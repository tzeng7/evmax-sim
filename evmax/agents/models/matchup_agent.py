"""MatchupAgent — style-based NBA matchup adjustments.

Analyzes how each team's offensive strengths interact with the opponent's
defensive weaknesses (and vice versa) to produce a probability nudge.

Three matchup dimensions:
  1. Paint scoring vs rim protection (BLK + OPP_PTS_PAINT)
  2. 3PT volume vs perimeter defense (OPP FG3_PCT, opponent 3PA allowed)
  3. Turnover forcing vs ball security (STL vs TOV)

Each dimension produces a point-margin nudge capped at ±1.5 points.
Combined nudge capped at ±4 points → ±5% probability via normal CDF.
"""

from __future__ import annotations

import asyncio
import json
from datetime import date
from pathlib import Path
from typing import Optional

import structlog

from evmax.agents.models.base import ModelAgent, ModelAgentPrediction
from evmax.models.market import PredictionMarket
from evmax.models.odds import SharpOdds

logger = structlog.get_logger(__name__)

STATE_PATH = Path(__file__).resolve().parents[3] / "data" / "models" / "matchup_state.json"

SCORE_STDEV = 12.0
MAX_DIM_NUDGE = 1.5  # max points per dimension
MAX_TOTAL_NUDGE = 4.0

TEAM_ABBREV: dict[str, str] = {
    "ATL": "hawks", "BOS": "celtics", "BKN": "nets", "CHA": "hornets",
    "CHI": "bulls", "CLE": "cavaliers", "DAL": "mavericks", "DEN": "nuggets",
    "DET": "pistons", "GSW": "warriors", "HOU": "rockets", "IND": "pacers",
    "LAC": "la clippers", "LAL": "lakers", "MEM": "grizzlies", "MIA": "heat",
    "MIL": "bucks", "MIN": "timberwolves", "NOP": "pelicans", "NYK": "knicks",
    "OKC": "thunder", "ORL": "magic", "PHI": "76ers", "PHX": "suns",
    "POR": "trail blazers", "SAC": "kings", "SAS": "spurs", "TOR": "raptors",
    "UTA": "jazz", "WAS": "wizards",
}


def _normal_cdf(x: float) -> float:
    import math
    if x < -6:
        return 0.0
    if x > 6:
        return 1.0
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = 0.3989422804014327
    p = d * math.exp(-x * x / 2.0) * t * (
        0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.8212560 + t * 1.3302744)))
    )
    return 1.0 - p if x > 0 else p


class MatchupAgent(ModelAgent):
    """Style-based matchup probability adjustments."""

    name = "matchup"
    weight = 0.10

    def __init__(self) -> None:
        super().__init__()
        self._off_stats: dict[str, dict] = {}
        self._def_stats: dict[str, dict] = {}
        self._league_avgs: dict[str, float] = {}
        self._fetched_date: str = ""

    async def _ensure_data(self) -> None:
        from evmax.agents.models._nba_freshness import (
            state_is_fresh,
            nba_api_in_backoff,
            mark_nba_api_failure,
        )

        if self._off_stats and await state_is_fresh(self._fetched_date):
            return

        sector_state = self._state.get("nba", {})
        sector_fetched = sector_state.get("fetched_date", "")
        if sector_fetched and await state_is_fresh(sector_fetched):
            self._off_stats = sector_state.get("offense", {})
            self._def_stats = sector_state.get("defense", {})
            self._league_avgs = sector_state.get("league_avgs", {})
            self._fetched_date = sector_fetched
            return

        if nba_api_in_backoff() and sector_state.get("offense"):
            self._off_stats = sector_state.get("offense", {})
            self._def_stats = sector_state.get("defense", {})
            self._league_avgs = sector_state.get("league_avgs", {})
            self._fetched_date = sector_fetched
            self.log.info("matchup_fetch_skipped_backoff", fetched_date=sector_fetched)
            return

        try:
            await self._fetch_data()
        except Exception as e:
            mark_nba_api_failure()
            self.log.warning("matchup_fetch_failed", error=str(e))

    async def _fetch_data(self) -> None:
        from nba_api.stats.endpoints import LeagueDashTeamStats
        from evmax.agents.models._nba_playoff_blend import (
            fetch_rs_and_po,
            gp_weighted_blend,
            parse_team_dash_rows,
        )

        def base_factory(season_type: str):
            return LeagueDashTeamStats(
                season="2025-26",
                season_type_all_star=season_type,
                per_mode_detailed="PerGame",
                timeout=15,
            )

        def def_factory(season_type: str):
            return LeagueDashTeamStats(
                season="2025-26",
                season_type_all_star=season_type,
                measure_type_detailed_defense="Defense",
                per_mode_detailed="PerGame",
                timeout=15,
            )

        # 4 calls in parallel: (base RS + base PO) + (def RS + def PO)
        (base_rs_data, base_po_data), (def_rs_data, def_po_data) = await asyncio.gather(
            fetch_rs_and_po(base_factory),
            fetch_rs_and_po(def_factory),
        )

        base_rs = parse_team_dash_rows(base_rs_data)
        base_po = parse_team_dash_rows(base_po_data)
        def_rs = parse_team_dash_rows(def_rs_data)
        def_po = parse_team_dash_rows(def_po_data)

        if not base_rs and not base_po:
            raise RuntimeError("nba_api returned no base rows for either season type")

        base_blended = gp_weighted_blend(
            base_rs, base_po,
            ["PTS", "FG3A", "FG3_PCT", "TOV", "OREB"],
        )
        def_blended = gp_weighted_blend(
            def_rs, def_po,
            ["DEF_RATING", "BLK", "STL", "OPP_PTS_PAINT", "OPP_PTS_FB"],
        )

        off_stats = {}
        def_stats = {}
        totals = {"pts_paint": [], "fg3a": [], "tov": [], "blk": [], "stl": [],
                  "opp_pts_paint": [], "opp_pts_fb": []}

        for key, b in base_blended.items():
            off_stats[key] = {
                "pts": b.get("PTS", 110),
                "fg3a": b.get("FG3A", 35),
                "fg3_pct": b.get("FG3_PCT", 0.36),
                "tov": b.get("TOV", 13),
                "oreb": b.get("OREB", 10),
                "gp": b["gp"],
                "rs_gp": b["rs_gp"],
                "po_gp": b["po_gp"],
            }
            totals["fg3a"].append(b.get("FG3A", 35))
            totals["tov"].append(b.get("TOV", 13))

        for key, b in def_blended.items():
            def_stats[key] = {
                "drtg": b.get("DEF_RATING", 114),
                "blk": b.get("BLK", 5),
                "stl": b.get("STL", 8),
                "opp_pts_paint": b.get("OPP_PTS_PAINT", 48),
                "opp_pts_fb": b.get("OPP_PTS_FB", 14),
                "gp": b["gp"],
                "rs_gp": b["rs_gp"],
                "po_gp": b["po_gp"],
            }
            totals["blk"].append(b.get("BLK", 5))
            totals["stl"].append(b.get("STL", 8))
            totals["opp_pts_paint"].append(b.get("OPP_PTS_PAINT", 48))
            totals["pts_paint"].append(b.get("OPP_PTS_PAINT", 48))
            totals["opp_pts_fb"].append(b.get("OPP_PTS_FB", 14))

        n = len(base_blended)
        po_count = sum(1 for v in off_stats.values() if v["po_gp"] > 0)
        self._league_avgs = {k: sum(v) / len(v) for k, v in totals.items() if v}
        self._off_stats = off_stats
        self._def_stats = def_stats
        self._fetched_date = date.today().isoformat()

        self._state["nba"] = {
            "fetched_date": self._fetched_date,
            "offense": off_stats,
            "defense": def_stats,
            "league_avgs": self._league_avgs,
        }
        self.save_state()
        self.log.info("matchup_fetched", teams=n, po_teams=po_count)

    def _resolve(self, team: str, store: dict) -> Optional[dict]:
        team = team.lower().strip()
        if team in store:
            return store[team]
        if " " in team:
            last = team.rsplit(" ", 1)[-1]
            if last in store:
                return store[last]
        for key in store:
            if team.endswith(key) or key.endswith(team) or team.startswith(key):
                return store[key]
        return None

    def _matchup_margin(self, off_a: dict, def_b: dict, off_b: dict, def_a: dict) -> float:
        """Compute net matchup margin advantage for team A.

        Uses direct opponent-adjusted stats: how many points does the opponent
        actually allow in each category, compared to league average?
        """
        avgs = self._league_avgs
        if not avgs:
            return 0.0

        nudge = 0.0
        avg_paint = avgs.get("opp_pts_paint", 48)
        avg_fb = avgs.get("opp_pts_fb", 14) if "opp_pts_fb" in avgs else 14.0
        avg_tov = avgs.get("tov", 13)
        avg_stl = avgs.get("stl", 8)

        # 1. Paint scoring: does opponent B allow more/fewer paint points?
        # Positive = B allows MORE than average → good for A's paint scoring
        paint_a = (def_b.get("opp_pts_paint", avg_paint) - avg_paint) * 0.15
        paint_b = (def_a.get("opp_pts_paint", avg_paint) - avg_paint) * 0.15
        nudge += max(-MAX_DIM_NUDGE, min(MAX_DIM_NUDGE, paint_a - paint_b))

        # 2. Transition: does opponent allow more fastbreak points?
        # Teams that allow lots of fastbreak pts are vulnerable to
        # teams that force turnovers (steals → fastbreak opportunities)
        fb_a = (def_b.get("opp_pts_fb", avg_fb) - avg_fb) * 0.12
        fb_b = (def_a.get("opp_pts_fb", avg_fb) - avg_fb) * 0.12
        nudge += max(-MAX_DIM_NUDGE, min(MAX_DIM_NUDGE, fb_a - fb_b))

        # 3. Turnover battle: team A's ball security vs B's steal rate
        # A team that turns it over a lot facing a steal-heavy defense
        # loses extra possessions
        a_excess_tov = (off_a.get("tov", avg_tov) - avg_tov)
        b_excess_stl = (def_b.get("stl", avg_stl) - avg_stl)
        b_excess_tov = (off_b.get("tov", avg_tov) - avg_tov)
        a_excess_stl = (def_a.get("stl", avg_stl) - avg_stl)

        # When a careless team faces an active-hands defense, amplify TO pain
        tov_a = -a_excess_tov * 0.08 - (a_excess_tov * b_excess_stl * 0.03 if a_excess_tov > 0 and b_excess_stl > 0 else 0)
        tov_b = -b_excess_tov * 0.08 - (b_excess_tov * a_excess_stl * 0.03 if b_excess_tov > 0 and a_excess_stl > 0 else 0)
        nudge += max(-MAX_DIM_NUDGE, min(MAX_DIM_NUDGE, tov_a - tov_b))

        return max(-MAX_TOTAL_NUDGE, min(MAX_TOTAL_NUDGE, nudge))

    async def predict_pair(
        self,
        market: PredictionMarket,
        sharp_odds: SharpOdds,
    ) -> Optional[ModelAgentPrediction]:
        sector = (market.sector or "").lower()
        if sector != "nba":
            return None

        await self._ensure_data()
        if not self._off_stats:
            return None

        team_a = (sharp_odds.outcome_a_label or market.team_home or "").lower().strip()
        team_b = (sharp_odds.outcome_b_label or market.team_away or "").lower().strip()

        off_a = self._resolve(team_a, self._off_stats)
        off_b = self._resolve(team_b, self._off_stats)
        def_a = self._resolve(team_a, self._def_stats)
        def_b = self._resolve(team_b, self._def_stats)

        if not all([off_a, off_b, def_a, def_b]):
            return None

        margin = self._matchup_margin(off_a, def_b, off_b, def_a)

        # Convert margin nudge to probability
        base_a = sharp_odds.true_prob_a
        prob_shift = _normal_cdf(margin / SCORE_STDEV) - 0.5
        prob_a = max(0.05, min(0.95, base_a + prob_shift))
        prob_b = 1.0 - prob_a

        return ModelAgentPrediction(
            event_id=sharp_odds.event_id,
            model_name=self.name,
            true_prob_a=prob_a,
            true_prob_b=prob_b,
            true_prob_draw=None,
            confidence=0.55,
            weight=self.weight,
            sample_size=82,
            notes=f"matchup_margin={margin:+.1f}pts",
        )

    def update(self, team_a: str, team_b: str, score_a: float, score_b: float,
               sector: str, event_date: Optional[str] = None) -> None:
        pass
