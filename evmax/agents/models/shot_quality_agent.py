"""ShotQualityAgent — shooting regression signal for NBA.

Identifies teams whose actual shooting is unsustainably high or low
relative to league baselines, and predicts regression.

Key signals:
  1. 3PT% vs league average — teams shooting >38% or <33% will regress
  2. Rim finishing (Restricted Area FG%) vs league average
  3. eFG% vs expected eFG% from shot distribution

A team shooting 40% from 3 when league avg is 36% is ~1.5% luckier than
sustainable. Over a game, that's ~1-2 points of margin. Convert to a
win probability nudge via the normal CDF.

This is an intelligence agent (probability adjustment), not a standalone
model — it produces adjustments that modify the ensemble output.
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

STATE_PATH = Path(__file__).resolve().parents[3] / "data" / "models" / "shot_quality_state.json"

# League-average baselines (updated when data is fetched)
DEFAULT_LEAGUE_3PT = 0.362
DEFAULT_LEAGUE_RIM = 0.635
DEFAULT_LEAGUE_MID = 0.420

# Regression strength: how many points of margin per 1% of shooting deviation
# A team 2% above avg from 3 on ~35 attempts ≈ 0.7 extra makes ≈ 2.1 pts
PTS_PER_3PT_PCT = 1.05  # per 1% deviation, per game
PTS_PER_RIM_PCT = 0.40  # rim attempts are fewer than 3s for most teams

# Standard deviation for NBA scoring
SCORE_STDEV = 12.0

# Team abbreviation mapping (reuse from player_impact)
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


# Shared Abramowitz & Stegun normal CDF (single source in models_ml).
from evmax.models_ml._math import normal_cdf as _normal_cdf


class ShotQualityAgent(ModelAgent):
    """Predicts shooting regression and adjusts win probability."""

    name = "shot_quality"
    weight = 0.15  # relatively low — this is a correction signal

    def __init__(self) -> None:
        super().__init__()
        self._team_shooting: dict[str, dict] = {}
        self._league_avgs: dict[str, float] = {}
        self._fetched_date: str = ""

    async def _ensure_data(self) -> None:
        from evmax.agents.models._nba_freshness import (
            state_is_fresh,
            nba_api_in_backoff,
            mark_nba_api_failure,
        )

        if self._team_shooting and await state_is_fresh(self._fetched_date):
            return

        sector_state = self._state.get("nba", {})
        sector_fetched = sector_state.get("fetched_date", "")
        if sector_fetched and await state_is_fresh(sector_fetched):
            self._team_shooting = sector_state.get("teams", {})
            self._league_avgs = sector_state.get("league_avgs", {})
            self._fetched_date = sector_fetched
            return

        if nba_api_in_backoff() and sector_state.get("teams"):
            self._team_shooting = sector_state.get("teams", {})
            self._league_avgs = sector_state.get("league_avgs", {})
            self._fetched_date = sector_fetched
            self.log.info("shot_quality_fetch_skipped_backoff", fetched_date=sector_fetched)
            return

        try:
            await self._fetch_shooting_data()
        except Exception as e:
            mark_nba_api_failure()
            self.log.warning("shot_quality_fetch_failed", error=str(e))

    async def _fetch_shooting_data(self) -> None:
        from nba_api.stats.endpoints import LeagueDashTeamStats, LeagueDashTeamShotLocations
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

        def loc_factory(season_type: str):
            return LeagueDashTeamShotLocations(
                season="2025-26",
                season_type_all_star=season_type,
                per_mode_detailed="PerGame",
                timeout=15,
            )

        # 4 parallel fetches: base RS+PO, locations RS+PO
        (base_rs_data, base_po_data), (loc_rs_data, loc_po_data) = await asyncio.gather(
            fetch_rs_and_po(base_factory),
            fetch_rs_and_po(loc_factory),
        )

        base_rs = parse_team_dash_rows(base_rs_data)
        base_po = parse_team_dash_rows(base_po_data)

        if not base_rs and not base_po:
            raise RuntimeError("nba_api returned no base rows for either season type")

        # Blend base stats — need GP, FG_PCT, FG3_PCT, FG3A, FG3M, FGA (last 3 for eFG)
        base_blended = gp_weighted_blend(
            base_rs, base_po,
            ["FG_PCT", "FG3_PCT", "FG3A", "FG3M", "FGA"],
        )

        # Shot locations payload has a single resultSets dict (not a list).
        # Row layout: [TEAM_ID, TEAM_NAME, RA_FGM, RA_FGA, RA_PCT,
        #              Paint_FGM, Paint_FGA, Paint_PCT,
        #              Mid_FGM, Mid_FGA, Mid_PCT, ...]
        def _parse_loc(payload: Optional[dict]) -> dict[str, dict]:
            if not payload:
                return {}
            try:
                rows = payload["resultSets"]["rowSet"]
            except (KeyError, TypeError):
                return {}
            out: dict[str, dict] = {}
            for row in rows:
                if len(row) < 11:
                    continue
                team_name = (row[1] or "").lower()
                if not team_name:
                    continue
                key = team_name.rsplit(" ", 1)[-1] if " " in team_name else team_name
                if key == "blazers":
                    key = "trail blazers"
                elif key == "clippers":
                    key = "la clippers"
                out[key] = {
                    "TEAM_ID": row[0],
                    "TEAM_NAME": row[1],
                    "rim_fgm": row[2] or 0,
                    "rim_fga": row[3] or 0,
                    "rim_pct": row[4] or 0,
                    "paint_pct": row[7] or 0,
                    "mid_fgm": row[8] or 0,
                    "mid_fga": row[9] or 0,
                    "mid_pct": row[10] or 0,
                }
            return out

        loc_rs = _parse_loc(loc_rs_data)
        loc_po = _parse_loc(loc_po_data)

        # GP-blend shot locations using base GP per team (same set of games per season type).
        teams: dict[str, dict] = {}
        totals = {"fg3_pct": [], "rim_pct": [], "mid_pct": [], "fg3a": [], "rim_fga": []}

        for key, b in base_blended.items():
            rs_gp = b["rs_gp"]
            po_gp = b["po_gp"]
            total_gp = b["gp"]

            # Weighted blend of RS + PO shot-location rows
            rs_loc = loc_rs.get(key, {})
            po_loc = loc_po.get(key, {})

            def _blend_loc(field: str, default: float) -> float:
                rs_v = rs_loc.get(field) if rs_loc else None
                po_v = po_loc.get(field) if po_loc else None
                if rs_gp == 0 or rs_v is None:
                    return po_v if po_v is not None else default
                if po_gp == 0 or po_v is None:
                    return rs_v
                return (rs_v * rs_gp + po_v * po_gp) / total_gp

            fg3_pct = b.get("FG3_PCT", 0.36)
            fg3a = b.get("FG3A", 35)
            fg3m = b.get("FG3M", 12)
            fga = b.get("FGA", 85) or 85
            efg = b.get("FG_PCT", 0.46) + 0.5 * fg3m / max(fga, 1)

            full_name = (rs_loc.get("TEAM_NAME") or po_loc.get("TEAM_NAME")
                         or b.get("TEAM_NAME", "")).lower()

            teams[key] = {
                "fg3_pct": fg3_pct,
                "fg3a": fg3a,
                "rim_pct": _blend_loc("rim_pct", 0.635),
                "rim_fga": _blend_loc("rim_fga", 25),
                "mid_pct": _blend_loc("mid_pct", 0.42),
                "mid_fga": _blend_loc("mid_fga", 15),
                "efg": efg,
                "gp": total_gp,
                "rs_gp": rs_gp,
                "po_gp": po_gp,
                "full_name": full_name,
            }

            totals["fg3_pct"].append(teams[key]["fg3_pct"])
            totals["rim_pct"].append(teams[key]["rim_pct"])
            totals["mid_pct"].append(teams[key]["mid_pct"])
            totals["fg3a"].append(teams[key]["fg3a"])
            totals["rim_fga"].append(teams[key]["rim_fga"])

        n = len(teams)
        po_count = sum(1 for v in teams.values() if v["po_gp"] > 0)
        self._league_avgs = {
            "fg3_pct": sum(totals["fg3_pct"]) / n,
            "rim_pct": sum(totals["rim_pct"]) / n,
            "mid_pct": sum(totals["mid_pct"]) / n,
            "fg3a": sum(totals["fg3a"]) / n,
            "rim_fga": sum(totals["rim_fga"]) / n,
        }
        self._team_shooting = teams
        self._fetched_date = date.today().isoformat()

        # Persist
        self._state["nba"] = {
            "fetched_date": self._fetched_date,
            "teams": teams,
            "league_avgs": self._league_avgs,
        }
        self.save_state()

        self.log.info(
            "shot_quality_fetched",
            teams=n,
            po_teams=po_count,
            league_3pt=f"{self._league_avgs['fg3_pct']:.3f}",
        )

    def _resolve_team(self, team: str) -> Optional[dict]:
        team = team.lower().strip()
        if team in self._team_shooting:
            return self._team_shooting[team]
        if " " in team:
            last = team.rsplit(" ", 1)[-1]
            if last in self._team_shooting:
                return self._team_shooting[last]
        for key, val in self._team_shooting.items():
            full = val.get("full_name", "")
            if team in full or full.endswith(team) or team.startswith(key):
                return val
        return None

    def _regression_margin(self, team_stats: dict) -> float:
        """Estimate how many points of margin a team gains/loses from
        unsustainable shooting. Positive = team is over-performing."""
        avgs = self._league_avgs
        if not avgs:
            return 0.0

        # 3PT regression
        fg3_dev = team_stats["fg3_pct"] - avgs["fg3_pct"]
        fg3_margin = fg3_dev * 100 * PTS_PER_3PT_PCT

        # Rim regression
        rim_dev = team_stats["rim_pct"] - avgs["rim_pct"]
        rim_margin = rim_dev * 100 * PTS_PER_RIM_PCT

        return fg3_margin + rim_margin

    def _expected_pts_per_shot(self, stats: dict) -> float:
        """Expected points per field goal attempt from shot distribution.

        Uses zone-level FGA and FG% to compute expected offensive output.
        Better than raw eFG% because it captures shot selection quality.
        """
        avgs = self._league_avgs
        if not avgs:
            return 1.0

        fg3a = stats.get("fg3a", avgs.get("fg3a", 35))
        rim_fga = stats.get("rim_fga", avgs.get("rim_fga", 25))
        total_fga = fg3a + rim_fga + stats.get("mid_fga", 15)
        if total_fga == 0:
            return 1.0

        # Regress each zone's FG% 30% toward league average to avoid
        # rewarding/penalizing pure variance
        regress = 0.30
        fg3 = stats["fg3_pct"] * (1 - regress) + avgs["fg3_pct"] * regress
        rim = stats["rim_pct"] * (1 - regress) + avgs["rim_pct"] * regress
        mid = stats.get("mid_pct", avgs.get("mid_pct", 0.42))
        mid = mid * (1 - regress) + avgs.get("mid_pct", 0.42) * regress

        pts_3 = fg3a * fg3 * 3.0
        pts_rim = rim_fga * rim * 2.0
        mid_fga = stats.get("mid_fga", 15)
        pts_mid = mid_fga * mid * 2.0

        return (pts_3 + pts_rim + pts_mid) / total_fga

    async def predict_pair(
        self,
        market: PredictionMarket,
        sharp_odds: SharpOdds,
    ) -> Optional[ModelAgentPrediction]:
        sector = (market.sector or "").lower()
        if sector != "nba":
            return None

        await self._ensure_data()
        if not self._team_shooting:
            return None

        team_a = (sharp_odds.outcome_a_label or market.team_home or "").lower().strip()
        team_b = (sharp_odds.outcome_b_label or market.team_away or "").lower().strip()

        stats_a = self._resolve_team(team_a)
        stats_b = self._resolve_team(team_b)

        if not stats_a or not stats_b:
            return None

        # Expected points per shot for each team (regressed toward mean)
        pps_a = self._expected_pts_per_shot(stats_a)
        pps_b = self._expected_pts_per_shot(stats_b)

        # Convert differential to projected margin over a full game (~85 FGA)
        fga_per_game = 85.0
        margin = (pps_a - pps_b) * fga_per_game + 2.5  # home edge

        prob_a = _normal_cdf(margin / SCORE_STDEV)
        prob_a = max(0.05, min(0.95, prob_a))
        prob_b = 1.0 - prob_a

        confidence = 0.60

        return ModelAgentPrediction(
            event_id=sharp_odds.event_id,
            model_name=self.name,
            true_prob_a=prob_a,
            true_prob_b=prob_b,
            true_prob_draw=None,
            confidence=confidence,
            weight=self.weight,
            sample_size=82,
            notes=(
                f"pps={pps_a:.3f}/{pps_b:.3f} "
                f"3PT={stats_a['fg3_pct']:.3f}/{stats_b['fg3_pct']:.3f} "
                f"margin={margin:+.1f}pts"
            ),
        )

    def update(
        self,
        team_a: str,
        team_b: str,
        score_a: float,
        score_b: float,
        sector: str,
        event_date: Optional[str] = None,
    ) -> None:
        """No-op: shooting stats are fetched in bulk from stats.nba.com."""
        pass
