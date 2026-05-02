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

        loop = asyncio.get_event_loop()

        # Fetch base stats (FG3_PCT, FG_PCT) and shot locations in parallel
        base_task = loop.run_in_executor(
            None,
            lambda: LeagueDashTeamStats(
                season="2025-26", per_mode_detailed="PerGame", timeout=15
            ).get_dict(),
        )
        loc_task = loop.run_in_executor(
            None,
            lambda: LeagueDashTeamShotLocations(
                season="2025-26", per_mode_detailed="PerGame", timeout=15
            ).get_dict(),
        )

        base_data, loc_data = await asyncio.gather(base_task, loc_task)

        # Parse base stats
        b_headers = base_data["resultSets"][0]["headers"]
        b_rows = base_data["resultSets"][0]["rowSet"]

        base_by_id = {}
        for row in b_rows:
            d = dict(zip(b_headers, row))
            base_by_id[d["TEAM_ID"]] = d

        # Parse shot locations
        # Row format: [TEAM_ID, TEAM_NAME, RA_FGM, RA_FGA, RA_PCT, Paint_FGM, Paint_FGA, Paint_PCT,
        #              Mid_FGM, Mid_FGA, Mid_PCT, LC3_FGM, LC3_FGA, LC3_PCT, RC3_FGM, RC3_FGA, RC3_PCT,
        #              AB3_FGM, AB3_FGA, AB3_PCT, BC_FGM, BC_FGA, BC_PCT, C3_FGM, C3_FGA, C3_PCT]
        loc_rows = loc_data["resultSets"]["rowSet"]

        teams = {}
        totals = {"fg3_pct": [], "rim_pct": [], "mid_pct": [], "fg3a": [], "rim_fga": []}

        for row in loc_rows:
            team_id = row[0]
            team_name = row[1]
            key = team_name.lower().rsplit(" ", 1)[-1] if " " in team_name.lower() else team_name.lower()
            if key == "blazers":
                key = "trail blazers"
            elif key == "clippers":
                key = "la clippers"

            rim_fgm, rim_fga, rim_pct = row[2], row[3], row[4]
            paint_fgm, paint_fga, paint_pct = row[5], row[6], row[7]
            mid_fgm, mid_fga, mid_pct = row[8], row[9], row[10]

            base = base_by_id.get(team_id, {})
            fg3_pct = base.get("FG3_PCT", 0.36)
            fg3a = base.get("FG3A", 35)
            efg = base.get("FG_PCT", 0.46) + 0.5 * base.get("FG3M", 12) / max(base.get("FGA", 85), 1)

            teams[key] = {
                "fg3_pct": fg3_pct,
                "fg3a": fg3a,
                "rim_pct": rim_pct,
                "rim_fga": rim_fga,
                "mid_pct": mid_pct,
                "mid_fga": mid_fga,
                "efg": efg,
                "full_name": team_name.lower(),
            }

            totals["fg3_pct"].append(fg3_pct)
            totals["rim_pct"].append(rim_pct)
            totals["mid_pct"].append(mid_pct)
            totals["fg3a"].append(fg3a)
            totals["rim_fga"].append(rim_fga)

        n = len(teams)
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
