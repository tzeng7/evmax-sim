"""EfficiencyModelAgent — NBA win probability from team offensive/defensive ratings.

Uses ORTG, DRTG, and Pace from stats.nba.com to project point differentials,
then converts to win probability via a normal CDF (calibrated to NBA variance).

Projected margin:
  pace = (pace_home + pace_away) / 2
  possessions = pace × 48 / 48  (≈ pace, in per-game terms)
  home_pts = ORTG_home × DRTG_factor_away × possessions / 100
  away_pts = ORTG_away × DRTG_factor_home × possessions / 100
  margin = home_pts - away_pts + HOME_EDGE

Win probability:
  P(home) = Φ(margin / σ)   where σ ≈ 12.0 for NBA (empirical)

State file: data/models/efficiency_state.json
  {
    "nba": {
      "league_avg_ortg": 114.5,
      "league_avg_drtg": 114.5,
      "league_avg_pace": 100.2,
      "teams": {
        "thunder": {"ortg": 117.6, "drtg": 106.5, "pace": 100.4, "net": 11.1, "gp": 82},
        ...
      },
      "fetched_at": "2026-04-18"
    }
  }
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import structlog

from evmax.agents.models.base import ModelAgent, ModelAgentPrediction
from evmax.models.market import PredictionMarket
from evmax.models.odds import SharpOdds

logger = structlog.get_logger(__name__)

STATE_PATH = Path(__file__).resolve().parents[3] / "data" / "models" / "efficiency_state.json"
CACHE_TTL_HOURS = 12

# NBA-specific constants
HOME_EDGE_PTS = 3.2
SCORE_STDEV = 12.0
MIN_GAMES = 20


def _normal_cdf(x: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun)."""
    import math
    if x < -6:
        return 0.0
    if x > 6:
        return 1.0
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = 0.3989422804014327  # 1/sqrt(2π)
    p = d * math.exp(-x * x / 2.0) * t * (
        0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.8212560 + t * 1.3302744)))
    )
    return 1.0 - p if x > 0 else p


class EfficiencyModelAgent(ModelAgent):
    """Win probability from team efficiency ratings (ORTG/DRTG/Pace).

    More predictive than Elo for NBA because it uses per-possession
    efficiency rather than raw W/L records.
    """

    name = "efficiency"
    weight = 0.40

    def __init__(self) -> None:
        super().__init__()
        self._stats_cache: Optional[dict] = None
        self._cache_time: float = 0

    async def _ensure_stats(self, sector: str) -> dict:
        """Load or fetch efficiency stats. Caches for CACHE_TTL_HOURS."""
        if sector != "nba":
            return {}

        from evmax.agents.models._nba_freshness import (
            state_is_fresh,
            nba_api_in_backoff,
            mark_nba_api_failure,
        )

        sector_state = self._state.get("nba", {})
        fetched_at = sector_state.get("fetched_at", "")

        if sector_state.get("teams") and await state_is_fresh(fetched_at):
            return sector_state

        if nba_api_in_backoff() and sector_state.get("teams"):
            self.log.info("efficiency_fetch_skipped_backoff", fetched_at=fetched_at)
            return sector_state

        try:
            stats = await self._fetch_from_nba_api()
            self._state["nba"] = stats
            self.save_state()
            return stats
        except Exception as e:
            mark_nba_api_failure()
            self.log.warning("efficiency_fetch_failed", error=str(e))
            return sector_state

    async def _fetch_from_nba_api(self) -> dict:
        """Fetch team advanced stats from stats.nba.com."""
        from nba_api.stats.endpoints import LeagueDashTeamStats

        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            None,
            lambda: LeagueDashTeamStats(
                season="2025-26",
                measure_type_detailed_defense="Advanced",
                per_mode_detailed="PerGame",
                timeout=15,
            ).get_dict(),
        )

        headers = data["resultSets"][0]["headers"]
        rows = data["resultSets"][0]["rowSet"]

        teams = {}
        total_ortg = 0.0
        total_drtg = 0.0
        total_pace = 0.0

        for row in rows:
            d = dict(zip(headers, row))
            name = d["TEAM_NAME"].lower()
            # Extract last word for matching (e.g. "Oklahoma City Thunder" → "thunder")
            key = name.rsplit(" ", 1)[-1] if " " in name else name
            # Handle two-word team names
            if key in ("blazers",):
                key = "trail blazers"
            elif key in ("clippers",):
                key = "la clippers"

            teams[key] = {
                "ortg": d["OFF_RATING"],
                "drtg": d["DEF_RATING"],
                "pace": d["PACE"],
                "net": d["NET_RATING"],
                "efg": d["EFG_PCT"],
                "ts": d["TS_PCT"],
                "tov_pct": d["TM_TOV_PCT"],
                "oreb_pct": d["OREB_PCT"],
                "gp": d["GP"],
                "full_name": d["TEAM_NAME"].lower(),
            }
            total_ortg += d["OFF_RATING"]
            total_drtg += d["DEF_RATING"]
            total_pace += d["PACE"]

        n = len(teams)
        result = {
            "league_avg_ortg": round(total_ortg / n, 2),
            "league_avg_drtg": round(total_drtg / n, 2),
            "league_avg_pace": round(total_pace / n, 2),
            "teams": teams,
            "fetched_at": date.today().isoformat(),
        }

        self.log.info("efficiency_fetched", teams=n, avg_ortg=result["league_avg_ortg"])
        return result

    def _resolve_team(self, teams: dict, team: str) -> Optional[dict]:
        """Resolve team name to stats dict."""
        team = team.lower().strip()
        if team in teams:
            return teams[team]
        # Try last word
        if " " in team:
            last = team.rsplit(" ", 1)[-1]
            if last in teams:
                return teams[last]
        # Prefix/suffix match
        for key, val in teams.items():
            full = val.get("full_name", "")
            if (team.startswith(key) or key.startswith(team)
                    or team in full or full.endswith(team)):
                return val
        return None

    async def predict_pair(
        self,
        market: PredictionMarket,
        sharp_odds: SharpOdds,
    ) -> Optional[ModelAgentPrediction]:
        sector = (market.sector or "").lower()
        if sector != "nba":
            return None

        stats = await self._ensure_stats(sector)
        teams = stats.get("teams", {})
        if not teams:
            return None

        team_a = (sharp_odds.outcome_a_label or market.team_home or "").lower().strip()
        team_b = (sharp_odds.outcome_b_label or market.team_away or "").lower().strip()

        stats_a = self._resolve_team(teams, team_a)
        stats_b = self._resolve_team(teams, team_b)

        if not stats_a or not stats_b:
            return None

        if stats_a["gp"] < MIN_GAMES or stats_b["gp"] < MIN_GAMES:
            return None

        league_ortg = stats.get("league_avg_ortg", 114.5)
        league_drtg = stats.get("league_avg_drtg", 114.5)

        # Offensive efficiency factor: team's ORTG relative to league average
        off_factor_a = stats_a["ortg"] / league_ortg
        off_factor_b = stats_b["ortg"] / league_ortg

        # Defensive efficiency factor: team's DRTG relative to league average
        # Lower DRTG = better defense, so we invert
        def_factor_a = stats_a["drtg"] / league_drtg
        def_factor_b = stats_b["drtg"] / league_drtg

        # Expected pace (average of both teams)
        expected_pace = (stats_a["pace"] + stats_b["pace"]) / 2.0

        # Project points: team A offense vs team B defense, scaled by pace
        possessions = expected_pace
        proj_pts_a = off_factor_a * def_factor_b * league_ortg * possessions / 100.0
        proj_pts_b = off_factor_b * def_factor_a * league_ortg * possessions / 100.0

        # Projected margin (positive = home team A favored)
        margin = proj_pts_a - proj_pts_b + HOME_EDGE_PTS

        # Win probability via normal CDF
        prob_a = _normal_cdf(margin / SCORE_STDEV)
        prob_a = max(0.02, min(0.98, prob_a))
        prob_b = 1.0 - prob_a

        # Confidence based on sample size and net rating clarity
        min_gp = min(stats_a["gp"], stats_b["gp"])
        if min_gp >= 60:
            confidence = 0.85
        elif min_gp >= 40:
            confidence = 0.75
        else:
            confidence = 0.60

        return ModelAgentPrediction(
            event_id=sharp_odds.event_id,
            model_name=self.name,
            true_prob_a=prob_a,
            true_prob_b=prob_b,
            true_prob_draw=None,
            confidence=confidence,
            weight=self.weight,
            sample_size=min_gp,
            notes=(
                f"ORTG={stats_a['ortg']:.1f}/{stats_b['ortg']:.1f} "
                f"DRTG={stats_a['drtg']:.1f}/{stats_b['drtg']:.1f} "
                f"margin={margin:+.1f} pace={expected_pace:.1f}"
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
        """No-op: efficiency stats are fetched in bulk from stats.nba.com."""
        pass
