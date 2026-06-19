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


# Shared Abramowitz & Stegun normal CDF (single source in models_ml).
from evmax.models_ml._math import normal_cdf as _normal_cdf


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
        """Fetch team advanced stats from stats.nba.com — Regular Season + Playoffs blended."""
        from nba_api.stats.endpoints import LeagueDashTeamStats
        from evmax.agents.models._nba_playoff_blend import (
            fetch_rs_and_po,
            gp_weighted_blend,
            parse_team_dash_rows,
        )

        def factory(season_type: str):
            return LeagueDashTeamStats(
                season="2025-26",
                season_type_all_star=season_type,
                measure_type_detailed_defense="Advanced",
                per_mode_detailed="PerGame",
                timeout=15,
            )

        rs_data, po_data = await fetch_rs_and_po(factory)
        rs_rows = parse_team_dash_rows(rs_data)
        po_rows = parse_team_dash_rows(po_data)

        if not rs_rows and not po_rows:
            raise RuntimeError("nba_api returned no team rows for either season type")

        fields = ["OFF_RATING", "DEF_RATING", "PACE", "NET_RATING",
                  "EFG_PCT", "TS_PCT", "TM_TOV_PCT", "OREB_PCT"]
        blended = gp_weighted_blend(rs_rows, po_rows, fields)

        teams = {}
        total_ortg = 0.0
        total_drtg = 0.0
        total_pace = 0.0

        for key, b in blended.items():
            teams[key] = {
                "ortg": round(b["OFF_RATING"], 2),
                "drtg": round(b["DEF_RATING"], 2),
                "pace": round(b["PACE"], 2),
                "net": round(b["NET_RATING"], 2),
                "efg": round(b["EFG_PCT"], 4),
                "ts": round(b["TS_PCT"], 4),
                "tov_pct": round(b["TM_TOV_PCT"], 4),
                "oreb_pct": round(b["OREB_PCT"], 4),
                "gp": b["gp"],
                "rs_gp": b["rs_gp"],
                "po_gp": b["po_gp"],
                "full_name": b["TEAM_NAME"].lower(),
            }
            total_ortg += teams[key]["ortg"]
            total_drtg += teams[key]["drtg"]
            total_pace += teams[key]["pace"]

        n = len(teams)
        po_team_count = sum(1 for v in teams.values() if v["po_gp"] > 0)
        result = {
            "league_avg_ortg": round(total_ortg / n, 2),
            "league_avg_drtg": round(total_drtg / n, 2),
            "league_avg_pace": round(total_pace / n, 2),
            "teams": teams,
            "fetched_at": date.today().isoformat(),
        }

        self.log.info(
            "efficiency_fetched",
            teams=n,
            po_teams=po_team_count,
            avg_ortg=result["league_avg_ortg"],
        )
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
