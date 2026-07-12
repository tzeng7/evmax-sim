"""NcaawEfficiencyModelAgent — NCAAW win probability from opponent-adjusted efficiency.

Women's college basketball analogue of the NCAAB efficiency agent. Same math
core (_college_efficiency.py — shared ONLY within the college stack), its own
state file and constants: women's college has a wider talent spread (top-10
teams routinely post +40 net efficiencies and blowout margins), so the margin
sigma and shrinkage are calibrated separately via
scripts/backtest_ncaab_efficiency.py --league womens.

Parallel-stack rules: never imports from, reads, or references the NBA or
WNBA stacks.

State file: data/models/ncaaw_efficiency_state.json, seeded by
scripts/seed_ncaab_efficiency.py --league womens. update() is a no-op; the
staleness guard silences prior-season ratings once a new season starts.

See ncaab_efficiency_agent.py for the projection formula documentation —
identical model, per-league constants.
"""

from __future__ import annotations

from typing import Optional

import structlog

from evmax.agents.models._college_efficiency import (
    project_matchup,
    resolve_team,
    smooth_confidence,
    state_is_stale_for_today,
)
from evmax.agents.models.base import ModelAgent, ModelAgentPrediction
from evmax.models.market import PredictionMarket
from evmax.models.odds import SharpOdds

logger = structlog.get_logger(__name__)

SECTOR = "ncaaw"

# Swept on 2024-25, held out on 2025-26 (scripts/backtest_ncaab_efficiency.py
# --league womens): 10.5 won at every shrink_k — the wider women's blowout
# margins are already carried by the larger rating gaps, not the noise term.
SCORE_STDEV = 10.5

# Swept in {6, 10, 14}: k=6 won at every score_stdev, same as NCAAB (the
# solver's ridge already regularizes early-season noise).
SHRINK_K = 6.0

MIN_GAMES = 4

# Fallback home edge (efficiency pts / 100 poss) for legacy states.
HCA_EFF_FALLBACK = 4.5

FULL_SEASON_GP = 30


class NcaawEfficiencyModelAgent(ModelAgent):
    """Win probability from opponent-adjusted NCAAW efficiency ratings.

    Fires only for sector == "ncaaw"; NCAAB has its own agent, state file,
    and constants.
    """

    name = "ncaaw_efficiency"
    weight = 0.30

    def __init__(self) -> None:
        super().__init__()
        self._normalizer = None

    def _normalize(self, name: str) -> str:
        if self._normalizer is None:
            from evmax.matching.normalizer import NameNormalizer
            self._normalizer = NameNormalizer(SECTOR)
        return self._normalizer.normalize(name)

    async def predict_pair(
        self,
        market: PredictionMarket,
        sharp_odds: SharpOdds,
    ) -> Optional[ModelAgentPrediction]:
        sector = (market.sector or "").lower()
        if sector != SECTOR:
            return None

        if state_is_stale_for_today(self._state):
            logger.warning(
                "ncaaw_efficiency_stale_source_season",
                source_season=self._state.get("source_season"),
                hint="re-run scripts/seed_ncaab_efficiency.py --league womens",
            )
            return None

        teams: dict = self._state.get("teams", {})
        if not teams:
            return None

        team_a = (sharp_odds.outcome_a_label or market.team_home or "").lower().strip()
        team_b = (sharp_odds.outcome_b_label or market.team_away or "").lower().strip()

        stats_a = resolve_team(teams, team_a, normalize=self._normalize)
        stats_b = resolve_team(teams, team_b, normalize=self._normalize)
        if not stats_a or not stats_b:
            return None
        if stats_a.get("gp", 0) < MIN_GAMES or stats_b.get("gp", 0) < MIN_GAMES:
            return None

        league = {
            "league_avg_ortg": self._state.get("league_avg_ortg", 95.0),
            "league_avg_drtg": self._state.get("league_avg_drtg", 95.0),
            "league_avg_pace": self._state.get("league_avg_pace", 70.0),
        }
        hca_eff = self._state.get("hca_eff", HCA_EFF_FALLBACK)

        proj = project_matchup(
            stats_a, stats_b, league,
            hca_eff=hca_eff, score_stdev=SCORE_STDEV, shrink_k=SHRINK_K,
        )
        prob_a = proj["prob_home"]

        min_gp = min(stats_a["gp"], stats_b["gp"])
        confidence = smooth_confidence(min_gp, full_gp=FULL_SEASON_GP)

        sh, sa = proj["stats_home"], proj["stats_away"]
        return ModelAgentPrediction(
            event_id=sharp_odds.event_id,
            model_name=self.name,
            true_prob_a=prob_a,
            true_prob_b=1.0 - prob_a,
            true_prob_draw=None,
            confidence=confidence,
            weight=self.weight,
            sample_size=min_gp,
            notes=(
                f"adjO={sh['ortg']:.1f}/{sa['ortg']:.1f} "
                f"adjD={sh['drtg']:.1f}/{sa['drtg']:.1f} "
                f"margin={proj['margin']:+.1f} pace={proj['pace']:.1f}"
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
        """No-op — re-run scripts/seed_ncaab_efficiency.py --league womens weekly."""
        return
