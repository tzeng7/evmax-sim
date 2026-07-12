"""NcaabEfficiencyModelAgent — NCAAB win probability from opponent-adjusted efficiency.

Men's college basketball analogue of the WNBA efficiency agent, but the
ratings it reads are OPPONENT-ADJUSTED (KenPom-style iterative solver in
evmax/agents/models/_college_efficiency.py) because raw Dean Oliver ratings
are meaningless across a 360-team league with wildly unequal schedules.

Parallel-stack rules:
  - Never imports from, reads, or references the NBA or WNBA stacks.
  - Shares the math core (_college_efficiency.py) with the NCAAW stack only —
    same data source and formulas; constants and state files stay per-league.

Inputs per team (seeded by scripts/seed_ncaab_efficiency.py --league mens):
  - ortg / drtg : opponent-ADJUSTED offensive/defensive efficiency (pts/100)
  - pace        : possessions per game (Dean Oliver)
  - tov_pct     : turnovers per possession (used by the possession sim)
  - gp          : D1 games in the seeded season

Projected margin:
  pace      = (pace_home + pace_away) / 2
  home_pts  = ortg_h/lg · drtg_a/lg · lg · pace / 100     (mirrored for away)
  margin    = home_pts − away_pts + hca_eff · pace / 100

  hca_eff is estimated jointly by the seed solver (efficiency points per 100
  possessions) and stored in the state file; a constant fallback covers
  legacy states. Neutral-site games (conference + NCAA tournaments) are NOT
  detected at predict time (the market feed carries no neutral flag) — the
  home edge is applied uniformly, same as the live Elo model. Known
  limitation, shared with every other sector.

Win probability: P(home) = Φ(margin / SCORE_STDEV)

Constants below were validated on the 2024-25 season and held out against
2025-26 via scripts/backtest_ncaab_efficiency.py. Do not copy NBA/WNBA values.

State file: data/models/ncaab_efficiency_state.json (auto-resolved from `name`).
The agent's update() is a no-op — re-run the seed script weekly in season.
A staleness guard silences prior-season ratings once a new season starts
(the WNBA +24pp chalk-bias lesson; college roster turnover is worse).
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

SECTOR = "ncaab"

# Margin standard deviation for the Φ transform. Swept on the 2024-25 season
# (scripts/backtest_ncaab_efficiency.py); men's college margins vs a rating
# model land near the NCAA spread-market sigma (~10.5).
SCORE_STDEV = 10.5

# Empirical-Bayes pseudo-count pulling adjusted ratings toward league mean.
# Swept on 2024-25 in {6, 10, 14}: k=6 won at every score_stdev (the
# opponent-adjustment solver's ridge already regularizes early-season noise,
# so predict-time shrinkage lighter than the WNBA's 8 is optimal here).
SHRINK_K = 6.0

# Hard floor — below this the ratings are pure noise even with shrinkage.
MIN_GAMES = 4

# Fallback home edge (efficiency pts / 100 poss) when the state file predates
# the solver's hca_eff field. ~3 points at a 68-possession pace.
HCA_EFF_FALLBACK = 4.5

# Confidence ramp completes at a full college regular season.
FULL_SEASON_GP = 30


class NcaabEfficiencyModelAgent(ModelAgent):
    """Win probability from opponent-adjusted NCAAB efficiency ratings.

    Fires only for sector == "ncaab"; returns None for everything else
    (NCAAW has its own agent, state file, and constants).
    """

    name = "ncaab_efficiency"
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
                "ncaab_efficiency_stale_source_season",
                source_season=self._state.get("source_season"),
                hint="re-run scripts/seed_ncaab_efficiency.py --league mens",
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
            "league_avg_ortg": self._state.get("league_avg_ortg", 104.0),
            "league_avg_drtg": self._state.get("league_avg_drtg", 104.0),
            "league_avg_pace": self._state.get("league_avg_pace", 68.0),
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
        """No-op — adjusted efficiencies are rebuilt in bulk by the seed script.

        A (score_a, score_b) pair carries neither the box-score detail needed
        for possessions nor the schedule context the opponent-adjustment
        solver requires. Run scripts/seed_ncaab_efficiency.py weekly in season.
        """
        return
