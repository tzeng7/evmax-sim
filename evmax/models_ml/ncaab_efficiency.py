"""NCAAB Efficiency Model — Phase 2 probability model.

Uses Barttorvik T-Rank efficiency ratings (AdjOE, AdjDE, AdjT) to estimate
true win probability for NCAAB moneyline markets independently of sharp books.

Formula
-------
avg_possessions  = (home_AdjT + away_AdjT) / 2

Expected scoring (per 100 possessions, adjusted for opponent quality):
  home_pts_per_100 = home_AdjOE × (away_AdjDE / 100)
  away_pts_per_100 = away_AdjOE × (home_AdjDE / 100)

Scale to full game:
  home_pts = avg_possessions × home_pts_per_100 / 100
           = avg_possessions × home_AdjOE × away_AdjDE / 10000
  away_pts = avg_possessions × away_AdjOE × home_AdjDE / 10000

Expected margin (home perspective):
  expected_margin = (home_pts − away_pts) + HOME_COURT_ADV

Convert to win probability:
  P(home wins) = Φ(expected_margin / σ)
  σ = 12.5 (empirical NCAAB point-margin std dev, from spread_distribution.py)

Home court advantage: ~3.0 pts (NCAAB empirical average at campus sites).
Set to 0.0 for neutral-court games (NCAA Tournament venues).

Blend weight: 0.4 (SharpBooksModel carries 0.6 in the ensemble).
"""

from __future__ import annotations

from typing import Optional

import structlog
from scipy.stats import norm

from evmax.data.barttorvik import fetch_team_efficiencies
from evmax.models.market import PredictionMarket
from evmax.models.odds import SharpOdds
from evmax.models_ml.base import ModelBase, ModelPrediction

logger = structlog.get_logger(__name__)

_SIGMA = 12.5         # NCAAB point-margin std dev
_HOME_COURT_ADV = 3.0  # pts; use 0.0 for neutral-site games


def _is_neutral_site(market: PredictionMarket) -> bool:
    """
    Heuristic: treat as neutral site if title contains "neutral" or common
    tournament keywords. For now, NCAA Tournament games in March/April are
    played at neutral sites.
    """
    title = (market.title or "").lower()
    keywords = ("ncaa tournament", "march madness", "sweet 16", "elite 8",
                "final four", "neutral", "first four", "first round",
                "second round")
    return any(kw in title for kw in keywords)


class NCAABEfficiencyModel(ModelBase):
    """
    NCAAB win probability model using Barttorvik efficiency ratings.

    Phase 2 model blended with SharpBooksModel at 40% weight.
    Only activates for moneyline markets where both teams have T-Rank data.
    Falls back gracefully (returns None) when data is unavailable.
    """

    name = "ncaab_efficiency"
    weight = 0.4

    async def predict(
        self,
        market: PredictionMarket,
        sharp_odds: SharpOdds,
    ) -> Optional[ModelPrediction]:
        """
        Compute home/away win probabilities from team efficiency ratings.

        The market's outcome_a is the home team (Pinnacle convention).
        Returns ModelPrediction where true_prob_a = P(home wins).
        """
        if not market.team_home or not market.team_away:
            logger.debug("ncaab_eff_no_teams", market_id=market.id)
            return None

        efficiencies = await fetch_team_efficiencies()
        if not efficiencies:
            logger.debug("ncaab_eff_no_data", market_id=market.id)
            return None

        home_eff = efficiencies.get(market.team_home)
        away_eff = efficiencies.get(market.team_away)

        if home_eff is None:
            logger.debug(
                "ncaab_eff_missing_home",
                team=market.team_home,
                market_id=market.id,
            )
            return None

        if away_eff is None:
            logger.debug(
                "ncaab_eff_missing_away",
                team=market.team_away,
                market_id=market.id,
            )
            return None

        avg_tempo = (home_eff.adj_t + away_eff.adj_t) / 2.0

        # Expected points per game
        home_pts = avg_tempo * home_eff.adj_oe * away_eff.adj_de / 10000.0
        away_pts = avg_tempo * away_eff.adj_oe * home_eff.adj_de / 10000.0

        home_adv = 0.0 if _is_neutral_site(market) else _HOME_COURT_ADV
        expected_margin = (home_pts - away_pts) + home_adv

        # P(home wins) = P(margin > 0) = Φ(expected_margin / σ)
        home_win_prob = float(norm.cdf(expected_margin / _SIGMA))
        home_win_prob = max(0.02, min(0.98, home_win_prob))
        away_win_prob = 1.0 - home_win_prob

        logger.debug(
            "ncaab_eff_prediction",
            market_id=market.id,
            home=market.team_home,
            away=market.team_away,
            home_adjoe=round(home_eff.adj_oe, 1),
            away_adjde=round(away_eff.adj_de, 1),
            away_adjoe=round(away_eff.adj_oe, 1),
            home_adjde=round(home_eff.adj_de, 1),
            avg_tempo=round(avg_tempo, 1),
            home_pts=round(home_pts, 1),
            away_pts=round(away_pts, 1),
            expected_margin=round(expected_margin, 2),
            home_win_prob=round(home_win_prob, 3),
            neutral_site=home_adv == 0.0,
        )

        return ModelPrediction(
            true_prob_a=home_win_prob,
            true_prob_b=away_win_prob,
            confidence=0.85,
            model_name=self.name,
        )
