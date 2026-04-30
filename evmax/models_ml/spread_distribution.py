"""Spread Distribution Model.

Estimates true probability for a specific spread line using a normal
distribution of point margins, calibrated from Pinnacle's posted line.

Method:
  1. Infer the implied mean of the scoring margin distribution from Pinnacle's
     devigged probability and posted spread line.
  2. Use P(margin > target_line) = 1 - Φ((target_line - μ) / σ) to estimate
     the true probability for any line (e.g., Kalshi's "wins by over X.5").

Typical NBA margin standard deviation: ~11.5 points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import structlog
from scipy.stats import norm

from evmax.models.odds import SharpOdds

logger = structlog.get_logger(__name__)

# Empirical point-margin standard deviations per sector
_SECTOR_SIGMA: dict[str, float] = {
    "nba": 11.5,
    "wnba": 12.5,    # matches WNBAPossessionSimAgent SCORE_STDEV; ~40-min games
    "nfl": 14.0,
    "ncaab": 12.5,
    "soccer": 1.9,   # goals, not points
    "default": 11.5,
}


@dataclass
class SpreadPrediction:
    true_prob: float      # P(yes_team covers target_line)
    implied_mean: float   # inferred scoring margin mean
    sigma: float          # standard deviation used


class SpreadDistributionModel:
    """Estimates cover probability for any line given Pinnacle's posted spread."""

    def predict(
        self,
        sharp_odds: SharpOdds,
        target_line: float,
        sector: str = "nba",
        yes_is_underdog: bool = False,
    ) -> Optional[SpreadPrediction]:
        """
        Estimate P(YES side covers target_line) using a normal distribution.

        Margin is defined as outcome_a_score - outcome_b_score (positive = a wins).

        Args:
            sharp_odds: Pinnacle spread SharpOdds (spread_line + true_prob_a set).
                        spread_line is outcome_a's line (e.g. -7.5 means a is -7.5 fav).
            target_line: Kalshi line from YES team's perspective (always negative,
                         e.g. -8.5 means YES team wins by more than 8.5 pts).
            sector: Sector key for standard deviation lookup.
            yes_is_underdog: True when the YES side is outcome_b (the underdog).

        Returns:
            SpreadPrediction with true_prob = P(YES side covers target_line).
        """
        if sharp_odds.spread_line is None:
            logger.debug("spread_model_no_line", event_id=sharp_odds.event_id)
            return None

        pinnacle_line = sharp_odds.spread_line   # e.g. -7.5 (outcome_a is favorite)
        true_prob_a = sharp_odds.true_prob_a     # devigged P(a covers pinnacle_line)

        if true_prob_a <= 0 or true_prob_a >= 1:
            return None

        # Reject Kalshi lines that are more than 1 sigma away from Pinnacle's line.
        # Beyond this range the normal distribution extrapolation becomes unreliable
        # (tail probabilities are very sensitive to small errors in the inferred mean).
        sigma = _SECTOR_SIGMA.get(sector, _SECTOR_SIGMA["default"])
        line_distance = abs(abs(target_line) - abs(pinnacle_line))
        if line_distance > 1.0 * sigma:
            logger.debug(
                "spread_model_line_too_far",
                event_id=sharp_odds.event_id,
                pinnacle_line=pinnacle_line,
                target_line=target_line,
                distance=line_distance,
            )
            return None

        # Infer implied mean margin for outcome_a:
        #   P(margin > |pinnacle_line|) = true_prob_a
        #   ⟹ (|pinnacle_line| - μ) / σ = Φ⁻¹(1 - true_prob_a)
        z_pinnacle = norm.ppf(1.0 - true_prob_a)
        implied_mean = abs(pinnacle_line) - z_pinnacle * sigma

        margin_threshold = abs(target_line)

        if not yes_is_underdog:
            # YES = outcome_a (favorite): P(margin > threshold)
            z_target = (margin_threshold - implied_mean) / sigma
            true_prob = float(1.0 - norm.cdf(z_target))
        else:
            # YES = outcome_b (underdog): P(margin < -threshold)
            # i.e. outcome_b wins by more than threshold
            z_target = (-margin_threshold - implied_mean) / sigma
            true_prob = float(norm.cdf(z_target))

        return SpreadPrediction(
            true_prob=max(0.01, min(0.99, true_prob)),
            implied_mean=implied_mean,
            sigma=sigma,
        )
