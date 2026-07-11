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
    "nba": 12.5,     # bumped 11.5→12.5 on 2026-05-07 — backtest_nba_score_stdev
                     # found Brier −0.00077 on 82 resolved spread bets, fixed
                     # tail over-confidence (>80% bucket bias +7.7→+8.4pp w/ no
                     # change at sigma=14.0 was overfitting noise; 12.5 captures
                     # ~60% of the gain and aligns with empirical NBA margin σ).
    "wnba": 12.5,    # matches WNBAPossessionSimAgent SCORE_STDEV; ~40-min games
    "nfl": 14.0,
    "ncaab": 12.5,
    "baseball": 4.0, # MLB run-margin σ — defense-in-depth; gated below
    "nhl": 2.0,      # NHL goal-margin σ — defense-in-depth; gated below
    "soccer": 1.9,   # goals, not points — gated below
    "default": 11.5,
}

# Low-scoring sports where margin distributions are non-Gaussian (Poisson/Skellam-
# like, with mass concentrated near 0 and a long thin tail). The normal-CDF
# extrapolation in this model is unreliable for these sectors: a 3-run jump from
# Pinnacle's MLB run line (−1.5) to a Kalshi alt-spread (−4.5) produced 165%+ EV
# at the default σ because Φ overstates tail mass. For these sectors, only allow
# predictions where the Kalshi line directly matches Pinnacle's posted line
# (within LOW_SCORING_LINE_TOLERANCE).
_LOW_SCORING_SECTORS: set[str] = {"baseball", "nhl", "soccer"}
LOW_SCORING_LINE_TOLERANCE: float = 0.5

# Maximum absolute alt-line magnitude we will price for a low-scoring sport, even
# when the sharp book posts that exact ladder (line_distance == 0 defeats the
# tolerance gate above). Empirical motivation: baseball −4.5 alt run lines went
# 2-for-15 live+shadow (Apr–Jun 2026) while the model predicted 27–46% cover.
# Devigging Pinnacle's own thin −4.5 ladder still overstates cover mass because
# the run-margin distribution is skewed/fat-tailed, not Gaussian, and the alt
# ladder itself is low-liquidity. We only price the standard line (run line /
# puck line / standard Asian handicap = 1.5) and tighter. This is the gate the
# `fix/baseball-alt-runline-gate` branch was meant to ship but never did.
_LOW_SCORING_MAX_ABS_LINE: dict[str, float] = {
    "baseball": 1.5,  # standard MLB run line; reject −2.5/−4.5 alt ladders
    "nhl": 1.5,       # standard puck line
    "soccer": 1.5,    # standard Asian handicap ceiling
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

        sigma = _SECTOR_SIGMA.get(sector, _SECTOR_SIGMA["default"])
        line_distance = abs(abs(target_line) - abs(pinnacle_line))

        # Low-scoring sports: cap the absolute alt-line magnitude we will price.
        # The tolerance gate below keys off distance from Pinnacle's posted line,
        # but sharp books post their own −4.5 alt ladders (distance == 0), so a
        # far-from-pickem runline sails through. The normal-CDF cover prob is
        # unreliable that deep into a skewed margin distribution regardless of
        # where the sharp line sits — reject it outright.
        max_abs = _LOW_SCORING_MAX_ABS_LINE.get(sector)
        if max_abs is not None and abs(target_line) > max_abs:
            logger.debug(
                "spread_model_low_scoring_alt_line_too_large",
                event_id=sharp_odds.event_id,
                sector=sector,
                target_line=target_line,
                max_abs=max_abs,
            )
            return None

        # Low-scoring sports: margin distribution is not Gaussian, so extrapolation
        # is unsafe even within 1σ. Require a direct line match (±0.5).
        if sector in _LOW_SCORING_SECTORS and line_distance > LOW_SCORING_LINE_TOLERANCE:
            logger.debug(
                "spread_model_low_scoring_alt_line_skipped",
                event_id=sharp_odds.event_id,
                sector=sector,
                pinnacle_line=pinnacle_line,
                target_line=target_line,
                distance=line_distance,
            )
            return None

        # Reject Kalshi lines that are more than 1 sigma away from Pinnacle's line.
        # Beyond this range the normal distribution extrapolation becomes unreliable
        # (tail probabilities are very sensitive to small errors in the inferred mean).
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

        # YES covers iff (yes_team_score + target_line) > opponent_score.
        # In terms of margin = outcome_a_score - outcome_b_score:
        #   YES = outcome_a: covers iff margin > -target_line
        #   YES = outcome_b: covers iff target_line > margin, i.e. margin < target_line
        # target_line's SIGN must be preserved here (not abs()'d) — for the
        # common underdog case target_line is POSITIVE (e.g. +17.5, the
        # points they're getting), and using abs() silently flipped the
        # condition to "underdog wins outright by more than 17.5", which
        # produced near-zero cover probabilities for ordinary underdog
        # spreads instead of the correct near-certain ones. Caught 2026-07-10
        # via a WNBA Polymarket US +17.5 underdog line scoring 1.6% instead
        # of the correct ~74%.
        if not yes_is_underdog:
            # YES = outcome_a (favorite): P(margin > -target_line)
            z_target = (-target_line - implied_mean) / sigma
            true_prob = float(1.0 - norm.cdf(z_target))
        else:
            # YES = outcome_b (underdog): P(margin < target_line)
            z_target = (target_line - implied_mean) / sigma
            true_prob = float(norm.cdf(z_target))

        return SpreadPrediction(
            true_prob=max(0.01, min(0.99, true_prob)),
            implied_mean=implied_mean,
            sigma=sigma,
        )
