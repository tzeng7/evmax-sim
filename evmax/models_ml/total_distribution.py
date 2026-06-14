"""Total Distribution Model.

Estimates true probability for a specific over/under total line using a normal
distribution of total game scores, calibrated from Pinnacle's posted line.

Method:
  1. Infer the implied mean total score from Pinnacle's devigged true_prob_over
     and posted total_line.
  2. Use P(total > target_line) = 1 - Φ((target_line - μ) / σ) to compute
     the true probability for any line (e.g. Kalshi's line differs by 0.5–1.5 pts).

Typical total-score standard deviations per sector:
  NBA:      ~22 pts  (game totals typically 210–230)
  NFL:      ~14 pts  (game totals typically 42–52)
  NCAAB:    ~14 pts  (game totals typically 130–155)
  Soccer:   ~2.5 goals
  Baseball: ~3.0 runs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import structlog
from scipy.stats import norm

from evmax.models.odds import SharpOdds

logger = structlog.get_logger(__name__)

# Empirical standard deviations of total game score per sector
_TOTAL_SIGMA: dict[str, float] = {
    "nba":      22.0,
    "wnba":     16.0,    # WNBA totals avg ~155-165; lower variance than NBA
    "nfl":      14.0,
    "ncaab":    14.0,
    "soccer":   2.5,
    "baseball": 3.0,
    "default":  18.0,
}

# Per-sector cap on how far a target (Kalshi) line may sit from Pinnacle's
# posted line before we refuse to price it. Defaults to 1.0 * sigma (the
# extrapolation gets unreliable in the tails). Baseball overrides to a tight
# absolute 1.0 run: combined with main-line-only ingestion
# (TOTALS_MAIN_LINE_ONLY_SECTORS), this keeps each game to its standard total
# and rejects the deep alternate O/U ladder rather than flooding the scan with
# tail lines off a skewed run distribution. Sectors not listed fall through to
# the sigma-based default.
_MAX_LINE_DISTANCE: dict[str, float] = {
    "baseball": 1.0,
}

# Minimum total line to be a game total (not a team scoring prop).
# e.g. "Will Lakers score over 109.5?" → team prop, not game total.
# Game total floors: NBA ~170 (team props ~100-120), NCAAB ~110 (team props ~60-75).
_GAME_TOTAL_FLOOR: dict[str, float] = {
    "nba":      170.0,
    "wnba":     130.0,   # WNBA team scores ~75-85, game totals ~155+
    "nfl":       35.0,
    "ncaab":    110.0,
    "soccer":     1.5,
    "baseball":   6.0,
    "default":   30.0,
}


@dataclass
class TotalPrediction:
    true_prob: float       # P(YES side is correct: over or under)
    implied_mean: float    # inferred mean total score
    sigma: float           # standard deviation used


class TotalDistributionModel:
    """Estimates over/under probability for any line given Pinnacle's posted total."""

    def predict(
        self,
        sharp_odds: SharpOdds,
        target_line: float,
        sector: str = "nba",
        yes_is_under: bool = False,
    ) -> Optional[TotalPrediction]:
        """
        Estimate P(YES side is correct) for a target total line.

        Uses a normal distribution of total game scores. The implied mean is
        inferred from Pinnacle's devigged true_prob_over and posted total_line.

        Rejects if target_line is more than 1 sigma from Pinnacle's posted line
        (distribution extrapolation becomes unreliable in the tails).

        Args:
            sharp_odds:   Pinnacle totals SharpOdds (total_line + true_prob_over set).
            target_line:  Kalshi total line (e.g. 220.5).
            sector:       Sector key for standard deviation lookup.
            yes_is_under: True when YES side is "under".

        Returns:
            TotalPrediction or None if line is too far / data missing.
        """
        if sharp_odds.total_line is None or sharp_odds.true_prob_over is None:
            logger.debug("total_model_missing_data", event_id=sharp_odds.event_id)
            return None

        prob_over = sharp_odds.true_prob_over
        if prob_over <= 0 or prob_over >= 1:
            return None

        sigma = _TOTAL_SIGMA.get(sector, _TOTAL_SIGMA["default"])
        pinnacle_line = sharp_odds.total_line

        # Reject lines that are too far from Pinnacle's posted line. Most
        # sectors use 1*sigma (tail extrapolation gets unreliable); capped
        # sectors (baseball) use a tight absolute distance so only the standard
        # total survives — see _MAX_LINE_DISTANCE.
        max_distance = _MAX_LINE_DISTANCE.get(sector, 1.0 * sigma)
        distance = abs(target_line - pinnacle_line)
        if distance > max_distance:
            logger.debug(
                "total_model_line_too_far",
                event_id=sharp_odds.event_id,
                pinnacle_line=pinnacle_line,
                target_line=target_line,
                distance=distance,
                max_distance=max_distance,
            )
            return None

        # Infer implied mean total score:
        #   P(total > pinnacle_line) = prob_over
        #   ⟹ (pinnacle_line - μ) / σ = Φ⁻¹(1 - prob_over)
        z_pinnacle = norm.ppf(1.0 - prob_over)
        implied_mean = pinnacle_line - z_pinnacle * sigma

        # P(total > target_line) = 1 - Φ((target_line - μ) / σ)
        z_target = (target_line - implied_mean) / sigma
        prob_over_target = float(1.0 - norm.cdf(z_target))

        true_prob = (1.0 - prob_over_target) if yes_is_under else prob_over_target

        return TotalPrediction(
            true_prob=max(0.01, min(0.99, true_prob)),
            implied_mean=implied_mean,
            sigma=sigma,
        )


def is_game_total(line: float, sector: str) -> bool:
    """Return True if the line is plausibly a game total (not a team scoring prop)."""
    floor = _GAME_TOTAL_FLOOR.get(sector, _GAME_TOTAL_FLOOR["default"])
    return line >= floor
