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


TOTAL_DIST_TOKEN = "total_dist"    # normal curve off the matched Pinnacle total
TOTAL_PMF_TOKEN = "total_pmf"      # fitted LinePMF off the matched Pinnacle total
TOTAL_LADDER_TOKEN = "sharp_ladder"  # Pinnacle's own price for this exact rung

# Layered totals (2026-11-16). For these sectors a total is priced, in order:
#   1. Pinnacle's own price when the matched Pinnacle total is this EXACT line
#      (TOTAL_LADDER_TOKEN): the book's ladder, no curve at all;
#   2. the fitted LinePMF anchored on the matched Pinnacle total (TOTAL_PMF_TOKEN),
#      when data/models/line_pmf/{sector}_total.json exists;
#   3. the normal curve (TOTAL_DIST_TOKEN), as before.
# The possession-sim overlay is not applied in these sectors: the fair value is
# the sharp book's distribution, read or reconstructed.
TOTAL_LADDER_SECTORS: set[str] = {"nfl", "nba", "ncaab", "ncaaw"}
TOTAL_LADDER_EXACT_TOLERANCE: float = 0.01


def total_ladder_enabled(sector: Optional[str]) -> bool:
    """True when ``sector`` prices totals with the layered stack."""
    return (sector or "").lower() in TOTAL_LADDER_SECTORS


def total_rung_is_exact(sector: Optional[str], pinnacle_line: Optional[float],
                        target_line: Optional[float]) -> bool:
    """True when the matched Pinnacle total IS the venue's line (layered sectors only)."""
    if not total_ladder_enabled(sector) or pinnacle_line is None or target_line is None:
        return False
    return abs(float(pinnacle_line) - float(target_line)) <= TOTAL_LADDER_EXACT_TOLERANCE


@dataclass
class TotalPrediction:
    true_prob: float       # P(YES side is correct: over or under)
    implied_mean: float    # inferred mean total score
    sigma: float           # standard deviation used
    method: str = TOTAL_DIST_TOKEN   # model_sources token for the pricing path


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

        # Layered sectors: the fitted LinePMF replaces the normal curve when an
        # artifact exists. Same gate (applied above), same anchor: the matched
        # Pinnacle total and its devigged over probability.
        if total_ladder_enabled(sector):
            from evmax.models_ml.line_pmf import load_line_pmf

            pmf = load_line_pmf(sector, "total")
            p_over_target = (
                pmf.over_probability(pinnacle_line, prob_over, target_line) if pmf else None
            )
            if p_over_target is not None:
                true_prob = (1.0 - p_over_target) if yes_is_under else p_over_target
                return TotalPrediction(
                    true_prob=max(0.01, min(0.99, true_prob)),
                    implied_mean=pmf.anchor(pinnacle_line, prob_over) or pinnacle_line,
                    sigma=pmf.scale(pinnacle_line),
                    method=TOTAL_PMF_TOKEN,
                )

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
