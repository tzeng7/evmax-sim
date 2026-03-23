"""Pinnacle Power Method devigging.

Industry-standard approach that handles favorite/underdog asymmetry by finding
the exponent k such that sum(raw_prob_i ^ k) == 1.0.

Supports 2-way markets (most sports) and 3-way markets (soccer with draw).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import brentq

_log = logging.getLogger(__name__)


@dataclass
class DevigResult:
    true_probs: list[float]
    margin: float  # vig as fraction, e.g. 0.04 = 4%
    k: float  # power exponent found


def _objective(k: float, raw_probs: list[float]) -> float:
    """f(k) = sum(p_i^k) - 1.  Root gives us devigged probs."""
    return sum(p**k for p in raw_probs) - 1.0


def devig_power_method(
    decimals: list[float],
    k_bracket: tuple[float, float] = (0.01, 20.0),
) -> DevigResult:
    """
    Apply the Power Method to remove vig from decimal odds.

    Args:
        decimals: List of decimal odds for each outcome (2 or 3 outcomes).
                  e.g. [1.91, 2.05] for a two-way market.
        k_bracket: Search range for the exponent k.

    Returns:
        DevigResult with true_probs summing to 1.0, margin, and k.

    Raises:
        ValueError: If fewer than 2 outcomes provided or odds <= 1.0.
    """
    if len(decimals) < 2:
        raise ValueError("Need at least 2 decimal odds")
    if any(d <= 1.0 for d in decimals):
        raise ValueError(f"All decimal odds must be > 1.0, got {decimals}")

    raw_probs = [1.0 / d for d in decimals]
    overround = sum(raw_probs)
    margin = overround - 1.0

    # Find k such that sum(raw_prob^k) = 1
    try:
        k = brentq(_objective, k_bracket[0], k_bracket[1], args=(raw_probs,))
    except ValueError:
        # Fallback: proportional devig (multiplicative)
        _log.warning(
            "devig_power_method_fallback: brentq failed for decimals=%s, "
            "using proportional devig (k=1.0)",
            decimals,
        )
        k = 1.0
        true_probs = [p / overround for p in raw_probs]
        return DevigResult(true_probs=true_probs, margin=margin, k=k)

    true_probs = [p**k for p in raw_probs]

    # Normalise to handle floating-point drift
    total = sum(true_probs)
    true_probs = [p / total for p in true_probs]

    return DevigResult(true_probs=true_probs, margin=margin, k=k)


def devig_two_way(
    decimal_a: float,
    decimal_b: float,
) -> tuple[float, float, float]:
    """
    Convenience wrapper for standard two-way markets.

    Returns:
        (true_prob_a, true_prob_b, margin)
    """
    result = devig_power_method([decimal_a, decimal_b])
    return result.true_probs[0], result.true_probs[1], result.margin


def devig_three_way(
    decimal_a: float,
    decimal_b: float,
    decimal_draw: float,
) -> tuple[float, float, float, float]:
    """
    Convenience wrapper for three-way markets (soccer).

    Returns:
        (true_prob_a, true_prob_b, true_prob_draw, margin)
    """
    result = devig_power_method([decimal_a, decimal_b, decimal_draw])
    return result.true_probs[0], result.true_probs[1], result.true_probs[2], result.margin


def american_to_decimal(american: int) -> float:
    """Convert American odds to decimal.

    Raises:
        ValueError: If american == 0 (invalid odds) or would produce <= 1.0.
    """
    if american == 0:
        raise ValueError("American odds cannot be 0")
    if american > 0:
        return 1.0 + american / 100.0
    else:
        return 1.0 + 100.0 / abs(american)


def decimal_to_american(decimal: float) -> int:
    """Convert decimal odds to American.

    Raises:
        ValueError: If decimal <= 1.0 (no profit possible).
    """
    if decimal <= 1.0:
        raise ValueError(f"Decimal odds must be > 1.0, got {decimal}")
    if decimal >= 2.0:
        return int((decimal - 1) * 100)
    else:
        return int(-100 / (decimal - 1))
