"""Pinnacle Power Method devigging.

Industry-standard approach that handles favorite/underdog asymmetry by finding
the exponent k such that sum(raw_prob_i ^ k) == 1.0.

Supports 2-way markets (most sports) and 3-way markets (soccer with draw).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import brentq

_log = logging.getLogger(__name__)

# Selectable devig methods. Power is the shipped default (its favorite/underdog
# exponent already handles asymmetry); shin and multiplicative are A/B
# alternatives — see settings.devig_method and scripts/backtest_devig_ab.py.
DEVIG_METHODS = ("power", "shin", "multiplicative")

# Per-sector devig, baked from CLV evidence (scripts/backtest_devig_ab.py) — the
# same "set once in code, never flip at runtime" pattern as
# ensemble_agent.SECTOR_WEIGHT_OVERRIDES. Empty = power everywhere, the shipped
# state, so the operator changes NOTHING out of the box. To promote a sector to
# a non-power method after its CLV clears, add ONE line here (e.g.
# {"soccer": "shin"}); it is then permanent and automatic. The global
# settings.devig_method is only the fallback default / A/B experiment override.
DEVIG_METHOD_BY_SECTOR: dict[str, str] = {}


def resolve_devig_method(sector: Optional[str], default: str = "power") -> str:
    """The devig method for ``sector``: its baked-in override, else ``default``.

    ``default`` is the global fallback (settings.devig_method). This is what the
    Pinnacle client calls per line, so each sector automatically gets its
    validated method with no runtime configuration.
    """
    return DEVIG_METHOD_BY_SECTOR.get((sector or "").lower(), default or "power")


@dataclass
class DevigResult:
    true_probs: list[float]
    margin: float  # vig as fraction, e.g. 0.04 = 4%
    k: float  # power exponent found (1.0 for non-power methods)
    # Shin insider-trading fraction z (None for non-Shin methods). Exposed so a
    # caller/backtest can inspect how much longshot-shading Shin applied.
    z: Optional[float] = None
    method: str = "power"


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


def devig_multiplicative(decimals: list[float]) -> DevigResult:
    """Proportional (multiplicative) devig: ``p_i = (1/d_i) / Σ(1/d_j)``.

    The simplest method — divide out the booksum. Keeps every outcome's share
    of the overround identical, so it does NOT correct favorite-longshot bias
    (unlike power and shin). Included as an A/B baseline.
    """
    if len(decimals) < 2:
        raise ValueError("Need at least 2 decimal odds")
    if any(d <= 1.0 for d in decimals):
        raise ValueError(f"All decimal odds must be > 1.0, got {decimals}")
    raw = [1.0 / d for d in decimals]
    total = sum(raw)
    return DevigResult(
        true_probs=[p / total for p in raw],
        margin=total - 1.0,
        k=1.0,
        z=None,
        method="multiplicative",
    )


def devig_shin(
    decimals: list[float],
    max_iter: int = 200,
    tol: float = 1e-12,
) -> DevigResult:
    """Shin (1992, 1993) devig.

    Models the book as facing a fraction ``z`` of insider traders and backs out
    the true probabilities that shade LONGSHOTS DOWN and FAVOURITES UP relative
    to the proportional devig — the standard favorite-longshot-bias correction.
    Solves ``Σ_i p_i = 1`` for ``z ∈ [0, 1)`` where::

        p_i = (sqrt(z² + 4(1 − z)·π_i²/o) − z) / (2(1 − z))

    with ``π_i = 1/d_i`` the raw inverse odds and ``o = Σ π_i`` the booksum.
    Reduces to the proportional devig (``π_i/o``, ``z ≈ 0``) for a fair or
    near-fair book. On a non-vigged/arbitrage book (``o ≤ 1``) or if the root
    solve fails, falls back to :func:`devig_multiplicative`.
    """
    if len(decimals) < 2:
        raise ValueError("Need at least 2 decimal odds")
    if any(d <= 1.0 for d in decimals):
        raise ValueError(f"All decimal odds must be > 1.0, got {decimals}")

    raw = [1.0 / d for d in decimals]
    o = sum(raw)
    margin = o - 1.0
    n = len(raw)

    if o <= 1.0 + 1e-12:
        # No overround to remove — Shin's z collapses to 0.
        mult = devig_multiplicative(decimals)
        return DevigResult(
            true_probs=mult.true_probs, margin=margin, k=1.0, z=0.0, method="shin"
        )

    def _sum_sqrt(z: float) -> float:
        return sum(
            math.sqrt(z * z + 4.0 * (1.0 - z) * (p * p) / o) for p in raw
        )

    # Σ p_i = 1  ⇔  Σ sqrt(...) = 2 + (n − 2)·z.
    def _f(z: float) -> float:
        return _sum_sqrt(z) - (2.0 + (n - 2) * z)

    try:
        # f(0) = 2(sqrt(o) − 1) > 0 for an over-round book; f is negative just
        # below the degenerate z=1 root, so the interior insider fraction is
        # bracketed on (0, 1).
        z = brentq(_f, 1e-12, 1.0 - 1e-9, maxiter=max_iter, xtol=tol)
    except (ValueError, RuntimeError):
        _log.warning(
            "devig_shin_fallback: root solve failed for decimals=%s, "
            "using multiplicative devig",
            decimals,
        )
        mult = devig_multiplicative(decimals)
        return DevigResult(
            true_probs=mult.true_probs, margin=margin, k=1.0, z=None, method="shin"
        )

    denom = 2.0 * (1.0 - z)
    true_probs = [
        (math.sqrt(z * z + 4.0 * (1.0 - z) * (p * p) / o) - z) / denom for p in raw
    ]
    # Normalise out any floating-point drift.
    total = sum(true_probs)
    true_probs = [p / total for p in true_probs]
    return DevigResult(true_probs=true_probs, margin=margin, k=1.0, z=z, method="shin")


def devig(decimals: list[float], method: str = "power") -> DevigResult:
    """Devig ``decimals`` with the named method (``power``/``shin``/``multiplicative``).

    ``power`` is the shipped default. An unknown method falls back to power (and
    logs) rather than raising — a stray config value must never crash the scan.
    """
    m = (method or "power").lower()
    if m == "power":
        return devig_power_method(decimals)
    if m == "shin":
        return devig_shin(decimals)
    if m == "multiplicative":
        return devig_multiplicative(decimals)
    _log.warning("devig_unknown_method=%s, falling back to power", method)
    return devig_power_method(decimals)


def devig_two_way(
    decimal_a: float,
    decimal_b: float,
    method: str = "power",
) -> tuple[float, float, float]:
    """
    Convenience wrapper for standard two-way markets.

    ``method`` selects the devig algorithm (default ``power`` — unchanged).

    Returns:
        (true_prob_a, true_prob_b, margin)
    """
    result = devig([decimal_a, decimal_b], method=method)
    return result.true_probs[0], result.true_probs[1], result.margin


def devig_three_way(
    decimal_a: float,
    decimal_b: float,
    decimal_draw: float,
    method: str = "power",
) -> tuple[float, float, float, float]:
    """
    Convenience wrapper for three-way markets (soccer).

    ``method`` selects the devig algorithm (default ``power`` — unchanged).

    Returns:
        (true_prob_a, true_prob_b, true_prob_draw, margin)
    """
    result = devig([decimal_a, decimal_b, decimal_draw], method=method)
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


# Share of a regulation draw resolved with the favorite's edge intact (extra
# time) vs a near coin flip (penalties). Historically ~45-50% of knockout ET
# draws reach pens; 0.55 also minimizes error against Kalshi's liquid
# KXWCADVANCE mids (6-game R16/QF calibration 2026-07-05: mean abs error
# 2.0pp shaded vs 3.4pp pure-ratio, near-exact on the lopsided ties
# ARG-EGY/FRA-MAR/POR-ESP where the two formulas actually differ).
ADVANCE_ET_SHARE = 0.55


def derive_advance_prob(
    p_a: Optional[float], p_b: Optional[float], p_draw: Optional[float],
) -> Optional[float]:
    """P(team A advances a knockout tie) from regulation 3-way probabilities.

    A advances by winning in 90', or by surviving a regulation draw through
    extra time / penalties. Conditional on the draw, the win share blends the
    no-draw strength ratio r = p_a / (p_a + p_b) (extra time — the favorite's
    edge persists) with a coin flip (penalties):

        P(A advances) = p_a + p_draw * (0.55 * r + 0.45 * 0.5)

    Complements sum to 1 by construction. Used on BOTH sides of the advance
    EV gap: on devigged Pinnacle regulation probs to synthesize the sharp
    anchor (Pinnacle's own "To Reach X" specials cut off at the team's
    PREVIOUS kickoff and never reopen, so there is no live per-match advance
    special — verified 2026-07-05), and on the model blend's 3-way to produce
    the model advance prob.
    """
    if p_a is None or p_b is None:
        return None
    denom = p_a + p_b
    if denom <= 0:
        return None
    ratio = p_a / denom
    draw_share = ADVANCE_ET_SHARE * ratio + (1.0 - ADVANCE_ET_SHARE) * 0.5
    return p_a + (p_draw or 0.0) * draw_share
