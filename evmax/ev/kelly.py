"""Fractional Kelly Criterion bet sizing.

K_full = (p × b - q) / b
  where b = payout - 1, p = true_prob, q = 1 - p

Adjusted Kelly applies three discounts:
  1. Base fraction (0.25) — quarter Kelly for Phase 1
  2. Confidence discount — scales with edge, maxes at 100% at 20% edge
  3. Liquidity discount — penalizes wide spreads

Final K is hard-capped at max_kelly_fraction (default 5% of bankroll).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class KellyResult:
    kelly_full: float
    kelly_fraction: float  # after all discounts
    confidence_discount: float
    liquidity_discount: float
    suggested_units: float  # same as kelly_fraction


def kelly_full(true_prob: float, payout_decimal: float) -> float:
    """
    Compute full Kelly fraction.

    Args:
        true_prob: True probability of winning (0.0–1.0).
        payout_decimal: Decimal payout (e.g. 2.5 means win 1.5x stake).

    Returns:
        Full Kelly fraction (may be negative for -EV bets).
    """
    if payout_decimal <= 1.0:
        return 0.0
    b = payout_decimal - 1.0  # net odds
    q = 1.0 - true_prob
    return (true_prob * b - q) / b


def compute_kelly(
    true_prob: float,
    payout_decimal: float,
    edge_pct: float,
    spread_pct: float = 0.0,
    base_fraction: float = 0.25,
    max_kelly: float = 0.05,
    min_kelly: float = 0.0,
) -> KellyResult:
    """
    Compute adjusted fractional Kelly bet size.

    Args:
        true_prob: Devigged true probability.
        payout_decimal: Market payout (1 / market_price).
        edge_pct: EV edge as a fraction (e.g. 0.05 for 5%).
        spread_pct: Bid-ask spread as fraction — used for liquidity discount.
        base_fraction: Base Kelly multiplier (0.25 = quarter Kelly).
        max_kelly: Hard cap on fraction of bankroll.

    Returns:
        KellyResult with full and adjusted Kelly fractions.
    """
    # Guard against NaN/Inf from upstream model drift or bad data
    if not math.isfinite(true_prob) or not math.isfinite(payout_decimal):
        return KellyResult(
            kelly_full=0.0,
            kelly_fraction=0.0,
            confidence_discount=0.0,
            liquidity_discount=0.0,
            suggested_units=0.0,
        )

    k_full = kelly_full(true_prob, payout_decimal)

    if k_full <= 0:
        return KellyResult(
            kelly_full=k_full,
            kelly_fraction=0.0,
            confidence_discount=0.0,
            liquidity_discount=0.0,
            suggested_units=0.0,
        )

    # Confidence discount removed — Pinnacle devigged edges are trusted directly.
    # Edge size is already baked into the full Kelly fraction.
    confidence_discount = 1.0

    # Liquidity discount: spread_pct > 0 means thinner market
    # Discount = max(0.25, 1.0 - spread_pct × 5)
    liquidity_discount = max(0.25, 1.0 - spread_pct * 5.0)

    k_adjusted = k_full * base_fraction * confidence_discount * liquidity_discount
    k_final = min(k_adjusted, max_kelly)
    k_final = max(k_final, min_kelly)

    return KellyResult(
        kelly_full=k_full,
        kelly_fraction=k_final,
        confidence_discount=confidence_discount,
        liquidity_discount=liquidity_discount,
        suggested_units=k_final,
    )
