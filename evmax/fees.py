"""Venue trading-fee models for Kalshi and Polymarket US.

Both venues charge a symmetric quadratic fee on expected earnings:

    fee = rate × contracts × p × (1 − p)

where ``p`` is the execution price in dollars (0.01–0.99). What differs is
the rate, the maker treatment, and the rounding rule:

Kalshi (fee schedule, July 2026):
    - Taker rate 0.07, rounded UP to the next cent per order.
    - Maker fees apply only on designated series, at 25% of the taker rate
      (``KALSHI_MAKER_RATE_MULT`` — encoded from the published schedule;
      re-verify against https://kalshi.com/fee-schedule when relying on the
      maker path, the PDF is the authority and rates change per series).

Polymarket US (docs.polymarket.us/fees, effective 2026-07-01):
    - Taker theta 0.06, maker theta −0.0125 (a REBATE — makers are paid).
    - Rounded to the nearest cent, banker's rounding (half to even).

For EV/arb math we usually want the *unrounded per-contract* fee in
probability units (the ``*_prob`` helpers); the rounded dollar helpers
mirror what the venue actually charges on an order.
"""

from __future__ import annotations

import math

KALSHI_TAKER_RATE = 0.07
KALSHI_MAKER_RATE_MULT = 0.25  # × taker rate, designated series only
POLYMARKET_US_TAKER_THETA = 0.06
POLYMARKET_US_MAKER_THETA = -0.0125


def _quad(rate: float, price: float, contracts: float) -> float:
    if not (0.0 < price < 1.0):
        raise ValueError(f"price must be in (0, 1), got {price}")
    if contracts < 0:
        raise ValueError(f"contracts must be >= 0, got {contracts}")
    return rate * contracts * price * (1.0 - price)


# ---------------------------------------------------------------------------
# Per-contract, probability units (unrounded) — for EV / arb edge math
# ---------------------------------------------------------------------------

def kalshi_fee_prob(price: float, maker: bool = False) -> float:
    """Kalshi fee per contract in probability units, unrounded."""
    rate = KALSHI_TAKER_RATE * (KALSHI_MAKER_RATE_MULT if maker else 1.0)
    return _quad(rate, price, 1.0)


def polymarket_us_fee_prob(price: float, maker: bool = False) -> float:
    """Polymarket US fee per contract in probability units, unrounded.

    Negative when ``maker=True`` — makers receive a rebate.
    """
    theta = POLYMARKET_US_MAKER_THETA if maker else POLYMARKET_US_TAKER_THETA
    return _quad(theta, price, 1.0)


def venue_fee_prob(venue: str, price: float, maker: bool = False) -> float:
    """Dispatch per-contract fee by venue name (``kalshi``/``polymarket_us``)."""
    v = venue.lower()
    if v == "kalshi":
        return kalshi_fee_prob(price, maker=maker)
    if v == "polymarket_us":
        return polymarket_us_fee_prob(price, maker=maker)
    raise ValueError(f"unknown venue: {venue!r}")


# ---------------------------------------------------------------------------
# Order-level dollar fees (venue rounding applied)
# ---------------------------------------------------------------------------

def kalshi_order_fee(price: float, contracts: float, maker: bool = False) -> float:
    """Dollar fee Kalshi charges on one order — rounded UP to the cent."""
    raw = _quad(
        KALSHI_TAKER_RATE * (KALSHI_MAKER_RATE_MULT if maker else 1.0),
        price,
        contracts,
    )
    return math.ceil(raw * 100.0 - 1e-9) / 100.0


def _bankers_round_cents(amount: float) -> float:
    """Round to the nearest cent, half to even (Polymarket US rule)."""
    cents = amount * 100.0
    floor = math.floor(cents)
    frac = cents - floor
    if abs(frac - 0.5) < 1e-9:
        rounded = floor if floor % 2 == 0 else floor + 1
    else:
        rounded = math.floor(cents + 0.5)
    return rounded / 100.0


def polymarket_us_order_fee(
    price: float, contracts: float, maker: bool = False
) -> float:
    """Dollar fee (or negative rebate) Polymarket US applies on one order."""
    theta = POLYMARKET_US_MAKER_THETA if maker else POLYMARKET_US_TAKER_THETA
    raw = _quad(theta, price, contracts)
    sign = -1.0 if raw < 0 else 1.0
    return sign * _bankers_round_cents(abs(raw))
