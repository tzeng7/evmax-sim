"""Field devigging for golf markets.

Golf markets come in two probability-mass shapes:

  * exactly-one-winner (``win``, ``rN_leader``) — true probs sum to 1.0;
  * non-exclusive (``top_5/10/20``, ``make_cut``) — true probs sum to N (or
    ~cut_size), because many golfers can each finish top-10 / make the cut.

So a single "normalise the field to 1" is only right for the winner market.
The venue-native fix is per-contract 2-way devig: both Kalshi and PolyUS quote
a YES *and* a NO ask for every golfer, so each contract's own vig is removed
with the Power Method (``evmax.ev.devig``) independent of what the field sums
to. That path is market-type-agnostic and is the default.

When only YES asks are available (thin Kalshi contracts), we fall back to
scaling the raw YES-implied field to the market's known probability mass
(1.0 for winner/leader, N for top-N, an explicit ``cut_target`` for make-cut).
"""

from __future__ import annotations

import logging
from typing import Optional

from evmax.ev.devig import devig_power_method, devig_two_way
from evmax.golf.models import FieldPrices, GolfMarketType

_log = logging.getLogger(__name__)


def devig_contract(yes_raw: float, no_raw: float) -> float:
    """Per-contract 2-way devig → calibrated true YES probability.

    ``yes_raw`` / ``no_raw`` are the vigged YES / NO ask-implied probs (0-1).
    Converts to decimal odds and removes the pair's vig with the Power Method.
    """
    if not (0.0 < yes_raw < 1.0) or not (0.0 < no_raw < 1.0):
        # Degenerate quote; fall back to the raw YES (renormalised against the
        # pair if possible) rather than raising inside a field loop.
        total = yes_raw + no_raw
        return yes_raw / total if total > 0 else yes_raw
    true_yes, _true_no, _margin = devig_two_way(1.0 / yes_raw, 1.0 / no_raw)
    return true_yes


def _market_mass(mt: GolfMarketType, cut_target: Optional[float]) -> Optional[float]:
    """Total true-probability mass a fully-devigged field should sum to."""
    if mt in (GolfMarketType.win, GolfMarketType.r1_leader,
              GolfMarketType.r2_leader, GolfMarketType.r3_leader):
        return 1.0
    if mt.top_n is not None:
        return float(mt.top_n)
    if mt is GolfMarketType.make_cut:
        return cut_target  # None → caller couldn't supply; skip normalisation
    return None


_EXACTLY_ONE_WINNER = (
    GolfMarketType.win,
    GolfMarketType.r1_leader,
    GolfMarketType.r2_leader,
    GolfMarketType.r3_leader,
)


def devig_field(
    field: FieldPrices,
    cut_target: Optional[float] = None,
    max_spread: float = 0.05,
) -> FieldPrices:
    """Populate ``field.devigged`` (canonical name → true prob) in place.

    Two shapes, handled differently:

    * exactly-one-winner (``win``/``rN_leader``): the true probs sum to 1, but a
      naive per-contract devig is poisoned by illiquid longshots' placeholder
      asks (a no-hoper quoted 0.45 ask would steal mass from the favourites).
      So we build a liquidity-robust implied per golfer (``robust_implied`` —
      mid when tight, bid when wide) and normalise the field to 1.0.

    * non-exclusive (``top_5/10/20``, ``make_cut``): each contract is its own
      YES/NO binary summing to the market's mass (N, or ~cut). Per-contract
      2-way devig where both sides are present; YES-only contracts are scaled to
      the remaining known mass.
    """
    priced = field.priced_markets()
    if not priced:
        field.devigged = {}
        field.overround = None
        return field

    field.overround = sum(m.implied_raw for m in priced)

    if field.market_type in _EXACTLY_ONE_WINNER:
        implied = {m.golfer: (m.robust_implied(max_spread) or m.implied_raw) for m in priced}
        s = sum(implied.values())
        field.devigged = {k: v / s for k, v in implied.items()} if s > 0 else implied
        return field

    # Non-exclusive markets.
    devigged: dict[str, float] = {}
    yes_only: list = []
    for m in priced:
        if m.has_both_sides:
            devigged[m.golfer] = devig_contract(m.implied_raw, m.implied_no_raw)
        else:
            yes_only.append(m)

    mass = _market_mass(field.market_type, cut_target)
    if yes_only:
        yes_sum = sum(m.implied_raw for m in yes_only)
        if mass is not None and yes_sum > 0:
            remaining = max(mass - sum(devigged.values()), 0.0)
            scale = remaining / yes_sum if remaining > 0 else 0.0
            for m in yes_only:
                devigged[m.golfer] = m.implied_raw * scale
        else:
            for m in yes_only:
                devigged[m.golfer] = m.implied_raw

    field.devigged = devigged
    return field
