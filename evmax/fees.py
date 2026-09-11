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

Novig (P2P exchange, CFTC-designated contract market since 2026-08-04):
    - Same quadratic entry-fee form as Kalshi/Poly. Taker theta 0.03
      (peaks at $0.0075/contract at p=0.5 — the published cap). Makers pay
      nothing and earn a credit up to HALF the taker fee, so the maker path is
      a rebate at ``-0.5 ×`` the taker rate (``NOVIG_MAKER_RATE_MULT``).
      Sources: oddsassist.com/prediction-markets/novig-fees, predictionscout.

ProphetX (P2P exchange, CFTC-regulated):
    - NOT an entry fee. A COMMISSION on NET WINNINGS, charged at settlement on
      WINNING contracts only (nothing on losses or unmatched orders) — the
      betting-exchange model. Standard rate 2% (``PROPHETX_COMMISSION_RATE``;
      VIP 1.5%). Maker and taker pay the same commission.
      Sources: prophethelp.zendesk.com "Understanding Commission",
      legalsportsreport / predictionscout ProphetX reviews (2026).

      Because it applies only to the winning profit, its drag is much smaller
      than an entry fee at every price (~0.5pp of true prob at p=0.5 vs Kalshi
      1.75pp). It is expressed in the same units as the entry-fee venues via
      the breakeven-probability SHIFT it induces (see ``prophetx_fee_prob``),
      so the EV gate (``effective_price``) and arb math treat every venue
      uniformly. Realized P&L (``bet_pnl``) applies it as a winners-only profit
      commission — its true mechanism.

For EV/arb math we usually want the *unrounded per-contract* fee in
probability units (the ``*_prob`` helpers); the rounded dollar helpers
mirror what the venue actually charges on an order.
"""

from __future__ import annotations

import math
from typing import Optional

KALSHI_TAKER_RATE = 0.07
KALSHI_MAKER_RATE_MULT = 0.25  # × taker rate, designated series only
POLYMARKET_US_TAKER_THETA = 0.06
POLYMARKET_US_MAKER_THETA = -0.0125
NOVIG_TAKER_RATE = 0.03
NOVIG_MAKER_RATE_MULT = -0.5   # makers earn a rebate up to half the taker fee
PROPHETX_COMMISSION_RATE = 0.02  # on net winnings, winners only (VIP 1.5%)

# Venues whose fee is a commission on net WINNINGS (winners only), not a
# per-contract entry fee. Modeled as a breakeven-prob shift for EV/arb and a
# winners-only profit commission for realized P&L. Keyed by venue name.
_WINNINGS_COMMISSION_RATE: dict[str, float] = {
    "prophetx": PROPHETX_COMMISSION_RATE,
}


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


def novig_fee_prob(price: float, maker: bool = False) -> float:
    """Novig fee per contract in probability units, unrounded.

    Same quadratic form as Kalshi/Poly (``θ·p·(1−p)``) with taker θ=0.03. The
    maker path is a rebate at half the taker rate — negative when ``maker=True``.
    """
    rate = NOVIG_TAKER_RATE * (NOVIG_MAKER_RATE_MULT if maker else 1.0)
    return _quad(rate, price, 1.0)


def prophetx_fee_prob(
    price: float, rate: float = PROPHETX_COMMISSION_RATE
) -> float:
    """ProphetX commission expressed as a breakeven-probability shift.

    ProphetX charges no entry fee; it keeps ``rate`` of the NET WINNINGS on a
    winning contract. A contract bought at ask ``p`` pays ``(1/p − 1)`` profit,
    of which the buyer keeps ``(1 − rate)``. The buyer's breakeven true
    probability therefore rises from ``p`` (no fee) to::

        q = 1 / (1 + (1/p − 1)·(1 − rate))

    and the fee, in the same additive units the entry-fee venues use (a bump to
    the effective acquisition price — see ``effective_price``), is ``q − p``.
    Returns 0 at ``rate=0`` and is >= 0 for ``rate`` in [0, 1). Maker and taker
    pay the same commission on ProphetX, so there is no ``maker`` argument.
    """
    if not (0.0 < price < 1.0):
        raise ValueError(f"price must be in (0, 1), got {price}")
    a = (1.0 / price - 1.0) * (1.0 - rate)
    q = 1.0 / (1.0 + a)
    return q - price


def venue_fee_prob(venue: str, price: float, maker: bool = False) -> float:
    """Dispatch per-contract fee by venue name in probability units.

    For entry-fee venues (``kalshi``/``polymarket_us``/``novig``) this is the
    per-contract ``θ·p·(1−p)`` fee; for a winnings-commission venue
    (``prophetx``) it is the breakeven-probability SHIFT that commission
    induces, so ``effective_price`` and arb math price every venue uniformly.
    """
    v = venue.lower()
    if v == "kalshi":
        return kalshi_fee_prob(price, maker=maker)
    if v == "polymarket_us":
        return polymarket_us_fee_prob(price, maker=maker)
    if v == "novig":
        return novig_fee_prob(price, maker=maker)
    if v in _WINNINGS_COMMISSION_RATE:
        return prophetx_fee_prob(price, rate=_WINNINGS_COMMISSION_RATE[v])
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


def novig_order_fee(price: float, contracts: float, maker: bool = False) -> float:
    """Dollar fee (or negative rebate) Novig applies on one order.

    Same quadratic form as Polymarket US; banker's-rounded to the cent (Novig's
    exact rounding rule is unpublished — nearest-cent is the reasonable default
    and the unrounded ``novig_fee_prob`` is what EV math uses regardless).
    """
    rate = NOVIG_TAKER_RATE * (NOVIG_MAKER_RATE_MULT if maker else 1.0)
    raw = _quad(rate, price, contracts)
    sign = -1.0 if raw < 0 else 1.0
    return sign * _bankers_round_cents(abs(raw))


def venue_order_fee(
    venue: Optional[str], price: float, contracts: float, maker: bool = False
) -> float:
    """Dollar order fee CHARGED AT ENTRY by venue name, venue rounding applied.

    Dispatches to the per-venue entry-fee helper. Returns ``0.0`` for
    ``venue=None``, an unrecognised venue name, OR a winnings-commission venue
    (``prophetx``): those charge nothing at entry — their commission is taken at
    settlement on the winning profit and is applied in ``bet_pnl``, not here. A
    caller that opts out of fee accounting (``settings.fees_in_pricing`` off)
    passes ``None`` to price gross. This differs from ``venue_fee_prob``, which
    raises on an unknown venue — the P&L path must never crash a report on a
    stray venue string, so it degrades to gross instead.
    """
    if venue is None:
        return 0.0
    v = venue.lower()
    if v == "kalshi":
        return kalshi_order_fee(price, contracts, maker=maker)
    if v == "polymarket_us":
        return polymarket_us_order_fee(price, contracts, maker=maker)
    if v == "novig":
        return novig_order_fee(price, contracts, maker=maker)
    return 0.0


# ---------------------------------------------------------------------------
# Net-of-fee realized P&L for a settled binary bet
# ---------------------------------------------------------------------------

def bet_pnl(
    stake: float,
    price: float,
    won: bool,
    venue: Optional[str] = None,
    maker: bool = False,
) -> float:
    """Net-of-fee dollar P&L for one settled binary contract bet.

    ``stake`` is the contract NOTIONAL — the dollars spent on contracts,
    ``= contracts × price`` — NOT fee-inclusive. The payout ratio ``1/price``
    multiplies the notional only, so the fee must NOT be folded into ``stake``:
    doing so would scale the fee by ``1/price`` on a win and over-credit it. The
    venue's trading fee is instead a flat cost paid once at entry, collected by
    the venue at trade time regardless of outcome, so it reduces the win P&L and
    the loss P&L by the SAME amount::

        contracts = stake / price
        fee       = venue_order_fee(venue, price, contracts, maker)
        won  →  stake · (1/price − 1) − fee
        lost →  −stake − fee

    ``venue=None`` prices gross (``fee = 0``) — the behaviour before fee
    accounting, and what a caller passes when ``settings.fees_in_pricing`` is
    off. A degenerate ``price`` outside the open interval (0, 1) yields ``0.0``.

    A winnings-commission venue (``prophetx``) charges nothing at entry: the
    commission is taken at settlement on the winning profit only, so a win nets
    ``profit·(1 − rate)`` and a loss is the plain ``−stake`` (no fee). This is a
    different mechanism from the flat entry fee and is handled here explicitly.
    """
    if not (0.0 < price < 1.0):
        return 0.0
    v = (venue or "").lower()
    if v in _WINNINGS_COMMISSION_RATE:
        if won:
            profit = stake * (1.0 / price - 1.0)
            return profit * (1.0 - _WINNINGS_COMMISSION_RATE[v])
        return -stake
    contracts = stake / price
    fee = venue_order_fee(venue, price, contracts, maker=maker)
    if won:
        return stake * (1.0 / price - 1.0) - fee
    return -stake - fee
