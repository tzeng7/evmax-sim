"""EV Calculator.

EV = (true_prob × payout) - 1
payout = 1.0 / market_price   (for binary prediction markets)

Flag any market where EV >= threshold (default 2%).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from evmax.fees import venue_fee_prob
from evmax.models.market import PredictionMarket
from evmax.models.odds import SharpOdds


def effective_price(
    market_price: float,
    venue: Optional[str] = None,
    *,
    maker: bool = False,
) -> float:
    """Fee-inclusive per-contract entry cost on ``venue``.

    Returns ``market_price + per-contract trading fee`` (probability / dollar
    units, 0–1). Winning a binary contract still pays $1, but the venue's
    trading fee raises what you actually paid to acquire it. The EV gate and
    Kelly sizing must both price the contract at this cost, not the raw ask —
    otherwise a gross edge that the fee fully eats still reads as +EV and gets
    staked.

    ``venue=None`` (or an unknown venue) returns ``market_price`` unchanged, so
    every existing gross-of-fee caller keeps its behaviour until it opts in.

    The fee is symmetric in ``p`` / ``1-p`` (``fee = θ·p·(1-p)``), so passing a
    NO-side ask here is correct with no side-specific handling. The Polymarket
    US maker path is a *rebate* (negative fee) — makers are paid — so the result
    can dip below ``market_price`` there; the clamp keeps it inside the open
    interval so downstream ``payout = 1 / effective_price`` stays finite.
    """
    if venue is None or not (0.0 < market_price < 1.0):
        return market_price
    try:
        fee = venue_fee_prob(venue, market_price, maker=maker)
    except ValueError:
        # Unknown venue name — degrade to gross rather than raise on the hot path.
        return market_price
    return min(max(market_price + fee, 1e-4), 0.9999)


def tiered_min_ev(true_prob: float, *, min_ev: float, min_prob: float) -> float:
    """Scale the minimum EV up for low-probability bets.

    Formula: ``min_ev + max(0, min_prob - true_prob) * 0.5``. Examples
    (min_ev=2%, min_prob=15%):
      true_prob=0.08 → 2% + (0.15-0.08)*0.5 = 5.5%
      true_prob=0.12 → 2% + (0.15-0.12)*0.5 = 3.5%
      true_prob≥0.15 → 2% (floor, no scaling)

    Single source for the scan/verify/pick commands, which all gate on the
    same ramp. Takes the floors explicitly (the CLI copies closed over their
    command's ``min_ev``/``min_prob`` params).
    """
    return min_ev + max(0.0, min_prob - true_prob) * 0.5


@dataclass
class FeePricedEV:
    """EV of one binary contract priced under BOTH execution modes.

    A resting limit order that fills is charged the venue's *maker* fee (on
    Kalshi, 25% of the taker rate on maker-fee series, ~$0 elsewhere; on
    Polymarket US, a rebate). Crossing the spread is charged the *taker* fee.
    The scanner gates on the taker fee by default, so a gap that only clears
    net-of-fee as a maker is dropped and never surfaces. This struct exposes
    both so the caller can surface maker-only opportunities explicitly.

    Fields:
      taker_* — EV / edge / decimal payout at the taker effective price.
      maker_* — the same at the maker effective price (>= taker EV; the maker
                fee is never larger than the taker fee).
      passes  — True when EITHER mode clears ``ev_floor`` (the relaxed gate).
      maker_only — True when the taker EV is below the floor but the maker EV
                clears it. These are fill-contingent: they require a resting
                order and are NOT crossable at the current ask, so callers size
                them at zero live stake and log them as shadow.
    """

    taker_ev: float
    taker_edge: float
    taker_payout: float
    maker_ev: float
    maker_edge: float
    maker_payout: float
    passes: bool
    maker_only: bool


def dual_ev(
    market_price: float,
    true_prob: float,
    venue: Optional[str],
    ev_floor: float,
) -> FeePricedEV:
    """Price a contract net of BOTH the taker and maker fee, against ``ev_floor``.

    ``venue=None`` (fee accounting off) collapses both modes to the gross price,
    so ``maker_ev == taker_ev``, ``maker_only`` is always False, and ``passes``
    reduces to the pre-fee ``taker_ev >= ev_floor`` gate — byte-identical to the
    old single-mode path. This keeps the ``fees_in_pricing=False`` behaviour
    unchanged.
    """
    eff_taker = effective_price(market_price, venue, maker=False)
    eff_maker = effective_price(market_price, venue, maker=True)
    taker_ev, taker_edge = calculate_ev(eff_taker, true_prob)
    maker_ev, maker_edge = calculate_ev(eff_maker, true_prob)
    taker_payout = 1.0 / eff_taker if eff_taker > 0 else 0.0
    maker_payout = 1.0 / eff_maker if eff_maker > 0 else 0.0
    passes = taker_ev >= ev_floor or maker_ev >= ev_floor
    maker_only = taker_ev < ev_floor <= maker_ev
    return FeePricedEV(
        taker_ev=taker_ev,
        taker_edge=taker_edge,
        taker_payout=taker_payout,
        maker_ev=maker_ev,
        maker_edge=maker_edge,
        maker_payout=maker_payout,
        passes=passes,
        maker_only=maker_only,
    )


@dataclass
class EVResult:
    outcome: str  # "yes" or "no"
    market_implied_prob: float
    true_prob: float
    payout_decimal: float
    ev: float
    edge_pct: float
    is_positive_ev: bool


def calculate_ev(
    market_price: float,
    true_prob: float,
) -> tuple[float, float]:
    """
    Compute EV and edge percentage for a binary outcome.

    Args:
        market_price: Market price (0.0–1.0) — the cost to buy 1 unit of YES.
        true_prob: Devigged true probability of the outcome.

    Returns:
        (ev, edge_pct) where ev is raw EV and edge_pct is as a fraction.
    """
    if market_price <= 0 or market_price >= 1.0:
        return 0.0, 0.0
    if not (0.0 < true_prob <= 1.0):
        return 0.0, 0.0

    payout = 1.0 / market_price  # e.g. price=0.40 → payout=2.5x
    ev = (true_prob * payout) - 1.0
    edge_pct = ev  # same as EV for unit bet
    return ev, edge_pct


def evaluate_market(
    market: PredictionMarket,
    sharp_odds: SharpOdds,
    ev_threshold: float = 0.02,
) -> list[EVResult]:
    """
    Evaluate both YES and NO sides of a prediction market against sharp odds.

    The market's YES corresponds to outcome_a of the sharp event.
    The market's NO corresponds to outcome_b.

    Returns:
        List of EVResult for outcomes that are at or above threshold.
    """
    results: list[EVResult] = []

    # YES side only: each Kalshi game has a dedicated YES market per outcome
    # (home win, away win, draw). Evaluating NO sides would double-count positions
    # that are already covered by the opponent's YES market.
    yes_ev, yes_edge = calculate_ev(market.yes_price, sharp_odds.true_prob_a)
    yes_payout = 1.0 / market.yes_price if market.yes_price > 0 else 0.0
    yes_result = EVResult(
        outcome="yes",
        market_implied_prob=market.yes_price,
        true_prob=sharp_odds.true_prob_a,
        payout_decimal=yes_payout,
        ev=yes_ev,
        edge_pct=yes_edge,
        is_positive_ev=yes_ev >= ev_threshold,
    )
    results.append(yes_result)

    return [r for r in results if r.is_positive_ev]
