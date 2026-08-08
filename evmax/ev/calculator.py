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
