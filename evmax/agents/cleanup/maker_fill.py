"""Recording a filled maker limit order as a real placed position.

A maker-only play clears the EV floor ONLY as a resting limit order — it is not
crossable at the current ask, so :func:`evmax.agents.cleanup.logger.log_gaps`
demotes it to ``mode='shadow'`` and zeroes its taker Kelly. The row is therefore
fill-contingent: it becomes a real position only when the resting order actually
fills. This module owns that promotion::

    placed=1, placed_at, placed_price, placed_stake, mode='live', maker_fill=1

``maker_fill=1`` is what makes the net-of-fee P&L charge the MAKER fee instead
of the taker fee (see :func:`evmax.fees.venue_order_fee`).

Shared by ``evmax agents fill`` (CLI) and ``POST /api/fill`` (dashboard) so both
entry points validate, normalize and write identically. The helper never commits
and never closes — the caller owns the transaction, so a batch of dashboard
fills lands atomically.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from evmax.fees import venue_order_fee


class MakerFillError(ValueError):
    """A maker fill that cannot be recorded.

    ``reason`` is a short machine-stable slug the dashboard reports per row;
    ``str(err)`` is the human-readable message the CLI prints.
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class MakerFillResult:
    """What a successful :func:`record_maker_fill` wrote, for display."""

    market_id: str
    fill_prob: float          # normalized probability (0-1) the order filled at
    stake: float              # dollar notional = contracts x fill_prob
    contracts: float
    maker_fee: float          # dollar maker fee charged at this venue/price/size
    venue: str
    prior_mode: str           # row's mode BEFORE the promotion ('shadow' normally)
    event_title: str
    yes_team: str
    market_type: str
    line: Optional[float]
    sector: str


def normalize_fill_price(price: float) -> float:
    """Normalize a user-entered fill price to a probability in (0, 1).

    Accepts either cents (e.g. ``47``) or a fraction (e.g. ``0.47``). A value
    greater than 1.0 is read as cents; anything else is read as a fraction.
    Both unit conventions are accepted because the CLI takes cents by habit
    while the dashboard posts the fraction it already stores.

    Raises :class:`MakerFillError` when the normalized value is not strictly
    between 0 and 1. Note the boundary consequence of the shared unit rule: an
    input of exactly ``1`` reads as the fraction 1.0 and is rejected, and so is
    ``100``. Neither is a fillable binary-contract price.
    """
    try:
        numeric = float(price)
    except (TypeError, ValueError):
        raise MakerFillError(
            "invalid_price", f"Invalid price: {price!r} - give cents (0-100) or a fraction (0-1)."
        ) from None
    fill_prob = numeric / 100.0 if numeric > 1.0 else numeric
    if not (0.0 < fill_prob < 1.0):
        raise MakerFillError(
            "invalid_price", f"Invalid price: {price!r} - give cents (0-100) or a fraction (0-1)."
        )
    return fill_prob


def record_maker_fill(
    conn: sqlite3.Connection,
    market_id: str,
    price: float,
    stake: float,
    *,
    now: Optional[datetime] = None,
) -> MakerFillResult:
    """Promote one shadow maker row to a live placed maker bet.

    Validates the row is fillable, normalizes ``price`` through
    :func:`normalize_fill_price`, and writes the promotion columns. Does NOT
    commit — the caller commits so a multi-row fill is atomic.

    Raises :class:`MakerFillError` with one of these ``reason`` slugs:

    ``missing_market_id``
        No ``market_id`` was supplied.
    ``invalid_price`` / ``invalid_stake``
        The price is not in (0, 1) after normalization, or the stake is <= 0.
    ``not_found``
        No ``ev_predictions`` row for this market. Re-scan to log it.
    ``already_placed``
        The row is already a position. Un-place it first to re-record a fill,
        so an accidental double-submit can never overwrite a real fill price.
    ``voided``
        The row was voided (cancelled game, or auto-voided by the stale-position
        pruner). A voided row is not a live market to have filled against.

    A row whose mode is not ``shadow`` is still filled — ``prior_mode`` on the
    result reports what it was, so the caller can surface a note. This keeps a
    genuine maker fill recordable on a row that was left live (for instance a
    play whose taker EV also cleared the floor, which you chose to rest anyway).
    """
    if not market_id:
        raise MakerFillError("missing_market_id", "Missing market_id.")

    fill_prob = normalize_fill_price(price)

    try:
        stake_amt = float(stake)
    except (TypeError, ValueError):
        raise MakerFillError("invalid_stake", f"Invalid stake: {stake!r} - must be > 0.") from None
    if not stake_amt > 0:
        raise MakerFillError("invalid_stake", f"Invalid stake: {stake!r} - must be > 0.")

    row = conn.execute(
        """SELECT market_id, event_title, yes_team, market_type, line, sector,
                  venue, mode, placed, voided
           FROM ev_predictions WHERE market_id = ?""",
        (market_id,),
    ).fetchone()
    if row is None:
        raise MakerFillError(
            "not_found", f"No prediction row for market_id {market_id!r} (try rescanning)."
        )
    if row["placed"]:
        raise MakerFillError(
            "already_placed",
            f"{market_id} is already placed - un-place it first to re-record the fill.",
        )
    if row["voided"]:
        raise MakerFillError("voided", f"{market_id} is voided - not filling a voided row.")

    venue = row["venue"] or "kalshi"
    contracts = stake_amt / fill_prob
    maker_fee = venue_order_fee(venue, fill_prob, contracts, maker=True)

    now_str = (now or datetime.now(timezone.utc)).isoformat()
    conn.execute(
        """UPDATE ev_predictions
           SET placed = 1, placed_at = ?, placed_price = ?, placed_stake = ?,
               mode = 'live', maker_fill = 1
           WHERE market_id = ?""",
        (now_str, fill_prob, stake_amt, market_id),
    )

    return MakerFillResult(
        market_id=market_id,
        fill_prob=fill_prob,
        stake=stake_amt,
        contracts=contracts,
        maker_fee=maker_fee,
        venue=venue,
        prior_mode=row["mode"] or "",
        event_title=row["event_title"] or "",
        yes_team=row["yes_team"] or "",
        market_type=row["market_type"] or "",
        line=row["line"],
        sector=row["sector"] or "",
    )


def _as_dict(result: MakerFillResult) -> dict[str, Any]:
    """JSON-safe view of a recorded fill, for the dashboard response."""
    return {
        "market_id": result.market_id,
        "fill_price": round(result.fill_prob, 4),
        "fill_stake": round(result.stake, 2),
        "contracts": round(result.contracts, 2),
        "maker_fee": round(result.maker_fee, 2),
        "venue": result.venue,
        "prior_mode": result.prior_mode,
        "event_title": result.event_title,
    }
