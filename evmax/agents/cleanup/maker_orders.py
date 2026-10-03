"""Real maker-order attempts from Kalshi order history → fill rate + fee check.

``agents fill`` only records orders that FILLED, so there is no denominator for
a fill rate and no way to see an order that rested and was cancelled. This
module reads the account's own order history (``KalshiClient.get_orders``,
read-only), keeps one row per order in ``maker_order_attempts``, and
summarizes per Kalshi series:

  * fill rate by order and by contract (resting orders excluded — unresolved),
  * hours from creation to the last update on fully executed orders,
  * the fee Kalshi actually charged, as an IMPLIED MULTIPLIER versus the
    published formulas (taker ``0.07·C·P·(1−P)``, maker ``0.0175·C·P·(1−P)``).
    This is the empirical answer to "what fee does this series really charge":
    a multiplier near 1.0 confirms ``evmax.fees``; near 0.5 would mean the
    series API's ``fee_multiplier`` of 0.5 (KXMLBGAME) is what Kalshi bills.

Payload shape follows docs.kalshi.com ``GET /portfolio/orders`` (checked
2026-10-03): ``status`` resting|canceled|executed, prices and counts as
fixed-point strings (``yes_price_dollars``, ``fill_count_fp``, ...),
``outcome_side`` yes|no, ``taker_fees_dollars`` / ``maker_fees_dollars``,
``taker_fill_cost_dollars`` / ``maker_fill_cost_dollars``. It has NOT been run
against a live account payload — the parser skips (and counts) anything it
cannot read rather than guessing.
"""

from __future__ import annotations

import sqlite3
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

KALSHI_TAKER_RATE = 0.07
KALSHI_MAKER_RATE = 0.0175  # 0.07 × 0.25, the maker rate in the fee schedule

_TABLE = """
CREATE TABLE IF NOT EXISTS maker_order_attempts (
    order_id        TEXT PRIMARY KEY,
    ticker          TEXT NOT NULL,
    series          TEXT NOT NULL,
    outcome_side    TEXT NOT NULL,
    limit_price     REAL NOT NULL,
    initial_count   REAL NOT NULL,
    fill_count      REAL NOT NULL,
    status          TEXT NOT NULL,
    created_at      TEXT,
    last_update_at  TEXT,
    maker_fill_cost REAL NOT NULL DEFAULT 0,
    taker_fill_cost REAL NOT NULL DEFAULT 0,
    maker_fees      REAL NOT NULL DEFAULT 0,
    taker_fees      REAL NOT NULL DEFAULT 0,
    synced_at       TEXT NOT NULL
)
"""


@dataclass(frozen=True)
class OrderAttempt:
    order_id: str
    ticker: str
    series: str
    outcome_side: str          # "yes" | "no" — the side we are long
    limit_price: float         # dollars, on OUR side
    initial_count: float
    fill_count: float
    status: str                # resting | canceled | executed
    created_at: Optional[str]
    last_update_at: Optional[str]
    maker_fill_cost: float
    taker_fill_cost: float
    maker_fees: float
    taker_fees: float

    @property
    def filled(self) -> bool:
        return self.fill_count > 0

    @property
    def kind(self) -> str:
        """``maker`` / ``taker`` / ``mixed`` by where the fill cost landed, ``none`` if unfilled."""
        if self.maker_fill_cost > 0 and self.taker_fill_cost > 0:
            return "mixed"
        if self.maker_fill_cost > 0:
            return "maker"
        if self.taker_fill_cost > 0:
            return "taker"
        return "none"


def series_of(ticker: str) -> str:
    return (ticker or "").split("-", 1)[0].upper()


def _num(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first(d: dict, *keys: str) -> Optional[float]:
    for k in keys:
        v = _num(d.get(k))
        if v is not None:
            return v
    return None


def parse_order(d: dict) -> Optional[OrderAttempt]:
    """One raw order dict → ``OrderAttempt``, or None when it cannot be read."""
    try:
        order_id = str(d["order_id"])
        ticker = str(d["ticker"])
        status = str(d["status"]).lower()
    except KeyError:
        return None
    side = str(d.get("outcome_side") or d.get("side") or "").lower()
    if side not in ("yes", "no"):
        return None
    price_key = "yes_price_dollars" if side == "yes" else "no_price_dollars"
    price = _num(d.get(price_key))
    if price is None:  # legacy integer-cent fields
        cents = _num(d.get("yes_price" if side == "yes" else "no_price"))
        price = cents / 100.0 if cents is not None else None
    initial = _first(d, "initial_count_fp", "initial_count")
    fill = _first(d, "fill_count_fp", "fill_count")
    if price is None or initial is None or fill is None or not (0.0 < price < 1.0):
        return None
    return OrderAttempt(
        order_id=order_id,
        ticker=ticker,
        series=series_of(ticker),
        outcome_side=side,
        limit_price=price,
        initial_count=initial,
        fill_count=fill,
        status=status,
        created_at=d.get("created_time"),
        last_update_at=d.get("last_update_time"),
        maker_fill_cost=_num(d.get("maker_fill_cost_dollars")) or 0.0,
        taker_fill_cost=_num(d.get("taker_fill_cost_dollars")) or 0.0,
        maker_fees=_num(d.get("maker_fees_dollars")) or 0.0,
        taker_fees=_num(d.get("taker_fees_dollars")) or 0.0,
    )


def parse_orders(raw: Iterable[dict]) -> tuple[list[OrderAttempt], int]:
    """Parse a batch; returns ``(attempts, n_skipped)``."""
    out: list[OrderAttempt] = []
    skipped = 0
    for d in raw:
        a = parse_order(d)
        if a is None:
            skipped += 1
        else:
            out.append(a)
    return out, skipped


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(_TABLE)


def upsert_attempts(
    conn: sqlite3.Connection,
    attempts: Iterable[OrderAttempt],
    now: Optional[datetime] = None,
) -> int:
    """Insert or refresh by ``order_id`` (an order's fills/status change over time)."""
    ensure_table(conn)
    stamp = (now or datetime.now(timezone.utc)).isoformat()
    n = 0
    for a in attempts:
        conn.execute(
            """INSERT INTO maker_order_attempts
               (order_id, ticker, series, outcome_side, limit_price, initial_count,
                fill_count, status, created_at, last_update_at, maker_fill_cost,
                taker_fill_cost, maker_fees, taker_fees, synced_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(order_id) DO UPDATE SET
                 fill_count=excluded.fill_count, status=excluded.status,
                 last_update_at=excluded.last_update_at,
                 maker_fill_cost=excluded.maker_fill_cost,
                 taker_fill_cost=excluded.taker_fill_cost,
                 maker_fees=excluded.maker_fees, taker_fees=excluded.taker_fees,
                 synced_at=excluded.synced_at""",
            (a.order_id, a.ticker, a.series, a.outcome_side, a.limit_price,
             a.initial_count, a.fill_count, a.status, a.created_at,
             a.last_update_at, a.maker_fill_cost, a.taker_fill_cost,
             a.maker_fees, a.taker_fees, stamp),
        )
        n += 1
    conn.commit()
    return n


def load_attempts(conn: sqlite3.Connection) -> list[OrderAttempt]:
    ensure_table(conn)
    cur = conn.execute(
        """SELECT order_id, ticker, series, outcome_side, limit_price, initial_count,
                  fill_count, status, created_at, last_update_at, maker_fill_cost,
                  taker_fill_cost, maker_fees, taker_fees FROM maker_order_attempts"""
    )
    return [OrderAttempt(*row) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------

def _parse_ts(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def hours_to_fill(a: OrderAttempt) -> Optional[float]:
    """Creation → last update on a fully executed order (None otherwise)."""
    if a.status != "executed":
        return None
    t0, t1 = _parse_ts(a.created_at), _parse_ts(a.last_update_at)
    if t0 is None or t1 is None or t1 < t0:
        return None
    return (t1 - t0).total_seconds() / 3600.0


def fill_summary(attempts: Iterable[OrderAttempt]) -> dict[str, dict]:
    """Per-series fill behaviour of maker (resting) orders.

    Orders still ``resting`` are excluded from the denominators: their outcome
    is not known yet. Orders that crossed as takers are counted separately and
    are not part of the maker fill rate.
    """
    by: dict[str, list[OrderAttempt]] = {}
    for a in attempts:
        by.setdefault(a.series, []).append(a)
    out: dict[str, dict] = {}
    for series, xs in sorted(by.items()):
        done = [a for a in xs if a.status != "resting"]
        maker_side = [a for a in done if a.kind != "taker"]
        n = len(maker_side)
        filled = [a for a in maker_side if a.filled]
        contracts = sum(a.initial_count for a in maker_side)
        hrs = [h for a in filled if (h := hours_to_fill(a)) is not None]
        out[series] = {
            "orders": len(xs),
            "resting": len(xs) - len(done),
            "crossed_as_taker": sum(1 for a in done if a.kind == "taker"),
            "resolved_maker_orders": n,
            "filled_orders": len(filled),
            "fill_rate_orders": (len(filled) / n) if n else None,
            "fill_rate_contracts": (
                sum(a.fill_count for a in maker_side) / contracts if contracts else None
            ),
            "median_hours_to_fill": statistics.median(hrs) if hrs else None,
        }
    return out


def implied_fee_multipliers(attempts: Iterable[OrderAttempt]) -> dict[str, dict]:
    """Observed fee ÷ published-formula fee, per series and fee kind.

    Uses only orders whose fills were ALL maker or ALL taker, so the average fill
    price is ``fill_cost / fill_count``. ``implied_multiplier`` of ~1.0 means the
    series bills the published rate; ~0.5 means half of it. Per-order rounding
    (fees are rounded up) inflates the ratio on small orders, so the contract
    total is reported — do not trust a ratio from a handful of contracts.
    """
    acc: dict[tuple[str, str], dict[str, float]] = {}
    for a in attempts:
        if a.fill_count <= 0 or a.kind not in ("maker", "taker"):
            continue
        cost = a.maker_fill_cost if a.kind == "maker" else a.taker_fill_cost
        fee = a.maker_fees if a.kind == "maker" else a.taker_fees
        px = cost / a.fill_count
        if not (0.0 < px < 1.0):
            continue
        rate = KALSHI_MAKER_RATE if a.kind == "maker" else KALSHI_TAKER_RATE
        expected = rate * a.fill_count * px * (1.0 - px)
        d = acc.setdefault((a.series, a.kind), {"observed": 0.0, "expected": 0.0, "contracts": 0.0})
        d["observed"] += fee
        d["expected"] += expected
        d["contracts"] += a.fill_count
    out: dict[str, dict] = {}
    for (series, kind), d in sorted(acc.items()):
        out[f"{series}:{kind}"] = {
            "contracts": d["contracts"],
            "observed_fee": d["observed"],
            "expected_fee": d["expected"],
            "implied_multiplier": d["observed"] / d["expected"] if d["expected"] > 0 else None,
        }
    return out
