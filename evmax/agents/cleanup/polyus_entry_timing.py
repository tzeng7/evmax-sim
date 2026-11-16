"""Entry-timing analysis for Polymarket US candidates, on price history.

Built on the replay in ``polyus_maker_replay`` (same data, same side
orientation, same no-look-ahead quote rule). It answers three questions about
WHEN and HOW to place a flagged candidate:

  1. Edge decay. After a candidate is flagged at ``t0``, how fast does our
     side's ask move toward its close? ``drift(d) = ask(t0 + d) − ask(t0)`` is
     the price paid for reacting ``d`` minutes late.
  2. Sampling interval. How many resting-bid fills does a point sample every
     ``k`` minutes miss, compared with the lowest ask inside every bucket, and
     how does that bias the value of the fills it does see?
  3. Placement policy. Value per candidate, net of fees, of: crossing now;
     waiting then crossing; resting a bid until tip; resting for ``H`` minutes
     then cancelling; resting for ``H`` minutes then crossing if unfilled.

Value is closing-line value: ``close ask − price paid − fee``, in percentage
points of contract price. An order that never trades is worth zero. A
candidate too close to tip for a policy's horizon falls back to the policy's
terminal action at the last usable quote, so every policy is scored on the
SAME candidates and the numbers are comparable.

All functions are pure. Fetching and caching live in
scripts/eval_polyus_maker_replay.py; the CLI is
scripts/eval_polyus_entry_timing.py.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from evmax.agents.cleanup.polyus_maker_replay import (
    BUCKET_S,
    ENTRY_MAX_STALENESS_S,
    EPS,
    MIN_LEAD_S,
    Bucket,
    MarketMeta,
    _mean_ci,
    _utc,
    our_side_is_long,
    side_quote,
)

# Latest moment a taker order is still allowed, in seconds before tip.
LAST_ENTRY_BEFORE_TIP_S = 300


@dataclass(frozen=True)
class Path:
    """One candidate's own-side quote path from the signal to tip-off."""

    market_id: str
    sector: Optional[str]
    market_type: Optional[str]
    game: str
    t0: float                 # signal time (unix)
    tip: float
    tick: float
    # Completed buckets from the entry bucket on, oldest first:
    # (bucket_end_ts, ask_last, bid_last, ask_min)
    quotes: tuple[tuple[float, float, float, float], ...]
    close_ask: float
    close_mid: float

    @property
    def lead_h(self) -> float:
        return (self.tip - self.t0) / 3600.0


def build_path(row: dict, meta: MarketMeta, buckets: list[Bucket]) -> tuple[Optional[Path], str]:
    """Own-side quote path for one logged row. (None, reason) when unusable.

    ``quotes[0]`` is the entry quote: the last bucket that had fully ended by
    ``logged_at``. Every later element is a bucket that started at or after
    ``logged_at`` and ended at or before tip-off.
    """
    from datetime import timedelta

    long = our_side_is_long(row["market_id"], meta)
    if long is None:
        return None, "side_unknown"
    t0 = _utc(row.get("logged_at"))
    if t0 is None:
        return None, "no_logged_at"
    tip = meta.tip
    if tip is None and row.get("minutes_to_tipoff"):
        tip = t0 + timedelta(minutes=float(row["minutes_to_tipoff"]))
    if tip is None:
        return None, "no_tip"
    t0s, tips = t0.timestamp(), tip.timestamp()
    if tips - t0s < MIN_LEAD_S:
        return None, "logged_at_or_after_tip"

    entry = None
    for b in buckets:
        if b[0] + BUCKET_S <= t0s:
            entry = b
        else:
            break
    if entry is None or t0s - (entry[0] + BUCKET_S) > ENTRY_MAX_STALENESS_S:
        return None, "no_entry_quote"
    ask, bid, low = side_quote(entry, long)
    if not (0.0 < bid < ask < 1.0):
        return None, "bad_entry_book"
    quotes = [(float(entry[0] + BUCKET_S), ask, bid, low)]
    for b in buckets:
        if b[0] >= t0s and b[0] + BUCKET_S <= tips:
            a, bd, lo = side_quote(b, long)
            quotes.append((float(b[0] + BUCKET_S), a, bd, lo))
    if len(quotes) < 2:
        return None, "no_pre_tip_path"
    close_ask, close_bid = quotes[-1][1], quotes[-1][2]
    if not (0.0 < close_bid <= close_ask < 1.0):
        return None, "bad_close_book"
    return Path(
        market_id=row["market_id"], sector=row.get("sector"),
        market_type=row.get("market_type"), game=row.get("game") or row["market_id"],
        t0=t0s, tip=tips, tick=meta.tick, quotes=tuple(quotes),
        close_ask=close_ask, close_mid=(close_ask + close_bid) / 2.0,
    ), "ok"


def limit_price(p: Path) -> float:
    """Resting limit at the signal: bid + one tick, never at or above the ask."""
    _, ask, bid, _ = p.quotes[0]
    return round(max(bid, min(bid + p.tick, ask - p.tick)), 6)


def quote_at(p: Path, t: float) -> Optional[tuple[float, float]]:
    """(ask, bid) standing at time ``t``: the last bucket that had ended by ``t``.

    None when ``t`` is before the entry quote, or the standing quote is older
    than the staleness limit (a gap in the history is not a price).
    """
    found = None
    for end, ask, bid, _ in p.quotes:
        if end <= t + EPS:
            found = (end, ask, bid)
        else:
            break
    if found is None or t - found[0] > ENTRY_MAX_STALENESS_S:
        return None
    return found[1], found[2]


def first_fill(p: Path, limit: float, *, thru: bool = False,
               until: Optional[float] = None, sample_s: Optional[int] = None) -> Optional[float]:
    """Time the resting bid first fills, or None.

    Default test: the LOWEST ask inside a bucket reached the limit (``thru``:
    went strictly below it). ``until`` stops the order resting at that time.
    ``sample_s`` switches to a POINT sample: only the quote standing at every
    ``sample_s`` seconds after the signal is seen, the way a periodic snapshot
    sees the market. It exists to measure what such sampling misses.
    """
    next_sample = p.t0 + sample_s if sample_s else None
    for end, ask, _, low in p.quotes[1:]:
        if until is not None and end > until + EPS:
            break
        if sample_s:
            if end + EPS < next_sample:
                continue
            while next_sample <= end + EPS:
                next_sample += sample_s
            seen = ask
        else:
            seen = low
        if (seen < limit - EPS) if thru else (seen <= limit + EPS):
            return end
    return None


def _fees(price: float, maker: bool) -> float:
    from evmax.fees import polymarket_us_fee_prob

    return polymarket_us_fee_prob(price, maker=maker)


def _last_entry(p: Path) -> float:
    return p.tip - LAST_ENTRY_BEFORE_TIP_S


def taker_value(p: Path, delay_s: float = 0.0) -> float:
    """Cross the ask ``delay_s`` after the signal (capped at the last entry time)."""
    q = quote_at(p, min(p.t0 + delay_s, _last_entry(p))) or p.quotes[0][1:3]
    ask = q[0]
    return (p.close_ask - ask - _fees(ask, maker=False)) * 100.0


def maker_value(p: Path, *, rest_s: Optional[float] = None, then_cross: bool = False,
                thru: bool = False, mid_basis: bool = False) -> float:
    """Rest a bid at the signal's limit.

    ``rest_s=None`` rests until tip. Otherwise the order is cancelled
    ``rest_s`` after the signal; with ``then_cross`` an unfilled order crosses
    the ask standing at that moment (never later than the last entry time).
    """
    limit = limit_price(p)
    close = p.close_mid if mid_basis else p.close_ask
    until = None if rest_s is None else min(p.t0 + rest_s, p.tip)
    if first_fill(p, limit, thru=thru, until=until) is not None:
        return (close - limit - _fees(limit, maker=True)) * 100.0
    if not then_cross or rest_s is None:
        return 0.0
    q = quote_at(p, min(p.t0 + rest_s, _last_entry(p))) or p.quotes[0][1:3]
    ask = q[0]
    return (close - ask - _fees(ask, maker=False)) * 100.0


def drift(p: Path, delay_s: float) -> Optional[float]:
    """ask(t0 + delay) − ask(t0), in pp. None when that moment is not pre-tip."""
    t = p.t0 + delay_s
    if t > _last_entry(p):
        return None
    q = quote_at(p, t)
    if q is None:
        return None
    return (q[0] - p.quotes[0][1]) * 100.0


# ---------------------------------------------------------------------------
# Aggregates (game-clustered via polyus_maker_replay._mean_ci)
# ---------------------------------------------------------------------------


def _ci(paths: list[Path], values: list[float]) -> dict:
    rows = [{"game": p.game, "v": v} for p, v in zip(paths, values)]
    return _mean_ci(rows, lambda r: r["v"])


def decay_curve(paths: list[Path], delays_min: tuple[int, ...]) -> list[dict]:
    """Per delay: how much of the move to the close has already happened.

    Each delay uses only candidates still pre-tip at ``t0 + delay``, and
    reports that SAME cohort's total move (close − ask at the signal), so
    ``share_consumed`` compares like with like.
    """
    out = []
    for d in delays_min:
        ps, drifts, totals = [], [], []
        for p in paths:
            x = drift(p, d * 60.0)
            if x is None:
                continue
            ps.append(p)
            drifts.append(x)
            totals.append((p.close_ask - p.quotes[0][1]) * 100.0)
        if not ps:
            continue
        dr, tot = _ci(ps, drifts), _ci(ps, totals)
        out.append({
            "delay_min": d, "n": dr["n"], "games": dr["games"],
            "drift_pp": dr, "total_move_pp": tot,
            "share_consumed": (dr["mean"] / tot["mean"]) if tot["mean"] else None,
        })
    return out


def sampling_bias(paths: list[Path], sample_minutes: tuple[int, ...]) -> list[dict]:
    """Fill rate and filled-order value seen by a point sample every k minutes.

    Row ``k=None`` is the reference: the lowest ask inside every bucket.
    """
    out = []
    for k in (None, *sample_minutes):
        filled_p, filled_v, per_cand = [], [], []
        for p in paths:
            limit = limit_price(p)
            hit = first_fill(p, limit, sample_s=(k * 60 if k else None))
            v = (p.close_ask - limit - _fees(limit, maker=True)) * 100.0
            per_cand.append(v if hit is not None else 0.0)
            if hit is not None:
                filled_p.append(p)
                filled_v.append(v)
        out.append({
            "sample_min": k, "n": len(paths),
            "fill_rate": len(filled_p) / len(paths) if paths else None,
            "value_if_filled_pp": _ci(filled_p, filled_v),
            "value_per_candidate_pp": _ci(paths, per_cand),
        })
    return out


def time_to_fill(paths: list[Path], marks_min: tuple[int, ...] = (15, 60, 180, 360)) -> dict:
    """Among orders that fill before tip: how long after the signal."""
    waits = []
    for p in paths:
        t = first_fill(p, limit_price(p))
        if t is not None:
            waits.append((t - p.t0) / 60.0)
    if not waits:
        return {"n": 0}
    waits.sort()
    return {
        "n": len(waits), "median_min": waits[len(waits) // 2],
        "share_within": {m: sum(1 for w in waits if w <= m) / len(waits) for m in marks_min},
    }


def policy_table(paths: list[Path], rest_minutes: tuple[int, ...] = (15, 30, 60, 180, 360),
                 wait_minutes: tuple[int, ...] = (15, 60, 180)) -> list[dict]:
    """Value per candidate (pp, net of fees) for each placement policy."""
    rows: list[tuple[str, list[float]]] = [
        ("cross now", [taker_value(p) for p in paths]),
    ]
    for w in wait_minutes:
        rows.append((f"wait {w}m, then cross", [taker_value(p, w * 60.0) for p in paths]))
    rows.append(("rest to tip (touch)", [maker_value(p) for p in paths]))
    rows.append(("rest to tip (thru)", [maker_value(p, thru=True) for p in paths]))
    for h in rest_minutes:
        rows.append((f"rest {h}m, then cancel",
                     [maker_value(p, rest_s=h * 60.0) for p in paths]))
    for h in rest_minutes:
        rows.append((f"rest {h}m, then cross",
                     [maker_value(p, rest_s=h * 60.0, then_cross=True) for p in paths]))
    for h in rest_minutes:
        rows.append((f"rest {h}m, then cross (thru fills)",
                     [maker_value(p, rest_s=h * 60.0, then_cross=True, thru=True)
                      for p in paths]))
    return [{"policy": name, **_ci(paths, vals)} for name, vals in rows]


def by_lead(paths: list[Path],
            edges_h: tuple[float, ...] = (0, 1, 3, 6, 12, 24, 72, 1e9)) -> list[dict]:
    """Taker and maker value per candidate, split by hours-to-tip at the signal."""
    buckets: dict[tuple[float, float], list[Path]] = defaultdict(list)
    for p in paths:
        for lo, hi in zip(edges_h, edges_h[1:]):
            if lo <= p.lead_h < hi:
                buckets[(lo, hi)].append(p)
                break
    out = []
    for (lo, hi), ps in sorted(buckets.items()):
        fills = sum(1 for p in ps if first_fill(p, limit_price(p)) is not None)
        out.append({
            "lead_h": (lo, hi), "n": len(ps), "games": len({p.game for p in ps}),
            "fill_rate": fills / len(ps),
            "taker_gross_pp": _ci(ps, [(p.close_ask - p.quotes[0][1]) * 100.0 for p in ps]),
            "taker_net_pp": _ci(ps, [taker_value(p) for p in ps]),
            "maker_net_pp": _ci(ps, [maker_value(p) for p in ps]),
        })
    return out
