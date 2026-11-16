"""Maker-fill replay for Polymarket US, on the venue's own price history.

Question: if a logged PolyUS candidate had been entered with a RESTING bid
instead of crossing the ask, would the order have filled before tip-off, and
what was it worth once filled?

Data. ``GET gateway.polymarket.us/v1/price-history`` is public and keeps
roughly one point per minute for settled markets (verified back to
2026-07-08). Each point carries the LONG instrument's two display prices:

    longPrice  = best ask of the long side
    shortPrice = 1 − best bid of the long side  (= best ask of the short side)

So one point gives the ask AND the bid of either side:

    long side :  ask = longPrice ,  bid = 1 − shortPrice
    short side:  ask = shortPrice,  bid = 1 − longPrice

History has no sizes, so the replay cannot see queue position or depth.

Method (mirrors scripts/eval_maker_fill_replay.py, the Kalshi replay):

  1. Entry quote = the last completed bucket at or before the row's
     ``logged_at`` (never a later one: no look-ahead).
  2. Resting limit = best bid + one tick, capped at ask − one tick, never
     below the bid.
  3. Fill test over every bucket from entry to tip-off, on the bucket's
     MINIMUM ask of our side:
       touch = the ask reached the limit        (optimistic: ignores queue)
       thru  = the ask went strictly below it   (conservative: we were hit)
  4. Close = the last fully pre-tip bucket. Maker CLV = close − limit, on two
     bases: the close ASK (comparable to the Kalshi replay and the CLV gate)
     and the close MID (stricter: an ask-based close credits the maker with
     half the closing spread).
  5. Fees: ``evmax.fees.polymarket_us_fee_prob`` (taker at the ask, maker at
     the limit).

Tip-off is the market's own ``gameStartTime``; in-play buckets never enter.
All statistics are per GAME (bootstrap over games): rungs of one game share
its news and are not independent.

The functions here are pure. Fetching and caching live in
scripts/eval_polyus_maker_replay.py.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

BUCKET_S = 300                 # 5-minute buckets
ENTRY_MAX_STALENESS_S = 1800   # entry quote must be within 30 min of logged_at
MIN_LEAD_S = 600               # a row logged < 10 min before tip is not replayed
DEFAULT_TICK = 0.01
EPS = 1e-9

# A bucket is [ts, long_last, short_last, long_min, short_min].
Bucket = list


def bucketize(history: list[dict], bucket_s: int = BUCKET_S) -> list[Bucket]:
    """Collapse raw history points into fixed buckets.

    Per bucket: the LAST long/short price (the quote standing at its end) and
    the MINIMUM long/short price inside it. The minimum of a side's ask is
    what a resting bid on that side is tested against, so bucketing loses no
    fills. Points missing either price, or outside [0, 1], are dropped.
    """
    out: dict[int, Bucket] = {}
    for p in sorted(history, key=lambda h: h.get("timestamp", 0)):
        ts, lp, sp = p.get("timestamp"), p.get("longPrice"), p.get("shortPrice")
        if ts is None or lp is None or sp is None:
            continue
        lp, sp = float(lp), float(sp)
        if not (0.0 <= lp <= 1.0 and 0.0 <= sp <= 1.0):
            continue
        key = int(ts) - int(ts) % bucket_s
        b = out.get(key)
        if b is None:
            out[key] = [key, lp, sp, lp, sp]
        else:
            b[1], b[2] = lp, sp
            b[3], b[4] = min(b[3], lp), min(b[4], sp)
    return [out[k] for k in sorted(out)]


def day_windows(start: datetime, end: datetime) -> list[str]:
    """UTC calendar days (YYYY-MM-DD) covering [start, end].

    Day-aligned windows let every row and both sides of a slug share one
    cached fetch per day (the endpoint allows custom ranges up to 24 h).
    """
    if end < start:
        return []
    d, last = start.astimezone(timezone.utc).date(), end.astimezone(timezone.utc).date()
    days = []
    while d <= last:
        days.append(d.isoformat())
        d += timedelta(days=1)
    return days


@dataclass(frozen=True)
class MarketMeta:
    slug: str
    tip: Optional[datetime]
    tick: float
    sides: tuple[tuple[Optional[str], str, bool], ...]  # (team abbr, description, long)


def parse_market_meta(slug: str, market: dict) -> MarketMeta:
    """Read tip-off, tick size and side orientation from ``/v1/market/slug``."""
    tip = None
    raw = market.get("gameStartTime")
    if raw:
        try:
            tip = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            if tip.tzinfo is None:
                tip = tip.replace(tzinfo=timezone.utc)
        except ValueError:
            tip = None
    try:
        tick = float(market.get("orderPriceMinTickSize") or DEFAULT_TICK)
    except (TypeError, ValueError):
        tick = DEFAULT_TICK
    if not (0.0 < tick <= 0.05):
        tick = DEFAULT_TICK
    sides = tuple(
        ((s.get("team") or {}).get("abbreviation"),
         str(s.get("description") or ""), bool(s.get("long")))
        for s in market.get("marketSides") or []
    )
    return MarketMeta(slug, tip, tick, sides)


def parse_market_id(market_id: str) -> Optional[tuple[str, Optional[str], bool]]:
    """``polymarket_us:{slug}[:{side}][:no]`` → (slug, side_key, is_no)."""
    if not market_id.startswith("polymarket_us:"):
        return None
    body = market_id.removeprefix("polymarket_us:")
    is_no = body.endswith(":no")
    if is_no:
        body = body[: -len(":no")]
    slug, _, side = body.partition(":")
    if not slug:
        return None
    return slug, (side or None), is_no


def our_side_is_long(market_id: str, meta: MarketMeta) -> Optional[bool]:
    """Is the row's OWN side the market's long instrument? None = unknown.

    A per-team id names its side by team abbreviation. An id with no side
    (spread, total, drawable outcome) has YES = the long side, except a total
    whose "Over" side is not long (read from the metadata, never assumed).
    A ``:no`` suffix is the opposite side of that YES contract.
    """
    parsed = parse_market_id(market_id)
    if parsed is None:
        return None
    _, side_key, is_no = parsed
    yes_long: Optional[bool]
    if side_key is None:
        over = [lg for _, desc, lg in meta.sides if desc.strip().lower() == "over"]
        yes_long = over[0] if over else True
    else:
        match = [lg for abbr, _, lg in meta.sides if abbr is not None and abbr == side_key]
        if len(match) == 1:
            yes_long = match[0]
        elif side_key in ("a", "b") and not match:
            yes_long = side_key == "a"   # the parser's fallback key
        else:
            yes_long = None
    if yes_long is None:
        return None
    return (not yes_long) if is_no else yes_long


def side_quote(bucket: Bucket, long: bool) -> tuple[float, float, float]:
    """(ask_last, bid_last, ask_min) of one side from a bucket."""
    _, long_last, short_last, long_min, short_min = bucket
    if long:
        return long_last, 1.0 - short_last, long_min
    return short_last, 1.0 - long_last, short_min


def _utc(raw) -> Optional[datetime]:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    try:
        d = datetime.fromisoformat(str(raw).replace("T", " ").replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def replay_row(row: dict, meta: MarketMeta, buckets: list[Bucket]) -> tuple[Optional[dict], str]:
    """Replay one logged candidate. Returns (result, reason); result None = skipped.

    ``row`` needs market_id, logged_at, minutes_to_tipoff; optional
    entry_price (the ask logged at scan time, used only as an orientation
    check), blended_true_prob, outcome, sector, market_type, game.
    """
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

    # Entry: last bucket that had fully ended by logged_at.
    entry = None
    for b in buckets:
        if b[0] + BUCKET_S <= t0s:
            entry = b
        else:
            break
    if entry is None or t0s - (entry[0] + BUCKET_S) > ENTRY_MAX_STALENESS_S:
        return None, "no_entry_quote"
    ask, bid, _ = side_quote(entry, long)
    if not (0.0 < bid < ask < 1.0):
        return None, "bad_entry_book"

    limit = min(bid + meta.tick, ask - meta.tick)
    if limit < bid:
        limit = bid
    limit = round(limit, 6)

    future = [b for b in buckets if b[0] >= t0s and b[0] + BUCKET_S <= tips]
    if not future:
        return None, "no_pre_tip_path"
    lows = [side_quote(b, long)[2] for b in future]
    close_ask, close_bid, _ = side_quote(future[-1], long)
    if not (0.0 < close_bid <= close_ask < 1.0):
        return None, "bad_close_book"

    from evmax.fees import polymarket_us_fee_prob

    logged = row.get("entry_price")
    return {
        "market_id": row["market_id"],
        "sector": row.get("sector"),
        "market_type": row.get("market_type"),
        "game": row.get("game") or row["market_id"],
        "long": long,
        "ask": ask,
        "bid": bid,
        "spread": ask - bid,
        "limit": limit,
        "touch": any(x <= limit + EPS for x in lows),
        "thru": any(x < limit - EPS for x in lows),
        "close_ask": close_ask,
        "close_mid": (close_ask + close_bid) / 2.0,
        "taker_fee": polymarket_us_fee_prob(ask),
        "maker_fee": polymarket_us_fee_prob(limit, maker=True),
        "lead_h": (tips - t0s) / 3600.0,
        "logged_ask_diff": (ask - float(logged)) if logged is not None else None,
        "blend": row.get("blended_true_prob"),
        "outcome": row.get("outcome"),
    }, "ok"


# ---------------------------------------------------------------------------
# Statistics (per game)
# ---------------------------------------------------------------------------


def _game_bootstrap(per_game: dict[str, list[float]], n_boot: int = 2000,
                    seed: int = 7) -> Optional[tuple[float, float]]:
    """95% CI of the mean over ROWS, resampling GAMES."""
    games = [v for v in per_game.values() if v]
    if len(games) < 5:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        tot = cnt = 0.0
        for _ in games:
            g = games[rng.randrange(len(games))]
            tot += sum(g)
            cnt += len(g)
        means.append(tot / cnt if cnt else 0.0)
    means.sort()
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def _mean_ci(rows: list[dict], value) -> dict:
    per_game: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        per_game[r["game"]].append(value(r))
    vals = [v for g in per_game.values() for v in g]
    if not vals:
        return {"n": 0, "games": 0, "mean": None, "ci95": None}
    return {
        "n": len(vals),
        "games": len(per_game),
        "mean": sum(vals) / len(vals),
        "ci95": _game_bootstrap(per_game),
    }


def summarize(results: list[dict], fill: str = "touch") -> Optional[dict]:
    """Aggregate one group of replay results. ``fill`` = 'touch' or 'thru'.

    All CLV / EV figures are in percentage points of contract price.
    ``maker_net_per_candidate`` is the figure to compare with
    ``taker_net_per_candidate``: an unfilled resting order earns zero, so the
    maker value of a CANDIDATE is (net edge when filled) × (fill rate). It is
    computed as the mean over ALL candidates of (filled ? edge : 0), which
    keeps the fill-rate and the edge in one game-clustered statistic.
    """
    if fill not in ("touch", "thru"):
        raise ValueError("fill must be 'touch' or 'thru'")
    n = len(results)
    if n == 0:
        return None
    pp = 100.0
    filled = [r for r in results if r[fill]]
    unfilled = [r for r in results if not r[fill]]
    maker_ask = lambda r: (r["close_ask"] - r["limit"] - r["maker_fee"]) * pp   # noqa: E731
    maker_mid = lambda r: (r["close_mid"] - r["limit"] - r["maker_fee"]) * pp   # noqa: E731
    taker = lambda r: (r["close_ask"] - r["ask"] - r["taker_fee"]) * pp         # noqa: E731
    diffs = sorted(abs(r["logged_ask_diff"]) for r in results
                   if r["logged_ask_diff"] is not None)
    wins = [r for r in filled if r["outcome"] in (0, 1)]
    return {
        "n": n,
        "games": len({r["game"] for r in results}),
        "fill_rate": len(filled) / n,
        "mean_spread_pp": sum(r["spread"] for r in results) / n * pp,
        "median_lead_h": sorted(r["lead_h"] for r in results)[n // 2],
        "median_abs_logged_ask_diff_pp": (diffs[len(diffs) // 2] * pp) if diffs else None,
        "taker_clv_gross": _mean_ci(results, lambda r: (r["close_ask"] - r["ask"]) * pp),
        "taker_net_per_candidate": _mean_ci(results, taker),
        "maker_clv_filled_ask": _mean_ci(filled, lambda r: (r["close_ask"] - r["limit"]) * pp),
        "maker_clv_filled_mid": _mean_ci(filled, lambda r: (r["close_mid"] - r["limit"]) * pp),
        "unfilled_taker_clv_gross": _mean_ci(
            unfilled, lambda r: (r["close_ask"] - r["ask"]) * pp),
        "maker_net_per_candidate": _mean_ci(
            results, lambda r: maker_ask(r) if r[fill] else 0.0),
        "maker_net_per_candidate_mid": _mean_ci(
            results, lambda r: maker_mid(r) if r[fill] else 0.0),
        "filled_realized": {
            "n": len(wins),
            "win_rate": (sum(r["outcome"] for r in wins) / len(wins)) if wins else None,
            "mean_limit": (sum(r["limit"] for r in wins) / len(wins)) if wins else None,
            "roi_net_pct": (
                sum(r["outcome"] - r["limit"] - r["maker_fee"] for r in wins)
                / sum(r["limit"] + r["maker_fee"] for r in wins) * 100.0
            ) if wins else None,
        },
    }

