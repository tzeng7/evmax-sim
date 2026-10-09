#!/usr/bin/env python3
"""Offline EV lens for Kalshi's NFL scalar prop products (Ladders + Escalators).

READ-ONLY research tool. Nothing here is wired into scans; see
docs/nfl-ladders-escalators-eval.md for the verdict this script produced and
the gate for revisiting it.

Products (Kalshi pays YES a fraction of $1 that depends on the stat line):

    KXNFLLADDER{RECYDS,RSHYDS}   0.25c per yard, cap 400 yd  (linear)
    KXNFLLADDERREC               5c per reception, cap 20    (linear)
    KXNFLESCALATOR{RECYDS,RSHYDS} floor(1e4*(y/200)^3)/1e4, y = 10-yd floor, cap 200
    KXNFLESCALATORREC            floor(1e4*(min(s,14)/14)^3)/1e4

Fair value ("KB") replicates each payoff from Kalshi's OWN binary threshold
ladder for the same player (KXNFLRECYDS / KXNFLRSHYDS / KXNFLREC, archived by
the nfl_props scans in archive.db) at the bid/ask MID taken at the same time:

    escalator = sum_k [f(k) - f(k-step)] * P(Y >= k)
    ladder    = per-unit * sum_{y>=1} P(Y >= y)

P(Y >= y) between binary thresholds is linear in logit; tails are extrapolated
(see Survival). ``--bias-correct`` re-scores with binary mids shifted by their
own measured calibration (mid vs Kalshi's official settlement, by price bucket)
because the binary mids are not exactly unbiased.

Usage:
    python scripts/eval_nfl_scalar_props.py --fetch          # refresh public-API cache
    python scripts/eval_nfl_scalar_props.py                  # report from cache
    python scripts/eval_nfl_scalar_props.py --bias-correct --max-spread 0.02
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sqlite3
from bisect import bisect_right
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "data" / "backtest" / "nfl_scalar_props"
DEFAULT_ARCHIVE = ROOT / "data" / "archive.db"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"

# scalar series -> (product, stat, binary series used for replication)
SCALAR_SERIES: dict[str, tuple[str, str, str]] = {
    "KXNFLLADDERRECYDS": ("ladder", "recyds", "KXNFLRECYDS"),
    "KXNFLLADDERRSHYDS": ("ladder", "rshyds", "KXNFLRSHYDS"),
    "KXNFLLADDERREC": ("ladder", "rec", "KXNFLREC"),
    "KXNFLESCALATORRECYDS": ("escalator", "recyds", "KXNFLRECYDS"),
    "KXNFLESCALATORRSHYDS": ("escalator", "rshyds", "KXNFLRSHYDS"),
    "KXNFLESCALATORREC": ("escalator", "rec", "KXNFLREC"),
}
BINARY_SERIES = ("KXNFLRECYDS", "KXNFLRSHYDS", "KXNFLREC")
KICK_BEFORE_OCCURRENCE = timedelta(hours=3)  # occurrence_datetime = kickoff + 3h (verified 1160/1160)
TAKER_RATE = 0.07  # these series are fee_type "quadratic"


# ── payoff schedules (Kalshi custom_strike) ─────────────────────────────────
def _cube(x: float) -> float:
    # +1e-9: (60/200)**3 evaluates to 0.026999.. in floats; Kalshi pays 0.0270
    return math.floor(10000 * x ** 3 + 1e-9) / 10000


def escalator_payoff(stat: str, s: float) -> float:
    if stat == "rec":
        return _cube(min(max(s, 0.0), 14.0) / 14.0)
    y = min(200.0, max(0.0, 10.0 * math.floor(s / 10.0)))
    return _cube(y / 200.0)


def ladder_payoff(stat: str, s: float) -> float:
    if stat == "rec":
        return min(max(s, 0.0), 20.0) * 0.05
    return min(max(s, 0.0), 400.0) * 0.0025


def payoff(product: str, stat: str, s: float) -> float:
    return escalator_payoff(stat, s) if product == "escalator" else ladder_payoff(stat, s)


def taker_fee(p: float) -> float:
    """Kalshi quadratic taker fee per contract (large-order limit, no cent rounding)."""
    return TAKER_RATE * p * (1.0 - p)


# ── survival function from binary threshold points ──────────────────────────
def isotonic_decreasing(y: list[float]) -> list[float]:
    """Pool-adjacent-violators fit of a non-increasing sequence."""
    blocks: list[list[float]] = []
    for v in y:
        blocks.append([v, 1.0])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            v2, w2 = blocks.pop()
            v1, w1 = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2])
    out: list[float] = []
    for v, w in blocks:
        out.extend([v] * int(w))
    return out


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _expit(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class Survival:
    """S(y) = P(Y >= y) from threshold points (T_i, S_i).

    Between knots: linear in logit(S). Below the lowest knot: continue the
    lowest segment's logit slope, capped at ``cap_lo``. Above the top knot:
    exponential tail with the log-slope fitted on the top three knots, never
    flatter than ``max_log_slope``.
    """

    def __init__(self, points: list[tuple[float, float]], stat: str):
        pts = sorted((float(t), float(s)) for t, s in points)
        if len(pts) < 2:
            raise ValueError("need at least two threshold points")
        self.ts = [t for t, _ in pts]
        self.ss = [min(max(s, 1e-4), 0.995) for s in isotonic_decreasing([s for _, s in pts])]
        self.cap_lo = 0.95 if stat == "rshyds" else 0.97
        lo = (_logit(self.ss[1]) - _logit(self.ss[0])) / (self.ts[1] - self.ts[0]) \
            if self.ts[1] > self.ts[0] else -0.05
        self.lo_slope = min(lo, -1e-3)
        top_t = np.array(self.ts[-3:])
        top_s = np.log(np.array(self.ss[-3:]))
        slope = float(np.polyfit(top_t, top_s, 1)[0]) if len(top_t) >= 2 else -0.03
        self.hi_slope = min(slope, -0.15 if stat == "rec" else -0.01)

    def __call__(self, y: float) -> float:
        ts, ss = self.ts, self.ss
        if y <= ts[0]:
            v = _expit(_logit(ss[0]) + self.lo_slope * (y - ts[0]))
            return min(max(v, ss[0]), self.cap_lo)
        if y >= ts[-1]:
            return ss[-1] * math.exp(self.hi_slope * (y - ts[-1]))
        i = bisect_right(ts, y) - 1
        w = (y - ts[i]) / (ts[i + 1] - ts[i])
        return _expit((1 - w) * _logit(ss[i]) + w * _logit(ss[i + 1]))


def fair_value(product: str, stat: str, S) -> float:
    """E[payoff] given S(y) = P(Y >= y) on integer stat values."""
    if stat == "rec":
        if product == "escalator":
            return sum((escalator_payoff("rec", k) - escalator_payoff("rec", k - 1)) * S(k)
                       for k in range(1, 15))
        return 0.05 * sum(S(k) for k in range(1, 21))
    if product == "escalator":
        return sum((escalator_payoff(stat, k) - escalator_payoff(stat, k - 10)) * S(k)
                   for k in range(10, 201, 10))
    return 0.0025 * sum(S(y) for y in range(1, 401))


def clustered_mean(x, groups) -> tuple[float, float, int]:
    """Mean and cluster-robust SE (one cluster per group)."""
    x = np.asarray(x, dtype=float)
    m = float(x.mean())
    s = pd.DataFrame({"r": x - m, "g": list(groups)}).groupby("g")["r"].sum().to_numpy()
    G = len(s)
    if G < 2:
        return m, float("nan"), G
    return m, float(np.sqrt((s ** 2).sum()) / len(x) * np.sqrt(G / (G - 1))), G


# ── public-API fetch (cache) ────────────────────────────────────────────────
async def _get(client, path, params=None, tries=14):
    delay = 1.0
    for _ in range(tries):
        r = await client.get(KALSHI + path, params=params)
        if r.status_code == 429:
            await asyncio.sleep(delay)
            delay = min(delay * 2, 20)
            continue
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"rate-limited out: {path}")


async def _list_markets(client, series: str, status: str | None = None) -> list[dict]:
    out, cursor = [], None
    while True:
        params = {"series_ticker": series, "limit": 1000}
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor
        d = await _get(client, "/markets", params)
        out.extend(d.get("markets", []))
        cursor = d.get("cursor")
        if not cursor or not d.get("markets"):
            return out


async def fetch(cache: Path) -> None:
    import httpx

    (cache / "candles").mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(2)
    async with httpx.AsyncClient(timeout=60) as client:
        settled = []
        for s in SCALAR_SERIES:
            ms = await _list_markets(client, s)
            (cache / f"markets_{s}.json").write_text(json.dumps(ms))
            settled += [(s, m) for m in ms if m["status"] in ("settled", "finalized")]
            print(f"{s}: {len(ms)} markets", flush=True)
        for s in BINARY_SERIES:
            ms = await _list_markets(client, s, status="settled")
            keep = ("ticker", "result", "occurrence_datetime")
            (cache / f"binary_settled_{s}.json").write_text(json.dumps([{k: m.get(k) for k in keep} for m in ms]))
            print(f"{s}: {len(ms)} settled binaries", flush=True)

        async def candles(series, m):
            path = cache / "candles" / f"{m['ticker']}.json"
            if path.exists():
                return
            start = _ts(m["open_time"]) - 3600
            end = min(_ts(m["close_time"]), _ts(m["occurrence_datetime"]) + 6 * 3600)
            async with sem:
                d = await _get(client, f"/series/{series}/markets/{m['ticker']}/candlesticks",
                               {"start_ts": start, "end_ts": end, "period_interval": 60})
            path.write_text(json.dumps((d or {}).get("candlesticks", [])))

        await asyncio.gather(*(candles(s, m) for s, m in settled))


def _ts(s: str) -> int:
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


# ── build + report ──────────────────────────────────────────────────────────
def _load_binaries(archive: Path) -> pd.DataFrame:
    con = sqlite3.connect(f"file:{archive}?mode=ro", uri=True)
    frames = [pd.read_sql_query(
        "SELECT ticker, fetched_at, yes_price, no_price FROM archived_kalshi_markets "
        "WHERE ticker >= ? AND ticker < ?", con, params=(s + "-", s + "-~")) for s in BINARY_SERIES]
    con.close()
    b = pd.concat(frames)
    b = b[(b.yes_price + b.no_price - 1).abs() > 1e-9]  # synthesized NO price = no real bid
    b["bid"] = 1 - b.no_price
    b = b[(b.bid >= 0) & (b.bid <= b.yes_price)].copy()
    b["mid"] = (b.yes_price + b.bid) / 2
    b["fetched"] = pd.to_datetime(b.fetched_at, utc=True, format="ISO8601")
    sp = b.ticker.str.split("-")
    b["series"], b["event"], b["pcode"], b["thr"] = sp.str[0], sp.str[1], sp.str[2], sp.str[3].astype(float)
    return b.sort_values("fetched")


def binary_calibration(binaries: pd.DataFrame, cache: Path, offset_h: int = 1) -> list[tuple[float, float]]:
    """(bucket upper edge, realized - mid) for binary mids vs Kalshi's official result."""
    res, kick = {}, {}
    for s in BINARY_SERIES:
        f = cache / f"binary_settled_{s}.json"
        if not f.exists():
            continue
        for m in json.loads(f.read_text()):
            if m.get("result") in ("yes", "no") and m.get("occurrence_datetime"):
                res[m["ticker"]] = 1 if m["result"] == "yes" else 0
                kick[m["ticker"]] = pd.Timestamp(m["occurrence_datetime"]) - KICK_BEFORE_OCCURRENCE
    b = binaries[binaries.ticker.isin(res)].copy()
    b["t"] = b.ticker.map(kick) - timedelta(hours=offset_h)
    b = b[(b.fetched <= b.t) & (b.fetched >= b.t - timedelta(hours=6))].groupby("ticker").tail(1)
    b["y"] = b.ticker.map(res)
    edges = [0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 1.0]
    b["bk"] = pd.cut(b.mid, [0.0] + edges)
    table = b.groupby("bk", observed=False).apply(lambda h: (h.y - h.mid).mean() if len(h) >= 30 else 0.0)
    return [(e, float(0.0 if np.isnan(v) else v)) for e, v in zip(edges, table.to_numpy())]


def _shift(mid: float, table: list[tuple[float, float]] | None) -> float:
    if not table:
        return mid
    for hi, d in table:
        if mid <= hi:
            return min(max(mid + d, 0.0), 1.0)
    return mid


def build(cache: Path, archive: Path, offsets: list[int], bias_table=None) -> pd.DataFrame:
    binaries = _load_binaries(archive)
    by_player = {k: g for k, g in binaries.groupby(["series", "event", "pcode"])}
    rows = []
    for series, (product, stat, bin_series) in SCALAR_SERIES.items():
        f = cache / f"markets_{series}.json"
        if not f.exists():
            continue
        for m in json.loads(f.read_text()):
            if m["status"] not in ("settled", "finalized"):
                continue
            cpath = cache / "candles" / f"{m['ticker']}.json"
            if not cpath.exists():
                continue
            candles = sorted(json.loads(cpath.read_text()), key=lambda c: c["end_period_ts"])
            _, event, pcode = m["ticker"].split("-")[:3]
            kick = pd.Timestamp(m["occurrence_datetime"]) - KICK_BEFORE_OCCURRENCE
            g = by_player.get((bin_series, event, pcode))
            for off in offsets:
                t = kick - timedelta(hours=off)
                bid, ask = _quote(candles, int(t.timestamp()))
                kb = None
                if g is not None:
                    h = g[(g.fetched <= t) & (g.fetched >= t - timedelta(hours=6))].groupby("thr").tail(1)
                    if len(h) >= 3:
                        pts = [(th, _shift(mid, bias_table)) for th, mid in zip(h.thr, h.mid)]
                        kb = fair_value(product, stat, Survival(pts, stat))
                rows.append(dict(series=series, product=product, stat=stat, ticker=m["ticker"], event=event,
                                 kick=kick, offset_h=off, bid=bid, ask=ask, kb=kb,
                                 settle=float(m["settlement_value_dollars"])))
    df = pd.DataFrame(rows)
    k = df.kick.dt.tz_convert("America/New_York")
    df["slate"] = k.dt.strftime("%m-%d ") + np.select(
        [k.dt.hour < 12, k.dt.hour < 15, k.dt.hour < 18], ["intl", "early", "late"], "night")
    return df


def _quote(candles: list[dict], ts: int) -> tuple[float | None, float | None]:
    best = None
    for c in candles:
        if c["end_period_ts"] > ts:
            break
        best = c
    if best is None:
        return None, None
    b = (best.get("yes_bid") or {}).get("close_dollars")
    a = (best.get("yes_ask") or {}).get("close_dollars")
    b = float(b) if b is not None and float(b) > 0 else None
    a = float(a) if a is not None and float(a) < 1 else None
    return b, a


def report(df: pd.DataFrame, max_spread: float) -> None:
    q = df.dropna(subset=["bid", "ask", "kb"]).copy()
    q = q[(q.ask - q.bid) <= max_spread]
    q["mid"] = (q.bid + q.ask) / 2
    q["no_exp"] = q.bid - q.kb - q.bid.map(taker_fee)
    q["no_pnl"] = q.bid - q.settle - q.bid.map(taker_fee)
    q["yes_pnl"] = q.settle - q.ask - q.ask.map(taker_fee)
    print(f"quotes with spread <= {max_spread*100:.0f}c, cents per contract, SE clustered by kickoff slate\n")
    hdr = (f"{'series':22s} {'T-':>4s} {'n':>4s} {'mid':>6s} {'KB':>6s} {'settle':>6s} {'mid/KB':>7s} "
           f"{'rich slates':>11s} {'NO exp':>7s} {'NO real':>15s} {'YES real':>9s}")
    print(hdr)
    for (s, off), g in q.groupby(["series", "offset_h"]):
        rich = g.groupby("slate").apply(lambda h: (h.mid - h.kb).mean() > 0)
        m, se, _ = clustered_mean(g.no_pnl, g.slate)
        print(f"{s:22s} {off:>3d}h {len(g):4d} {g.mid.mean()*100:6.2f} {g.kb.mean()*100:6.2f} "
              f"{g.settle.mean()*100:6.2f} {(g.mid / g.kb).median()-1:+7.0%} {int(rich.sum()):>5d}/{len(rich):<5d} "
              f"{g.no_exp.mean()*100:+7.2f} {m*100:+7.2f} ±{se*100:5.2f} {g.yes_pnl.mean()*100:+9.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fetch", action="store_true", help="refresh the public-API cache first")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    ap.add_argument("--offsets", default="24,1", help="entry offsets in hours before kickoff")
    ap.add_argument("--max-spread", type=float, default=0.02)
    ap.add_argument("--bias-correct", action="store_true",
                    help="shift binary mids by their measured calibration before replicating")
    args = ap.parse_args()
    if args.fetch:
        asyncio.run(fetch(args.cache))
    offsets = [int(x) for x in args.offsets.split(",")]
    table = None
    if args.bias_correct:
        table = binary_calibration(_load_binaries(args.archive), args.cache)
        print("binary calibration (bucket upper edge: realized - mid, pp):",
              ", ".join(f"{hi:.2f}: {d*100:+.1f}" for hi, d in table), "\n")
    report(build(args.cache, args.archive, offsets, table), args.max_spread)


if __name__ == "__main__":
    main()
