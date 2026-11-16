"""Polymarket US maker-fill replay on backfilled price history.

For every resolved Polymarket US candidate in predictions.db, fetch the
market's own minute-level ask/bid history (public /v1/price-history, cached on
disk), simulate a resting bid from the time the row was logged to tip-off, and
compare the maker path with crossing the ask.

Logic and definitions: evmax/agents/cleanup/polyus_maker_replay.py.
Read-only on predictions.db. Writes only the history cache.

Usage:
    python scripts/eval_polyus_maker_replay.py --pred-db data/predictions.db
    python scripts/eval_polyus_maker_replay.py --offline            # cache only
    python scripts/eval_polyus_maker_replay.py --since 2026-09-01 --json out.json

Requests carry market slugs only. Default pace is 8 requests/s (the public
limit is 20/s per IP).
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from evmax.agents.cleanup import polyus_maker_replay as mr  # noqa: E402

GATEWAY = "https://gateway.polymarket.us"
CACHE_VERSION = 1


def load_rows(pred_db: str, since: str | None, sectors: set[str]) -> list[dict]:
    from evmax.agents.cleanup.contamination import is_contaminated
    from evmax.cli.commands.shadow import game_key

    conn = sqlite3.connect(f"file:{pred_db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    sql = """SELECT p.market_id, p.event_id, p.sector, p.market_type, p.line,
                    p.model_sources, p.logged_at, p.minutes_to_tipoff, p.event_date,
                    p.kalshi_yes_price, p.captured_yes_price, p.blended_true_prob,
                    p.mode, o.outcome
             FROM ev_predictions p JOIN ev_outcomes o USING(market_id)
             WHERE p.venue = 'polymarket_us' AND o.outcome IS NOT NULL
               AND (COALESCE(p.voided, 0) = 0 OR p.void_reason = 'stale_reverted')
               AND p.minutes_to_tipoff > 0 AND p.logged_at IS NOT NULL"""
    params: list = []
    if since:
        sql += " AND p.event_date >= ?"
        params.append(since)
    seen, out = set(), []
    for r in conn.execute(sql + " ORDER BY p.id", params):
        if r["market_id"] in seen or (sectors and r["sector"] not in sectors):
            continue
        if is_contaminated(r["sector"], r["market_type"], r["model_sources"], r["line"]):
            continue
        seen.add(r["market_id"])
        d = dict(r)
        d["game"] = game_key(r["event_id"])
        # `:no` rows store their own side's ask in kalshi_yes_price.
        d["entry_price"] = r["kalshi_yes_price"]
        out.append(d)
    conn.close()
    return out


class Cache:
    """One gzip JSON per slug: {"v", "meta", "days": {date: [buckets]}}."""

    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def _path(self, slug: str) -> Path:
        return self.root / f"{slug}.json.gz"

    def load(self, slug: str) -> dict:
        p = self._path(slug)
        if p.exists():
            try:
                with gzip.open(p, "rt") as f:
                    d = json.load(f)
                if d.get("v") == CACHE_VERSION:
                    return d
            except (OSError, ValueError):
                pass
        return {"v": CACHE_VERSION, "meta": None, "days": {}}

    def save(self, slug: str, data: dict) -> None:
        tmp = self._path(slug).with_suffix(".tmp")
        with gzip.open(tmp, "wt") as f:
            json.dump(data, f, separators=(",", ":"))
        tmp.replace(self._path(slug))


async def fetch_all(rows: list[dict], cache: Cache, rps: float, offline: bool) -> dict:
    """Fill the cache for every slug the rows need. Returns {slug: cache dict}."""
    import httpx
    from aiolimiter import AsyncLimiter

    by_slug: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        parsed = mr.parse_market_id(r["market_id"])
        if parsed:
            by_slug[parsed[0]].append(r)

    limiter = AsyncLimiter(max(1, int(rps)), 1.0)
    stats = Counter()
    out: dict[str, dict] = {}
    now = datetime.now(timezone.utc)

    async def get(client, path, params=None):
        for attempt in range(4):
            async with limiter:
                resp = await client.get(path, params=params)
            if resp.status_code == 429:
                stats["429"] += 1
                await asyncio.sleep(float(resp.headers.get("retry-after", 2 ** attempt)))
                continue
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        return None

    async def one(client, slug, slug_rows, sem):
        async with sem:
            data = cache.load(slug)
            dirty = False
            if data["meta"] is None and not offline:
                try:
                    payload = await get(client, f"/v1/market/slug/{slug}")
                    stats["meta_calls"] += 1
                except Exception:  # noqa: BLE001
                    payload = None
                    stats["meta_errors"] += 1
                if payload and payload.get("market"):
                    m = payload["market"]
                    data["meta"] = {
                        "gameStartTime": m.get("gameStartTime"),
                        "orderPriceMinTickSize": m.get("orderPriceMinTickSize"),
                        "marketSides": [
                            {"description": s.get("description"), "long": s.get("long"),
                             "team": {"abbreviation": (s.get("team") or {}).get("abbreviation")}}
                            for s in m.get("marketSides") or []
                        ],
                    }
                    dirty = True
            if data["meta"] is not None:
                meta = mr.parse_market_meta(slug, data["meta"])
                starts = [t for t in (mr._utc(r["logged_at"]) for r in slug_rows) if t]
                tip = meta.tip or max(
                    (t + timedelta(minutes=float(r["minutes_to_tipoff"]))
                     for t, r in zip(starts, slug_rows)), default=None)
                if starts and tip:
                    # one bucket of slack before the earliest entry
                    first = min(starts) - timedelta(seconds=mr.ENTRY_MAX_STALENESS_S)
                    for day in mr.day_windows(first, min(tip, now)):
                        if day in data["days"] or offline:
                            continue
                        a = int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp())
                        try:
                            payload = await get(client, "/v1/price-history", {
                                "symbol": slug, "fidelity": 1,
                                "timestamp.startTimestamp": a,
                                "timestamp.endTimestamp": a + 86400,
                            })
                            stats["history_calls"] += 1
                        except Exception:  # noqa: BLE001
                            stats["history_errors"] += 1
                            continue
                        hist = (payload or {}).get("history") or []
                        # keep only the day's own buckets (ranges are inclusive)
                        data["days"][day] = [
                            b for b in mr.bucketize(hist) if a <= b[0] < a + 86400
                        ]
                        dirty = True
            if dirty:
                cache.save(slug, data)
            out[slug] = data
            stats["slugs_done"] += 1
            if stats["slugs_done"] % 200 == 0:
                print(f"  ... {stats['slugs_done']}/{len(by_slug)} slugs, "
                      f"{stats['history_calls']} history calls, {stats['429']} 429s",
                      file=sys.stderr, flush=True)

    sem = asyncio.Semaphore(16)
    async with httpx.AsyncClient(base_url=GATEWAY, timeout=30.0,
                                 headers={"User-Agent": "evmax-replay/1.0"}) as client:
        await asyncio.gather(*(one(client, s, rs, sem) for s, rs in by_slug.items()))
    out["__stats__"] = dict(stats)
    return out


def run_replay(rows: list[dict], cached: dict) -> tuple[list[dict], Counter]:
    results, skipped = [], Counter()
    for r in rows:
        parsed = mr.parse_market_id(r["market_id"])
        data = cached.get(parsed[0]) if parsed else None
        if not data or data.get("meta") is None:
            skipped["no_market_meta"] += 1
            continue
        meta = mr.parse_market_meta(parsed[0], data["meta"])
        buckets = sorted(b for day in data["days"].values() for b in day)
        res, why = mr.replay_row(r, meta, buckets)
        if res is None:
            skipped[why] += 1
            continue
        res["mode"] = r.get("mode")
        results.append(res)
    return results, skipped


def _ci(block: dict, nd: int = 2) -> str:
    if not block or block["mean"] is None:
        return "—"
    s = f"{block['mean']:+.{nd}f}"
    if block["ci95"]:
        s += f" [{block['ci95'][0]:+.{nd}f}, {block['ci95'][1]:+.{nd}f}]"
    return s


def print_group(label: str, results: list[dict]) -> dict | None:
    out = {}
    if len(results) < 8:
        print(f"\n== {label}: n={len(results)} (too few)")
        return None
    for fill in ("touch", "thru"):
        s = mr.summarize(results, fill)
        out[fill] = s
        if fill == "touch":
            diff = s["median_abs_logged_ask_diff_pp"]
            print(f"\n== {label}: {s['n']} rows / {s['games']} games | spread "
                  f"{s['mean_spread_pp']:.2f}pp | median lead {s['median_lead_h']:.0f}h | "
                  f"history ask vs logged ask, median |diff| "
                  f"{'—' if diff is None else f'{diff:.2f}pp'}")
            print(f"   taker: CLV gross {_ci(s['taker_clv_gross'])} | net of fee per candidate "
                  f"{_ci(s['taker_net_per_candidate'])}")
        fr = s["filled_realized"]
        print(f"   maker ({fill:5}): fill {s['fill_rate'] * 100:4.0f}% | CLV if filled, to close ask "
              f"{_ci(s['maker_clv_filled_ask'])} / to close mid {_ci(s['maker_clv_filled_mid'])} | "
              f"unfilled would-be taker CLV {_ci(s['unfilled_taker_clv_gross'])}")
        roi = "—" if fr["roi_net_pct"] is None else f"{fr['roi_net_pct']:+.1f}%"
        print(f"                  net per candidate: ask basis {_ci(s['maker_net_per_candidate'])} | "
              f"mid basis {_ci(s['maker_net_per_candidate_mid'])} | filled realized ROI {roi} "
              f"(n={fr['n']})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-db", default="data/predictions.db")
    ap.add_argument("--cache-dir", default=str(_REPO / "data" / "backtest" / "polyus_history"))
    ap.add_argument("--since", default="2026-09-01", help="event_date >= (YYYY-MM-DD)")
    ap.add_argument("--sectors", default="", help="comma list; empty = all")
    ap.add_argument("--rps", type=float, default=8.0)
    ap.add_argument("--offline", action="store_true", help="use the cache only")
    ap.add_argument("--min-games", type=int, default=15, help="smallest group to print")
    ap.add_argument("--json", default="", help="write the summaries to this file")
    args = ap.parse_args()

    sectors = {s.strip() for s in args.sectors.split(",") if s.strip()}
    rows = load_rows(args.pred_db, args.since, sectors)
    print(f"candidates: {len(rows)} resolved, clean, pre-game Polymarket US rows "
          f"(event_date >= {args.since})")
    cached = asyncio.run(fetch_all(rows, Cache(Path(args.cache_dir)), args.rps, args.offline))
    print("fetch:", cached.pop("__stats__"))
    results, skipped = run_replay(rows, cached)
    print(f"replayed: {len(results)}   skipped: {dict(skipped)}")

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in results:
        groups[(r["sector"], r["market_type"])].append(r)
    report = {"all": print_group("ALL", results)}
    for key, rs in sorted(groups.items(), key=lambda kv: -len({r['game'] for r in kv[1]})):
        if len({r["game"] for r in rs}) < args.min_games:
            continue
        report["/".join(key)] = print_group(" ".join(key), rs)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
