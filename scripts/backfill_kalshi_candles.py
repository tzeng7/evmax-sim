"""Backfill hourly Kalshi candlestick history for archived laddered tickers.

Kalshi keeps per-market candle history (bid/ask OHLC) even after settlement,
so the full listing→tip price trail of every spread/total ticker this season
can be reconstructed NOW — instead of waiting for weeks of live hourly
watch-listings capture to accumulate an evaluable sample.

For each distinct (sector, spread/total) ticker in archived_kalshi_markets,
fetch hourly candles over [tip - 72h, tip] and write one snapshot row per
candle into archived_kalshi_markets:

    session_id = f"candlebf-{end_period_ts}"      (varies per candle-hour, so
                                                   UNIQUE(session_id, ticker)
                                                   admits the full trail)
    fetched_at = candle end time (ISO UTC)
    yes_price  = yes_ask close   (crossable YES price, matches convention)
    no_price   = 1 - yes_bid close (crossable NO price); 1.0 when no bid

Idempotent: INSERT OR IGNORE, plus tickers that already have candlebf rows are
skipped unless --refresh. Read the result back with
`evmax cleanup listings-eval --session-prefix candlebf --bid-from-no-price`
or scripts/eval_wnba_anchored_backfill.py.

Usage:
    python scripts/backfill_kalshi_candles.py [--sector wnba] [--dry-run]
        [--market-types spread,total] [--window-h 72] [--refresh] [--limit N]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# The worktree carries its own near-empty data/archive.db; point at the main
# checkout's live archive (same pattern as analyze_wnba_listings_lay_robustness).
MAIN_REPO = Path("/Users/ktzeng/Projects/evmax")
import evmax.archiver as _archiver_mod  # noqa: E402
if (MAIN_REPO / "data" / "archive.db").exists():
    _archiver_mod.DB_PATH = MAIN_REPO / "data" / "archive.db"

from evmax.archiver import DataArchiver, _get_connection  # noqa: E402
from evmax.clients.kalshi import KalshiClient  # noqa: E402

CHUNK = 150  # tickers fetched concurrently per gather batch


def _parse_ts(raw: str) -> datetime:
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def discover_tickers(sector: str, market_types: tuple[str, ...], refresh: bool) -> list[dict]:
    """One representative context row per archived laddered ticker."""
    placeholders = ",".join("?" * len(market_types))
    with _get_connection() as conn:
        rows = conn.execute(
            f"""SELECT ticker, market_type,
                       MAX(line) AS line, MAX(yes_team) AS yes_team,
                       MAX(team_home) AS team_home, MAX(team_away) AS team_away,
                       MAX(event_date) AS event_date, MAX(event_id) AS event_id
                FROM archived_kalshi_markets
                WHERE sector = ? AND market_type IN ({placeholders})
                  AND session_id NOT LIKE 'candlebf%'
                GROUP BY ticker""",
            (sector, *market_types),
        ).fetchall()
        done: set[str] = set()
        if not refresh:
            done = {
                r["ticker"]
                for r in conn.execute(
                    "SELECT DISTINCT ticker FROM archived_kalshi_markets "
                    "WHERE sector = ? AND session_id LIKE 'candlebf%'",
                    (sector,),
                )
            }
    out = []
    for r in rows:
        if r["ticker"] in done:
            continue
        if not r["event_date"]:
            continue  # no tip anchor — evaluate_sector would skip it anyway
        out.append(dict(r))
    return out


async def fetch_one(client: KalshiClient, info: dict, window_h: float) -> tuple[dict, list[dict]]:
    ticker = info["ticker"]
    series = ticker.split("-", 1)[0]
    # archived_kalshi_markets.event_date is usually the noon-UTC ticker-date
    # anchor, NOT the real tip (which lives on archived_sharp_odds and lands
    # up to ~14h later). Extend the window +24h past the anchor; the
    # evaluator's pre-tip filter (real Pinnacle tip) trims precisely, and
    # close selection ignores post-tip rows too.
    anchor = _parse_ts(info["event_date"])
    end_ts = int((anchor + timedelta(hours=24)).timestamp())
    start_ts = int((anchor - timedelta(hours=window_h)).timestamp())
    candles = await client.get_market_candlesticks(
        series, ticker, start_ts=start_ts, end_ts=end_ts,
        period_interval=60, include_latest_before_start=True,
    )
    return info, candles


def snapshot_rows(info: dict, candles: list[dict]) -> list[tuple[int, dict]]:
    """(end_ts, snapshot-dict) pairs for archive_kalshi_snapshot."""
    out = []
    for c in candles:
        ask = c["yes_ask_close"]
        if ask is None or not (0 < ask < 1):
            continue
        bid = c["yes_bid_close"]
        no_price = (1.0 - bid) if bid is not None and 0 < bid < 1 else 1.0
        out.append((c["end_ts"], {
            "ticker": info["ticker"],
            "yes_price": ask,
            "no_price": no_price,
            "market_type": info["market_type"],
            "line": info["line"],
            "yes_team": info["yes_team"],
            "team_home": info["team_home"],
            "team_away": info["team_away"],
            "event_date": info["event_date"],
            "event_id": info["event_id"],
        }))
    return out


async def run(args: argparse.Namespace) -> None:
    market_types = tuple(t.strip() for t in args.market_types.split(",") if t.strip())
    tickers = discover_tickers(args.sector, market_types, args.refresh)
    if args.limit:
        tickers = tickers[: args.limit]
    by_type: dict[str, int] = defaultdict(int)
    for t in tickers:
        by_type[t["market_type"]] += 1
    print(f"sector={args.sector}  tickers to backfill: {len(tickers)} "
          f"({dict(by_type)})  window={args.window_h}h  db={_archiver_mod.DB_PATH}")
    if args.dry_run:
        for t in tickers[:10]:
            anchor = _parse_ts(t["event_date"])
            print(f"  {t['ticker']:<44} {t['market_type']:<7} "
                  f"[{(anchor - timedelta(hours=args.window_h)).isoformat()} → "
                  f"{(anchor + timedelta(hours=24)).isoformat()}]")
        if len(tickers) > 10:
            print(f"  ... and {len(tickers) - 10} more")
        print("dry-run: no requests made, nothing written")
        return

    archiver = DataArchiver()
    total_rows = 0
    empty = 0
    async with KalshiClient() as client:
        for i in range(0, len(tickers), CHUNK):
            batch = tickers[i:i + CHUNK]
            results = await asyncio.gather(
                *(fetch_one(client, t, args.window_h) for t in batch),
                return_exceptions=True,
            )
            # Group rows by candle-hour so one archive call covers every ticker
            # sharing that session_id/fetched_at.
            by_hour: dict[int, list[dict]] = defaultdict(list)
            for res in results:
                if isinstance(res, Exception):
                    empty += 1
                    continue
                info, candles = res
                rows = snapshot_rows(info, candles)
                if not rows:
                    empty += 1
                    continue
                for end_ts, snap in rows:
                    by_hour[end_ts].append(snap)
            for end_ts, snaps in by_hour.items():
                fa = datetime.fromtimestamp(end_ts, tz=timezone.utc)
                total_rows += archiver.archive_kalshi_snapshot(
                    f"candlebf-{end_ts}", args.sector, snaps, fetched_at=fa,
                )
            print(f"  [{min(i + CHUNK, len(tickers))}/{len(tickers)}] "
                  f"rows so far: {total_rows}  no-data tickers: {empty}")
    print(f"done: {total_rows} candle rows written, {empty} tickers had no candle data")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sector", default="wnba")
    p.add_argument("--market-types", default="spread,total")
    p.add_argument("--window-h", type=float, default=72.0,
                   help="Hours before tip to start the candle window.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the request plan; no API calls, no writes.")
    p.add_argument("--refresh", action="store_true",
                   help="Re-fetch tickers that already have candlebf rows.")
    p.add_argument("--limit", type=int, default=0,
                   help="Backfill at most N tickers (smoke-test aid).")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
