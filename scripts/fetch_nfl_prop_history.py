"""Pull settled NFL player-prop history from Kalshi for backtesting.

Enumerates every settled event across the six real NFL prop series via
/events?series_ticker=...&status=settled, then fetches the full market
list for each event from /historical/markets?event_ticker=... and writes
the raw responses to data/backtest/nfl_props/kalshi_raw.json.

Usage:
    uv run python scripts/fetch_nfl_prop_history.py
    uv run python scripts/fetch_nfl_prop_history.py --out path/to/dir
    uv run python scripts/fetch_nfl_prop_history.py --series KXNFLPASSYDS

The Kalshi historical API is unauthenticated but gated at a rolling
3-month cutoff (see /historical/cutoff). Everything older than that is
fair game. Series, event enumeration, and market fetches are all
paginated via cursor.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import httpx

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"

DEFAULT_SERIES = [
    "KXNFLPASSYDS",
    "KXNFLRSHYDS",
    "KXNFLRECYDS",
    "KXNFLANYTD",
    "KXNFLPASSTDS",
    "KXNFLREC",
]

DEFAULT_OUT = Path("data/backtest/nfl_props")

# Be polite — Kalshi public API is generous but not infinite.
REQUEST_DELAY_S = 0.05
TIMEOUT_S = 20.0


async def _get_json(
    client: httpx.AsyncClient, path: str, params: Optional[dict] = None
) -> dict:
    """GET a Kalshi API path with modest retry on transient errors."""
    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            r = await client.get(path, params=params, timeout=TIMEOUT_S)
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            last_err = e
            await asyncio.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"GET {path} failed after 3 attempts: {last_err}")


async def fetch_cutoff(client: httpx.AsyncClient) -> dict:
    """Return the /historical/cutoff response — tells us where history starts."""
    return await _get_json(client, "/historical/cutoff")


async def enumerate_events(
    client: httpx.AsyncClient, series_ticker: str
) -> list[dict]:
    """Paginate /events for one series, return every settled event dict."""
    events: list[dict] = []
    cursor = ""
    pages = 0
    while True:
        params = {
            "series_ticker": series_ticker,
            "status": "settled",
            "limit": 200,
        }
        if cursor:
            params["cursor"] = cursor
        data = await _get_json(client, "/events", params=params)
        batch = data.get("events", []) or []
        events.extend(batch)
        cursor = data.get("cursor") or ""
        pages += 1
        await asyncio.sleep(REQUEST_DELAY_S)
        if not cursor or not batch:
            break
        if pages > 50:
            print(
                f"  ! {series_ticker}: stopping event enumeration at 50 pages",
                file=sys.stderr,
            )
            break
    return events


async def fetch_event_markets(
    client: httpx.AsyncClient, event_ticker: str
) -> list[dict]:
    """Fetch all historical markets for one event_ticker via /historical/markets."""
    markets: list[dict] = []
    cursor = ""
    pages = 0
    while True:
        params: dict[str, Any] = {"event_ticker": event_ticker, "limit": 200}
        if cursor:
            params["cursor"] = cursor
        data = await _get_json(client, "/historical/markets", params=params)
        batch = data.get("markets", []) or []
        markets.extend(batch)
        cursor = data.get("cursor") or ""
        pages += 1
        await asyncio.sleep(REQUEST_DELAY_S)
        if not cursor or not batch:
            break
        if pages > 10:
            break
    return markets


async def fetch_series(
    client: httpx.AsyncClient, series_ticker: str
) -> tuple[list[dict], list[dict]]:
    """For one series, return (events, all_markets)."""
    print(f"  [{series_ticker}] enumerating events …")
    events = await enumerate_events(client, series_ticker)
    print(f"  [{series_ticker}] {len(events)} events found")

    all_markets: list[dict] = []
    for i, ev in enumerate(events, start=1):
        et = ev.get("event_ticker")
        if not et:
            continue
        try:
            mkts = await fetch_event_markets(client, et)
        except Exception as e:
            print(f"    ! {et}: {e}", file=sys.stderr)
            continue
        # Annotate each market with the series it came from for easier joins later
        for m in mkts:
            m["_series_ticker"] = series_ticker
        all_markets.extend(mkts)
        if i % 25 == 0 or i == len(events):
            print(
                f"    [{series_ticker}] events {i}/{len(events)} "
                f"→ {len(all_markets)} markets so far"
            )
    return events, all_markets


async def main_async(out_dir: Path, series_list: list[str]) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    async with httpx.AsyncClient(base_url=KALSHI_BASE) as client:
        cutoff = await fetch_cutoff(client)
        print(f"historical cutoff: {cutoff}")
        all_markets: list[dict] = []
        per_series_events: dict[str, int] = {}
        per_series_markets: dict[str, int] = {}
        for s in series_list:
            events, mkts = await fetch_series(client, s)
            per_series_events[s] = len(events)
            per_series_markets[s] = len(mkts)
            all_markets.extend(mkts)

    raw_path = out_dir / "kalshi_raw.json"
    raw_path.write_text(json.dumps(all_markets))
    manifest = {
        "fetched_at": int(time.time()),
        "cutoff": cutoff,
        "series": series_list,
        "n_events_per_series": per_series_events,
        "n_markets_per_series": per_series_markets,
        "total_markets": len(all_markets),
        "duration_s": round(time.time() - t0, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print()
    print(f"wrote {raw_path} ({len(all_markets)} markets)")
    print(f"wrote {out_dir / 'manifest.json'}")
    print(f"elapsed: {manifest['duration_s']}s")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"output directory (default: {DEFAULT_OUT})",
    )
    p.add_argument(
        "--series",
        type=str,
        default=",".join(DEFAULT_SERIES),
        help="comma-separated series tickers to pull",
    )
    args = p.parse_args()
    series_list = [s.strip() for s in args.series.split(",") if s.strip()]
    return asyncio.run(main_async(args.out, series_list))


if __name__ == "__main__":
    sys.exit(main())
