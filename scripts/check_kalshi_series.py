#!/usr/bin/env python3
"""Check SECTOR_SERIES_MAP against live Kalshi series API for drift.

Three buckets:
  [OK]    — our ticker exists on Kalshi
  [STALE] — our ticker is NOT in Kalshi's series list (phantom / typo / retired)
  [NEW]   — a Kalshi sports series matches one of our sector prefixes but
             isn't in SECTOR_SERIES_MAP (potential missed opportunity)

Exit code 0 if no STALE entries. Exit code 1 if any STALE found.
NEW entries are informational — they don't fail the check.

Usage:
    python3 scripts/check_kalshi_series.py
    python3 scripts/check_kalshi_series.py --probe  # also check market counts for [OK] tickers
"""

from __future__ import annotations

import argparse
import sys

import httpx

KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"
TIMEOUT = 30


def fetch_sports_series() -> set[str]:
    """Return the set of all sports series tickers on Kalshi."""
    r = httpx.get(
        f"{KALSHI_API}/series",
        params={"category": "Sports", "limit": 5000},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return {s["ticker"] for s in r.json().get("series", [])}


def fetch_market_count(series_ticker: str) -> int:
    """Return the total number of markets (any status) for a series."""
    r = httpx.get(
        f"{KALSHI_API}/markets",
        params={"series_ticker": series_ticker, "limit": 1},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return len(r.json().get("markets", []))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Also probe each [OK] ticker for market count (slower, 1 req per ticker).",
    )
    args = parser.parse_args()

    from evmax.clients.kalshi import SECTOR_SERIES_MAP

    print("Fetching Kalshi sports series …")
    kalshi_tickers = fetch_sports_series()
    print(f"  {len(kalshi_tickers)} sports series on Kalshi\n")

    # Flatten our map into {ticker: sector}
    our_tickers: dict[str, str] = {}
    for sector, tickers in SECTOR_SERIES_MAP.items():
        for t in tickers:
            our_tickers[t] = sector

    # Build sector prefixes for NEW detection. E.g., if we have KXNBAGAME
    # and KXNBASPREAD under "nba", the prefix is "KXNBA".
    sector_prefixes: dict[str, set[str]] = {}
    for sector, tickers in SECTOR_SERIES_MAP.items():
        prefixes = set()
        for t in tickers:
            # Take the longest alpha prefix before any digit or special char
            prefix = ""
            for ch in t:
                if ch.isalpha():
                    prefix += ch
                else:
                    break
            if len(prefix) >= 4:
                prefixes.add(prefix)
        sector_prefixes[sector] = prefixes

    ok_tickers: list[tuple[str, str]] = []
    stale_tickers: list[tuple[str, str]] = []
    new_tickers: list[tuple[str, str]] = []  # (ticker, matched_sector)

    for ticker, sector in sorted(our_tickers.items(), key=lambda x: (x[1], x[0])):
        if ticker in kalshi_tickers:
            ok_tickers.append((ticker, sector))
        else:
            stale_tickers.append((ticker, sector))

    # Find NEW: Kalshi tickers matching our sector prefixes but not in our map
    our_set = set(our_tickers.keys())
    for kt in sorted(kalshi_tickers):
        if kt in our_set:
            continue
        for sector, prefixes in sector_prefixes.items():
            if any(kt.startswith(p) for p in prefixes):
                new_tickers.append((kt, sector))
                break

    # Report
    for ticker, sector in ok_tickers:
        suffix = ""
        if args.probe:
            count = fetch_market_count(ticker)
            suffix = f"  ({count} markets)" if count > 0 else "  ⚠ 0 markets!"
        print(f"  [OK]    {ticker:<20s}  ({sector}){suffix}")

    for ticker, sector in stale_tickers:
        print(f"  [STALE] {ticker:<20s}  ({sector}) — not in Kalshi series API")

    if new_tickers:
        print(f"\n  {len(new_tickers)} new Kalshi series matching our sector prefixes:")
        for ticker, sector in new_tickers[:30]:
            print(f"  [NEW]   {ticker:<20s}  (matches {sector} prefix)")
        if len(new_tickers) > 30:
            print(f"  … and {len(new_tickers) - 30} more")

    # Summary
    print(f"\n{'=' * 50}")
    print(f"  OK: {len(ok_tickers)}  |  STALE: {len(stale_tickers)}  |  NEW: {len(new_tickers)}")
    if stale_tickers:
        print(f"\n  ❌ {len(stale_tickers)} stale ticker(s) — these return zero markets.")
        print("  Fix SECTOR_SERIES_MAP in evmax/clients/kalshi.py")
        return 1
    else:
        print("\n  ✓ All tickers verified against Kalshi.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
