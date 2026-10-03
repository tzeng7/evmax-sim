#!/usr/bin/env python3
"""Compare Kalshi's per-series fee fields with what ``evmax.fees`` assumes.

``evmax.fees`` applies ONE schedule to every Kalshi series: the quadratic
``0.07·P·(1−P)`` taker fee, a maker rate of 25% of that, and a multiplier of 1.
Kalshi's series API (``GET /series/{ticker}``) publishes ``fee_type`` and
``fee_multiplier`` per series, and they can disagree with the published PDF
(2026-10-03: KXMLBGAME reports 0.5 where the PDF table says 1). The PDF check in
``check_kalshi_fees.py`` cannot see that.

This canary fetches the series we trade (``SECTOR_SERIES_MAP``) and reports every
series whose ``fee_type`` is not ``quadratic_with_maker_fees`` or whose multiplier
is not 1.0. It reports only — it never edits ``fees.py``. Which side of a
disagreement Kalshi really bills is settled by real fills:
``evmax cleanup maker-orders`` prints the implied multiplier from the account's
own order history.

Usage:
  check_kalshi_series_fees.py                  # fetch + print; exit 3 on any mismatch
  check_kalshi_series_fees.py --sectors nfl,ncaaf
  check_kalshi_series_fees.py --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

ASSUMED_FEE_TYPE = "quadratic_with_maker_fees"
ASSUMED_MULTIPLIER = 1.0
_TOL = 1e-9


def classify_series_fees(api: dict[str, Optional[dict]]) -> list[dict]:
    """PURE. ``{series: series_payload_or_None}`` → one finding per series.

    ``status`` is ``match`` (type + multiplier as assumed), ``mismatch`` (either
    differs; both values are reported) or ``unavailable`` (no payload).
    """
    out: list[dict] = []
    for series, d in sorted(api.items()):
        if not d:
            out.append({"series": series, "status": "unavailable"})
            continue
        fee_type = d.get("fee_type")
        mult = d.get("fee_multiplier")
        try:
            mult_f = float(mult)
        except (TypeError, ValueError):
            mult_f = None
        ok = fee_type == ASSUMED_FEE_TYPE and mult_f is not None and abs(mult_f - ASSUMED_MULTIPLIER) <= _TOL
        out.append({
            "series": series,
            "status": "match" if ok else "mismatch",
            "fee_type": fee_type,
            "fee_multiplier": mult_f,
        })
    return out


async def _fetch(series: list[str]) -> dict[str, Optional[dict]]:
    import httpx

    from evmax.settings import get_settings

    base = get_settings().kalshi_base_url.rstrip("/")
    out: dict[str, Optional[dict]] = {}
    async with httpx.AsyncClient(timeout=20.0) as client:
        sem = asyncio.Semaphore(5)

        async def one(s: str) -> None:
            async with sem:
                try:
                    r = await client.get(f"{base}/series/{s}")
                    r.raise_for_status()
                    out[s] = r.json().get("series")
                except Exception:  # noqa: BLE001 — a failed fetch is reported, not fatal
                    out[s] = None

        await asyncio.gather(*(one(s) for s in series))
    return out


def traded_series(sectors: Optional[list[str]] = None) -> list[str]:
    from evmax.clients.kalshi import SECTOR_SERIES_MAP

    seen: list[str] = []
    for sector, series in SECTOR_SERIES_MAP.items():
        if sectors and sector not in sectors:
            continue
        for s in series:
            if s not in seen:
                seen.append(s)
    return seen


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sectors", help="comma-separated sectors (default: all)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    sectors = [s.strip().lower() for s in args.sectors.split(",")] if args.sectors else None
    findings = classify_series_fees(asyncio.run(_fetch(traded_series(sectors))))
    bad = [f for f in findings if f["status"] != "match"]
    if args.json:
        print(json.dumps(findings, indent=2))
    else:
        for f in bad:
            if f["status"] == "unavailable":
                print(f"UNAVAILABLE  {f['series']}")
            else:
                print(f"MISMATCH     {f['series']:<22} fee_type={f['fee_type']} fee_multiplier={f['fee_multiplier']}")
        print(f"{len(findings) - len(bad)}/{len(findings)} series match fees.py "
              f"({ASSUMED_FEE_TYPE}, multiplier {ASSUMED_MULTIPLIER:g}).")
    return 3 if any(f["status"] == "mismatch" for f in bad) else 0


if __name__ == "__main__":
    sys.exit(main())
