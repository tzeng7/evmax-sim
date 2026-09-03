"""Verify every Pinnacle guest league id in SECTOR_SPORT_LEAGUES is still served.

Companion to scripts/check_kalshi_series.py for season-start checklist item 4
(docs/SEASON_START.md §2). Pinnacle re-cuts league ids between seasons — the
NFL went 258 → 889 for 2026 — and a stale id silently returns ZERO matchups,
i.e. no sharp anchor and no EV for the whole sector. Nothing else in the
pipeline can tell that apart from an off-season.

For each id-configured sector this hits `GET /sports/{sport_id}/leagues` once
and reports each configured id as OK (listed) or STALE (not listed), and lists
the sport's currently-served leagues whenever an id is stale so the operator
can read the replacement straight off the output. Exit 1 on any STALE id.

Usage:
    uv run python scripts/check_pinnacle_leagues.py            # all id-configured sectors
    uv run python scripts/check_pinnacle_leagues.py -s nfl,nba
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
TOP_N = 15  # leagues shown per stale id (sorted by matchupCount)
sys.path.insert(0, str(_REPO_ROOT))

from evmax.clients.esports_pinnacle import (  # noqa: E402
    PinnacleGuestClient,
    SECTOR_SPORT_LEAGUES,
    classify_pinnacle_error,
)


def classify_league_ids(configured: list[int], listed: dict[int, str]) -> dict[str, list]:
    """Split configured ids into OK / STALE against the served league map.

    `listed` maps league id → league name as the guest API returns it.
    Returns {"ok": [(id, name)], "stale": [id]} preserving configured order.
    """
    ok: list[tuple[int, str]] = []
    stale: list[int] = []
    for lid in configured:
        if lid in listed:
            ok.append((lid, listed[lid]))
        else:
            stale.append(lid)
    return {"ok": ok, "stale": stale}


async def fetch_served_leagues(client: PinnacleGuestClient, sport_id: int) -> dict[int, dict]:
    data = await client._get(f"/sports/{sport_id}/leagues")
    out: dict[int, dict] = {}
    for lg in data if isinstance(data, list) else []:
        try:
            out[int(lg["id"])] = lg
        except (KeyError, TypeError, ValueError):
            continue
    return out


async def run(sectors: Optional[list[str]]) -> int:
    targets = {
        s: cfg for s, cfg in SECTOR_SPORT_LEAGUES.items()
        if cfg[1] and (sectors is None or s in sectors)
    }
    if not targets:
        print("no id-configured sectors selected", file=sys.stderr)
        return 2
    exit_code = 0
    served: dict[int, dict[int, dict]] = {}
    async with PinnacleGuestClient() as client:
        for sport_id in sorted({cfg[0] for cfg in targets.values()}):
            try:
                served[sport_id] = await fetch_served_leagues(client, sport_id)
            except Exception as e:  # noqa: BLE001 — report, keep going
                status, reason = classify_pinnacle_error(e)
                print(f"sport {sport_id}: FETCH FAILED ({status} {reason}) — cannot verify", file=sys.stderr)
                served[sport_id] = {}
                exit_code = max(exit_code, 2)

    for sector, (sport_id, league_ids) in sorted(targets.items()):
        listed = {lid: lg.get("name", "?") for lid, lg in served.get(sport_id, {}).items()}
        if not listed:
            print(f"{sector:10} sport {sport_id:3}  UNVERIFIED (no league list)")
            continue
        verdict = classify_league_ids(league_ids, listed)
        for lid, name in verdict["ok"]:
            n = served[sport_id][lid].get("matchupCount")
            print(f"{sector:10} sport {sport_id:3}  OK     {lid:5} {name} (matchups={n})")
        for lid in verdict["stale"]:
            exit_code = max(exit_code, 1)
            ranked = sorted(served[sport_id].items(), key=lambda kv: -int(kv[1].get("matchupCount") or 0))
            print(f"{sector:10} sport {sport_id:3}  STALE  {lid:5} — not served; top {min(TOP_N, len(ranked))} "
                  f"of {len(ranked)} leagues now listed for sport {sport_id}:")
            for oid, lg in ranked[:TOP_N]:
                print(f"{'':10} {'':11}         {oid:6} {lg.get('name')} (matchups={lg.get('matchupCount')})")
    return exit_code


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-s", "--sectors", default=None, help="Comma-separated sector keys (default: all id-configured)")
    args = ap.parse_args(argv)
    sectors = [s.strip().lower() for s in args.sectors.split(",")] if args.sectors else None
    return asyncio.run(run(sectors))


if __name__ == "__main__":
    raise SystemExit(main())
