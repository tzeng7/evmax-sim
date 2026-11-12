"""Verify every Pinnacle guest league id in SECTOR_SPORT_LEAGUES is still served.

Companion to scripts/check_kalshi_series.py for season-start checklist item 4
(docs/SEASON_START.md §2). Pinnacle re-cuts league ids between seasons — the
NFL went 258 → 889 for 2026 — and a stale id silently returns ZERO matchups,
i.e. no sharp anchor and no EV for the whole sector. Nothing else in the
pipeline can tell that apart from an off-season.

For each id-configured sector this hits `GET /sports/{sport_id}/leagues` once
and reports each configured id as OK (listed), IDLE or STALE, and lists
the sport's currently-served leagues whenever an id is stale so the operator
can read the replacement straight off the output. Exit 1 on any STALE id.

The default league list carries only leagues with open matchups. An id absent
from it is re-checked against `?all=true` (the full catalog): present there
means IDLE — the id is valid and Pinnacle has posted no games (WNCAA 583 in
early November 2026) — and only an id absent from the catalog is STALE. IDLE
does not change the exit code.

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


def classify_league_ids(
    configured: list[int],
    listed: dict[int, str],
    catalog: Optional[dict[int, str]] = None,
) -> dict[str, list]:
    """Split configured ids into OK / STALE against the served league map.

    `listed` maps league id → league name as the guest API returns it.
    Returns {"ok": [(id, name)], "stale": [id]} preserving configured order.

    `catalog` is the full league catalog (`?all=true`). When given, an id that
    is not listed but is in the catalog is reported under "idle" (valid id, no
    open matchups) instead of "stale", and the result carries an "idle" key.
    Without a catalog the result keeps the two-key shape.
    """
    ok: list[tuple[int, str]] = []
    stale: list[int] = []
    idle: list[tuple[int, str]] = []
    for lid in configured:
        if lid in listed:
            ok.append((lid, listed[lid]))
        elif catalog is not None and lid in catalog:
            idle.append((lid, catalog[lid]))
        else:
            stale.append(lid)
    out: dict[str, list] = {"ok": ok, "stale": stale}
    if catalog is not None:
        out["idle"] = idle
    return out


async def fetch_served_leagues(
    client: PinnacleGuestClient, sport_id: int, *, all_leagues: bool = False
) -> dict[int, dict]:
    suffix = "?all=true" if all_leagues else ""
    data = await client._get(f"/sports/{sport_id}/leagues{suffix}")
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
    catalogs: dict[int, dict[int, dict]] = {}
    async with PinnacleGuestClient() as client:
        for sport_id in sorted({cfg[0] for cfg in targets.values()}):
            try:
                served[sport_id] = await fetch_served_leagues(client, sport_id)
            except Exception as e:  # noqa: BLE001 — report, keep going
                status, reason = classify_pinnacle_error(e)
                print(f"sport {sport_id}: FETCH FAILED ({status} {reason}) — cannot verify", file=sys.stderr)
                served[sport_id] = {}
                exit_code = max(exit_code, 2)
                continue
            # Full catalog only when a configured id is missing from the
            # active list. A catalog fetch failure leaves the id STALE.
            wanted = {lid for sid, ids in targets.values() if sid == sport_id for lid in ids}
            if wanted - set(served[sport_id]):
                try:
                    catalogs[sport_id] = await fetch_served_leagues(client, sport_id, all_leagues=True)
                except Exception:  # noqa: BLE001 — fall back to STALE
                    pass

    for sector, (sport_id, league_ids) in sorted(targets.items()):
        listed = {lid: lg.get("name", "?") for lid, lg in served.get(sport_id, {}).items()}
        if not listed:
            print(f"{sector:10} sport {sport_id:3}  UNVERIFIED (no league list)")
            continue
        catalog = (
            {lid: lg.get("name", "?") for lid, lg in catalogs[sport_id].items()}
            if sport_id in catalogs else None
        )
        verdict = classify_league_ids(league_ids, listed, catalog)
        for lid, name in verdict["ok"]:
            n = served[sport_id][lid].get("matchupCount")
            print(f"{sector:10} sport {sport_id:3}  OK     {lid:5} {name} (matchups={n})")
        for lid, name in verdict.get("idle", []):
            print(f"{sector:10} sport {sport_id:3}  IDLE   {lid:5} {name} (id valid; Pinnacle has no open matchups)")
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
