"""Helpers for blending Regular Season + Playoffs NBA team stats.

The nba_api `LeagueDashTeamStats` endpoint accepts a `season_type_all_star`
parameter that defaults to "Regular Season" and silently ignores playoff
games. After mid-April we want both buckets folded together, weighted by
games played so a team's blended ORTG/DRTG/Pace is the per-game average
across all 2025-26 games (regular + post).

If the Playoffs request fails (off-season, network error) we fall back
to the Regular Season payload alone so callers never break.
"""

from __future__ import annotations

import asyncio
from typing import Callable, Iterable, Optional


def parse_team_dash_rows(data: Optional[dict]) -> dict[str, dict]:
    """Parse a LeagueDashTeamStats payload into {team_key: row_dict}.

    Returns an empty dict if the payload is missing or malformed (e.g. a
    Playoffs query before the postseason starts).
    """
    if not data:
        return {}
    try:
        result_set = data["resultSets"][0]
        headers = result_set["headers"]
        rows = result_set["rowSet"]
    except (KeyError, IndexError, TypeError):
        return {}

    out: dict[str, dict] = {}
    for row in rows:
        d = dict(zip(headers, row))
        name = (d.get("TEAM_NAME") or "").lower()
        if not name:
            continue
        key = name.rsplit(" ", 1)[-1] if " " in name else name
        if key == "blazers":
            key = "trail blazers"
        elif key == "clippers":
            key = "la clippers"
        out[key] = d
    return out


def gp_weighted_blend(
    rs_rows: dict[str, dict],
    po_rows: dict[str, dict],
    fields: Iterable[str],
) -> dict[str, dict]:
    """GP-weighted per-team blend of regular-season and playoff rows.

    For each team present in either input, returns a dict containing the
    blended numeric `fields` plus `gp` (total), `rs_gp`, `po_gp`, and the
    team identity columns (TEAM_NAME, TEAM_ID). Teams with zero combined
    games are dropped.
    """
    fields = list(fields)
    blended: dict[str, dict] = {}
    all_keys = set(rs_rows) | set(po_rows)
    for key in all_keys:
        rs = rs_rows.get(key) or {}
        po = po_rows.get(key) or {}
        rs_gp = rs.get("GP") or 0
        po_gp = po.get("GP") or 0
        total_gp = rs_gp + po_gp
        if total_gp == 0:
            continue
        out: dict = {"gp": total_gp, "rs_gp": rs_gp, "po_gp": po_gp}
        for f in fields:
            rs_v = rs.get(f) if rs else None
            po_v = po.get(f) if po else None
            if rs_gp == 0 or rs_v is None:
                out[f] = po_v if po_v is not None else 0
            elif po_gp == 0 or po_v is None:
                out[f] = rs_v
            else:
                out[f] = (rs_v * rs_gp + po_v * po_gp) / total_gp
        out["TEAM_NAME"] = rs.get("TEAM_NAME") or po.get("TEAM_NAME") or ""
        out["TEAM_ID"] = rs.get("TEAM_ID") or po.get("TEAM_ID")
        blended[key] = out
    return blended


async def fetch_rs_and_po(factory: Callable[[str], object]) -> tuple[Optional[dict], Optional[dict]]:
    """Fetch the same nba_api endpoint for Regular Season and Playoffs concurrently.

    `factory(season_type)` must return an endpoint instance with `.get_dict()`.
    Either side may come back None if its request fails — callers are
    expected to treat None as "no data for that bucket".
    """
    loop = asyncio.get_event_loop()
    rs_future = loop.run_in_executor(None, lambda: factory("Regular Season").get_dict())
    po_future = loop.run_in_executor(None, lambda: factory("Playoffs").get_dict())
    results = await asyncio.gather(rs_future, po_future, return_exceptions=True)
    rs = results[0] if not isinstance(results[0], BaseException) else None
    po = results[1] if not isinstance(results[1], BaseException) else None
    return rs, po
