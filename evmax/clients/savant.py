"""Baseball Savant expected-statistics client — pitcher xERA.

Fetches the public CSV export of Savant's expected_statistics leaderboard
(one request per season per refresh) and returns per-pitcher expected ERA
(xERA) keyed by MLB player_id — the same stable id the MLB Stats API
probable-starter feed carries, so the join needs no name matching (an
ascii-folded name key is included as a fallback).

xERA is park- and defense-neutral by construction (built from exit
velocity / launch angle quality-of-contact), which is exactly what makes
it a better starter-quality input than results-ERA: it strips the
sequencing and BABIP noise FIP only partially removes.

TOS posture: a single daily fetch of a documented public CSV export with
an identifying User-Agent, cached to disk (24h TTL for the current season,
permanent for past seasons), exponential backoff, and degrade-to-{} on any
failure so the pitcher model falls back to FIP/ERA. No scraping loops.

Consumed at SEED time by scripts/seed_pitcher_fip.py, which attaches
``xera`` / ``xera_prior`` to each pitcher's state record — the scan path
only ever reads state and never calls Savant.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import time
import unicodedata
from datetime import date
from pathlib import Path
from typing import Optional

import httpx
import structlog

logger = structlog.get_logger(__name__)

_BASE_URL = "https://baseballsavant.mlb.com/leaderboard/expected_statistics"
CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "savant_xera_cache.json"
CACHE_TTL_CURRENT_S = 24 * 3600  # current season refreshes daily
_RETRIES = 3
_TIMEOUT_S = 30.0


def _ascii_fold(name: str) -> str:
    return (
        unicodedata.normalize("NFKD", name)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
        .strip()
    )


def _load_cache() -> dict:
    try:
        return json.loads(CACHE_PATH.read_text())
    except (OSError, ValueError):
        return {}


def _save_cache(cache: dict) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(cache))
    except OSError:  # cache write failure is never fatal
        logger.warning("savant_cache_write_failed", path=str(CACHE_PATH))


def parse_expected_stats_csv(text: str) -> dict[int, dict]:
    """Parse the Savant expected_statistics CSV into {player_id: record}.

    Records carry name / name_key (ascii-folded "first last") / pa / bip /
    era / xera. Column naming is defensive: xERA has appeared as both
    ``xera`` and ``est_era`` across export revisions.

    The export starts with a UTF-8 BOM that must be stripped from the TEXT
    before csv parsing: attached to the opening quote of the first header
    (``\\ufeff"last_name, first_name"``) it breaks quote detection, the
    header splits on its inner comma, and every column shifts by one.
    """
    out: dict[int, dict] = {}
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    for raw in reader:
        row = {(k or "").strip(): (v or "").strip() for k, v in raw.items()}
        try:
            player_id = int(row.get("player_id", ""))
        except (TypeError, ValueError):
            continue

        def _f(*keys: str) -> Optional[float]:
            for k in keys:
                v = row.get(k)
                if v not in (None, ""):
                    try:
                        return float(v)
                    except ValueError:
                        continue
            return None

        xera = _f("xera", "est_era")
        if xera is None or xera <= 0:
            continue

        # Name arrives as "Last, First" in a single column.
        raw_name = row.get("last_name, first_name") or row.get("player_name") or ""
        if "," in raw_name:
            last, _, first = raw_name.partition(",")
            display = f"{first.strip()} {last.strip()}".strip()
        else:
            display = raw_name.strip()

        out[player_id] = {
            "name": display,
            "name_key": _ascii_fold(display),
            "pa": int(_f("pa") or 0),
            "bip": int(_f("bip") or 0),
            "era": _f("era"),
            "xera": xera,
        }
    return out


async def fetch_expected_stats(
    year: int, min_bip: int = 25, *, force: bool = False
) -> dict[int, dict]:
    """Pitcher expected stats for a season, keyed by MLB player_id.

    Disk-cached: permanent for past seasons, 24h TTL for the current one.
    Returns {} on any fetch/parse failure (callers degrade to FIP/ERA).
    """
    cache = _load_cache()
    key = str(year)
    entry = cache.get(key)
    is_past_season = year < date.today().year
    if entry and not force:
        fresh = is_past_season or (
            time.time() - entry.get("fetched_at", 0) < CACHE_TTL_CURRENT_S
        )
        if fresh:
            return {int(pid): rec for pid, rec in entry.get("players", {}).items()}

    params = {
        "type": "pitcher",
        "year": str(year),
        "position": "",
        "team": "",
        "min": str(min_bip),
        "csv": "true",
    }
    last_err: Optional[Exception] = None
    for attempt in range(_RETRIES):
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(_TIMEOUT_S),
                headers={"User-Agent": "evmax-seeder/1.0"},
                follow_redirects=True,
            ) as client:
                resp = await client.get(_BASE_URL, params=params)
                resp.raise_for_status()
                players = parse_expected_stats_csv(resp.text)
            if not players:
                raise ValueError("savant CSV parsed to 0 pitchers")
            cache[key] = {
                "fetched_at": time.time(),
                "players": {str(pid): rec for pid, rec in players.items()},
            }
            _save_cache(cache)
            logger.info("savant_xera_fetched", year=year, pitchers=len(players))
            return players
        except Exception as e:  # noqa: BLE001 — degrade, never raise to callers
            last_err = e
            await asyncio.sleep(2.0 * (attempt + 1))

    logger.warning("savant_xera_fetch_failed", year=year, error=str(last_err))
    # Stale cache beats nothing: serve an expired current-season entry if present.
    if entry:
        return {int(pid): rec for pid, rec in entry.get("players", {}).items()}
    return {}
