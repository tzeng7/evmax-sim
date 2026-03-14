"""Barttorvik T-Rank data fetcher.

Fetches current-season adjusted efficiency metrics (AdjOE, AdjDE, AdjT)
from barttorvik.com. Results are cached in-process for 4 hours.

URL: https://barttorvik.com/trank.php?year=YEAR&top=400&csv=1

CSV columns of interest (names may vary slightly by year):
  Team, AdjOE, AdjDE, AdjT (tempo)
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import httpx
import structlog

logger = structlog.get_logger(__name__)

_BARTTORVIK_URL = "https://barttorvik.com/trank.php"
_CACHE_TTL = timedelta(hours=4)

# Module-level cache
_cache: dict[str, "TeamEfficiency"] = {}
_cache_fetched_at: Optional[datetime] = None


@dataclass
class TeamEfficiency:
    """Barttorvik adjusted efficiency metrics for one team."""

    team: str        # normalized team name
    adj_oe: float    # Adjusted Offensive Efficiency (pts per 100 possessions vs avg D)
    adj_de: float    # Adjusted Defensive Efficiency (pts allowed per 100 poss vs avg O)
    adj_t: float     # Adjusted Tempo (possessions per 40-min game)


async def fetch_team_efficiencies(year: Optional[int] = None) -> dict[str, TeamEfficiency]:
    """
    Fetch current NCAAB team efficiency ratings from Barttorvik.

    Returns a dict keyed by normalized team name (lowercase, alias-resolved).
    Cached for 4 hours. Returns empty dict on fetch failure, preserving last
    good cache if available.

    Args:
        year: Season ending year (e.g. 2026 for 2025-26 season).
              Defaults to current calendar year.
    """
    global _cache, _cache_fetched_at

    now = datetime.utcnow()
    if _cache and _cache_fetched_at and (now - _cache_fetched_at) < _CACHE_TTL:
        return _cache

    if year is None:
        year = now.year

    url_params = {"year": str(year), "top": "400", "csv": "1"}

    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            resp = await client.get(_BARTTORVIK_URL, params=url_params)
            resp.raise_for_status()
            text = resp.text

        parsed = _parse_csv(text)

        if parsed:
            _cache = parsed
            _cache_fetched_at = now
            logger.info("barttorvik_fetched", teams=len(parsed), year=year)
        else:
            logger.warning("barttorvik_empty_response", year=year)

    except Exception as e:
        logger.warning("barttorvik_fetch_failed", error=str(e), year=year)
        # Return stale cache rather than empty if available
        if _cache:
            logger.info("barttorvik_using_stale_cache", teams=len(_cache))

    return _cache


def get_cached() -> dict[str, TeamEfficiency]:
    """Return current in-process cache without triggering a fetch."""
    return _cache


def _parse_csv(text: str) -> dict[str, TeamEfficiency]:
    """
    Parse Barttorvik trank CSV response into a team efficiency dict.

    Handles column name variations across years. Falls back to positional
    indices if named headers don't match:
      col[1]  = Team
      col[5]  = AdjOE
      col[6]  = AdjDE
      col[20] = AdjT  (tempo, last column)
    """
    from evmax.matching.normalizer import NameNormalizer

    normalizer = NameNormalizer("ncaab")
    result: dict[str, TeamEfficiency] = {}

    # Possible header name variants
    _OE_KEYS = ("adjoe", "adj oe", "adj. oe", "offadjem", "oadjeff", "aoe")
    _DE_KEYS = ("adjde", "adj de", "adj. de", "defadjem", "dadjeff", "ade")
    _T_KEYS  = ("adjt", "adj t", "adj. t", "adjtempo", "tempo", "pace", "at")
    _TEAM_KEYS = ("team", "school", "name")

    try:
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
    except Exception as e:
        logger.warning("barttorvik_csv_parse_error", error=str(e))
        return result

    if not rows:
        return result

    # Find header row (first row with 5+ string fields)
    header_idx = 0
    header: list[str] = []
    for i, row in enumerate(rows):
        if len(row) >= 5 and not row[0].lstrip("-").isdigit():
            header = [h.strip().lower() for h in row]
            header_idx = i
            break

    if not header:
        logger.warning("barttorvik_no_header_found")
        return result

    # Map column names to indices
    def find_col(keys: tuple[str, ...]) -> Optional[int]:
        for key in keys:
            for i, h in enumerate(header):
                if h == key:
                    return i
        return None

    team_col = find_col(_TEAM_KEYS)
    oe_col   = find_col(_OE_KEYS)
    de_col   = find_col(_DE_KEYS)
    t_col    = find_col(_T_KEYS)

    # Fallback to known positional indices if header matching failed
    if team_col is None: team_col = 1
    if oe_col is None:   oe_col = 5
    if de_col is None:   de_col = 6
    if t_col is None:    t_col = len(header) - 1  # AdjT is typically last

    logger.debug(
        "barttorvik_columns",
        team_col=team_col, oe_col=oe_col, de_col=de_col, t_col=t_col,
    )

    for row in rows[header_idx + 1:]:
        if len(row) <= max(team_col, oe_col, de_col, t_col):
            continue

        try:
            team_raw = row[team_col].strip()
            if not team_raw:
                continue

            adj_oe = float(row[oe_col].strip() or 0)
            adj_de = float(row[de_col].strip() or 0)
            adj_t  = float(row[t_col].strip() or 0)

            if adj_oe <= 0 or adj_de <= 0 or adj_t <= 0:
                continue

            team_norm = normalizer.normalize(team_raw)
            if not team_norm:
                continue

            result[team_norm] = TeamEfficiency(
                team=team_norm,
                adj_oe=adj_oe,
                adj_de=adj_de,
                adj_t=adj_t,
            )

        except (ValueError, IndexError):
            continue

    return result
