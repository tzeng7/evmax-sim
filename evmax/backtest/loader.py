"""Download and cache historical odds CSV/XLSX files."""

from __future__ import annotations

import time
from pathlib import Path

import httpx
import structlog

logger = structlog.get_logger(__name__)

CACHE_DIR = Path("data/backtest")

# football-data.co.uk league codes → human names
SOCCER_LEAGUES = {
    "E0":  "EPL",
    "SP1": "La Liga",
    "D1":  "Bundesliga",
    "I1":  "Serie A",
    "F1":  "Ligue 1",
}

SOCCER_SEASON_URLS: dict[str, dict[str, str]] = {
    "2324": {k: f"https://www.football-data.co.uk/mmz4281/2324/{k}.csv" for k in SOCCER_LEAGUES},
    "2425": {k: f"https://www.football-data.co.uk/mmz4281/2425/{k}.csv" for k in SOCCER_LEAGUES},
    "2526": {k: f"https://www.football-data.co.uk/mmz4281/2526/{k}.csv" for k in SOCCER_LEAGUES},
}

TENNIS_SEASON_URLS: dict[int, str] = {
    2024: "http://www.tennis-data.co.uk/2024/2024.xlsx",
    2025: "http://www.tennis-data.co.uk/2025/2025.xlsx",
    2026: "http://www.tennis-data.co.uk/2026/2026.xlsx",
}

# Current season files may update daily — cache for 6h
CURRENT_SEASON = "2526"
CURRENT_TENNIS_YEAR = 2026
CACHE_TTL_CURRENT = 6 * 3600
CACHE_TTL_HISTORICAL = float("inf")  # historical files never change


def _is_stale(path: Path, ttl: float) -> bool:
    if not path.exists():
        return True
    if ttl == float("inf"):
        return False
    return (time.time() - path.stat().st_mtime) > ttl


def fetch_soccer_csv(season: str, league: str, force: bool = False) -> Path:
    """Download a football-data.co.uk CSV, returning the local cache path."""
    if season not in SOCCER_SEASON_URLS:
        raise ValueError(f"Unknown soccer season: {season!r}. Available: {list(SOCCER_SEASON_URLS)}")
    if league not in SOCCER_LEAGUES:
        raise ValueError(f"Unknown league code: {league!r}. Available: {list(SOCCER_LEAGUES)}")

    url = SOCCER_SEASON_URLS[season][league]
    dest = CACHE_DIR / "soccer" / season / f"{league}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)

    ttl = CACHE_TTL_CURRENT if season == CURRENT_SEASON else CACHE_TTL_HISTORICAL
    if not force and not _is_stale(dest, ttl):
        logger.debug("backtest_cache_hit", file=str(dest))
        return dest

    logger.info("backtest_download", url=url, dest=str(dest))
    resp = httpx.get(url, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def fetch_tennis_xlsx(year: int, force: bool = False) -> Path:
    """Download tennis-data.co.uk XLSX, returning the local cache path."""
    if year not in TENNIS_SEASON_URLS:
        raise ValueError(f"Unknown tennis year: {year}. Available: {list(TENNIS_SEASON_URLS)}")

    url = TENNIS_SEASON_URLS[year]
    dest = CACHE_DIR / "tennis" / f"{year}.xlsx"
    dest.parent.mkdir(parents=True, exist_ok=True)

    ttl = CACHE_TTL_CURRENT if year == CURRENT_TENNIS_YEAR else CACHE_TTL_HISTORICAL
    if not force and not _is_stale(dest, ttl):
        logger.debug("backtest_cache_hit", file=str(dest))
        return dest

    logger.info("backtest_download", url=url, dest=str(dest))
    resp = httpx.get(url, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest
