"""Daily NBA player stats cache — sample-size + minutes diagnostics only.

Refreshes player game logs from stats.nba.com (or ESPN fallback) into
data/nba_props_cache.json once per day. The production scan reads only
sample-size and minutes-volatility metadata via :func:`compute_prop_diagnostics`;
prop probabilities come from the Pinnacle-anchored pricing module
:mod:`evmax.ev.prop_pricing`, not from this cache.

Historical context: this module used to project a recency-weighted mean +
opponent adjustment + isotonic calibration into a P(over) — that L15 model
was demoted to shadow on 2026-05-01 after −223u over 575 bets, then ripped
out entirely on 2026-05-10 once anchor pricing replaced it. Cache machinery
stayed because :func:`lookup_player_team` (used by the injury boost) and the
diagnostic columns still need it.

Usage:
    # Daily refresh (run once, e.g. morning cron or first scan of the day):
    await refresh_props_cache()

    # During scan (instant):
    diag = compute_prop_diagnostics("lebron_james")
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "nba_props_cache.json"
_CACHE_TTL = 8 * 3600  # 8 hours — refresh once per day is fine

# Diagnostic params — kept lean
_LAST_N_GAMES = 15
_MIN_GAMES = 5
_MIN_VOL_THRESHOLD = 1.5  # flag games with minutes > mean ± 1.5σ


@dataclass
class PropDiagnostics:
    """Sample-size + minutes-volatility diagnostics from the daily L15 cache.

    No probability math — anchor-based pricing in :mod:`evmax.ev.prop_pricing`
    owns the prob estimate. These columns ride along on each :class:`SharpOdds`
    so downstream filters can gate on data quality (thin samples, erratic
    minutes) without re-deriving them.
    """
    n_games: int
    minutes_volatile: bool = False
    minutes_cv: float = 0.0
    avg_minutes: float = 0.0


# In-memory cache (loaded from disk on first access)
_mem_cache: dict | None = None
_mem_cache_time: float = 0


def _load_cache() -> dict | None:
    """Load cache from disk into memory."""
    global _mem_cache, _mem_cache_time
    if _mem_cache is not None and (time.monotonic() - _mem_cache_time) < _CACHE_TTL:
        return _mem_cache
    try:
        if _CACHE_PATH.exists():
            data = json.loads(_CACHE_PATH.read_text())
            age = time.time() - data.get("fetched_at", 0)
            if age < _CACHE_TTL:
                _mem_cache = data
                _mem_cache_time = time.monotonic()
                return data
    except Exception as e:
        logger.debug("nba_props_cache_load_failed", error=str(e))
    return None


def _load_cache_any_age() -> dict | None:
    """Load cache regardless of TTL — used for fallback when refresh fails."""
    try:
        if _CACHE_PATH.exists():
            return json.loads(_CACHE_PATH.read_text())
    except Exception as e:
        logger.debug("nba_props_cache_load_any_failed", error=str(e))
    return None


def _save_cache(data: dict) -> None:
    """Write cache to disk."""
    global _mem_cache, _mem_cache_time
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(data))
        _mem_cache = data
        _mem_cache_time = time.monotonic()
    except Exception as e:
        logger.warning("nba_props_cache_save_failed", error=str(e))


async def refresh_props_cache(
    force: bool = False,
    player_names: list[str] | None = None,
) -> int:
    """Fetch L15 game logs for NBA players + team stats. Save to disk.

    If player_names is provided, only fetch those players (used during scan
    to cache only players with active Kalshi prop markets). Otherwise fetches
    all players from the existing cache or skips if fresh.

    Only hits stats.nba.com if cache is stale (>8h) or force=True.
    Returns number of players cached.
    """
    if not force:
        existing = _load_cache()
        if existing is not None:
            n = len(existing.get("players", {}))
            logger.info("nba_props_cache_fresh", players=n,
                        age_h=round((time.time() - existing.get("fetched_at", 0)) / 3600, 1))
            return n

    logger.info("nba_props_cache_refreshing",
                targeted=len(player_names) if player_names else 0)

    # Fetch player game logs (targeted or all)
    players_data = await _fetch_player_stats_async(player_names)

    # Merge with existing cache if doing a targeted refresh
    existing = _load_cache()
    if existing and player_names:
        merged = existing.get("players", {})
        merged.update(players_data)
        players_data = merged

    # Cache-safety: if the fetch produced zero players (network blocked,
    # ESPN+nba_api both down) AND we have an existing non-empty cache,
    # keep the prior cache instead of wiping it. Prevents a death-spiral
    # where one bad refresh clears the cache and every subsequent prop
    # scan returns empty.
    if not players_data:
        prior = existing or _load_cache_any_age()
        prior_players = (prior or {}).get("players") or {}
        if prior_players:
            logger.warning(
                "nba_props_cache_refresh_empty_keeping_prior",
                prior_count=len(prior_players),
            )
            return len(prior_players)

    from datetime import date
    today = date.today().isoformat()

    # Cache only stores per-player game logs now — opponent stats and the
    # daily schedule were inputs to the deleted L15 model. Anchor pricing
    # gets opponent context from Pinnacle's own line; schedule lookups for
    # injury boost go through the live ESPN scoreboard fetch instead.
    cache_data = {
        "fetched_at": time.time(),
        "date": today,
        "players": players_data,
    }
    _save_cache(cache_data)
    logger.info("nba_props_cache_saved", players=len(players_data))
    return len(players_data)


# Semaphore for stats.nba.com (3 concurrent to avoid rate-limiting)
_SEM = asyncio.Semaphore(3)
_SEASON = "2025-26"
_TIMEOUT = 10


async def _fetch_player_stats_async(
    player_names: list[str] | None = None,
) -> dict[str, dict]:
    """Fetch L15 game logs for players. Uses per-player endpoint (reliable).

    If player_names is given, only fetch those. Otherwise fetch all active
    players who have played recently.
    """
    try:
        from nba_api.stats.static import players as nba_players
    except ImportError:
        logger.warning("nba_api_not_installed")
        return {}

    # Resolve player name → ID. nba_api ID is optional now — when missing
    # (or stats.nba.com is in backoff), we fall through to ESPN using the
    # name_key directly. Targets carry pid=0 as the sentinel for "ESPN-only".
    if player_names:
        from evmax.clients.nba_stats import _find_player_id
        targets: list[tuple[str, int]] = []
        for name in set(player_names):
            pid = _find_player_id(name) or 0
            targets.append((name, pid))
    else:
        all_active = [p for p in nba_players.get_active_players()]
        targets = [
            (p["full_name"].lower().replace(" ", "_"), p["id"])
            for p in all_active
        ]

    # Skip nba_api entirely when the circuit breaker is tripped — every call
    # would just burn the per-player timeout (3 retries × 10s) before failing.
    from evmax.agents.models._nba_freshness import nba_api_in_backoff, mark_nba_api_failure
    nba_api_blocked = nba_api_in_backoff()
    if nba_api_blocked:
        logger.info("nba_props_skipping_nba_api_breaker_tripped", targets=len(targets))

    logger.info("nba_props_fetching_players", count=len(targets))

    nba_failures = 0  # track failures to trip the breaker on first sign of trouble

    async def _fetch_one(name_key: str, player_id: int) -> tuple[str, dict | None]:
        nonlocal nba_failures
        loop = asyncio.get_event_loop()

        if not nba_api_blocked and player_id:
            async with _SEM:
                data = await loop.run_in_executor(
                    None, _fetch_single_player_sync, player_id,
                )
            if data is not None:
                return name_key, data
            nba_failures += 1

        # ESPN fallback. No semaphore — ESPN handles concurrent requests fine
        # and we want to recover the cache fast when stats.nba.com is down.
        data = await loop.run_in_executor(None, _fetch_player_via_espn_sync, name_key)
        return name_key, data

    results = await asyncio.gather(
        *(_fetch_one(name, pid) for name, pid in targets),
        return_exceptions=True,
    )

    players: dict[str, dict] = {}
    for r in results:
        if isinstance(r, Exception):
            continue
        name_key, data = r
        if data is not None:
            players[name_key] = data

    # If every nba_api call failed but ESPN saved us, trip the breaker so
    # the next refresh skips nba_api straight away.
    if not nba_api_blocked and nba_failures and nba_failures >= len(targets) // 2:
        mark_nba_api_failure()
        logger.warning("nba_props_nba_api_breaker_tripped", failures=nba_failures, total=len(targets))

    logger.info("nba_props_players_fetched", count=len(players))
    return players


def _fetch_single_player_sync(player_id: int) -> dict | None:
    """Fetch L15 game log for a single player. Returns parsed dict or None."""
    try:
        from nba_api.stats.endpoints import playergamelog

        logs = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=_SEASON,
            season_type_all_star="Regular Season",
            timeout=_TIMEOUT,
        )
        df = logs.get_data_frames()[0]

        if df is None or len(df) < _MIN_GAMES:
            return None

        # Take last N games (already sorted most recent first)
        df = df.head(_LAST_N_GAMES)

        stat_cols = ["PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV", "MIN"]
        stats: dict[str, list[float]] = {}
        for col in stat_cols:
            if col in df.columns:
                stats[col] = [float(v) for v in df[col].values]

        if not stats.get("PTS"):
            return None

        # Extract player name and team from MATCHUP
        player_name = ""
        if "PLAYER_NAME" in df.columns:
            player_name = str(df.iloc[0]["PLAYER_NAME"])

        matchup = str(df.iloc[0].get("MATCHUP", ""))
        team_abbrev = matchup.split(" ")[0].upper() if matchup else ""

        return {
            "player_name": player_name,
            "player_id": int(player_id),
            "team": team_abbrev,
            "n_games": len(df),
            "stats": stats,
        }
    except Exception as e:
        logger.debug("nba_props_player_fetch_failed", player_id=player_id, error=str(e))
        return None


# ESPN fallback path. stats.nba.com routinely blocks scraping during playoffs;
# ESPN's public gamelog API is unauthenticated, fast (~350ms), and exposes the
# same per-game stat columns we need. Used when nba_api returns None.
#
# Stat indices come from the response's `labels` array, fixed across requests:
#   ['MIN','FG','FG%','3PT','3P%','FT','FT%','REB','AST','BLK','STL','PF','TO','PTS']
_ESPN_STAT_IDX = {"MIN": 0, "REB": 7, "AST": 8, "BLK": 9, "STL": 10, "TOV": 12, "PTS": 13}
_ESPN_3PT_IDX = 3  # "made-attempted" string, e.g. "4-9"
_ESPN_SEARCH_URL = "https://site.web.api.espn.com/apis/common/v3/search"
_ESPN_GAMELOG_URL = "https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/{id}/gamelog"
_ESPN_SEASON_YEAR = 2026  # 2025-26 season; bump in offseason

# In-memory cache of name → ESPN athlete ID so repeated scans don't re-search
# (lookup costs ~300ms per player; cache survives the process lifetime).
_espn_id_cache: dict[str, str] = {}


def _resolve_espn_athlete_id(player_name: str) -> str | None:
    """Resolve a name like 'lebron_james' to ESPN athlete id via search."""
    cached = _espn_id_cache.get(player_name)
    if cached is not None:
        return cached or None
    query = player_name.replace("_", " ").strip()
    if not query:
        return None
    try:
        import httpx
        r = httpx.get(
            _ESPN_SEARCH_URL,
            params={"query": query, "limit": 5, "type": "player"},
            timeout=8.0,
        )
        r.raise_for_status()
        for item in (r.json() or {}).get("items", []):
            if item.get("league") == "nba" and item.get("sport") == "basketball":
                aid = str(item.get("id") or "")
                if aid:
                    _espn_id_cache[player_name] = aid
                    return aid
        _espn_id_cache[player_name] = ""  # negative-cache so we don't retry
        return None
    except Exception as e:
        logger.debug("nba_props_espn_search_failed", player=player_name, error=str(e))
        return None


def _parse_espn_stat(stats_row: list, idx: int) -> float:
    """Coerce ESPN stat-cell to float; '-' / '' → 0.0."""
    try:
        v = stats_row[idx]
    except (IndexError, TypeError):
        return 0.0
    if v in (None, "", "-"):
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _parse_espn_made(stats_row: list, idx: int) -> float:
    """Parse a 'made-attempted' cell like '4-9' → 4.0."""
    try:
        cell = str(stats_row[idx])
    except (IndexError, TypeError):
        return 0.0
    if "-" not in cell:
        return 0.0
    try:
        return float(cell.split("-", 1)[0])
    except (TypeError, ValueError):
        return 0.0


def _fetch_player_via_espn_sync(player_name: str) -> dict | None:
    """ESPN fallback. Returns same shape as _fetch_single_player_sync, or None.

    Walks seasonTypes → categories → events in the order ESPN returns them
    (postseason first, then regular season most-recent-month first), collects
    up to _LAST_N_GAMES, parses stats by fixed label index.
    """
    aid = _resolve_espn_athlete_id(player_name)
    if not aid:
        return None
    try:
        import httpx
        r = httpx.get(
            _ESPN_GAMELOG_URL.format(id=aid),
            params={"season": _ESPN_SEASON_YEAR},
            timeout=10.0,
        )
        r.raise_for_status()
        data = r.json() or {}
    except Exception as e:
        logger.debug("nba_props_espn_gamelog_failed", player=player_name, aid=aid, error=str(e))
        return None

    events_meta = data.get("events") or {}
    collected: list[tuple[str, list]] = []  # (event_id, stats_row)
    for st in data.get("seasonTypes") or []:
        for cat in st.get("categories") or []:
            for ev in cat.get("events") or []:
                eid = str(ev.get("eventId") or "")
                row = ev.get("stats") or []
                if eid and row:
                    collected.append((eid, row))

    # ESPN returns most-recent-first within each category, but categories
    # are ordered postseason→regular, recent→old. Sort by event gameDate
    # descending using the events meta map so playoff-and-RS interleave
    # is handled correctly.
    def _date_key(eid: str) -> str:
        return (events_meta.get(eid, {}) or {}).get("gameDate", "")
    collected.sort(key=lambda p: _date_key(p[0]), reverse=True)
    collected = collected[:_LAST_N_GAMES]

    if len(collected) < _MIN_GAMES:
        return None

    stats: dict[str, list[float]] = {k: [] for k in ("PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV", "MIN")}
    for _eid, row in collected:
        stats["PTS"].append(_parse_espn_stat(row, _ESPN_STAT_IDX["PTS"]))
        stats["REB"].append(_parse_espn_stat(row, _ESPN_STAT_IDX["REB"]))
        stats["AST"].append(_parse_espn_stat(row, _ESPN_STAT_IDX["AST"]))
        stats["STL"].append(_parse_espn_stat(row, _ESPN_STAT_IDX["STL"]))
        stats["BLK"].append(_parse_espn_stat(row, _ESPN_STAT_IDX["BLK"]))
        stats["TOV"].append(_parse_espn_stat(row, _ESPN_STAT_IDX["TOV"]))
        stats["MIN"].append(_parse_espn_stat(row, _ESPN_STAT_IDX["MIN"]))
        stats["FG3M"].append(_parse_espn_made(row, _ESPN_3PT_IDX))

    if not stats["PTS"]:
        return None

    # Pull team + display name from the most recent event meta.
    most_recent_eid = collected[0][0]
    meta = events_meta.get(most_recent_eid, {}) or {}
    team_abbrev = ((meta.get("team") or {}).get("abbreviation") or "").upper()
    display_name = player_name.replace("_", " ").title()

    return {
        "player_name": display_name,
        "player_id": 0,  # not an nba_api id; downstream only uses name keys
        "team": team_abbrev,
        "n_games": len(collected),
        "stats": stats,
        "source": "espn",
    }






def compute_prop_diagnostics(player_name: str) -> PropDiagnostics | None:
    """Cheap L15 diagnostic lookup — sample size + minutes volatility only.

    Production scan path. Returns None if the player isn't in cache or has
    fewer than _MIN_GAMES recent games. Otherwise returns a PropDiagnostics
    with n_games / minutes_volatile / minutes_cv / avg_minutes — no
    probability math, no opponent adjustment, no isotonic calibration.

    See PropDiagnostics docstring for the design rationale.
    """
    cache = _load_cache()
    if cache is None:
        return None
    player = (cache.get("players") or {}).get(player_name)
    if player is None:
        return None

    n_games = player.get("n_games", 0)
    if n_games < _MIN_GAMES:
        return None

    stats = player.get("stats", {})
    minutes_list = stats.get("MIN", [])
    if not minutes_list:
        # No minutes data — return what we can.
        return PropDiagnostics(n_games=n_games)

    n = min(len(minutes_list), _LAST_N_GAMES)
    minutes = np.array(minutes_list[:n])
    avg_min = float(np.mean(minutes))
    min_std = float(np.std(minutes)) if n >= 3 else 0.0
    minutes_cv = min_std / avg_min if avg_min > 5.0 else 0.0

    volatile_games = 0
    if avg_min >= 5.0 and min_std > 1.0:
        volatile_mask = np.abs(minutes - avg_min) > (_MIN_VOL_THRESHOLD * min_std)
        volatile_games = int(np.sum(volatile_mask))
    minutes_volatile = volatile_games >= 2 or (volatile_games >= 1 and minutes_cv > 0.25)

    return PropDiagnostics(
        n_games=n,
        minutes_volatile=minutes_volatile,
        minutes_cv=round(minutes_cv, 3),
        avg_minutes=round(avg_min, 1),
    )




def is_cache_fresh() -> bool:
    """Check if the props cache exists and is within TTL."""
    return _load_cache() is not None


# Abbreviation → team nickname used by InjuryReportAgent (keys are lowercase
# full team names like "los angeles lakers", and substring matching reduces
# them via the nickname). Keeping a small local map avoids pulling in the
# sector alias loader just for injury boosts.
_NBA_ABBREV_TO_NICKNAME: dict[str, str] = {
    "atl": "hawks", "bos": "celtics", "bkn": "nets", "cha": "hornets",
    "chi": "bulls", "cle": "cavaliers", "dal": "mavericks", "den": "nuggets",
    "det": "pistons", "gsw": "warriors", "hou": "rockets", "ind": "pacers",
    "lac": "clippers", "lal": "lakers", "mem": "grizzlies", "mia": "heat",
    "mil": "bucks", "min": "timberwolves", "nop": "pelicans", "nyk": "knicks",
    "okc": "thunder", "orl": "magic", "phi": "76ers", "phx": "suns",
    "por": "trail blazers", "sac": "kings", "sas": "spurs", "tor": "raptors",
    "uta": "jazz", "was": "wizards",
}


def lookup_player_team(player_name: str) -> str | None:
    """Return the canonical team nickname for a cached NBA player, or None.

    Used by the prop-injury-boost logic so it can identify the player's team
    directly instead of trying to parse a game slug out of the prop event_id
    (which has no game slug).
    """
    cache = _load_cache()
    if cache is None:
        return None
    player = cache.get("players", {}).get(player_name)
    if not player:
        return None
    abbrev = (player.get("team") or "").lower().strip()
    return _NBA_ABBREV_TO_NICKNAME.get(abbrev)
