"""Cache freshness helper shared by the 3 NBA model agents.

The previous logic refreshed state whenever the calendar day rolled over,
forcing 5 nba_api calls on the first scan of each day even when no games
had been played. This helper instead asks ESPN for the most recent
completed NBA game date and treats the agent's state as fresh whenever
`fetched_at >= last_completed_game_date`. On off-days (no games since
last fetch) all three agents skip the network entirely.

Cached process-locally so the three agents share a single ESPN round-trip
within a scan.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import httpx
import structlog

logger = structlog.get_logger(__name__)

_BREAKER_PATH = Path(__file__).resolve().parents[3] / "data" / "models" / ".nba_api_breaker.json"

_ESPN_NBA_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
_LOOKBACK_DAYS = 7
_CACHE_TTL_S = 600  # 10 minutes — long enough for one scan, short enough to follow real game completion

_cache_value: Optional[str] = None
_cache_time: float = 0.0
_lock = asyncio.Lock()

# Circuit breaker for stats.nba.com. When the underlying API is down (or
# rate-limited), nba_api retries internally — 3 retries × 15s timeout
# = 45s per agent. Three agents trying in parallel each consume the
# entire sector budget. Once any agent reports a failure, every NBA model
# agent skips its fetch attempt for FAILURE_BACKOFF_S and uses whatever
# stale state it has on disk.
_FAILURE_BACKOFF_S = 600  # 10 minutes
_last_failure_time: float = 0.0


def _reset_cache_for_tests() -> None:
    """Test hook — clear the in-process cache and circuit breaker."""
    global _cache_value, _cache_time, _last_failure_time
    _cache_value = None
    _cache_time = 0.0
    _last_failure_time = 0.0
    try:
        _BREAKER_PATH.unlink()
    except FileNotFoundError:
        pass


def mark_nba_api_failure() -> None:
    """Trip the circuit breaker. Persisted to disk so subsequent scans see it."""
    global _last_failure_time
    _last_failure_time = time.time()
    try:
        _BREAKER_PATH.parent.mkdir(parents=True, exist_ok=True)
        _BREAKER_PATH.write_text(json.dumps({"failed_at": _last_failure_time}))
    except Exception as e:
        logger.debug("nba_breaker_write_failed", error=str(e))


def _load_persisted_failure_time() -> float:
    """Read last failure timestamp from disk; 0 if missing/unreadable."""
    try:
        return float(json.loads(_BREAKER_PATH.read_text()).get("failed_at", 0))
    except Exception:
        return 0.0


def nba_api_in_backoff() -> bool:
    """True when stats.nba.com recently failed and we should not retry yet.

    Reads disk-persisted failure timestamp on each call so siblings in
    parallel scans / subsequent invocations all observe the same state.
    """
    global _last_failure_time
    persisted = _load_persisted_failure_time()
    if persisted > _last_failure_time:
        _last_failure_time = persisted
    return (time.time() - _last_failure_time) < _FAILURE_BACKOFF_S


async def last_completed_nba_game_date() -> Optional[str]:
    """YYYY-MM-DD of the most recently completed NBA game, or None if unknown.

    Walks back up to 7 days from today via ESPN scoreboard. Returns None
    when no completed games are found in the window (e.g. All-Star break,
    pre-season, or ESPN is unreachable).
    """
    global _cache_value, _cache_time
    async with _lock:
        if _cache_value is not None and (time.time() - _cache_time) < _CACHE_TTL_S:
            return _cache_value or None

        today = date.today()
        async with httpx.AsyncClient(timeout=10.0) as client:
            for offset in range(_LOOKBACK_DAYS):
                d = today - timedelta(days=offset)
                try:
                    resp = await client.get(_ESPN_NBA_SCOREBOARD, params={"dates": d.strftime("%Y%m%d")})
                    resp.raise_for_status()
                    events = resp.json().get("events", [])
                    if any(e.get("status", {}).get("type", {}).get("completed") for e in events):
                        _cache_value = d.isoformat()
                        _cache_time = time.time()
                        return _cache_value
                except Exception as e:
                    logger.debug("nba_freshness_probe_failed", date=d.isoformat(), error=str(e))
                    continue

        _cache_value = ""  # cache the negative result so we don't retry on every agent
        _cache_time = time.time()
        return None


async def state_is_fresh(fetched_at: str) -> bool:
    """True when state was fetched on/after the most recent completed NBA game.

    Falls back to True when ESPN can't be reached and `fetched_at` looks
    valid — better to use slightly stale stats than to bust the sector
    timeout on a transient network blip.
    """
    if not fetched_at:
        return False
    last = await last_completed_nba_game_date()
    if last is None:
        return True
    return fetched_at >= last
