"""Disk-backed JSON cache for API responses.

Usage in dev: set EVMAX_CACHE_TTL_SECS=3600 in .env to reuse responses for 1 hour.
Set EVMAX_OFFLINE=true to never hit live APIs (raises CacheOfflineError if no cached data).

Cache files are stored in data/cache/<key>.json and are safe to delete at any time.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_CACHE_DIR = Path("data/cache")


class CacheOfflineError(RuntimeError):
    """Raised when offline_mode=True and no cached data exists for the key."""


def _cache_path(key: str) -> Path:
    # Sanitize key to a safe filename
    safe = key.replace("/", "_").replace(":", "_").replace(" ", "_")
    return _CACHE_DIR / f"{safe}.json"


def cache_get(key: str, ttl_secs: int) -> Any | None:
    """Return cached value if it exists and is not expired. Returns None otherwise."""
    if ttl_secs <= 0:
        return None
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        age = time.time() - data["ts"]
        if age > ttl_secs:
            logger.debug("cache_expired", key=key, age_s=round(age))
            return None
        logger.debug("cache_hit", key=key, age_s=round(age))
        return data["payload"]
    except Exception as exc:
        logger.warning("cache_read_error", key=key, error=str(exc))
        return None


def cache_set(key: str, payload: Any) -> None:
    """Write payload to disk cache."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(key)
    try:
        path.write_text(json.dumps({"ts": time.time(), "payload": payload}))
        logger.debug("cache_write", key=key)
    except Exception as exc:
        logger.warning("cache_write_error", key=key, error=str(exc))


def cache_get_offline(key: str) -> Any:
    """Return any cached value regardless of TTL. Raises CacheOfflineError if missing."""
    path = _cache_path(key)
    if not path.exists():
        raise CacheOfflineError(
            f"offline_mode=True but no cache exists for key={key!r}. "
            "Run once with EVMAX_OFFLINE=false to populate the cache."
        )
    try:
        data = json.loads(path.read_text())
        logger.debug("cache_offline_hit", key=key)
        return data["payload"]
    except Exception as exc:
        raise CacheOfflineError(f"Failed to read cache for key={key!r}: {exc}") from exc
