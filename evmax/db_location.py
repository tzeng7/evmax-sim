"""Database location overrides for predictions.db and archive.db.

A git worktree has no ``data/*.db``. Run from a worktree, the CLI used to
create an EMPTY ``archive.db`` on first open, so read-side commands silently
saw zero rows (``--max-staleness-h`` excluded 100% of rows on 2026-10-07).
Two environment variables point a checkout at another checkout's data instead:

``EVMAX_DB_DIR``
    Directory that holds ``predictions.db`` and ``archive.db``. Default: the
    checkout's own ``data/``. Resolved once, at import of the owning module.

``EVMAX_DB_READONLY``
    ``1`` / ``true`` / ``yes`` / ``on`` → every connection from
    ``cleanup.db.get_connection``, ``archiver._get_connection`` and the
    portfolio store opens with a ``mode=ro`` URI and SKIPS schema creation and
    migrations. Any write raises ``sqlite3.OperationalError``; a missing file
    raises ``FileNotFoundError`` instead of creating an empty database.

Typical use from a worktree (the opportunity-scout backtester does this)::

    EVMAX_DB_DIR=/path/to/main/checkout/data EVMAX_DB_READONLY=1 \\
        uv run evmax cleanup shadow clv nfl -m spread

Modules that open their own literal paths (``web/app.py``, the calibration and
meta-model trainers) are not affected.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

ENV_DB_DIR = "EVMAX_DB_DIR"
ENV_DB_READONLY = "EVMAX_DB_READONLY"
_TRUTHY = {"1", "true", "yes", "on"}


def resolve_db_path(filename: str, default: Path) -> Path:
    """Return ``$EVMAX_DB_DIR/filename`` when the variable is set, else ``default``."""
    db_dir = os.environ.get(ENV_DB_DIR, "").strip()
    if not db_dir:
        return default
    return Path(db_dir).expanduser() / filename


def readonly_enabled() -> bool:
    """True when ``EVMAX_DB_READONLY`` holds a truthy value. Read on every call."""
    return os.environ.get(ENV_DB_READONLY, "").strip().lower() in _TRUTHY


def connect_readonly(path: Path, timeout: float = 5.0) -> sqlite3.Connection:
    """Open ``path`` read-only. Never creates the file and never runs DDL."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{ENV_DB_READONLY} is set but {path} does not exist; point "
            f"{ENV_DB_DIR} at the main checkout's data/ directory"
        )
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=timeout)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    return conn
