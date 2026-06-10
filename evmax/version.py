"""Code provenance for persisted predictions.

``code_version()`` returns a short identifier of the code state that
produced a scan row — git short SHA, plus the branch name when not on
``main`` and a ``-dirty`` suffix when the working tree has uncommitted
changes. Example values::

    "5901aa1"                     # clean main
    "5901aa1-dirty"               # main with local edits
    "3f2c9ab@fix/runline-cap"     # clean feature branch
    "3f2c9ab@fix/runline-cap-dirty"

Written into the ``model_version`` column of ``ev_predictions`` /
``prop_observations`` so mixed-code periods are identifiable from the DB
instead of having to be reconstructed from ``git reflog`` (which is how
the 2026-06-01 → 06-10 baseball contamination had to be diagnosed).

Resolved once per process and cached — scans are short-lived processes,
and a mid-process git change would be exactly the confusion this exists
to prevent.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(_REPO_ROOT), *args],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    ).stdout.strip()


@lru_cache(maxsize=1)
def code_version() -> str:
    """Short git SHA + branch (if not main) + dirty flag, or 'unknown'.

    Never raises — a missing git binary or a non-repo install (e.g. a
    bare pip install) yields ``"unknown"`` rather than breaking a scan.
    """
    try:
        sha = _git("rev-parse", "--short", "HEAD")
    except Exception:
        return "unknown"
    version = sha
    try:
        branch = _git("rev-parse", "--abbrev-ref", "HEAD")
        if branch not in ("main", "HEAD"):
            version = f"{sha}@{branch}"
    except Exception:
        pass
    try:
        if _git("status", "--porcelain"):
            version = f"{version}-dirty"
    except Exception:
        pass
    return version
