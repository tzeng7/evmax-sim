"""Code provenance — identifies which code state produced a scan row.

Scans run from whatever branch/commit the checkout happens to be on, and
between 2026-06-01 and 2026-06-10 that silently mixed three different code
states into the live predictions.db. ``code_version()`` stamps each row's
`model_version` column with the short git SHA, a ``-dirty`` suffix when the
working tree has uncommitted changes, and an ``@branch`` suffix when HEAD is
not on main — so validation queries can segment or exclude rows by the exact
code that wrote them.

Examples: ``5901aa1`` (clean main), ``5901aa1-dirty`` (main, uncommitted
changes), ``80d491b-dirty@fix/baseball-alt-runline-magnitude-cap``.

Returns None (column stays NULL) when git is unavailable or the package is
not running from a git checkout — provenance must never break a scan.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Optional

# evmax is installed editable (pip install -e), so the package directory's
# parent is the repo root the scan process is actually running from.
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _git(repo_root: Path, *args: str) -> Optional[str]:
    """Run a git command in repo_root; return stripped stdout or None on failure."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


@lru_cache(maxsize=None)
def code_version(repo_root: Optional[str] = None) -> Optional[str]:
    """Short git SHA + '-dirty' if the tree is dirty + '@branch' if not main.

    Cached per process — a scan invocation is a single process, so the code
    state cannot change mid-scan. Pass an explicit repo_root only in tests.
    """
    root = Path(repo_root) if repo_root else _REPO_ROOT

    sha = _git(root, "rev-parse", "--short", "HEAD")
    if not sha:
        return None
    version = sha

    status = _git(root, "status", "--porcelain")
    if status:  # non-empty porcelain output = uncommitted changes
        version += "-dirty"

    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    # "HEAD" means detached — the SHA already identifies the state exactly.
    if branch and branch not in ("main", "HEAD"):
        version += f"@{branch}"

    return version
