#!/usr/bin/env python3
"""Isolated-worktree git helper for evmax scheduled tasks.

WHY THIS EXISTS
---------------
Every state/code-altering scheduled task (``weekly-seasonal-model-reseed``,
``weekly-tennis-surface-elo-refresh``, ``daily-resolve-and-model-update``,
``daily-evening-resolve``, ``weekly-model-calibration``, ``weekly-drift-audit``,
``biweekly-model-improve-graph``) used to run against the ONE shared checkout at
``/Users/ktzeng/Projects/evmax`` and branch off *local* ``main``. Two failure
modes followed:

1. **Working-tree collision.** A checkout has one checked-out branch and one
   working tree. When two tasks overlapped, task A's ``git switch main`` /
   ``git pull`` yanked the tree out from under task B mid-run — silently
   discarding uncommitted state (the 2026-07-21 collision, the 2026-08-20
   pre-commit stash-timeout loss).
2. **Un-mergeable PRs.** The owned artifacts (``data/models/*.json``,
   ``data/model_config.json``) are whole-file regenerated JSON. Two PRs that
   both rewrite ``elo_state.json`` cannot three-way-merge, and each branched off
   a potentially-stale *local* main. Concurrent PRs deadlock on
   "require up-to-date" branch protection → reads as "won't pass CI/CD".

THE FIX (this helper)
---------------------
* ``open`` — fetch ``origin``, then create an **isolated git worktree** whose
  branch is cut from the **live remote tip** (``origin/main``), never local
  main. Worktrees share the object store (cheap) but have independent working
  trees and HEADs, so parallel runs can never fight over one checked-out
  branch. This is the same isolation the Claude Code harness already uses under
  ``.claude/worktrees/``.
* ``ship`` — stage ONLY the files the task owns (never ``git add -A``), commit
  (``--no-verify`` by default so a slow pre-commit stash cannot swallow another
  task's tree — the repo's hooks are the advisory doc-sync/test-sync reminders,
  which no-op on a JSON-only commit), push, ensure exactly one open PR, then
  remove the worktree.

ROLLING vs DATED BRANCHES
-------------------------
State tasks pass ``--rolling``: all of them accumulate onto ONE long-lived
branch (default ``bot/model-state``) so there is only ever a single open state
PR. Because each task owns a **disjoint** set of files and the branch is always
re-anchored to ``origin/main``, an already-merged PR leaves the branch clean and
the next run rebuilds from live main; an unmerged PR is preserved and the new
task's owned files are added on top. Regenerated state is idempotent, so on any
rebase trouble the safe fallback is simply to re-anchor to ``origin/main`` and
let the seed script reproduce the current-truth file.

Code tasks (drift-audit, model-improve) omit ``--rolling`` and get a fresh dated
branch off ``origin/main`` each run — their diffs are real code that must pass
review + CI on its own, so they should not share a branch.

USAGE
-----
    # STEP 0 — open an isolated tree on the live remote tip, print its path
    WT=$(python scripts/sched_worktree.py open --rolling \
            --branch bot/model-state --print-path)
    cd "$WT"
    # ... run seed scripts here; they rewrite the owned files in $WT ...

    # STEP N — ship only the owned files as a single rolling PR, then clean up
    python scripts/sched_worktree.py ship --rolling \
        --branch bot/model-state --worktree "$WT" \
        --title "chore(models): daily state refresh" \
        --body  "Automated model-state reseed." \
        -- data/models/elo_state.json data/models/form_state.json

The helper is deliberately fail-soft on the PR step: if ``gh`` fails, the push
already preserved the work and the branch name is reported so the PR can be
opened by hand. It NEVER runs ``git reset --hard`` / ``git checkout -- .`` /
``git clean`` against the shared checkout — all destructive operations happen
only inside the throwaway worktree it created.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_REPO = Path("/Users/ktzeng/Projects/evmax")
DEFAULT_ROLLING_BRANCH = "bot/model-state"
BASE_BRANCH = "main"


def _run(
    args: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a git/gh command, returning the CompletedProcess.

    ``capture=True`` collects stdout/stderr (text). ``check=True`` raises on a
    non-zero exit. Callers that tolerate failure pass ``check=False`` and read
    ``returncode``.
    """
    return subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        check=check,
        text=True,
        capture_output=capture,
    )


def _git(
    repo: Path,
    *args: str,
    check: bool = True,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    return _run(["git", "-C", str(repo), *args], check=check, capture=capture)


def _log(msg: str) -> None:
    """Progress goes to stderr so ``--print-path`` keeps stdout parseable."""
    print(msg, file=sys.stderr, flush=True)


def _sanitize_branch(branch: str) -> str:
    """Turn a branch name into a filesystem-safe directory component."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", branch).strip("-") or "branch"


def _default_worktree_path(branch: str) -> Path:
    base = Path("/tmp") / "evmax-sched"
    return base / _sanitize_branch(branch)


def _remote_ref_exists(repo: Path, ref: str) -> bool:
    cp = _git(repo, "rev-parse", "--verify", "--quiet", ref, check=False)
    return cp.returncode == 0


def _unmerged_commit_count(repo: Path, branch_ref: str, base_ref: str) -> int:
    """Number of commits on ``branch_ref`` not yet in ``base_ref``."""
    cp = _git(repo, "rev-list", "--count", f"{base_ref}..{branch_ref}", check=False)
    if cp.returncode != 0:
        return 0
    try:
        return int(cp.stdout.strip() or "0")
    except ValueError:
        return 0


def _remove_worktree_if_present(repo: Path, path: Path) -> None:
    """Detach any existing worktree at ``path`` so ``open`` is idempotent."""
    if path.exists():
        _git(repo, "worktree", "remove", "--force", str(path), check=False)
    # Prune stale administrative entries whose directory is already gone.
    _git(repo, "worktree", "prune", check=False)


def cmd_open(args: argparse.Namespace) -> int:
    repo: Path = args.repo
    branch: str = args.branch
    base_remote = f"origin/{BASE_BRANCH}"
    worktree = args.worktree or _default_worktree_path(branch)

    _log(f"[open] fetch origin (repo={repo})")
    _git(repo, "fetch", "origin", "--prune", "--quiet")

    if not _remote_ref_exists(repo, base_remote):
        _log(f"[open] FATAL: {base_remote} does not exist after fetch")
        return 2

    # Decide the start point.
    start_point = base_remote
    rebase_after = False
    branch_remote = f"origin/{branch}"
    if args.rolling and _remote_ref_exists(repo, branch_remote):
        unmerged = _unmerged_commit_count(repo, branch_remote, base_remote)
        if unmerged > 0:
            # Preserve another task's not-yet-merged owned files; re-anchor to
            # live main so the eventual PR diff stays against the current tip.
            start_point = branch_remote
            rebase_after = True
            _log(
                f"[open] rolling branch {branch_remote} has {unmerged} unmerged "
                f"commit(s); basing on it and rebasing onto {base_remote}"
            )
        else:
            _log(
                f"[open] rolling branch {branch_remote} fully merged; "
                f"basing fresh on {base_remote}"
            )
    else:
        _log(f"[open] basing branch {branch} on {start_point}")

    _remove_worktree_if_present(repo, worktree)
    worktree.parent.mkdir(parents=True, exist_ok=True)

    # -B force-creates/resets the LOCAL branch to start_point in the new tree.
    _git(repo, "worktree", "add", "--quiet", "-B", branch, str(worktree), start_point)

    if rebase_after:
        cp = _git(worktree, "rebase", base_remote, check=False)
        if cp.returncode != 0:
            # Regenerated state is idempotent: abort and re-anchor to main; the
            # seed script will reproduce the current-truth file from scratch.
            _log(
                "[open] rebase onto main conflicted; aborting and re-anchoring "
                "to origin/main (regenerated state will be rebuilt fresh)"
            )
            _git(worktree, "rebase", "--abort", check=False)
            _git(worktree, "reset", "--hard", base_remote)

    head = _git(worktree, "rev-parse", "--short", "HEAD").stdout.strip()
    _log(f"[open] worktree ready at {worktree} on {branch} @ {head}")

    if args.print_path:
        # The ONLY thing on stdout, so `WT=$(... --print-path)` works.
        print(str(worktree))
    return 0


def _open_pr_number(repo: Path, branch: str) -> str | None:
    cp = _run(
        [
            "gh", "pr", "list",
            "--repo", _repo_slug(repo),
            "--head", branch,
            "--state", "open",
            "--json", "number",
            "--jq", ".[0].number // empty",
        ],
        cwd=repo,
        check=False,
    )
    if cp.returncode != 0:
        return None
    num = cp.stdout.strip()
    return num or None


def _repo_slug(repo: Path) -> str:
    """owner/name for gh --repo, derived from the origin URL."""
    cp = _git(repo, "remote", "get-url", "origin", check=False)
    url = cp.stdout.strip()
    m = re.search(r"[:/]([^/:]+/[^/]+?)(?:\.git)?$", url)
    return m.group(1) if m else ""


def cmd_ship(args: argparse.Namespace) -> int:
    repo: Path = args.repo
    branch: str = args.branch
    worktree: Path = args.worktree
    owned: list[str] = args.files
    base_remote = f"origin/{BASE_BRANCH}"

    if not owned:
        _log("[ship] FATAL: no owned files given after `--`")
        return 2
    if not worktree.exists():
        _log(f"[ship] FATAL: worktree {worktree} does not exist")
        return 2

    # Stage ONLY the owned files that actually changed. Never `git add -A`.
    # `git add` on an unchanged path is a harmless no-op.
    _git(worktree, "add", "--", *owned)
    staged = _git(worktree, "diff", "--cached", "--name-only").stdout.strip()
    if not staged:
        _log("[ship] no owned files changed — nothing to ship")
        if not args.keep_worktree:
            _remove_worktree_if_present(repo, worktree)
        return 0
    _log(f"[ship] staged:\n{staged}")

    commit_args = ["commit", "-m", args.message or args.title]
    if not args.run_hooks:
        commit_args.append("--no-verify")
    _git(worktree, *commit_args)

    # Push. Rolling branches rewrite history (rebased onto main) → force-with-
    # lease + one retry after re-anchoring to the current remote branch.
    rc = _push(worktree, repo, branch, rolling=args.rolling, base_remote=base_remote)
    if rc != 0:
        _log(
            f"[ship] push failed for {branch}; work is committed locally in "
            f"{worktree} — resolve by hand, do NOT delete the worktree"
        )
        return rc

    pr_url = _ensure_pr(args, repo, branch)

    if not args.keep_worktree:
        _remove_worktree_if_present(repo, worktree)

    if pr_url:
        _log(f"[ship] PR: {pr_url}")
        print(pr_url)
    else:
        _log(f"[ship] pushed {branch}; open the PR manually (gh unavailable)")
        print(branch)
    return 0


def _push(
    worktree: Path,
    repo: Path,
    branch: str,
    *,
    rolling: bool,
    base_remote: str,
) -> int:
    if not rolling:
        cp = _git(worktree, "push", "-u", "origin", branch, check=False, capture=True)
        if cp.returncode != 0:
            _log(cp.stderr.strip())
        return cp.returncode

    cp = _git(
        worktree, "push", "--force-with-lease", "origin", branch,
        check=False, capture=True,
    )
    if cp.returncode == 0:
        return 0

    # Another task moved the rolling branch between our open and ship. Re-fetch,
    # rebase our (disjoint-file) commit onto the updated remote branch, retry.
    _log("[ship] force-with-lease rejected; re-fetching and rebasing onto remote branch")
    _git(worktree, "fetch", "origin", "--quiet", check=False)
    branch_remote = f"origin/{branch}"
    if _remote_ref_exists(repo, branch_remote):
        reb = _git(worktree, "rebase", branch_remote, check=False)
        if reb.returncode != 0:
            # Same-file conflict (should not happen for disjoint owners): drop
            # our commit onto the remote branch and re-commit the owned files,
            # which are still present on disk in the worktree.
            _log("[ship] rebase onto remote branch conflicted; re-applying owned files")
            _git(worktree, "rebase", "--abort", check=False)
            _git(worktree, "reset", "--soft", branch_remote, check=False)
            _git(worktree, "commit", "--no-verify", "-m",
                 "chore(models): re-applied rolling state after concurrent push",
                 check=False)
    cp2 = _git(
        worktree, "push", "--force-with-lease", "origin", branch,
        check=False, capture=True,
    )
    if cp2.returncode != 0:
        _log(cp2.stderr.strip())
    return cp2.returncode


def _ensure_pr(args: argparse.Namespace, repo: Path, branch: str) -> str | None:
    if args.no_pr:
        _log("[ship] --no-pr: skipping PR creation")
        return None

    existing = _open_pr_number(repo, branch)
    if existing:
        _log(f"[ship] PR #{existing} already open for {branch}; leaving it")
        cp = _run(
            ["gh", "pr", "view", existing, "--repo", _repo_slug(repo),
             "--json", "url", "--jq", ".url"],
            cwd=repo, check=False,
        )
        return cp.stdout.strip() or None

    body = (args.body or "") + (
        "\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)"
    )
    cp = _run(
        ["gh", "pr", "create",
         "--repo", _repo_slug(repo),
         "--base", BASE_BRANCH,
         "--head", branch,
         "--title", args.title,
         "--body", body],
        cwd=repo, check=False,
    )
    if cp.returncode != 0:
        _log(f"[ship] gh pr create failed:\n{cp.stderr.strip()}")
        return None
    return cp.stdout.strip() or None


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repo", type=Path, default=DEFAULT_REPO,
                        help="Path to the shared checkout (default: %(default)s)")
    common.add_argument("--branch", default=DEFAULT_ROLLING_BRANCH,
                        help="Branch name (default rolling branch: %(default)s)")
    common.add_argument("--rolling", action="store_true",
                        help="Single long-lived accumulating branch (state tasks). "
                             "Omit for a fresh dated branch off origin/main (code tasks).")

    po = sub.add_parser("open", parents=[common],
                        help="Fetch origin and open an isolated worktree off the live remote tip")
    po.add_argument("--worktree", type=Path, default=None,
                    help="Worktree path (default: /tmp/evmax-sched/<branch>)")
    po.add_argument("--print-path", action="store_true",
                    help="Print the worktree path (and nothing else) on stdout")
    po.set_defaults(func=cmd_open)

    ps = sub.add_parser("ship", parents=[common],
                        help="Stage owned files, commit, push, ensure one PR, remove worktree")
    ps.add_argument("--worktree", type=Path, required=True,
                    help="Worktree path returned by `open --print-path`")
    ps.add_argument("--title", required=True, help="PR title (used on creation only)")
    ps.add_argument("--body", default="", help="PR body (used on creation only)")
    ps.add_argument("--message", default=None,
                    help="Commit message (defaults to --title)")
    ps.add_argument("--run-hooks", action="store_true",
                    help="Run pre-commit hooks (default: --no-verify; use for code tasks)")
    ps.add_argument("--no-pr", action="store_true",
                    help="Push only; do not create/inspect a PR (offline/tests)")
    ps.add_argument("--keep-worktree", action="store_true",
                    help="Do not remove the worktree after shipping")
    ps.add_argument("files", nargs="*",
                    help="Owned files to stage (place after `--`)")
    ps.set_defaults(func=cmd_ship)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
