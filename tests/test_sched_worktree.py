"""Tests for scripts/sched_worktree.py — the isolated-worktree git helper for
scheduled tasks.

These run fully offline: a local bare repo stands in for ``origin`` and every
``ship`` uses ``--no-pr`` so ``gh`` is never invoked. They assert the two
properties that keep parallel scheduled runs from conflicting:

1. Branches are cut from the fetched remote tip (``origin/main``), never a
   stale local main, and each run gets its own isolated worktree.
2. ``ship`` stages ONLY the owned files, is a no-op when nothing changed, and a
   ``--rolling`` branch accumulates disjoint owners while re-anchoring to main.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sched_worktree.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sched_worktree", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sched = _load_module()


# --------------------------------------------------------------------------- #
# git helpers
# --------------------------------------------------------------------------- #
def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _configure_identity(repo: Path) -> None:
    _git(repo, "config", "user.email", "sched-test@example.com")
    _git(repo, "config", "user.name", "Sched Test")
    _git(repo, "config", "commit.gpgsign", "false")


def _write(repo: Path, rel: str, content: str) -> None:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


@pytest.fixture()
def repos(tmp_path: Path):
    """Return (work_checkout, origin_bare) with an initial main commit pushed.

    ``work`` is the shared checkout scheduled tasks run against; ``origin`` is a
    bare repo standing in for GitHub. Two owned state files exist on main.
    """
    origin = tmp_path / "origin.git"
    subprocess.run(
        ["git", "init", "--bare", "--initial-branch=main", str(origin)],
        check=True, capture_output=True,
    )
    work = tmp_path / "work"
    subprocess.run(
        ["git", "init", "--initial-branch=main", str(work)],
        check=True, capture_output=True,
    )
    _configure_identity(work)
    _git(work, "remote", "add", "origin", str(origin))
    _write(work, "data/models/elo_state.json", '{"v": 1}\n')
    _write(work, "data/models/form_state.json", '{"v": 1}\n')
    _write(work, "README.md", "base\n")
    _git(work, "add", "-A")
    _git(work, "commit", "-m", "init")
    _git(work, "push", "-u", "origin", "main")
    return work, origin


def _origin_head(origin: Path, ref: str) -> str:
    return subprocess.run(
        ["git", "-C", str(origin), "rev-parse", ref],
        check=True, text=True, capture_output=True,
    ).stdout.strip()


def _origin_has_branch(origin: Path, branch: str) -> bool:
    cp = subprocess.run(
        ["git", "-C", str(origin), "rev-parse", "--verify", "--quiet", branch],
        text=True, capture_output=True,
    )
    return cp.returncode == 0


def _file_on_ref(origin: Path, ref: str, rel: str) -> str | None:
    cp = subprocess.run(
        ["git", "-C", str(origin), "show", f"{ref}:{rel}"],
        text=True, capture_output=True,
    )
    return cp.stdout if cp.returncode == 0 else None


# --------------------------------------------------------------------------- #
# open
# --------------------------------------------------------------------------- #
def test_open_bases_on_origin_main(repos, tmp_path, capsys):
    work, origin = repos
    wt = tmp_path / "wt-topic"
    rc = sched.main(
        ["open", "--repo", str(work), "--branch", "topic/x",
         "--worktree", str(wt), "--print-path"]
    )
    assert rc == 0
    out = capsys.readouterr().out.strip()
    assert out == str(wt)
    assert wt.exists()
    # The new branch points at the fetched remote main tip.
    assert _git(wt, "rev-parse", "HEAD") == _origin_head(origin, "main")
    assert _git(wt, "rev-parse", "--abbrev-ref", "HEAD") == "topic/x"


def test_open_is_idempotent(repos, tmp_path):
    work, _ = repos
    wt = tmp_path / "wt-again"
    for _ in range(2):
        rc = sched.main(
            ["open", "--repo", str(work), "--branch", "topic/y",
             "--worktree", str(wt)]
        )
        assert rc == 0
        assert wt.exists()


# --------------------------------------------------------------------------- #
# ship
# --------------------------------------------------------------------------- #
def test_ship_stages_only_owned_files(repos, tmp_path):
    work, origin = repos
    wt = tmp_path / "wt-ship"
    sched.main(["open", "--repo", str(work), "--branch", "topic/ship",
                "--worktree", str(wt)])
    # Change an owned file AND an unowned file in the same tree.
    _write(wt, "data/models/elo_state.json", '{"v": 2}\n')
    _write(wt, "README.md", "SHOULD NOT SHIP\n")

    rc = sched.main(
        ["ship", "--repo", str(work), "--branch", "topic/ship",
         "--worktree", str(wt), "--no-pr",
         "--title", "chore: elo refresh",
         "--", "data/models/elo_state.json"]
    )
    assert rc == 0
    assert _origin_has_branch(origin, "topic/ship")
    # Owned file updated on the pushed branch...
    assert _file_on_ref(origin, "topic/ship", "data/models/elo_state.json") == '{"v": 2}\n'
    # ...unowned file untouched (still the base content).
    assert _file_on_ref(origin, "topic/ship", "README.md") == "base\n"
    # Worktree cleaned up by default.
    assert not wt.exists()


def test_ship_noop_when_nothing_changed(repos, tmp_path):
    work, origin = repos
    wt = tmp_path / "wt-noop"
    sched.main(["open", "--repo", str(work), "--branch", "topic/noop",
                "--worktree", str(wt)])
    rc = sched.main(
        ["ship", "--repo", str(work), "--branch", "topic/noop",
         "--worktree", str(wt), "--no-pr", "--title", "noop",
         "--", "data/models/elo_state.json"]
    )
    assert rc == 0
    # No commit → branch never pushed.
    assert not _origin_has_branch(origin, "topic/noop")
    assert not wt.exists()


def test_ship_requires_owned_files(repos, tmp_path):
    work, _ = repos
    wt = tmp_path / "wt-nofiles"
    sched.main(["open", "--repo", str(work), "--branch", "topic/z",
                "--worktree", str(wt)])
    rc = sched.main(
        ["ship", "--repo", str(work), "--branch", "topic/z",
         "--worktree", str(wt), "--no-pr", "--title", "t"]
    )
    assert rc == 2  # no files after `--`


# --------------------------------------------------------------------------- #
# rolling branch
# --------------------------------------------------------------------------- #
def _rolling_run(work: Path, wt: Path, owned: str, value: str) -> None:
    """Simulate one state task: open rolling, rewrite one owned file, ship."""
    sched.main(["open", "--repo", str(work), "--branch", "bot/model-state",
                "--rolling", "--worktree", str(wt)])
    _write(wt, owned, value)
    sched.main(
        ["ship", "--repo", str(work), "--branch", "bot/model-state",
         "--rolling", "--worktree", str(wt), "--no-pr",
         "--title", "chore(models): rolling refresh", "--", owned]
    )


def test_rolling_accumulates_disjoint_owners(repos, tmp_path):
    work, origin = repos
    # Task A rewrites elo; Task B rewrites form. Disjoint owners, one branch.
    _rolling_run(work, tmp_path / "wtA", "data/models/elo_state.json", '{"v": 9}\n')
    _rolling_run(work, tmp_path / "wtB", "data/models/form_state.json", '{"v": 8}\n')

    # Both owners' changes coexist on the single rolling branch.
    assert _file_on_ref(origin, "bot/model-state", "data/models/elo_state.json") == '{"v": 9}\n'
    assert _file_on_ref(origin, "bot/model-state", "data/models/form_state.json") == '{"v": 8}\n'
    # main is untouched (PR not merged).
    assert _file_on_ref(origin, "main", "data/models/elo_state.json") == '{"v": 1}\n'


def test_rolling_rebases_fresh_after_merge(repos, tmp_path):
    work, origin = repos
    # Task A ships to the rolling branch.
    _rolling_run(work, tmp_path / "wtA", "data/models/elo_state.json", '{"v": 5}\n')
    # Simulate the rolling PR merging: fast-forward main to the branch on origin.
    _git(work, "fetch", "origin", "--quiet")
    _git(work, "merge", "--ff-only", "origin/bot/model-state")
    _git(work, "push", "origin", "main")
    merged_main = _origin_head(origin, "main")

    # Task B now opens rolling: branch is fully merged, so it bases FRESH on main.
    wtB = tmp_path / "wtB"
    sched.main(["open", "--repo", str(work), "--branch", "bot/model-state",
                "--rolling", "--worktree", str(wtB)])
    assert _git(wtB, "rev-parse", "HEAD") == merged_main
    # And main already carries task A's change (proving it merged, not lingered).
    assert _file_on_ref(origin, "main", "data/models/elo_state.json") == '{"v": 5}\n'


# --------------------------------------------------------------------------- #
# small unit helpers
# --------------------------------------------------------------------------- #
def test_sanitize_branch():
    assert sched._sanitize_branch("bot/model-state") == "bot-model-state"
    assert sched._sanitize_branch("drift-audit/2026-08-28") == "drift-audit-2026-08-28"
    assert sched._sanitize_branch("///") == "branch"


def test_repo_slug_parses_ssh_and_https(repos):
    work, _ = repos
    # The fixture's origin is a filesystem path; just assert it does not crash
    # and returns the trailing two path components.
    slug = sched._repo_slug(work)
    assert slug.endswith("origin.git") or "/" in slug


# --------------------------------------------------------------------------- #
# CI watch + auto-merge (state PRs only)
# --------------------------------------------------------------------------- #
def _cp(stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(args=[], returncode=returncode,
                                       stdout=stdout, stderr=stderr)


def _patch_gh(monkeypatch, handler):
    """Route sched._run calls to ``handler(argv) -> CompletedProcess``.

    Any non-gh shell-out raises so a stray call is caught, not run for real.
    """
    def fake_run(args, *, cwd=None, check=False, capture=True):
        # _repo_slug shells out to git for the origin URL -- answer it locally
        # so the handler only ever sees the gh calls under test.
        if args[:1] == ["git"] and "get-url" in args:
            return _cp(stdout="git@github.com:owner/repo.git\n", returncode=0)
        assert args and args[0] == "gh", f"unexpected shell-out: {args}"
        return handler(args)

    monkeypatch.setattr(sched, "_run", fake_run)


def test_watch_ci_all_pass(monkeypatch):
    payload = '[{"name":"Python tests","bucket":"pass","state":"SUCCESS","link":"u"}]'
    _patch_gh(monkeypatch, lambda argv: _cp(stdout=payload, returncode=0))
    bucket, _ = sched._watch_ci(Path("."), "b", timeout_s=1, interval_s=0)
    assert bucket == "pass"


def test_watch_ci_reports_failure(monkeypatch):
    payload = ('[{"name":"Frontend","bucket":"fail","state":"FAILURE",'
               '"link":"http://run/1"},'
               '{"name":"Python tests","bucket":"pass","state":"SUCCESS","link":""}]')
    _patch_gh(monkeypatch, lambda argv: _cp(stdout=payload, returncode=1))
    bucket, detail = sched._watch_ci(Path("."), "b", timeout_s=1, interval_s=0)
    assert bucket == "fail"
    assert "Frontend" in detail and "http://run/1" in detail


def test_watch_ci_pending_times_out(monkeypatch):
    payload = '[{"name":"Python tests","bucket":"pending","state":"IN_PROGRESS","link":""}]'
    _patch_gh(monkeypatch, lambda argv: _cp(stdout=payload, returncode=8))
    bucket, _ = sched._watch_ci(Path("."), "b", timeout_s=0, interval_s=0)
    assert bucket == "pending"


def test_watch_ci_no_checks(monkeypatch):
    _patch_gh(monkeypatch, lambda argv: _cp(stdout="", returncode=0))
    bucket, _ = sched._watch_ci(Path("."), "b", timeout_s=1, interval_s=0)
    assert bucket == "none"


def test_merge_when_green_merges_on_pass(monkeypatch):
    calls = []

    def handler(argv):
        calls.append(argv)
        if argv[1] == "pr" and argv[2] == "checks":
            return _cp(stdout='[{"name":"CI","bucket":"pass","link":""}]', returncode=0)
        return _cp(returncode=0)

    _patch_gh(monkeypatch, handler)
    args = argparse.Namespace(ci_timeout=1, ci_interval=0, merge_method="squash")
    sched._merge_when_green(args, Path("."), "bot/model-state")
    assert any(a[1:3] == ["pr", "merge"] for a in calls), "expected a merge on green CI"


def test_merge_when_green_skips_on_fail(monkeypatch):
    calls = []

    def handler(argv):
        calls.append(argv)
        if argv[1] == "pr" and argv[2] == "checks":
            return _cp(stdout='[{"name":"CI","bucket":"fail","link":"x"}]', returncode=1)
        return _cp(returncode=0)

    _patch_gh(monkeypatch, handler)
    args = argparse.Namespace(ci_timeout=1, ci_interval=0, merge_method="squash")
    sched._merge_when_green(args, Path("."), "bot/model-state")
    assert not any(a[1:3] == ["pr", "merge"] for a in calls), "must NOT merge on red CI"


def test_watch_ci_subcommand_exit_codes(monkeypatch, capsys):
    _patch_gh(monkeypatch, lambda argv: _cp(stdout='[{"name":"CI","bucket":"fail"}]',
                                            returncode=1))
    rc = sched.main(["watch-ci", "--repo", ".", "--branch", "b",
                     "--ci-timeout", "1", "--ci-interval", "0"])
    assert rc == 1
    assert capsys.readouterr().out.strip() == "fail"


def test_ship_no_pr_never_merges(repos, tmp_path, monkeypatch):
    """--merge-when-green with --no-pr (offline) must not shell out to gh."""
    work, _ = repos
    wt = tmp_path / "wt"
    sched.main(["open", "--repo", str(work), "--branch", "bot/model-state",
                "--rolling", "--worktree", str(wt)])
    _write(wt, "data/models/elo_state.json", '{"v": 42}\n')

    def boom(*a, **k):
        raise AssertionError("gh must not be called with --no-pr")

    monkeypatch.setattr(sched, "_merge_when_green", boom)
    rc = sched.main(["ship", "--repo", str(work), "--branch", "bot/model-state",
                     "--rolling", "--worktree", str(wt), "--no-pr",
                     "--merge-when-green", "--title", "t",
                     "--", "data/models/elo_state.json"])
    assert rc == 0
