"""Tests for evmax.provenance.code_version — git code provenance stamping.

Covers:
  1. Clean checkout on main → bare short SHA
  2. Dirty working tree → '-dirty' suffix
  3. Non-main branch → '@branch' suffix (composes with '-dirty')
  4. Not a git repo → None (column stays NULL, scan never breaks)
  5. Result is cached per (repo_root) within a process
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from evmax.provenance import code_version


def _run(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A fresh git repo on branch 'main' with one commit."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(repo, "init", "-b", "main")
    _run(repo, "config", "user.email", "test@evmax.local")
    _run(repo, "config", "user.name", "evmax-test")
    (repo / "file.txt").write_text("v1\n")
    _run(repo, "add", "file.txt")
    _run(repo, "commit", "-m", "initial")
    yield repo
    code_version.cache_clear()


def test_clean_main_is_bare_short_sha(git_repo: Path):
    sha = _run(git_repo, "rev-parse", "--short", "HEAD")
    assert code_version(str(git_repo)) == sha


def test_dirty_tree_appends_dirty_suffix(git_repo: Path):
    (git_repo / "file.txt").write_text("v2 uncommitted\n")
    sha = _run(git_repo, "rev-parse", "--short", "HEAD")
    assert code_version(str(git_repo)) == f"{sha}-dirty"


def test_non_main_branch_appends_branch_name(git_repo: Path):
    _run(git_repo, "checkout", "-b", "fix/some-feature")
    sha = _run(git_repo, "rev-parse", "--short", "HEAD")
    assert code_version(str(git_repo)) == f"{sha}@fix/some-feature"


def test_dirty_feature_branch_composes_both_suffixes(git_repo: Path):
    _run(git_repo, "checkout", "-b", "fix/some-feature")
    (git_repo / "file.txt").write_text("v2 uncommitted\n")
    sha = _run(git_repo, "rev-parse", "--short", "HEAD")
    assert code_version(str(git_repo)) == f"{sha}-dirty@fix/some-feature"


def test_non_git_directory_returns_none(tmp_path: Path):
    plain = tmp_path / "not-a-repo"
    plain.mkdir()
    try:
        assert code_version(str(plain)) is None
    finally:
        code_version.cache_clear()


def test_result_is_cached_per_process(git_repo: Path):
    first = code_version(str(git_repo))
    # Dirty the tree AFTER the first call — the cached value must not change,
    # because a scan is a single process and stamps one code state throughout.
    (git_repo / "file.txt").write_text("v2 uncommitted\n")
    assert code_version(str(git_repo)) == first
