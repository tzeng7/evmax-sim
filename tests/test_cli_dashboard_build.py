"""Tests for the dashboard SPA staleness check that prevents serving a stale UI."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from evmax.cli.commands import dashboard


@pytest.fixture
def fake_frontend(tmp_path, monkeypatch):
    """Point the dashboard module at a throwaway frontend tree."""
    frontend = tmp_path / "frontend"
    src = frontend / "src"
    dist = frontend / "dist"
    src.mkdir(parents=True)
    dist.mkdir(parents=True)
    monkeypatch.setattr(dashboard, "FRONTEND", frontend)
    monkeypatch.setattr(dashboard, "DIST", dist)
    return frontend


def _touch(path: Path, mtime: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")
    os.utime(path, (mtime, mtime))


def test_stale_when_dist_missing(fake_frontend):
    _touch(fake_frontend / "src" / "App.tsx", 1000.0)
    # dist/index.html does not exist yet
    assert dashboard._spa_is_stale() is True


def test_fresh_when_dist_newer_than_source(fake_frontend):
    _touch(fake_frontend / "src" / "App.tsx", 1000.0)
    _touch(fake_frontend / "dist" / "index.html", 2000.0)
    assert dashboard._spa_is_stale() is False


def test_stale_when_source_newer_than_dist(fake_frontend):
    _touch(fake_frontend / "dist" / "index.html", 1000.0)
    _touch(fake_frontend / "src" / "App.tsx", 2000.0)  # edited after build
    assert dashboard._spa_is_stale() is True


def test_stale_when_nested_source_newer(fake_frontend):
    _touch(fake_frontend / "dist" / "index.html", 1000.0)
    _touch(fake_frontend / "src" / "components" / "ActionBar.tsx", 2000.0)
    assert dashboard._spa_is_stale() is True


def test_stale_when_top_level_config_newer(fake_frontend):
    """A change to vite.config.ts / package.json must also trigger a rebuild."""
    _touch(fake_frontend / "src" / "App.tsx", 1000.0)
    _touch(fake_frontend / "dist" / "index.html", 1500.0)
    _touch(fake_frontend / "vite.config.ts", 2000.0)
    assert dashboard._spa_is_stale() is True


def test_maybe_build_skips_when_fresh(fake_frontend, monkeypatch):
    _touch(fake_frontend / "src" / "App.tsx", 1000.0)
    _touch(fake_frontend / "dist" / "index.html", 2000.0)
    (fake_frontend / "node_modules").mkdir()
    calls: list[list[str]] = []
    monkeypatch.setattr(dashboard.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    dashboard._maybe_build_spa(rebuild=False)
    assert calls == []  # nothing built


def test_maybe_build_runs_when_stale(fake_frontend, monkeypatch):
    _touch(fake_frontend / "dist" / "index.html", 1000.0)
    _touch(fake_frontend / "src" / "App.tsx", 2000.0)
    (fake_frontend / "node_modules").mkdir()
    calls: list[list[str]] = []
    monkeypatch.setattr(dashboard.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    dashboard._maybe_build_spa(rebuild=False)
    assert ["npm", "run", "build"] in calls


def test_maybe_build_installs_deps_when_missing(fake_frontend, monkeypatch):
    _touch(fake_frontend / "dist" / "index.html", 1000.0)
    _touch(fake_frontend / "src" / "App.tsx", 2000.0)
    # no node_modules → should npm install before building
    calls: list[list[str]] = []
    monkeypatch.setattr(dashboard.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    dashboard._maybe_build_spa(rebuild=False)
    assert ["npm", "install"] in calls
    assert ["npm", "run", "build"] in calls
