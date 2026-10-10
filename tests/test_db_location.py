"""EVMAX_DB_DIR / EVMAX_DB_READONLY overrides (evmax/db_location.py).

Regression for the worktree gotcha: run from a worktree, the CLI created an
EMPTY archive.db and read-side lenses silently saw zero rows. The read-only
switch must (a) read an existing database, (b) refuse every write, (c) never
create a missing file, and (d) skip schema/migration DDL in all three
connection factories that honor it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from evmax import archiver, db_location, portfolios
from evmax.agents.cleanup import db as cleanup_db


def _make_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (a INTEGER)")
    conn.execute("INSERT INTO t VALUES (1)")
    conn.commit()
    conn.close()


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv(db_location.ENV_DB_DIR, raising=False)
    monkeypatch.delenv(db_location.ENV_DB_READONLY, raising=False)
    return monkeypatch


def test_resolve_db_path_default_and_override(clean_env, tmp_path):
    default = tmp_path / "default" / "predictions.db"
    assert db_location.resolve_db_path("predictions.db", default) == default
    clean_env.setenv(db_location.ENV_DB_DIR, str(tmp_path / "main" / "data"))
    assert db_location.resolve_db_path("predictions.db", default) == tmp_path / "main" / "data" / "predictions.db"
    clean_env.setenv(db_location.ENV_DB_DIR, "   ")
    assert db_location.resolve_db_path("archive.db", default) == default


@pytest.mark.parametrize("value,expected", [
    ("1", True), ("true", True), ("YES", True), (" on ", True),
    ("0", False), ("false", False), ("", False), ("ro", False),
])
def test_readonly_enabled_values(clean_env, value, expected):
    clean_env.setenv(db_location.ENV_DB_READONLY, value)
    assert db_location.readonly_enabled() is expected


def test_readonly_disabled_when_unset(clean_env):
    assert db_location.readonly_enabled() is False


def test_connect_readonly_reads_but_blocks_writes(tmp_path):
    path = tmp_path / "x.db"
    _make_db(path)
    conn = db_location.connect_readonly(path)
    assert conn.execute("SELECT a FROM t").fetchone()["a"] == 1
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        conn.execute("INSERT INTO t VALUES (2)")
    conn.close()


def test_connect_readonly_never_creates_missing_file(tmp_path):
    missing = tmp_path / "nope" / "archive.db"
    with pytest.raises(FileNotFoundError, match="EVMAX_DB_DIR"):
        db_location.connect_readonly(missing)
    assert not missing.exists() and not missing.parent.exists()


def test_path_with_spaces(tmp_path):
    path = tmp_path / "dir with space" / "p.db"
    path.parent.mkdir()
    _make_db(path)
    assert db_location.connect_readonly(path).execute("SELECT count(*) FROM t").fetchone()[0] == 1


@pytest.mark.parametrize("module,factory", [
    (cleanup_db, "get_connection"),
    (archiver, "_get_connection"),
    (portfolios, "_get_conn"),
])
def test_factories_skip_ddl_and_block_writes_in_readonly_mode(clean_env, tmp_path, module, factory):
    path = tmp_path / "live.db"
    _make_db(path)
    target = cleanup_db if module is portfolios else module   # portfolios imports cleanup_db.DB_PATH by name
    clean_env.setattr(module, "DB_PATH", path)
    clean_env.setattr(target, "DB_PATH", path)
    clean_env.setenv(db_location.ENV_DB_READONLY, "1")
    conn = getattr(module, factory)()
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert tables == {"t"}                      # no schema / migrations ran
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("INSERT INTO t VALUES (3)")
    conn.close()


def test_factories_unchanged_when_readonly_off(clean_env, tmp_path):
    path = tmp_path / "fresh.db"
    clean_env.setattr(cleanup_db, "DB_PATH", path)
    conn = cleanup_db.get_connection()          # normal mode creates + migrates as before
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "ev_predictions" in tables
    conn.close()


def test_db_dir_alone_implies_readonly(clean_env, tmp_path):
    clean_env.setenv(db_location.ENV_DB_DIR, str(tmp_path))
    assert db_location.readonly_enabled() is True
    clean_env.setenv(db_location.ENV_DB_READONLY, "0")          # explicit opt-out
    assert db_location.readonly_enabled() is False
    clean_env.setenv(db_location.ENV_DB_READONLY, "maybe")      # unrecognized → falls back to the DIR rule
    assert db_location.readonly_enabled() is True


def test_sizing_replay_opens_read_only(clean_env, tmp_path):
    from evmax.backtest import sizing

    calls = []

    def spy(path, timeout=5.0):
        calls.append(Path(path))
        raise RuntimeError("spy")

    clean_env.setattr(sizing, "connect_readonly", spy)
    clean_env.setenv(db_location.ENV_DB_READONLY, "1")
    with pytest.raises(RuntimeError, match="spy"):
        sizing.load_resolved_rows(db_path=tmp_path / "p.db")
    assert calls == [tmp_path / "p.db"]


def test_archive_cli_uses_overridable_archive_path():
    from evmax.cli.commands import archive as archive_cli

    assert archive_cli.DB_PATH is archiver.DB_PATH


def test_backfill_clv_recompute_refused_when_readonly(clean_env, tmp_path):
    from typer.testing import CliRunner

    from evmax.cli.commands.cleanup import app

    clean_env.setenv(db_location.ENV_DB_DIR, str(tmp_path))
    result = CliRunner().invoke(app, ["backfill-clv", "--recompute"])
    assert result.exit_code == 1
    assert "refusing" in result.output
    assert not list(tmp_path.iterdir())                         # no backup file written
