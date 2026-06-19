"""Regression: `evmax cleanup void` must NOT void shadow-mode rows.

Before the fix the candidate query lacked a mode='live' guard, so stale
unresolved shadow predictions were swept into the void, shrinking the MODEL-9
shadow-validation sample."""

from __future__ import annotations

import sqlite3

import pytest

from evmax.cli.commands.cleanup import _stale_unmatched_candidates


def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE ev_predictions (
            id INTEGER PRIMARY KEY,
            market_id TEXT, event_id TEXT, sector TEXT, yes_team TEXT,
            event_date TEXT, scan_date TEXT, ev_pct REAL,
            voided INTEGER NOT NULL DEFAULT 0, mode TEXT NOT NULL DEFAULT 'live'
        )
        """
    )
    conn.execute("CREATE TABLE ev_outcomes (market_id TEXT, outcome INTEGER)")
    return conn


def _insert(conn, market_id, *, mode, event_date="2026-01-01", voided=0):
    conn.execute(
        "INSERT INTO ev_predictions (market_id, event_id, sector, yes_team, "
        "event_date, scan_date, ev_pct, voided, mode) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (market_id, f"evt:{market_id}", "tennis", "X", event_date, event_date, 0.05, voided, mode),
    )
    conn.commit()


def test_void_candidates_exclude_shadow_rows():
    conn = _make_db()
    _insert(conn, "live_stale", mode="live")
    _insert(conn, "shadow_stale", mode="shadow")

    ids = {r["market_id"] for r in _stale_unmatched_candidates(conn, "2026-06-01")}

    assert "live_stale" in ids
    assert "shadow_stale" not in ids  # the bug: shadow rows must be left alone


def test_void_candidates_skip_resolved_and_already_voided():
    conn = _make_db()
    _insert(conn, "resolved", mode="live")
    conn.execute("INSERT INTO ev_outcomes (market_id, outcome) VALUES ('resolved', 1)")
    _insert(conn, "already_void", mode="live", voided=1)
    _insert(conn, "open_live", mode="live")
    conn.commit()

    ids = {r["market_id"] for r in _stale_unmatched_candidates(conn, "2026-06-01")}
    assert ids == {"open_live"}


def test_void_candidates_respect_cutoff():
    conn = _make_db()
    _insert(conn, "past", mode="live", event_date="2026-01-01")
    _insert(conn, "future", mode="live", event_date="2026-12-01")

    ids = {r["market_id"] for r in _stale_unmatched_candidates(conn, "2026-06-01")}
    assert ids == {"past"}


def test_void_command_is_registered_with_its_own_options():
    # Guards against the helper accidentally stealing the @app.command("void")
    # decorator (which would expose CONN/CUTOFF_STR args instead of the real
    # --before/--days/--dry-run command).
    from typer.testing import CliRunner

    from evmax.cli.commands.cleanup import app

    result = CliRunner().invoke(app, ["void", "--help"])
    assert result.exit_code == 0
    assert "--before" in result.output
    assert "--dry-run" in result.output
    assert "CUTOFF_STR" not in result.output
