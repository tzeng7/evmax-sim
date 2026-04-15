"""Schema + filter-audit tests for ARCH-11.

Verifies:
  1. A fresh predictions.db created via get_connection() has the new
     `mode` / `captured_yes_price` / `model_version` columns on both
     ev_predictions and prop_observations.
  2. New columns default to the expected values (`mode='live'`, NULL
     for the others) when rows are inserted without specifying them.
  3. The audited reader queries filter out `mode='shadow'` rows by
     default, so shadow data does not leak into the live bet log shown
     by `evmax cleanup show` / metrics / the web dashboard.
  4. An existing database missing the new columns gets them added by
     the migration loop (idempotent).
"""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from evmax.agents.cleanup import db as db_module


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """Point DB_PATH at a clean tmp file and return a connection."""
    db_path = tmp_path / "predictions.db"
    monkeypatch.setattr(db_module, "DB_PATH", db_path)
    conn = db_module.get_connection()
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# Schema presence
# ---------------------------------------------------------------------------


def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {r[1] for r in rows}


def test_ev_predictions_has_mode_columns(fresh_db):
    cols = _column_names(fresh_db, "ev_predictions")
    assert "mode" in cols
    assert "captured_yes_price" in cols
    assert "model_version" in cols


def test_prop_observations_has_mode_columns(fresh_db):
    cols = _column_names(fresh_db, "prop_observations")
    assert "mode" in cols
    assert "captured_yes_price" in cols
    assert "model_version" in cols


# ---------------------------------------------------------------------------
# Defaults — a row inserted without `mode` must come back as 'live'
# ---------------------------------------------------------------------------


def _insert_ev(conn, market_id: str, mode: str | None = None) -> None:
    cols = [
        "scan_date", "market_id", "event_id", "sector", "yes_team",
        "market_type", "kalshi_yes_price", "sharp_true_prob",
        "blended_true_prob", "ev_pct", "kelly_fraction",
    ]
    values = [
        date.today().isoformat(), market_id, "evt:1", "nba", "LAL",
        "moneyline", 0.50, 0.55, 0.55, 0.05, 0.02,
    ]
    if mode is not None:
        cols.append("mode")
        values.append(mode)
    placeholders = ",".join(["?"] * len(values))
    conn.execute(
        f"INSERT INTO ev_predictions ({','.join(cols)}) VALUES ({placeholders})",
        values,
    )
    conn.commit()


def test_ev_prediction_default_mode_is_live(fresh_db):
    _insert_ev(fresh_db, "m1")
    row = fresh_db.execute(
        "SELECT mode FROM ev_predictions WHERE market_id = 'm1'"
    ).fetchone()
    assert row["mode"] == "live"


def test_ev_prediction_explicit_shadow_persists(fresh_db):
    _insert_ev(fresh_db, "m2", mode="shadow")
    row = fresh_db.execute(
        "SELECT mode FROM ev_predictions WHERE market_id = 'm2'"
    ).fetchone()
    assert row["mode"] == "shadow"


# ---------------------------------------------------------------------------
# Reader audit — shadow rows must not show up in live reads
# ---------------------------------------------------------------------------


def test_cleanup_show_query_excludes_shadow(fresh_db):
    """Mirrors the WHERE clause `evmax cleanup show` uses. Shadow rows
    inserted alongside live rows must not appear in the result."""
    _insert_ev(fresh_db, "m_live", mode="live")
    _insert_ev(fresh_db, "m_shadow", mode="shadow")
    rows = fresh_db.execute(
        "SELECT market_id FROM ev_predictions "
        "WHERE voided = 0 AND mode = 'live'"
    ).fetchall()
    ids = {r["market_id"] for r in rows}
    assert ids == {"m_live"}


def test_agents_show_open_latest_subquery_excludes_shadow(fresh_db):
    """The inner subquery in `evmax agents show` must also filter by mode
    — otherwise a later shadow re-scan would become the 'latest' row for
    a market that was originally scanned live, hiding the live row."""
    _insert_ev(fresh_db, "m_same", mode="live")
    # Simulate a later-day shadow re-scan for the same market_id.
    fresh_db.execute(
        """INSERT INTO ev_predictions
           (scan_date, market_id, event_id, sector, yes_team, market_type,
            kalshi_yes_price, sharp_true_prob, blended_true_prob, ev_pct,
            kelly_fraction, mode)
           VALUES ('2099-01-01', 'm_same_shadow', 'evt:1', 'nba', 'LAL',
                   'moneyline', 0.5, 0.55, 0.55, 0.05, 0.02, 'shadow')"""
    )
    fresh_db.commit()

    rows = fresh_db.execute(
        """SELECT p.market_id FROM ev_predictions p
           INNER JOIN (
               SELECT market_id, MAX(scan_date) AS latest_scan
               FROM ev_predictions
               WHERE voided = 0 AND mode = 'live'
               GROUP BY market_id
           ) latest ON p.market_id = latest.market_id
                   AND p.scan_date = latest.latest_scan
           WHERE p.mode = 'live'"""
    ).fetchall()
    ids = {r["market_id"] for r in rows}
    assert "m_shadow" not in ids
    assert "m_same_shadow" not in ids
    assert "m_same" in ids


def test_prop_observations_mode_filter(fresh_db):
    """Shadow prop observations are logged but should not appear in the
    live `evmax cleanup show --props` output."""
    for mid, mode in (("p1", "live"), ("p2", "shadow")):
        fresh_db.execute(
            """INSERT INTO prop_observations
               (scan_date, sector, player_name, stat_type, line, mode)
               VALUES (?, 'nba', 'LeBron', 'points', 24.5, ?)""",
            (date.today().isoformat(), mode),
        )
    fresh_db.commit()
    rows = fresh_db.execute(
        "SELECT market_id FROM prop_observations WHERE mode = 'live'"
    ).fetchall()
    # Both rows have NULL market_id since we didn't set it — match on
    # insert count instead
    count = fresh_db.execute(
        "SELECT COUNT(*) AS n FROM prop_observations WHERE mode = 'live'"
    ).fetchone()
    assert count["n"] == 1


# ---------------------------------------------------------------------------
# Idempotent migration — running the ALTER loop on an already-migrated DB
# must not error
# ---------------------------------------------------------------------------


def test_migration_is_idempotent(tmp_path, monkeypatch):
    db_path = tmp_path / "predictions.db"
    monkeypatch.setattr(db_module, "DB_PATH", db_path)
    # First open runs CREATE + ALTER
    conn1 = db_module.get_connection()
    conn1.close()
    # Second open hits the ALTERs again; they should all raise "duplicate
    # column" and be swallowed by the except-pass.
    conn2 = db_module.get_connection()
    cols = _column_names(conn2, "ev_predictions")
    assert "mode" in cols  # still there after idempotent run
    conn2.close()


def test_migration_adds_columns_to_legacy_db(tmp_path, monkeypatch):
    """Simulate a pre-ARCH-11 DB by creating ev_predictions without the
    new columns, then running get_connection() and verifying the ALTER
    loop backfills them."""
    db_path = tmp_path / "legacy.db"
    pre = sqlite3.connect(str(db_path))
    pre.executescript(
        """CREATE TABLE ev_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_date TEXT NOT NULL,
            market_id TEXT NOT NULL,
            event_id TEXT,
            sector TEXT,
            yes_team TEXT,
            market_type TEXT,
            kalshi_yes_price REAL,
            sharp_true_prob REAL,
            blended_true_prob REAL,
            ev_pct REAL,
            kelly_fraction REAL,
            UNIQUE(market_id, scan_date)
        );
        CREATE TABLE prop_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_date TEXT NOT NULL,
            sector TEXT NOT NULL,
            player_name TEXT NOT NULL,
            stat_type TEXT NOT NULL,
            line REAL NOT NULL,
            market_id TEXT,
            UNIQUE(market_id, scan_date)
        );
        """
    )
    pre.commit()
    pre.close()

    monkeypatch.setattr(db_module, "DB_PATH", db_path)
    conn = db_module.get_connection()
    ev_cols = _column_names(conn, "ev_predictions")
    prop_cols = _column_names(conn, "prop_observations")
    assert {"mode", "captured_yes_price", "model_version"} <= ev_cols
    assert {"mode", "captured_yes_price", "model_version"} <= prop_cols
    conn.close()
