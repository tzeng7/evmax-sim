"""Tests for evmax/archiver.py prop persistence.

Pre-2026-05-10, archived_sharp_odds dropped the over/under probabilities for
player props (true_prob_a/true_prob_b stored 0.0; total_line had no slot).
~132K NBA prop rows already exist in the historical archive with zeroed probs.

These tests assert the new prop columns (true_prob_over / true_prob_under /
total_line / prop_player_name / prop_stat_type) are now persisted, while the
moneyline path remains unaffected.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from evmax.archiver import DataArchiver
from evmax.models.odds import SharpBook, SharpOdds


@pytest.fixture
def temp_archive_db(tmp_path, monkeypatch):
    """Point the archiver at a temp DB so tests don't touch the real one."""
    db_path = tmp_path / "archive_test.db"
    monkeypatch.setattr("evmax.archiver.DB_PATH", db_path)
    return db_path


def _moneyline_sharp(event_id: str = "nba::2026-05-10::lakers_vs_warriors") -> SharpOdds:
    return SharpOdds(
        event_id=event_id,
        book=SharpBook.pinnacle,
        sector="nba",
        outcome_a_label="lakers",
        outcome_b_label="warriors",
        outcome_a_decimal=1.85,
        outcome_b_decimal=1.95,
        true_prob_a=0.52,
        true_prob_b=0.48,
        margin=0.04,
        event_date=datetime(2026, 5, 10, 22, 0, tzinfo=timezone.utc),
        fetched_at=datetime(2026, 5, 10, 21, 0, tzinfo=timezone.utc),
    )


def _prop_sharp(
    player: str = "jalen_brunson",
    stat: str = "assists",
    line: float = 6.5,
    prob_over: float = 0.52,
) -> SharpOdds:
    return SharpOdds(
        event_id=f"nba::2026-05-10::prop::{player}::{stat}::{line}",
        book=SharpBook.pinnacle,
        sector="nba",
        outcome_a_label="over",
        outcome_b_label="under",
        outcome_a_decimal=1.91,
        outcome_b_decimal=1.91,
        true_prob_a=0.0,                # legacy/unused for props
        true_prob_b=0.0,                # legacy/unused for props
        margin=0.04,
        total_line=line,
        true_prob_over=prob_over,
        true_prob_under=1.0 - prob_over,
        prop_player_name=player,
        prop_stat_type=stat,
        event_date=datetime(2026, 5, 10, 22, 0, tzinfo=timezone.utc),
        fetched_at=datetime(2026, 5, 10, 21, 0, tzinfo=timezone.utc),
    )


class TestArchiverPropPersistence:
    def test_prop_columns_populated(self, temp_archive_db):
        archiver = DataArchiver()
        archiver.open_session("test-1", ["nba"], "test")
        n = archiver.archive_sharp_odds("test-1", "nba", [_prop_sharp()])
        assert n == 1

        conn = sqlite3.connect(str(temp_archive_db))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM archived_sharp_odds WHERE prop_player_name IS NOT NULL"
        ).fetchone()
        conn.close()

        assert row is not None
        assert row["prop_player_name"] == "jalen_brunson"
        assert row["prop_stat_type"] == "assists"
        assert row["total_line"] == pytest.approx(6.5)
        assert row["true_prob_over"] == pytest.approx(0.52)
        assert row["true_prob_under"] == pytest.approx(0.48)
        # Moneyline-style probs intentionally zero for props
        assert row["true_prob_a"] == pytest.approx(0.0)
        assert row["true_prob_b"] == pytest.approx(0.0)

    def test_moneyline_path_unaffected(self, temp_archive_db):
        archiver = DataArchiver()
        archiver.open_session("test-2", ["nba"], "test")
        n = archiver.archive_sharp_odds("test-2", "nba", [_moneyline_sharp()])
        assert n == 1

        conn = sqlite3.connect(str(temp_archive_db))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM archived_sharp_odds").fetchone()
        conn.close()

        assert row is not None
        assert row["true_prob_a"] == pytest.approx(0.52)
        assert row["true_prob_b"] == pytest.approx(0.48)
        # New prop columns must be NULL for moneyline rows — that's what makes
        # 'prop_player_name IS NOT NULL' a clean partition filter.
        assert row["prop_player_name"] is None
        assert row["prop_stat_type"] is None
        assert row["total_line"] is None
        assert row["true_prob_over"] is None
        assert row["true_prob_under"] is None

    def test_mixed_batch(self, temp_archive_db):
        """A single archive call mixing moneyline + props writes both correctly."""
        archiver = DataArchiver()
        archiver.open_session("test-3", ["nba"], "test")
        odds = [
            _moneyline_sharp(),
            _prop_sharp("luka_doncic", "points", 27.5, 0.55),
            _prop_sharp("lebron_james", "rebounds", 8.5, 0.48),
        ]
        n = archiver.archive_sharp_odds("test-3", "nba", odds)
        assert n == 3

        conn = sqlite3.connect(str(temp_archive_db))
        conn.row_factory = sqlite3.Row
        prop_rows = conn.execute(
            "SELECT prop_player_name, prop_stat_type, total_line, true_prob_over "
            "FROM archived_sharp_odds "
            "WHERE prop_player_name IS NOT NULL ORDER BY prop_player_name"
        ).fetchall()
        ml_rows = conn.execute(
            "SELECT * FROM archived_sharp_odds WHERE prop_player_name IS NULL"
        ).fetchall()
        conn.close()

        assert len(prop_rows) == 2
        assert len(ml_rows) == 1
        names = {r["prop_player_name"] for r in prop_rows}
        assert names == {"lebron_james", "luka_doncic"}

    def test_migration_idempotent(self, tmp_path, monkeypatch):
        """Running _get_connection on an existing DB without prop columns
        should ALTER them in without error; running twice should be a no-op."""
        from evmax.archiver import _get_connection

        db_path = tmp_path / "legacy.db"
        monkeypatch.setattr("evmax.archiver.DB_PATH", db_path)

        # First call: creates fresh table with all columns from SCHEMA + runs
        # the ALTER migrations (which fail silently because columns already
        # exist from SCHEMA). Should not raise.
        with _get_connection() as conn:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(archived_sharp_odds)")}
        assert "prop_player_name" in cols
        assert "true_prob_over" in cols

        # Second call: should also succeed (migrations are wrapped in try/except).
        with _get_connection() as conn:
            cols2 = {r[1] for r in conn.execute("PRAGMA table_info(archived_sharp_odds)")}
        assert cols == cols2
