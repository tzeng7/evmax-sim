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


class TestDerivedPropRungs:
    """2026-10-09: the scanner re-lines ONE Pinnacle prop anchor to every
    Kalshi threshold. Those rungs are evmax's model output, and were archived
    as book='pinnacle' — indistinguishable from real Pinnacle alt lines (the
    tell was identical decimals on every rung). They now go under
    book='pinnacle_derived' with the real anchor alongside."""

    def _derived(self, threshold: float = 30.0) -> SharpOdds:
        return _prop_sharp("luka_doncic", "points", threshold, 0.21).model_copy(
            update={"derived": True, "anchor_line": 27.5, "anchor_prob_over": 0.55}
        )

    def test_book_label(self):
        from evmax.archiver import archive_book_label

        assert archive_book_label(_prop_sharp()) == "pinnacle"
        assert archive_book_label(self._derived()) == "pinnacle_derived"
        assert archive_book_label(_moneyline_sharp()) == "pinnacle"

    def test_derived_rung_archived_under_distinct_book(self, temp_archive_db):
        archiver = DataArchiver()
        archiver.open_session("d-1", ["nba"], "test")
        anchor = _prop_sharp("luka_doncic", "points", 27.5, 0.55)
        archiver.archive_sharp_odds("d-1", "nba", [self._derived(), anchor, _moneyline_sharp()])

        conn = sqlite3.connect(str(temp_archive_db))
        conn.row_factory = sqlite3.Row
        rows = {
            (r["book"], r["total_line"]): r
            for r in conn.execute("SELECT * FROM archived_sharp_odds")
        }
        conn.close()

        derived = rows[("pinnacle_derived", 30.0)]
        assert derived["derived"] == 1
        assert derived["anchor_line"] == pytest.approx(27.5)
        assert derived["anchor_prob_over"] == pytest.approx(0.55)
        assert derived["true_prob_over"] == pytest.approx(0.21)

        quote = rows[("pinnacle", 27.5)]
        assert quote["derived"] == 0
        assert quote["anchor_line"] is None
        assert quote["anchor_prob_over"] is None

        ml = rows[("pinnacle", None)]
        assert ml["derived"] == 0

    def test_book_filter_returns_only_posted_lines(self, temp_archive_db):
        """The query a backtest would write must not see model output."""
        archiver = DataArchiver()
        archiver.open_session("d-2", ["nba"], "test")
        archiver.archive_sharp_odds("d-2", "nba", [
            self._derived(25.0), self._derived(30.0), self._derived(35.0),
            _prop_sharp("luka_doncic", "points", 27.5, 0.55),
        ])
        conn = sqlite3.connect(str(temp_archive_db))
        lines = [r[0] for r in conn.execute(
            "SELECT total_line FROM archived_sharp_odds "
            "WHERE book='pinnacle' AND prop_player_name IS NOT NULL"
        )]
        conn.close()
        assert lines == [27.5]

    def test_derived_and_quote_at_same_event_id_coexist(self, temp_archive_db):
        """UNIQUE(session_id, event_id, book): a rung whose threshold equals
        the anchor line shares its event_id; the distinct book keeps both."""
        archiver = DataArchiver()
        archiver.open_session("d-3", ["nba"], "test")
        anchor = _prop_sharp("luka_doncic", "points", 27.5, 0.55)
        rung = self._derived(27.5)
        assert rung.event_id == anchor.event_id
        archiver.archive_sharp_odds("d-3", "nba", [anchor, rung])
        conn = sqlite3.connect(str(temp_archive_db))
        n = conn.execute("SELECT COUNT(*) FROM archived_sharp_odds").fetchone()[0]
        conn.close()
        assert n == 2

    def test_legacy_db_gains_columns_and_keeps_rows_null(self, tmp_path, monkeypatch):
        """An archive created before the change (no derived/anchor columns)
        is migrated in place; its existing rows read derived IS NULL, which is
        how scripts/relabel_derived_prop_rungs.py finds them."""
        db_path = tmp_path / "legacy_props.db"
        monkeypatch.setattr("evmax.archiver.DB_PATH", db_path)
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """CREATE TABLE archived_sharp_odds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
                    fetched_at TEXT NOT NULL, sector TEXT NOT NULL,
                    event_id TEXT NOT NULL, book TEXT NOT NULL,
                    outcome_a_label TEXT, outcome_b_label TEXT,
                    outcome_a_decimal REAL NOT NULL, outcome_b_decimal REAL NOT NULL,
                    outcome_draw_decimal REAL, true_prob_a REAL NOT NULL,
                    true_prob_b REAL NOT NULL, true_prob_draw REAL,
                    margin REAL NOT NULL, spread_line REAL, event_date TEXT,
                    true_prob_over REAL, true_prob_under REAL, total_line REAL,
                    prop_player_name TEXT, prop_stat_type TEXT,
                    UNIQUE(session_id, event_id, book))"""
            )
            conn.execute(
                "INSERT INTO archived_sharp_odds (session_id, fetched_at, sector, "
                "event_id, book, outcome_a_decimal, outcome_b_decimal, true_prob_a, "
                "true_prob_b, margin, total_line, prop_player_name, prop_stat_type) "
                "VALUES ('old', '2026-10-01', 'nfl', 'nfl::x::prop::p::receiving_yards::90.0', "
                "'pinnacle', 1.91, 1.91, 0, 0, 0.04, 90.0, 'p', 'receiving_yards')"
            )

        archiver = DataArchiver()
        archiver.open_session("new", ["nba"], "test")
        archiver.archive_sharp_odds("new", "nba", [self._derived()])

        with sqlite3.connect(db_path) as conn:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(archived_sharp_odds)")}
            rows = dict(conn.execute("SELECT session_id, derived FROM archived_sharp_odds"))
        assert {"derived", "anchor_line", "anchor_prob_over"} <= cols
        assert rows == {"old": None, "new": 1}
