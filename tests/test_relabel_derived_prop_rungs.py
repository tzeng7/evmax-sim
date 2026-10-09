"""Tests for scripts/relabel_derived_prop_rungs.py (temp archive DBs only)."""
from __future__ import annotations

import importlib.util
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from evmax.archiver import DataArchiver
from evmax.models.odds import SharpBook, SharpOdds

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "relabel_derived_prop_rungs.py"
_spec = importlib.util.spec_from_file_location("relabel_derived_prop_rungs", _SCRIPT)
relabel = importlib.util.module_from_spec(_spec)
sys.modules["relabel_derived_prop_rungs"] = relabel
_spec.loader.exec_module(relabel)

_LEGACY_SCHEMA = """CREATE TABLE archived_sharp_odds (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
    fetched_at TEXT NOT NULL, sector TEXT NOT NULL, event_id TEXT NOT NULL,
    book TEXT NOT NULL, outcome_a_label TEXT, outcome_b_label TEXT,
    outcome_a_decimal REAL NOT NULL, outcome_b_decimal REAL NOT NULL,
    outcome_draw_decimal REAL, true_prob_a REAL NOT NULL, true_prob_b REAL NOT NULL,
    true_prob_draw REAL, margin REAL NOT NULL, spread_line REAL, event_date TEXT,
    true_prob_over REAL, true_prob_under REAL, total_line REAL,
    prop_player_name TEXT, prop_stat_type TEXT,
    UNIQUE(session_id, event_id, book))"""


def _insert_legacy(conn, session, event_id, player=None, line=None, sector="nfl"):
    conn.execute(
        "INSERT INTO archived_sharp_odds (session_id, fetched_at, sector, event_id, "
        "book, outcome_a_decimal, outcome_b_decimal, true_prob_a, true_prob_b, margin, "
        "total_line, prop_player_name, prop_stat_type) "
        "VALUES (?, '2026-10-04T16:00:00', ?, ?, 'pinnacle', 1.91, 1.91, ?, ?, 0.04, ?, ?, ?)",
        (session, sector, event_id, 0.0 if player else 0.5, 0.0 if player else 0.5,
         line, player, "receiving_yards" if player else None),
    )


def _prop(line: float, derived: bool) -> SharpOdds:
    odds = SharpOdds(
        event_id=f"nfl::2026-10-11::prop::ja'marr_chase::receiving_yards::{line}",
        book=SharpBook.pinnacle,
        sector="nfl",
        outcome_a_label="over",
        outcome_b_label="under",
        outcome_a_decimal=1.91,
        outcome_b_decimal=1.91,
        margin=0.04,
        total_line=line,
        true_prob_over=0.5,
        true_prob_under=0.5,
        prop_player_name="ja'marr_chase",
        prop_stat_type="receiving_yards",
        event_date=datetime(2026, 10, 11, 17, tzinfo=timezone.utc),
    )
    if derived:
        odds = odds.model_copy(update={"derived": True, "anchor_line": 83.5,
                                       "anchor_prob_over": 0.5})
    return odds


@pytest.fixture
def legacy_db(tmp_path, monkeypatch):
    """A pre-change archive: three legacy prop rungs + one game row, then a
    post-change scan (real anchor + derived rung) written by DataArchiver."""
    db = tmp_path / "archive.db"
    with sqlite3.connect(db) as conn:
        conn.execute(_LEGACY_SCHEMA)
        for k in (80.0, 90.0, 100.0):
            _insert_legacy(conn, "old", f"nfl::2026-10-04::prop::p::receiving_yards::{k}",
                           player="p", line=k)
        _insert_legacy(conn, "old", "nfl::2026-10-04::a_vs_b")
    monkeypatch.setattr("evmax.archiver.DB_PATH", db)
    archiver = DataArchiver()
    archiver.open_session("new", ["nfl"], "test")
    archiver.archive_sharp_odds("new", "nfl", [_prop(83.5, False), _prop(90.0, True)])
    return db


def _rows(db):
    with sqlite3.connect(db) as conn:
        return sorted(conn.execute(
            "SELECT session_id, COALESCE(total_line, -1), book, derived FROM archived_sharp_odds"
        ))


def test_dry_run_counts_only_legacy_prop_rows(legacy_db):
    before = _rows(legacy_db)
    assert relabel.main(["--db", str(legacy_db)]) == 0
    assert _rows(legacy_db) == before
    with sqlite3.connect(legacy_db) as conn:
        assert relabel.plan(conn) == [("nfl", "pinnacle", 3)]


def test_apply_relabels_legacy_rungs_and_nothing_else(legacy_db):
    assert relabel.main(["--db", str(legacy_db), "--apply", "--batch", "2"]) == 0
    assert _rows(legacy_db) == [
        ("new", 83.5, "pinnacle", 0),            # real anchor untouched
        ("new", 90.0, "pinnacle_derived", 1),    # new-code rung untouched
        ("old", -1, "pinnacle", None),           # game row untouched
        ("old", 80.0, "pinnacle_derived", 1),
        ("old", 90.0, "pinnacle_derived", 1),
        ("old", 100.0, "pinnacle_derived", 1),
    ]


def test_apply_is_idempotent(legacy_db):
    relabel.main(["--db", str(legacy_db), "--apply"])
    once = _rows(legacy_db)
    with sqlite3.connect(legacy_db) as conn:
        assert relabel.plan(conn) == []
        assert relabel.apply(conn) == 0
    assert _rows(legacy_db) == once


def test_revert_restores_only_script_relabelled_rows(legacy_db):
    before = _rows(legacy_db)
    relabel.main(["--db", str(legacy_db), "--apply"])
    assert relabel.main(["--db", str(legacy_db), "--apply", "--revert"]) == 0
    assert _rows(legacy_db) == before


def test_pre_migration_archive_is_planned_and_migrated(tmp_path):
    """An archive the new archiver never opened lacks the derived columns:
    the dry-run still plans every prop row, --apply adds the columns."""
    db = tmp_path / "old_only.db"
    with sqlite3.connect(db) as conn:
        conn.execute(_LEGACY_SCHEMA)
        _insert_legacy(conn, "old", "nfl::d::prop::p::receiving_yards::90.0", player="p", line=90.0)
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        assert not relabel.has_derived_columns(conn)
        assert relabel.plan(conn) == [("nfl", "pinnacle", 1)]
    assert relabel.main(["--db", str(db), "--apply"]) == 0
    assert _rows(db) == [("old", 90.0, "pinnacle_derived", 1)]


def test_missing_db_returns_error(tmp_path):
    assert relabel.main(["--db", str(tmp_path / "nope.db")]) == 1
