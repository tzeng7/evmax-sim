"""Regression test for the missing archived_sharp_odds(event_id) index.

Every per-event closing-line lookup (``get_line_history``,
``get_kalshi_close_price`` / ``get_kalshi_close_staleness_h`` via
``_select_kalshi_close``, ``get_closing_line_aligned``) filters
``archived_sharp_odds`` by ``event_id`` alone. Without an index on that column
sqlite falls back to a full table scan — fine at small scale, but the
promotion board's CLV staleness filter (``evmax cleanup shadow board`` /
``GET /api/promotion-board``) calls one of these helpers once per resolved
Kalshi row across *all* history for a sector/market_type, so the missing
index turned a sub-2s endpoint into a 110s+ one against the production
archive.db (2026-07-19 diagnosis). The fix adds ``idx_sharp_event_id`` to the
schema; this test locks in that the index exists and that sqlite actually
uses it for the lookup shape these helpers issue.
"""
from __future__ import annotations

import sqlite3

import pytest

from evmax.archiver import DataArchiver, _get_connection


@pytest.fixture
def temp_archive_db(tmp_path, monkeypatch):
    db_path = tmp_path / "archive_index_test.db"
    monkeypatch.setattr("evmax.archiver.DB_PATH", db_path)
    return db_path


def test_archived_sharp_odds_has_event_id_index(temp_archive_db):
    DataArchiver()  # trigger schema creation
    conn = _get_connection()
    try:
        indices = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='archived_sharp_odds'"
            ).fetchall()
        }
    finally:
        conn.close()
    assert "idx_sharp_event_id" in indices


def test_event_id_lookup_uses_index_not_full_scan(temp_archive_db):
    DataArchiver()  # trigger schema creation
    conn = _get_connection()
    try:
        # Same query shape as DataArchiver._select_kalshi_close /
        # get_closing_line_aligned: WHERE event_id = ? ... ORDER BY fetched_at DESC.
        plan = conn.execute(
            """EXPLAIN QUERY PLAN
               SELECT event_date FROM archived_sharp_odds
               WHERE event_id = ? AND event_date IS NOT NULL
               ORDER BY fetched_at DESC LIMIT 1""",
            ("some-event",),
        ).fetchall()
    finally:
        conn.close()
    plan_text = " ".join(row["detail"] for row in plan)
    assert "USING INDEX idx_sharp_event_id" in plan_text
    assert "SCAN archived_sharp_odds" not in plan_text
