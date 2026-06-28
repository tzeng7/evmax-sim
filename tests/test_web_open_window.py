"""Open Positions is scoped to a FORWARD rolling window — only games on or
after today are surfaced, so past unplaced/unresolved markets don't pile up in
the panel. Plus unit coverage for the persistence date-window predicate that
keeps a ranged scan from writing out-of-range rows. See
evmax.web.app._open_bets and _gap_in_scan_window.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

import evmax.agents.cleanup.db as dbmod
from evmax.web.app import _gap_in_scan_window, _open_bets


def _seed_open(conn, *, market_id, sector, event_date, market_type="moneyline"):
    """Insert an unplaced, unresolved, live ev_predictions row (no outcome)."""
    conn.execute(
        """
        INSERT INTO ev_predictions
            (scan_date, market_id, event_id, sector, yes_team, market_type,
             event_title, event_date, kalshi_yes_price, sharp_true_prob,
             blended_true_prob, ev_pct, kelly_fraction, bankroll_used,
             mode, voided, placed)
        VALUES (?, ?, ?, ?, 'A', ?, 'A vs B', ?, 0.5, 0.55, 0.56, 0.06,
                0.03, 250.0, 'live', 0, 0)
        """,
        (event_date, market_id, market_id + "_evt", sector, market_type,
         event_date),
    )


@pytest.fixture
def open_db(tmp_path, monkeypatch):
    db_path = tmp_path / "predictions.db"
    monkeypatch.setattr(dbmod, "DB_PATH", db_path)
    conn = dbmod.get_connection()
    today = date.today()
    _seed_open(conn, market_id="past", sector="nba",
               event_date=(today - timedelta(days=2)).isoformat())
    _seed_open(conn, market_id="today", sector="nba",
               event_date=today.isoformat())
    _seed_open(conn, market_id="future", sector="nba",
               event_date=(today + timedelta(days=3)).isoformat())
    _seed_open(conn, market_id="dateless", sector="nba", event_date="")
    conn.commit()
    conn.close()
    return db_path


def test_open_positions_hides_past_keeps_today_and_future(open_db):
    mids = {b["market_id"] for b in _open_bets()}
    assert "today" in mids
    assert "future" in mids
    assert "past" not in mids


def test_open_positions_keeps_dateless_rows(open_db):
    mids = {b["market_id"] for b in _open_bets()}
    # Rare dateless markets shouldn't be silently dropped by the window.
    assert "dateless" in mids


# ---- _gap_in_scan_window unit coverage ----

def _dt(d: date) -> datetime:
    # tz-aware UTC so kalshi_game_day's non-US branch formats the same day.
    return datetime(d.year, d.month, d.day, 12, 0, tzinfo=timezone.utc)


def test_window_explicit_range_inclusive():
    today = date.today()
    assert _gap_in_scan_window(_dt(today), "tennis",
                               today.isoformat(), today.isoformat())
    assert not _gap_in_scan_window(_dt(today + timedelta(days=1)), "tennis",
                                   today.isoformat(), today.isoformat())


def test_window_open_ended_bounds():
    today = date.today()
    # only date_from → everything on/after it
    assert _gap_in_scan_window(_dt(today + timedelta(days=5)), "tennis",
                               today.isoformat(), "")
    assert not _gap_in_scan_window(_dt(today - timedelta(days=1)), "tennis",
                                   today.isoformat(), "")
    # only date_to → everything on/before it
    assert _gap_in_scan_window(_dt(today - timedelta(days=1)), "tennis",
                               "", today.isoformat())
    assert not _gap_in_scan_window(_dt(today + timedelta(days=1)), "tennis",
                                   "", today.isoformat())


def test_window_empty_range_defaults_to_today_tomorrow():
    today = date.today()
    assert _gap_in_scan_window(_dt(today), "tennis", "", "")
    assert _gap_in_scan_window(_dt(today + timedelta(days=1)), "tennis", "", "")
    assert not _gap_in_scan_window(_dt(today + timedelta(days=2)), "tennis", "", "")
    assert not _gap_in_scan_window(_dt(today - timedelta(days=1)), "tennis", "", "")


def test_window_dateless_gap_kept():
    # None event_date → preserve prior persist-all behavior for dateless markets.
    assert _gap_in_scan_window(None, "tennis", date.today().isoformat(), "")
