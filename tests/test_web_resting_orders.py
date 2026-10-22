"""Dashboard "Resting orders" route — GET /api/resting-orders (record-only)."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from evmax.agents.cleanup.resting_orders import kickoff_time, order_status
from evmax.web.app import app

_SCHEMA = """
CREATE TABLE ev_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_date TEXT NOT NULL, market_id TEXT NOT NULL, event_id TEXT, sector TEXT,
    yes_team TEXT, market_type TEXT, event_title TEXT, line REAL,
    kalshi_yes_price REAL, captured_yes_price REAL, ev_pct REAL, kelly_fraction REAL,
    voided INTEGER NOT NULL DEFAULT 0, placed INTEGER NOT NULL DEFAULT 0,
    placed_at TEXT, placed_price REAL, placed_stake REAL,
    mode TEXT NOT NULL DEFAULT 'live', venue TEXT NOT NULL DEFAULT 'kalshi',
    maker_ev_pct REAL, maker_fill INTEGER NOT NULL DEFAULT 0,
    model_sources TEXT, logged_at TEXT, minutes_to_tipoff INTEGER,
    UNIQUE(market_id)
);
CREATE TABLE ev_outcomes (market_id TEXT UNIQUE, outcome INTEGER);
"""


class _NoCloseConn:
    def __init__(self, inner):
        self._inner = inner

    def close(self):
        pass

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _ts(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _insert(conn, mid, *, logged_ago_min, tip_in_min_at_log, mode="shadow", placed=0,
            maker_fill=0, maker_ev=0.03, sources="spread_pmf+maker_pilot", voided=0):
    now = datetime.now(timezone.utc)
    conn.execute(
        """INSERT INTO ev_predictions
           (scan_date, market_id, sector, yes_team, market_type, event_title, line,
            kalshi_yes_price, captured_yes_price, kelly_fraction, voided, placed,
            placed_price, mode, maker_ev_pct, maker_fill, model_sources, logged_at,
            minutes_to_tipoff)
           VALUES (?, ?, 'nfl', 'Texans', 'spread', 'A vs B', 7.5, 0.4, 0.4, 0.01, ?, ?,
                   ?, ?, ?, ?, ?, ?, ?)""",
        (now.strftime("%Y-%m-%d"), mid, voided, placed, 0.38 if placed else None, mode,
         maker_ev, maker_fill, sources, _ts(now - timedelta(minutes=logged_ago_min)),
         tip_in_min_at_log),
    )
    conn.commit()


@pytest.fixture
def db(monkeypatch):
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    monkeypatch.setattr("evmax.agents.cleanup.db.get_connection", lambda: _NoCloseConn(conn))
    monkeypatch.setattr(
        "evmax.cli.commands.shadow._fetch_clv_rows", lambda *a, **k: ([], 0)
    )
    yield conn
    conn.close()


def _get(sector="nfl"):
    with TestClient(app) as client:
        res = client.get("/api/resting-orders", params={"sector": sector})
    assert res.status_code == 200
    return res.json()


def test_statuses_resting_past_kickoff_and_filled(db):
    _insert(db, "m-rest", logged_ago_min=30, tip_in_min_at_log=300)       # kickoff in ~270m
    _insert(db, "m-past", logged_ago_min=600, tip_in_min_at_log=300)      # kickoff 300m ago
    _insert(db, "m-fill", logged_ago_min=60, tip_in_min_at_log=300,
            mode="live", placed=1, maker_fill=1)
    body = _get()
    by_id = {o["market_id"]: o for o in body["orders"]}
    assert by_id["m-rest"]["status"] == "resting"
    assert by_id["m-past"]["status"] == "past_kickoff"
    assert by_id["m-fill"]["status"] == "filled"
    assert 25 <= by_id["m-rest"]["age_minutes"] <= 35
    assert by_id["m-rest"]["pilot"] is True
    assert by_id["m-rest"]["kickoff_at"] is not None


def test_excludes_voided_resolved_and_non_maker_rows(db):
    _insert(db, "m-void", logged_ago_min=10, tip_in_min_at_log=300, voided=1)
    _insert(db, "m-done", logged_ago_min=10, tip_in_min_at_log=300)
    db.execute("INSERT INTO ev_outcomes VALUES ('m-done', 1)")
    _insert(db, "m-taker", logged_ago_min=10, tip_in_min_at_log=300, maker_ev=None, sources="spread_pmf")
    _insert(db, "m-live-unfilled", logged_ago_min=10, tip_in_min_at_log=300, mode="live")
    assert _get()["orders"] == []


def test_unknown_kickoff_is_resting_not_warned(db):
    _insert(db, "m-nokick", logged_ago_min=10, tip_in_min_at_log=None)
    o = _get()["orders"][0]
    assert o["kickoff_at"] is None
    assert o["status"] == "resting"


def test_readout_collecting_when_no_filled_games(db):
    r = _get()["readout"]
    assert r["verdict"] == "COLLECTING"
    assert r["games"] == 0
    assert r["checkpoint_games"] == 15 and r["gate_games"] == 30
    assert r["fill_rate"] is None


def test_readout_scores_filled_rows_net_of_maker_fee(db, monkeypatch):
    rows = [
        {"event_id": f"G{i}::spread", "kalshi_clv_pct": 2.0, "placed_price": 0.4, "venue": "kalshi"}
        for i in range(15)
    ]
    monkeypatch.setattr("evmax.cli.commands.shadow._fetch_clv_rows", lambda *a, **k: (rows, 1))
    r = _get()["readout"]
    assert r["games"] == 15
    assert r["verdict"] in {"STEP-UP", "HOLD", "STOP"}
    assert r["mean_net_pp"] < 2.0  # maker fee is subtracted
    assert r["stale_close_excluded"] == 1


def test_helpers():
    now = datetime(2026, 10, 18, 18, 0, tzinfo=timezone.utc)
    k = kickoff_time("2026-10-18 12:00:00", 360)
    assert k == now
    assert order_status(0, k, now) == "past_kickoff"
    assert order_status(0, k, now - timedelta(minutes=1)) == "resting"
    assert order_status(1, k, now) == "filled"
    assert kickoff_time(None, 10) is None
