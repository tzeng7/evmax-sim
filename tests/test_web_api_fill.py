"""Dashboard maker-fill endpoint — POST /api/fill.

Regression cover for the gap this closed: a maker-only play logs as
mode='shadow' (it clears the EV floor only as a resting limit order, so it is
not crossable at the ask). /api/pick correctly refuses every non-live row, which
left the dashboard with NO way to record such a position at all — the only path
was the `evmax agents fill` CLI. /api/fill is that path in the dashboard.
"""

from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

from evmax.web.app import app

_SCHEMA = """
CREATE TABLE ev_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_date TEXT NOT NULL,
    market_id TEXT NOT NULL,
    event_id TEXT,
    sector TEXT,
    yes_team TEXT,
    market_type TEXT,
    event_title TEXT,
    event_date TEXT,
    kalshi_yes_price REAL,
    sharp_true_prob REAL,
    blended_true_prob REAL,
    ev_pct REAL,
    kelly_fraction REAL,
    bankroll_used REAL,
    line REAL,
    voided INTEGER NOT NULL DEFAULT 0,
    placed INTEGER NOT NULL DEFAULT 0,
    placed_at TEXT,
    placed_price REAL,
    placed_stake REAL,
    mode TEXT NOT NULL DEFAULT 'live',
    venue TEXT NOT NULL DEFAULT 'kalshi',
    maker_ev_pct REAL,
    maker_fill INTEGER NOT NULL DEFAULT 0,
    UNIQUE(market_id)
);
"""


class _NoCloseConn:
    """Keep the in-memory DB alive across the endpoint's conn.close()."""

    def __init__(self, inner):
        self._inner = inner

    def close(self):
        pass

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _insert(conn, market_id, *, mode="shadow", placed=0, voided=0):
    conn.execute(
        """INSERT INTO ev_predictions
           (scan_date, market_id, event_id, sector, yes_team, market_type,
            event_title, kalshi_yes_price, sharp_true_prob, blended_true_prob,
            ev_pct, kelly_fraction, bankroll_used, voided, placed, mode, venue,
            maker_ev_pct)
           VALUES ('2026-03-21', ?, 'evt:1', 'nba', 'celtics', 'moneyline',
                   'Celtics vs Heat', 0.50, 0.525, 0.525, 0.014, 0.0, 500.0,
                   ?, ?, ?, 'kalshi', 0.041)""",
        (market_id, voided, placed, mode),
    )
    conn.commit()


@pytest.fixture
def db(monkeypatch):
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    _insert(conn, "kalshi:MAKER-BOS")
    monkeypatch.setattr(
        "evmax.agents.cleanup.db.get_connection", lambda: _NoCloseConn(conn)
    )
    yield conn
    conn.close()


def _row(conn, market_id):
    return conn.execute(
        "SELECT * FROM ev_predictions WHERE market_id = ?", (market_id,)
    ).fetchone()


def test_pick_still_refuses_a_shadow_maker_row(db):
    """The taker path must stay closed — this is correct, not the bug."""
    with TestClient(app) as client:
        res = client.post(
            "/api/pick",
            json={"bets": [{"market_id": "kalshi:MAKER-BOS", "fill_price": 0.5, "fill_stake": 10}]},
        )
    assert res.status_code == 200
    body = res.json()
    assert body["placed"] == 0
    assert "mode=shadow" in body["skipped"][0]["reason"]
    assert _row(db, "kalshi:MAKER-BOS")["placed"] == 0


def test_fill_records_the_shadow_maker_row_as_a_live_position(db):
    with TestClient(app) as client:
        res = client.post(
            "/api/fill",
            json={"fills": [
                {"market_id": "kalshi:MAKER-BOS", "fill_price": 0.47, "fill_stake": 25.0},
            ]},
        )
    assert res.status_code == 200
    body = res.json()
    assert body["filled"] == 1
    assert body["skipped"] == []
    assert body["fills"][0]["maker_fee"] >= 0
    assert body["fills"][0]["prior_mode"] == "shadow"

    row = _row(db, "kalshi:MAKER-BOS")
    assert row["placed"] == 1
    assert row["mode"] == "live"
    assert row["maker_fill"] == 1
    assert row["placed_price"] == pytest.approx(0.47)
    assert row["placed_stake"] == pytest.approx(25.0)


def test_fill_accepts_cents(db):
    with TestClient(app) as client:
        res = client.post(
            "/api/fill",
            json={"fills": [{"market_id": "kalshi:MAKER-BOS", "fill_price": 47, "fill_stake": 25.0}]},
        )
    assert res.json()["filled"] == 1
    assert _row(db, "kalshi:MAKER-BOS")["placed_price"] == pytest.approx(0.47)


def test_fill_is_partial_success_not_all_or_nothing(db):
    _insert(db, "kalshi:ALREADY", mode="live", placed=1)
    with TestClient(app) as client:
        res = client.post(
            "/api/fill",
            json={"fills": [
                {"market_id": "kalshi:ALREADY", "fill_price": 0.47, "fill_stake": 25.0},
                {"market_id": "kalshi:MAKER-BOS", "fill_price": 0.47, "fill_stake": 25.0},
            ]},
        )
    body = res.json()
    # One bad selection must not discard the good fill next to it.
    assert body["filled"] == 1
    assert len(body["skipped"]) == 1
    assert body["skipped"][0]["market_id"] == "kalshi:ALREADY"
    assert _row(db, "kalshi:MAKER-BOS")["placed"] == 1


def test_fill_reports_a_reason_per_bad_row(db):
    _insert(db, "kalshi:VOIDED", voided=1)
    with TestClient(app) as client:
        res = client.post(
            "/api/fill",
            json={"fills": [
                {"market_id": "kalshi:VOIDED", "fill_price": 0.47, "fill_stake": 25.0},
                {"market_id": "kalshi:GHOST", "fill_price": 0.47, "fill_stake": 25.0},
                {"market_id": "kalshi:MAKER-BOS", "fill_price": 0.47, "fill_stake": 0},
            ]},
        )
    body = res.json()
    assert body["filled"] == 0
    reasons = {s["market_id"]: s["reason"] for s in body["skipped"]}
    assert "voided" in reasons["kalshi:VOIDED"]
    assert "No prediction row" in reasons["kalshi:GHOST"]
    assert "stake" in reasons["kalshi:MAKER-BOS"]
    assert _row(db, "kalshi:MAKER-BOS")["placed"] == 0


def test_fill_requires_a_body(db):
    with TestClient(app) as client:
        res = client.post("/api/fill", json={"fills": []})
    assert res.status_code == 400


# --- the rest of the loop: edit the recorded fill, then undo it -------------

def test_recorded_fill_can_be_corrected_afterwards(db):
    """Wrong odds/size entered? Placed Bets edits them via /api/update-placed."""
    with TestClient(app) as client:
        client.post(
            "/api/fill",
            json={"fills": [{"market_id": "kalshi:MAKER-BOS", "fill_price": 0.47, "fill_stake": 25.0}]},
        )
        res = client.post(
            "/api/update-placed",
            json={"edits": [{"market_id": "kalshi:MAKER-BOS", "fill_price": 0.44, "fill_stake": 40.0}]},
        )
    assert res.json()["updated"] == 1
    row = _row(db, "kalshi:MAKER-BOS")
    assert row["placed_price"] == pytest.approx(0.44)
    assert row["placed_stake"] == pytest.approx(40.0)
    # Correcting the numbers must not silently drop the maker-fee flag.
    assert row["maker_fill"] == 1


def test_unplacing_a_maker_fill_reverts_the_whole_promotion(db):
    """Undo must clear maker_fill AND mode, not just the placement.

    _bet_pnl charges the maker fee off maker_fill, so a stale flag would price a
    later TAKER pick at the maker rate. And a maker-only row left at mode='live'
    becomes taker-pickable at an ask that was never +EV.
    """
    with TestClient(app) as client:
        client.post(
            "/api/fill",
            json={"fills": [{"market_id": "kalshi:MAKER-BOS", "fill_price": 0.47, "fill_stake": 25.0}]},
        )
        res = client.post("/api/unplace", json={"market_ids": ["kalshi:MAKER-BOS"]})
    assert res.json()["removed"] == 1

    row = _row(db, "kalshi:MAKER-BOS")
    assert row["placed"] == 0
    assert row["placed_price"] is None
    assert row["placed_stake"] is None
    assert row["maker_fill"] == 0
    assert row["mode"] == "shadow"


def test_unplacing_a_taker_bet_leaves_its_live_mode_alone(db):
    """The maker revert must not demote an ordinary taker pick to shadow."""
    _insert(db, "kalshi:TAKER-LAL", mode="live")
    with TestClient(app) as client:
        client.post(
            "/api/pick",
            json={"bets": [{"market_id": "kalshi:TAKER-LAL", "fill_price": 0.45, "fill_stake": 10.0}]},
        )
        client.post("/api/unplace", json={"market_ids": ["kalshi:TAKER-LAL"]})
    row = _row(db, "kalshi:TAKER-LAL")
    assert row["placed"] == 0
    assert row["mode"] == "live"
    assert row["maker_fill"] == 0


def test_a_maker_row_can_be_re_recorded_after_an_undo(db):
    """Undo then re-record — the round trip must not stick."""
    with TestClient(app) as client:
        client.post(
            "/api/fill",
            json={"fills": [{"market_id": "kalshi:MAKER-BOS", "fill_price": 0.47, "fill_stake": 25.0}]},
        )
        client.post("/api/unplace", json={"market_ids": ["kalshi:MAKER-BOS"]})
        res = client.post(
            "/api/fill",
            json={"fills": [{"market_id": "kalshi:MAKER-BOS", "fill_price": 0.49, "fill_stake": 12.0}]},
        )
    assert res.json()["filled"] == 1
    row = _row(db, "kalshi:MAKER-BOS")
    assert row["placed"] == 1
    assert row["maker_fill"] == 1
    assert row["placed_price"] == pytest.approx(0.49)
    assert row["placed_stake"] == pytest.approx(12.0)
