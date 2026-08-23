"""POST /api/pick — server-side per-venue cash cap.

The scan→ScanResults flow already caps stakes client-side, but Open-Positions
'Pick All' places from stored rows that bypass it. /api/pick caps the batch to
each venue's deployable cash server-side (fail-soft), so a placed bet can't
exceed a venue's cash.
"""

from __future__ import annotations

import sqlite3
from unittest.mock import patch

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


def _insert_live(conn, market_id, venue="kalshi"):
    conn.execute(
        """INSERT INTO ev_predictions
           (scan_date, market_id, event_id, sector, yes_team, market_type,
            event_title, kalshi_yes_price, kelly_fraction, bankroll_used,
            voided, placed, mode, venue)
           VALUES ('2026-03-21', ?, 'evt:1', 'nba', 'celtics', 'moneyline',
                   'Celtics vs Heat', 0.50, 0.05, 500.0, 0, 0, 'live', ?)""",
        (market_id, venue),
    )
    conn.commit()


@pytest.fixture
def db(monkeypatch):
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    monkeypatch.setattr(
        "evmax.agents.cleanup.db.get_connection", lambda: _NoCloseConn(conn)
    )
    yield conn
    conn.close()


def _stake(conn, market_id):
    return conn.execute(
        "SELECT placed_stake FROM ev_predictions WHERE market_id = ?", (market_id,)
    ).fetchone()["placed_stake"]


def _fake_cash(mapping):
    async def _f(venues):
        return dict(mapping)
    return _f


def _pick(bets):
    with TestClient(app) as client:
        return client.post("/api/pick", json={"bets": bets}).json()


def test_caps_venue_over_cash(db):
    """Two kalshi picks summing to $60 > $40 cash → each scaled by 40/60."""
    _insert_live(db, "kalshi:A")
    _insert_live(db, "kalshi:B")
    with patch("evmax.clients.balances.fetch_cash_balances", _fake_cash({"kalshi": 40.0})):
        body = _pick([
            {"market_id": "kalshi:A", "fill_price": 0.5, "fill_stake": 30.0},
            {"market_id": "kalshi:B", "fill_price": 0.5, "fill_stake": 30.0},
        ])
    assert body["placed"] == 2
    assert body["cash_capped"]["kalshi"] == pytest.approx(40.0 / 60.0, abs=1e-3)
    assert _stake(db, "kalshi:A") == pytest.approx(20.0)
    assert _stake(db, "kalshi:B") == pytest.approx(20.0)


def test_no_cap_when_under_cash(db):
    _insert_live(db, "kalshi:A")
    with patch("evmax.clients.balances.fetch_cash_balances", _fake_cash({"kalshi": 1000.0})):
        body = _pick([{"market_id": "kalshi:A", "fill_price": 0.5, "fill_stake": 30.0}])
    assert "cash_capped" not in body
    assert _stake(db, "kalshi:A") == pytest.approx(30.0)


def test_failsoft_when_cash_unavailable(db):
    """No creds / balance None → no cap, stake recorded as-is."""
    _insert_live(db, "kalshi:A")
    with patch("evmax.clients.balances.fetch_cash_balances", _fake_cash({"kalshi": None})):
        body = _pick([{"market_id": "kalshi:A", "fill_price": 0.5, "fill_stake": 30.0}])
    assert "cash_capped" not in body
    assert _stake(db, "kalshi:A") == pytest.approx(30.0)


def test_failsoft_when_fetch_raises(db):
    _insert_live(db, "kalshi:A")
    async def _boom(venues):
        raise RuntimeError("balance host down")
    with patch("evmax.clients.balances.fetch_cash_balances", _boom):
        body = _pick([{"market_id": "kalshi:A", "fill_price": 0.5, "fill_stake": 30.0}])
    assert "cash_capped" not in body
    assert _stake(db, "kalshi:A") == pytest.approx(30.0)


def test_caps_each_venue_independently(db):
    """Kalshi under its cash stays; PolyUS over its cash scales."""
    _insert_live(db, "kalshi:A", venue="kalshi")
    _insert_live(db, "polymarket_us:A", venue="polymarket_us")
    with patch("evmax.clients.balances.fetch_cash_balances",
               _fake_cash({"kalshi": 1000.0, "polymarket_us": 10.0})):
        body = _pick([
            {"market_id": "kalshi:A", "fill_price": 0.5, "fill_stake": 30.0},
            {"market_id": "polymarket_us:A", "fill_price": 0.5, "fill_stake": 30.0},
        ])
    assert _stake(db, "kalshi:A") == pytest.approx(30.0)         # under cash
    assert _stake(db, "polymarket_us:A") == pytest.approx(10.0)  # $30 → $10
    assert set(body["cash_capped"].keys()) == {"polymarket_us"}
