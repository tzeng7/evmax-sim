"""POST /api/verify — dashboard live re-price (mirrors `agents verify`/`pick --live`)."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from evmax.web import verify as verify_mod
from evmax.web.app import app

_SCHEMA = """
CREATE TABLE ev_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_date TEXT NOT NULL, market_id TEXT NOT NULL, event_id TEXT,
    sector TEXT, yes_team TEXT, market_type TEXT, event_title TEXT,
    kalshi_yes_price REAL, sharp_true_prob REAL, blended_true_prob REAL,
    ev_pct REAL, kelly_fraction REAL, bankroll_used REAL, line REAL,
    sharp_weight_used REAL,
    voided INTEGER NOT NULL DEFAULT 0, placed INTEGER NOT NULL DEFAULT 0,
    mode TEXT NOT NULL DEFAULT 'live', venue TEXT NOT NULL DEFAULT 'kalshi'
);
"""


class _NoCloseConn:
    def __init__(self, inner):
        self._inner = inner

    def close(self):
        pass

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _insert(conn, mid, *, venue="kalshi", mtype="moneyline", ask=0.40, fair=0.46,
            placed=0, voided=0, mode="live"):
    conn.execute(
        """INSERT INTO ev_predictions
           (scan_date, market_id, event_id, sector, yes_team, market_type,
            event_title, kalshi_yes_price, sharp_true_prob, blended_true_prob,
            ev_pct, kelly_fraction, bankroll_used, sharp_weight_used,
            voided, placed, mode, venue)
           VALUES ('2026-10-21', ?, ?, 'nba', 'Celtics', ?, 'Celtics vs Heat',
                   ?, ?, ?, 0.15, 0.02, 500.0, 0.85, ?, ?, ?, ?)""",
        (mid, f"evt:{mid}", mtype, ask, fair, fair, voided, placed, mode, venue),
    )
    conn.commit()


class _FakeKalshi:
    asks: dict = {}
    fail = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get_market_asks_batch(self, tickers):
        if self.fail:
            raise RuntimeError("ws down")
        return {t: self.asks.get(t) for t in tickers}


class _FakePinnacle:
    odds: list = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get_odds(self, sector):
        return list(self.odds)


@pytest.fixture
def env(monkeypatch):
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    monkeypatch.setattr("evmax.agents.cleanup.db.get_connection", lambda: _NoCloseConn(conn))
    _FakeKalshi.asks, _FakeKalshi.fail = {}, False
    _FakePinnacle.odds = []
    monkeypatch.setattr("evmax.clients.kalshi.KalshiClient", _FakeKalshi)
    monkeypatch.setattr("evmax.clients.esports_pinnacle.PinnacleGuestClient", _FakePinnacle)

    async def _no_poly(rows):
        return {}

    monkeypatch.setattr("evmax.agents.cleanup.reprice.fetch_polymarket_live_asks", _no_poly)
    yield conn
    conn.close()


def _post(ids, **extra):
    with TestClient(app) as c:
        return c.post("/api/verify", json={"market_ids": ids, **extra})


def _by_id(res):
    return {r["market_id"]: r for r in res.json()["results"]}


def test_requires_market_ids(env):
    assert _post([]).status_code == 400


def test_keeps_row_whose_edge_survives_and_returns_live_ask(env):
    _insert(env, "T-A", ask=0.40, fair=0.46)
    _FakeKalshi.asks = {"T-A": 0.41}
    r = _by_id(_post(["T-A"]))["T-A"]
    assert r["keep"] is True
    assert r["live_ask"] == pytest.approx(0.41)
    assert r["scan_ask"] == pytest.approx(0.40)
    assert r["delta"] == pytest.approx(0.01)
    assert r["live_ev"] is not None and r["stake"] > 0


def test_drops_row_whose_edge_reverted_below_floor(env):
    _insert(env, "T-B", ask=0.40, fair=0.46)
    _FakeKalshi.asks = {"T-B": 0.46}  # ask walked up to fair
    r = _by_id(_post(["T-B"]))["T-B"]
    assert r["keep"] is False
    assert "below EV floor" in r["reason"]
    assert r["stake"] == 0.0
    assert r["live_ask"] == pytest.approx(0.46)


@pytest.mark.parametrize("quote", [None, 0.99, 1.0])
def test_never_trusts_missing_or_empty_book_quote(env, quote):
    _insert(env, "T-C")
    _FakeKalshi.asks = {"T-C": quote}
    r = _by_id(_post(["T-C"]))["T-C"]
    assert r["keep"] is False
    assert r["live_ask"] is None or r["live_ask"] == quote
    assert r["stake"] == 0.0


def test_kalshi_fetch_failure_keeps_nothing_and_does_not_fall_back_to_scan_ask(env):
    _insert(env, "T-D")
    _FakeKalshi.fail = True
    res = _post(["T-D"])
    r = _by_id(res)["T-D"]
    assert r["keep"] is False
    assert r["live_ask"] is None
    assert res.json()["live"] is False
    assert res.json()["warnings"]


def test_in_play_game_is_dropped(env):
    _insert(env, "T-E", ask=0.40, fair=0.46)
    _FakeKalshi.asks = {"T-E": 0.40}
    started = datetime.now(timezone.utc) - timedelta(minutes=30)
    _FakePinnacle.odds = [SimpleNamespace(
        event_id="evt:T-E", event_date=started,
        outcome_a_label="Celtics", outcome_b_label="Heat",
        true_prob_a=0.46, true_prob_b=0.54,
    )]
    r = _by_id(_post(["T-E"]))["T-E"]
    assert r["keep"] is False
    assert r["in_play"] is True
    assert "in-play" in r["reason"]


def test_fresh_pinnacle_line_reblends_fair(env):
    _insert(env, "T-F", ask=0.40, fair=0.46)
    _FakeKalshi.asks = {"T-F": 0.40}
    future = datetime.now(timezone.utc) + timedelta(hours=3)
    # Sharp prob dropped 0.46 -> 0.40; fair shifts by 0.85 * -0.06 = -0.051.
    _FakePinnacle.odds = [SimpleNamespace(
        event_id="evt:T-F", event_date=future,
        outcome_a_label="Celtics", outcome_b_label="Heat",
        true_prob_a=0.40, true_prob_b=0.60,
    )]
    r = _by_id(_post(["T-F"]))["T-F"]
    assert r["pin_is_live"] is True
    assert r["live_fair"] == pytest.approx(0.46 - 0.85 * 0.06, abs=1e-6)
    assert r["keep"] is False  # edge gone once the sharp line moved


def test_rejects_placed_voided_shadow_and_unknown_without_network(env):
    _insert(env, "P", placed=1)
    _insert(env, "V", voided=1)
    _insert(env, "S", mode="shadow")
    res = _post(["P", "V", "S", "NOPE"])
    by = _by_id(res)
    assert all(not r["keep"] for r in by.values())
    assert by["P"]["reason"] == "already placed"
    assert by["V"]["reason"] == "voided"
    assert by["S"]["reason"].startswith("mode=")
    assert by["NOPE"]["reason"] == "not in ev_predictions"


def test_polyus_failure_demotes_only_polyus_rows(env, monkeypatch):
    _insert(env, "K-1", ask=0.40, fair=0.46)
    _insert(env, "polymarket_us:abc", venue="polymarket_us", ask=0.40, fair=0.46)
    _FakeKalshi.asks = {"K-1": 0.40}

    async def _boom(rows):
        raise RuntimeError("gateway 500")

    monkeypatch.setattr("evmax.agents.cleanup.reprice.fetch_polymarket_live_asks", _boom)
    by = _by_id(_post(["K-1", "polymarket_us:abc"]))
    assert by["K-1"]["keep"] is True
    assert by["polymarket_us:abc"]["keep"] is False
    assert by["polymarket_us:abc"]["price_is_live"] is False


def test_classify_is_pure_and_flags_unknown_start():
    bet = {
        "market_id": "x", "blended_true_prob": 0.5, "ev_pct": 0.1,
        "live_ev_real": 0.08, "scan_ask": 0.4, "live_ask": 0.42,
        "price_is_live": True, "is_live": True, "stake": 12.345,
        "event_start": None, "venue": "kalshi",
    }
    out = verify_mod.classify([bet], min_ev=0.02, min_prob=0.15)[0]
    assert out["keep"] and out["start_unknown"] and out["stake"] == 12.35
