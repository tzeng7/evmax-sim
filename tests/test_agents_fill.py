"""Tests for `evmax agents fill` — promoting a maker-only shadow row to a
real placed maker bet (placed=1, mode='live', maker_fill=1).

A maker-only play surfaces tagged MAKER and logs as shadow (fill-contingent).
When the resting limit order fills, `fill` records it as a live position and
flags maker_fill=1 so the net-of-fee P&L charges the maker fee, not the taker.
"""

from __future__ import annotations

import sqlite3

import pytest
from typer.testing import CliRunner

from evmax.cli.commands import agents as agents_cmd

runner = CliRunner()


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
    """Wrap an in-memory conn so the command's .close() doesn't drop the DB."""

    def __init__(self, inner):
        self._inner = inner

    def close(self):  # command calls this; keep the DB alive for assertions
        pass

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _seed(mode="shadow", placed=0, voided=0, maker_ev=0.041):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.execute(
        """INSERT INTO ev_predictions
           (scan_date, market_id, event_id, sector, yes_team, market_type,
            event_title, kalshi_yes_price, sharp_true_prob, blended_true_prob,
            ev_pct, kelly_fraction, voided, placed, mode, venue, maker_ev_pct)
           VALUES ('2026-03-21', 'kalshi:TEST-BOS', 'evt:1', 'nba', 'celtics',
                   'moneyline', 'Celtics vs Heat', 0.50, 0.525, 0.525,
                   0.014, 0.0, ?, ?, ?, 'kalshi', ?)""",
        (voided, placed, mode, maker_ev),
    )
    conn.commit()
    return conn


@pytest.fixture
def patched_conn(monkeypatch):
    conn = _seed()
    # fill imports get_connection from the db module at call time.
    monkeypatch.setattr(
        "evmax.agents.cleanup.db.get_connection", lambda: _NoCloseConn(conn)
    )
    yield conn
    conn.close()


def test_fill_promotes_shadow_to_live_placed_maker(patched_conn):
    result = runner.invoke(
        agents_cmd.app, ["fill", "kalshi:TEST-BOS", "-p", "47", "-s", "25"]
    )
    assert result.exit_code == 0, result.output
    row = patched_conn.execute(
        "SELECT placed, mode, maker_fill, placed_price, placed_stake "
        "FROM ev_predictions WHERE market_id = 'kalshi:TEST-BOS'"
    ).fetchone()
    assert row["placed"] == 1
    assert row["mode"] == "live"
    assert row["maker_fill"] == 1
    assert row["placed_price"] == pytest.approx(0.47)
    assert row["placed_stake"] == pytest.approx(25.0)


def test_fill_accepts_fractional_price(patched_conn):
    result = runner.invoke(
        agents_cmd.app, ["fill", "kalshi:TEST-BOS", "-p", "0.47", "-s", "10"]
    )
    assert result.exit_code == 0, result.output
    row = patched_conn.execute(
        "SELECT placed_price FROM ev_predictions WHERE market_id = 'kalshi:TEST-BOS'"
    ).fetchone()
    assert row["placed_price"] == pytest.approx(0.47)


def test_fill_unknown_market_id_errors(patched_conn):
    result = runner.invoke(
        agents_cmd.app, ["fill", "kalshi:NOPE", "-p", "47", "-s", "25"]
    )
    assert result.exit_code == 1
    # original row untouched
    row = patched_conn.execute(
        "SELECT placed FROM ev_predictions WHERE market_id = 'kalshi:TEST-BOS'"
    ).fetchone()
    assert row["placed"] == 0


def test_fill_rejects_already_placed(monkeypatch):
    conn = _seed(mode="live", placed=1)
    monkeypatch.setattr(
        "evmax.agents.cleanup.db.get_connection", lambda: _NoCloseConn(conn)
    )
    result = runner.invoke(
        agents_cmd.app, ["fill", "kalshi:TEST-BOS", "-p", "47", "-s", "25"]
    )
    assert result.exit_code == 1
    assert "already placed" in result.output.lower()
    conn.close()


def test_fill_rejects_voided(monkeypatch):
    conn = _seed(voided=1)
    monkeypatch.setattr(
        "evmax.agents.cleanup.db.get_connection", lambda: _NoCloseConn(conn)
    )
    result = runner.invoke(
        agents_cmd.app, ["fill", "kalshi:TEST-BOS", "-p", "47", "-s", "25"]
    )
    assert result.exit_code == 1
    conn.close()


def test_fill_rejects_bad_price(patched_conn):
    result = runner.invoke(
        agents_cmd.app, ["fill", "kalshi:TEST-BOS", "-p", "150", "-s", "25"]
    )
    assert result.exit_code == 1
