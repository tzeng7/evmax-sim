"""Shared maker-fill recorder — `evmax agents fill` and POST /api/fill both
route through `record_maker_fill`, so validation and column writes stay
identical between the CLI and the dashboard.

A maker-only play is fill-contingent: it clears the EV floor only as a resting
limit order, logs as mode='shadow', and must NOT be pickable as a taker. This
module is the one sanctioned way it becomes a real position.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

from evmax.agents.cleanup.maker_fill import (
    MakerFillError,
    normalize_fill_price,
    record_maker_fill,
)

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


def _conn(mode="shadow", placed=0, voided=0, venue="kalshi", market_id="kalshi:TEST-BOS"):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.execute(
        """INSERT INTO ev_predictions
           (scan_date, market_id, event_id, sector, yes_team, market_type,
            event_title, kalshi_yes_price, sharp_true_prob, blended_true_prob,
            ev_pct, kelly_fraction, line, voided, placed, mode, venue, maker_ev_pct)
           VALUES ('2026-03-21', ?, 'evt:1', 'nba', 'celtics', 'moneyline',
                   'Celtics vs Heat', 0.50, 0.525, 0.525, 0.014, 0.0, NULL,
                   ?, ?, ?, ?, 0.041)""",
        (market_id, voided, placed, mode, venue),
    )
    conn.commit()
    return conn


def _row(conn, market_id="kalshi:TEST-BOS"):
    return conn.execute(
        "SELECT * FROM ev_predictions WHERE market_id = ?", (market_id,)
    ).fetchone()


# --- normalize_fill_price --------------------------------------------------

@pytest.mark.parametrize("given,expected", [(47, 0.47), (0.47, 0.47), (99, 0.99), (0.01, 0.01)])
def test_normalize_accepts_cents_and_fractions(given, expected):
    assert normalize_fill_price(given) == pytest.approx(expected)


@pytest.mark.parametrize("bad", [0, 0.0, -5, 1.0, 100, 150, "abc", None])
def test_normalize_rejects_unfillable_prices(bad):
    # 1.0 and 100 both normalize to certainty, which is not a fillable binary
    # contract price; the rest are out of range or non-numeric.
    with pytest.raises(MakerFillError) as err:
        normalize_fill_price(bad)
    assert err.value.reason == "invalid_price"


# --- record_maker_fill happy path ------------------------------------------

def test_records_promotion_columns():
    conn = _conn()
    result = record_maker_fill(conn, "kalshi:TEST-BOS", 47, 25.0)
    row = _row(conn)
    assert row["placed"] == 1
    assert row["mode"] == "live"
    assert row["maker_fill"] == 1
    assert row["placed_price"] == pytest.approx(0.47)
    assert row["placed_stake"] == pytest.approx(25.0)
    assert row["placed_at"]
    # contracts = notional / price; the fee is charged at the MAKER rate.
    assert result.contracts == pytest.approx(25.0 / 0.47)
    assert result.maker_fee > 0
    assert result.prior_mode == "shadow"
    conn.close()


def test_maker_fee_is_cheaper_than_taker():
    from evmax.fees import venue_order_fee

    conn = _conn()
    result = record_maker_fill(conn, "kalshi:TEST-BOS", 47, 25.0)
    taker = venue_order_fee("kalshi", result.fill_prob, result.contracts, maker=False)
    # The whole point of recording maker_fill=1: P&L must charge the lower fee.
    assert result.maker_fee < taker
    conn.close()


def test_fee_is_venue_aware():
    conn = _conn(venue="polymarket_us")
    result = record_maker_fill(conn, "kalshi:TEST-BOS", 47, 25.0)
    assert result.venue == "polymarket_us"
    # Polymarket US pays the maker a rebate, so the fee is negative there.
    assert result.maker_fee < 0
    conn.close()


def test_caller_owns_the_transaction():
    conn = _conn()
    record_maker_fill(conn, "kalshi:TEST-BOS", 47, 25.0)
    conn.rollback()
    # No commit inside the helper, so a rollback discards the write. This is what
    # lets a batch of dashboard fills land atomically.
    assert _row(conn)["placed"] == 0
    conn.close()


def test_accepts_injected_timestamp():
    conn = _conn()
    stamp = datetime(2026, 3, 21, 18, 30, tzinfo=timezone.utc)
    record_maker_fill(conn, "kalshi:TEST-BOS", 47, 25.0, now=stamp)
    assert _row(conn)["placed_at"] == stamp.isoformat()
    conn.close()


def test_non_shadow_row_still_fills_and_reports_prior_mode():
    conn = _conn(mode="live")
    result = record_maker_fill(conn, "kalshi:TEST-BOS", 47, 25.0)
    assert result.prior_mode == "live"
    assert _row(conn)["maker_fill"] == 1
    conn.close()


# --- record_maker_fill rejections ------------------------------------------

def test_rejects_unknown_market():
    conn = _conn()
    with pytest.raises(MakerFillError) as err:
        record_maker_fill(conn, "kalshi:NOPE", 47, 25.0)
    assert err.value.reason == "not_found"
    assert _row(conn)["placed"] == 0
    conn.close()


def test_rejects_missing_market_id():
    conn = _conn()
    with pytest.raises(MakerFillError) as err:
        record_maker_fill(conn, "", 47, 25.0)
    assert err.value.reason == "missing_market_id"
    conn.close()


def test_rejects_already_placed_without_overwriting_the_fill():
    conn = _conn(mode="live", placed=1)
    conn.execute(
        "UPDATE ev_predictions SET placed_price = 0.41, placed_stake = 9.0 "
        "WHERE market_id = 'kalshi:TEST-BOS'"
    )
    with pytest.raises(MakerFillError) as err:
        record_maker_fill(conn, "kalshi:TEST-BOS", 47, 25.0)
    assert err.value.reason == "already_placed"
    # A double-submit must never clobber the real recorded fill.
    row = _row(conn)
    assert row["placed_price"] == pytest.approx(0.41)
    assert row["placed_stake"] == pytest.approx(9.0)
    conn.close()


def test_rejects_voided_row():
    conn = _conn(voided=1)
    with pytest.raises(MakerFillError) as err:
        record_maker_fill(conn, "kalshi:TEST-BOS", 47, 25.0)
    assert err.value.reason == "voided"
    assert _row(conn)["placed"] == 0
    conn.close()


@pytest.mark.parametrize("bad_stake", [0, -3.0, "abc", None])
def test_rejects_non_positive_stake(bad_stake):
    conn = _conn()
    with pytest.raises(MakerFillError) as err:
        record_maker_fill(conn, "kalshi:TEST-BOS", 47, bad_stake)
    assert err.value.reason == "invalid_stake"
    assert _row(conn)["placed"] == 0
    conn.close()


def test_price_is_validated_before_the_row_is_touched():
    conn = _conn()
    with pytest.raises(MakerFillError):
        record_maker_fill(conn, "kalshi:TEST-BOS", 0, 25.0)
    assert _row(conn)["placed"] == 0
    conn.close()
