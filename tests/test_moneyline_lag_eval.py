"""Tests for the moneyline venue-lag lens (moneyline_lag_eval.py).

Entry = first sweep with a concurrent Pinnacle anchor; close = last pre-tip
snapshot at least MIN_HOLD_H later; placeholder asks never score; net CLV
subtracts the Kalshi taker fee at the entry price; slope/verdict use game
clusters.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from evmax.agents.cleanup.moneyline_lag_eval import (
    LagStats,
    Unit,
    cluster_slope,
    evaluate_moneyline,
    mean_t,
    verdict,
)
from evmax.archiver import _get_connection
from evmax.fees import kalshi_fee_prob

TIP = "2026-10-11T17:00:00+00:00"
EVENT = "nfl::2026-10-11::patriots_vs_seahawks"
TICKER = "KXNFLGAME-26OCT11NESEA-SEA"


@pytest.fixture
def conn(tmp_path, monkeypatch):
    db_path = tmp_path / "archive.db"
    monkeypatch.setattr("evmax.archiver.DB_PATH", db_path)
    with _get_connection():
        pass
    c = sqlite3.connect(str(db_path))
    c.row_factory = sqlite3.Row
    yield c
    c.close()


def pin(conn, fetched_at, p_a=0.40, p_b=0.60):
    conn.execute(
        """INSERT INTO archived_sharp_odds
           (session_id, fetched_at, sector, event_id, book, outcome_a_label,
            outcome_b_label, outcome_a_decimal, outcome_b_decimal, true_prob_a,
            true_prob_b, margin, event_date)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (f"p-{fetched_at}", fetched_at, "nfl", EVENT, "pinnacle", "Patriots",
         "Seahawks", 2.5, 1.7, p_a, p_b, 0.0, TIP),
    )


def kal(conn, session, fetched_at, yes_price, ticker=TICKER, yes_team="seahawks"):
    conn.execute(
        """INSERT INTO archived_kalshi_markets
           (session_id, fetched_at, sector, ticker, title, market_type, yes_price,
            no_price, team_home, team_away, yes_team, line, event_date)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (session, fetched_at, "nfl", ticker, None, "moneyline", yes_price,
         1 - yes_price, "sea", "ne", yes_team, None, "2026-10-11T12:00:00+00:00"),
    )


def test_entry_close_and_net_clv(conn):
    pin(conn, "2026-10-09T12:00:00+00:00", p_a=0.40, p_b=0.60)
    kal(conn, "watchlist-1", "2026-10-09T12:05:00+00:00", 0.55)   # entry (anchored)
    kal(conn, "watchlist-2", "2026-10-11T15:30:00+00:00", 0.58)   # close, T-1.5h
    stats = evaluate_moneyline(conn, "nfl", "KXNFLGAME")
    (u,) = stats.units
    assert u.entry_ask == 0.55
    assert u.edge_pp == pytest.approx((0.60 - 0.55) * 100)
    assert u.close_ask == 0.58
    assert u.gross_clv_pp == pytest.approx(3.0)
    assert u.net_clv_pp == pytest.approx(3.0 - kalshi_fee_prob(0.55) * 100)


def test_unanchored_sweep_is_skipped_and_stale_anchor_not_used(conn):
    pin(conn, "2026-10-09T12:00:00+00:00")
    kal(conn, "watchlist-0", "2026-10-09T05:00:00+00:00", 0.50)   # before any anchor
    kal(conn, "watchlist-1", "2026-10-09T20:00:00+00:00", 0.55)   # anchor 8h old: stale
    stats = evaluate_moneyline(conn, "nfl", "KXNFLGAME")
    assert stats.units == [] and stats.never_anchored == 1


def test_no_close_means_no_clv_not_zero(conn):
    pin(conn, "2026-10-11T10:00:00+00:00")
    kal(conn, "watchlist-1", "2026-10-11T10:05:00+00:00", 0.55)   # only snapshot
    (u,) = evaluate_moneyline(conn, "nfl", "KXNFLGAME").units
    assert u.gross_clv_pp is None and u.net_clv_pp is None


def test_placeholder_close_is_ignored(conn):
    pin(conn, "2026-10-09T12:00:00+00:00")
    kal(conn, "watchlist-1", "2026-10-09T12:05:00+00:00", 0.55)
    kal(conn, "watchlist-2", "2026-10-11T15:30:00+00:00", 0.995)  # empty-book print
    (u,) = evaluate_moneyline(conn, "nfl", "KXNFLGAME").units
    assert u.close_ask is None


def test_close_beyond_staleness_window_is_ignored(conn):
    pin(conn, "2026-10-09T12:00:00+00:00")
    kal(conn, "watchlist-1", "2026-10-09T12:05:00+00:00", 0.55)
    kal(conn, "watchlist-2", "2026-10-10T12:00:00+00:00", 0.58)   # T-29h
    (u,) = evaluate_moneyline(conn, "nfl", "KXNFLGAME").units
    assert u.close_ask is None


def test_cluster_slope_recovers_known_line():
    xs = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    ys = [1 + 0.5 * x + (0.1 if i % 2 else -0.1) for i, x in enumerate(xs)]
    b, se, z, g = cluster_slope(xs, ys, list(range(6)))
    assert b == pytest.approx(0.5, abs=0.05) and z > 5 and g == 6


def test_cluster_slope_degenerate_returns_none():
    assert cluster_slope([1.0, 1.0, 1.0], [1.0, 2.0, 3.0], [0, 1, 2]) is None
    assert cluster_slope([1.0], [1.0], [0]) is None


def test_mean_t_needs_two_games():
    assert mean_t([1.0]) is None
    m, t, pos = mean_t([1.0, 2.0, -1.0, 3.0])
    assert m == pytest.approx(1.25) and pos == 75.0


def _unit(i, edge, clv, game=None):
    now = datetime(2026, 10, 1, tzinfo=timezone.utc)
    return Unit(
        ticker=f"t{i}", game=game or (f"d{i}", frozenset({"a", "b"})), entry_at=now,
        entry_ask=0.5, fair=0.5, edge_pp=edge, depth_usd=None, tip_at=now + timedelta(days=1),
        entry_lead_h=24, close_ask=0.5, close_lead_h=1, gross_clv_pp=clv, net_clv_pp=clv - 1.0,
    )


def test_verdict_refuted_with_flat_slope_and_enough_games():
    units = [_unit(i, (i % 7) - 3.0, 0.2 * ((-1) ** i)) for i in range(80)]
    assert verdict(LagStats(units=units)).verdict == "REFUTED"


def test_verdict_underpowered_with_few_games():
    units = [_unit(i, float(i), 0.0) for i in range(10)]
    assert verdict(LagStats(units=units)).verdict == "UNDERPOWERED"


def test_verdict_lag_but_fee_negative():
    # CLV tracks edge (slope 0.5) but flagged units net out below zero after fee
    units = [
        _unit(i, e, 0.5 * e + (0.2 if i % 2 else -0.2))   # noise: a perfect fit has se 0
        for i, e in enumerate([-3, -2, -1, 0, 1, 2, 2.5, 3, 3, 2.2] * 4)
    ]
    for u in units:
        u.net_clv_pp = u.gross_clv_pp - 3.0
    assert verdict(LagStats(units=units)).verdict == "LAG-BUT-FEE-NEGATIVE"
