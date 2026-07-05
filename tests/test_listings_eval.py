"""Tests for the offline anchored-entry evaluator (listings_eval.py).

Covers the first-anchored-sweep entry rule end-to-end against a synthetic
archive.db: unanchored listing sweeps are skipped, the entry lands on the
first sweep with a concurrent Pinnacle anchor, the EV and depth gates drop
non-qualifying tickers (visibly, via the counters), both YES and NO sides
are priced with the correct crossable prices, and CLV is scored against the
last pre-tip snapshot.
"""

from __future__ import annotations

import sqlite3

import pytest

from evmax.agents.cleanup.listings_eval import evaluate_sector, summarize
from evmax.archiver import _get_connection

GAME_EVENT = "wnba::2026-07-02::mercury_vs_storm"
TIP = "2026-07-03T02:00:00+00:00"

S1 = ("watchlist-1", "2026-07-01T22:00:00+00:00")   # pre-Pinnacle (unanchored)
S2 = ("watchlist-2", "2026-07-02T16:00:00+00:00")   # anchor arrives
S3 = ("watchlist-3", "2026-07-02T20:00:00+00:00")   # close sweep


@pytest.fixture
def archive_conn(tmp_path, monkeypatch):
    db_path = tmp_path / "archive.db"
    monkeypatch.setattr("evmax.archiver.DB_PATH", db_path)
    with _get_connection():  # creates schema
        pass
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


def insert_kalshi(conn, session, fetched_at, ticker, market_type, yes_price,
                  line, yes_team, home="phx", away="sea"):
    conn.execute(
        """INSERT INTO archived_kalshi_markets
           (session_id, fetched_at, sector, ticker, title, market_type,
            yes_price, no_price, team_home, team_away, yes_team, line, event_date)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (session, fetched_at, "wnba", ticker, None, market_type, yes_price,
         1 - yes_price, home, away, yes_team, line, "2026-07-02T12:00:00+00:00"),
    )


def insert_depth(conn, session, fetched_at, ticker, ask, ask_usd, bid, bid_usd):
    conn.execute(
        """INSERT INTO archived_orderbook_depth
           (session_id, fetched_at, sector, ticker, yes_ask, yes_ask_depth_usd,
            yes_bid, yes_bid_depth_usd, source)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (session, fetched_at, "wnba", ticker, ask, ask_usd, bid, bid_usd, "ws"),
    )


def insert_sharp_spread(conn, session, fetched_at, spread_line=-4.5,
                        prob_a=0.5, a="Seattle Storm", b="Phoenix Mercury"):
    conn.execute(
        """INSERT INTO archived_sharp_odds
           (session_id, fetched_at, sector, event_id, book, outcome_a_label,
            outcome_b_label, outcome_a_decimal, outcome_b_decimal,
            true_prob_a, true_prob_b, margin, spread_line, event_date)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (session, fetched_at, "wnba", GAME_EVENT + "::spread", "pinnacle",
         a, b, 2.0, 2.0, prob_a, 1 - prob_a, 0.03, spread_line, TIP),
    )


def insert_sharp_total(conn, session, fetched_at, total_line, p_over):
    conn.execute(
        """INSERT INTO archived_sharp_odds
           (session_id, fetched_at, sector, event_id, book, outcome_a_label,
            outcome_b_label, outcome_a_decimal, outcome_b_decimal,
            true_prob_a, true_prob_b, margin, total_line, true_prob_over,
            true_prob_under, event_date)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (session, fetched_at, "wnba", f"{GAME_EVENT}::total::{total_line}",
         "pinnacle", "Over", "Under", 2.0, 2.0, p_over, 1 - p_over, 0.03,
         total_line, p_over, 1 - p_over, TIP),
    )


def spread_trail(conn, ticker="KXWNBASPREAD-26JUL02SEAPHX-SEA7",
                 asks=(0.30, 0.30, 0.40), bids=(0.20, 0.20, 0.35),
                 ask_usd=100.0, bid_usd=100.0, line=-6.5, yes_team="storm"):
    for (session, t), ask, bid in zip((S1, S2, S3), asks, bids):
        insert_kalshi(conn, session, t, ticker, "spread", ask, line, yes_team)
        insert_depth(conn, session, t, ticker, ask, ask_usd, bid, bid_usd)


class TestFirstAnchoredSweepEntry:
    def test_entry_lands_on_first_anchored_sweep_not_listing(self, archive_conn):
        spread_trail(archive_conn)
        insert_sharp_spread(archive_conn, *S2)   # anchor only exists from S2
        archive_conn.commit()

        stats = evaluate_sector(archive_conn, "wnba")
        lay = [e for e in stats.entries if e.side == "lay"]
        assert len(lay) == 1
        e = lay[0]
        # entered at S2 (first anchored), NOT at the S1 listing snapshot
        assert e.entry_at.isoformat() == S2[1]
        assert e.entry_price == pytest.approx(0.30)
        # storm -4.5 @ 0.5 -> mu=4.5, sigma=12.5; P(cover -6.5) ~ 0.4364
        assert e.fair_prob == pytest.approx(0.4364, abs=0.001)
        assert e.edge_pp == pytest.approx(13.64, abs=0.1)
        # CLV to the last pre-tip ask
        assert e.clv_pp == pytest.approx(10.0, abs=0.01)
        assert e.event_label == "storm vs mercury"

    def test_no_side_not_entered_when_bid_below_fair(self, archive_conn):
        spread_trail(archive_conn)               # bid 0.20 << fair 0.4364
        insert_sharp_spread(archive_conn, *S2)
        archive_conn.commit()

        stats = evaluate_sector(archive_conn, "wnba")
        assert not [e for e in stats.entries if e.side == "take"]

    def test_take_side_entry_and_clv_use_bid_prices(self, archive_conn):
        # bid at anchor sweep ABOVE fair -> NO side is +EV: edge = bid - fair
        spread_trail(archive_conn, bids=(0.55, 0.55, 0.45))
        insert_sharp_spread(archive_conn, *S2)
        archive_conn.commit()

        stats = evaluate_sector(archive_conn, "wnba")
        take = [e for e in stats.entries if e.side == "take"]
        assert len(take) == 1
        e = take[0]
        assert e.entry_price == pytest.approx(0.45)          # 1 - yes_bid
        assert e.edge_pp == pytest.approx(55 - 43.64, abs=0.1)
        assert e.fair_prob == pytest.approx(1 - 0.4364, abs=0.001)
        # NO CLV = bid_entry - bid_close = 0.55 - 0.45
        assert e.clv_pp == pytest.approx(10.0, abs=0.01)
        assert e.outcome_label == "mercury +6.5"

    def test_depth_gate_blocks_thin_books(self, archive_conn):
        spread_trail(archive_conn, ask_usd=10.0, bid_usd=10.0)
        insert_sharp_spread(archive_conn, *S2)
        archive_conn.commit()

        stats = evaluate_sector(archive_conn, "wnba", depth_min_usd=50.0)
        assert stats.entries == []
        assert stats.anchored_no_entry == 1

    def test_ev_gate_blocks_thin_edges(self, archive_conn):
        spread_trail(archive_conn, asks=(0.43, 0.43, 0.44))   # edge ~ +0.6pp
        insert_sharp_spread(archive_conn, *S2)
        archive_conn.commit()

        stats = evaluate_sector(archive_conn, "wnba", ev_min_pp=2.0)
        assert stats.entries == []
        assert stats.anchored_no_entry == 1

    def test_never_anchored_ticker_is_counted_not_silently_dropped(self, archive_conn):
        spread_trail(archive_conn)               # no sharp rows at all
        archive_conn.commit()

        stats = evaluate_sector(archive_conn, "wnba")
        assert stats.tickers == 1
        assert stats.never_anchored == 1
        assert stats.entries == []


class TestTotals:
    def total_trail(self, conn, line=164.5, asks=(0.50, 0.50, 0.55),
                    bids=(0.40, 0.40, 0.50)):
        ticker = "KXWNBATOTAL-26JUL02SEAPHX-165"
        for (session, t), ask, bid in zip((S1, S2, S3), asks, bids):
            insert_kalshi(conn, session, t, ticker, "total", ask, line, "over")
            insert_depth(conn, session, t, ticker, ask, 80.0, bid, 80.0)

    def test_over_entry_priced_from_exact_ladder_line(self, archive_conn):
        self.total_trail(archive_conn)
        insert_sharp_total(archive_conn, *S2, total_line=164.5, p_over=0.60)
        archive_conn.commit()

        stats = evaluate_sector(archive_conn, "wnba")
        over = [e for e in stats.entries if e.side == "over"]
        assert len(over) == 1
        assert over[0].fair_prob == pytest.approx(0.60)
        assert over[0].edge_pp == pytest.approx(10.0, abs=0.01)
        assert over[0].clv_pp == pytest.approx(5.0, abs=0.01)

    def test_under_entry_via_no_side(self, archive_conn):
        # yes_bid 0.65 -> under costs 0.35 vs fair_under 0.40 -> +5pp edge
        self.total_trail(archive_conn, asks=(0.70, 0.70, 0.66),
                         bids=(0.65, 0.65, 0.60))
        insert_sharp_total(archive_conn, *S2, total_line=164.5, p_over=0.60)
        archive_conn.commit()

        stats = evaluate_sector(archive_conn, "wnba")
        under = [e for e in stats.entries if e.side == "under"]
        assert len(under) == 1
        assert under[0].entry_price == pytest.approx(0.35)
        assert under[0].edge_pp == pytest.approx(5.0, abs=0.01)
        assert under[0].clv_pp == pytest.approx(5.0, abs=0.01)  # 0.65 - 0.60

    def test_far_ladder_line_is_not_an_anchor(self, archive_conn):
        self.total_trail(archive_conn, line=170.5)   # ladder only has 164.5
        insert_sharp_total(archive_conn, *S2, total_line=164.5, p_over=0.60)
        archive_conn.commit()

        stats = evaluate_sector(archive_conn, "wnba")
        assert stats.entries == []
        assert stats.never_anchored == 1


class TestSummarize:
    def test_groups_by_market_and_side(self, archive_conn):
        spread_trail(archive_conn)
        insert_sharp_spread(archive_conn, *S2)
        archive_conn.commit()

        stats = evaluate_sector(archive_conn, "wnba")
        rows = summarize(stats)
        assert len(rows) == 1
        s = rows[0]
        assert (s.market_type, s.side, s.n) == ("spread", "lay", 1)
        assert s.mean_clv_pp == pytest.approx(10.0, abs=0.01)
        assert s.pct_clv_pos == 100.0
