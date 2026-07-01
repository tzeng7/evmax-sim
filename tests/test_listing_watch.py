"""Tests for the listing-window watcher (2026-07-01, WNBA spread promotion path).

Covers the three new pieces:
  - ``book_depth_metrics`` / ``KalshiWSClient._parse_book`` (kalshi.py) — turn a
    raw orderbook snapshot into taker-perspective depth numbers, and keep the
    refactored ``_parse_message`` byte-compatible with the old ask extraction.
  - ``DataArchiver.archive_orderbook_depth`` (archiver.py) — the depth time
    series table, including the (session_id, ticker) dedup.
  - ``listing_window_markets`` (cli/commands/cleanup.py) — the pure sweep
    filter (market-type set + [now−24h, now+window] event window).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from evmax.archiver import DataArchiver
from evmax.cli.commands.cleanup import listing_window_markets, resolve_watch_sectors
from evmax.clients.kalshi import SECTOR_SERIES_MAP, KalshiWSClient, book_depth_metrics
from evmax.models.market import MarketSource, MarketType, PredictionMarket


# ---------------------------------------------------------------------------
# book_depth_metrics
# ---------------------------------------------------------------------------
class TestBookDepthMetrics:
    def test_liquid_book(self):
        # Best NO bid 0.55/$300 → YES ask 0.45 with $300 fillable.
        m = book_depth_metrics(
            yes_bids=[(0.10, 50.0), (0.40, 120.0)],
            no_bids=[(0.20, 80.0), (0.55, 300.0)],
        )
        assert m["yes_ask"] == pytest.approx(0.45)
        assert m["yes_ask_depth_usd"] == 300.0
        assert m["yes_bid"] == pytest.approx(0.40)
        assert m["yes_bid_depth_usd"] == 120.0
        assert m["yes_book_usd"] == 170.0
        assert m["no_book_usd"] == 380.0

    def test_empty_ladders_are_the_placeholder_case(self):
        # Freshly-listed market: quote may exist upstream but the book is bare.
        m = book_depth_metrics(yes_bids=[], no_bids=[])
        assert m["yes_ask"] is None
        assert m["yes_ask_depth_usd"] is None
        assert m["yes_bid"] is None
        assert m["yes_bid_depth_usd"] is None
        assert m["yes_book_usd"] == 0.0
        assert m["no_book_usd"] == 0.0

    def test_malformed_price_nulls_that_side(self):
        # A best NO bid of 0 would imply yes_ask=1.0 — out of (0,1), so both
        # the price AND its depth must be None (no fake fillable quote).
        m = book_depth_metrics(yes_bids=[(0.30, 10.0)], no_bids=[(0.0, 999.0)])
        assert m["yes_ask"] is None
        assert m["yes_ask_depth_usd"] is None
        assert m["yes_bid"] == pytest.approx(0.30)


# ---------------------------------------------------------------------------
# KalshiWSClient._parse_book / _parse_message parity
# ---------------------------------------------------------------------------
def _ws_client() -> KalshiWSClient:
    return KalshiWSClient(sign_fn=lambda m, p: {}, key_id="", ws_url="wss://x")


SNAPSHOT = {
    "type": "orderbook_snapshot",
    "msg": {
        "market_ticker": "KXWNBASPREAD-26JUL02SEALV-LV7",
        "yes_dollars_fp": [["0.10", "50.00"], ["0.40", "120.00"]],
        "no_dollars_fp": [["0.20", "80.00"], ["0.55", "300.00"]],
    },
}


class TestParseBook:
    def test_parses_ladders_as_floats(self):
        ticker, yes_bids, no_bids = _ws_client()._parse_book(SNAPSHOT)
        assert ticker == "KXWNBASPREAD-26JUL02SEALV-LV7"
        assert yes_bids == [(0.10, 50.0), (0.40, 120.0)]
        assert no_bids == [(0.20, 80.0), (0.55, 300.0)]

    def test_non_snapshot_message_returns_empty(self):
        ticker, yes_bids, no_bids = _ws_client()._parse_book({"type": "subscribed"})
        assert ticker is None and yes_bids == [] and no_bids == []

    def test_malformed_entries_skipped(self):
        msg = {
            "type": "orderbook_snapshot",
            "msg": {
                "market_ticker": "T1",
                "yes_dollars_fp": [["bad"], ["0.30", "10.00"]],
                "no_dollars_fp": [],
            },
        }
        _, yes_bids, no_bids = _ws_client()._parse_book(msg)
        assert yes_bids == [(0.30, 10.0)] and no_bids == []

    def test_parse_message_still_derives_asks(self):
        # The old contract: yes_ask = 1 − best NO bid, no_ask = 1 − best YES bid.
        ticker, yes_ask, no_ask = _ws_client()._parse_message(SNAPSHOT)
        assert ticker == "KXWNBASPREAD-26JUL02SEALV-LV7"
        assert yes_ask == pytest.approx(0.45)
        assert no_ask == pytest.approx(0.60)


# ---------------------------------------------------------------------------
# DataArchiver.archive_orderbook_depth
# ---------------------------------------------------------------------------
@pytest.fixture
def temp_archive_db(tmp_path, monkeypatch):
    db_path = tmp_path / "archive.db"
    monkeypatch.setattr("evmax.archiver.DB_PATH", db_path)
    return db_path


class TestArchiveOrderbookDepth:
    def test_inserts_and_dedups_per_session(self, temp_archive_db):
        archiver = DataArchiver()
        rows = [
            {
                "ticker": "T1", "yes_ask": 0.45, "yes_ask_depth_usd": 300.0,
                "yes_bid": 0.40, "yes_bid_depth_usd": 120.0,
                "yes_book_usd": 170.0, "no_book_usd": 380.0, "source": "ws",
            },
            {"ticker": "T2", "yes_ask": None, "yes_book_usd": 0.0,
             "no_book_usd": 0.0, "source": "rest"},
        ]
        assert archiver.archive_orderbook_depth("s1", "wnba", rows) == 2
        # Same session re-insert is IGNOREd; a new session appends.
        archiver.archive_orderbook_depth("s1", "wnba", rows)
        archiver.archive_orderbook_depth("s2", "wnba", rows[:1])

        import sqlite3
        conn = sqlite3.connect(str(temp_archive_db))
        n, = conn.execute("SELECT COUNT(*) FROM archived_orderbook_depth").fetchone()
        assert n == 3
        got = conn.execute(
            "SELECT yes_ask, yes_ask_depth_usd, source FROM archived_orderbook_depth "
            "WHERE session_id='s1' AND ticker='T1'"
        ).fetchone()
        assert got == (0.45, 300.0, "ws")
        conn.close()

    def test_empty_list_is_noop(self, temp_archive_db):
        assert DataArchiver().archive_orderbook_depth("s1", "wnba", []) == 0


# ---------------------------------------------------------------------------
# listing_window_markets
# ---------------------------------------------------------------------------
def _mkt(ticker: str, market_type: MarketType, event_date: datetime | None) -> PredictionMarket:
    return PredictionMarket(
        id=f"kalshi:{ticker}", source=MarketSource.kalshi, sector="wnba",
        market_type=market_type, ticker=ticker,
        yes_price=0.5, no_price=0.5, event_date=event_date,
    )


class TestListingWindowMarkets:
    NOW = datetime(2026, 7, 1, 18, 0, tzinfo=timezone.utc)

    def test_keeps_spread_inside_window(self):
        m = _mkt("S1", MarketType.spread, self.NOW + timedelta(hours=40))
        assert listing_window_markets([m], {"spread"}, 72, now=self.NOW) == [m]

    def test_drops_wrong_type_and_beyond_window(self):
        ml = _mkt("M1", MarketType.moneyline, self.NOW + timedelta(hours=40))
        far = _mkt("S2", MarketType.spread, self.NOW + timedelta(hours=80))
        assert listing_window_markets([ml, far], {"spread"}, 72, now=self.NOW) == []

    def test_empty_type_set_means_all_types(self):
        ml = _mkt("M1", MarketType.moneyline, self.NOW + timedelta(hours=10))
        assert listing_window_markets([ml], set(), 72, now=self.NOW) == [ml]

    def test_noon_utc_anchor_slack_keeps_tonights_game(self):
        # Kalshi ticker dates anchor at noon UTC, so a game tipping tonight has
        # event_date hours BEFORE `now` — the 24h lower bound must keep it.
        tonight = _mkt("S3", MarketType.spread, self.NOW - timedelta(hours=6))
        assert listing_window_markets([tonight], {"spread"}, 72, now=self.NOW) == [tonight]

    def test_drops_null_event_date_and_stale(self):
        no_date = _mkt("S4", MarketType.spread, None)
        stale = _mkt("S5", MarketType.spread, self.NOW - timedelta(hours=30))
        assert listing_window_markets([no_date, stale], {"spread"}, 72, now=self.NOW) == []


# ---------------------------------------------------------------------------
# resolve_watch_sectors
# ---------------------------------------------------------------------------
class TestResolveWatchSectors:
    def test_all_expands_to_game_sectors_without_props(self):
        got = resolve_watch_sectors("all")
        assert set(got) == {s for s in SECTOR_SERIES_MAP if not s.endswith("_props")}
        assert "nba_props" not in got and "nfl_props" not in got
        assert "wnba" in got and "tennis" in got

    def test_explicit_list_passes_through_normalized(self):
        assert resolve_watch_sectors("WNBA, baseball ,") == ["wnba", "baseball"]
