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


# ---------------------------------------------------------------------------
# near_tip_snapshot_candidates (watch-closes bet selection)
# ---------------------------------------------------------------------------
class TestNearTipSnapshotCandidates:
    """Regression for the 2026-07-05 widening: shadow rows MUST be included.

    Laddered categories (WNBA spread/total) live in shadow mode; their CLV
    promotion gate anchors against the last archived Kalshi price before tip.
    With the old mode='live' filter, shadow bets never got a watch-closes
    near-tip snapshot.
    """

    @pytest.fixture
    def conn(self):
        import sqlite3

        c = sqlite3.connect(":memory:")
        c.row_factory = sqlite3.Row
        c.execute(
            """CREATE TABLE ev_predictions (
                   id INTEGER PRIMARY KEY, market_id TEXT, sector TEXT,
                   event_id TEXT, event_date TEXT, market_type TEXT,
                   yes_team TEXT, line REAL, mode TEXT, voided INTEGER)"""
        )
        c.execute(
            """CREATE TABLE ev_outcomes (
                   id INTEGER PRIMARY KEY, market_id TEXT, outcome INTEGER)"""
        )
        yield c
        c.close()

    def _bet(self, conn, market_id, event_id, mode="live", voided=0):
        conn.execute(
            """INSERT INTO ev_predictions
               (market_id, sector, event_id, event_date, market_type,
                yes_team, line, mode, voided)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (market_id, "wnba", event_id, "2026-07-02", "spread",
             "storm", -6.5, mode, voided),
        )

    def test_includes_both_live_and_shadow_unresolved(self, conn):
        from evmax.cli.commands.cleanup import near_tip_snapshot_candidates

        self._bet(conn, "kalshi:T1", "ev1", mode="live")
        self._bet(conn, "kalshi:T2", "ev1", mode="shadow")   # the regression
        got = {r["market_id"] for r in near_tip_snapshot_candidates(conn, ["ev1"])}
        assert got == {"kalshi:T1", "kalshi:T2"}

    def test_excludes_voided_resolved_other_events_and_stray_modes(self, conn):
        from evmax.cli.commands.cleanup import near_tip_snapshot_candidates

        self._bet(conn, "kalshi:VOID", "ev1", voided=1)
        self._bet(conn, "kalshi:DONE", "ev1", mode="shadow")
        conn.execute(
            "INSERT INTO ev_outcomes (market_id, outcome) VALUES ('kalshi:DONE', 1)"
        )
        self._bet(conn, "kalshi:OTHER", "ev2")                # not in event list
        self._bet(conn, "kalshi:STRAY", "ev1", mode="disabled")
        assert near_tip_snapshot_candidates(conn, ["ev1"]) == []

    def test_unresolved_outcome_row_still_included(self, conn):
        from evmax.cli.commands.cleanup import near_tip_snapshot_candidates

        # outcome row exists but is NULL (pending) — must still snapshot
        self._bet(conn, "kalshi:PEND", "ev1", mode="shadow")
        conn.execute(
            "INSERT INTO ev_outcomes (market_id, outcome) VALUES ('kalshi:PEND', NULL)"
        )
        got = near_tip_snapshot_candidates(conn, ["ev1"])
        assert [r["market_id"] for r in got] == ["kalshi:PEND"]

    def test_empty_event_list_returns_empty(self, conn):
        from evmax.cli.commands.cleanup import near_tip_snapshot_candidates

        assert near_tip_snapshot_candidates(conn, []) == []


# ---------------------------------------------------------------------------
# snapshot_targets_by_venue (watch-closes venue partition)
#
# Before the 2026-07-07 extension, _capture_kalshi filtered on
# market_id.startswith("kalshi:"), so every polymarket_us row was silently
# dropped from the near-tip sweep — no venue close was ever archived and the
# venue promotion gate had nothing to score CLV against.
# ---------------------------------------------------------------------------
class TestSnapshotTargetsByVenue:
    def _row(self, market_id, sector="wnba"):
        return {"market_id": market_id, "sector": sector, "event_id": "ev1",
                "event_date": "2026-07-08", "market_type": "moneyline",
                "yes_team": "storm", "line": None}

    def test_partitions_kalshi_and_polymarket_us(self):
        from evmax.cli.commands.cleanup import snapshot_targets_by_venue

        targets = snapshot_targets_by_venue([
            self._row("kalshi:KXWNBAGAME-X"),
            self._row("polymarket_us:lva-sea-2026-07-08:LVA"),
        ])
        assert set(targets["kalshi"]) == {"KXWNBAGAME-X"}
        # PolyUS keeps its venue prefix — that IS the archive ticker.
        assert set(targets["polymarket_us"]) == {"polymarket_us:lva-sea-2026-07-08:LVA"}

    def test_no_suffix_stripped_and_deduped_per_ticker(self):
        from evmax.cli.commands.cleanup import snapshot_targets_by_venue

        targets = snapshot_targets_by_venue([
            self._row("kalshi:KXT-1"),
            self._row("kalshi:KXT-1:no"),
            self._row("polymarket_us:slug-total:no"),
        ])
        assert set(targets["kalshi"]) == {"KXT-1"}
        assert set(targets["polymarket_us"]) == {"polymarket_us:slug-total"}

    def test_unknown_venue_and_null_ids_skipped(self):
        from evmax.cli.commands.cleanup import snapshot_targets_by_venue

        targets = snapshot_targets_by_venue([
            self._row(None),
            self._row("otherbook:XYZ"),
        ])
        assert targets == {}


# ---------------------------------------------------------------------------
# capture_polymarket_us_closes (watch-closes PolyUS snapshot branch)
# ---------------------------------------------------------------------------
class TestCapturePolymarketUsCloses:
    MID = "polymarket_us:lva-sea-2026-07-08:LVA"

    def _targets(self):
        return {self.MID: {"market_id": self.MID, "sector": "wnba",
                           "event_id": "wnba::2026-07-08::aces_vs_storm",
                           "event_date": "2026-07-08", "market_type": "moneyline",
                           "yes_team": "aces", "line": None}}

    class _FakeClient:
        def __init__(self, markets_by_sector):
            self._markets = markets_by_sector
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            return None
        async def get_markets(self, sector):
            return self._markets.get(sector, [])

    def _fake_market(self, mid, yes_price, no_price=None):
        from types import SimpleNamespace
        return SimpleNamespace(id=mid, yes_price=yes_price, no_price=no_price)

    def _patch_settings(self, monkeypatch, enabled=True):
        from types import SimpleNamespace
        monkeypatch.setattr(
            "evmax.settings.get_settings",
            lambda: SimpleNamespace(polymarket_us_enabled=enabled),
        )

    def test_archives_snapshot_under_prefixed_ticker(self, tmp_path, monkeypatch):
        import asyncio
        import evmax.archiver as archiver_mod
        from evmax.cli.commands.cleanup import capture_polymarket_us_closes

        monkeypatch.setattr(archiver_mod, "DB_PATH", tmp_path / "archive.db")
        self._patch_settings(monkeypatch)
        fake = self._FakeClient({"wnba": [self._fake_market(self.MID, 0.61, 0.41)]})
        monkeypatch.setattr(
            "evmax.clients.polymarket_us.PolymarketUSClient", lambda: fake
        )

        n = asyncio.run(capture_polymarket_us_closes(self._targets()))
        assert n == 1
        with archiver_mod._get_connection() as conn:
            row = conn.execute(
                "SELECT ticker, sector, yes_price, no_price, event_id "
                "FROM archived_kalshi_markets"
            ).fetchone()
        assert row["ticker"] == self.MID          # venue-prefixed on purpose
        assert row["sector"] == "wnba"
        assert row["yes_price"] == pytest.approx(0.61)
        assert row["no_price"] == pytest.approx(0.41)
        assert row["event_id"] == "wnba::2026-07-08::aces_vs_storm"

    def test_kill_switch_skips_fetch(self, tmp_path, monkeypatch):
        import asyncio
        from evmax.cli.commands.cleanup import capture_polymarket_us_closes

        self._patch_settings(monkeypatch, enabled=False)

        def _boom():
            raise AssertionError("client must not be constructed when disabled")

        monkeypatch.setattr("evmax.clients.polymarket_us.PolymarketUSClient", _boom)
        assert asyncio.run(capture_polymarket_us_closes(self._targets())) == 0

    def test_empty_book_and_missing_market_skipped(self, tmp_path, monkeypatch):
        import asyncio
        import evmax.archiver as archiver_mod
        from evmax.cli.commands.cleanup import capture_polymarket_us_closes

        monkeypatch.setattr(archiver_mod, "DB_PATH", tmp_path / "archive.db")
        self._patch_settings(monkeypatch)
        targets = self._targets()
        targets["polymarket_us:gone-market"] = dict(
            targets[self.MID], market_id="polymarket_us:gone-market"
        )
        # 0.99+ mirrors the Kalshi empty-book filter; gone-market isn't listed.
        fake = self._FakeClient({"wnba": [self._fake_market(self.MID, 0.99)]})
        monkeypatch.setattr(
            "evmax.clients.polymarket_us.PolymarketUSClient", lambda: fake
        )
        assert asyncio.run(capture_polymarket_us_closes(targets)) == 0

    def test_empty_targets_noop(self):
        import asyncio
        from evmax.cli.commands.cleanup import capture_polymarket_us_closes

        assert asyncio.run(capture_polymarket_us_closes({})) == 0
