"""Tests for the PolyUS price-history close backfill (polyus_history.py)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from evmax.agents.cleanup import polyus_history as ph
from evmax.archiver import DataArchiver
from evmax.models.odds import SharpBook, SharpOdds

TIP = datetime(2026, 7, 9, 0, 0, tzinfo=timezone.utc)
TIP_TS = TIP.timestamp()


def _pt(minutes_before_tip, long_p, short_p):
    return {"timestamp": int(TIP_TS - minutes_before_tip * 60),
            "longPrice": long_p, "shortPrice": short_p}


def test_parse_ticker():
    assert ph.parse_ticker("polymarket_us:aec-wnba-ind-la-2026-07-08:ind") == (
        "aec-wnba-ind-la-2026-07-08", "ind")
    assert ph.parse_ticker("polymarket_us:asc-mlb-x-2026-07-08") == (
        "asc-mlb-x-2026-07-08", None)


def test_pick_points_never_uses_in_play():
    hist = [_pt(60, .5, .5), _pt(31, .6, .41), _pt(30, .62, .39), _pt(10, .7, .31),
            _pt(-5, .9, .11)]  # last one is after tip-off
    t30, pre = ph.pick_points(hist, TIP_TS)
    assert t30["longPrice"] == .62          # ts <= tip-30m is inclusive
    assert pre["longPrice"] == .7           # latest strictly before tip
    assert ph.pick_points([_pt(-1, .9, .1)], TIP_TS) == (None, None)


def test_side_prices_swap_for_short_side_and_reject_degenerate():
    p = _pt(20, .62, .39)
    assert ph.side_prices(p, True) == (.62, .39)
    assert ph.side_prices(p, False) == (.39, .62)
    assert ph.side_prices(_pt(20, .995, .02), True) is None   # settled / empty book
    assert ph.side_prices({"timestamp": 1, "longPrice": None, "shortPrice": .5}, True) is None


def test_resolve_is_long():
    sides = [{"long": True, "team": {"abbreviation": "IND"}},
             {"long": False, "team": {"abbreviation": "LA"}}]
    assert ph.resolve_is_long("IND", sides) is True
    assert ph.resolve_is_long("LA", sides) is False
    assert ph.resolve_is_long(None, []) is True          # spread/total: long side
    assert ph.resolve_is_long("zzz", sides) is None      # unknown -> never guess
    assert ph.resolve_is_long("b", []) is False          # no-abbreviation fallback key


class FakeClient:
    def __init__(self, hist):
        self.hist, self.calls = hist, []

    async def get_price_history(self, slug, start, end):
        self.calls.append((slug, start, end))
        assert end - start <= 24 * 3600               # docs: custom window <= 24h
        return self.hist

    async def get_market_sides(self, slug):
        return [{"long": True, "team": {"abbreviation": "ind"}},
                {"long": False, "team": {"abbreviation": "la"}}]


@pytest.fixture
def archive(tmp_path, monkeypatch):
    monkeypatch.setattr("evmax.archiver.DB_PATH", tmp_path / "archive.db")
    a = DataArchiver()
    a.archive_sharp_odds("s0", "wnba", [SharpOdds(
        event_id="wnba::2026-07-08::ind_vs_la", book=SharpBook.pinnacle, sector="wnba",
        outcome_a_label="ind", outcome_b_label="la", outcome_a_decimal=1.9,
        outcome_b_decimal=1.9, true_prob_a=.5, true_prob_b=.5, margin=.04,
        event_date=TIP, fetched_at=TIP - timedelta(hours=5))])
    return a


def _targets():
    mk = lambda side: ph.Target(f"polymarket_us:slug-1:{side}", "slug-1", side,
                                "wnba::2026-07-08::ind_vs_la", "wnba", "moneyline",
                                side, None, "2026-07-08")
    return [mk("ind"), mk("la")]


def test_backfill_writes_both_sides_and_close_is_selectable(archive):
    client = FakeClient([_pt(60, .5, .52), _pt(31, .6, .42), _pt(10, .7, .32), _pt(-3, .97, .04)])
    tally = asyncio.run(ph.backfill(_targets(), client))
    assert tally["written_t30"] == 2 and tally["written_pretip"] == 2
    assert len(client.calls) == 1                        # one history call per slug
    ind = archive.get_kalshi_close_price("polymarket_us:slug-1:ind", "wnba::2026-07-08::ind_vs_la")
    la = archive.get_kalshi_close_price("polymarket_us:slug-1:la", "wnba::2026-07-08::ind_vs_la")
    assert ind == pytest.approx(.6) and la == pytest.approx(.42)   # short side swaps
    # point timestamp, not "now", is the snapshot time => staleness ~ (T-30 - T-31)
    assert archive.get_kalshi_close_staleness_h(
        "polymarket_us:slug-1:ind", "wnba::2026-07-08::ind_vs_la") == pytest.approx(1 / 60, abs=1e-3)


def test_backfill_is_idempotent_and_skips_fresh(archive):
    client = FakeClient([_pt(31, .6, .42), _pt(10, .7, .32)])
    asyncio.run(ph.backfill(_targets(), client))
    again = asyncio.run(ph.backfill(_targets(), FakeClient([_pt(31, .6, .42)])))
    assert again["skipped_fresh"] == 2 and again["written_t30"] == 0


def test_dry_run_writes_nothing(archive):
    tally = asyncio.run(ph.backfill(_targets(), FakeClient([_pt(31, .6, .42)]), dry_run=True))
    assert tally["written_t30"] == 2
    assert archive.get_kalshi_close_price(
        "polymarket_us:slug-1:ind", "wnba::2026-07-08::ind_vs_la") is None


def test_missing_tip_or_history_is_tallied_not_fatal(archive):
    t = [ph.Target("polymarket_us:nope:ind", "nope", "ind", "wnba::unknown::x", "wnba",
                   "moneyline", "ind", None, None)]
    assert asyncio.run(ph.backfill(t, FakeClient([_pt(31, .6, .4)])))["no_tip"] == 1
    assert asyncio.run(ph.backfill(_targets(), FakeClient([]), force=True))["no_history"] == 2
