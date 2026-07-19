"""Tests for the live anchored-entry trigger (anchored_entry.py).

Covers: crossable-price capture (order-book ask, not the market's displayed
price), NO-side scanner conventions (":no" market_id, negated spread line +
opponent yes_team, totals side flip with unflipped line), the EV and depth
gates, in-play/imminent skip, and the anchored_entry model_sources token.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from evmax.agents.cleanup.anchored_entry import build_anchored_entries
from evmax.models.market import MarketSource, MarketType, PredictionMarket
from evmax.models.odds import SharpBook, SharpOdds

NOW = datetime(2026, 7, 2, 12, 0, tzinfo=timezone.utc)
TIP = datetime(2026, 7, 3, 2, 0, tzinfo=timezone.utc)
GAME_EVENT = "wnba::2026-07-02::mercury_vs_storm"


def spread_market(ticker="KXWNBASPREAD-26JUL02SEAPHX-SEA7", yes_price=0.32,
                  line=-6.5, yes_team="storm") -> PredictionMarket:
    return PredictionMarket(
        id=f"kalshi:{ticker}",
        source=MarketSource.kalshi,
        sector="wnba",
        market_type=MarketType.spread,
        ticker=ticker,
        yes_price=yes_price,
        no_price=1 - yes_price + 0.05,   # vig
        team_home="phx", team_away="sea",
        yes_team=yes_team, line=line,
        event_date=datetime(2026, 7, 2, 12, 0, tzinfo=timezone.utc),
        volume_usd=100.0,
    )


def total_market(ticker="KXWNBATOTAL-26JUL02SEAPHX-165", yes_price=0.50,
                 line=164.5) -> PredictionMarket:
    return PredictionMarket(
        id=f"kalshi:{ticker}",
        source=MarketSource.kalshi,
        sector="wnba",
        market_type=MarketType.total,
        ticker=ticker,
        yes_price=yes_price,
        no_price=1 - yes_price + 0.05,
        team_home="phx", team_away="sea",
        yes_team="over", line=line,
        event_date=datetime(2026, 7, 2, 12, 0, tzinfo=timezone.utc),
        volume_usd=100.0,
    )


def spread_sharp(prob_a=0.5, spread_line=-4.5, tip=TIP) -> SharpOdds:
    return SharpOdds(
        event_id=GAME_EVENT + "::spread",
        book=SharpBook.pinnacle,
        sector="wnba",
        outcome_a_label="Seattle Storm",
        outcome_b_label="Phoenix Mercury",
        outcome_a_decimal=2.0, outcome_b_decimal=2.0,
        true_prob_a=prob_a, true_prob_b=1 - prob_a,
        margin=0.03, spread_line=spread_line, event_date=tip,
    )


def total_sharp(total_line=164.5, p_over=0.60, tip=TIP) -> SharpOdds:
    return SharpOdds(
        event_id=f"{GAME_EVENT}::total::{total_line}",
        book=SharpBook.pinnacle,
        sector="wnba",
        outcome_a_label="Over", outcome_b_label="Under",
        outcome_a_decimal=2.0, outcome_b_decimal=2.0,
        true_prob_a=p_over, true_prob_b=1 - p_over,
        margin=0.03, total_line=total_line,
        true_prob_over=p_over, true_prob_under=1 - p_over,
        event_date=tip,
    )


def book(ask=0.30, ask_usd=100.0, bid=0.20, bid_usd=100.0) -> dict:
    return {
        "yes_ask": ask, "yes_ask_depth_usd": ask_usd,
        "yes_bid": bid, "yes_bid_depth_usd": bid_usd,
        "yes_book_usd": ask_usd + bid_usd, "no_book_usd": ask_usd + bid_usd,
        "source": "ws",
    }


class TestYesSide:
    def test_entry_priced_at_orderbook_ask_not_display_price(self):
        m = spread_market(yes_price=0.45)          # displayed price — must NOT be used
        books = {m.ticker: book(ask=0.30)}
        gaps = build_anchored_entries([m], books, [spread_sharp()], "wnba", now=NOW)
        lay = [g for g in gaps if g.market_id == m.id]
        assert len(lay) == 1
        # storm -4.5 @ 0.5 -> P(cover -6.5) ~ 0.4364; entry at the crossable ask
        assert lay[0].kalshi_yes_price == pytest.approx(0.30)
        assert lay[0].sharp_true_prob == pytest.approx(0.4364, abs=0.001)
        assert lay[0].blended_true_prob == lay[0].sharp_true_prob  # pure anchor
        assert lay[0].kelly_fraction == 0.0

    def test_sources_token_present(self):
        m = spread_market()
        gaps = build_anchored_entries(
            [m], {m.ticker: book(ask=0.30)}, [spread_sharp()], "wnba", now=NOW)
        assert gaps and "anchored_entry" in gaps[0].model_sources
        assert "spread_dist" in gaps[0].model_sources
        assert "possession_sim" not in gaps[0].model_sources  # contamination-safe

    def test_ev_gate_blocks_thin_edge(self):
        m = spread_market()
        gaps = build_anchored_entries(
            [m], {m.ticker: book(ask=0.43, bid=0.20)}, [spread_sharp()], "wnba", now=NOW)
        assert not [g for g in gaps if not g.market_id.endswith(":no")]

    def test_depth_gate_blocks_thin_book(self):
        m = spread_market()
        gaps = build_anchored_entries(
            [m], {m.ticker: book(ask=0.30, ask_usd=10.0)}, [spread_sharp()], "wnba", now=NOW)
        assert not [g for g in gaps if not g.market_id.endswith(":no")]

    def test_fallback_to_market_price_without_book(self):
        m = spread_market(yes_price=0.30)
        gaps = build_anchored_entries(
            [m], {}, [spread_sharp()], "wnba", now=NOW, depth_min_usd=0.0)
        lay = [g for g in gaps if g.market_id == m.id]
        assert len(lay) == 1
        assert lay[0].kalshi_yes_price == pytest.approx(0.30)


class TestNoSide:
    def test_spread_take_follows_scanner_conventions(self):
        # bid 0.55 above fair 0.4364 -> NO side +EV
        m = spread_market()
        gaps = build_anchored_entries(
            [m], {m.ticker: book(ask=0.60, bid=0.55)}, [spread_sharp()], "wnba", now=NOW)
        take = [g for g in gaps if g.market_id.endswith(":no")]
        assert len(take) == 1
        g = take[0]
        assert g.market_id == f"{m.id}:no"
        assert g.kalshi_yes_price == pytest.approx(0.45)      # 1 - bid
        assert g.line == pytest.approx(6.5)                    # negated
        assert g.yes_team == "mercury"                         # opponent
        assert g.sharp_true_prob == pytest.approx(1 - 0.4364, abs=0.001)
        assert "no_side" in g.model_sources and "anchored_entry" in g.model_sources

    def test_total_under_flips_side_not_line(self):
        m = total_market(yes_price=0.70)
        # yes_bid 0.65 -> under costs 0.35 vs fair_under 0.40 -> +5pp edge
        gaps = build_anchored_entries(
            [m], {m.ticker: book(ask=0.70, bid=0.65)}, [total_sharp()], "wnba", now=NOW)
        under = [g for g in gaps if g.market_id.endswith(":no")]
        assert len(under) == 1
        g = under[0]
        assert g.yes_team == "under"
        assert g.line == pytest.approx(164.5)                  # NOT flipped
        assert g.kalshi_yes_price == pytest.approx(0.35)

    def test_no_side_skipped_without_book(self):
        m = spread_market()
        gaps = build_anchored_entries(
            [m], {}, [spread_sharp()], "wnba", now=NOW, depth_min_usd=0.0)
        assert not [g for g in gaps if g.market_id.endswith(":no")]


class TestGates:
    def test_imminent_market_skipped(self):
        m = spread_market()
        near_tip = TIP - timedelta(minutes=10)
        gaps = build_anchored_entries(
            [m], {m.ticker: book(ask=0.30)}, [spread_sharp()], "wnba", now=near_tip)
        assert gaps == []

    def test_unanchored_game_skipped(self):
        m = spread_market()
        gaps = build_anchored_entries([m], {m.ticker: book(ask=0.30)}, [], "wnba", now=NOW)
        assert gaps == []

    def test_total_far_from_ladder_line_skipped(self):
        m = total_market(line=170.5)   # ladder only carries 164.5
        gaps = build_anchored_entries(
            [m], {m.ticker: book(ask=0.30)}, [total_sharp(164.5)], "wnba", now=NOW)
        assert gaps == []

    def test_moneyline_markets_ignored(self):
        m = spread_market()
        m = m.model_copy(update={"market_type": MarketType.moneyline})
        gaps = build_anchored_entries(
            [m], {m.ticker: book(ask=0.30)}, [spread_sharp()], "wnba", now=NOW)
        assert gaps == []


class TestOverSide:
    def test_total_over_priced_from_exact_ladder_line(self):
        m = total_market(yes_price=0.50)
        gaps = build_anchored_entries(
            [m], {m.ticker: book(ask=0.50, bid=0.40)}, [total_sharp(164.5, 0.60)],
            "wnba", now=NOW)
        over = [g for g in gaps if not g.market_id.endswith(":no")]
        assert len(over) == 1
        assert over[0].sharp_true_prob == pytest.approx(0.60)
        assert over[0].yes_team == "over"
        assert "anchored_entry" in over[0].model_sources
