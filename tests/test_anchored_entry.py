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


class TestNflSectorGeneric:
    """NFL spread flows through the same sector-generic path (--entry-sectors
    now defaults wnba,nfl). Proves the anchored-entry stream can accrue the NFL
    spread CLV sample without any WNBA-specific coupling."""

    def _nfl_spread_market(self, ask_line=-6.5, yes_team="patriots"):
        return PredictionMarket(
            id="kalshi:KXNFLSPREAD-26SEP13SEANE-NE6",
            source=MarketSource.kalshi,
            sector="nfl",
            market_type=MarketType.spread,
            ticker="KXNFLSPREAD-26SEP13SEANE-NE6",
            yes_price=0.28,
            no_price=0.77,
            team_home="New England Patriots", team_away="Seattle Seahawks",
            yes_team=yes_team, line=ask_line,
            event_date=NOW,
            volume_usd=100.0,
        )

    def _nfl_spread_sharp(self, prob_a=0.60, spread_line=-3.5):
        # Pinnacle: Patriots -3.5 favorite. event_id tokens normalize to the
        # same nicks the NFL handler produces for the market teams.
        return SharpOdds(
            event_id="nfl::2026-07-02::seahawks_vs_patriots::spread",
            book=SharpBook.pinnacle,
            sector="nfl",
            outcome_a_label="New England Patriots",
            outcome_b_label="Seattle Seahawks",
            outcome_a_decimal=2.0, outcome_b_decimal=2.0,
            true_prob_a=prob_a, true_prob_b=1 - prob_a,
            margin=0.03, spread_line=spread_line, event_date=TIP,
        )

    def test_nfl_spread_anchored_entry_built(self):
        m = self._nfl_spread_market()
        gaps = build_anchored_entries(
            [m], {m.ticker: book(ask=0.20)}, [self._nfl_spread_sharp()], "nfl", now=NOW)
        lay = [g for g in gaps if g.market_id == m.id]
        assert len(lay) == 1
        assert lay[0].sector == "nfl"
        assert lay[0].kalshi_yes_price == pytest.approx(0.20)  # crossable ask
        assert "anchored_entry" in lay[0].model_sources
        # NFL prices alt rungs with the key-number margin PMF (_PMF_SECTORS);
        # the token lets `--sources-token spread_pmf` isolate this stream.
        assert "spread_pmf" in lay[0].model_sources
        assert "spread_dist" not in lay[0].model_sources
        assert lay[0].kelly_fraction == 0.0  # shadow-only, never bankroll-sized

    def test_nfl_spread_anchored_entry_normal_fallback_token(self, monkeypatch):
        """With NFL removed from _PMF_SECTORS the stream reverts to the normal
        CDF and carries the spread_dist token again (one-line revert path)."""
        import evmax.models_ml.spread_distribution as sd

        monkeypatch.setattr(sd, "_PMF_SECTORS", set())
        m = self._nfl_spread_market()
        gaps = build_anchored_entries(
            [m], {m.ticker: book(ask=0.20)}, [self._nfl_spread_sharp()], "nfl", now=NOW)
        lay = [g for g in gaps if g.market_id == m.id]
        assert len(lay) == 1
        assert "spread_dist" in lay[0].model_sources
        assert "spread_pmf" not in lay[0].model_sources

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


# ---------------------------------------------------------------------------
# Early-totals stream (NFL, shadow): unposted rungs priced by the fitted model
# ---------------------------------------------------------------------------

NFL_GAME = "nfl::2026-11-22::seahawks_vs_patriots"
NFL_NOW = datetime(2026, 11, 19, 12, 0, tzinfo=timezone.utc)
NFL_TIP = datetime(2026, 11, 22, 18, 0, tzinfo=timezone.utc)      # 78h out


def _nfl_total_market(line=47.5, yes_price=0.40) -> PredictionMarket:
    ticker = f"KXNFLTOTAL-26NOV22SEANE-{int(line)}"
    return PredictionMarket(
        id=f"kalshi:{ticker}", source=MarketSource.kalshi, sector="nfl",
        market_type=MarketType.total, ticker=ticker,
        yes_price=yes_price, no_price=1 - yes_price + 0.03,
        team_home="patriots", team_away="seahawks", yes_team="over", line=line,
        event_date=datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc), volume_usd=100.0,
    )


def _nfl_total_sharp(total_line=44.5, p_over=0.50, tip=NFL_TIP) -> SharpOdds:
    return SharpOdds(
        event_id=f"{NFL_GAME}::total::{total_line}", book=SharpBook.pinnacle, sector="nfl",
        outcome_a_label="Over", outcome_b_label="Under",
        outcome_a_decimal=2.0, outcome_b_decimal=2.0,
        true_prob_a=p_over, true_prob_b=1 - p_over, margin=0.03, total_line=total_line,
        true_prob_over=p_over, true_prob_under=1 - p_over, event_date=tip,
    )


class TestEarlyTotals:
    @pytest.fixture(autouse=True)
    def _needs_artifact(self):
        from evmax.models_ml.line_pmf import load_line_pmf
        if load_line_pmf("nfl", "total") is None:
            pytest.skip("no fitted NFL total artifact")

    def _model_fair(self, line=47.5):
        from evmax.agents.cleanup.anchored_entry import _model_total_fair
        return _model_total_fair({44.5: (0.50, 0.50)}, "over", line, "nfl")

    def test_unposted_rung_is_model_priced_and_tagged(self):
        fair = self._model_fair()
        assert fair is not None and 0.30 < fair < 0.49      # 3 points above a 44.5 main
        m = _nfl_total_market()
        ask = round(fair - 0.04, 2)                           # ~2.5pp edge after the fee
        gaps = build_anchored_entries(
            [m], {m.ticker: book(ask=ask, bid=ask - 0.02)}, [_nfl_total_sharp()],
            "nfl", now=NFL_NOW)
        over = [g for g in gaps if g.market_id == m.id]
        assert len(over) == 1
        g = over[0]
        assert g.model_sources == "sharp+total_pmf+early_total"
        assert "anchored_entry" not in g.model_sources      # never pools into the 2pp gate
        assert g.sharp_true_prob == pytest.approx(fair)
        assert g.kalshi_yes_price == pytest.approx(ask)
        assert g.kelly_fraction == 0.0 and g.venue == "kalshi"

    def test_one_point_edge_threshold_not_two(self):
        fair = self._model_fair()
        m = _nfl_total_market()
        # ~1.3pp net edge: passes this stream's 1pp gate (the 2pp gate would block).
        from evmax.ev.calculator import effective_price
        ask = next(a / 100 for a in range(int(fair * 100), 0, -1)
                   if 1.0 <= (fair - effective_price(a / 100, "kalshi")) * 100 < 2.0)
        gaps = build_anchored_entries(
            [m], {m.ticker: book(ask=ask, bid=ask - 0.02)}, [_nfl_total_sharp()],
            "nfl", now=NFL_NOW)
        assert [g.market_id for g in gaps if g.market_id == m.id] == [m.id]
        # under 1pp: nothing
        thin = next(a / 100 for a in range(int(fair * 100) + 1, 0, -1)
                    if 0.0 < (fair - effective_price(a / 100, "kalshi")) * 100 < 1.0)
        gaps = build_anchored_entries(
            [m], {m.ticker: book(ask=thin, bid=thin - 0.02)}, [_nfl_total_sharp()],
            "nfl", now=NFL_NOW)
        assert [g for g in gaps if g.market_id == m.id] == []

    def test_inside_24h_the_stream_is_off(self):
        fair = self._model_fair()
        m = _nfl_total_market()
        late = NFL_TIP - timedelta(hours=23)
        gaps = build_anchored_entries(
            [m], {m.ticker: book(ask=round(fair - 0.06, 2), bid=0.20)},
            [_nfl_total_sharp()], "nfl", now=late)
        assert gaps == []                                    # 3 pts from the ladder: unpriced

    def test_posted_rung_stays_on_the_anchored_stream(self):
        m = _nfl_total_market(line=44.5)
        gaps = build_anchored_entries(
            [m], {m.ticker: book(ask=0.50, bid=0.45)}, [_nfl_total_sharp(44.5, 0.60)],
            "nfl", now=NFL_NOW)
        over = [g for g in gaps if g.market_id == m.id]
        assert len(over) == 1
        assert over[0].sharp_true_prob == pytest.approx(0.60)
        assert over[0].model_sources == "sharp+anchored_entry"

    def test_under_side_is_the_complement_with_no_side_token(self):
        fair = self._model_fair()
        m = _nfl_total_market()
        bid = round(fair + 0.05, 2)                           # NO costs 1-bid, fair NO = 1-fair
        gaps = build_anchored_entries(
            [m], {m.ticker: book(ask=0.99, bid=bid)}, [_nfl_total_sharp()],
            "nfl", now=NFL_NOW)
        no = [g for g in gaps if g.market_id == f"{m.id}:no"]
        assert len(no) == 1
        assert no[0].yes_team == "under" and no[0].line == 47.5
        assert no[0].sharp_true_prob == pytest.approx(1 - fair)
        assert no[0].model_sources == "sharp+total_pmf+no_side+early_total"

    def test_depth_gate_still_applies(self):
        fair = self._model_fair()
        m = _nfl_total_market()
        gaps = build_anchored_entries(
            [m], {m.ticker: book(ask=round(fair - 0.05, 2), ask_usd=10.0, bid=0.10)},
            [_nfl_total_sharp()], "nfl", now=NFL_NOW)
        assert [g for g in gaps if g.market_id == m.id] == []

    def test_other_sectors_and_missing_model_do_not_use_the_stream(self, monkeypatch):
        # wnba: an unposted rung far from the ladder stays unpriced (existing test
        # test_total_far_from_ladder_line_skipped); nfl without the fitted model too.
        import evmax.models_ml.total_distribution as td
        monkeypatch.setattr(td, "TOTAL_LADDER_SECTORS", set())      # normal-curve fallback
        m = _nfl_total_market()
        gaps = build_anchored_entries(
            [m], {m.ticker: book(ask=0.20, bid=0.10)}, [_nfl_total_sharp()],
            "nfl", now=NFL_NOW)
        assert gaps == []

    def test_token_is_registered_as_non_model(self):
        from evmax.agents.cleanup import improve_triggers, integrity
        assert "early_total" in integrity._NON_MODEL_SOURCE_TOKENS
        assert any("early_total" in v for v in vars(improve_triggers).values()
                   if isinstance(v, (set, frozenset, tuple, list)))
