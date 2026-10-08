"""NHL spreads price only Pinnacle's own posted puck-line rung (2026-10-07).

Off a −1.5 puck line (outcome_a lays 1.5), the venue contracts split in two:

* direct — "a wins by 2+" (a −1.5) and its complement "b +1.5". Pinnacle posts
  this rung, so the fair is the book's own devigged price.
* mirror — "b wins by 2+" (b −1.5) and its complement "a +1.5". Pinnacle does
  not post it; the normal CDF (σ 2.0) extrapolated it 3.0 goals out on the
  true margin axis and priced it ~10pp too high (Pinnacle's own quote for the
  same contract, when the favourite flipped: 0.71–0.72; the normal: ~0.82).

The mirror rung must go unpriced on BOTH sides; the direct rung must equal the
book's price and carry the ``puck_line`` token, never ``spread_dist``.
"""
from __future__ import annotations

import pytest

from evmax.agents.cleanup.contamination import is_contaminated
from evmax.agents.odds.ev_gap_agent import EVGapAgent
from evmax.models.market import MarketSource, MarketType, PredictionMarket
from evmax.models.odds import SharpBook, SharpOdds
from evmax.models_ml.spread_distribution import (
    PUCK_LINE_TOKEN,
    SPREAD_DIST_TOKEN,
    SpreadDistributionModel,
)

TOR, MTL = "toronto maple leafs", "montreal canadiens"
P_A = 0.30  # Pinnacle: Toronto −1.5 covers 30%; Montreal +1.5 covers 70%


def _puck_line(sector: str = "nhl", p_a: float = P_A) -> SharpOdds:
    return SharpOdds(
        event_id=f"{sector}::2026-10-08::toronto_maple_leafs_vs_montreal_canadiens::spread",
        book=SharpBook.pinnacle, sector=sector,
        outcome_a_label=TOR, outcome_b_label=MTL,
        outcome_a_decimal=1.0 / p_a, outcome_b_decimal=1.0 / (1.0 - p_a),
        true_prob_a=p_a, true_prob_b=round(1.0 - p_a, 6),
        spread_line=-1.5, margin=0.04,
    )


class TestModel:
    def setup_method(self):
        self.model = SpreadDistributionModel()

    def test_favourite_minus_1_5_is_the_book_price(self):
        pred = self.model.predict(_puck_line(), target_line=-1.5, sector="nhl",
                                  yes_is_underdog=False)
        assert pred is not None
        assert pred.true_prob == pytest.approx(P_A)
        assert pred.method == PUCK_LINE_TOKEN

    def test_underdog_plus_1_5_is_the_book_price(self):
        pred = self.model.predict(_puck_line(), target_line=1.5, sector="nhl",
                                  yes_is_underdog=True)
        assert pred is not None
        assert pred.true_prob == pytest.approx(1.0 - P_A)
        assert pred.method == PUCK_LINE_TOKEN

    def test_underdog_wins_by_2_is_not_priced(self):
        # Mirror rung: Montreal −1.5. Previously Φ put it at ~0.17.
        assert self.model.predict(_puck_line(), target_line=-1.5, sector="nhl",
                                  yes_is_underdog=True) is None

    def test_favourite_plus_1_5_is_not_priced(self):
        # Mirror complement: Toronto +1.5. Previously Φ put it at ~0.83.
        assert self.model.predict(_puck_line(), target_line=1.5, sector="nhl",
                                  yes_is_underdog=False) is None

    def test_baseball_mirror_rung_still_prices(self):
        # Only NHL is posted-rung-only; baseball keeps the folded gate.
        pred = self.model.predict(_puck_line("baseball"), target_line=-1.5,
                                  sector="baseball", yes_is_underdog=True)
        assert pred is not None
        assert pred.method == SPREAD_DIST_TOKEN


def _market(yes_team: str, yes_price: float, no_price: float) -> PredictionMarket:
    """Kalshi "{yes_team} wins by over 1.5 goals"."""
    return PredictionMarket(
        id=f"KXNHLSPREAD-26OCT08MTLTOR-{yes_team[:3].upper()}2",
        source=MarketSource.kalshi, sector="nhl", market_type=MarketType.spread,
        yes_price=yes_price, no_price=no_price,
        team_home=TOR, team_away=MTL, yes_team=yes_team, line=-1.5,
    )


def _evaluate(agent: EVGapAgent, market: PredictionMarket):
    return agent._evaluate_pair(
        market=market, sharp=_puck_line(), confidence=95.0, sector="nhl",
        blended_preds={}, injuries={}, model_sources={}, kelly_base=0.25,
        steam_events=set(), return_blend=True,
    )


class TestEvGap:
    def test_mirror_market_prices_neither_side(self):
        # "Montreal wins by over 1.5": YES = Montreal −1.5, NO = Toronto +1.5.
        # NO at 0.72 looked like +14% EV against the old ~0.83 fair.
        gap, payload = _evaluate(EVGapAgent(), _market(MTL, yes_price=0.30, no_price=0.72))
        assert gap is None
        assert payload is None  # caller skips the NO side on a None payload

    def test_direct_market_yes_side_uses_book_price(self):
        # "Toronto wins by over 1.5" at 0.25 vs the book's 0.30.
        gap, payload = _evaluate(EVGapAgent(), _market(TOR, yes_price=0.25, no_price=0.78))
        assert gap is not None and payload is not None
        assert gap.blended_true_prob == pytest.approx(P_A, abs=1e-6)
        assert PUCK_LINE_TOKEN in gap.model_sources
        assert SPREAD_DIST_TOKEN not in gap.model_sources

    def test_direct_market_no_side_uses_book_price(self):
        # NO on "Toronto wins by over 1.5" = Montreal +1.5 at 0.64 vs the book's 0.70.
        agent = EVGapAgent()
        market = _market(TOR, yes_price=0.38, no_price=0.64)
        _, payload = _evaluate(agent, market)
        assert payload is not None
        no_gap = agent._build_no_side_spread_gap(
            market=market, sharp=_puck_line(), blend_payload=payload,
            confidence=95.0, sector="nhl", kelly_base=0.25, steam_events=set(),
        )
        assert no_gap is not None
        assert no_gap.blended_true_prob == pytest.approx(1.0 - P_A, abs=1e-6)
        assert PUCK_LINE_TOKEN in no_gap.model_sources
        assert SPREAD_DIST_TOKEN not in no_gap.model_sources


class TestContamination:
    def test_pre_fix_spread_dist_rows_are_dated_out(self):
        assert is_contaminated("nhl", "spread", "sharp+spread_dist+no_side", 1.5)
        assert is_contaminated("nhl", "spread", "sharp+spread_dist", 1.5)

    def test_puck_line_rows_are_clean(self):
        assert not is_contaminated("nhl", "spread", "sharp+puck_line+no_side", 1.5)

    def test_moneyline_untouched(self):
        assert not is_contaminated("nhl", "moneyline", "elo+nhl_xg+sharp", None)
