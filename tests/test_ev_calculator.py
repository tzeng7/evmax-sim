"""Tests for the EV calculator."""

import pytest
from datetime import datetime, timezone

from evmax.ev.calculator import calculate_ev, evaluate_market
from evmax.models.market import MarketSource, MarketType, PredictionMarket
from evmax.models.odds import SharpBook, SharpOdds


def make_market(yes_price: float, no_price: float, sector: str = "nfl") -> PredictionMarket:
    return PredictionMarket(
        id=f"kalshi:TEST-{sector}",
        source=MarketSource.kalshi,
        sector=sector,
        yes_price=yes_price,
        no_price=no_price,
        event_date=datetime(2026, 2, 22, tzinfo=timezone.utc),
        team_home="Team A",
        team_away="Team B",
    )


def make_sharp_odds(
    prob_a: float,
    prob_b: float,
    sector: str = "nfl",
) -> SharpOdds:
    return SharpOdds(
        event_id=f"{sector}::2026-02-22::team_a_vs_team_b",
        book=SharpBook.pinnacle,
        sector=sector,
        true_prob_a=prob_a,
        true_prob_b=prob_b,
        outcome_a_decimal=1.0 / prob_a if prob_a > 0 else 2.0,
        outcome_b_decimal=1.0 / prob_b if prob_b > 0 else 2.0,
        margin=0.03,
    )


class TestCalculateEV:
    def test_positive_ev(self):
        """Market price underestimates true probability → positive EV."""
        # True prob 55%, market prices it at 50% (0.50 implied)
        ev, edge = calculate_ev(market_price=0.50, true_prob=0.55)
        assert ev > 0
        assert edge > 0

    def test_negative_ev(self):
        """Market price overestimates true probability → negative EV."""
        # True prob 45%, market prices it at 50%
        ev, edge = calculate_ev(market_price=0.50, true_prob=0.45)
        assert ev < 0

    def test_zero_ev(self):
        """Fair price = true prob → zero EV."""
        ev, edge = calculate_ev(market_price=0.50, true_prob=0.50)
        assert abs(ev) < 1e-10

    def test_ev_formula(self):
        """EV = (true_prob × payout) - 1."""
        market_price = 0.40
        true_prob = 0.50
        payout = 1.0 / market_price  # 2.5
        expected_ev = (true_prob * payout) - 1.0  # 0.25
        ev, _ = calculate_ev(market_price, true_prob)
        assert abs(ev - expected_ev) < 1e-10

    def test_invalid_price_returns_zero(self):
        """Invalid prices return 0 EV."""
        ev, edge = calculate_ev(0.0, 0.5)
        assert ev == 0.0
        ev, edge = calculate_ev(1.0, 0.5)
        assert ev == 0.0


class TestEvaluateMarket:
    def test_finds_yes_ev(self):
        """Should find EV when yes_price is below true prob."""
        market = make_market(yes_price=0.45, no_price=0.55)
        sharp = make_sharp_odds(prob_a=0.52, prob_b=0.48)

        results = evaluate_market(market, sharp, ev_threshold=0.02)
        yes_results = [r for r in results if r.outcome == "yes"]
        assert len(yes_results) == 1
        assert yes_results[0].ev > 0.02

    def test_no_side_not_evaluated(self):
        """NO side is never evaluated — each outcome has its own YES market on Kalshi.
        Evaluating NO sides would double-count the same position via the opponent's YES market."""
        market = make_market(yes_price=0.60, no_price=0.40)
        sharp = make_sharp_odds(prob_a=0.55, prob_b=0.45)

        results = evaluate_market(market, sharp, ev_threshold=0.02)
        no_results = [r for r in results if r.outcome == "no"]
        assert len(no_results) == 0, "NO side should never be returned"

    def test_no_ev_fair_market(self):
        """Fair market: market prices match true probs exactly → no +EV."""
        # Yes price = 0.50, no price = 0.50, true_prob_a = 0.50, true_prob_b = 0.50
        # YES: EV = (0.50 × 2.0) - 1 = 0.0
        # NO:  EV = (0.50 × 2.0) - 1 = 0.0
        # Neither exceeds 2% threshold
        market = make_market(yes_price=0.50, no_price=0.50)
        sharp = make_sharp_odds(prob_a=0.50, prob_b=0.50)

        results = evaluate_market(market, sharp, ev_threshold=0.02)
        assert len(results) == 0

    def test_threshold_filtering(self):
        """Only return results above threshold."""
        market = make_market(yes_price=0.49, no_price=0.51)
        sharp = make_sharp_odds(prob_a=0.50, prob_b=0.50)

        # At threshold 5%, the 2% edge shouldn't appear
        results = evaluate_market(market, sharp, ev_threshold=0.05)
        assert len(results) == 0

        # At threshold 1%, it should appear
        results = evaluate_market(market, sharp, ev_threshold=0.01)
        assert len(results) >= 1
