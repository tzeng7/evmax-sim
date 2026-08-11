"""Tests for the EV calculator."""

import math

import pytest
from datetime import datetime, timezone

from evmax.ev.calculator import (
    calculate_ev,
    dual_ev,
    effective_price,
    evaluate_market,
    max_maker_limit_price,
    suggested_maker_bid,
)
from evmax.ev.kelly import compute_kelly
from evmax.fees import kalshi_fee_prob, polymarket_us_fee_prob
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


class TestEffectivePrice:
    def test_kalshi_taker_adds_quadratic_fee(self):
        """Kalshi taker fee = 0.07·p·(1-p); at p=0.50 that is 1.75¢."""
        eff = effective_price(0.50, "kalshi")
        assert math.isclose(eff, 0.50 + 0.0175, abs_tol=1e-12)
        assert math.isclose(eff - 0.50, kalshi_fee_prob(0.50), abs_tol=1e-12)

    def test_none_venue_is_passthrough(self):
        """venue=None returns the raw price unchanged (gross-of-fee callers)."""
        assert effective_price(0.50, None) == 0.50
        assert effective_price(0.42) == 0.42  # default venue is None

    def test_unknown_venue_degrades_to_gross(self):
        """An unrecognised venue name must not raise on the hot path."""
        assert effective_price(0.50, "betfair") == 0.50

    def test_degenerate_prices_passthrough(self):
        """0/1 and out-of-range prices are returned unchanged."""
        assert effective_price(0.0, "kalshi") == 0.0
        assert effective_price(1.0, "kalshi") == 1.0
        assert effective_price(-0.1, "kalshi") == -0.1

    def test_polymarket_taker_cheaper_than_kalshi(self):
        """PolyUS taker theta (0.06) < Kalshi taker (0.07) at the same price."""
        assert effective_price(0.50, "polymarket_us") < effective_price(0.50, "kalshi")
        assert math.isclose(
            effective_price(0.50, "polymarket_us") - 0.50,
            polymarket_us_fee_prob(0.50),
            abs_tol=1e-12,
        )

    def test_polymarket_maker_rebate_lowers_cost(self):
        """PolyUS maker fee is a rebate (negative) → effective cost below the ask."""
        eff = effective_price(0.50, "polymarket_us", maker=True)
        assert eff < 0.50
        assert math.isclose(eff, 0.50 + polymarket_us_fee_prob(0.50, maker=True), abs_tol=1e-12)

    def test_fee_is_symmetric_across_yes_no(self):
        """fee(p) == fee(1-p): passing the NO-side ask needs no special handling."""
        yes_fee = effective_price(0.30, "kalshi") - 0.30
        no_fee = effective_price(0.70, "kalshi") - 0.70
        assert math.isclose(yes_fee, no_fee, abs_tol=1e-12)

    def test_fee_peaks_at_coinflip(self):
        """The quadratic fee is largest at p=0.50 and shrinks toward the extremes."""
        f_50 = effective_price(0.50, "kalshi") - 0.50
        f_75 = effective_price(0.75, "kalshi") - 0.75
        f_90 = effective_price(0.90, "kalshi") - 0.90
        assert f_50 > f_75 > f_90

    def test_high_price_clamped_below_one(self):
        """Adding the fee near p=0.99 must not push the effective price to/above 1."""
        assert effective_price(0.99, "kalshi") < 1.0


class TestNetOfFeeEV:
    def test_fee_reduces_ev_below_gate(self):
        """A 4% gross coin-flip edge nets ~0.5% after the Kalshi fee — below 2%."""
        true_prob = 0.52
        gross_ev, _ = calculate_ev(0.50, true_prob)
        net_ev, _ = calculate_ev(effective_price(0.50, "kalshi"), true_prob)
        assert gross_ev > 0.02  # would pass the default gate gross
        assert net_ev < gross_ev
        assert net_ev < 0.02  # fails it net — the whole point of the change

    def test_net_ev_drag_smaller_on_chalk(self):
        """Fee drag on a heavy favorite is far smaller than on a coin flip."""
        drag_coin = (
            calculate_ev(0.50, 0.52)[0]
            - calculate_ev(effective_price(0.50, "kalshi"), 0.52)[0]
        )
        drag_chalk = (
            calculate_ev(0.90, 0.92)[0]
            - calculate_ev(effective_price(0.90, "kalshi"), 0.92)[0]
        )
        assert drag_coin > drag_chalk

    def test_kelly_stake_shrinks_net_of_fee(self):
        """Sizing on the fee-inclusive price stakes strictly less than gross."""
        true_prob = 0.56
        gross_payout = 1.0 / 0.50
        net_payout = 1.0 / effective_price(0.50, "kalshi")
        gross = compute_kelly(true_prob, gross_payout, edge_pct=0.12, base_fraction=0.5)
        net = compute_kelly(true_prob, net_payout, edge_pct=0.12, base_fraction=0.5)
        assert net.kelly_fraction < gross.kelly_fraction
        assert net.kelly_fraction > 0  # still a real edge here, just smaller


class TestDualEV:
    def test_taker_clears_is_not_maker_only(self):
        """When the taker EV already clears the floor, maker_only is False."""
        r = dual_ev(0.50, 0.55, "kalshi", 0.02)
        assert r.passes is True
        assert r.maker_only is False
        assert r.taker_ev >= 0.02
        assert r.maker_ev >= r.taker_ev  # maker fee never larger than taker

    def test_maker_only_when_fee_straddles_floor(self):
        """Taker EV below the floor but maker EV above it → maker_only play."""
        # p=0.50, prob=0.525: taker nets ~1.4% (<2%), maker nets ~4% (>=2%).
        r = dual_ev(0.50, 0.525, "kalshi", 0.02)
        assert r.taker_ev < 0.02 <= r.maker_ev
        assert r.maker_only is True
        assert r.passes is True

    def test_neither_clears_does_not_pass(self):
        """A gross edge too small even for the maker fee fails the gate entirely."""
        r = dual_ev(0.50, 0.505, "kalshi", 0.02)
        assert r.maker_ev < 0.02
        assert r.passes is False
        assert r.maker_only is False

    def test_fees_off_collapses_both_modes(self):
        """venue=None makes maker == taker and can never be maker_only."""
        r = dual_ev(0.50, 0.53, None, 0.02)
        assert math.isclose(r.taker_ev, r.maker_ev, abs_tol=1e-12)
        assert r.maker_only is False
        # reduces to the pre-fee gate
        assert r.passes is (calculate_ev(0.50, 0.53)[0] >= 0.02)

    def test_payouts_match_effective_prices(self):
        """taker/maker payout = 1 / effective price under each mode."""
        r = dual_ev(0.60, 0.66, "kalshi", 0.02)
        assert math.isclose(r.taker_payout, 1.0 / effective_price(0.60, "kalshi"), abs_tol=1e-12)
        assert math.isclose(
            r.maker_payout, 1.0 / effective_price(0.60, "kalshi", maker=True), abs_tol=1e-12
        )


class TestMakerLimitPrice:
    def test_resting_at_limit_yields_floor_ev(self):
        L = max_maker_limit_price(0.55, "kalshi", 0.02)
        assert L is not None
        ev_at_limit, _ = calculate_ev(effective_price(L, "kalshi", maker=True), 0.55)
        assert ev_at_limit == pytest.approx(0.02, abs=1e-3)

    def test_below_limit_beats_floor_above_misses(self):
        L = max_maker_limit_price(0.55, "kalshi", 0.02)
        below, _ = calculate_ev(effective_price(L - 0.02, "kalshi", maker=True), 0.55)
        above, _ = calculate_ev(effective_price(L + 0.02, "kalshi", maker=True), 0.55)
        assert below > 0.02 > above

    def test_limit_is_below_true_prob(self):
        # You never rest a buy above fair value.
        assert max_maker_limit_price(0.55, "kalshi", 0.02) < 0.55

    def test_fees_off_is_fair_over_one_plus_floor(self):
        L = max_maker_limit_price(0.60, None, 0.02)
        assert L == pytest.approx(0.60 / 1.02, abs=1e-3)

    def test_maker_limit_above_taker_break_even(self):
        # The maker fee is smaller than the taker fee, so you can rest higher and
        # still clear the floor than you could paying the taker ask.
        target = 0.55 / 1.02
        maker_L = max_maker_limit_price(0.55, "kalshi", 0.02)
        # taker break-even for the same floor: eff_taker(L)=target
        assert effective_price(maker_L, "kalshi", maker=True) == pytest.approx(target, abs=1e-3)
        assert effective_price(maker_L, "kalshi", maker=False) > target  # taker would miss here

    def test_degenerate_prob_returns_none(self):
        assert max_maker_limit_price(0.0, "kalshi", 0.02) is None
        assert max_maker_limit_price(1.0, "kalshi", 0.02) is None


class TestSuggestedMakerBid:
    # Screenshot regime: fair 0.568, ask 0.54, floor 0.02 → ceiling ~0.5525.
    LIMIT = max_maker_limit_price(0.568, "kalshi", 0.02)

    def test_improves_best_bid_by_one_tick(self):
        # Best bid 0.51, ask 0.54 → rest one tick up at 0.52 (queue priority,
        # still below the ask, still under the +EV ceiling).
        assert suggested_maker_bid(0.51, 0.54, self.LIMIT) == pytest.approx(0.52)

    def test_never_crosses_the_ask(self):
        # 1-tick spread (bid 0.53, ask 0.54): improving would cross, so join the
        # bid at 0.53 instead.
        assert suggested_maker_bid(0.53, 0.54, self.LIMIT) == pytest.approx(0.53)

    def test_result_is_strictly_below_ask(self):
        for bid in (0.30, 0.45, 0.50, 0.53):
            rest = suggested_maker_bid(bid, 0.54, self.LIMIT)
            assert rest is not None and rest < 0.54

    def test_never_exceeds_the_maker_ceiling(self):
        # Wide book, bid near the ceiling: capped at the floored ceiling, never
        # above it (would drop below the EV floor).
        rest = suggested_maker_bid(0.55, 0.60, self.LIMIT)
        assert rest is not None
        assert rest <= math.floor(self.LIMIT * 100) / 100.0

    def test_none_when_bid_past_ceiling(self):
        # Best bid already above the +EV maker ceiling → no fillable +EV rest.
        assert suggested_maker_bid(0.56, 0.60, self.LIMIT) is None

    def test_none_on_missing_inputs(self):
        assert suggested_maker_bid(None, 0.54, self.LIMIT) is None       # no bid ladder
        assert suggested_maker_bid(0.51, 0.54, None) is None            # fees off
        assert suggested_maker_bid(0.60, 0.54, self.LIMIT) is None      # bid >= ask (degenerate)

    def test_rest_price_clears_ev_floor_as_maker(self):
        # Whatever price it returns, filling there as a maker is >= the floor.
        rest = suggested_maker_bid(0.51, 0.54, self.LIMIT)
        ev, _ = calculate_ev(effective_price(rest, "kalshi", maker=True), 0.568)
        assert ev >= 0.02


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
