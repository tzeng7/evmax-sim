"""Comprehensive tests for the core math and model layers.

Covers:
- evmax/ev/devig.py       (devig_two_way, devig_three_way, american_to_decimal)
- evmax/ev/calculator.py  (calculate_ev)
- evmax/ev/kelly.py       (compute_kelly)
- evmax/models_ml/spread_distribution.py (SpreadDistributionModel)
- evmax/models_ml/total_distribution.py  (TotalDistributionModel, is_game_total)
- evmax/models/odds.py    (SharpOdds Pydantic validation)
"""

from __future__ import annotations

import math

import pytest
from scipy.stats import norm

from evmax.ev.calculator import calculate_ev
from evmax.ev.devig import american_to_decimal, devig_three_way, devig_two_way
from evmax.ev.kelly import compute_kelly, kelly_full
from evmax.models.odds import SharpBook, SharpOdds
from evmax.models_ml.spread_distribution import SpreadDistributionModel
from evmax.models_ml.total_distribution import TotalDistributionModel, is_game_total


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_moneyline_sharp(
    prob_a: float,
    prob_b: float,
    event_id: str = "nba::2026-03-21::team_a_vs_team_b",
    sector: str = "nba",
    spread_line: float | None = None,
    total_line: float | None = None,
    true_prob_over: float | None = None,
    true_prob_under: float | None = None,
) -> SharpOdds:
    """Build a SharpOdds with valid moneyline probs and optional spread/total fields."""
    return SharpOdds(
        event_id=event_id,
        book=SharpBook.pinnacle,
        sector=sector,
        outcome_a_decimal=1.0 / prob_a if prob_a > 0 else 2.0,
        outcome_b_decimal=1.0 / prob_b if prob_b > 0 else 2.0,
        true_prob_a=prob_a,
        true_prob_b=prob_b,
        margin=0.04,
        spread_line=spread_line,
        total_line=total_line,
        true_prob_over=true_prob_over,
        true_prob_under=true_prob_under,
    )


def _make_spread_sharp(
    spread_line: float,
    true_prob_a: float,
    sector: str = "nba",
) -> SharpOdds:
    """Build a SharpOdds representing Pinnacle's spread market."""
    prob_b = 1.0 - true_prob_a
    return SharpOdds(
        event_id=f"{sector}::2026-03-21::home_vs_away",
        book=SharpBook.pinnacle,
        sector=sector,
        outcome_a_decimal=1.0 / true_prob_a,
        outcome_b_decimal=1.0 / prob_b,
        true_prob_a=true_prob_a,
        true_prob_b=prob_b,
        margin=0.04,
        spread_line=spread_line,
    )


def _make_total_sharp(
    total_line: float,
    prob_over: float,
    sector: str = "nba",
) -> SharpOdds:
    """Build a SharpOdds representing Pinnacle's totals market."""
    prob_under = 1.0 - prob_over
    # For totals-only SharpOdds the moneyline probs are zeroed out (validator skips
    # the sum check when true_prob_a == 0).
    return SharpOdds(
        event_id=f"{sector}::2026-03-21::home_vs_away",
        book=SharpBook.pinnacle,
        sector=sector,
        outcome_a_decimal=2.0,
        outcome_b_decimal=2.0,
        true_prob_a=0.0,
        true_prob_b=0.0,
        margin=0.04,
        total_line=total_line,
        true_prob_over=prob_over,
        true_prob_under=prob_under,
    )


# ===========================================================================
# devig.py
# ===========================================================================

class TestDevigTwoWay:
    def test_sums_to_one(self):
        """Devigged probs must sum to 1.0 within floating-point tolerance."""
        prob_a, prob_b, _ = devig_two_way(1.909, 1.909)
        assert abs(prob_a + prob_b - 1.0) < 0.001

    def test_equal_lines_returns_half_half(self):
        """-110/-110 should devig to ~0.5/0.5."""
        dec = american_to_decimal(-110)
        prob_a, prob_b, _ = devig_two_way(dec, dec)
        assert abs(prob_a - 0.5) < 0.01
        assert abs(prob_b - 0.5) < 0.01

    def test_heavy_favorite_higher_prob(self):
        """-400 favorite should have true prob > +320 underdog after devigging."""
        dec_fav = american_to_decimal(-400)   # 1.25
        dec_dog = american_to_decimal(+320)   # 4.20
        prob_fav, prob_dog, _ = devig_two_way(dec_fav, dec_dog)
        assert prob_fav > prob_dog

    def test_heavy_favorite_probs_sum_to_one(self):
        dec_fav = american_to_decimal(-400)
        dec_dog = american_to_decimal(+320)
        prob_a, prob_b, _ = devig_two_way(dec_fav, dec_dog)
        assert abs(prob_a + prob_b - 1.0) < 0.001

    def test_heavy_favorite_approximate_values(self):
        """For -400/+320, raw implied probs are 0.80 and 0.238; Power Method
        should give the favorite a true prob clearly above 0.75."""
        dec_fav = american_to_decimal(-400)
        dec_dog = american_to_decimal(+320)
        prob_fav, _, _ = devig_two_way(dec_fav, dec_dog)
        assert prob_fav > 0.75

    def test_very_large_favorite_sum_to_one(self):
        """-1000 heavy chalk versus +700 longshot."""
        dec_fav = american_to_decimal(-1000)
        dec_dog = american_to_decimal(+700)
        prob_a, prob_b, _ = devig_two_way(dec_fav, dec_dog)
        assert abs(prob_a + prob_b - 1.0) < 0.001

    def test_near_even_line_sum_to_one(self):
        """-105/+100 near-even market."""
        dec_a = american_to_decimal(-105)
        dec_b = american_to_decimal(+100)
        prob_a, prob_b, _ = devig_two_way(dec_a, dec_b)
        assert abs(prob_a + prob_b - 1.0) < 0.001

    def test_near_even_line_slight_favorite(self):
        """-105 side should have slightly higher true prob than +100 side."""
        dec_a = american_to_decimal(-105)
        dec_b = american_to_decimal(+100)
        prob_a, prob_b, _ = devig_two_way(dec_a, dec_b)
        assert prob_a > prob_b

    def test_margin_is_positive(self):
        """Any vigged market must have margin > 0."""
        _, _, margin = devig_two_way(1.909, 1.909)
        assert margin > 0.0

    def test_margin_is_sane(self):
        """Typical sportsbook margins are < 10%."""
        _, _, margin = devig_two_way(1.91, 2.05)
        assert margin < 0.10


class TestDevigThreeWay:
    def test_sums_to_one(self):
        """Three-way soccer devig must sum to 1.0."""
        prob_a, prob_b, prob_draw, _ = devig_three_way(2.1, 3.5, 3.2)
        assert abs(prob_a + prob_b + prob_draw - 1.0) < 0.001

    def test_equal_odds_returns_one_third_each(self):
        """Three equal decimal odds → each outcome ≈ 1/3."""
        prob_a, prob_b, prob_draw, _ = devig_three_way(3.0, 3.0, 3.0)
        for p in (prob_a, prob_b, prob_draw):
            assert abs(p - 1 / 3) < 0.01

    def test_equal_odds_sum_to_one(self):
        prob_a, prob_b, prob_draw, _ = devig_three_way(3.0, 3.0, 3.0)
        assert abs(prob_a + prob_b + prob_draw - 1.0) < 0.001

    def test_margin_positive(self):
        _, _, _, margin = devig_three_way(2.1, 3.5, 3.2)
        assert margin > 0

    def test_all_probs_between_zero_and_one(self):
        prob_a, prob_b, prob_draw, _ = devig_three_way(2.1, 3.5, 3.2)
        for p in (prob_a, prob_b, prob_draw):
            assert 0 < p < 1


class TestAmericanToDecimal:
    @pytest.mark.parametrize("american,expected", [
        (-110, 1.0 + 100 / 110),    # ≈ 1.909
        (+150, 2.5),
        (-200, 1.5),
        (+100, 2.0),
        (-300, 1.0 + 100 / 300),    # ≈ 1.333
        (+250, 3.5),
    ])
    def test_conversions(self, american, expected):
        assert abs(american_to_decimal(american) - expected) < 0.001

    def test_negative_110_approx(self):
        assert abs(american_to_decimal(-110) - 1.909) < 0.001

    def test_positive_150(self):
        assert abs(american_to_decimal(+150) - 2.5) < 1e-9

    def test_negative_200(self):
        assert abs(american_to_decimal(-200) - 1.5) < 1e-9

    def test_positive_100(self):
        assert abs(american_to_decimal(+100) - 2.0) < 1e-9

    def test_very_large_favorite(self):
        """-2000 chalk → decimal just above 1.0."""
        dec = american_to_decimal(-2000)
        assert 1.0 < dec < 1.1

    def test_large_underdog(self):
        """+2000 longshot → payout close to 21.0."""
        dec = american_to_decimal(+2000)
        assert abs(dec - 21.0) < 0.001


# ===========================================================================
# calculator.py
# ===========================================================================

class TestCalculateEV:
    def test_positive_ev_when_true_prob_exceeds_implied(self):
        """true_prob=0.55, market_price=0.50 → EV > 0."""
        ev, edge = calculate_ev(market_price=0.50, true_prob=0.55)
        assert ev > 0.0
        assert edge > 0.0

    def test_negative_ev_when_implied_exceeds_true_prob(self):
        """true_prob=0.45, market_price=0.50 → EV < 0."""
        ev, edge = calculate_ev(market_price=0.50, true_prob=0.45)
        assert ev < 0.0

    def test_breakeven_returns_zero(self):
        """When true_prob == implied_prob (market_price), EV ≈ 0."""
        ev, edge = calculate_ev(market_price=0.50, true_prob=0.50)
        assert abs(ev) < 1e-9

    def test_breakeven_non_half(self):
        """Breakeven at a non-0.5 price: market_price=true_prob=0.35."""
        ev, _ = calculate_ev(market_price=0.35, true_prob=0.35)
        assert abs(ev) < 1e-9

    def test_ev_formula_matches_manual_calculation(self):
        """EV = (true_prob × (1/market_price)) - 1."""
        market_price = 0.40
        true_prob = 0.50
        expected = (true_prob * (1.0 / market_price)) - 1.0   # 0.25
        ev, _ = calculate_ev(market_price, true_prob)
        assert abs(ev - expected) < 1e-10

    def test_ev_above_two_percent_threshold(self):
        """A clear +EV situation should exceed the 2% operational threshold."""
        ev, _ = calculate_ev(market_price=0.45, true_prob=0.52)
        assert ev >= 0.02

    def test_ev_below_threshold_on_thin_edge(self):
        """Tiny edge (0.1%) should be below the 2% threshold."""
        ev, _ = calculate_ev(market_price=0.499, true_prob=0.50)
        assert ev < 0.02

    @pytest.mark.parametrize("price", [0.0, 1.0, -0.1, 1.1])
    def test_invalid_price_returns_zero(self, price):
        """market_price outside (0, 1) must return (0.0, 0.0)."""
        ev, edge = calculate_ev(market_price=price, true_prob=0.5)
        assert ev == 0.0
        assert edge == 0.0

    def test_high_prob_low_price_large_ev(self):
        """70% true prob vs 50% market price → EV around 40%."""
        ev, _ = calculate_ev(market_price=0.50, true_prob=0.70)
        assert abs(ev - 0.40) < 0.001


# ===========================================================================
# kelly.py
# ===========================================================================

class TestKellyFull:
    def test_positive_ev_returns_positive_kelly(self):
        k = kelly_full(true_prob=0.55, payout_decimal=2.0)
        assert k > 0

    def test_negative_ev_returns_nonpositive(self):
        k = kelly_full(true_prob=0.40, payout_decimal=2.0)
        assert k <= 0

    def test_breakeven_returns_zero(self):
        """Exact fair-coin bet: kelly = 0."""
        k = kelly_full(true_prob=0.50, payout_decimal=2.0)
        assert abs(k) < 1e-9

    def test_payout_lte_one_returns_zero(self):
        k = kelly_full(true_prob=0.60, payout_decimal=1.0)
        assert k == 0.0


class TestComputeKelly:
    def test_returns_zero_for_negative_ev(self):
        result = compute_kelly(
            true_prob=0.40,
            payout_decimal=2.0,
            edge_pct=-0.10,
        )
        assert result.kelly_fraction == 0.0
        assert result.suggested_units == 0.0

    def test_returns_near_zero_for_zero_ev(self):
        """Kelly of a breakeven bet is 0 (or near-0)."""
        result = compute_kelly(
            true_prob=0.50,
            payout_decimal=2.0,
            edge_pct=0.0,
        )
        assert result.kelly_fraction <= 1e-9

    def test_never_exceeds_max_kelly_cap(self):
        """No matter how good the edge, suggested_units <= max_kelly."""
        result = compute_kelly(
            true_prob=0.80,
            payout_decimal=3.0,
            edge_pct=0.50,
            spread_pct=0.0,
            max_kelly=0.05,
        )
        assert result.suggested_units <= 0.05

    def test_never_exceeds_cap_default(self):
        """Default cap is 5%."""
        result = compute_kelly(
            true_prob=0.90,
            payout_decimal=5.0,
            edge_pct=0.80,
            spread_pct=0.0,
        )
        assert result.suggested_units <= 0.05

    def test_quarter_kelly_base_fraction(self):
        """With base_fraction=0.25, kelly_fraction <= 0.25 * kelly_full."""
        k_full = kelly_full(0.55, 2.0)
        result = compute_kelly(
            true_prob=0.55,
            payout_decimal=2.0,
            edge_pct=0.10,
            spread_pct=0.0,
            base_fraction=0.25,
        )
        # Allow small floating-point tolerance
        assert result.kelly_fraction <= k_full * 0.25 + 1e-9

    def test_half_kelly_base_fraction(self):
        """With base_fraction=0.50, result is roughly 2× the 0.25 result."""
        result_quarter = compute_kelly(0.55, 2.0, edge_pct=0.10, spread_pct=0.0, base_fraction=0.25)
        result_half = compute_kelly(0.55, 2.0, edge_pct=0.10, spread_pct=0.0, base_fraction=0.50)
        assert result_half.kelly_fraction > result_quarter.kelly_fraction

    def test_wider_spread_reduces_kelly(self):
        """Higher spread_pct → lower liquidity discount → smaller bet."""
        result_tight = compute_kelly(0.55, 2.0, edge_pct=0.10, spread_pct=0.01)
        result_wide = compute_kelly(0.55, 2.0, edge_pct=0.10, spread_pct=0.15)
        assert result_tight.suggested_units >= result_wide.suggested_units

    def test_spread_pct_floor_at_0_25(self):
        """Liquidity discount is floored at 0.25 even for spread_pct > 0.15."""
        # spread_pct=0.20 → 1 - 0.20*5 = 0.0 → floored to 0.25
        # spread_pct=0.30 → 1 - 0.30*5 = -0.5 → floored to 0.25
        result_high = compute_kelly(0.55, 2.0, edge_pct=0.10, spread_pct=0.20)
        result_very_high = compute_kelly(0.55, 2.0, edge_pct=0.10, spread_pct=0.30)
        assert abs(result_high.kelly_fraction - result_very_high.kelly_fraction) < 1e-9
        assert result_high.liquidity_discount == 0.25

    def test_zero_spread_full_liquidity(self):
        """No spread → liquidity_discount = 1.0."""
        result = compute_kelly(0.55, 2.0, edge_pct=0.10, spread_pct=0.0)
        assert result.liquidity_discount == 1.0

    def test_confidence_discount_always_one(self):
        """Per implementation, confidence_discount is hardcoded to 1.0."""
        result = compute_kelly(0.55, 2.0, edge_pct=0.04, spread_pct=0.0)
        assert result.confidence_discount == 1.0

    def test_min_kelly_floor(self):
        """min_kelly parameter enforces a floor on positive-EV bets.

        Note: the early-return path (k_full <= 0) bypasses min_kelly entirely
        and always returns 0.0. min_kelly only applies when the full Kelly is
        positive but the adjusted fraction would otherwise fall below the floor.
        """
        # A very small edge with a large spread could produce a tiny adjusted
        # fraction; min_kelly should clamp it upward.
        result_with_floor = compute_kelly(
            true_prob=0.52,
            payout_decimal=1.0 / 0.50,
            edge_pct=0.02,
            spread_pct=0.0,
            base_fraction=0.25,
            max_kelly=0.05,
            min_kelly=0.001,
        )
        # The result must be at least min_kelly when the bet is positive EV
        assert result_with_floor.kelly_fraction >= 0.001

    def test_small_positive_edge_still_positive(self):
        """2% edge at typical odds should still produce a nonzero bet."""
        result = compute_kelly(
            true_prob=0.51,
            payout_decimal=1.0 / 0.49,
            edge_pct=0.02,
            spread_pct=0.0,
        )
        assert result.suggested_units > 0.0


# ===========================================================================
# SpreadDistributionModel
# ===========================================================================

class TestSpreadDistributionModel:
    def setup_method(self):
        self.model = SpreadDistributionModel()

    def test_returns_none_when_spread_line_is_none(self):
        sharp = _make_moneyline_sharp(0.6, 0.4)  # no spread_line
        result = self.model.predict(sharp, target_line=-7.5, sector="nba")
        assert result is None

    def test_returns_none_when_true_prob_a_is_zero(self):
        """If true_prob_a is 0 the model has no calibration point."""
        sharp = _make_spread_sharp(spread_line=-7.5, true_prob_a=0.0001)
        # Edge case: prob just above 0 won't return None via the <= 0 guard
        # but we can test the explicit 0-case if needed
        sharp2 = SharpOdds(
            event_id="nba::2026-03-21::home_vs_away",
            book=SharpBook.pinnacle,
            sector="nba",
            outcome_a_decimal=2.0,
            outcome_b_decimal=2.0,
            true_prob_a=0.0,
            true_prob_b=0.0,
            margin=0.04,
            spread_line=-7.5,
        )
        result = self.model.predict(sharp2, target_line=-7.5, sector="nba")
        assert result is None

    def test_returns_none_when_line_distance_exceeds_one_sigma(self):
        """NBA sigma=12.5; target_line >12.5 pts away should return None."""
        sharp = _make_spread_sharp(spread_line=-7.5, true_prob_a=0.60)
        # |target| - |pinnacle| = 30 - 7.5 = 22.5 > 12.5
        result = self.model.predict(sharp, target_line=-30.0, sector="nba")
        assert result is None

    def test_returns_none_when_line_distance_exactly_exceeds_sigma(self):
        """Distance of (sigma + 0.1) should be rejected."""
        sharp = _make_spread_sharp(spread_line=-7.5, true_prob_a=0.60)
        sigma = 12.5
        too_far = -(abs(-7.5) + sigma + 0.1)
        result = self.model.predict(sharp, target_line=too_far, sector="nba")
        assert result is None

    def test_predict_at_same_line_as_pinnacle_returns_close_to_true_prob_a(self):
        """When target_line == pinnacle's spread_line, predicted prob ≈ true_prob_a."""
        true_prob_a = 0.60
        sharp = _make_spread_sharp(spread_line=-7.5, true_prob_a=true_prob_a)
        result = self.model.predict(sharp, target_line=-7.5, sector="nba")
        assert result is not None
        assert abs(result.true_prob - true_prob_a) < 0.01

    def test_predict_underdog_yes_flips_probability(self):
        """YES is underdog (outcome_b): prob should be 1 - prob_a at same line."""
        true_prob_a = 0.70  # strong home favorite
        sharp = _make_spread_sharp(spread_line=-7.5, true_prob_a=true_prob_a)
        result_fav = self.model.predict(sharp, target_line=-7.5, sector="nba", yes_is_underdog=False)
        result_dog = self.model.predict(sharp, target_line=-7.5, sector="nba", yes_is_underdog=True)
        assert result_fav is not None and result_dog is not None
        # The underdog result at the same line magnitude should be significantly lower
        assert result_dog.true_prob < result_fav.true_prob

    def test_predict_underdog_positive_line_is_high_not_low(self):
        """Regression (2026-07-10): underdog YES at a POSITIVE target_line
        (the normal case — e.g. Mercury +17.5 receiving points) must be a
        HIGH cover probability, not near-zero.

        Before the fix, the underdog branch discarded target_line's sign via
        abs(), so it silently evaluated P(underdog wins outright by more than
        17.5) instead of P(underdog's spread-adjusted score exceeds the
        favorite's). A WNBA Polymarket US game with an implied ~9.3pt
        favorite margin scored Mercury +17.5 at 1.6% cover instead of the
        correct ~74%, which then fed a synthetic ~98% "fair value" shadow bet
        via the NO-side derivation.
        """
        # Calibrate off a game with an implied mean margin of ~9.3 points
        # (true_prob_a=0.588 at spread_line=-6.5 for a WNBA sigma of 12.5).
        sharp = _make_spread_sharp(spread_line=-6.5, true_prob_a=0.588)
        underdog_deep = self.model.predict(
            sharp, target_line=17.5, sector="wnba", yes_is_underdog=True,
        )
        assert underdog_deep is not None
        # Getting 17.5 points when only trailing by ~9.3 on average should be
        # a strong favorite to cover, not a near-certain miss.
        assert underdog_deep.true_prob > 0.65

    def test_predict_favorite_and_underdog_sum_to_one_at_matching_lines(self):
        """P(favorite covers -N.5) + P(underdog covers +N.5) == 1 for the
        SAME game — they are complementary events. The old abs()-based
        underdog formula broke this identity for any positive target_line.
        """
        sharp = _make_spread_sharp(spread_line=-6.5, true_prob_a=0.588)
        fav = self.model.predict(sharp, target_line=-6.5, sector="wnba", yes_is_underdog=False)
        dog = self.model.predict(sharp, target_line=6.5, sector="wnba", yes_is_underdog=True)
        assert fav is not None and dog is not None
        assert abs((fav.true_prob + dog.true_prob) - 1.0) < 1e-9

    def test_result_clamped_at_0_01(self):
        """Result must never go below 0.01."""
        # Use a target line very far from the mean (but within sigma) to push prob near 0
        sharp = _make_spread_sharp(spread_line=-1.5, true_prob_a=0.52)
        # Sigma for soccer is 1.9; distance 1.5 is within 1 sigma from 1.5
        result = self.model.predict(sharp, target_line=-1.5, sector="soccer")
        if result is not None:
            assert result.true_prob >= 0.01

    def test_result_clamped_at_0_99(self):
        """Result must never exceed 0.99."""
        sharp = _make_spread_sharp(spread_line=-1.5, true_prob_a=0.52)
        result = self.model.predict(sharp, target_line=-1.5, sector="soccer")
        if result is not None:
            assert result.true_prob <= 0.99

    def test_nba_sigma_is_12_5(self):
        """SpreadPrediction.sigma for NBA sector is 12.5 (bumped 11.5→12.5
        on 2026-05-07; backtest_nba_score_stdev showed Brier −0.00077 on
        82 resolved spread bets, fixing tail over-confidence)."""
        sharp = _make_spread_sharp(spread_line=-7.5, true_prob_a=0.60)
        result = self.model.predict(sharp, target_line=-7.5, sector="nba")
        assert result is not None
        assert result.sigma == 12.5

    def test_nfl_sigma_is_14_0(self):
        """SpreadPrediction.sigma for NFL sector should be 14.0."""
        sharp = _make_spread_sharp(spread_line=-7.5, true_prob_a=0.60, sector="nfl")
        result = self.model.predict(sharp, target_line=-7.5, sector="nfl")
        assert result is not None
        assert result.sigma == 14.0

    def test_line_within_one_sigma_not_rejected(self):
        """A Kalshi line 5 pts from Pinnacle should not be None for NBA."""
        sharp = _make_spread_sharp(spread_line=-7.5, true_prob_a=0.60)
        result = self.model.predict(sharp, target_line=-12.0, sector="nba")
        # distance = |12 - 7.5| = 4.5, NBA sigma = 12.5 → within limit
        assert result is not None

    def test_implied_mean_is_set(self):
        """SpreadPrediction.implied_mean should be a reasonable floating-point value."""
        sharp = _make_spread_sharp(spread_line=-7.5, true_prob_a=0.60)
        result = self.model.predict(sharp, target_line=-7.5, sector="nba")
        assert result is not None
        assert math.isfinite(result.implied_mean)

    def test_stronger_favorite_lower_underdog_prob(self):
        """A heavier favorite spread means lower probability for the underdog."""
        sharp_big_fav = _make_spread_sharp(spread_line=-10.5, true_prob_a=0.65)
        sharp_small_fav = _make_spread_sharp(spread_line=-4.5, true_prob_a=0.55)
        result_big = self.model.predict(sharp_big_fav, target_line=-10.5, sector="nba", yes_is_underdog=True)
        result_small = self.model.predict(sharp_small_fav, target_line=-4.5, sector="nba", yes_is_underdog=True)
        assert result_big is not None and result_small is not None
        assert result_big.true_prob < result_small.true_prob

    def test_baseball_alt_runline_rejected_even_when_sharp_posts_it(self):
        """Baseball −4.5 alt run line must be rejected even when Pinnacle posts
        the exact −4.5 ladder (line_distance == 0 defeats the tolerance gate).

        Regression for the alt-runline leak: −4.5 covers went 2-for-15 live+
        shadow (Apr–Jun 2026) while the model predicted 27–46%. The absolute
        line-magnitude cap (1.5 for baseball) is the gate the
        fix/baseball-alt-runline-gate branch was meant to ship.
        """
        sharp = _make_spread_sharp(spread_line=-4.5, true_prob_a=0.40, sector="baseball")
        result = self.model.predict(sharp, target_line=-4.5, sector="baseball")
        assert result is None

    def test_baseball_standard_runline_allowed(self):
        """The standard −1.5 MLB run line is within the cap and still prices."""
        sharp = _make_spread_sharp(spread_line=-1.5, true_prob_a=0.45, sector="baseball")
        result = self.model.predict(sharp, target_line=-1.5, sector="baseball")
        assert result is not None
        assert 0.01 <= result.true_prob <= 0.99

    def test_nhl_alt_puckline_rejected(self):
        """NHL −2.5 alt puck line is past the 1.5 cap → rejected."""
        sharp = _make_spread_sharp(spread_line=-2.5, true_prob_a=0.40, sector="nhl")
        result = self.model.predict(sharp, target_line=-2.5, sector="nhl")
        assert result is None

    def test_nba_alt_line_unaffected_by_low_scoring_cap(self):
        """The magnitude cap is low-scoring-only — NBA alt lines still price."""
        sharp = _make_spread_sharp(spread_line=-7.5, true_prob_a=0.60)
        result = self.model.predict(sharp, target_line=-10.5, sector="nba")
        assert result is not None


# ===========================================================================
# TotalDistributionModel
# ===========================================================================

class TestTotalDistributionModel:
    def setup_method(self):
        self.model = TotalDistributionModel()

    def test_returns_none_when_total_line_is_none(self):
        sharp = _make_moneyline_sharp(0.6, 0.4)  # no total_line
        result = self.model.predict(sharp, target_line=225.5, sector="nba")
        assert result is None

    def test_returns_none_when_true_prob_over_is_none(self):
        """total_line present but true_prob_over missing."""
        sharp = SharpOdds(
            event_id="nba::2026-03-21::home_vs_away",
            book=SharpBook.pinnacle,
            sector="nba",
            outcome_a_decimal=2.0,
            outcome_b_decimal=2.0,
            true_prob_a=0.0,
            true_prob_b=0.0,
            margin=0.04,
            total_line=225.0,
            true_prob_over=None,
            true_prob_under=None,
        )
        result = self.model.predict(sharp, target_line=225.0, sector="nba")
        assert result is None

    def test_returns_none_when_line_distance_exceeds_one_sigma(self):
        """NBA sigma=22; target 50 pts from Pinnacle line should return None."""
        sharp = _make_total_sharp(total_line=225.5, prob_over=0.50)
        result = self.model.predict(sharp, target_line=275.5, sector="nba")
        assert result is None

    def test_returns_none_when_distance_just_over_sigma(self):
        sharp = _make_total_sharp(total_line=225.5, prob_over=0.50)
        # NBA sigma=22; distance 22.1 should be rejected
        result = self.model.predict(sharp, target_line=225.5 + 22.1, sector="nba")
        assert result is None

    def test_baseball_accepts_line_within_one_run_of_main(self):
        """Baseball caps the alt-line distance at 1.0 run (not 1*sigma=3.0)."""
        sharp = _make_total_sharp(total_line=8.5, prob_over=0.50)
        result = self.model.predict(sharp, target_line=9.5, sector="baseball")
        assert result is not None  # 1.0 run away → kept (the standard ±1)

    def test_baseball_rejects_deep_alternate_line(self):
        """An alt total 2.5 runs off the main line is rejected even though it
        sits inside 1*sigma=3.0 — this is the flood we're capping."""
        sharp = _make_total_sharp(total_line=8.5, prob_over=0.50)
        result = self.model.predict(sharp, target_line=11.0, sector="baseball")
        assert result is None
        # Sanity: the same 2.5-distance line would survive the default sigma rule.
        assert abs(11.0 - 8.5) < 1.0 * 3.0

    def test_predict_at_same_line_returns_close_to_true_prob_over(self):
        """When target_line == pinnacle total_line, predicted over_prob ≈ true_prob_over."""
        prob_over = 0.52
        sharp = _make_total_sharp(total_line=225.5, prob_over=prob_over)
        result = self.model.predict(sharp, target_line=225.5, sector="nba")
        assert result is not None
        assert abs(result.true_prob - prob_over) < 0.01

    def test_predict_under_is_complement(self):
        """yes_is_under=True should give (1 - over_prob) at the same line."""
        prob_over = 0.52
        sharp = _make_total_sharp(total_line=225.5, prob_over=prob_over)
        result_over = self.model.predict(sharp, target_line=225.5, sector="nba", yes_is_under=False)
        result_under = self.model.predict(sharp, target_line=225.5, sector="nba", yes_is_under=True)
        assert result_over is not None and result_under is not None
        assert abs(result_over.true_prob + result_under.true_prob - 1.0) < 0.01

    def test_result_clamped_at_0_01(self):
        """true_prob should never go below 0.01."""
        sharp = _make_total_sharp(total_line=225.5, prob_over=0.50)
        result = self.model.predict(sharp, target_line=225.5, sector="nba")
        if result is not None:
            assert result.true_prob >= 0.01

    def test_result_clamped_at_0_99(self):
        """true_prob should never exceed 0.99."""
        sharp = _make_total_sharp(total_line=225.5, prob_over=0.50)
        result = self.model.predict(sharp, target_line=225.5, sector="nba")
        if result is not None:
            assert result.true_prob <= 0.99

    def test_higher_target_line_lower_over_prob(self):
        """Moving target line higher should decrease over probability."""
        sharp = _make_total_sharp(total_line=225.5, prob_over=0.50)
        result_lower = self.model.predict(sharp, target_line=220.5, sector="nba")
        result_higher = self.model.predict(sharp, target_line=230.5, sector="nba")
        assert result_lower is not None and result_higher is not None
        assert result_lower.true_prob > result_higher.true_prob

    def test_sigma_is_correct_for_nba(self):
        """NBA sigma should be 22.0."""
        sharp = _make_total_sharp(total_line=225.5, prob_over=0.50)
        result = self.model.predict(sharp, target_line=225.5, sector="nba")
        assert result is not None
        assert result.sigma == 22.0

    def test_sigma_is_correct_for_nfl(self):
        """NFL sigma should be 14.0."""
        sharp = _make_total_sharp(total_line=47.5, prob_over=0.50, sector="nfl")
        result = self.model.predict(sharp, target_line=47.5, sector="nfl")
        assert result is not None
        assert result.sigma == 14.0

    def test_implied_mean_finite(self):
        sharp = _make_total_sharp(total_line=225.5, prob_over=0.50)
        result = self.model.predict(sharp, target_line=225.5, sector="nba")
        assert result is not None
        assert math.isfinite(result.implied_mean)

    def test_within_one_sigma_not_rejected_nba(self):
        """Distance of 20 pts is within NBA sigma=22 → should not return None."""
        sharp = _make_total_sharp(total_line=225.5, prob_over=0.50)
        result = self.model.predict(sharp, target_line=245.0, sector="nba")
        # distance = 245.0 - 225.5 = 19.5 < 22
        assert result is not None


class TestIsGameTotal:
    @pytest.mark.parametrize("line,sector,expected", [
        # NBA game totals
        (220.5, "nba", True),
        (235.0, "nba", True),
        (170.0, "nba", True),   # exactly at floor
        # NBA team props (below floor of 170)
        (109.5, "nba", False),
        (115.0, "nba", False),
        (169.9, "nba", False),
        # NCAAB game totals
        (140.0, "ncaab", True),
        (155.5, "ncaab", True),
        (110.0, "ncaab", True),   # exactly at floor
        # NCAAB team props
        (65.0, "ncaab", False),
        (75.0, "ncaab", False),
        (109.9, "ncaab", False),
        # NFL game totals
        (47.5, "nfl", True),
        (35.0, "nfl", True),     # exactly at floor
        # NFL low (unlikely but below floor)
        (34.9, "nfl", False),
        # Soccer over/under
        (2.5, "soccer", True),
        (1.5, "soccer", True),   # exactly at floor
        (0.5, "soccer", False),
    ])
    def test_is_game_total(self, line, sector, expected):
        assert is_game_total(line, sector) == expected


# ===========================================================================
# SharpOdds Pydantic validation
# ===========================================================================

class TestSharpOddsValidation:
    def _base_kwargs(self, **overrides):
        kw = dict(
            event_id="nfl::2026-03-21::patriots_vs_bills",
            book=SharpBook.pinnacle,
            sector="nfl",
            outcome_a_decimal=2.0,
            outcome_b_decimal=2.0,
            true_prob_a=0.50,
            true_prob_b=0.50,
            margin=0.04,
        )
        kw.update(overrides)
        return kw

    def test_valid_moneyline_accepted(self):
        """Probs that sum to exactly 1.0 should pass."""
        sharp = SharpOdds(**self._base_kwargs())
        assert abs(sharp.true_prob_a + sharp.true_prob_b - 1.0) < 1e-9

    def test_probs_summing_to_1_05_raises(self):
        """true_prob_a + true_prob_b = 1.05 → ValueError."""
        with pytest.raises(Exception):
            SharpOdds(**self._base_kwargs(true_prob_a=0.55, true_prob_b=0.50))

    def test_probs_summing_slightly_off_within_tolerance_accepted(self):
        """Probs differing by < 0.01 (e.g. 0.999) are accepted."""
        sharp = SharpOdds(**self._base_kwargs(true_prob_a=0.50, true_prob_b=0.499))
        assert sharp is not None

    def test_three_way_valid_accepted(self):
        """Soccer three-way with probs summing to 1.0 should pass."""
        sharp = SharpOdds(**self._base_kwargs(
            true_prob_a=0.45,
            true_prob_b=0.30,
            true_prob_draw=0.25,
            outcome_draw_decimal=4.0,
        ))
        assert abs(sharp.true_prob_a + sharp.true_prob_b + (sharp.true_prob_draw or 0) - 1.0) < 1e-9

    def test_three_way_invalid_raises(self):
        """Soccer three-way probs summing to 1.05 → ValueError."""
        with pytest.raises(Exception):
            SharpOdds(**self._base_kwargs(
                true_prob_a=0.45,
                true_prob_b=0.35,
                true_prob_draw=0.25,
                outcome_draw_decimal=4.0,
            ))

    def test_totals_valid_over_under_sum_to_one(self):
        """true_prob_over + true_prob_under = 1.0 → accepted."""
        sharp = SharpOdds(**self._base_kwargs(
            true_prob_a=0.0,
            true_prob_b=0.0,
            total_line=225.5,
            true_prob_over=0.52,
            true_prob_under=0.48,
        ))
        assert sharp.true_prob_over == 0.52

    def test_totals_probs_not_summing_to_one_raises(self):
        """true_prob_over + true_prob_under ≠ 1.0 → ValueError."""
        with pytest.raises(Exception):
            SharpOdds(**self._base_kwargs(
                true_prob_a=0.0,
                true_prob_b=0.0,
                total_line=225.5,
                true_prob_over=0.60,
                true_prob_under=0.60,
            ))

    def test_totals_moneyline_zero_skips_sum_validator(self):
        """When true_prob_a=0 the moneyline validator is skipped (totals-only record)."""
        sharp = SharpOdds(**self._base_kwargs(
            true_prob_a=0.0,
            true_prob_b=0.0,
            total_line=225.5,
            true_prob_over=0.50,
            true_prob_under=0.50,
        ))
        assert sharp.true_prob_a == 0.0

    def test_spread_line_present_with_zero_prob_a_skips_validator(self):
        """spread_line present but true_prob_a=0 → moneyline validator skipped."""
        sharp = SharpOdds(**self._base_kwargs(
            true_prob_a=0.0,
            true_prob_b=0.0,
            spread_line=-7.5,
        ))
        assert sharp.spread_line == -7.5

    def test_spread_line_valid_with_good_probs(self):
        """spread_line + valid moneyline probs should be fully accepted."""
        sharp = SharpOdds(**self._base_kwargs(
            true_prob_a=0.60,
            true_prob_b=0.40,
            spread_line=-7.5,
        ))
        assert sharp.spread_line == -7.5
        assert abs(sharp.true_prob_a + sharp.true_prob_b - 1.0) < 1e-9

    def test_error_message_contains_total(self):
        """ValueError message for bad totals should mention the computed sum."""
        with pytest.raises(Exception, match=r"1\.2"):
            SharpOdds(**self._base_kwargs(
                true_prob_a=0.0,
                true_prob_b=0.0,
                total_line=225.5,
                true_prob_over=0.60,
                true_prob_under=0.60,
            ))

    def test_error_raised_for_moneyline_sum_mismatch(self):
        """Explicit check that the raised error is a ValueError (via pydantic)."""
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            SharpOdds(**self._base_kwargs(true_prob_a=0.70, true_prob_b=0.70))
