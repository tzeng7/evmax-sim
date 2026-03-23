"""Pass 1 — Math/Logic robustness tests.

Covers:
  - devig.py: american_to_decimal(0) ZeroDivisionError guard,
               decimal_to_american(1.0) guard, fallback logging
  - kelly.py: NaN/Inf input guard
  - calculator.py: true_prob out-of-range guard
"""

import math
import pytest

from evmax.ev.devig import (
    american_to_decimal,
    decimal_to_american,
    devig_power_method,
    devig_two_way,
    devig_three_way,
)
from evmax.ev.kelly import compute_kelly, kelly_full
from evmax.ev.calculator import calculate_ev


# ---------------------------------------------------------------------------
# devig.py
# ---------------------------------------------------------------------------

class TestAmericanToDecimal:
    def test_positive_odds(self):
        assert american_to_decimal(100) == pytest.approx(2.0)
        assert american_to_decimal(200) == pytest.approx(3.0)
        assert american_to_decimal(150) == pytest.approx(2.5)

    def test_negative_odds(self):
        assert american_to_decimal(-110) == pytest.approx(1.909, rel=1e-3)
        assert american_to_decimal(-200) == pytest.approx(1.5)

    def test_zero_raises(self):
        """american_to_decimal(0) must raise ValueError, not ZeroDivisionError."""
        with pytest.raises(ValueError, match="cannot be 0"):
            american_to_decimal(0)

    def test_large_positive(self):
        # +10000 = 101x payout
        assert american_to_decimal(10000) == pytest.approx(101.0)

    def test_result_always_above_one(self):
        for odds in [-500, -200, -110, 100, 200, 500]:
            assert american_to_decimal(odds) > 1.0


class TestDecimalToAmerican:
    def test_standard_values(self):
        assert decimal_to_american(2.0) == 100
        assert decimal_to_american(3.0) == 200
        assert decimal_to_american(1.5) == -200

    def test_at_boundary_raises(self):
        """decimal_to_american(1.0) must raise, not divide by zero."""
        with pytest.raises(ValueError, match="must be > 1.0"):
            decimal_to_american(1.0)

    def test_below_one_raises(self):
        with pytest.raises(ValueError):
            decimal_to_american(0.5)

    def test_roundtrip(self):
        """american → decimal → american should be stable within 1."""
        for american in [-200, -150, -110, 100, 150, 200]:
            dec = american_to_decimal(american)
            back = decimal_to_american(dec)
            assert abs(back - american) <= 1, f"roundtrip failed for {american}"


class TestDevigPowerMethod:
    def test_two_way_probs_sum_to_one(self):
        result = devig_power_method([1.91, 2.05])
        assert sum(result.true_probs) == pytest.approx(1.0, abs=1e-6)

    def test_three_way_probs_sum_to_one(self):
        result = devig_power_method([2.4, 3.5, 3.2])
        assert sum(result.true_probs) == pytest.approx(1.0, abs=1e-6)

    def test_raises_on_too_few_outcomes(self):
        with pytest.raises(ValueError, match="at least 2"):
            devig_power_method([2.0])

    def test_raises_on_odds_below_one(self):
        with pytest.raises(ValueError, match="must be > 1.0"):
            devig_power_method([0.9, 2.0])

    def test_raises_on_odds_equal_one(self):
        with pytest.raises(ValueError):
            devig_power_method([1.0, 2.0])

    def test_margin_positive_with_vig(self):
        result = devig_power_method([1.91, 1.91])  # ~4.7% vig
        assert result.margin > 0

    def test_fair_odds_near_zero_margin(self):
        # 2.0 / 2.0 = exactly 0% margin
        result = devig_power_method([2.0, 2.0])
        assert abs(result.margin) < 1e-6

    def test_fallback_proportional_gives_valid_probs(self):
        # Extreme odds that might cause brentq to fail — still valid output
        result = devig_power_method([1.01, 100.0])
        assert abs(sum(result.true_probs) - 1.0) < 0.01


class TestDevigConvenienceWrappers:
    def test_devig_two_way(self):
        pa, pb, margin = devig_two_way(1.91, 2.05)
        assert 0 < pa < 1
        assert 0 < pb < 1
        assert abs(pa + pb - 1.0) < 1e-6

    def test_devig_three_way(self):
        pa, pb, pd, margin = devig_three_way(2.5, 3.0, 3.5)
        assert abs(pa + pb + pd - 1.0) < 1e-6
        assert margin >= 0


# ---------------------------------------------------------------------------
# kelly.py
# ---------------------------------------------------------------------------

class TestKellyFull:
    def test_positive_ev_bet(self):
        # 55% true prob, 2.0x payout — should be positive Kelly
        k = kelly_full(0.55, 2.0)
        assert k > 0

    def test_negative_ev_bet(self):
        # 40% true prob at 2.0x payout — negative EV
        k = kelly_full(0.40, 2.0)
        assert k < 0

    def test_breakeven_payout_returns_zero(self):
        # payout <= 1.0 → no profit possible
        assert kelly_full(0.9, 1.0) == 0.0
        assert kelly_full(0.9, 0.5) == 0.0


class TestComputeKelly:
    def test_standard_bet(self):
        result = compute_kelly(
            true_prob=0.55,
            payout_decimal=2.0,
            edge_pct=0.10,
            max_kelly=0.05,
        )
        assert result.kelly_full > 0
        assert result.kelly_fraction <= 0.05
        assert result.kelly_fraction > 0

    def test_nan_true_prob_returns_zero(self):
        result = compute_kelly(
            true_prob=float("nan"),
            payout_decimal=2.0,
            edge_pct=0.10,
        )
        assert result.kelly_fraction == 0.0
        assert result.suggested_units == 0.0

    def test_inf_payout_returns_zero(self):
        result = compute_kelly(
            true_prob=0.55,
            payout_decimal=float("inf"),
            edge_pct=0.10,
        )
        assert result.kelly_fraction == 0.0

    def test_nan_payout_returns_zero(self):
        result = compute_kelly(
            true_prob=0.55,
            payout_decimal=float("nan"),
            edge_pct=0.10,
        )
        assert result.kelly_fraction == 0.0

    def test_max_kelly_cap_respected(self):
        result = compute_kelly(
            true_prob=0.99,
            payout_decimal=10.0,
            edge_pct=0.90,
            max_kelly=0.05,
        )
        assert result.kelly_fraction <= 0.05

    def test_breakeven_payout_returns_zero_units(self):
        result = compute_kelly(
            true_prob=0.9,
            payout_decimal=1.0,
            edge_pct=0.0,
        )
        assert result.suggested_units == 0.0

    def test_liquidity_discount_applied(self):
        wide = compute_kelly(0.55, 2.0, 0.10, spread_pct=0.30)
        narrow = compute_kelly(0.55, 2.0, 0.10, spread_pct=0.01)
        # Wide spread → more discount → smaller fraction
        assert wide.kelly_fraction <= narrow.kelly_fraction

    def test_min_kelly_floor(self):
        # Very small edge — without floor would be tiny
        result = compute_kelly(
            true_prob=0.51,
            payout_decimal=2.0,
            edge_pct=0.02,
            min_kelly=0.005,
            max_kelly=0.05,
        )
        assert result.kelly_fraction >= 0.005


# ---------------------------------------------------------------------------
# calculator.py
# ---------------------------------------------------------------------------

class TestCalculateEV:
    def test_positive_ev(self):
        # true_prob=0.60, market_price=0.50 → EV = 0.60*2.0 - 1 = 0.20
        ev, edge = calculate_ev(0.50, 0.60)
        assert ev == pytest.approx(0.20)
        assert edge == pytest.approx(0.20)

    def test_negative_ev(self):
        ev, edge = calculate_ev(0.60, 0.40)
        assert ev < 0

    def test_market_price_zero_returns_zero(self):
        ev, edge = calculate_ev(0.0, 0.55)
        assert ev == 0.0
        assert edge == 0.0

    def test_market_price_one_returns_zero(self):
        ev, edge = calculate_ev(1.0, 0.55)
        assert ev == 0.0
        assert edge == 0.0

    def test_market_price_above_one_returns_zero(self):
        ev, edge = calculate_ev(1.5, 0.55)
        assert ev == 0.0

    def test_true_prob_zero_returns_zero(self):
        """Zero true prob should not produce a positive EV."""
        ev, edge = calculate_ev(0.50, 0.0)
        assert ev == 0.0

    def test_true_prob_negative_returns_zero(self):
        ev, edge = calculate_ev(0.50, -0.10)
        assert ev == 0.0

    def test_true_prob_above_one_returns_zero(self):
        ev, edge = calculate_ev(0.50, 1.5)
        assert ev == 0.0

    def test_breakeven(self):
        # market_price == true_prob → EV = 0
        ev, edge = calculate_ev(0.55, 0.55)
        assert ev == pytest.approx(0.0, abs=1e-6)

    def test_boundary_prices_excluded(self):
        # Just outside valid range
        assert calculate_ev(-0.01, 0.5) == (0.0, 0.0)
        assert calculate_ev(1.01, 0.5) == (0.0, 0.0)
