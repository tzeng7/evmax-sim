"""Tests for evmax.fees — Kalshi and Polymarket US fee models."""

import math

import pytest

from evmax.fees import (
    kalshi_fee_prob,
    kalshi_order_fee,
    polymarket_us_fee_prob,
    polymarket_us_order_fee,
    venue_fee_prob,
)


class TestKalshiFees:
    def test_taker_prob_at_midpoint(self):
        # 0.07 * 0.5 * 0.5 = 0.0175 — the documented per-contract max
        assert kalshi_fee_prob(0.5) == pytest.approx(0.0175)

    def test_taker_prob_at_longshot(self):
        assert kalshi_fee_prob(0.10) == pytest.approx(0.07 * 0.10 * 0.90)

    def test_maker_is_quarter_of_taker(self):
        assert kalshi_fee_prob(0.5, maker=True) == pytest.approx(0.0175 * 0.25)

    def test_order_fee_rounds_up_to_cent(self):
        # 1 contract at 50c: raw 0.0175 → ceil to 0.02
        assert kalshi_order_fee(0.5, 1) == pytest.approx(0.02)

    def test_order_fee_100_contracts_exact(self):
        # 100 contracts at 50c: raw 1.75 exactly — no rounding needed
        assert kalshi_order_fee(0.5, 100) == pytest.approx(1.75)

    def test_order_fee_never_rounds_down(self):
        raw = 0.07 * 7 * 0.33 * 0.67
        assert kalshi_order_fee(0.33, 7) >= raw

    def test_price_out_of_range_raises(self):
        with pytest.raises(ValueError):
            kalshi_fee_prob(0.0)
        with pytest.raises(ValueError):
            kalshi_fee_prob(1.0)


class TestPolymarketUSFees:
    def test_taker_prob_at_midpoint(self):
        # 0.06 * 0.5 * 0.5 = 0.015 → $1.50 per 100 contracts (documented max)
        assert polymarket_us_fee_prob(0.5) == pytest.approx(0.015)

    def test_maker_is_rebate(self):
        assert polymarket_us_fee_prob(0.5, maker=True) == pytest.approx(-0.0125 * 0.25)
        assert polymarket_us_fee_prob(0.5, maker=True) < 0

    def test_order_fee_100_contracts(self):
        assert polymarket_us_order_fee(0.5, 100) == pytest.approx(1.50)

    def test_bankers_rounding_half_to_even(self):
        # From the fee docs: $0.025 rounds to $0.02 (even), $0.035 to $0.04.
        # 0.025 raw: theta*C*p*(1-p) = 0.06*C*0.5*0.5 → C = 0.025/0.015
        contracts_a = 0.025 / 0.015
        assert polymarket_us_order_fee(0.5, contracts_a) == pytest.approx(0.02)
        contracts_b = 0.035 / 0.015
        assert polymarket_us_order_fee(0.5, contracts_b) == pytest.approx(0.04)

    def test_maker_order_fee_negative(self):
        assert polymarket_us_order_fee(0.5, 100, maker=True) == pytest.approx(-0.31, abs=0.005)


class TestVenueDispatch:
    def test_dispatch(self):
        assert venue_fee_prob("kalshi", 0.5) == pytest.approx(0.0175)
        assert venue_fee_prob("polymarket_us", 0.5) == pytest.approx(0.015)

    def test_unknown_venue_raises(self):
        with pytest.raises(ValueError):
            venue_fee_prob("pinnacle", 0.5)
