"""Tests for evmax.fees — Kalshi and Polymarket US fee models."""

import math

import pytest

from evmax.fees import (
    bet_pnl,
    kalshi_fee_prob,
    kalshi_order_fee,
    polymarket_us_fee_prob,
    polymarket_us_order_fee,
    venue_fee_prob,
    venue_order_fee,
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


class TestVenueOrderFee:
    def test_dispatch_matches_per_venue_helpers(self):
        assert venue_order_fee("kalshi", 0.5, 200) == pytest.approx(kalshi_order_fee(0.5, 200))
        assert venue_order_fee("polymarket_us", 0.5, 200) == pytest.approx(
            polymarket_us_order_fee(0.5, 200)
        )

    def test_maker_flag_forwarded(self):
        # PolyUS maker path is a rebate (negative) — the flag must reach it.
        assert venue_order_fee("polymarket_us", 0.5, 200, maker=True) < 0

    def test_none_venue_is_zero(self):
        # Unlike venue_fee_prob, the P&L path degrades to gross instead of raising.
        assert venue_order_fee(None, 0.5, 200) == 0.0

    def test_unknown_venue_degrades_to_zero(self):
        assert venue_order_fee("pinnacle", 0.5, 200) == 0.0


class TestBetPnl:
    def test_gross_when_no_venue(self):
        # venue=None reproduces the pre-fee formulas exactly.
        assert bet_pnl(100.0, 0.5, True) == pytest.approx(100.0)
        assert bet_pnl(100.0, 0.5, False) == pytest.approx(-100.0)

    def test_kalshi_win_nets_the_order_fee(self):
        # 100 notional at 50c = 200 contracts; fee = ceil(0.07*200*0.25) = $3.50.
        assert bet_pnl(100.0, 0.5, True, venue="kalshi") == pytest.approx(96.50)

    def test_kalshi_loss_adds_the_order_fee(self):
        assert bet_pnl(100.0, 0.5, False, venue="kalshi") == pytest.approx(-103.50)

    def test_fee_hits_win_and_loss_equally(self):
        # The core accounting invariant: the flat entry fee reduces the win P&L
        # and the loss P&L by the SAME amount. This is why the fee must NOT be
        # folded into the stake (that would scale it by 1/price on a win).
        stake, price = 100.0, 0.5
        fee = kalshi_order_fee(price, stake / price)
        win_drop = bet_pnl(stake, price, True) - bet_pnl(stake, price, True, venue="kalshi")
        loss_drop = bet_pnl(stake, price, False) - bet_pnl(stake, price, False, venue="kalshi")
        assert win_drop == pytest.approx(fee)
        assert loss_drop == pytest.approx(fee)

    def test_polymarket_maker_rebate_increases_pnl(self):
        # Maker rebate (negative fee) lifts the win above the gross figure.
        assert bet_pnl(100.0, 0.5, True, venue="polymarket_us", maker=True) > 100.0

    def test_degenerate_price_is_zero(self):
        assert bet_pnl(100.0, 0.0, True, venue="kalshi") == 0.0
        assert bet_pnl(100.0, 1.0, False, venue="kalshi") == 0.0
