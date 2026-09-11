"""Tests for evmax.fees — Kalshi and Polymarket US fee models."""

import math

import pytest

from evmax.fees import (
    NOVIG_TAKER_RATE,
    PROPHETX_COMMISSION_RATE,
    bet_pnl,
    kalshi_fee_prob,
    kalshi_order_fee,
    novig_fee_prob,
    novig_order_fee,
    polymarket_us_fee_prob,
    polymarket_us_order_fee,
    prophetx_fee_prob,
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


class TestNovigFees:
    def test_taker_prob_at_midpoint_is_the_cap(self):
        # 0.03 * 0.5 * 0.5 = 0.0075 — the published per-contract cap.
        assert novig_fee_prob(0.5) == pytest.approx(0.0075)

    def test_taker_prob_at_longshot(self):
        assert novig_fee_prob(0.10) == pytest.approx(0.03 * 0.10 * 0.90)

    def test_maker_is_half_taker_rebate(self):
        # Makers pay nothing and earn a credit up to HALF the taker fee.
        assert novig_fee_prob(0.5, maker=True) == pytest.approx(-0.5 * 0.0075)
        assert novig_fee_prob(0.5, maker=True) < 0

    def test_cheaper_than_kalshi_and_poly_at_every_price(self):
        for p in (0.1, 0.3, 0.5, 0.7, 0.9):
            assert novig_fee_prob(p) < polymarket_us_fee_prob(p) < kalshi_fee_prob(p)

    def test_order_fee_100_contracts(self):
        assert novig_order_fee(0.5, 100) == pytest.approx(0.75)

    def test_maker_order_fee_negative(self):
        assert novig_order_fee(0.5, 100, maker=True) < 0

    def test_price_out_of_range_raises(self):
        with pytest.raises(ValueError):
            novig_fee_prob(1.0)


class TestProphetXFees:
    def test_no_commission_means_no_shift(self):
        # rate=0 → breakeven prob unchanged from the ask.
        assert prophetx_fee_prob(0.5, rate=0.0) == pytest.approx(0.0)

    def test_shift_is_positive_and_small(self):
        # 2% commission on winnings ≈ 0.5pp of true prob at p=0.5 — much
        # smaller than any entry-fee venue there (Kalshi 1.75pp).
        shift = prophetx_fee_prob(0.5)
        assert shift == pytest.approx(1.0 / 1.98 - 0.5, abs=1e-9)
        assert 0.0 < shift < kalshi_fee_prob(0.5)

    def test_breakeven_prob_zeroes_expected_pnl(self):
        # THE validation: at the fee-implied breakeven true prob, the expected
        # net-of-commission P&L of a $1 stake is exactly zero. Confirms the
        # closed-form q = 1/(1+(1/p-1)(1-r)) is the correct breakeven.
        for p in (0.2, 0.4, 0.6, 0.85):
            q = p + prophetx_fee_prob(p)  # breakeven true prob
            ev = q * bet_pnl(1.0, p, True, venue="prophetx") + (1 - q) * bet_pnl(
                1.0, p, False, venue="prophetx"
            )
            assert ev == pytest.approx(0.0, abs=1e-9)

    def test_maker_and_taker_pay_the_same(self):
        # ProphetX charges a uniform commission — maker flag is a no-op.
        assert venue_fee_prob("prophetx", 0.5, maker=True) == pytest.approx(
            venue_fee_prob("prophetx", 0.5, maker=False)
        )

    def test_win_nets_commission_on_profit_only(self):
        # 100 at 0.5 → profit 100, minus 2% = 98 net; nothing on a loss.
        assert bet_pnl(100.0, 0.5, True, venue="prophetx") == pytest.approx(98.0)
        assert bet_pnl(100.0, 0.5, False, venue="prophetx") == pytest.approx(-100.0)

    def test_no_entry_fee_charged(self):
        # The commission is a settlement event, not an order fee.
        assert venue_order_fee("prophetx", 0.5, 200) == 0.0


class TestVenueDispatch:
    def test_dispatch(self):
        assert venue_fee_prob("kalshi", 0.5) == pytest.approx(0.0175)
        assert venue_fee_prob("polymarket_us", 0.5) == pytest.approx(0.015)
        assert venue_fee_prob("novig", 0.5) == pytest.approx(0.0075)
        assert venue_fee_prob("prophetx", 0.5) == pytest.approx(prophetx_fee_prob(0.5))

    def test_unknown_venue_raises(self):
        with pytest.raises(ValueError):
            venue_fee_prob("pinnacle", 0.5)

    def test_constants_match_sourced_rates(self):
        assert NOVIG_TAKER_RATE == pytest.approx(0.03)
        assert PROPHETX_COMMISSION_RATE == pytest.approx(0.02)


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


class TestEffectivePriceIntegration:
    """The EV gate prices every venue through effective_price = ask + fee_prob."""

    def test_novig_effective_price(self):
        from evmax.ev.calculator import effective_price

        assert effective_price(0.5, "novig") == pytest.approx(0.5 + novig_fee_prob(0.5))

    def test_prophetx_effective_price_is_the_breakeven_prob(self):
        # effective_price for a winnings-commission venue equals the breakeven
        # true prob q, so calculate_ev(q, q) = 0 — the fee is priced correctly
        # by the same generic gate the entry-fee venues use.
        from evmax.ev.calculator import calculate_ev, effective_price

        p = 0.4
        q = effective_price(p, "prophetx")
        assert q == pytest.approx(p + prophetx_fee_prob(p))
        ev, _ = calculate_ev(q, q)
        assert ev == pytest.approx(0.0, abs=1e-9)


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
