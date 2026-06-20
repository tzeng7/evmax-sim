"""Shared EV helpers extracted from copy-pasted CLI/notification logic:
tiered_min_ev (calculator) and american_odds (odds_format)."""

from __future__ import annotations

import pytest

from evmax.ev.calculator import tiered_min_ev
from evmax.ev.odds_format import american_odds, cents, cents_american


class TestTieredMinEv:
    def test_floor_no_scaling_at_or_above_min_prob(self):
        assert tiered_min_ev(0.15, min_ev=0.02, min_prob=0.15) == pytest.approx(0.02)
        assert tiered_min_ev(0.50, min_ev=0.02, min_prob=0.15) == pytest.approx(0.02)

    def test_scales_up_for_longshots(self):
        assert tiered_min_ev(0.08, min_ev=0.02, min_prob=0.15) == pytest.approx(0.055)
        assert tiered_min_ev(0.12, min_ev=0.02, min_prob=0.15) == pytest.approx(0.035)

    def test_respects_custom_floors(self):
        assert tiered_min_ev(0.0, min_ev=0.05, min_prob=0.20) == pytest.approx(0.15)


class TestAmericanOdds:
    def test_out_of_range_is_na(self):
        assert american_odds(0.0) == "N/A"
        assert american_odds(1.0) == "N/A"
        assert american_odds(-0.1) == "N/A"

    def test_favorite_is_negative(self):
        assert american_odds(0.6) == "-150"

    def test_underdog_has_plus(self):
        assert american_odds(0.4) == "+150"

    def test_even_money_boundary(self):
        # prob == 0.5 takes the favorite branch: -100
        assert american_odds(0.5) == "-100"


class TestCents:
    def test_out_of_range_is_na(self):
        assert cents(0.0) == "N/A"
        assert cents(1.0) == "N/A"
        assert cents(-0.1) == "N/A"
        assert cents(1.5) == "N/A"

    def test_whole_cent_has_no_decimal(self):
        # Plain binary markets quote whole cents — no trailing ".0".
        assert cents(0.12) == "12¢"
        assert cents(0.5) == "50¢"
        assert cents(0.99) == "99¢"

    def test_sub_cent_keeps_one_decimal(self):
        # Combo/spread asks price continuously — this is the +733-vs-+684.9
        # case: a 12¢ display that really fills at 12.7¢.
        assert cents(0.127) == "12.7¢"
        assert cents(0.1271) == "12.7¢"

    def test_rounds_to_one_decimal(self):
        assert cents(0.12349) == "12.3¢"
        assert cents(0.12350) == "12.3¢"  # banker's rounding on .5 → 12.3
        assert cents(0.12361) == "12.4¢"


class TestCentsAmerican:
    def test_out_of_range_is_na(self):
        assert cents_american(0.0) == "N/A"
        assert cents_american(1.0) == "N/A"

    def test_cents_primary_american_in_parens(self):
        # Underdog: cents first, American reference in parens.
        assert cents_american(0.127) == "12.7¢ (+687)"
        assert cents_american(0.4) == "40¢ (+150)"

    def test_favorite_renders_negative_american(self):
        assert cents_american(0.6) == "60¢ (-150)"
