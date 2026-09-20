"""Shared EV helpers extracted from copy-pasted CLI/notification logic:
passes_play_floor (calculator) and american_odds (odds_format)."""

from __future__ import annotations

import pytest

from evmax.ev.calculator import passes_play_floor
from evmax.ev.odds_format import american_odds, cents


class TestPassesPlayFloor:
    def test_both_floors_required(self):
        assert passes_play_floor(0.02, 0.15, min_ev=0.02, min_prob=0.15) is True
        assert passes_play_floor(0.019, 0.50, min_ev=0.02, min_prob=0.15) is False
        assert passes_play_floor(0.30, 0.149, min_ev=0.02, min_prob=0.15) is False

    def test_no_longshot_ramp(self):
        # The former tiered ramp raised the EV floor below min_prob; since a row
        # below min_prob is excluded outright, the ramp never fired — the flat
        # floor is the exact effective rule and the only one now.
        assert passes_play_floor(0.03, 0.16, min_ev=0.02, min_prob=0.15) is True
        assert passes_play_floor(0.99, 0.10, min_ev=0.02, min_prob=0.15) is False

    def test_removed_helper_is_gone(self):
        import evmax.ev.calculator as calc

        assert not hasattr(calc, "tiered_min_ev")


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
