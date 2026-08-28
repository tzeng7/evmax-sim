"""Unit tests for the pure helpers in evmax/backtest/sources/espn_walkforward.py.

This 1,599-line module produces the walk-forward Brier numbers that gate every
shadow->live promotion, yet had no direct coverage. These lock down the pure
parsing/aggregation helpers — a bug in _parse_ip (the '6.2 = 6 2/3 innings'
convention) or _ensemble would silently corrupt every pitcher/blend backtest.
"""
from __future__ import annotations

import math

import pytest

import evmax.backtest.sources.espn_walkforward as wf


class TestParseIp:
    def test_outs_are_thirds(self):
        assert wf._parse_ip("6.2") == pytest.approx(6 + 2 / 3)
        assert wf._parse_ip("5.1") == pytest.approx(5 + 1 / 3)
        assert wf._parse_ip("5.0") == 5.0

    def test_plain_integer(self):
        assert wf._parse_ip("7") == 7.0

    def test_bad_input_is_zero(self):
        assert wf._parse_ip("--") == 0.0
        assert wf._parse_ip(None) == 0.0


class TestParsePc:
    def test_ok(self):
        assert wf._parse_pc("103") == 103

    def test_bad_is_zero(self):
        assert wf._parse_pc("--") == 0
        assert wf._parse_pc("") == 0
        assert wf._parse_pc(None) == 0


class TestParseVariant:
    def test_v1_is_empty(self):
        assert wf._parse_pitcher_variant("v1") == set()

    def test_v2_is_all(self):
        assert wf._parse_pitcher_variant("v2") == {"pen", "park", "off", "xera"}

    def test_custom_component_set(self):
        assert wf._parse_pitcher_variant("pen, park") == {"pen", "park"}

    def test_default_and_none(self):
        assert wf._parse_pitcher_variant("") == set()
        assert wf._parse_pitcher_variant(None) == set()


class TestRunningToRates:
    def test_fip_and_era(self):
        r = {"ip": 9.0, "er": 3, "hr": 1, "bb": 2, "so": 9}
        fip, era = wf._running_to_rates(r)
        assert era == pytest.approx(3.0)  # 3 ER * 9 / 9 IP
        assert fip == pytest.approx((13 * 1 + 3 * 2 - 2 * 9) / 9 + wf.BACKTEST_CFIP)

    def test_zero_ip_is_nan(self):
        fip, era = wf._running_to_rates({"ip": 0, "er": 0, "hr": 0, "bb": 0, "so": 0})
        assert math.isnan(fip) and math.isnan(era)


class TestAdaptiveHomeBonus:
    def test_below_sample_uses_default(self):
        assert wf._adaptive_home_bonus(100, 60) == wf.PITCHER_HOME_BONUS

    def test_large_sample_is_clamped(self):
        # wp=0.60 -> 0.10, clamped into [MIN, MAX]
        val = wf._adaptive_home_bonus(1000, 600)
        expected = max(wf.PITCHER_HOME_BONUS_MIN, min(wf.PITCHER_HOME_BONUS_MAX, 0.10))
        assert val == pytest.approx(expected)


class TestEnsemble:
    def test_weighted_mean(self):
        assert wf._ensemble([(0.6, 0.5), (0.8, 0.5)]) == pytest.approx(0.7)

    def test_skips_none_and_reweights(self):
        assert wf._ensemble([(None, 0.5), (0.8, 0.5)]) == pytest.approx(0.8)

    def test_all_none_returns_none(self):
        assert wf._ensemble([(None, 1.0), (None, 0.5)]) is None


class TestParsePitcherLine:
    def test_valid_row(self):
        labels = ["IP", "ER", "BB", "K", "HR", "PC"]
        stats = ["6.2", "2", "1", "7", "1", "95"]
        d = wf._parse_pitcher_line(labels, stats, "  Gerrit Cole  ")
        assert d["name"] == "Gerrit Cole"
        assert d["ip"] == pytest.approx(6 + 2 / 3)
        assert (d["er"], d["bb"], d["so"], d["hr"], d["pc"]) == (2, 1, 7, 1, 95)

    def test_length_mismatch_returns_none(self):
        assert wf._parse_pitcher_line(["IP", "ER"], ["6.2"], "X") is None


class TestAccumulatePitcherLine:
    def test_adds_into_running_totals(self):
        running = {"ip": 0.0, "er": 0, "bb": 0, "so": 0, "hr": 0, "games": 0}
        wf._accumulate_pitcher_line(running, {"ip": 6.0, "er": 2, "bb": 1, "so": 7, "hr": 1})
        wf._accumulate_pitcher_line(running, {"ip": 5.0, "er": 3, "bb": 2, "so": 4, "hr": 0})
        assert running == {"ip": 11.0, "er": 5, "bb": 3, "so": 11, "hr": 1, "games": 2}
