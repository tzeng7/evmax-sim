"""Tests for the pure injury-mechanism helpers in
scripts/backtest_wnba_injury_impact.py. These mirror the live injury agent's
logic (OUT x tier x staleness, per-team cap, additive win-prob adjustment), so a
bug here would silently invalidate the MODEL-13 validation verdict.
"""
from __future__ import annotations

from collections import deque
from datetime import date

import pytest

from scripts.backtest_wnba_injury_impact import (
    _apply_variant,
    _parse_minutes,
    _team_scratch_impact,
    _tier,
)


class TestParseMinutes:
    def test_numeric(self):
        assert _parse_minutes("21") == 21.0
        assert _parse_minutes("0") == 0.0

    def test_dnp_markers(self):
        assert _parse_minutes("--") == 0.0
        assert _parse_minutes("") == 0.0
        assert _parse_minutes("DNP") == 0.0
        assert _parse_minutes(None) == 0.0


class TestTier:
    def test_thresholds(self):
        assert _tier(32.0) == "star"
        assert _tier(30.0) == "star"
        assert _tier(24.0) == "starter"
        assert _tier(22.0) == "starter"
        assert _tier(15.0) == "rotation"
        assert _tier(0.0) == "rotation"


class TestApplyVariant:
    def test_no_injury_is_identity(self):
        assert _apply_variant(0.60, 0.0, 0.0, 1.0) == pytest.approx(0.60)

    def test_home_scratch_lowers_home_prob(self):
        p = _apply_variant(0.60, raw_home=0.07, raw_away=0.0, magnitude=1.0)
        assert p < 0.60

    def test_away_scratch_raises_home_prob(self):
        p = _apply_variant(0.60, raw_home=0.0, raw_away=0.07, magnitude=1.0)
        assert p > 0.60

    def test_magnitude_scales_effect(self):
        p1 = _apply_variant(0.60, 0.05, 0.0, 1.0)
        p2 = _apply_variant(0.60, 0.05, 0.0, 2.0)
        assert p2 < p1  # more magnitude => bigger downward shift

    def test_per_team_cap(self):
        # raw*mag far exceeds the 0.10 cap; the shift is bounded.
        capped = _apply_variant(0.60, raw_home=0.50, raw_away=0.0, magnitude=2.0)
        # home adj is floored at -0.10; new_home ~ 0.60-0.10 renormalized
        assert capped >= 0.49  # not driven to zero


class TestTeamScratchImpact:
    _GAME_DATE = date(2025, 6, 3)

    def _hist(self, mins_list):
        return deque(mins_list, maxlen=5)

    def test_established_star_scratch(self):
        history = {"Star": self._hist([32, 34, 30, 33, 31])}
        last_played = {"Star": date(2025, 6, 1)}   # 2 days ago -> fresh
        minutes_now = {"Star": 0.0}                 # scratched
        impact, notes = _team_scratch_impact(
            "aces", minutes_now, history, last_played, self._GAME_DATE
        )
        # 0.045 OUT * 1.5 star * 1.0 staleness
        assert impact == pytest.approx(0.045 * 1.5)
        assert any("Star" in n for n in notes)

    def test_player_who_played_is_not_a_scratch(self):
        history = {"Star": self._hist([32, 34, 30])}
        last_played = {"Star": date(2025, 6, 1)}
        minutes_now = {"Star": 28.0}                # played
        impact, notes = _team_scratch_impact(
            "aces", minutes_now, history, last_played, self._GAME_DATE
        )
        assert impact == 0.0
        assert notes == []

    def test_thin_sample_ignored(self):
        # Only 2 prior appearances (< _MIN_APPEARANCES=3) -> not established.
        history = {"Sub": self._hist([18, 16])}
        last_played = {"Sub": date(2025, 6, 1)}
        minutes_now = {"Sub": 0.0}
        impact, _ = _team_scratch_impact(
            "aces", minutes_now, history, last_played, self._GAME_DATE
        )
        assert impact == 0.0

    def test_low_role_ignored(self):
        # Established but a deep-bench role (< _ROTATION_MIN=14) -> ignored.
        history = {"Deep": self._hist([8, 6, 10, 7])}
        last_played = {"Deep": date(2025, 6, 1)}
        minutes_now = {"Deep": 0.0}
        impact, _ = _team_scratch_impact(
            "aces", minutes_now, history, last_played, self._GAME_DATE
        )
        assert impact == 0.0

    def test_stale_absence_decays_to_zero(self):
        # Last played 33 days ago -> beyond INJURY_STALE_DAYS -> staleness 0.
        history = {"Star": self._hist([32, 34, 30, 33, 31])}
        last_played = {"Star": date(2025, 5, 1)}
        minutes_now = {"Star": 0.0}
        impact, _ = _team_scratch_impact(
            "aces", minutes_now, history, last_played, self._GAME_DATE
        )
        assert impact == 0.0
