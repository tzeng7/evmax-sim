"""Tests for scripts/seed_ncaaf_elo.py pure helpers (NCAAF warm-seed).

Covers the deterministic, network-free logic:
- regress_toward_mean — the FiveThirtyEight-style carry-forward regression.
- _elo_confidence — must reproduce EloModelAgent's confidence tiers exactly,
  since the whole point of the warm seed is that a team clears the 0.45 gate
  (>=5 lifetime games → confidence > 0.45) from week 0.

No network — the fetch-driven build_seed/validation paths are not exercised here.
"""

from __future__ import annotations

import pytest

from scripts.seed_ncaaf_elo import _elo_confidence, regress_toward_mean


class TestRegressTowardMean:
    def test_half_regression_pulls_halfway_to_1500(self):
        r = {"a": 1700.0, "b": 1300.0}
        regress_toward_mean(r, 0.5)
        assert r["a"] == pytest.approx(1600.0)  # 1500 + (1700-1500)*0.5
        assert r["b"] == pytest.approx(1400.0)

    def test_factor_one_is_noop(self):
        r = {"a": 1831.0, "b": 1211.0}
        regress_toward_mean(r, 1.0)
        assert r["a"] == pytest.approx(1831.0)
        assert r["b"] == pytest.approx(1211.0)

    def test_factor_zero_wipes_to_mean(self):
        r = {"a": 1831.0, "b": 1211.0}
        regress_toward_mean(r, 0.0)
        assert r["a"] == pytest.approx(1500.0)
        assert r["b"] == pytest.approx(1500.0)

    def test_team_at_mean_is_fixed_point(self):
        r = {"a": 1500.0}
        regress_toward_mean(r, 0.37)
        assert r["a"] == pytest.approx(1500.0)

    def test_custom_mean(self):
        r = {"a": 1200.0}
        regress_toward_mean(r, 0.5, mean=1000.0)
        assert r["a"] == pytest.approx(1100.0)


class TestEloConfidenceGate:
    """The seed exists so warm counts clear the ensemble's 0.45 gate at week 0.

    EloModelAgent.predict_pair tiers: 0 games -> 0.30, 1-4 -> 0.45, 5-14 -> 0.60,
    >=15 -> 0.80. The ensemble fires a model only when confidence > 0.45, so a
    team needs >=5 lifetime games to contribute — the exact darkness a warm seed
    removes.
    """

    def test_zero_games_is_below_gate(self):
        assert _elo_confidence(0) == pytest.approx(0.30)
        assert _elo_confidence(0) <= 0.45  # dark

    def test_one_to_four_games_at_gate_still_dark(self):
        for n in (1, 4):
            assert _elo_confidence(n) == pytest.approx(0.45)
            assert _elo_confidence(n) <= 0.45  # ensemble uses strict > 0.45

    def test_five_games_clears_gate(self):
        assert _elo_confidence(5) == pytest.approx(0.60)
        assert _elo_confidence(5) > 0.45  # fires

    def test_warm_seed_counts_fire_high(self):
        # A marquee program carries ~60-70 lifetime games after a 5-season seed.
        assert _elo_confidence(70) == pytest.approx(0.80)
        assert _elo_confidence(15) == pytest.approx(0.80)
        assert _elo_confidence(14) == pytest.approx(0.60)
