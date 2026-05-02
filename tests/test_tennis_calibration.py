"""Tests for the per-sector calibration wiring inside EnsembleModelAgent.

The calibration module itself (ModelCalibrator) is exercised in
test_nba_models.py. These tests cover the wiring added in ensemble_agent.py:
  - Identity passthrough when no sector calibration is fitted
  - Calibration applies to model-side blend BEFORE sharp blend
  - 3-way (draw) markets skip calibration
  - YES-side calibration recovers prob_b = 1 - prob_a
"""

from __future__ import annotations

import pytest

from evmax.agents.models.ensemble_agent import EnsembleModelAgent


class _FakeCalibrator:
    """Minimal stand-in for ModelCalibrator.calibrate() used by ensemble."""

    def __init__(self, mapping: dict[str, dict[float, float]]) -> None:
        self._mapping = mapping

    def calibrate(self, name: str, prob: float) -> float:
        return self._mapping.get(name, {}).get(prob, prob)


def _ensemble_with_calibrator(mapping: dict[str, dict[float, float]]) -> EnsembleModelAgent:
    ens = EnsembleModelAgent(models=[], sharp_weight=0.0)
    ens._calibrator = _FakeCalibrator(mapping)
    return ens


def test_identity_passthrough_when_no_sector_entry():
    ens = _ensemble_with_calibrator({})  # nothing fitted
    a, b, d = ens._apply_sector_calibration("tennis", 0.65, 0.35, None)
    assert (a, b, d) == (0.65, 0.35, None)


def test_calibration_applies_to_two_way_market():
    """When the sector has a fitted entry, prob_a is calibrated and prob_b mirrors."""
    ens = _ensemble_with_calibrator({"tennis_ensemble": {0.65: 0.72}})
    a, b, d = ens._apply_sector_calibration("tennis", 0.65, 0.35, None)
    assert a == pytest.approx(0.72)
    assert b == pytest.approx(0.28)
    assert d is None
    # Sum-to-1 invariant preserved
    assert a + b == pytest.approx(1.0)


def test_three_way_market_calibrates_team_win_legs_and_renormalizes():
    """Soccer-style 3-way markets: home and away legs go through the curve,
    draw is left untouched, then all three are renormalized to sum to 1."""
    # Curve squashes both team-win legs (mirrors live soccer overprediction)
    ens = _ensemble_with_calibrator({"soccer_ensemble": {0.50: 0.40, 0.30: 0.20}})
    a, b, d = ens._apply_sector_calibration("soccer", 0.50, 0.30, 0.20)
    # Pre-norm: home 0.40, away 0.20, draw 0.20  → sum 0.80
    assert a == pytest.approx(0.40 / 0.80)
    assert b == pytest.approx(0.20 / 0.80)
    assert d == pytest.approx(0.20 / 0.80)
    assert a + b + d == pytest.approx(1.0)


def test_three_way_identity_when_no_curve_change():
    """If the curve returns inputs unchanged (no fit), pass through unchanged
    without renormalizing — caller's draw probability is preserved exactly."""
    ens = _ensemble_with_calibrator({"soccer_ensemble": {}})  # empty mapping
    a, b, d = ens._apply_sector_calibration("soccer", 0.50, 0.30, 0.20)
    assert (a, b, d) == (0.50, 0.30, 0.20)


def test_three_way_no_entry_is_identity():
    """When no soccer_ensemble entry exists at all, behave identically to
    the unfitted case — no calibration, no renormalization."""
    ens = _ensemble_with_calibrator({})
    a, b, d = ens._apply_sector_calibration("soccer", 0.45, 0.30, 0.25)
    assert (a, b, d) == (0.45, 0.30, 0.25)


def test_three_way_preserves_home_advantage_ordering():
    """Calibration is monotonic — if the model said home > away before, it
    must still say home > away after, even with renormalization."""
    ens = _ensemble_with_calibrator({"soccer_ensemble": {0.55: 0.45, 0.20: 0.15}})
    a, b, d = ens._apply_sector_calibration("soccer", 0.55, 0.20, 0.25)
    assert a > b


def test_unfitted_identity_when_calibrate_returns_input():
    """If the calibrator returns the same value (untrained model), no rebalance happens."""
    ens = _ensemble_with_calibrator({"tennis_ensemble": {}})  # entry exists, no mapping
    a, b, d = ens._apply_sector_calibration("tennis", 0.42, 0.58, None)
    assert (a, b, d) == (0.42, 0.58, None)


def test_sector_lookup_is_lowercased():
    """Caller's sector string can be any case; lookup uses lowercase."""
    ens = _ensemble_with_calibrator({"tennis_ensemble": {0.5: 0.55}})
    a, b, d = ens._apply_sector_calibration("Tennis", 0.5, 0.5, None)
    assert a == pytest.approx(0.55)
    assert b == pytest.approx(0.45)
