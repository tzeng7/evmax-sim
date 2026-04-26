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


def test_three_way_market_skips_calibration():
    """Soccer-style markets with a draw probability bypass the YES-only calibrator."""
    ens = _ensemble_with_calibrator({"soccer_ensemble": {0.5: 0.6}})
    a, b, d = ens._apply_sector_calibration("soccer", 0.5, 0.3, 0.2)
    # Even though a fitted entry exists, draw markets are passed through
    assert (a, b, d) == (0.5, 0.3, 0.2)


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
