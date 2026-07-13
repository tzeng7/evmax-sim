"""Isotonic calibrator (PAV) tests — monotonicity, order preservation, and that
a compressed/under-confident input gets stretched toward the truth it was fit
on."""

import numpy as np
import pytest

from evmax.golf.calibration import IsotonicCalibrator, brier


class TestPav:
    def test_output_is_monotone_nondecreasing(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 1, 500)
        y = (rng.uniform(0, 1, 500) < x).astype(float)  # truth ≈ x
        cal = IsotonicCalibrator.fit(x, y)
        grid = np.linspace(0, 1, 50)
        out = cal.transform(grid)
        assert np.all(np.diff(out) >= -1e-9)  # non-decreasing

    def test_preserves_ranking(self):
        # calibration must never reorder players
        cal = IsotonicCalibrator.fit([0.1, 0.2, 0.3, 0.4], [0, 0, 1, 1])
        a, b = cal.transform_one(0.15), cal.transform_one(0.35)
        assert a <= b

    def test_recovers_underconfident_map(self):
        # Build data where truth is 2x the stated prob (under-confident), in a
        # region where the linear map stays < 1. Isotonic should lift predictions
        # toward the realised rate.
        rng = np.random.default_rng(1)
        p = rng.uniform(0, 0.4, 4000)
        y = (rng.uniform(0, 1, 4000) < np.minimum(2 * p, 1)).astype(float)
        cal = IsotonicCalibrator.fit(p, y)
        # a stated 0.2 should map up toward ~0.4
        assert cal.transform_one(0.2) > 0.30

    def test_identity_on_calibrated_data(self):
        rng = np.random.default_rng(2)
        p = rng.uniform(0, 1, 5000)
        y = (rng.uniform(0, 1, 5000) < p).astype(float)  # already calibrated
        cal = IsotonicCalibrator.fit(p, y)
        for q in (0.2, 0.5, 0.8):
            assert cal.transform_one(q) == pytest.approx(q, abs=0.08)


class TestTransform:
    def test_clips_to_unit_interval(self):
        cal = IsotonicCalibrator.fit([0.1, 0.9], [0, 1])
        out = cal.transform([-0.5, 0.5, 1.5])
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_empty_fit_is_identity(self):
        cal = IsotonicCalibrator.fit([], [])
        assert cal.transform_one(0.3) == pytest.approx(0.3)

    def test_weights_respected(self):
        # heavily-weighted high-outcome point pulls its level up
        cal = IsotonicCalibrator.fit([0.5, 0.5], [0.0, 1.0], weights=[1.0, 9.0])
        assert cal.transform_one(0.5) > 0.8


class TestBrier:
    def test_perfect_prediction_zero(self):
        assert brier([1.0, 0.0], [1.0, 0.0]) == 0.0

    def test_known_value(self):
        # (0.5-1)^2 + (0.5-0)^2 = 0.25 + 0.25, mean 0.25
        assert brier([0.5, 0.5], [1.0, 0.0]) == pytest.approx(0.25)
