"""Unit tests for evmax/backtest/metrics.py — the Brier/accuracy/log-loss math
that produces every walk-forward number gating a shadow->live promotion. A
silent bug here corrupts promotion decisions, so lock the math down directly.
"""
from __future__ import annotations

import math

import pytest

from evmax.backtest.metrics import (
    _ev_bucket,
    accuracy_score,
    brier_score,
    log_loss_score,
)


class TestBrier:
    def test_basic(self):
        # (0.7-1)^2 = 0.09, (0.3-0)^2 = 0.09 -> mean 0.09
        assert brier_score([0.7, 0.3], [True, False]) == pytest.approx(0.09)

    def test_perfect_is_zero(self):
        assert brier_score([1.0, 0.0], [True, False]) == 0.0

    def test_worst_is_one(self):
        assert brier_score([0.0, 1.0], [True, False]) == pytest.approx(1.0)

    def test_empty_is_zero(self):
        assert brier_score([], []) == 0.0


class TestAccuracy:
    def test_threshold_is_inclusive_half(self):
        assert accuracy_score([0.5], [True]) == 1.0    # 0.5 predicts True
        assert accuracy_score([0.49], [True]) == 0.0

    def test_mixed(self):
        # 0.9->T✓  0.2->F✓  0.6->T vs F✗  => 2/3
        assert accuracy_score([0.9, 0.2, 0.6], [True, False, False]) == pytest.approx(2 / 3)

    def test_empty_is_zero(self):
        assert accuracy_score([], []) == 0.0


class TestLogLoss:
    def test_known_value(self):
        assert log_loss_score([0.9], [True]) == pytest.approx(-math.log(0.9))

    def test_extreme_prediction_is_clipped_finite(self):
        # p=1.0 with a False outcome would be +inf without the eps clip.
        assert math.isfinite(log_loss_score([1.0], [False]))

    def test_empty_is_zero(self):
        assert log_loss_score([], []) == 0.0


class TestEvBucket:
    def test_boundaries(self):
        assert _ev_bucket(-0.01) == "< 0%"
        assert _ev_bucket(0.0) == "0–2%"
        assert _ev_bucket(0.019) == "0–2%"
        assert _ev_bucket(0.02) == "2–5%"
        assert _ev_bucket(0.049) == "2–5%"
        assert _ev_bucket(0.05) == "5–10%"
        assert _ev_bucket(0.10) == "> 10%"
