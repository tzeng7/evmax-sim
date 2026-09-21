"""Unit tests for the alt-spread ladder replay join helpers
(evmax/models_ml/spread_ladder_eval).

These pin the two error-prone pieces the replay depends on: the rung-distance
bucketing and the YES-side alignment of an archived favorite-cover ladder onto a
signed row line (favorite laying = negative, underdog getting = positive).
"""
from __future__ import annotations

import math

import pytest

from evmax.models_ml.spread_ladder_eval import (
    BUCKET_ORDER,
    brier,
    ladder_yes_prob,
    rung_distance_bucket,
)


class TestRungDistanceBucket:
    @pytest.mark.parametrize("dist,expected", [
        (0.0, "at-line (<=1)"),
        (1.0, "at-line (<=1)"),
        (1.5, "near (1-4)"),
        (4.0, "near (1-4)"),
        (4.5, "mid (4-8)"),
        (8.0, "mid (4-8)"),
        (8.5, "deep tail (>8)"),
        (20.0, "deep tail (>8)"),
    ])
    def test_boundaries(self, dist, expected):
        assert rung_distance_bucket(dist) == expected

    def test_sign_insensitive(self):
        assert rung_distance_bucket(-13.5) == rung_distance_bucket(13.5)

    def test_every_bucket_is_declared(self):
        seen = {rung_distance_bucket(d) for d in (0, 2, 6, 12)}
        assert seen == set(BUCKET_ORDER)


class TestLadderYesProb:
    # Favorite covers -7.5 with prob 0.40; the underdog side is the complement.
    LADDER = {-3.0: 0.52, -7.5: 0.40, -16.5: 0.12}

    def test_favorite_side_reads_cover_prob_directly(self):
        # row_line negative → YES is the favorite laying points.
        assert ladder_yes_prob(-7.5, self.LADDER) == pytest.approx(0.40)
        assert ladder_yes_prob(-16.5, self.LADDER) == pytest.approx(0.12)

    def test_underdog_side_is_complement(self):
        # row_line positive → YES is the underdog getting points.
        assert ladder_yes_prob(7.5, self.LADDER) == pytest.approx(0.60)
        assert ladder_yes_prob(16.5, self.LADDER) == pytest.approx(0.88)

    def test_matches_within_tolerance(self):
        # 7.6 rounds onto the -7.5 rung within the 0.5 tolerance.
        assert ladder_yes_prob(-7.6, self.LADDER, tolerance=0.5) == pytest.approx(0.40)

    def test_no_rung_within_tolerance_returns_none(self):
        assert ladder_yes_prob(-11.0, self.LADDER, tolerance=0.5) is None

    def test_empty_ladder_returns_none(self):
        assert ladder_yes_prob(-7.5, {}, tolerance=0.5) is None

    def test_pickem_line_returns_none(self):
        assert ladder_yes_prob(0.0, self.LADDER) is None

    def test_picks_nearest_when_multiple_in_tolerance(self):
        ladder = {-7.0: 0.42, -7.5: 0.40}
        # -7.4 is closer to -7.5 (0.1) than -7.0 (0.4).
        assert ladder_yes_prob(-7.4, ladder, tolerance=1.0) == pytest.approx(0.40)


class TestBrier:
    def test_known_value(self):
        # (0.4-0)^2 + (0.4-1)^2 = 0.16 + 0.36 = 0.52; /2 = 0.26
        assert brier([(0.4, 0), (0.4, 1)]) == pytest.approx(0.26)

    def test_perfect_prediction_is_zero(self):
        assert brier([(1.0, 1), (0.0, 0)]) == pytest.approx(0.0)

    def test_empty_is_nan(self):
        assert math.isnan(brier([]))
