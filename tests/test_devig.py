"""Tests for the Power Method devigging."""

import pytest
from evmax.ev.devig import (
    devig_power_method,
    devig_two_way,
    devig_three_way,
    american_to_decimal,
    decimal_to_american,
)


class TestDevigPowerMethod:
    def test_two_way_balanced(self):
        """Balanced 50/50 market with vig should return ~0.5/0.5."""
        # Both sides at -110 (American) = 1.909 decimal
        result = devig_power_method([1.909, 1.909])
        assert abs(result.true_probs[0] - 0.5) < 0.01
        assert abs(result.true_probs[1] - 0.5) < 0.01
        assert abs(sum(result.true_probs) - 1.0) < 1e-6
        assert result.margin > 0

    def test_two_way_favorite(self):
        """Favorite/underdog market: favorite should have higher true prob."""
        # Home favorite at -200 (1.5 decimal), away at +180 (2.8 decimal)
        result = devig_power_method([1.5, 2.8])
        prob_fav, prob_dog = result.true_probs
        assert prob_fav > prob_dog
        assert abs(sum(result.true_probs) - 1.0) < 1e-6

    def test_three_way_soccer(self):
        """Three-way soccer market should sum to 1."""
        # Home 2.1, Away 3.5, Draw 3.2
        result = devig_power_method([2.1, 3.5, 3.2])
        assert len(result.true_probs) == 3
        assert abs(sum(result.true_probs) - 1.0) < 1e-6
        assert all(p > 0 for p in result.true_probs)

    def test_margin_calculation(self):
        """Margin should be positive (vig exists)."""
        result = devig_power_method([1.91, 2.05])
        assert result.margin > 0
        assert result.margin < 0.10  # Sane margin < 10%

    def test_invalid_odds_raises(self):
        """Odds <= 1.0 should raise ValueError."""
        with pytest.raises(ValueError):
            devig_power_method([1.0, 2.0])
        with pytest.raises(ValueError):
            devig_power_method([0.5, 2.0])

    def test_single_outcome_raises(self):
        """Single outcome should raise ValueError."""
        with pytest.raises(ValueError):
            devig_power_method([2.0])


class TestDevigConvenience:
    def test_two_way_returns_tuple(self):
        prob_a, prob_b, margin = devig_two_way(1.91, 2.05)
        assert 0 < prob_a < 1
        assert 0 < prob_b < 1
        assert abs(prob_a + prob_b - 1.0) < 1e-6
        assert margin > 0

    def test_three_way_returns_tuple(self):
        prob_a, prob_b, prob_draw, margin = devig_three_way(2.1, 3.5, 3.2)
        assert abs(prob_a + prob_b + prob_draw - 1.0) < 1e-6

    def test_american_to_decimal_positive(self):
        """Positive American odds (underdog)."""
        assert abs(american_to_decimal(100) - 2.0) < 0.001
        assert abs(american_to_decimal(200) - 3.0) < 0.001

    def test_american_to_decimal_negative(self):
        """Negative American odds (favorite)."""
        assert abs(american_to_decimal(-110) - 1.909) < 0.01
        assert abs(american_to_decimal(-200) - 1.5) < 0.001

    def test_round_trip_conversion(self):
        """American → decimal → American should be close."""
        original = -150
        decimal = american_to_decimal(original)
        back = decimal_to_american(decimal)
        assert abs(back - original) <= 1  # Rounding tolerance
