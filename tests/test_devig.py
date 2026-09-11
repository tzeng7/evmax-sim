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


# ---------------------------------------------------------------------------
# Shin + multiplicative devig (A/B alternatives to Power) and the dispatcher.
# Validated by mathematical properties, not against prod — the closed form is
# deterministic and its invariants (sum-to-1, favorite-longshot direction,
# fair-book reduction) are checkable without market data.
# ---------------------------------------------------------------------------
from evmax.ev.devig import devig, devig_shin, devig_multiplicative  # noqa: E402


class TestMultiplicativeDevig:
    def test_proportional_and_sums_to_one(self):
        r = devig_multiplicative([1.5, 2.8])
        raw = [1 / 1.5, 1 / 2.8]
        total = sum(raw)
        assert r.true_probs[0] == pytest.approx(raw[0] / total)
        assert sum(r.true_probs) == pytest.approx(1.0)
        assert r.method == "multiplicative"

    def test_favorite_higher(self):
        r = devig_multiplicative([1.5, 2.8])
        assert r.true_probs[0] > r.true_probs[1]

    def test_rejects_bad_odds(self):
        with pytest.raises(ValueError):
            devig_multiplicative([1.0, 2.0])
        with pytest.raises(ValueError):
            devig_multiplicative([2.0])


class TestShinDevig:
    def test_sums_to_one_and_z_in_range(self):
        r = devig_shin([1.5, 2.8])
        assert sum(r.true_probs) == pytest.approx(1.0)
        assert r.z is not None and 0.0 <= r.z < 1.0
        assert r.method == "shin"

    def test_symmetric_book_is_half_half_with_known_z(self):
        # Hand-derived from the closed form for two -110 sides (1.90909 dec):
        # π=1/1.90909=0.52381, o=1.04762, π²/o=0.26191; setting p=0.5 gives
        # z² − 1.04763·z + 0.04763 = 0 → z ≈ 0.0476, probs 0.5/0.5.
        r = devig_shin([1.90909, 1.90909])
        assert r.true_probs[0] == pytest.approx(0.5, abs=1e-6)
        assert r.true_probs[1] == pytest.approx(0.5, abs=1e-6)
        assert r.z == pytest.approx(0.0476, abs=2e-3)

    def test_shades_longshot_down_vs_proportional(self):
        # THE point of Shin: correct favorite-longshot bias. The favourite's
        # true prob is HIGHER, the longshot's LOWER, than the proportional devig.
        odds = [1.4, 3.0]
        shin = devig_shin(odds)
        prop = devig_multiplicative(odds)
        assert shin.true_probs[0] > prop.true_probs[0]  # favourite up
        assert shin.true_probs[1] < prop.true_probs[1]  # longshot down
        # Hand-computed reference (z ≈ 0.0497): ~[0.690, 0.309].
        assert shin.true_probs[0] == pytest.approx(0.690, abs=3e-3)
        assert shin.true_probs[1] == pytest.approx(0.309, abs=3e-3)

    def test_fair_book_reduces_to_proportional(self):
        # No overround → no insider signal → z≈0 → proportional probs.
        r = devig_shin([2.0, 2.0])
        assert r.true_probs[0] == pytest.approx(0.5)
        assert r.z == pytest.approx(0.0, abs=1e-9)

    def test_three_way_sums_to_one(self):
        r = devig_shin([2.1, 3.5, 3.2])
        assert len(r.true_probs) == 3
        assert sum(r.true_probs) == pytest.approx(1.0)
        assert all(p > 0 for p in r.true_probs)

    def test_monotonic_in_raw_prob(self):
        # Higher raw implied (shorter odds) → higher devigged prob, always.
        r = devig_shin([1.3, 2.5, 6.0])
        assert r.true_probs[0] > r.true_probs[1] > r.true_probs[2]


class TestDevigDispatch:
    def test_power_is_default_and_unchanged(self):
        # The dispatcher's power path must equal the direct power method exactly
        # — this is what keeps the shipped devig_method="power" behaviour inert.
        from evmax.ev.devig import devig_power_method

        direct = devig_power_method([1.5, 2.8])
        via = devig([1.5, 2.8], method="power")
        assert via.true_probs == direct.true_probs

    def test_two_way_default_matches_power(self):
        # devig_two_way with no method arg is byte-identical to the old behaviour.
        a, b, m = devig_two_way(1.5, 2.8)
        from evmax.ev.devig import devig_power_method

        r = devig_power_method([1.5, 2.8])
        assert (a, b) == (r.true_probs[0], r.true_probs[1])

    def test_method_selects_algorithm(self):
        assert devig([1.5, 2.8], method="shin").method == "shin"
        assert devig([1.5, 2.8], method="multiplicative").method == "multiplicative"

    def test_unknown_method_falls_back_to_power(self):
        r = devig([1.5, 2.8], method="bogus")
        assert r.method == "power"

    def test_two_way_and_three_way_forward_method(self):
        # The Pinnacle client passes method= through these wrappers.
        a, b, _ = devig_two_way(1.4, 3.0, method="shin")
        assert a > b  # favourite still higher, just Shin-shaded
        pa, pb, pd, _ = devig_three_way(2.1, 3.5, 3.2, method="multiplicative")
        assert pa + pb + pd == pytest.approx(1.0)


class TestPerSectorDevigResolution:
    def test_default_is_power_everywhere(self):
        # Empty override map (shipped state) → every sector resolves to the
        # global fallback, so the operator changes nothing out of the box.
        from evmax.ev.devig import resolve_devig_method

        assert resolve_devig_method("soccer", "power") == "power"
        assert resolve_devig_method("nba", "power") == "power"
        assert resolve_devig_method(None, "power") == "power"

    def test_baked_override_wins_over_global_default(self, monkeypatch):
        # Simulate promoting soccer to shin once its CLV clears — one map entry,
        # permanent and automatic, with no runtime flag.
        import evmax.ev.devig as d

        monkeypatch.setattr(d, "DEVIG_METHOD_BY_SECTOR", {"soccer": "shin"})
        assert d.resolve_devig_method("soccer", "power") == "shin"
        assert d.resolve_devig_method("nba", "power") == "power"  # untouched sectors

    def test_case_insensitive_sector(self, monkeypatch):
        import evmax.ev.devig as d

        monkeypatch.setattr(d, "DEVIG_METHOD_BY_SECTOR", {"soccer": "shin"})
        assert d.resolve_devig_method("Soccer", "power") == "shin"
