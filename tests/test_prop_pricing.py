"""Tests for evmax/ev/prop_pricing.py — distribution-based Kalshi prop pricing.

Pinnacle posts ONE half-point line per (player, stat) with a devigged prob_over;
Kalshi posts MANY integer 'X+' thresholds per (player, stat). The pricing module
fits a parametric distribution (Poisson for count stats, Normal for high-mean /
yardage stats) to the Pinnacle anchor and reads off P(stat >= threshold) for
arbitrary Kalshi thresholds.

Coverage:
  - PoissonProp.from_anchor / prob_at_or_above — round-trip + monotonicity
  - NormalProp.from_anchor / prob_at_or_above — round-trip + σ sensitivity
  - fit_distribution dispatch by stat_type (NBA + NFL stats)
  - price_kalshi_threshold end-to-end convenience
  - Edge cases: degenerate probs, unknown stat, malformed inputs
"""
from __future__ import annotations

import math

import pytest

from evmax.ev.prop_pricing import (
    NegBinomialProp,
    NormalProp,
    PoissonProp,
    fit_distribution,
    price_kalshi_threshold,
    supported_stats,
)


# ===========================================================================
# PoissonProp
# ===========================================================================


class TestPoissonProp:
    def test_from_anchor_round_trip_at_anchor_cutoff(self):
        """Fitting (line=26.5, prob_over=0.55) and reading off P(X>=27) should
        return 0.55 — that's the constraint we fit to."""
        for prob in [0.30, 0.45, 0.50, 0.55, 0.65, 0.80]:
            d = PoissonProp.from_anchor(line=26.5, prob_over=prob)
            assert d is not None, f"Failed to fit for prob_over={prob}"
            recovered = d.prob_at_or_above(27)
            assert recovered == pytest.approx(prob, abs=1e-4)

    def test_from_anchor_low_line(self):
        """Threes line is typically 1.5-2.5. Fitting at low means must work."""
        d = PoissonProp.from_anchor(line=1.5, prob_over=0.55)
        assert d is not None
        assert d.prob_at_or_above(2) == pytest.approx(0.55, abs=1e-4)

    def test_prob_at_or_above_is_monotone_decreasing(self):
        d = PoissonProp.from_anchor(line=26.5, prob_over=0.50)
        probs = [d.prob_at_or_above(k) for k in range(20, 36)]
        # Strictly decreasing
        for a, b in zip(probs, probs[1:]):
            assert a > b, f"Not monotone: {probs}"

    def test_threshold_at_zero_returns_one(self):
        """Asking P(X >= 0) for any non-negative count is 1.0."""
        d = PoissonProp.from_anchor(line=26.5, prob_over=0.50)
        assert d.prob_at_or_above(0) == pytest.approx(1.0, abs=1e-9)

    def test_threshold_well_above_mean_is_near_zero(self):
        d = PoissonProp.from_anchor(line=5.5, prob_over=0.50)  # λ ≈ 6
        assert d.prob_at_or_above(50) < 1e-10

    def test_lambda_increases_with_higher_anchor_prob(self):
        """Higher prob_over at the same line means higher mean."""
        d_low  = PoissonProp.from_anchor(line=26.5, prob_over=0.40)
        d_high = PoissonProp.from_anchor(line=26.5, prob_over=0.60)
        assert d_low.lam < d_high.lam

    @pytest.mark.parametrize("bad_prob", [0.0, 1.0, -0.1, 1.1, float("nan")])
    def test_degenerate_probability_returns_none(self, bad_prob):
        assert PoissonProp.from_anchor(line=26.5, prob_over=bad_prob) is None

    def test_negative_line_returns_none(self):
        assert PoissonProp.from_anchor(line=-1.5, prob_over=0.50) is None

    def test_non_finite_inputs_return_none(self):
        assert PoissonProp.from_anchor(line=float("inf"), prob_over=0.50) is None
        assert PoissonProp.from_anchor(line=26.5, prob_over=float("nan")) is None


# ===========================================================================
# NegBinomialProp
# ===========================================================================


class TestNegBinomialProp:
    """NegBin handles overdispersed count stats — variance = μ + μ²/k. As k →∞
    it degenerates to Poisson. The basketball-analytics literature (Binomial
    Basketball, Squared Statistics) consistently picks NegBin for player
    scoring stats because Poisson under-weights the tail."""

    def test_from_anchor_round_trip(self):
        """Fitting (line=26.5, prob_over) and reading P(X>=27) returns prob_over."""
        for prob in [0.30, 0.45, 0.50, 0.55, 0.65, 0.80]:
            d = NegBinomialProp.from_anchor(line=26.5, prob_over=prob, k=13.0)
            assert d is not None
            recovered = d.prob_at_or_above(27)
            assert recovered == pytest.approx(prob, abs=1e-4)

    def test_low_line_works(self):
        """Threes line is typically 1.5-2.5. Fitting at low μ must work."""
        d = NegBinomialProp.from_anchor(line=1.5, prob_over=0.55, k=16.0)
        assert d is not None
        assert d.prob_at_or_above(2) == pytest.approx(0.55, abs=1e-4)

    def test_prob_at_or_above_is_monotone_decreasing(self):
        d = NegBinomialProp.from_anchor(line=26.5, prob_over=0.50, k=13.0)
        probs = [d.prob_at_or_above(k) for k in range(20, 60)]
        for a, b in zip(probs, probs[1:]):
            assert a > b

    def test_threshold_at_zero_returns_one(self):
        d = NegBinomialProp.from_anchor(line=5.5, prob_over=0.50, k=13.0)
        assert d.prob_at_or_above(0) == pytest.approx(1.0, abs=1e-9)

    def test_negbin_has_heavier_tail_than_poisson(self):
        """The whole reason for NegBin: at extreme thresholds, overdispersion
        gives meaningfully more probability mass than Poisson does. Both fit
        the same anchor (26.5, 0.55) — at threshold 50, NegBin must be
        substantially heavier."""
        nb = NegBinomialProp.from_anchor(line=26.5, prob_over=0.55, k=13.0)
        po = PoissonProp.from_anchor(line=26.5, prob_over=0.55)
        # At the anchor cutoff (27) both should hit ~0.55. Diverge in the tail.
        assert nb.prob_at_or_above(27) == pytest.approx(0.55, abs=1e-3)
        assert po.prob_at_or_above(27) == pytest.approx(0.55, abs=1e-3)
        # At threshold 45, NegBin should be at least 20× heavier than Poisson
        nb_far = nb.prob_at_or_above(45)
        po_far = po.prob_at_or_above(45)
        assert nb_far > 20 * po_far

    def test_higher_k_approaches_poisson(self):
        """As k → ∞ NegBin converges to Poisson — verify k=1000 produces
        nearly-identical tail to fitted Poisson."""
        nb = NegBinomialProp.from_anchor(line=26.5, prob_over=0.55, k=1000.0)
        po = PoissonProp.from_anchor(line=26.5, prob_over=0.55)
        for t in [25, 30, 35, 40]:
            assert nb.prob_at_or_above(t) == pytest.approx(po.prob_at_or_above(t), abs=0.01)

    def test_lower_k_produces_heavier_tail(self):
        """Lower k = more overdispersion = heavier tail at high thresholds."""
        nb_tight = NegBinomialProp.from_anchor(line=26.5, prob_over=0.50, k=50.0)
        nb_wide  = NegBinomialProp.from_anchor(line=26.5, prob_over=0.50, k=5.0)
        assert nb_wide.prob_at_or_above(45) > nb_tight.prob_at_or_above(45)

    @pytest.mark.parametrize("bad_prob", [0.0, 1.0, -0.1, 1.1, float("nan")])
    def test_degenerate_probability_returns_none(self, bad_prob):
        assert NegBinomialProp.from_anchor(line=26.5, prob_over=bad_prob, k=13.0) is None

    @pytest.mark.parametrize("bad_k", [0.0, -1.0, float("nan")])
    def test_invalid_k_returns_none(self, bad_k):
        assert NegBinomialProp.from_anchor(line=26.5, prob_over=0.50, k=bad_k) is None

    def test_negative_line_returns_none(self):
        assert NegBinomialProp.from_anchor(line=-1.5, prob_over=0.50, k=13.0) is None


# ===========================================================================
# NormalProp
# ===========================================================================


class TestNormalProp:
    def test_from_anchor_at_median(self):
        """When prob_over = 0.5, μ should equal the line exactly."""
        d = NormalProp.from_anchor(line=250.5, prob_over=0.50, sigma=70.0)
        assert d is not None
        assert d.mu == pytest.approx(250.5, abs=1e-9)

    def test_from_anchor_round_trip_at_anchor(self):
        """P(X > line) must equal prob_over after fitting (with continuity
        correction at threshold = line + 0.5)."""
        for prob in [0.40, 0.55, 0.70]:
            d = NormalProp.from_anchor(line=250.5, prob_over=prob, sigma=70.0)
            recovered = d.prob_at_or_above(251)  # 251 - 0.5 = 250.5 = line
            assert recovered == pytest.approx(prob, abs=1e-4)

    def test_higher_sigma_widens_tail(self):
        """At the same line and prob_over, larger σ increases tail probabilities
        far from the line (more uncertainty)."""
        d_tight = NormalProp.from_anchor(line=250.5, prob_over=0.55, sigma=30.0)
        d_wide  = NormalProp.from_anchor(line=250.5, prob_over=0.55, sigma=70.0)
        # P(X >= 400) is bigger for wide σ (more mass in upper tail)
        assert d_wide.prob_at_or_above(400) > d_tight.prob_at_or_above(400)

    def test_prob_at_or_above_is_monotone_decreasing(self):
        d = NormalProp.from_anchor(line=250.5, prob_over=0.55, sigma=70.0)
        probs = [d.prob_at_or_above(t) for t in range(100, 500, 25)]
        for a, b in zip(probs, probs[1:]):
            assert a > b

    @pytest.mark.parametrize("bad_sigma", [0.0, -1.0, float("nan")])
    def test_invalid_sigma_returns_none(self, bad_sigma):
        assert NormalProp.from_anchor(line=250.5, prob_over=0.55, sigma=bad_sigma) is None

    @pytest.mark.parametrize("bad_prob", [0.0, 1.0, -0.5, 2.0])
    def test_degenerate_probability_returns_none(self, bad_prob):
        assert NormalProp.from_anchor(line=250.5, prob_over=bad_prob, sigma=70.0) is None


# ===========================================================================
# fit_distribution dispatch
# ===========================================================================


class TestFitDistribution:
    @pytest.mark.parametrize("stat,line,prob,expected_cls", [
        # NBA count stats: NegBinomial (overdispersed; tail-heavier than Poisson)
        # Exception: points uses Normal — backtest shows Normal beats NegBin at
        # high-mean stats where σ ≪ μ.
        ("points",                  26.5,  0.55, NormalProp),
        ("rebounds",                 8.5,  0.50, NegBinomialProp),
        ("assists",                  6.5,  0.50, NegBinomialProp),
        ("threes",                   2.5,  0.45, NegBinomialProp),
        ("steals",                   1.5,  0.50, NegBinomialProp),
        ("blocks",                   1.5,  0.50, NegBinomialProp),
        # NBA composite + NFL yardage: Normal with per-stat σ
        ("points_rebounds_assists", 36.5,  0.50, NormalProp),
        ("passing_yards",          250.5,  0.55, NormalProp),
        ("rushing_yards",           75.5,  0.50, NormalProp),
        ("receiving_yards",         65.5,  0.50, NormalProp),
    ])
    def test_correct_distribution_per_stat(self, stat, line, prob, expected_cls):
        d = fit_distribution(stat, line, prob)
        assert isinstance(d, expected_cls)

    def test_unknown_stat_returns_none(self):
        assert fit_distribution("dunks", 5.5, 0.50) is None
        assert fit_distribution("", 5.5, 0.50) is None

    def test_supported_stats_lists_known_keys(self):
        s = supported_stats()
        assert "points" in s
        assert "passing_yards" in s
        assert "dunks" not in s


# ===========================================================================
# price_kalshi_threshold end-to-end
# ===========================================================================


class TestPriceKalshiThreshold:
    def test_nba_points_priced_at_multiple_thresholds(self):
        """The realistic case: Pinnacle anchor at 26.5 prob_over=0.55, Kalshi
        posts thresholds at 20/25/30/35. All should price under NegBin."""
        anchor = (26.5, 0.55)
        priced = {
            t: price_kalshi_threshold("points", *anchor, kalshi_threshold=t)
            for t in [20, 25, 30, 35]
        }
        for t, p in priced.items():
            assert p is not None, f"Failed to price threshold {t}"
            assert 0.0 <= p <= 1.0
        # Monotonically decreasing in threshold
        assert priced[20] > priced[25] > priced[30] > priced[35]

    def test_kalshi_threshold_at_anchor_cutoff_recovers_anchor_prob(self):
        """Kalshi threshold = ceil(line) should give back the anchor prob_over.
        Holds for both NegBin (exact at cutoff) and Normal (approximate via
        continuity correction)."""
        # NegBin path (rebounds)
        p_reb = price_kalshi_threshold("rebounds", 8.5, 0.55, kalshi_threshold=9)
        assert p_reb == pytest.approx(0.55, abs=1e-4)
        # Normal path (points + PRA) — continuity correction tolerance
        p_pts = price_kalshi_threshold("points", 26.5, 0.55, kalshi_threshold=27)
        assert p_pts == pytest.approx(0.55, abs=1e-3)
        p_pra = price_kalshi_threshold("points_rebounds_assists", 36.5, 0.55, kalshi_threshold=37)
        assert p_pra == pytest.approx(0.55, abs=1e-3)

    def test_nfl_pass_yards_priced(self):
        p = price_kalshi_threshold("passing_yards", 250.5, 0.55, kalshi_threshold=275)
        assert p is not None
        assert 0.0 < p < 0.55  # 275 above the line, below the anchor prob

    def test_unknown_stat_returns_none(self):
        assert price_kalshi_threshold("dunks", 5.5, 0.50, 4) is None

    def test_degenerate_prob_returns_none(self):
        assert price_kalshi_threshold("points", 26.5, 0.0, 25) is None
        assert price_kalshi_threshold("points", 26.5, 1.0, 25) is None
