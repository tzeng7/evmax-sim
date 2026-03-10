"""Tests for the Kelly Criterion calculator."""

import pytest
from evmax.ev.kelly import kelly_full, compute_kelly


class TestKellyFull:
    def test_even_money_edge(self):
        """At 55% win rate with 2.0 payout, full Kelly = 10%."""
        # K = (0.55 * 1 - 0.45) / 1 = 0.10
        k = kelly_full(true_prob=0.55, payout_decimal=2.0)
        assert abs(k - 0.10) < 0.001

    def test_negative_ev_returns_nonpositive(self):
        """Negative EV bet should return ≤ 0 Kelly."""
        k = kelly_full(true_prob=0.40, payout_decimal=2.0)
        assert k <= 0

    def test_zero_edge_zero_kelly(self):
        """Exactly fair bet: Kelly = 0."""
        k = kelly_full(true_prob=0.50, payout_decimal=2.0)
        assert abs(k) < 1e-9

    def test_large_payout_large_kelly(self):
        """Higher payout odds → larger Kelly fraction."""
        k_low = kelly_full(true_prob=0.60, payout_decimal=1.5)
        k_high = kelly_full(true_prob=0.60, payout_decimal=3.0)
        assert k_high > k_low

    def test_invalid_payout(self):
        """Payout ≤ 1 should return 0."""
        k = kelly_full(true_prob=0.60, payout_decimal=1.0)
        assert k == 0.0


class TestComputeKelly:
    def test_quarter_kelly_baseline(self):
        """Default base_fraction=0.25 → result ≤ 0.25 × full Kelly."""
        k_full = kelly_full(0.55, 2.0)
        result = compute_kelly(
            true_prob=0.55,
            payout_decimal=2.0,
            edge_pct=0.10,
            spread_pct=0.0,
        )
        assert result.suggested_units <= k_full * 0.25 + 1e-9

    def test_hard_cap_at_max_kelly(self):
        """Suggested units should never exceed max_kelly."""
        result = compute_kelly(
            true_prob=0.80,
            payout_decimal=3.0,
            edge_pct=0.50,
            spread_pct=0.0,
            max_kelly=0.05,
        )
        assert result.suggested_units <= 0.05

    def test_confidence_discount(self):
        """Lower edge_pct → lower confidence → smaller bet."""
        result_high_edge = compute_kelly(0.55, 2.0, edge_pct=0.20, spread_pct=0.0)
        result_low_edge = compute_kelly(0.55, 2.0, edge_pct=0.04, spread_pct=0.0)
        assert result_high_edge.suggested_units >= result_low_edge.suggested_units

    def test_liquidity_discount(self):
        """Higher spread → lower liquidity discount → smaller bet."""
        result_tight = compute_kelly(0.55, 2.0, edge_pct=0.10, spread_pct=0.01)
        result_wide = compute_kelly(0.55, 2.0, edge_pct=0.10, spread_pct=0.15)
        assert result_tight.suggested_units >= result_wide.suggested_units

    def test_negative_ev_returns_zero(self):
        """Negative EV market should return zero units."""
        result = compute_kelly(
            true_prob=0.40,
            payout_decimal=2.0,
            edge_pct=-0.10,
        )
        assert result.suggested_units == 0.0

    def test_at_threshold_edge(self):
        """2% edge should result in a small positive bet."""
        result = compute_kelly(
            true_prob=0.51,
            payout_decimal=1.0 / 0.49,
            edge_pct=0.02,
        )
        assert result.suggested_units > 0
