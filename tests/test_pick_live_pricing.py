"""Tests for recompute_at_price / reblend_with_fresh_sharp — the pure helpers
that let `evmax agents pick` gate and size a bet at CURRENT prices instead of
the stale scan-time snapshot.

This is the entry-timing fix: the fattest scan-time edges are the most likely to
have reverted by the time you place, so a bet only counts as live if its edge
survives to the live price. recompute_at_price handles the Kalshi side (the
ask); reblend_with_fresh_sharp handles the Pinnacle side (blended_true_prob) —
without it, EV/Kelly at pick time were computed from a sharp line frozen at
scan time even though Pinnacle drifts meaningfully (median ~0.8pp / p90 ~4pp
on resolved bets) between an early scan and a T-60 pick window, flipping the
bet/no-bet decision on ~19% of rows in a backtest over resolved live bets.
"""
from __future__ import annotations

import pytest

from evmax.cli.commands.agents import recompute_at_price, reblend_with_fresh_sharp

_COMMON = dict(base_kelly=0.5, max_kelly=0.05, bankroll=1000.0, min_ev=0.02, min_prob=0.15)


def test_live_edge_survives_is_live_with_stake():
    # Fair 0.60 vs a 0.50 ask → ~20% edge, well above threshold.
    rc = recompute_at_price(blended_prob=0.60, price=0.50, **_COMMON)
    assert rc["is_live"] is True
    assert rc["ev"] > 0.02
    assert rc["stake"] > 0
    assert 0 < rc["kelly_fraction"] <= 0.05  # capped at max_kelly


def test_edge_eroded_to_zero_is_not_live():
    # Same fair 0.60 but the ask has risen to 0.60 → no edge → drop the bet.
    rc = recompute_at_price(blended_prob=0.60, price=0.60, **_COMMON)
    assert rc["is_live"] is False
    assert rc["stake"] == 0.0
    assert rc["kelly_fraction"] == 0.0


def test_price_past_fair_is_not_live():
    # Ask above fair → negative EV → never live.
    rc = recompute_at_price(blended_prob=0.55, price=0.70, **_COMMON)
    assert rc["is_live"] is False
    assert rc["stake"] == 0.0


def test_none_price_is_not_live():
    rc = recompute_at_price(blended_prob=0.60, price=None, **_COMMON)
    assert rc["is_live"] is False
    assert rc["ev"] is None
    assert rc["stake"] == 0.0


@pytest.mark.parametrize("price", [0.0, 1.0, 0.99, 1.5, -0.1])
def test_degenerate_prices_are_not_live(price):
    # Settled (0/1), empty-book (>=0.99), or out-of-range prices yield no bet.
    rc = recompute_at_price(blended_prob=0.60, price=price, **_COMMON)
    assert rc["is_live"] is False
    assert rc["stake"] == 0.0


def test_stake_scales_with_bankroll():
    small = recompute_at_price(blended_prob=0.60, price=0.50, **{**_COMMON, "bankroll": 500.0})
    big = recompute_at_price(blended_prob=0.60, price=0.50, **{**_COMMON, "bankroll": 2000.0})
    assert big["kelly_fraction"] == pytest.approx(small["kelly_fraction"])
    assert big["stake"] == pytest.approx(4 * small["stake"])


def test_low_prob_longshot_gated_out():
    # blended_prob below min_prob is excluded even if nominal EV is positive.
    rc = recompute_at_price(blended_prob=0.10, price=0.05, **_COMMON)
    assert rc["is_live"] is False


class TestReblendWithFreshSharp:
    """reblend_with_fresh_sharp — re-derives blended_true_prob from a fresh
    Pinnacle line using the same linear formula EnsembleModelAgent uses
    (blended = sharp_weight*sharp + (1-sharp_weight)*model), holding the
    model component fixed."""

    def test_sharp_moved_up_shifts_blend_by_weight(self):
        # Pinnacle moved +10pp (0.40 -> 0.50) with sharp_weight=0.80 -> blend
        # should shift by 0.80*0.10 = +0.08pp.
        updated = reblend_with_fresh_sharp(
            blended_prob=0.45, scan_sharp_prob=0.40, fresh_sharp_prob=0.50, sharp_weight=0.80,
        )
        assert updated == pytest.approx(0.45 + 0.08)

    def test_sharp_moved_down_shifts_blend_down(self):
        updated = reblend_with_fresh_sharp(
            blended_prob=0.45, scan_sharp_prob=0.40, fresh_sharp_prob=0.30, sharp_weight=0.80,
        )
        assert updated == pytest.approx(0.45 - 0.08)

    def test_sharp_weight_zero_no_change(self):
        # A fully model-driven blend shouldn't move even if Pinnacle drifted.
        updated = reblend_with_fresh_sharp(
            blended_prob=0.45, scan_sharp_prob=0.40, fresh_sharp_prob=0.70, sharp_weight=0.0,
        )
        assert updated == pytest.approx(0.45)

    def test_sharp_weight_one_is_pure_passthrough(self):
        # sharp_weight=1.0 means blended == scan_sharp_prob at scan time, so
        # re-blending should land exactly on the fresh sharp prob.
        updated = reblend_with_fresh_sharp(
            blended_prob=0.40, scan_sharp_prob=0.40, fresh_sharp_prob=0.63, sharp_weight=1.0,
        )
        assert updated == pytest.approx(0.63)

    @pytest.mark.parametrize("missing_field", ["scan_sharp_prob", "fresh_sharp_prob", "sharp_weight"])
    def test_missing_input_falls_back_to_frozen_blend(self, missing_field):
        kwargs = dict(blended_prob=0.45, scan_sharp_prob=0.40, fresh_sharp_prob=0.55, sharp_weight=0.75)
        kwargs[missing_field] = None
        assert reblend_with_fresh_sharp(**kwargs) == 0.45

    def test_clamped_to_valid_probability_range(self):
        # A large upward drift on an already-high blend must not exceed 1.
        high = reblend_with_fresh_sharp(
            blended_prob=0.97, scan_sharp_prob=0.50, fresh_sharp_prob=0.99, sharp_weight=1.0,
        )
        assert high <= 0.999
        # A large downward drift on an already-low blend must not go below 0.
        low = reblend_with_fresh_sharp(
            blended_prob=0.03, scan_sharp_prob=0.50, fresh_sharp_prob=0.01, sharp_weight=1.0,
        )
        assert low >= 0.001
