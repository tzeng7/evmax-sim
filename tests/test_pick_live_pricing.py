"""Tests for recompute_at_price — the pure helper that lets `evmax agents pick`
gate and size a bet at the CURRENT Kalshi ask instead of the stale scan price.

This is the entry-timing fix: the fattest scan-time edges are the most likely to
have reverted by the time you place, so a bet only counts as live if its edge
survives to the live price.
"""
from __future__ import annotations

import pytest

from evmax.cli.commands.agents import recompute_at_price

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
