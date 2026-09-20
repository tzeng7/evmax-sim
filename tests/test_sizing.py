"""Tests for the shared sizing entry point (evmax/ev/sizing.py)."""

from __future__ import annotations

import math

import pytest

from evmax.ev.kelly import compute_kelly
from evmax.ev.sizing import (
    ShrinkageCoeffs,
    ShrinkageModel,
    SizingConfig,
    depth_liquidity_discount,
    edge_is_quarantined,
    load_shrinkage_model,
    size_position,
)


# --------------------------------------------------------------------------
# Identity: with all layers off, size_position == compute_kelly, byte for byte.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("true_prob", [0.30, 0.45, 0.55, 0.72, 0.90])
@pytest.mark.parametrize("price", [0.25, 0.40, 0.50, 0.65, 0.85])
@pytest.mark.parametrize("base_fraction", [0.25, 0.5])
def test_identity_when_all_layers_off(true_prob, price, base_fraction):
    payout = 1.0 / price
    edge = true_prob * payout - 1.0
    ref = compute_kelly(
        true_prob=true_prob, payout_decimal=payout, edge_pct=edge,
        spread_pct=0.03, base_fraction=base_fraction, max_kelly=0.05,
    )
    got = size_position(
        true_prob=true_prob, payout_decimal=payout, edge_pct=edge,
        sector="nba", price=price, spread_pct=0.03,
        base_fraction=base_fraction, max_kelly=0.05,
        config=SizingConfig(),  # all off
    )
    assert got.kelly_fraction == ref.kelly_fraction
    assert got.kelly_full == ref.kelly_full


def test_identity_when_config_none():
    got = size_position(
        true_prob=0.6, payout_decimal=2.0, edge_pct=0.2,
        price=0.5, spread_pct=0.0, base_fraction=0.5, max_kelly=0.05, config=None,
    )
    ref = compute_kelly(true_prob=0.6, payout_decimal=2.0, edge_pct=0.2,
                        spread_pct=0.0, base_fraction=0.5, max_kelly=0.05)
    assert got.kelly_fraction == ref.kelly_fraction


# --------------------------------------------------------------------------
# Edge shrinkage
# --------------------------------------------------------------------------

def _identity_model() -> ShrinkageModel:
    # a=0, s_b=1, s_p=0 → predict() returns blended unchanged.
    return ShrinkageModel(pooled=ShrinkageCoeffs(0.0, 1.0, 0.0, n=9999))


def test_shrinkage_identity_coeffs_leave_prob_unchanged():
    m = _identity_model()
    assert m.sizing_probability(0.7, 0.5, "nba") == pytest.approx(0.7, abs=1e-9)


def test_shrinkage_toward_price_reduces_favored_stake():
    # s_b < 1, s_p > 0 pulls a high blended prob down toward the (lower) price,
    # shrinking the sized fraction vs the raw blend.
    model = ShrinkageModel(pooled=ShrinkageCoeffs(0.0, 0.4, 0.5, n=9999))
    cfg = SizingConfig(edge_shrinkage_enabled=True, shrinkage_model=model)
    raw = size_position(true_prob=0.70, payout_decimal=1/0.45, edge_pct=0.55,
                        sector="nba", price=0.45, base_fraction=0.5, max_kelly=0.5,
                        config=SizingConfig())
    shr = size_position(true_prob=0.70, payout_decimal=1/0.45, edge_pct=0.55,
                        sector="nba", price=0.45, base_fraction=0.5, max_kelly=0.5,
                        config=cfg)
    assert shr.kelly_fraction < raw.kelly_fraction


def test_shrinkage_skipped_without_price():
    model = ShrinkageModel(pooled=ShrinkageCoeffs(0.0, 0.4, 0.5, n=9999))
    cfg = SizingConfig(edge_shrinkage_enabled=True, shrinkage_model=model)
    got = size_position(true_prob=0.70, payout_decimal=1/0.45, edge_pct=0.55,
                        sector="nba", price=None, base_fraction=0.5, max_kelly=0.5,
                        config=cfg)
    ref = size_position(true_prob=0.70, payout_decimal=1/0.45, edge_pct=0.55,
                        sector="nba", price=None, base_fraction=0.5, max_kelly=0.5,
                        config=SizingConfig())
    assert got.kelly_fraction == ref.kelly_fraction


def test_per_sector_fit_requires_min_n():
    # A thin per-sector fit (n below threshold) falls back to pooled.
    pooled = ShrinkageCoeffs(0.0, 1.0, 0.0, n=5000)
    thin = ShrinkageCoeffs(2.0, 0.1, 0.1, n=10)  # would be very different if used
    m = ShrinkageModel(pooled=pooled, sectors={"lol": thin})
    # thin sector → pooled (identity) applies
    assert m.sizing_probability(0.6, 0.5, "lol") == pytest.approx(0.6, abs=1e-9)


def test_coeffs_for_uses_powered_sector():
    pooled = ShrinkageCoeffs(0.0, 1.0, 0.0, n=5000)
    powered = ShrinkageCoeffs(0.0, 0.5, 0.5, n=500)
    m = ShrinkageModel(pooled=pooled, sectors={"tennis": powered})
    assert m.coeffs_for("tennis") is powered
    assert m.coeffs_for("unknown") is pooled


# --------------------------------------------------------------------------
# Depth-keyed liquidity discount
# --------------------------------------------------------------------------

def test_depth_discount_math():
    # $100 depth, $50 stake, alpha 1 → min(1, 100/50) = 1 (no discount).
    assert depth_liquidity_discount(100, 50, alpha=1.0) == pytest.approx(1.0)
    # $30 depth, $60 stake → 0.5.
    assert depth_liquidity_discount(30, 60, alpha=1.0) == pytest.approx(0.5)
    # empty book (0 depth) → floor.
    assert depth_liquidity_discount(0, 60, alpha=1.0, floor=0.1) == pytest.approx(0.1)


def test_depth_discount_none_without_inputs():
    assert depth_liquidity_discount(None, 50) is None
    assert depth_liquidity_discount(50, None) is None
    assert depth_liquidity_discount(50, 0) is None


def test_size_position_depth_caps_stake():
    # Big edge → cap 5% wants $250 of a $5000 bankroll, but only $50 rests.
    cfg = SizingConfig(liquidity_depth_enabled=True, depth_alpha=1.0)
    got = size_position(true_prob=0.75, payout_decimal=1/0.5, edge_pct=0.5,
                        sector="wnba", price=0.5, base_fraction=0.5, max_kelly=0.05,
                        depth_usd=50.0, bankroll=5000.0, config=cfg)
    # stake ≈ frac*5000 should be capped near $50 → frac ≈ 0.01.
    assert got.kelly_fraction * 5000.0 <= 55.0
    assert got.kelly_fraction > 0


def test_size_position_depth_noop_without_bankroll():
    cfg = SizingConfig(liquidity_depth_enabled=True)
    got = size_position(true_prob=0.75, payout_decimal=1/0.5, edge_pct=0.5,
                        sector="wnba", price=0.5, base_fraction=0.5, max_kelly=0.05,
                        depth_usd=50.0, bankroll=None, config=cfg)
    ref = size_position(true_prob=0.75, payout_decimal=1/0.5, edge_pct=0.5,
                        sector="wnba", price=0.5, base_fraction=0.5, max_kelly=0.05,
                        config=SizingConfig())
    assert got.kelly_fraction == ref.kelly_fraction


# --------------------------------------------------------------------------
# Quarantine safety gate
# --------------------------------------------------------------------------

def test_quarantine_off_by_default():
    assert edge_is_quarantined(0.9, 0.2, 0.2, pp=0.0) is False


def test_quarantine_fires_on_gross_double_disagreement():
    # blended 0.90 vs sharp 0.30 vs price 0.32 → 60pp / 58pp off both → quarantine.
    assert edge_is_quarantined(0.90, 0.30, 0.32, pp=0.25) is True


def test_quarantine_ignores_legitimate_edge():
    # blended agrees with sharp (0.62 vs 0.60), disagrees with a stale price (0.45).
    # That is an edge, not a wrong probability — must NOT quarantine.
    assert edge_is_quarantined(0.62, 0.60, 0.45, pp=0.25) is False


def test_quarantine_needs_both_anchors():
    assert edge_is_quarantined(0.9, None, 0.2, pp=0.25) is False
    assert edge_is_quarantined(0.9, 0.2, None, pp=0.25) is False


# --------------------------------------------------------------------------
# State loading
# --------------------------------------------------------------------------

def test_load_missing_state_returns_none(tmp_path):
    assert load_shrinkage_model(tmp_path / "nope.json") is None


def test_load_roundtrip(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(
        '{"pooled":{"a":0.1,"s_b":0.8,"s_p":0.3,"n":1000},'
        '"sectors":{"nba":{"a":0.0,"s_b":0.5,"s_p":0.5,"n":400}}}'
    )
    m = load_shrinkage_model(p)
    assert m is not None
    assert m.pooled.s_b == pytest.approx(0.8)
    assert m.coeffs_for("nba").s_p == pytest.approx(0.5)


def test_shipped_state_loads_if_present():
    # The committed state file, if present, must parse (guards a bad hand-edit).
    m = load_shrinkage_model()
    if m is not None:
        assert m.pooled is not None or m.sectors
