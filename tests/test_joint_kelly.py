"""Tests for correlation-aware joint Kelly sizing (evmax/ev/joint_kelly.py)."""

import pytest

from evmax.ev.joint_kelly import (
    AXIS_INDEPENDENT,
    AXIS_MARGIN,
    AXIS_TOTAL,
    JointKellyConfig,
    JointLeg,
    infer_axis_and_sign,
    joint_kelly_fractions,
)
from evmax.ev.kelly import compute_kelly


@pytest.fixture
def config() -> JointKellyConfig:
    return JointKellyConfig()


def _ml(prob: float, price: float, sign: float = 1.0) -> JointLeg:
    return JointLeg(win_prob=prob, decimal_odds=1.0 / price, axis=AXIS_MARGIN, sign=sign)


# --------------------------------------------------------------------------- #
# Reduction to fractional Kelly (the single-leg invariant)                    #
# --------------------------------------------------------------------------- #


def test_single_leg_reduces_to_fractional_kelly(config):
    """One leg per event matches plain fractional Kelly (uncapped edge)."""
    # Edge small enough that the 5% cap doesn't bind, so we test the math.
    leg = JointLeg(win_prob=0.53, decimal_odds=2.0)
    result = joint_kelly_fractions([leg], config)
    expected = compute_kelly(
        0.53, 2.0, edge_pct=0.06, base_fraction=config.kelly_multiplier,
        max_kelly=config.max_fraction,
    ).kelly_fraction
    assert result.fractions[0] == pytest.approx(expected, abs=2e-3)
    assert result.variance_ratio == pytest.approx(1.0, abs=1e-6)
    assert result.allowed_gross == pytest.approx(config.base_gross_cap)


def test_single_leg_respects_per_leg_cap(config):
    """A monster edge is clipped at max_fraction."""
    leg = JointLeg(win_prob=0.95, decimal_odds=2.0)
    result = joint_kelly_fractions([leg], config)
    assert result.fractions[0] == pytest.approx(config.max_fraction)


def test_single_leg_applies_confidence_and_liquidity(config):
    base = joint_kelly_fractions([JointLeg(win_prob=0.53, decimal_odds=2.0)], config)
    discounted = joint_kelly_fractions(
        [JointLeg(win_prob=0.53, decimal_odds=2.0, confidence=0.5, liquidity=0.8)],
        config,
    )
    assert discounted.fractions[0] == pytest.approx(base.fractions[0] * 0.5 * 0.8, abs=2e-3)


# --------------------------------------------------------------------------- #
# Hedged (contradictory) legs                                                 #
# --------------------------------------------------------------------------- #


def test_contradictory_legs_reduce_variance_and_expand_cap(config):
    """A ML + B ML hedge -> variance ratio < 1, gross cap expands above base."""
    a = _ml(0.55, 0.48, sign=1.0)
    b = _ml(0.50, 0.45, sign=-1.0)
    result = joint_kelly_fractions([a, b], config)

    assert result.variance_ratio < 1.0
    assert result.allowed_gross > config.base_gross_cap
    assert result.allowed_gross <= config.max_gross_cap + 1e-9
    assert all(f > 0 for f in result.fractions)


def test_contradictory_legs_beat_base_gross_cap(config):
    """Joint sizing of a hedge deploys more than the old fixed 8% cap allowed."""
    a = _ml(0.55, 0.48, sign=1.0)
    b = _ml(0.50, 0.45, sign=-1.0)
    result = joint_kelly_fractions([a, b], config)
    assert sum(result.fractions) > config.base_gross_cap


# --------------------------------------------------------------------------- #
# Redundant (same-direction) legs                                             #
# --------------------------------------------------------------------------- #


def test_same_direction_legs_stay_tight(config):
    """A ML + A -line reinforce -> variance ratio >= ~1, cap stays at base."""
    c = _ml(0.60, 0.50, sign=1.0)
    d = _ml(0.58, 0.52, sign=1.0)
    result = joint_kelly_fractions([c, d], config)

    assert result.variance_ratio >= 0.95
    assert result.allowed_gross == pytest.approx(config.base_gross_cap)


def test_redundant_legs_do_not_double_count(config):
    """Two near-identical legs stay within the per-event cap and don't stack."""
    single = joint_kelly_fractions([_ml(0.60, 0.50, sign=1.0)], config)
    pair = joint_kelly_fractions(
        [_ml(0.60, 0.50, sign=1.0), _ml(0.60, 0.50, sign=1.0)], config
    )
    assert sum(pair.fractions) <= config.base_gross_cap + 1e-9
    assert sum(pair.fractions) < 2.0 * single.fractions[0]


# --------------------------------------------------------------------------- #
# Cross-axis (margin vs total) legs                                           #
# --------------------------------------------------------------------------- #


def test_margin_and_total_legs_are_independent(config):
    """A margin leg and a total leg are ~uncorrelated -> no cap expansion."""
    margin = JointLeg(win_prob=0.58, decimal_odds=2.0, axis=AXIS_MARGIN, sign=1.0)
    total = JointLeg(win_prob=0.58, decimal_odds=2.0, axis=AXIS_TOTAL, sign=1.0)
    result = joint_kelly_fractions([margin, total], config)
    assert result.variance_ratio == pytest.approx(1.0, abs=0.08)
    assert result.allowed_gross == pytest.approx(config.base_gross_cap, abs=2e-3)


# --------------------------------------------------------------------------- #
# Caps, edges, determinism                                                    #
# --------------------------------------------------------------------------- #


def test_gross_and_per_leg_caps_respected(config):
    legs = [
        _ml(0.70, 0.40, sign=1.0),
        _ml(0.65, 0.42, sign=-1.0),
        JointLeg(win_prob=0.66, decimal_odds=1.0 / 0.45, axis=AXIS_TOTAL, sign=1.0),
    ]
    result = joint_kelly_fractions(legs, config)
    assert all(f <= config.max_fraction + 1e-9 for f in result.fractions)
    assert sum(result.fractions) <= result.allowed_gross + 1e-9
    assert result.allowed_gross <= config.max_gross_cap + 1e-9


def test_negative_ev_leg_gets_zero_stake(config):
    """Model prob below the implied price -> no bet."""
    leg = JointLeg(win_prob=0.40, decimal_odds=2.0)  # 50c price implies 0.50 > 0.40
    result = joint_kelly_fractions([leg], config)
    assert result.fractions[0] == 0.0


def test_negative_ev_leg_zeroed_within_mixed_event(config):
    good = _ml(0.60, 0.50, sign=1.0)
    bad = _ml(0.40, 0.50, sign=-1.0)
    result = joint_kelly_fractions([good, bad], config)
    assert result.fractions[0] > 0
    assert result.fractions[1] == 0.0


def test_invalid_legs_handled(config):
    """Zero odds / degenerate probabilities never crash and get zero stake."""
    legs = [
        JointLeg(win_prob=0.60, decimal_odds=0.0),   # no price
        JointLeg(win_prob=0.0, decimal_odds=2.0),    # impossible
        JointLeg(win_prob=1.0, decimal_odds=2.0),    # certain
    ]
    result = joint_kelly_fractions(legs, config)
    assert result.fractions == [0.0, 0.0, 0.0]


def test_empty_event(config):
    assert joint_kelly_fractions([], config).fractions == []


def test_determinism(config):
    legs = [_ml(0.55, 0.48, sign=1.0), _ml(0.50, 0.45, sign=-1.0)]
    first = joint_kelly_fractions(legs, config).fractions
    second = joint_kelly_fractions(legs, config).fractions
    assert first == second


def test_output_length_matches_input(config):
    """Inactive legs keep their slot so the caller can zip back to gaps."""
    legs = [
        JointLeg(win_prob=0.60, decimal_odds=0.0),   # inactive
        _ml(0.60, 0.50, sign=1.0),                   # active
    ]
    result = joint_kelly_fractions(legs, config)
    assert len(result.fractions) == 2
    assert result.fractions[0] == 0.0
    assert result.fractions[1] > 0


# --------------------------------------------------------------------------- #
# Axis / sign inference                                                        #
# --------------------------------------------------------------------------- #


def test_infer_moneyline_reference_team():
    assert infer_axis_and_sign("moneyline", "Lakers", "Lakers") == (AXIS_MARGIN, 1.0)


def test_infer_moneyline_opponent():
    assert infer_axis_and_sign("moneyline", "Celtics", "Lakers") == (AXIS_MARGIN, -1.0)


def test_infer_spread_uses_margin_axis():
    assert infer_axis_and_sign("spread", "Lakers", "Lakers") == (AXIS_MARGIN, 1.0)
    assert infer_axis_and_sign("spread", "Celtics", "Lakers") == (AXIS_MARGIN, -1.0)


def test_infer_total_over_under():
    # EVGap stores totals' yes_team as "over" / "under".
    assert infer_axis_and_sign("total", "over", None) == (AXIS_TOTAL, 1.0)
    assert infer_axis_and_sign("total", "under", None) == (AXIS_TOTAL, -1.0)


def test_infer_total_defaults_to_over():
    assert infer_axis_and_sign("total", "220.5", None) == (AXIS_TOTAL, 1.0)


def test_infer_player_prop_is_independent():
    axis, sign = infer_axis_and_sign("player_prop", "LeBron James", None)
    assert axis == AXIS_INDEPENDENT
    assert sign == 1.0


def test_infer_no_reference_team_defaults_positive():
    assert infer_axis_and_sign("moneyline", "Lakers", None) == (AXIS_MARGIN, 1.0)
