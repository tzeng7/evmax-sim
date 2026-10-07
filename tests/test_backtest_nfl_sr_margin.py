"""Tests for the margin-scale fit helpers in scripts/backtest_nfl_sr_margin.py."""

from __future__ import annotations

import pytest

from evmax.agents.models.nfl_efficiency_agent import HOME_EDGE_PTS
from scripts.backtest_nfl_sr_margin import _paired, fit_epa_only, fit_epa_scale_only


def _rows(epa_pts: float, home: float) -> list[dict]:
    """Noise-free games whose final margin is exactly epa_pts·Δepa + home."""
    out = []
    for d in (-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20):
        margin = epa_pts * d + home
        out.append({"d_epa": d, "d_sr": 0.0, "margin": margin, "home_won": 1 if margin > 0 else 0})
    return out


def test_scale_only_fit_recovers_the_slope_with_home_held():
    assert fit_epa_scale_only(_rows(36.0, HOME_EDGE_PTS)) == pytest.approx(36.0)


def test_epa_only_fit_recovers_home_and_slope():
    home, epa = fit_epa_only(_rows(40.0, 2.5))
    assert home == pytest.approx(2.5)
    assert epa == pytest.approx(40.0)


def test_paired_is_zero_for_identical_margins():
    rows = _rows(36.0, HOME_EDGE_PTS)
    b0, b1, delta, z = _paired(rows, (36.0, 0.0, HOME_EDGE_PTS), (36.0, 0.0, HOME_EDGE_PTS))
    assert b0 == b1
    assert delta == 0.0
    assert z == 0.0


def test_paired_sign_favors_the_better_candidate():
    # Outcomes are coin flips around a small margin: a narrower scale is better calibrated.
    rows = [
        {"d_epa": d, "d_sr": 0.0, "margin": 0.0, "home_won": w}
        for d in (-0.2, -0.1, 0.1, 0.2) for w in (0, 1)
    ]
    _, _, delta, _ = _paired(rows, (64.0, 0.0, 0.0), (20.0, 0.0, 0.0))
    assert delta > 0  # + = candidate better
