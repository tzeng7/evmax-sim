"""Games-played ramp for the NFL efficiency seed (scripts/seed_nfl_efficiency.py).

The ramp is INERT by default (RAMP_K = None): live seeding keeps the per-play
season-decay pooling. These tests pin the mixing math the backtest gates on.
"""

from __future__ import annotations

import pytest

import scripts.seed_nfl_efficiency as seed
from scripts.seed_nfl_efficiency import mix_team_stats, ramp_weight


def _row(off: float, defn: float, gp: int) -> dict:
    return {"off_epa_adj": off, "def_epa_adj": defn, "off_success_adj": 0.0,
            "def_success_adj": 0.0, "gp": gp}


def test_ramp_is_inert_by_default():
    assert seed.RAMP_K is None


@pytest.mark.parametrize("gp,k,expected", [
    (0, 4.0, 0.0),      # no current games -> prior only
    (4, 4.0, 0.5),      # gp == k -> half and half
    (12, 4.0, 0.75),
    (3, 0.0, 1.0),      # k = 0 -> current season only
    (0, 0.0, 0.0),      # ...but never with zero current games
])
def test_ramp_weight(gp, k, expected):
    assert ramp_weight(gp, k) == pytest.approx(expected)


def test_mix_weights_current_by_games_played():
    cur = {"a": _row(0.20, -0.10, 4)}
    pri = {"a": _row(0.00, 0.10, 51)}
    out = mix_team_stats(cur, pri, {"a": 4}, {"a": 55}, ramp_k=4.0)
    assert out["a"]["off_epa_adj"] == pytest.approx(0.10)
    assert out["a"]["def_epa_adj"] == pytest.approx(0.0)
    assert out["a"]["ramp_w"] == pytest.approx(0.5)


def test_gp_stays_the_pooled_count():
    """Confidence / MIN_GAMES in the agent read gp; the ramp must not move it."""
    cur = {"a": _row(0.2, 0.0, 2)}
    pri = {"a": _row(0.0, 0.0, 51)}
    out = mix_team_stats(cur, pri, {"a": 2}, {"a": 53}, ramp_k=4.0)
    assert out["a"]["gp"] == 53


def test_team_without_current_games_uses_prior():
    pri = {"a": _row(0.05, -0.02, 34)}
    out = mix_team_stats({}, pri, {}, {"a": 34}, ramp_k=4.0)
    assert out["a"]["off_epa_adj"] == 0.05
    assert out["a"]["def_epa_adj"] == -0.02


def test_team_without_prior_uses_current_at_full_weight():
    cur = {"a": _row(0.30, 0.10, 3)}
    out = mix_team_stats(cur, {}, {"a": 3}, {"a": 3}, ramp_k=4.0)
    assert out["a"]["off_epa_adj"] == 0.30


def test_field_missing_from_current_falls_back_to_prior():
    cur = {"a": _row(0.1, 0.0, 5)}
    pri = {"a": {**_row(0.0, 0.0, 40), "ridge_net": 0.07}}
    out = mix_team_stats(cur, pri, {"a": 5}, {"a": 45}, ramp_k=4.0)
    assert out["a"]["ridge_net"] == 0.07
