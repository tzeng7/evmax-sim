"""Tests for evmax.clients.nfl_props_cache — pure compute functions.

Covers:
  - compute_nfl_prop_prob dispatches correctly by stat_type
  - Yardage branch: monotone in threshold, handles streak/margin paths
  - Poisson branch: anytime TD lambda → P(X≥1), passing TDs over 1.5/2.5
  - compute_opponent_adjustment clips to [1-MAX, 1+MAX]
  - Thin sample returns None
"""

from __future__ import annotations

import numpy as np
import pytest

from evmax.clients.nfl_props_cache import (
    MAX_OPP_ADJ,
    MIN_GAMES,
    compute_nfl_prop_prob,
    compute_opponent_adjustment,
)


# -------------------------------------------------------------------------
# Yardage branch — passing_yards is the canonical test case
# -------------------------------------------------------------------------


def test_yardage_returns_reasonable_prob_for_average_player():
    hist = [240, 260, 220, 280, 250, 300, 210, 270]  # mean ~253
    prob, n = compute_nfl_prop_prob(
        stat_type="passing_yards",
        threshold=249.5,
        values=hist,
    )
    assert n == 8
    # Mean is slightly above threshold → prob should be > 0.5
    assert 0.5 <= prob <= 0.8


def test_yardage_prob_monotone_decreasing_in_threshold():
    hist = [240, 260, 220, 280, 250, 300, 210, 270]
    p_low, _ = compute_nfl_prop_prob("passing_yards", 199.5, hist)
    p_mid, _ = compute_nfl_prop_prob("passing_yards", 249.5, hist)
    p_high, _ = compute_nfl_prop_prob("passing_yards", 349.5, hist)
    assert p_low > p_mid > p_high


def test_yardage_opponent_adjustment_scales_prob():
    hist = [240, 260, 220, 280, 250, 300, 210, 270]
    base, _ = compute_nfl_prop_prob("passing_yards", 249.5, hist, opp_adj=1.0)
    soft, _ = compute_nfl_prop_prob("passing_yards", 249.5, hist, opp_adj=1.15)
    tough, _ = compute_nfl_prop_prob("passing_yards", 249.5, hist, opp_adj=0.85)
    assert soft > base > tough


def test_yardage_thin_sample_returns_none():
    hist = [240, 260, 220]  # fewer than MIN_GAMES
    assert compute_nfl_prop_prob("passing_yards", 249.5, hist) is None


def test_yardage_recent_hot_games_still_boost_prob():
    # Streak adjustment was removed (MODEL-8 step 2), but decay-weighted
    # mean still gives more weight to recent games, so a player with
    # recent big games should still be favored vs a cold baseline.
    hot = [320, 330, 310, 200, 210, 220, 180, 190]
    cold = [180, 190, 200, 200, 210, 220, 180, 190]
    hot_prob, _ = compute_nfl_prop_prob("passing_yards", 249.5, hot)
    cold_prob, _ = compute_nfl_prop_prob("passing_yards", 249.5, cold)
    assert hot_prob > cold_prob


def test_rushing_yards_uses_appropriate_std_floor():
    # A RB with 60/80/40/90/50/70/65/55 yards vs line of 59.5
    hist = [60, 80, 40, 90, 50, 70, 65, 55]
    prob, n = compute_nfl_prop_prob("rushing_yards", 59.5, hist)
    assert n == 8
    # Mean = 63.75, std ~ 15.4 → should land 55-70%
    assert 0.45 <= prob <= 0.75


def test_receiving_yards_branch():
    hist = [45, 32, 80, 28, 65, 40, 55, 38]
    prob, _ = compute_nfl_prop_prob("receiving_yards", 49.5, hist)
    assert 0.3 <= prob <= 0.7


# -------------------------------------------------------------------------
# Count branch — receptions
# -------------------------------------------------------------------------


def test_receptions_branch():
    hist = [4, 6, 3, 5, 7, 4, 5, 6]  # mean 5.0
    prob_under, _ = compute_nfl_prop_prob("receptions", 3.5, hist)
    prob_over, _ = compute_nfl_prop_prob("receptions", 6.5, hist)
    assert prob_under > prob_over
    assert prob_under > 0.5


# -------------------------------------------------------------------------
# Poisson branch — touchdowns
# -------------------------------------------------------------------------


def test_anytime_td_poisson_prob():
    # A RB with 0.8 TDs/game on average
    hist = [1, 1, 0, 2, 1, 0, 1, 0]
    prob, n = compute_nfl_prop_prob("anytime_td", 0.5, hist)
    assert n == 8
    # Expected λ ~ 0.75; P(X≥1) = 1 - e^-0.75 ≈ 0.53
    assert 0.40 <= prob <= 0.65


def test_passing_tds_poisson_threshold_step():
    # A QB who averages 2 passing TDs per game
    hist = [2, 3, 1, 2, 2, 3, 1, 2]
    p_over_15, _ = compute_nfl_prop_prob("passing_tds", 1.5, hist)
    p_over_25, _ = compute_nfl_prop_prob("passing_tds", 2.5, hist)
    p_over_35, _ = compute_nfl_prop_prob("passing_tds", 3.5, hist)
    assert p_over_15 > p_over_25 > p_over_35
    # λ ≈ 2, so P(X≥2) = 1 - poisson.cdf(1, 2) ≈ 0.594
    assert 0.50 <= p_over_15 <= 0.70


def test_poisson_opponent_adj_scales_lambda():
    hist = [1, 1, 0, 2, 1, 0, 1, 0]
    base, _ = compute_nfl_prop_prob("anytime_td", 0.5, hist, opp_adj=1.0)
    boost, _ = compute_nfl_prop_prob("anytime_td", 0.5, hist, opp_adj=1.15)
    assert boost > base


# -------------------------------------------------------------------------
# Opponent adjustment
# -------------------------------------------------------------------------


def test_opp_adjustment_clips_extremes():
    # Opponent allows 2x league average → should clip to 1 + MAX
    adj = compute_opponent_adjustment("passing_yards", 500.0, 250.0)
    assert adj == pytest.approx(1.0 + MAX_OPP_ADJ)


def test_opp_adjustment_clips_on_stingy_defense():
    adj = compute_opponent_adjustment("passing_yards", 100.0, 250.0)
    assert adj == pytest.approx(1.0 - MAX_OPP_ADJ)


def test_opp_adjustment_neutral_when_at_average():
    adj = compute_opponent_adjustment("passing_yards", 250.0, 250.0)
    assert adj == pytest.approx(1.0)


def test_opp_adjustment_zero_league_avg_is_neutral():
    adj = compute_opponent_adjustment("passing_yards", 200.0, 0.0)
    assert adj == 1.0


# -------------------------------------------------------------------------
# Unknown stat
# -------------------------------------------------------------------------


def test_unknown_stat_returns_none():
    assert compute_nfl_prop_prob("fumbles_recovered", 0.5, [0, 1, 0, 0, 1, 0, 1, 0]) is None


def test_prob_is_bounded():
    """Any valid call must return a probability in [0.01, 0.99]."""
    hist = [500, 600, 550, 580, 570, 590, 560, 540]  # QB destroying threshold
    prob, _ = compute_nfl_prop_prob("passing_yards", 100.5, hist)
    assert 0.01 <= prob <= 0.99
    prob, _ = compute_nfl_prop_prob("passing_yards", 800.5, hist)
    assert 0.01 <= prob <= 0.99
