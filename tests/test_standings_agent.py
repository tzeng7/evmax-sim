"""Tests for StandingsAgent rest-risk classification."""

from __future__ import annotations

from evmax.agents.intelligence.standings_agent import TeamStanding


def _team(
    *,
    games_remaining: int,
    clinch_code: str = "",
    is_b2b: bool = False,
) -> TeamStanding:
    return TeamStanding(
        team_name="test",
        abbreviation="TST",
        wins=0,
        losses=0,
        clinch_code=clinch_code,
        clinch_desc="",
        playoff_seed=1,
        games_remaining=games_remaining,
        is_back_to_back=is_b2b,
    )


def test_rest_risk_clinched_with_games_left_is_high():
    s = _team(games_remaining=2, clinch_code="x")
    assert s.rest_risk == "high"
    assert s.probability_adjustment == -0.05


def test_rest_risk_clinched_mid_zone_is_medium():
    s = _team(games_remaining=5, clinch_code="z")
    assert s.rest_risk == "medium"
    assert s.probability_adjustment == -0.02


def test_rest_risk_eliminated_with_games_left_is_high():
    s = _team(games_remaining=1, clinch_code="e")
    assert s.rest_risk == "high"


def test_rest_risk_suppressed_in_postseason():
    """Once regular season is over (games_remaining == 0), rest risk must not fire.

    Regression: ESPN standings still report clinch codes during the playoffs, which
    previously caused every clinched team to be flagged 'high' rest risk for every
    NBA scan after the regular season ended.
    """
    clinched = _team(games_remaining=0, clinch_code="x")
    assert clinched.rest_risk == "none"
    assert clinched.probability_adjustment == 0.0

    eliminated = _team(games_remaining=0, clinch_code="e")
    assert eliminated.rest_risk == "none"
    assert eliminated.probability_adjustment == 0.0


def test_rest_risk_no_clinch_no_b2b_is_none():
    s = _team(games_remaining=10, clinch_code="")
    assert s.rest_risk == "none"
