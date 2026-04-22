"""Tests for PlayoffAgent — playoff series detection and probability adjustments."""

from __future__ import annotations

import pytest

from evmax.agents.intelligence.playoff_agent import (
    PlayoffAgent,
    PlayoffSeries,
    _infer_round_number,
    _parse_series_summary,
)


def _make_series(
    team_a: str = "boston celtics",
    team_b: str = "indiana pacers",
    wins_a: int = 2,
    wins_b: int = 1,
    round_num: int = 1,
    round_name: str = "First Round",
    sector: str = "nba",
) -> PlayoffSeries:
    return PlayoffSeries(
        team_a=team_a,
        team_b=team_b,
        team_a_abbrev="BOS",
        team_b_abbrev="IND",
        series_wins_a=wins_a,
        series_wins_b=wins_b,
        round_num=round_num,
        round_name=round_name,
        sector=sector,
    )


class TestPlayoffSeriesProperties:
    def test_game_number(self):
        s = _make_series(wins_a=2, wins_b=1)
        assert s.game_number == 4

    def test_is_elimination_for_b(self):
        s = _make_series(wins_a=3, wins_b=1)
        assert s.is_closeout_for_a is True
        # At 3-1, team B IS facing elimination (one more loss = done)
        assert s.is_elimination_for_b is True
        assert s.is_elimination_for_a is False

    def test_is_elimination_trailing_3_2(self):
        s = _make_series(wins_a=3, wins_b=2)
        assert s.is_elimination_for_b is True
        assert s.is_closeout_for_a is True

    def test_game_7(self):
        s = _make_series(wins_a=3, wins_b=3)
        assert s.is_game_7 is True
        assert s.is_elimination_for_a is False
        assert s.is_elimination_for_b is False

    def test_not_elimination_early_series(self):
        s = _make_series(wins_a=1, wins_b=0)
        assert s.is_elimination_for_a is False
        assert s.is_elimination_for_b is False
        assert s.is_game_7 is False

    def test_is_finals(self):
        s = _make_series(round_num=4, round_name="NBA Finals")
        assert s.is_finals is True

    def test_not_finals(self):
        s = _make_series(round_num=1, round_name="First Round")
        assert s.is_finals is False

    def test_wnba_best_of_5(self):
        s = _make_series(wins_a=2, wins_b=2, sector="wnba")
        assert s.is_game_7 is True  # Game 5 in WNBA = their "Game 7"
        assert s._clinch == 3


class TestApplyAdjustments:
    def test_no_playoff_data_passthrough(self):
        a, b, notes, is_playoff = PlayoffAgent.apply_adjustments(
            playoff_data={},
            true_prob_a=0.55,
            true_prob_b=0.45,
            team_a="celtics",
            team_b="pacers",
        )
        assert a == 0.55
        assert b == 0.45
        assert notes == ""
        assert is_playoff is False

    def test_no_matching_series_passthrough(self):
        data = {"lakers_vs_nuggets": _make_series(
            team_a="los angeles lakers", team_b="denver nuggets",
        )}
        a, b, notes, is_playoff = PlayoffAgent.apply_adjustments(
            playoff_data=data,
            true_prob_a=0.55,
            true_prob_b=0.45,
            team_a="celtics",
            team_b="pacers",
        )
        assert a == 0.55
        assert is_playoff is False

    def test_playoff_hca_boost(self):
        """Home team in a normal playoff game gets +1.5% HCA bump."""
        data = {"boston celtics_vs_indiana pacers": _make_series(wins_a=1, wins_b=1)}
        a, b, notes, is_playoff = PlayoffAgent.apply_adjustments(
            playoff_data=data,
            true_prob_a=0.55,
            true_prob_b=0.45,
            team_a="boston celtics",
            team_b="indiana pacers",
        )
        assert is_playoff is True
        # Home team (boston = team_a in series, matches team_a in eval) gets boost
        assert a > 0.55
        assert "playoff_hca" in notes

    def test_elimination_boost(self):
        """Team facing elimination gets +3% boost."""
        # Boston leads 3-2, Indiana facing elimination (is away team in series)
        data = {"boston celtics_vs_indiana pacers": _make_series(wins_a=3, wins_b=2)}
        # Indiana is team_b in the eval AND team_b in the series
        a, b, notes, is_playoff = PlayoffAgent.apply_adjustments(
            playoff_data=data,
            true_prob_a=0.60,
            true_prob_b=0.40,
            team_a="boston celtics",
            team_b="indiana pacers",
        )
        assert is_playoff is True
        # Indiana (facing elimination) should get a boost, Boston (closeout) gets discount
        # Net: b should be higher than 0.40
        assert b > 0.40
        assert "elim_boost" in notes or "closeout" in notes

    def test_game_7_home_boost(self):
        """Game 7 gives massive home court advantage."""
        data = {"boston celtics_vs_indiana pacers": _make_series(wins_a=3, wins_b=3)}
        a, b, notes, is_playoff = PlayoffAgent.apply_adjustments(
            playoff_data=data,
            true_prob_a=0.55,
            true_prob_b=0.45,
            team_a="boston celtics",  # home
            team_b="indiana pacers",
        )
        assert is_playoff is True
        assert a > 0.55  # Home team gets game 7 + playoff HCA boost
        assert "game7_home" in notes

    def test_finals_bonus(self):
        """Finals games get small additional intensity adjustment."""
        data = {"boston celtics_vs_indiana pacers": _make_series(
            wins_a=2, wins_b=1, round_num=4, round_name="NBA Finals",
        )}
        a, b, notes, is_playoff = PlayoffAgent.apply_adjustments(
            playoff_data=data,
            true_prob_a=0.55,
            true_prob_b=0.45,
            team_a="boston celtics",
            team_b="indiana pacers",
        )
        assert "finals" in notes

    def test_probs_renormalize_to_1(self):
        """After adjustments, probabilities should sum to ~1.0."""
        data = {"boston celtics_vs_indiana pacers": _make_series(wins_a=3, wins_b=3)}
        a, b, _, _ = PlayoffAgent.apply_adjustments(
            playoff_data=data,
            true_prob_a=0.55,
            true_prob_b=0.45,
            team_a="boston celtics",
            team_b="indiana pacers",
        )
        assert abs(a + b - 1.0) < 0.001

    def test_probs_clamped(self):
        """Extreme probabilities should be clamped to [0.02, 0.98]."""
        data = {"boston celtics_vs_indiana pacers": _make_series(wins_a=3, wins_b=3)}
        a, b, _, _ = PlayoffAgent.apply_adjustments(
            playoff_data=data,
            true_prob_a=0.98,
            true_prob_b=0.02,
            team_a="boston celtics",
            team_b="indiana pacers",
        )
        assert a <= 0.98
        assert b >= 0.02


class TestPaceAdjustment:
    def test_nba_pace(self):
        assert PlayoffAgent.pace_adjustment("nba") == 0.95

    def test_nhl_pace(self):
        assert PlayoffAgent.pace_adjustment("nhl") == 0.97

    def test_non_playoff_sport(self):
        assert PlayoffAgent.pace_adjustment("soccer") == 1.0


class TestHelpers:
    def test_parse_series_tied(self):
        h, a = _parse_series_summary("Series tied 2-2", "BOS", "IND")
        assert h == 2
        assert a == 2

    def test_parse_series_leads(self):
        h, a = _parse_series_summary("BOS leads 3-1", "BOS", "IND")
        assert h == 3
        assert a == 1

    def test_parse_series_away_leads(self):
        h, a = _parse_series_summary("IND leads 2-1", "BOS", "IND")
        assert h == 1
        assert a == 2

    def test_parse_series_empty(self):
        h, a = _parse_series_summary("", "BOS", "IND")
        assert h == 0
        assert a == 0

    def test_infer_round_first(self):
        assert _infer_round_number("First Round - Game 3", "nba") == 1

    def test_infer_round_semis(self):
        assert _infer_round_number("Conference Semifinals - Game 5", "nba") == 2

    def test_infer_round_conf_finals(self):
        assert _infer_round_number("Conference Finals - Game 7", "nba") == 3

    def test_infer_round_finals(self):
        assert _infer_round_number("NBA Finals - Game 1", "nba") == 4

    def test_infer_round_wnba_finals(self):
        assert _infer_round_number("WNBA Finals - Game 3", "wnba") == 3


class TestPlayInTournament:
    """Tests for Play-In tournament (ESPN season type 5) handling."""

    def _make_playin(
        self,
        team_a: str = "phoenix suns",
        team_b: str = "golden state warriors",
        sector: str = "nba",
    ) -> PlayoffSeries:
        return PlayoffSeries(
            team_a=team_a,
            team_b=team_b,
            team_a_abbrev="PHX",
            team_b_abbrev="GSW",
            series_wins_a=0,
            series_wins_b=0,
            round_num=0,
            round_name="Play-In Tournament",
            sector=sector,
            is_play_in=True,
        )

    def test_playin_properties(self):
        s = self._make_playin()
        assert s.is_play_in is True
        assert s.is_playoff is True
        assert s.game_number == 1
        assert s.is_elimination_for_a is True
        assert s.is_elimination_for_b is True
        assert s.is_game_7 is False
        assert s.is_closeout_for_a is False
        assert s.is_closeout_for_b is False

    def test_playin_hca_boost_home(self):
        """Play-in home team gets +2.0% HCA boost (on top of +1.5% playoff HCA)."""
        data = {"phoenix suns_vs_golden state warriors": self._make_playin()}
        a, b, notes, is_playoff = PlayoffAgent.apply_adjustments(
            playoff_data=data,
            true_prob_a=0.55,
            true_prob_b=0.45,
            team_a="phoenix suns",
            team_b="golden state warriors",
        )
        assert is_playoff is True
        assert a > 0.55  # Home team gets both playoff HCA + play-in HCA
        assert "playin_hca" in notes
        assert "playoff_hca" in notes

    def test_playin_no_elimination_boost(self):
        """Play-in should NOT trigger one-sided elimination boost."""
        data = {"phoenix suns_vs_golden state warriors": self._make_playin()}
        a, b, notes, is_playoff = PlayoffAgent.apply_adjustments(
            playoff_data=data,
            true_prob_a=0.55,
            true_prob_b=0.45,
            team_a="phoenix suns",
            team_b="golden state warriors",
        )
        assert "elim_boost" not in notes

    def test_playin_symmetric_for_away(self):
        """When eval team_a is the away team in the series, away gets no HCA boost."""
        data = {"phoenix suns_vs_golden state warriors": self._make_playin()}
        # Swap: team_a in eval = warriors (away), team_b in eval = suns (home)
        a, b, notes, is_playoff = PlayoffAgent.apply_adjustments(
            playoff_data=data,
            true_prob_a=0.45,
            true_prob_b=0.55,
            team_a="golden state warriors",
            team_b="phoenix suns",
        )
        assert is_playoff is True
        # Suns (team_b in eval, home in series) should get the boosts
        assert b > 0.55

    def test_playin_probs_sum_to_1(self):
        data = {"phoenix suns_vs_golden state warriors": self._make_playin()}
        a, b, _, _ = PlayoffAgent.apply_adjustments(
            playoff_data=data,
            true_prob_a=0.55,
            true_prob_b=0.45,
            team_a="phoenix suns",
            team_b="golden state warriors",
        )
        assert abs(a + b - 1.0) < 0.001
