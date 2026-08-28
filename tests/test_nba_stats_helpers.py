"""Unit tests for the pure helpers in evmax/clients/nba_stats.py.

nba_stats.py is a LIVE path (NBA prop resolution / cache) that had no direct
coverage. These cover the pure, network-free helpers: league-average rollup,
opponent lookup, and MATCHUP-string team parsing.
"""
from __future__ import annotations

import pandas as pd

from evmax.clients.nba_stats import (
    _find_opponent,
    _league_averages,
    _player_team_abbrev,
)


class TestLeagueAverages:
    def test_empty_returns_defaults(self):
        avgs = _league_averages({})
        assert avgs["def_rating"] == 112.0
        assert avgs["pace"] == 98.5

    def test_averages_across_teams(self):
        stats = {
            "A": {"def_rating": 110.0, "pace": 100.0, "opp_pts": 110.0},
            "B": {"def_rating": 114.0, "pace": 96.0, "opp_pts": 114.0},
        }
        avgs = _league_averages(stats)
        assert avgs["def_rating"] == 112.0
        assert avgs["pace"] == 98.0
        assert avgs["opp_pts"] == 112.0


class TestFindOpponent:
    _MATCHUPS = [
        {"home_abbrev": "NYK", "away_abbrev": "MIL"},
        {"home_abbrev": "LAL", "away_abbrev": "BOS"},
    ]

    def test_home_side(self):
        assert _find_opponent("nyk", self._MATCHUPS) == "MIL"

    def test_away_side(self):
        assert _find_opponent("MIL", self._MATCHUPS) == "NYK"

    def test_not_playing_returns_none(self):
        assert _find_opponent("GSW", self._MATCHUPS) is None


class TestPlayerTeamAbbrev:
    def test_home_matchup(self):
        df = pd.DataFrame([{"MATCHUP": "NYK vs. MIL"}])
        assert _player_team_abbrev(df) == "NYK"

    def test_away_matchup_uppercased(self):
        df = pd.DataFrame([{"MATCHUP": "nyk @ lal"}])
        assert _player_team_abbrev(df) == "NYK"

    def test_empty_df_is_none(self):
        assert _player_team_abbrev(pd.DataFrame()) is None

    def test_missing_column_is_none(self):
        assert _player_team_abbrev(pd.DataFrame([{"X": 1}])) is None

    def test_none_df_is_none(self):
        assert _player_team_abbrev(None) is None
