"""Regression tests for scripts/backtest_nfl_efficiency.py.

The walk-forward replays past seasons on a state whose newest season is in
the past. The NFL staleness guard used to read the wall clock, so any run
between September and February blanked nfl_efficiency for every replayed game.
`_agent_predict` now passes the game date as the market's event_date and the
agents judge staleness at that date.
"""

from __future__ import annotations

from datetime import date

import evmax.agents.models.nfl_efficiency_agent as eff_mod
from evmax.agents.models.nfl_efficiency_agent import NflEfficiencyModelAgent
from scripts.backtest_nfl_efficiency import _agent_predict


def _team(off: float, defn: float) -> dict:
    return {"off_epa_adj": off, "def_epa_adj": defn, "gp": 40}


class _Oct2026(date):
    """Wall clock pinned inside the 2026 season."""

    @classmethod
    def today(cls):
        return cls(2026, 10, 6)


def _replay_agent() -> NflEfficiencyModelAgent:
    agent = NflEfficiencyModelAgent()
    agent._state = {"nfl": {
        "teams": {
            "kansas city chiefs": _team(0.10, -0.05),
            "buffalo bills": _team(0.0, 0.0),
        },
        "seasons_used": [2024, 2025],  # replay state: newest season is 2025
    }}
    return agent


def test_replayed_game_predicts_during_a_later_season(monkeypatch):
    monkeypatch.setattr(eff_mod, "date", _Oct2026)
    p = _agent_predict(_replay_agent(), "kansas city chiefs", "buffalo bills", date(2025, 11, 2))
    assert p is not None
    assert p > 0.5  # better home team favored


def test_without_game_date_the_guard_reads_the_wall_clock(monkeypatch):
    monkeypatch.setattr(eff_mod, "date", _Oct2026)
    assert _agent_predict(_replay_agent(), "kansas city chiefs", "buffalo bills") is None


def test_week_one_game_still_blanks_on_prior_season_state(monkeypatch):
    """The first game of a season is replayed on prior-seasons-only state.
    Live blanks it, so the replay must blank it too."""
    monkeypatch.setattr(eff_mod, "date", _Oct2026)
    agent = _replay_agent()
    assert _agent_predict(agent, "kansas city chiefs", "buffalo bills", date(2026, 9, 10)) is None


def test_parse_epa_margin_pts_single_value_applies_to_every_season():
    from scripts.backtest_nfl_efficiency import _parse_epa_margin_pts
    assert _parse_epa_margin_pts("38", ["2324", "2425"]) == {"2324": 38.0, "2425": 38.0}


def test_parse_epa_margin_pts_per_season_and_unset():
    from scripts.backtest_nfl_efficiency import _parse_epa_margin_pts
    assert _parse_epa_margin_pts("2324=38.8,2526=37.5", ["2324", "2425", "2526"]) == {
        "2324": 38.8, "2526": 37.5,
    }
    assert _parse_epa_margin_pts(None, ["2324"]) == {}


def test_state_cutoff_excludes_a_primetime_games_own_plays():
    """ESPN dates the Thu 8:20 pm ET opener 2023-09-08 (UTC); its PBP rows are
    dated 2023-09-07 (ET). The state that prices it must not contain them."""
    import polars as pl
    from scripts.backtest_nfl_efficiency import _state_cutoff

    pbp = pl.DataFrame({
        "game_date": [date(2023, 2, 12), date(2023, 9, 7)],  # prior Super Bowl, the opener itself
        "season": [2022, 2023],
    })
    kept = pbp.filter(pl.col("game_date") < pl.lit(_state_cutoff(date(2023, 9, 8))))
    assert kept["season"].to_list() == [2022]
