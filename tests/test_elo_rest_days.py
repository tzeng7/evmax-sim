"""EloModelAgent rest layer — kickoff-measured reference date + the NFL verdict.

Two things shipped 2026-09-03:
  1. `_days_of_rest` / `_rest_elo_bonus` / `_win_probs` take the GAME date.
     Before, rest was measured to today, so a Sunday game scanned on
     Wednesday looked like a 3-day turnaround for both teams.
  2. NFL has NO rest entry. Its old table {0:-30,1:0,2:10,3:10} was dead
     (any 4–7 day gap → +10, a bye → 0) and a proper kickoff-keyed table was
     walk-forward REJECTED (within noise, slightly worse on the 2025 holdout,
     see the REST_ELO_ADJ comment). `rest_adjustment` is the pure lookup.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from evmax.agents.models import elo_agent
from evmax.agents.models.elo_agent import REST_ELO_ADJ, EloModelAgent


def test_nfl_has_no_rest_adjustment():
    assert "nfl" not in REST_ELO_ADJ
    for days in (3, 4, 7, 14, 200, None):
        assert EloModelAgent.rest_adjustment("nfl", days) == 0.0


def test_legacy_tables_keep_their_old_semantics():
    nba = REST_ELO_ADJ["nba"]
    assert EloModelAgent.rest_adjustment("nba", 0) == nba[0]
    assert EloModelAgent.rest_adjustment("nba", 1) == nba[1]
    for d in (4, 5, 6, 7):                       # old lookup: table[min(days, 3)]
        assert EloModelAgent.rest_adjustment("nba", d) == nba[3]
    assert EloModelAgent.rest_adjustment("nba", 8) == 0.0   # old: > 7 days → 0
    assert EloModelAgent.rest_adjustment("no-such-sector", 3) == 0.0


def test_step_function_honours_explicit_long_rest_keys(monkeypatch):
    monkeypatch.setitem(REST_ELO_ADJ, "toy", {3: -20.0, 5: 0.0, 8: 10.0, 13: 20.0})
    # horizon = max(7, largest key): 13 is the last keyed day, 14+ carries no signal
    assert [EloModelAgent.rest_adjustment("toy", d) for d in (2, 3, 4, 5, 7, 8, 12, 13, 14, 15)] == \
        [-20.0, -20.0, -20.0, 0.0, 0.0, 10.0, 10.0, 20.0, 0.0, 0.0]


def test_days_of_rest_measured_to_the_game_date(tmp_path, monkeypatch):
    form = tmp_path / "form_state.json"
    form.write_text(json.dumps({"nba": {"lakers": [{"date": "2026-11-10", "won": True, "opp": "x", "home": True}]}}))
    monkeypatch.setattr(elo_agent, "FORM_STATE_PATH", form)
    a = EloModelAgent(); a._state = {}
    assert a._days_of_rest("nba", "lakers", date(2026, 11, 11)) == 1
    assert a._days_of_rest("nba", "lakers", date(2026, 11, 13)) == 3
    assert a._days_of_rest("nba", "nobody", date(2026, 11, 13)) is None
    # bonus follows the reference, not the wall clock
    assert a._rest_elo_bonus("nba", "lakers", date(2026, 11, 10)) == REST_ELO_ADJ["nba"][0]   # back-to-back
    assert a._rest_elo_bonus("nba", "lakers", date(2026, 11, 13)) == REST_ELO_ADJ["nba"][3]


def test_win_probs_threads_reference_into_rest(tmp_path, monkeypatch):
    form = tmp_path / "form_state.json"
    form.write_text(json.dumps({"nba": {
        "lakers": [{"date": "2026-11-10", "won": True, "opp": "x", "home": True}],
        "celtics": [{"date": "2026-11-07", "won": True, "opp": "y", "home": True}],
    }}))
    monkeypatch.setattr(elo_agent, "FORM_STATE_PATH", form)
    a = EloModelAgent(); a._state = {}
    a._sector_state("nba")["ratings"] = {"lakers": 1500.0, "celtics": 1500.0}
    nba = REST_ELO_ADJ["nba"]
    assert nba[0] < nba[3]
    # 11-11: lakers on a back-to-back (0 days), celtics on 4 days → lakers penalised relative to a rested date
    p_b2b, _, _ = a._win_probs("nba", "lakers", "celtics", date(2026, 11, 11))
    p_rested, _, _ = a._win_probs("nba", "lakers", "celtics", date(2026, 11, 20))  # both beyond horizon → 0
    assert p_b2b < p_rested
