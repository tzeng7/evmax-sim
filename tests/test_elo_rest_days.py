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


# ---------------------------------------------------------------------------
# Rest-layer read key (2026-09-23, docs/elo-h2h-rest-eval.md). At predict time
# the team is the lowercased Pinnacle label; form_state is keyed by the Form
# agent's resolved key. REST_RESOLVED_SECTORS resolve the label first (rest
# switched ON — walk-forward validated for nba/ncaab); every other sector keeps
# the raw-label lookup. H2H stays raw everywhere (walk-forward rejected).
# ---------------------------------------------------------------------------

from evmax.agents.models.elo_agent import REST_RESOLVED_SECTORS  # noqa: E402


def _rec(d: str) -> dict:
    return {"date": d, "won": True, "opp": "x", "home": True}


@pytest.fixture
def form_file(tmp_path, monkeypatch):
    def _write(state: dict):
        path = tmp_path / "form_state.json"
        path.write_text(json.dumps(state))
        monkeypatch.setattr(elo_agent, "FORM_STATE_PATH", path)
        return path
    return _write


def test_resolved_sectors_all_carry_a_rest_table():
    # A resolved sector with no REST_ELO_ADJ entry would change nothing — the
    # set must only name sectors whose rest layer actually fires.
    assert REST_RESOLVED_SECTORS == {"nba", "ncaab"}
    assert REST_RESOLVED_SECTORS <= set(REST_ELO_ADJ)


def test_nba_rest_reads_the_resolved_key_for_a_pinnacle_label(form_file):
    form_file({"nba": {"lakers": [_rec("2026-11-10")], "celtics": [_rec("2026-11-07")]}})
    a = EloModelAgent(); a._state = {}
    # Before: form.get("los angeles lakers") → nothing → rest silently 0.
    assert a._days_of_rest("nba", "los angeles lakers", date(2026, 11, 11)) == 1
    assert a._days_of_rest("nba", "boston celtics", date(2026, 11, 11)) == 4
    assert a._rest_elo_bonus("nba", "los angeles lakers", date(2026, 11, 13)) == REST_ELO_ADJ["nba"][3]
    assert a._days_of_rest("nba", "no such team", date(2026, 11, 11)) is None


def test_nba_rest_now_moves_the_pinnacle_label_prediction(form_file):
    form_file({"nba": {"lakers": [_rec("2026-11-10")], "celtics": [_rec("2026-11-07")]}})
    a = EloModelAgent(); a._state = {}
    a._sector_state("nba")["ratings"] = {"lakers": 1500.0, "celtics": 1500.0}
    # Lakers on 1 day (table key 1 → 0), Celtics on 4 days (→ table[3]): the
    # Pinnacle-label prediction must match the key-label one and differ from
    # a date where both sides are past the rest horizon.
    p_label, _, _ = a._win_probs("nba", "los angeles lakers", "boston celtics", date(2026, 11, 11))
    p_key, _, _ = a._win_probs("nba", "lakers", "celtics", date(2026, 11, 11))
    p_far, _, _ = a._win_probs("nba", "los angeles lakers", "boston celtics", date(2026, 11, 30))
    assert p_label == p_key
    assert p_label < p_far


def test_ncaab_rest_resolves_a_mascot_decorated_key(form_file):
    form_file({"ncaab": {"furman paladins": [_rec("2026-01-10")]}})
    a = EloModelAgent(); a._state = {}
    assert a._days_of_rest("ncaab", "furman", date(2026, 1, 12)) == 2


def test_ncaab_rest_never_borrows_another_schools_history(form_file):
    # "george washington" must not read "washington"'s games (college head-word guard).
    form_file({"ncaab": {"washington": [_rec("2026-01-10")]}})
    a = EloModelAgent(); a._state = {}
    assert a._days_of_rest("ncaab", "george washington", date(2026, 1, 12)) is None


@pytest.mark.parametrize("sector,label,key", [
    ("wnba", "las vegas aces", "aces"),              # rest walk-forward rejected → stays dead
    ("soccer", "manchester city", "man city"),       # rejected → keeps its partial (raw) firing
])
def test_unvalidated_sectors_keep_the_raw_label_lookup(form_file, sector, label, key):
    form_file({sector: {key: [_rec("2026-08-10"), _rec("2026-08-08")]}})
    a = EloModelAgent(); a._state = {}
    assert sector not in REST_RESOLVED_SECTORS
    assert a._days_of_rest(sector, label, date(2026, 8, 11)) is None
    assert a._games_in_last_n_days(sector, label, 7, date(2026, 8, 11)) == 0
    # the exact key still reads (current behaviour, byte-identical)
    assert a._days_of_rest(sector, key, date(2026, 8, 11)) == 1
    assert a._games_in_last_n_days(sector, key, 7, date(2026, 8, 11)) == 2


def test_h2h_stays_on_the_raw_label_read_by_design():
    # H2H read through resolve_team_key was walk-forward REJECTED in every
    # sector tested (nba +1.46 Brier/1000 z +4.8, baseball +1.00 z +5.8, ...).
    # A Pinnacle label must NOT pick up the canonical-keyed record.
    a = EloModelAgent(); a._state = {}
    for _ in range(4):
        a.update("lakers", "celtics", 110, 100, "nba", event_date="2026-01-01")
    assert a._h2h_adjustment("nba", "lakers", "celtics") > 0
    assert a._h2h_adjustment("nba", "los angeles lakers", "boston celtics") == 0.0
