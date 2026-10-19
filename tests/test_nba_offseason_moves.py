"""scripts/nba_offseason_moves.py — additive, idempotent NBA roster-move Elo deltas."""

from __future__ import annotations

import json

import pytest

from scripts.nba_offseason_moves import (
    AlreadyApplied,
    apply_moves,
    load_moves,
    main,
    sum_deltas,
)


def _state():
    return {"nba": {"ratings": {"heat": 1500.0, "bucks": 1550.0}, "season_games": {"heat": 7}}}


def test_sum_deltas_merges_per_team_and_normalizes_case():
    moves = [{"team": "Heat", "delta": 80}, {"team": "heat ", "delta": -10}]
    assert sum_deltas(moves) == {"heat": 70.0}


def test_apply_is_additive_and_leaves_season_games_alone():
    s = _state()
    apply_moves(s, [{"team": "heat", "delta": 80}], "d1")
    assert s["nba"]["ratings"] == {"heat": 1580.0, "bucks": 1550.0}
    assert s["nba"]["season_games"] == {"heat": 7}
    assert s["nba"]["offseason_moves"]["digest"] == "d1"


def test_repeat_digest_refused_unless_forced():
    s = _state()
    apply_moves(s, [{"team": "heat", "delta": 10}], "d1")
    with pytest.raises(AlreadyApplied):
        apply_moves(s, [{"team": "heat", "delta": 10}], "d1")
    apply_moves(s, [{"team": "heat", "delta": 10}], "d1", force=True)
    assert s["nba"]["ratings"]["heat"] == 1520.0


def test_unknown_team_raises_and_mutates_nothing():
    s = _state()
    with pytest.raises(KeyError):
        apply_moves(s, [{"team": "heat", "delta": 5}, {"team": "nope", "delta": 5}], "d1")
    assert s["nba"]["ratings"]["heat"] == 1500.0


def test_shipped_yaml_loads_and_every_team_exists_in_live_state():
    moves, _ = load_moves(__import__("scripts.nba_offseason_moves", fromlist=["MOVES_YAML"]).MOVES_YAML)
    state = json.load(open(__import__("scripts.nba_offseason_moves", fromlist=["x"]).ELO_STATE_PATH))
    assert set(sum_deltas(moves)) <= set(state["nba"]["ratings"])


def test_cli_dry_run_writes_nothing(tmp_path):
    y = tmp_path / "m.yaml"
    y.write_text("moves:\n  - {team: heat, delta: 40, reason: x}\n")
    st = tmp_path / "elo.json"
    st.write_text(json.dumps(_state()))
    before = st.read_text()
    assert main(["--dry-run", "--yaml", str(y), "--state", str(st)]) == 0
    assert st.read_text() == before


def test_cli_apply_then_repeat_fails(tmp_path):
    y = tmp_path / "m.yaml"
    y.write_text("moves:\n  - {team: heat, delta: 40, reason: x}\n")
    st = tmp_path / "elo.json"
    st.write_text(json.dumps(_state()))
    assert main(["--yaml", str(y), "--state", str(st)]) == 0
    assert json.loads(st.read_text())["nba"]["ratings"]["heat"] == 1540.0
    assert list(tmp_path.glob("elo.backup.nba_moves_*.json"))
    assert main(["--yaml", str(y), "--state", str(st)]) == 1
