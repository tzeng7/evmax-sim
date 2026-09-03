"""scripts/offseason_regress.py — generalized offseason Elo regression.

Pure-function coverage (shrink / apply_regression / prune_form_state) plus a
dry-run CLI check that nothing is written. The NFL keep=0.667 default is a
walk-forward result (scripts/backtest_nfl_elo_regression.py, 2026-09-02) and
is pinned here so a change forces a documented re-sweep.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from scripts.offseason_regress import (
    DEFAULT_ELO,
    SECTOR_DEFAULT_KEEP,
    apply_regression,
    main,
    prune_form_state,
    shrink,
)


def _state():
    return {
        "nfl": {
            "ratings": {"seahawks": 1650.0, "raiders": 1390.0, "chiefs": 1500.0, "afc": 1510.0, "nfc": 1490.0},
            "game_counts": {"seahawks": 20, "raiders": 17, "chiefs": 17, "afc": 1, "nfc": 1},
            "h2h": {"seahawks::raiders": {"a_wins": 1, "b_wins": 0, "games": 1},
                    "afc::nfc": {"a_wins": 1, "b_wins": 0, "games": 1}},
        },
        "wnba": {"ratings": {"aces": 1600.0}, "game_counts": {"aces": 40}, "season_games": {"aces": 40}, "h2h": {}},
    }


def test_nfl_default_keep_is_the_swept_value():
    assert SECTOR_DEFAULT_KEEP["nfl"] == pytest.approx(0.667)


@pytest.mark.parametrize("elo, keep, expected", [
    (1650.0, 0.667, 1500 + 0.667 * 150),
    (1390.0, 0.667, 1500 - 0.667 * 110),
    (1500.0, 0.5, 1500.0),
    (1700.0, 1.0, 1700.0),
])
def test_shrink_moves_toward_mean(elo, keep, expected):
    assert shrink(elo, keep) == pytest.approx(expected)


def test_apply_regression_shrinks_resets_and_drops():
    state = _state()
    summary = apply_regression(state, "nfl", 0.667, drop=["afc", "nfc"], today=date(2026, 9, 3))
    sec = state["nfl"]
    assert set(sec["ratings"]) == {"seahawks", "raiders", "chiefs"}
    assert sec["ratings"]["seahawks"] == pytest.approx(round(1500 + 0.667 * 150, 2))
    assert sec["ratings"]["raiders"] == pytest.approx(round(1500 - 0.667 * 110, 2))
    assert sec["ratings"]["chiefs"] == 1500.0
    # season_games reset for every rated team; lifetime counts untouched
    assert sec["season_games"] == {"seahawks": 0, "raiders": 0, "chiefs": 0}
    assert sec["game_counts"] == {"seahawks": 20, "raiders": 17, "chiefs": 17}
    # h2h rows referencing a dropped key go too
    assert set(sec["h2h"]) == {"seahawks::raiders"}
    assert sec["offseason_regression"] == {"applied_on": "2026-09-03", "keep": 0.667, "dropped": ["afc", "nfc"], "moves": 0}
    assert summary["dropped"] == ["afc", "nfc"]
    assert summary["before"]["seahawks"] == 1650.0
    # other sectors untouched
    assert state["wnba"]["ratings"] == {"aces": 1600.0}


def test_apply_regression_keep_one_is_identity_on_ratings():
    state = _state()
    apply_regression(state, "nfl", 1.0)
    assert state["nfl"]["ratings"]["seahawks"] == 1650.0
    assert state["nfl"]["season_games"]["seahawks"] == 0


def test_apply_regression_moves_and_expansion():
    state = _state()
    moves = [{"team": "Chiefs", "delta": 25}, {"team": "chiefs", "delta": -5}, {"team": "raiders", "delta": 10}]
    apply_regression(state, "nfl", 0.5, moves=moves, expansion={"NewTeam": 1450})
    r = state["nfl"]["ratings"]
    assert r["chiefs"] == pytest.approx(1520.0)           # 1500 + 20
    assert r["raiders"] == pytest.approx(1445.0 + 10.0)   # shrink then delta
    assert r["newteam"] == 1450.0                          # expansion prior, not shrunk
    assert state["nfl"]["season_games"]["newteam"] == 0
    assert state["nfl"]["offseason_regression"]["moves"] == 3


@pytest.mark.parametrize("keep", [0.0, -0.1, 1.5])
def test_apply_regression_rejects_bad_keep(keep):
    with pytest.raises(ValueError):
        apply_regression(_state(), "nfl", keep)


def test_prune_form_state_removes_only_listed_keys():
    form = {"nfl": {"eagles": [{"date": "2026-02-08"}], "afc": [{"date": "2026-02-01"}], "nfc": []}, "wnba": {"aces": []}}
    assert prune_form_state(form, "nfl", ["afc", "nfc", "missing"]) == ["afc", "nfc"]
    assert set(form["nfl"]) == {"eagles"}
    assert prune_form_state(form, "nhl", ["x"]) == []


def test_cli_dry_run_writes_nothing_and_apply_writes_with_backup(tmp_path):
    elo = tmp_path / "elo_state.json"
    form = tmp_path / "form_state.json"
    elo.write_text(json.dumps(_state()))
    form.write_text(json.dumps({"nfl": {"eagles": [], "afc": [], "nfc": []}}))
    before_elo, before_form = elo.read_text(), form.read_text()

    assert main(["--sector", "nfl", "--drop", "afc,nfc", "--dry-run",
                 "--state", str(elo), "--form-state", str(form)]) == 0
    assert elo.read_text() == before_elo and form.read_text() == before_form
    assert not list(tmp_path.glob("*.backup.*"))

    assert main(["--sector", "nfl", "--drop", "afc,nfc",
                 "--state", str(elo), "--form-state", str(form)]) == 0
    new = json.loads(elo.read_text())["nfl"]
    assert "afc" not in new["ratings"] and new["ratings"]["seahawks"] == pytest.approx(round(1500 + 0.667 * 150, 2))
    assert set(json.loads(form.read_text())["nfl"]) == {"eagles"}
    backups = sorted(p.name for p in tmp_path.glob("*.backup.nfl_offseason_*.json"))
    assert len(backups) == 2 and backups[0].startswith("elo_state") and backups[1].startswith("form_state")


def test_cli_requires_swept_keep_for_unknown_sector(tmp_path, capsys):
    elo = tmp_path / "elo_state.json"
    elo.write_text(json.dumps({"nhl": {"ratings": {"bruins": 1550.0}, "game_counts": {}}}))
    assert main(["--sector", "nhl", "--dry-run", "--state", str(elo)]) == 1
    assert "sweep" in capsys.readouterr().err
    assert main(["--sector", "nhl", "--keep", "0.8", "--dry-run", "--state", str(elo)]) == 0
