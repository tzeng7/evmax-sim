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
    parse_renames,
    prune_elo_keys,
    prune_form_state,
    rename_form_keys,
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
    # (nhl was the example here until it got its own swept value, 2026-09-22.)
    elo = tmp_path / "elo_state.json"
    elo.write_text(json.dumps({"nba": {"ratings": {"celtics": 1550.0}, "game_counts": {}}}))
    assert main(["--sector", "nba", "--dry-run", "--state", str(elo)]) == 1
    assert "sweep" in capsys.readouterr().err
    assert main(["--sector", "nba", "--keep", "0.8", "--dry-run", "--state", str(elo)]) == 0


# ── NHL: keep=1.0 + --prune-only / --rename (2026-09-22) ───────────────────

def test_nhl_default_keep_is_no_regression():
    # Walk-forward: keep 0.75 was worse than 1.0 inside the NHL blend.
    assert SECTOR_DEFAULT_KEEP["nhl"] == 1.0


def _nhl_state():
    return {
        "nhl": {
            "ratings": {"boston bruins": 1540.12, "st. louis blues": 1501.79,
                        "tampa bay lightning": 1530.0, "canada": 1510.0, "arizona coyotes": 1483.51},
            "game_counts": {"boston bruins": 246, "st. louis blues": 246,
                            "tampa bay lightning": 246, "arizona coyotes": 82},
            "season_games": {"boston bruins": 246, "st. louis blues": 246,
                             "tampa bay lightning": 246, "canada": 4, "arizona coyotes": 82},
            "h2h": {
                "boston bruins::st. louis blues": {"a_wins": 3, "b_wins": 1, "games": 4},
                "st. louis blues::tampa bay lightning": {"a_wins": 2, "b_wins": 5, "games": 7},
                "arizona coyotes::boston bruins": {"a_wins": 0, "b_wins": 2, "games": 2},
                "canada::usa": {"a_wins": 1, "b_wins": 0, "games": 1},
            },
            "last_updated": "2026-04-17",
        },
        "nfl": {"ratings": {"chiefs": 1600.0}, "game_counts": {}, "season_games": {}, "h2h": {}},
    }


def test_prune_elo_keys_drops_and_renames_without_regressing():
    state = _nhl_state()
    nfl_before = json.loads(json.dumps(state["nfl"]))
    out = prune_elo_keys(state, "nhl", drop=["canada", "usa", "arizona coyotes"],
                         renames=[("st. louis blues", "st louis blues")])
    sec = state["nhl"]
    # "usa" only lived in the canada::usa h2h entry, already gone with canada.
    assert out["dropped"] == ["canada", "arizona coyotes"]
    assert out["renamed"] == ["st. louis blues->st louis blues"]
    # Ratings / counts are untouched (no shrink, no rounding, no season reset).
    assert sec["ratings"] == {"boston bruins": 1540.12, "st louis blues": 1501.79,
                              "tampa bay lightning": 1530.0}
    assert sec["season_games"] == {"boston bruins": 246, "st louis blues": 246,
                                   "tampa bay lightning": 246}
    assert list(sec["ratings"]) == ["boston bruins", "st louis blues", "tampa bay lightning"]
    assert sec["h2h"] == {
        "boston bruins::st louis blues": {"a_wins": 3, "b_wins": 1, "games": 4},
        "st louis blues::tampa bay lightning": {"a_wins": 2, "b_wins": 5, "games": 7},
    }
    assert "offseason_regression" not in sec and sec["last_updated"] == "2026-04-17"
    assert state["nfl"] == nfl_before


def test_prune_elo_rename_swaps_h2h_when_order_flips():
    state = {"nhl": {"ratings": {"zeta": 1500.0, "mid": 1500.0}, "game_counts": {},
                     "season_games": {}, "h2h": {"mid::zeta": {"a_wins": 4, "b_wins": 1, "games": 5}}}}
    prune_elo_keys(state, "nhl", renames=[("zeta", "alpha")])
    assert state["nhl"]["h2h"] == {"alpha::mid": {"a_wins": 1, "b_wins": 4, "games": 5}}


def test_prune_elo_rename_refuses_to_merge():
    state = _nhl_state()
    state["nhl"]["ratings"]["st louis blues"] = 1500.0
    with pytest.raises(ValueError):
        prune_elo_keys(state, "nhl", renames=[("st. louis blues", "st louis blues")])


def test_rename_form_keys_moves_value_only():
    form = {"nhl": {"st. louis blues": [{"date": "2026-04-16", "opp": "utah mammoth"}],
                    "utah mammoth": [{"date": "2026-04-16", "opp": "st. louis blues"}]}}
    assert rename_form_keys(form, "nhl", [("st. louis blues", "st louis blues")]) == [
        "st. louis blues->st louis blues"]
    assert list(form["nhl"]) == ["st louis blues", "utah mammoth"]
    # opp labels are history used only for de-dup — left as recorded.
    assert form["nhl"]["utah mammoth"][0]["opp"] == "st. louis blues"


def test_parse_renames_validates():
    assert parse_renames(["St. Louis Blues = st louis blues"]) == [("st. louis blues", "st louis blues")]
    for bad in ("no-equals", "a=a", "=b"):
        with pytest.raises(ValueError):
            parse_renames([bad])


def test_cli_prune_only_writes_only_the_pruned_keys(tmp_path):
    elo = tmp_path / "elo_state.json"
    form = tmp_path / "form_state.json"
    elo.write_text(json.dumps(_nhl_state(), indent=2))
    form.write_text(json.dumps({"nhl": {"canada": [], "boston bruins": [{"date": "2026-04-14"}],
                                        "st. louis blues": []}}, indent=2))
    args = ["--sector", "nhl", "--prune-only", "--drop", "canada,usa,arizona coyotes",
            "--rename", "st. louis blues=st louis blues", "--state", str(elo), "--form-state", str(form)]
    assert main(args + ["--dry-run"]) == 0
    assert "canada" in json.loads(elo.read_text())["nhl"]["ratings"]  # dry run wrote nothing
    assert main(args) == 0
    after = json.loads(elo.read_text())["nhl"]
    assert sorted(after["ratings"]) == ["boston bruins", "st louis blues", "tampa bay lightning"]
    assert after["season_games"]["boston bruins"] == 246  # NOT reset to 0
    assert sorted(json.loads(form.read_text())["nhl"]) == ["boston bruins", "st louis blues"]


def test_cli_rename_requires_prune_only(tmp_path, capsys):
    elo = tmp_path / "elo_state.json"
    elo.write_text(json.dumps(_nhl_state()))
    assert main(["--sector", "nhl", "--rename", "a=b", "--dry-run", "--state", str(elo)]) == 1
    assert "--prune-only" in capsys.readouterr().err
