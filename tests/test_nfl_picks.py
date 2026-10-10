"""evmax.nfl_projections.picks — model picks vs the Vegas line, the stored record, ESPN openers."""
from __future__ import annotations

import pandas as pd
import pytest

from evmax.nfl_projections import data, picks
from evmax.nfl_projections.game_model import GameModelConfig, walk_forward


# ── making picks ─────────────────────────────────────────────────────────────

def test_spread_pick_takes_the_side_the_model_prefers():
    # market DAL (home) -8.5, model DAL -2.6 -> the dog
    p = picks.make_pick("DAL", "TB", 2.6, 49.2, 8.5, 47.5, week=5)
    assert (p.spread_pick_home, p.spread_pick) == (False, "TB +8.5")
    assert p.spread_edge == pytest.approx(5.9)
    # market JAX -7, model JAX -9.6 -> the favorite
    p = picks.make_pick("JAX", "PHI", 9.6, 44.4, 7.0, 42.5, week=5)
    assert (p.spread_pick_home, p.spread_pick) == (True, "JAX -7")


def test_spread_pick_when_the_away_team_is_favored():
    # market BAL (away) -3.5 -> home margin -3.5; model has ATL (home) by 0.1
    p = picks.make_pick("ATL", "BAL", 0.1, 46.0, -3.5, 43.5, week=5)
    assert p.spread_pick == "ATL +3.5" and p.spread_pick_home
    p = picks.make_pick("ATL", "BAL", -6.0, 46.0, -3.5, 43.5, week=5)
    assert p.spread_pick == "BAL -3.5" and not p.spread_pick_home
    assert picks.make_pick("NE", "LV", 1.0, 40.0, 0.0, 40.0, week=5).spread_pick == "NE PK"


def test_no_pick_without_a_line_or_without_disagreement():
    p = picks.make_pick("DAL", "TB", 3.0, 47.0, None, float("nan"), week=5)
    assert p.spread_pick is None and p.spread_edge is None and p.total_pick is None
    p = picks.make_pick("DAL", "TB", 3.0, 47.0, 3.0, 47.0 + picks.TOTAL_MEDIAN_SHIFT, week=5)
    assert p.spread_pick is None and p.spread_edge == 0
    assert p.total_pick is None and p.total_edge == 0


def test_total_pick_uses_the_median_not_the_mean():
    # a mean projection 0.5 above the line is still an UNDER once shifted to the median
    p = picks.make_pick("DAL", "TB", 3.0, 48.0, 3.0, 47.5, week=5)
    assert picks.TOTAL_MEDIAN_SHIFT < -0.5
    assert p.total_median == pytest.approx(48.0 + picks.TOTAL_MEDIAN_SHIFT)
    assert (p.total_pick, p.total_pick_over) == ("Under 47.5", False)
    p = picks.make_pick("DAL", "TB", 3.0, 52.0, 3.0, 47.5, week=5)
    assert (p.total_pick, p.total_pick_over) == ("Over 47.5", True)


def test_check_flags():
    assert picks.make_pick("DAL", "TB", 3.0, 47, 3.5, 47, week=5).flags == ()
    assert "big edge" in picks.make_pick("DAL", "TB", 3.0, 47, 8.5, 47, week=5).flags
    assert "QB change" in picks.make_pick("DAL", "TB", 3.0, 47, 3.5, 47, week=5, away_qb_delta=-0.08).flags
    assert "QB change" not in picks.make_pick("DAL", "TB", 3.0, 47, 3.5, 47, week=5, home_qb_delta=0.04).flags
    assert "late season" in picks.make_pick("DAL", "TB", 3.0, 47, 3.5, 47, week=14).flags


# ── grading ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("pick_home,line,margin,res", [
    (False, 8.5, -8, "W"),    # TB +8.5 and DAL lost by 8 -> TB covers
    (False, 8.5, 9, "L"),
    (True, 3.0, 3, "P"),      # whole-number push
    (True, -3.5, -3, "W"),    # home +3.5 loses by 3
    (None, 3.0, 10, None),
    (True, None, 10, None),
])
def test_grade_spread(pick_home, line, margin, res):
    assert picks.grade_spread(pick_home, line, margin) == res


def test_grade_total_and_line_moves():
    assert picks.grade_total(True, 47.5, 48) == "W"
    assert picks.grade_total(False, 47.5, 48) == "L"
    assert picks.grade_total(True, 44.0, 44) == "P"
    assert picks.grade_total(None, 44.0, 44) is None
    # home pick wants the home-margin line to RISE; away pick wants it to fall
    assert picks.line_move_toward(True, 3.0, 4.5) == pytest.approx(1.5)
    assert picks.line_move_toward(False, 8.0, 9.5) == pytest.approx(-1.5)   # home 8 -> 9.5 favored: away from the away pick
    assert picks.line_move_toward(False, 9.5, 8.0) == pytest.approx(1.5)
    assert picks.line_move_toward(True, None, 4.0) is None


def test_add_picks_appends_pick_columns_without_a_cover_probability():
    proj = pd.DataFrame({
        "home_team": ["DAL", "JAX"], "away_team": ["TB", "PHI"], "proj_margin": [2.6, 9.6],
        "proj_total": [49.2, 44.4], "market_spread_line": [8.5, 7.0], "market_total_line": [47.5, 42.5],
        "week": [5, 5], "home_qb_delta": [0.0, -0.2], "away_qb_delta": [0.0, 0.0]})
    out = picks.add_picks(proj)
    assert list(out["spread_pick"]) == ["TB +8.5", "JAX -7"]
    assert list(out["flags"]) == [("big edge",), ("QB change",)]
    assert "cover_prob" not in out.columns and len(out) == 2


# ── the stored record ────────────────────────────────────────────────────────

def _week(spread=8.5, total=47.5, gameday="2026-10-08", proj_margin=2.6):
    proj = pd.DataFrame({
        "game_id": ["2026_05_TB_DAL"], "season": [2026], "week": [5], "gameday": [pd.Timestamp(gameday).date()],
        "gametime": ["20:15"], "home_team": ["DAL"], "away_team": ["TB"], "proj_home": [25.9],
        "proj_away": [25.9 - proj_margin], "proj_margin": [proj_margin], "proj_total": [51.8 - proj_margin],
        "market_spread_line": [spread], "market_total_line": [total], "home_qb_delta": [0.0], "away_qb_delta": [0.0]})
    return picks.add_picks(proj)


def test_record_keeps_the_first_pre_kickoff_pick_and_never_reprices(tmp_path):
    conn = picks.connect(tmp_path / "p.db")
    before = pd.Timestamp("2026-10-06 12:00", tz="America/New_York")
    assert picks.record_picks(conn, _week(spread=8.5), now=before) == 1
    # the line moves to 9.5 and the model updates: the stored pick does not change
    assert picks.record_picks(conn, _week(spread=9.5, proj_margin=4.0), now=before) == 0
    row = conn.execute("SELECT * FROM nfl_picks").fetchone()
    assert (row["line_margin"], row["spread_pick"], row["proj_margin"]) == (8.5, "TB +8.5", 2.6)


def test_record_skips_kicked_off_games_and_games_without_a_line(tmp_path):
    conn = picks.connect(tmp_path / "p.db")
    after = pd.Timestamp("2026-10-08 20:15", tz="America/New_York")
    assert picks.record_picks(conn, _week(), now=after) == 0
    before = pd.Timestamp("2026-10-06 12:00", tz="America/New_York")
    assert picks.record_picks(conn, _week(spread=float("nan")), now=before) == 0
    assert conn.execute("SELECT COUNT(*) FROM nfl_picks").fetchone()[0] == 0


def test_record_fills_a_total_posted_after_the_spread(tmp_path):
    conn = picks.connect(tmp_path / "p.db")
    before = pd.Timestamp("2026-10-06 12:00", tz="America/New_York")
    picks.record_picks(conn, _week(total=float("nan")), now=before)
    assert conn.execute("SELECT total_pick FROM nfl_picks").fetchone()[0] is None
    picks.record_picks(conn, _week(total=47.5), now=before)
    row = conn.execute("SELECT total_pick, line_total FROM nfl_picks").fetchone()
    assert (row["total_pick"], row["line_total"]) == ("Over 47.5", 47.5)


def test_grade_picks_at_published_line_and_close(tmp_path):
    conn = picks.connect(tmp_path / "p.db")
    picks.record_picks(conn, _week(spread=8.5, total=47.5), now=pd.Timestamp("2026-10-06 12:00", tz="America/New_York"))
    games = pd.DataFrame({"game_id": ["2026_05_TB_DAL", "2026_05_X_Y"], "home_score": [16.0, None],
                          "away_score": [24.0, None], "spread_line": [9.5, 3.0], "total_line": [49.0, 44.0]})
    assert picks.grade_picks(conn, games) == 1
    r = conn.execute("SELECT * FROM nfl_picks").fetchone()
    assert (r["spread_result"], r["spread_result_close"]) == ("W", "W")       # TB won outright
    assert (r["total_result"], r["total_result_close"]) == ("L", "L")         # 40 points: under
    assert r["spread_move"] == pytest.approx(-1.0)                            # DAL -8.5 -> -9.5: away from TB +8.5
    assert r["total_move"] == pytest.approx(1.5)                              # 47.5 -> 49: toward the over
    assert picks.grade_picks(conn, games) == 0                                # graded once
    s = picks.record_summary(pd.read_sql_query("SELECT * FROM nfl_picks", conn))
    assert s["spread_result"].startswith("1-0") and s["total_result"].startswith("0-1")


def test_kickoff_is_eastern_with_a_1pm_default():
    assert picks.kickoff("2026-10-08", "20:15") == pd.Timestamp("2026-10-08 20:15", tz="America/New_York")
    assert picks.kickoff(pd.Timestamp("2026-10-11"), None).hour == 13


# ── ESPN opening lines ───────────────────────────────────────────────────────

def test_parse_espn_open_shapes():
    modern = {"items": [
        {"provider": {"name": "ESPN Bet - Live Odds"}, "homeTeamOdds": {"open": {"pointSpread": {"american": "-27.5"}}}},
        {"provider": {"name": "ESPN BET"}, "homeTeamOdds": {"open": {"pointSpread": {"american": "-2.5"}}},
         "open": {"total": {"american": "48.5"}}}]}
    assert data.parse_espn_open(modern) == (-2.5, 48.5, "ESPN BET")       # live provider skipped
    even = {"items": [{"provider": {"name": "Draft Kings"}, "homeTeamOdds": {"open": {"pointSpread": {"american": "EVEN"}}}}]}
    assert data.parse_espn_open(even)[0] == 0.0
    legacy = {"items": [{"provider": {"name": "consensus"}, "spread": -6.0},
                        {"provider": {"name": "Opening"}, "spread": -2.0, "overUnder": 43.5}]}
    assert data.parse_espn_open(legacy) == (-2.0, 43.5, "Opening")
    assert data.parse_espn_open({"items": [{"provider": {"name": "consensus"}, "spread": -6.0}]}) == (None, None, None)
    assert data.parse_espn_open(None) == (None, None, None)


def test_load_espn_open_lines_reads_cache_without_fetching(tmp_path):
    pd.DataFrame({"game_id": ["g1"], "open_line_margin": [2.5], "open_total": [48.5],
                  "open_source": ["ESPN BET"]}).to_parquet(data.espn_open_file(tmp_path))
    games = pd.DataFrame({"game_id": ["g1", "g2"], "home_score": [20.0, 21.0], "espn": ["1", "2"]})
    out = data.load_espn_open_lines(games, tmp_path, fetch=False)
    assert list(out["game_id"]) == ["g1"] and out.loc[0, "open_line_margin"] == 2.5


def test_walk_forward_exposes_qb_deltas(monkeypatch):
    """walk_forward returns each side's QB delta (the picks' QB-change flag) when the qb feature is on."""
    import evmax.nfl_projections.game_model as gm
    ft = pd.DataFrame({
        "game_id": ["g1", "g1", "g2", "g2"], "season": [2015, 2015, 2016, 2016], "week": [1, 1, 1, 1],
        "side": ["home", "away"] * 2, "points": [24.0, 20.0, 27.0, 17.0],
        **{f: [1.0, 0.5, 1.2, 0.4] for f in ("pts", "epa", "sr", "dome", "wind")}, "qb": [0.0, -0.1, 0.02, 0.0]})
    monkeypatch.setattr(gm, "feature_table", lambda *a, **k: ft)
    games = pd.DataFrame({"game_id": ["g2"], "game_type": ["REG"], "home_team": ["A"], "away_team": ["B"],
                          "home_score": [27.0], "away_score": [17.0], "spread_line": [3.0], "total_line": [44.0],
                          "location": ["Home"], "roof": ["outdoors"], "wind": [5.0], "temp": [60.0], "div_game": [0]})
    out = walk_forward(pd.DataFrame(), games, [2016], GameModelConfig())
    assert {"proj_home", "proj_away", "home_qb_delta", "away_qb_delta"} <= set(out.columns)
    assert out.loc[0, "home_qb_delta"] == pytest.approx(0.02) and out.loc[0, "away_qb_delta"] == 0.0


def test_nfl_record_cli_grades_and_renders(tmp_path, monkeypatch):
    from typer.testing import CliRunner
    import evmax.cli.commands.project as project_cli

    monkeypatch.setattr(project_cli, "_PROJ_DB_PATH", tmp_path / "p.db")
    conn = picks.connect(tmp_path / "p.db")
    picks.record_picks(conn, _week(), now=pd.Timestamp("2026-10-06 12:00", tz="America/New_York"))
    conn.close()
    games = pd.DataFrame({"game_id": ["2026_05_TB_DAL"], "home_score": [16.0], "away_score": [24.0],
                          "spread_line": [9.5], "total_line": [49.0]})
    monkeypatch.setattr(data, "load_games", lambda d=None: games)
    res = CliRunner().invoke(project_cli.app, ["nfl-record", "--no-refresh"])
    assert res.exit_code == 0, res.output
    assert "TB +8.5" in res.output and "1-0" in res.output and "Graded 1 new game" in res.output
