"""evmax.nfl_projections: who is expected to play — weekly roster status, starting QB,
participation probability and the raw ESPN injury feed (the 2026-10-10 IR / stale-QB fixes)."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from evmax.clients.nfl_depth_charts import QbChartRow
from evmax.nfl_projections import data, live
from evmax.nfl_projections.live import StarterInputs, pick_starter
from evmax.nfl_projections.player_model import (
    PARTICIPATION, PlayerModelConfig, fit_player_state, participation_probability, pregame_rosters,
    project_players, recent_team_players, roster_unavailable, team_volume_rows, walk_forward,
)

from tests.test_nfl_player_projections import _synthetic


def _roster(rows):
    return pd.DataFrame(rows, columns=["season", "week", "team", "gsis_id", "full_name", "position", "status"])


# ── weekly roster status ─────────────────────────────────────────────────────

def test_roster_unavailable_rules_out_reserve_practice_squad_released_and_departed():
    rosters = _roster([
        (2026, 5, "SEA", "act", "Act", "WR", "ACT"),
        (2026, 5, "SEA", "ir", "Jadarian Price", "RB", "RES"),     # injured reserve: off the injury report
        (2026, 5, "SEA", "ps", "Practice", "WR", "DEV"),
        (2026, 5, "SEA", "cut", "Cut", "WR", "CUT"),
        (2026, 4, "SEA", "gone", "Gone", "TE", "ACT"),             # last week's roster only
        (2026, 5, "SF", "ina", "Inactive", "WR", "INA"),
    ])
    players = pd.DataFrame({"team": ["SEA"] * 6 + ["SF", "KC"],
                            "player_id": ["act", "ir", "ps", "cut", "gone", "nobody", "ina", "kc1"]})
    out = roster_unavailable(rosters, players, 2026, 5)
    # a player missing from his team's current roster has left; a team with no rows is left alone
    assert out == {("SEA", "ir"), ("SEA", "ps"), ("SEA", "cut"), ("SEA", "gone"), ("SEA", "nobody"), ("SF", "ina")}
    # the week's own INA list counts only with inactive=True (the backtest runs before inactives)
    assert ("SF", "ina") not in roster_unavailable(rosters, players, 2026, 5, inactive=False)
    assert roster_unavailable(None, players, 2026, 5) == set()
    assert roster_unavailable(rosters, players, 2025, 5) == set()       # no rows for that season


def test_roster_unavailable_falls_back_to_the_latest_published_week():
    rosters = _roster([
        (2026, 4, "SEA", "ir", "IR", "RB", "RES"),
        (2026, 4, "SEA", "scratch", "Scratch", "WR", "INA"),           # last week's game-day inactive
        (2026, 4, "SEA", "act", "Act", "WR", "ACT"),
    ])
    players = pd.DataFrame({"team": ["SEA", "SEA", "SEA"], "player_id": ["ir", "scratch", "act"]})
    # week 5 not published: week 4's reserve list still applies, its INA (another game) does not
    assert roster_unavailable(rosters, players, 2026, 5) == {("SEA", "ir")}


def test_load_rosters_normalizes_teams_and_is_empty_without_files(tmp_path):
    assert data.load_rosters([2026], tmp_path).empty
    f = data.rosters_file(2026, tmp_path)
    f.parent.mkdir(parents=True)
    pd.DataFrame([{"season": 2026, "week": 5, "game_type": "REG", "team": "OAK", "gsis_id": "x",
                   "full_name": "X", "position": "WR", "status": "ACT", "status_description_abbr": "A01"}]
                 ).to_parquet(f)
    r = data.load_rosters([2025, 2026], tmp_path)
    assert list(r["team"]) == ["LV"] and list(r.columns) == data.ROSTER_COLUMNS
    # a file without the status column is skipped, not read as "everyone ruled out"
    pd.DataFrame([{"season": 2026, "week": 5, "team": "LV", "gsis_id": "x"}]).to_parquet(f)
    assert data.load_rosters([2026], tmp_path).empty


# ── participation ────────────────────────────────────────────────────────────

def test_recent_team_players_reports_recent_games_and_the_last_game():
    day = pd.Timestamp("2026-09-01")
    rows = [{"game_id": f"g{i}", "team": "AAA", "player_id": "every", "gameday": day + pd.Timedelta(weeks=i)}
            for i in range(3)]
    rows += [{"game_id": "g0", "team": "AAA", "player_id": "early", "gameday": day},
             {"game_id": "g2", "team": "AAA", "player_id": "latest", "gameday": day + pd.Timedelta(weeks=2)}]
    pg = pd.DataFrame(rows).assign(player_display_name="x", position="WR")
    r = recent_team_players(pg, {"AAA"}, day + pd.Timedelta(weeks=4)).set_index("player_id")
    assert r.loc["every", "recent_games"] == 3 and r.loc["every", "played_last"]
    assert r.loc["early", "recent_games"] == 1 and not r.loc["early", "played_last"]
    assert r.loc["latest", "recent_games"] == 1 and r.loc["latest", "played_last"]


def test_played_last_is_each_teams_own_last_game():
    # AAA played BBB in week 1; BBB had a bye in week 2 while AAA played CCC. A AAA player who
    # missed the week-2 game did not play AAA's last game, even though week 1 is BBB's last game.
    d1, d2 = pd.Timestamp("2026-09-10"), pd.Timestamp("2026-09-17")
    pg = pd.DataFrame([
        {"game_id": "w1", "team": "AAA", "player_id": "missed_w2", "gameday": d1},
        {"game_id": "w1", "team": "AAA", "player_id": "both", "gameday": d1},
        {"game_id": "w1", "team": "BBB", "player_id": "bbb", "gameday": d1},
        {"game_id": "w2", "team": "AAA", "player_id": "both", "gameday": d2},
        {"game_id": "w2", "team": "CCC", "player_id": "ccc", "gameday": d2},
    ]).assign(player_display_name="x", position="WR")
    cutoff = pd.Timestamp("2026-09-24")
    for teams in ({"AAA", "BBB"}, {"AAA"}):                       # same answer whatever the slate
        r = recent_team_players(pg, teams, cutoff).set_index("player_id")
        assert not r.loc["missed_w2", "played_last"] and r.loc["both", "played_last"]
    assert recent_team_players(pg, {"BBB"}, cutoff).set_index("player_id").loc["bbb", "played_last"]


def test_participation_probability_reads_the_table_and_always_plays_the_starter():
    p = participation_probability([3, 1, 2, 3, 0, 1], [True, False, True, True, False, False],
                                  [False, False, True, True, False, False], [False, False, False, False, True, True])
    assert list(p) == [PARTICIPATION[(3, True, False)], PARTICIPATION[(1, False, False)],
                       PARTICIPATION[(2, True, True)], PARTICIPATION[(3, True, True)], 1.0, 1.0]
    assert participation_probability([0], [False], [False], [False])[0] == 1.0   # no evidence (an added player)
    assert all(0 < v < 1 for v in PARTICIPATION.values())
    assert PARTICIPATION[(3, True, False)] > PARTICIPATION[(1, False, False)]


def test_project_players_adds_p_active_only_for_a_pre_game_roster():
    tg, games, pg = _synthetic(seasons=(2021,))
    st = fit_player_state(pg, team_volume_rows(pg, tg), pd.Timestamp("2022-01-01"), PlayerModelConfig())
    base = pd.DataFrame([{"game_id": "X", "team": "AAA", "opp": "BBB", "home": 1, "player_id": pid,
                          "position": "WR", "is_starting_qb": False} for pid in ("AAA-WR1", "AAA-WR2")])
    assert "p_active" not in project_players(st, base)            # who-played roster: everyone plays
    live_roster = base.assign(recent_games=[3, 1], played_last=[True, False], questionable=[False, False])
    p = project_players(st, live_roster).set_index("player_id")["p_active"]
    assert p["AAA-WR1"] == PARTICIPATION[(3, True, False)] and p["AAA-WR2"] == PARTICIPATION[(1, False, False)]


# ── live roster in the walk-forward ──────────────────────────────────────────

def test_walk_forward_live_roster_drops_reserve_players_without_redistributing():
    tg, games, pg = _synthetic()
    rosters = _roster([(2022, w, t, f"{t}-{s}", f"{t} {s}", "WR", "ACT")
                       for w in range(1, 13) for t in ("AAA", "BBB", "CCC", "DDD")
                       for s in ("WR1", "WR2", "RB1", "TE1", "QB1")])
    rosters.loc[(rosters["week"] >= 4) & (rosters["gsis_id"] == "AAA-WR1"), "status"] = "RES"   # IR from week 4
    played = walk_forward(tg, pg, games, [2022], rosters=rosters)
    lv = walk_forward(tg, pg, games, [2022], rosters=rosters, roster="live")
    assert lv["played"].all() and "p_active" in lv                     # the synthetic league never sits anyone
    on_ir = lv[(lv["player_id"] == "AAA-WR1") & (lv["week"] >= 4)]
    assert on_ir.empty                                                  # dropped from the expected roster
    # ...but teammates keep their usage (redistribution from roster-out players was rejected)
    m = played.merge(lv, on=["game_id", "player_id"], suffixes=("", "_l"))
    assert np.allclose(m["proj_targets"], m["proj_targets_l"])
    redistributed = walk_forward(tg, pg, games, [2022], rosters=rosters,
                                 cfg=replace(PlayerModelConfig(), roster_out_redistribution=True))
    r = played.merge(redistributed, on=["game_id", "player_id"], suffixes=("", "_r"))
    moved = r[~np.isclose(r["proj_targets"], r["proj_targets_r"])]
    assert len(moved) and (moved["team"] == "AAA").all() and (moved["week"] >= 4).all()
    with pytest.raises(ValueError, match="roster"):
        walk_forward(tg, pg, games, [2022], roster="nope")


def test_pregame_rosters_match_the_live_walk_forward_roster():
    tg, games, pg = _synthetic()
    r = pregame_rosters(tg, pg, games, [2022], None, None)
    lv = walk_forward(tg, pg, games, [2022], roster="live")
    assert set(zip(r["game_id"], r["player_id"])) == set(zip(lv["game_id"], lv["player_id"]))
    assert {"recent_games", "played_last", "questionable", "played", "season", "week"} <= set(r.columns)
    assert pregame_rosters(tg, pg, games, [1999], None, None).empty


def test_active_roster_drops_backup_qbs_and_carries_participation_evidence():
    day = pd.Timestamp("2026-09-20")
    rows = []
    for i in range(3):
        g = f"g{i}"
        rows += [{"game_id": g, "team": "SEA", "player_id": "darnold", "player_display_name": "Sam Darnold",
                  "position": "QB", "gameday": day + pd.Timedelta(weeks=i)},
                 {"game_id": g, "team": "SEA", "player_id": "jsn", "player_display_name": "JSN", "position": "WR",
                  "gameday": day + pd.Timedelta(weeks=i)}]
    rows.append({"game_id": "g0", "team": "SEA", "player_id": "lock", "player_display_name": "Drew Lock",
                 "position": "QB", "gameday": day})
    rows.append({"game_id": "g0", "team": "SEA", "player_id": "rb3", "player_display_name": "RB3",
                 "position": "RB", "gameday": day})
    pg = pd.DataFrame(rows)
    inj = pd.DataFrame([{"week": 5, "team": "SEA", "gsis_id": "jsn", "report_status": "Questionable"}])
    wk = pd.DataFrame([{"game_id": "g5", "home_team": "SEA", "away_team": "SF", "location": "Home",
                        "home_qb_id": "lock", "away_qb_id": None}])                  # stale schedule QB
    r = live.active_roster(pg, inj, wk, pd.Timestamp("2026-10-11"), 5, starters={"SEA": "darnold"})
    sea = r[r["team"] == "SEA"].set_index("player_id")
    assert set(sea.index) == {"darnold", "jsn", "rb3"}                 # Lock (backup QB) dropped
    assert sea.loc["darnold", "is_starting_qb"]
    assert sea.loc["jsn", "questionable"] and sea.loc["jsn", "recent_games"] == 3
    assert sea.loc["rb3", "recent_games"] == 1 and not sea.loc["rb3", "played_last"]
    esp = live.active_roster(pg, pd.DataFrame(columns=inj.columns), wk, pd.Timestamp("2026-10-11"), 5,
                             starters={"SEA": "darnold"}, extra_questionable={("SEA", "rb3")})
    assert esp.set_index("player_id").loc["rb3", "questionable"]


# ── starting QB ──────────────────────────────────────────────────────────────

def test_pick_starter_order():
    assert pick_starter(StarterInputs(depth=["a"], override="o")) == ("o", "override")
    assert pick_starter(StarterInputs(depth=["a", "b"], last_start="b", schedule="c")) == ("a", "depth chart")
    assert pick_starter(StarterInputs(depth=["a", "b"], ruled_out={"a"})) == ("b", "depth chart")
    # an unknown QB1 stops the walk instead of promoting QB2
    assert pick_starter(StarterInputs(depth=[None, "b"], last_start="l")) == ("l", "last start")
    assert pick_starter(StarterInputs(depth=[], last_start="l", schedule="s")) == ("l", "last start")
    assert pick_starter(StarterInputs(last_start="l", schedule="s", ruled_out={"l"})) == ("s", "schedule")
    assert pick_starter(StarterInputs(last_start="l", schedule="s", ruled_out={"l", "s"})) == ("s", "schedule")
    assert pick_starter(StarterInputs(last_start="l", ruled_out={"l"})) == ("l", "last start")
    assert pick_starter(StarterInputs()) == (None, "none")
    # a Questionable QB1 returning from injury yields to last week's starter (CHI 2026 Week 5)
    back = StarterInputs(depth=["williams", "bagent"], questionable={"williams"}, last_start="bagent")
    assert pick_starter(back) == ("bagent", "last start (QB1 questionable)")
    # ...but not when he started last week, nor when last week's starter is out
    assert pick_starter(replace(back, last_start="williams"))[0] == "williams"
    assert pick_starter(replace(back, ruled_out={"bagent"}))[0] == "williams"


def _snap(team, players, day="2026-10-10"):
    t = datetime.fromisoformat(f"{day}T13:00:00+00:00")
    return [QbChartRow(team, p, i + 1, t, None) for i, p in enumerate(players)]


def _starter_case(depth_rows, injuries=None, rosters=None, espn=None):
    wk = pd.DataFrame([{"game_id": "2026_05_SF_SEA", "home_team": "SEA", "away_team": "SF",
                        "home_qb_id": "lock", "away_qb_id": "purdy", "home_qb_name": "Drew Lock",
                        "away_qb_name": "Brock Purdy", "home_score": np.nan}])
    tg = pd.DataFrame([
        {"team": "SEA", "gameday": pd.Timestamp("2026-10-04"), "first_qb_id": "darnold", "first_qb_name": "S.Darnold"},
        {"team": "SEA", "gameday": pd.Timestamp("2026-09-20"), "first_qb_id": "lock", "first_qb_name": "D.Lock"},
        {"team": "SF", "gameday": pd.Timestamp("2026-10-04"), "first_qb_id": "purdy", "first_qb_name": "B.Purdy"},
    ])
    return live.week_starters(2026, 5, wk, tg, pd.Timestamp("2026-10-11"), injuries, rosters, espn_reports=espn,
                              qb_depth=depth_rows)


def test_week_starters_prefers_the_depth_chart_over_a_stale_schedule():
    rows = _snap("seattle seahawks", ["Sam Darnold", "Drew Lock"]) + _snap("san francisco 49ers", ["Brock Purdy"])
    rosters = _roster([(2026, 5, "SEA", "darnold", "Sam Darnold", "QB", "ACT"),
                       (2026, 5, "SEA", "lock", "Drew Lock", "QB", "ACT"),
                       (2026, 5, "SF", "purdy", "Brock Purdy", "QB", "ACT")])
    out = _starter_case(rows, rosters=rosters)
    assert out["SEA"] == ("darnold", "depth chart", "Sam Darnold")     # schedule said Lock (2026 Weeks 3-5)
    assert out["SF"] == ("purdy", "depth chart", "Brock Purdy")
    # QB1 ruled out by the injury report -> QB2; by the ESPN feed -> QB2; by the roster -> QB2
    inj = pd.DataFrame([{"season": 2026, "week": 5, "team": "SEA", "gsis_id": "darnold", "report_status": "Out"}])
    assert _starter_case(rows, injuries=inj, rosters=rosters)["SEA"][0] == "lock"
    espn = {"seattle seahawks": live.EspnTeamReport("seattle seahawks",
                                                    [live.EspnStatus("Sam Darnold", "QB", "Injured Reserve")])}
    assert _starter_case(rows, rosters=rosters, espn=espn)["SEA"][0] == "lock"
    # Questionable on the report: Darnold started last week, so he stays the starter
    q = pd.DataFrame([{"season": 2026, "week": 5, "team": "SEA", "gsis_id": "darnold",
                       "report_status": "Questionable"}])
    assert _starter_case(rows, injuries=q, rosters=rosters)["SEA"] == ("darnold", "depth chart", "Sam Darnold")
    flipped = _snap("seattle seahawks", ["Drew Lock", "Sam Darnold"])          # QB1 Lock, Questionable
    q2 = q.assign(gsis_id="lock")
    assert _starter_case(flipped, injuries=q2, rosters=rosters)["SEA"][:2] == ("darnold", "last start (QB1 questionable)")
    res = rosters.copy()
    res.loc[res["gsis_id"] == "darnold", "status"] = "RES"
    assert _starter_case(rows, rosters=res)["SEA"][0] == "lock"


def test_game_day_inactive_qb_counts_only_for_an_unplayed_week():
    # a played week's INA list is published ~90 minutes before kickoff: a replay must not see it
    rows = _snap("seattle seahawks", ["Sam Darnold", "Drew Lock"])
    rosters = _roster([(2026, 5, "SEA", "darnold", "Sam Darnold", "QB", "INA"),
                       (2026, 5, "SEA", "lock", "Drew Lock", "QB", "ACT")])
    assert _starter_case(rows, rosters=rosters)["SEA"][0] == "lock"           # live: inactive -> QB2
    wk = pd.DataFrame([{"game_id": "g", "home_team": "SEA", "away_team": "SF", "home_qb_id": None,
                        "away_qb_id": None}])
    tg = pd.DataFrame(columns=["team", "gameday", "first_qb_id", "first_qb_name"])
    inp, _ = live.starter_inputs(2026, 5, wk, tg, pd.Timestamp("2026-10-11"), None, rosters, qb_depth=rows,
                                 inactive=False)
    assert pick_starter(inp["SEA"])[0] == "darnold"


def test_week_starters_without_a_usable_depth_chart_uses_the_last_start():
    # names resolve through the past starters' pbp names when the roster is missing
    out = _starter_case([], rosters=None)
    assert out["SEA"] == ("darnold", "last start", "S.Darnold")
    weekly = [QbChartRow("seattle seahawks", "Drew Lock", 1, None, 5)]   # pre-2025 weekly chart: ignored
    assert _starter_case(weekly)["SEA"][0] == "darnold"
    pbp_only = _snap("seattle seahawks", ["Sam Darnold"])                 # depth name via pbp passer name
    assert _starter_case(pbp_only)["SEA"] == ("darnold", "depth chart", "S.Darnold")


def test_qb_name_index_marks_ambiguous_names():
    rosters = _roster([(2026, 5, "AAA", "j1", "Josh Allen", "QB", "ACT"),
                       (2026, 5, "AAA", "j2", "Josh Allen", "QB", "ACT")])
    by_team, names = live.qb_name_index(rosters, pd.DataFrame(columns=["team", "gameday", "first_qb_id"]),
                                        2026, 5, pd.Timestamp("2026-10-11"))
    assert by_team["AAA"]["josh allen"] is None and names["j1"] == "Josh Allen"


# ── ESPN feed, read raw ──────────────────────────────────────────────────────

def _espn_doc():
    def inj(name, status, pos="RB"):
        return {"athlete": {"displayName": name, "position": {"abbreviation": pos}}, "status": status,
                "date": "2026-10-03T20:02Z"}
    return {"injuries": [
        {"displayName": "Seattle Seahawks", "injuries": [
            inj("Jadarian Price", "Injured Reserve"), inj("George Holani", "Questionable"),
            inj("Cooper Kupp", "Active", "WR"), inj("Zach Charbonnet", {"name": "Out"}), "junk"]},
        {"displayName": "", "injuries": [inj("X", "Out")]},
    ]}


def test_parse_espn_injuries_keeps_every_non_active_status():
    r = live.parse_espn_injuries(_espn_doc())
    assert set(r) == {"seattle seahawks"}
    got = {(p.name, p.status) for p in r["seattle seahawks"].players}
    # the scanner's InjuryReportAgent drops 'Injured Reserve' (no win-probability impact) — this keeps it
    assert got == {("Jadarian Price", "Injured Reserve"), ("George Holani", "Questionable"),
                   ("Zach Charbonnet", "Out")}
    assert live.parse_espn_injuries({}) == {} and live.parse_espn_injuries(None) == {}


def test_espn_status_sets_split_dropped_from_redistributed():
    reports = live.parse_espn_injuries(_espn_doc())
    recent = pd.DataFrame([{"team": "SEA", "player_id": pid, "player_display_name": n, "position": "RB"}
                           for pid, n in (("price", "Jadarian Price"), ("holani", "George Holani"),
                                          ("charb", "Zach Charbonnet"))])
    assert live.espn_out_players(reports, recent) == {("SEA", "price"), ("SEA", "charb")}
    assert live.espn_out_players(reports, recent, live.ESPN_REDISTRIBUTED_STATUSES) == {("SEA", "charb")}
    assert live.espn_out_players(reports, recent, frozenset({"QUESTIONABLE"})) == {("SEA", "holani")}
    assert live.espn_statuses(reports)["SEA"]["jadarian price"] == "INJURED RESERVE"


def test_fetch_espn_injury_reports_is_fail_soft(monkeypatch):
    import httpx

    def boom(*a, **k):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx, "get", boom)
    assert live.fetch_espn_injury_reports() == {}

    class Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return _espn_doc()

    monkeypatch.setattr(httpx, "get", lambda *a, **k: Resp())
    assert "seattle seahawks" in live.fetch_espn_injury_reports()


def test_game_starters_reads_project_week_output():
    gp = pd.DataFrame([{"home_team": "SEA", "away_team": "SF", "home_qb_id": "darnold", "away_qb_id": None}])
    assert live.game_starters(gp) == {"SEA": "darnold", "SF": None}
    assert live.game_starters(pd.DataFrame()) == {} and live.game_starters(None) == {}
