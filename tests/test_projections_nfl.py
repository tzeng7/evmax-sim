"""evmax.projections.nfl: the NFL engine behind the Projections tab (live calls stubbed)."""

from __future__ import annotations

import json
from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from evmax.nfl_projections import data, live, pipeline, simulate, store
from evmax.projections import nfl
from evmax.projections.base import ProjectionError

UTC = ZoneInfo("UTC")


def _schedule():
    return pd.DataFrame({
        "season": [2026, 2026, 2026], "week": [5, 5, 6],
        "gameday": pd.to_datetime(["2026-10-11", "2026-10-11", "2026-10-18"]),
        "home_score": [np.nan, np.nan, np.nan],
    })


def _games():
    return pd.DataFrame([
        {"game_id": "2026_05_KC_LV", "season": 2026, "week": 5, "gameday": date(2026, 10, 11), "gametime": "16:25",
         "home_team": "LV", "away_team": "KC", "neutral": False, "roof": "dome", "proj_home": 17.0,
         "proj_away": 27.0, "proj_margin": -10.0, "proj_total": 44.0, "p_home_win": 0.23,
         "home_qb_id": "q3", "away_qb_id": "q4", "home_qb_name": "C", "away_qb_name": "D",
         "market_spread_line": -7.5, "market_total_line": 46.0},
        {"game_id": "2026_05_DAL_PHI", "season": 2026, "week": 5, "gameday": date(2026, 10, 11), "gametime": "13:00",
         "home_team": "PHI", "away_team": "DAL", "neutral": False, "roof": "outdoors", "proj_home": 24.0,
         "proj_away": 20.0, "proj_margin": 4.0, "proj_total": 44.0, "p_home_win": 0.6,
         "home_qb_id": "q1", "away_qb_id": "q2", "home_qb_name": "A", "away_qb_name": None,
         "market_spread_line": float("nan"), "market_total_line": float("nan")},
    ])


def _player(pid, team="PHI", opp="DAL", targets=6.0, carries=0.0, qb=False, game="2026_05_DAL_PHI", **kw):
    row = {"game_id": game, "season": 2026, "week": 5, "gameday": pd.Timestamp("2026-10-11"), "gametime": "13:00",
           "player_id": pid, "player_display_name": pid.upper(), "position": "QB" if qb else "WR",
           "team": team, "opp": opp, "home": 1 if team == "PHI" else 0, "is_starting_qb": qb,
           "proj_targets": targets, "proj_carries": carries, "p_anytime_td": 0.3, "proj_passing_tds": 1.6 if qb else 0.0,
           # simulate_team inputs
           "tgt_share": 0.0 if qb else 0.25, "car_share": 0.1, "catch_rate": 0.65, "ypt": 8.0, "ypc": 4.0,
           "rush_xtd_share": 0.1, "rec_xtd_share": 0.0 if qb else 0.3,
           "exp_team_targets": 34.0, "exp_team_carries": 26.0, "exp_team_rush_tds": 0.9, "exp_team_rec_tds": 1.4}
    for st, med in (("receptions", 4.0), ("receiving_yards", 50.0), ("rushing_yards", 10.0), ("passing_yards", 240.0)):
        row.update({f"proj_{st}": med, f"p10_{st}": med * 0.4, f"p90_{st}": med * 1.8})
    row.update(kw)
    return row


def _players():
    return pd.DataFrame([
        _player("qb1", targets=0.0, carries=3.0, qb=True),
        _player("wr1"),
        _player("ol1", targets=0.2, carries=0.0),                     # irrelevant: dropped
        _player("dal1", team="DAL", opp="PHI"),
    ])


@pytest.fixture
def stubs(monkeypatch, tmp_path):
    monkeypatch.setenv("EVMAX_PROJ_DB", str(tmp_path / "projections.db"))
    calls = {"project_week": [], "players": [], "sim": [], "run_week": []}
    monkeypatch.setattr(data, "ensure_games", lambda *a, **k: True)
    monkeypatch.setattr(data, "load_games", lambda *a, **k: _schedule())
    monkeypatch.setattr(live, "fetch_espn_injury_reports", lambda: {"Philadelphia Eagles": object()})

    def project_week(season, week, **kw):
        calls["project_week"].append((season, week, kw))
        return _games()

    def project_week_players(season, week, **kw):
        calls["players"].append((season, week, kw))
        return _players()

    def simulate_game(season, week, team, n=10000, refresh=True, proj=None, **kw):
        calls["sim"].append({"season": season, "week": week, "team": team, "n": n, "proj": proj, "refresh": refresh})
        out = {}
        for t, grp in proj[proj["game_id"] == "2026_05_DAL_PHI"].groupby("team"):
            grp = grp.reset_index(drop=True)
            out[t] = (grp, simulate.simulate_team(grp, 34.0, 26.0, 0.9, 1.4, simulate.SimParams(), n=n,
                                                  rng=np.random.default_rng(0)))
        return out

    def run_week(conn, season, week, refresh=True, espn=True, d=None):
        calls["run_week"].append((season, week, refresh, espn))
        return pipeline.WeekRun(season, week, _games(), _players(), games_logged=2, players_logged=4, picks_logged=1)

    monkeypatch.setattr(live, "project_week", project_week)
    monkeypatch.setattr(live, "project_week_players", project_week_players)
    monkeypatch.setattr(live, "simulate_game", simulate_game)
    monkeypatch.setattr(pipeline, "run_week", run_week)
    return calls


OPTS = {"season": None, "week": None, "players": True, "espn": True, "refresh": False, "store": False}


TODAY = date(2026, 10, 10)          # the Saturday before the fixture's Week 5 games


def engine(**kw) -> nfl.NflProjectionEngine:
    return nfl.NflProjectionEngine(today=lambda: TODAY, **kw)


def test_run_slate_rows(stubs):
    out = engine().run_slate("nfl", OPTS)
    json.dumps(out)
    assert out["title"] == "NFL 2026 · Week 5" and out["source"] == "run" and out["period"] == "2026-5"
    assert [g["game_id"] for g in out["games"]] == ["2026_05_DAL_PHI", "2026_05_KC_LV"]   # by kickoff
    phi, lv = out["games"]
    assert phi["kickoff"] == "2026-10-11T17:00+00:00" and phi["date"] == "2026-10-11"
    assert phi["home_name"] == "Philadelphia Eagles" and phi["model_line"] == "PHI -4.0"
    assert phi["market_line"] is None and phi["market_total"] is None and phi["subtitle"] == "? / A"
    assert lv["market_line"] == "KC -7.5" and phi["context"] == {"season": 2026, "week": 5}
    # players: the 0.2-target, 0-carry non-QB is dropped; cells follow the usage gates
    players = {p["player_id"]: p for p in out["players"]}
    assert set(players) == {"qb1", "wr1", "dal1"}
    qb, wr = players["qb1"]["cells"], players["wr1"]["cells"]
    assert qb["receiving_yards"] is None and qb["passing_yards"]["value"] == 240.0 and qb["rushing_yards"]
    assert qb["anytime_td"] == {"value": 0.3, "sub": "1.6 pass TD"}
    assert wr["passing_yards"] is None and wr["rushing_yards"] is None and wr["receiving_yards"]["hi"] == 90.0
    assert players["wr1"]["event"] == "PHI vs DAL" and players["wr1"]["detail"] == "WR, PHI"
    assert [c["key"] for c in out["player_columns"]][-1] == "anytime_td" and out["player_sort"] == "receiving_yards"
    # the player model got this run's game projection and the ESPN reports
    (_, _, kw), = stubs["players"]
    assert kw["game_proj"] is not None and kw["espn_reports"] and kw["refresh"] is False


def test_run_slate_without_players_or_espn(stubs, monkeypatch):
    monkeypatch.setattr(live, "fetch_espn_injury_reports", lambda: pytest.fail("ESPN fetched with espn off"))
    out = engine().run_slate("nfl", {**OPTS, "season": 2026, "week": 5, "players": False,
                                                       "espn": False})
    assert out["players"] is None and stubs["players"] == []


def test_run_slate_store_goes_through_the_pipeline(stubs):
    out = engine().run_slate("nfl", {**OPTS, "season": 2026, "week": 5, "store": True,
                                                       "players": False})
    assert stubs["run_week"] == [(2026, 5, False, True)] and stubs["project_week"] == []
    assert any("Stored 2 games, 4 player rows and 1 new model picks" in n for n in out["notes"])
    assert out["players"]                                               # a stored run always has players


def test_run_slate_rejects_a_week_not_in_the_schedule(stubs):
    with pytest.raises(ProjectionError, match="No NFL games in the schedule for 2026 week 9"):
        engine().run_slate("nfl", {**OPTS, "season": 2026, "week": 9})


def test_game_run_reuses_the_slate_run(stubs):
    eng = engine(clock=lambda: 1000.0)
    slate = eng.run_slate("nfl", {**OPTS, "season": 2026, "week": 5})
    stubs["players"].clear()
    phi = slate["games"][0]
    out = eng.run_game("nfl", phi, {"sims": 2000})
    json.dumps(out)
    assert stubs["players"] == []                                       # no second player projection
    [sim] = stubs["sim"]
    assert (sim["season"], sim["week"], sim["team"], sim["n"], sim["refresh"]) == (2026, 5, "PHI", 2000, False)
    assert set(sim["proj"]["player_id"]) == {"qb1", "wr1", "ol1", "dal1"}
    assert out["title"] == "DAL @ PHI — 2,000 simulated games"
    kinds = [s["kind"] for s in out["sections"]]
    assert kinds == ["players", "players", "table"]                    # away, home, stacks
    assert out["sections"][0]["title"].startswith("Dallas Cowboys")
    home_rows = out["sections"][1]["rows"]
    assert home_rows[0]["player_id"] == "qb1" and home_rows[0]["cells"]["passing_yards"]["lo"] is not None
    assert all(r["player_id"] != "ol1" for r in home_rows)             # below the sim display gate
    stacks = out["sections"][2]["rows"]
    assert stacks[0]["outcome"].startswith("QB1 ") and stacks[0]["joint"].endswith("%")
    assert "this page's" in out["notes"][0]


def test_game_run_reprojects_after_the_cache_expires(stubs):
    now = [1000.0]
    eng = engine(clock=lambda: now[0])
    slate = eng.run_slate("nfl", {**OPTS, "season": 2026, "week": 5})
    stubs["players"].clear()
    now[0] += nfl.CACHE_TTL_S + 1
    out = eng.run_game("nfl", slate["games"][0], {"sims": 1000})
    [(season, week, kw)] = stubs["players"]
    assert (season, week, kw["refresh"]) == (2026, 5, False) and kw["espn_reports"]
    assert "fresh player projection" in out["notes"][0]


def test_game_run_parses_the_week_from_the_game_id(stubs):
    eng = engine()
    out = eng.run_game("nfl", {"game_id": "2026_05_DAL_PHI", "home": "PHI", "away": "DAL"}, {"sims": 1000})
    assert stubs["sim"][0]["week"] == 5 and out["sections"]
    with pytest.raises(ProjectionError, match="no season/week"):
        eng.run_game("nfl", {"game_id": "x", "home": "PHI", "away": "DAL"}, {"sims": 1000})
    with pytest.raises(ProjectionError, match="NYG has no projected game"):
        eng.run_game("nfl", {"game_id": "2026_05_DAL_NYG", "home": "NYG", "away": "DAL"}, {"sims": 1000})


def test_cache_keeps_the_most_recent_weeks(stubs, monkeypatch):
    monkeypatch.setattr(data, "load_games", lambda *a, **k: pd.DataFrame(
        {"season": [2026] * 6, "week": list(range(1, 7)), "gameday": pd.to_datetime(["2026-10-11"] * 6),
         "home_score": [np.nan] * 6}))
    t = [0.0]

    def clock():
        t[0] += 1
        return t[0]

    eng = engine(clock=clock)
    for w in range(1, 7):
        eng.run_slate("nfl", {**OPTS, "season": 2026, "week": w, "players": False})
    assert sorted(w for _, w in eng._runs) == [3, 4, 5, 6]


def test_stored_view(stubs):
    eng = engine()
    empty = eng.stored("nfl", {})
    assert empty["games"] == [] and "No stored NFL projections" in empty["notes"][0]

    conn = store.connect()
    early = datetime(2026, 10, 1, tzinfo=UTC)
    store.log_games(conn, _games(), "v1", now=early)
    store.log_players(conn, _players(), "v1", now=early)
    store.resolve(conn, pd.DataFrame([{"game_id": "2026_05_DAL_PHI", "home_score": 27.0, "away_score": 17.0,
                                       "spread_line": 2.5, "total_line": 44.0}]),
                  pd.DataFrame([{"game_id": "2026_05_DAL_PHI", "player_id": "wr1", "receptions": 6,
                                 "receiving_yards": 120.0, "rushing_yards": 0.0, "passing_yards": 0.0,
                                 "rushing_tds": 0, "receiving_tds": 1, "passing_tds": 0}]),
                  ready_games={"2026_05_DAL_PHI"})
    conn.close()

    out = eng.stored("nfl", {"season": "2026", "week": "5"})
    json.dumps(out)
    assert out["source"] == "stored" and out["periods"] == [
        {"key": "2026-5", "label": "2026 · Week 5", "params": {"season": 2026, "week": 5}}]
    phi = next(g for g in out["games"] if g["game_id"] == "2026_05_DAL_PHI")
    assert (phi["actual_home"], phi["actual_away"]) == (27.0, 17.0) and phi["kickoff"] == "2026-10-11T17:00+00:00"
    wr = next(p for p in out["players"] if p["player_id"] == "wr1")
    assert wr["cells"]["receiving_yards"]["result"] == "actual 120" and wr["cells"]["receiving_yards"]["hit"] is False
    assert wr["cells"]["anytime_td"]["result"] == "scored 1" and wr["cells"]["anytime_td"]["hit"] is True
    dnp = next(p for p in out["players"] if p["player_id"] == "qb1")
    assert dnp["dimmed"] and dnp["note"] == "did not play"            # graded game, no stat row
    assert out["summary"][0].startswith("2026 tracked over 1 games: margin MAE")
    assert eng.stored("nfl", {})["period"] == "2026-5"                 # latest week by default
    with pytest.raises(ProjectionError, match="season must be a whole number"):
        eng.stored("nfl", {"season": "x", "week": "5"})


def _schedule_with_played_week():
    return pd.DataFrame({
        "season": [2026, 2026, 2026], "week": [4, 5, 5],
        "gameday": pd.to_datetime(["2026-10-04", "2026-10-11", "2026-10-11"]),
        "home_score": [21.0, np.nan, np.nan],
    })


def test_espn_injuries_are_never_applied_to_a_played_week(stubs, monkeypatch):
    """The ESPN feed lists today's injuries: applied to Week 4 it would prune players who played."""
    monkeypatch.setattr(data, "load_games", lambda *a, **k: _schedule_with_played_week())
    monkeypatch.setattr(live, "fetch_espn_injury_reports", lambda: pytest.fail("ESPN fetched for a played week"))
    eng = engine()
    out = eng.run_slate("nfl", {**OPTS, "season": 2026, "week": 4})
    (_, _, kw), = stubs["players"]
    assert kw["espn_reports"] is None and "ESPN injuries skipped" in out["notes"][0]

    eng.run_slate("nfl", {**OPTS, "season": 2026, "week": 4, "store": True})
    assert stubs["run_week"][-1] == (2026, 4, False, False)            # store path gets espn=False too

    stubs["players"].clear()
    detail = engine().run_game("nfl", {"game_id": "2026_04_DAL_PHI", "home": "PHI", "away": "DAL"}, {"sims": 1000})
    (_, _, kw), = stubs["players"]
    assert kw["espn_reports"] is None and "ESPN injuries skipped for a played week" in detail["notes"][0]


def test_espn_injuries_apply_to_the_next_unplayed_week(stubs, monkeypatch):
    monkeypatch.setattr(data, "load_games", lambda *a, **k: _schedule_with_played_week())
    out = engine().run_slate("nfl", {**OPTS, "season": 2026, "week": 5})
    (_, _, kw), = stubs["players"]
    assert kw["espn_reports"] and not any("ESPN injuries skipped" in n for n in out["notes"])
