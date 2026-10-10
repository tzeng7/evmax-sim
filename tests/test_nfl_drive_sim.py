"""evmax.nfl_projections.drive_model: drive table, zone rates / transitions, possession simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evmax.nfl_projections import drive_model as M


def test_build_drives_uses_first_scrimmage_play_and_maps_outcomes():
    rows = [
        # game, posteam, defteam, home, drive, result, yardline, play_type
        ("g", "AAA", "BBB", "AAA", 1, "Touchdown", 65, "kickoff"),      # kickoff spot is ignored
        ("g", "AAA", "BBB", "AAA", 1, "Touchdown", 75, "run"),
        ("g", "AAA", "BBB", "AAA", 1, "Touchdown", 3, "pass"),
        ("g", "BBB", "AAA", "AAA", 2, "Punt", 82, "pass"),
        ("g", "BBB", "AAA", "AAA", 2, "Punt", 70, "punt"),
        ("g", "AAA", "BBB", "AAA", 3, "Turnover", 40, "run"),
        ("g", "AAA", "BBB", "AAA", 4, "Weird", 40, "run"),               # unknown result dropped
    ]
    pbp = pd.DataFrame(rows, columns=["game_id", "posteam", "defteam", "home_team", "fixed_drive",
                                      "fixed_drive_result", "yardline_100", "play_type"])
    d = M.build_drives(pbp)
    assert list(d["outcome"]) == ["td", "punt", "turnover"]
    assert list(d["start"]) == [75, 82, 40]
    assert list(d["zone"]) == [2, 3, 0]                       # 65-79, 80+, opponent territory
    assert list(d["home"]) == [1, 0, 1]
    assert d["next_zone"].iloc[0] == 3 and d["next_team"].iloc[0] == "BBB"


def _drives(n_games=60, seed=0, strong="AAA"):
    rng = np.random.default_rng(seed)
    rows, gameday = [], {}
    day = pd.Timestamp("2025-09-07")
    teams = ["AAA", "BBB", "CCC", "DDD"]
    for g in range(n_games):
        h, a = teams[g % 4], teams[(g + 1) % 4]
        gid = f"2025_{g:03d}_{a}_{h}"
        gameday[gid] = day + pd.Timedelta(days=g)
        for k in range(22):
            team = h if k % 2 == 0 else a
            p_td = 0.45 if team == strong else 0.18
            o = rng.choice(["td", "fg", "punt", "turnover"], p=[p_td, 0.15, 0.85 - p_td - 0.1, 0.1])
            rows.append({"game_id": gid, "fixed_drive": k, "team": team, "opp": a if team == h else h,
                         "home": int(team == h), "start": 75, "zone": 2, "outcome": o})
    d = pd.DataFrame(rows)
    d["next_zone"] = d.groupby("game_id")["zone"].shift(-1)
    d["next_team"] = d.groupby("game_id")["team"].shift(-1)
    return d, pd.Series(gameday)


def test_fit_drive_state_rates_and_transitions_are_distributions():
    d, gd = _drives()
    st = M.fit_drive_state(d, gd, pd.Timestamp("2026-01-01"))
    assert np.allclose(st.base.sum(axis=1), 1.0)
    assert all(np.isclose(v.sum(), 1.0) for v in st.transition.values())
    assert st.td.off["AAA"] > st.td.off["BBB"]               # the strong offense scores more TDs than zones imply


def test_simulate_game_is_seeded_and_rewards_the_better_offense():
    d, gd = _drives()
    st = M.fit_drive_state(d, gd, pd.Timestamp("2026-01-01"))
    h1, a1 = M.simulate_game(st, "AAA", "BBB", n=3000, rng=np.random.default_rng(5))
    h2, a2 = M.simulate_game(st, "AAA", "BBB", n=3000, rng=np.random.default_rng(5))
    assert np.array_equal(h1, h2) and np.array_equal(a1, a2)
    assert h1.mean() > a1.mean() + 5
    assert (h1 >= 0).all() and (a1 >= 0).all()


def test_drive_points_feeds_the_game_model(monkeypatch):
    from evmax.nfl_projections import game_model

    d, gd = _drives()
    monkeypatch.setattr(M, "load_drives", lambda seasons, d_=None, **k: d)
    games = pd.DataFrame({"game_id": list(gd.index), "gameday": list(gd.values)})
    tg = pd.DataFrame({"season": [2025]})
    wk = pd.DataFrame([{"game_id": "X", "home_team": "AAA", "away_team": "BBB", "location": "Home"}])
    pts = game_model.drive_points(tg, games, pd.Timestamp("2026-01-01"), wk)
    assert set(pts) == {("X", "AAA"), ("X", "BBB")} and pts[("X", "AAA")] > pts[("X", "BBB")]
    assert pts == game_model.drive_points(tg, games, pd.Timestamp("2026-01-01"), wk)   # deterministic per game
