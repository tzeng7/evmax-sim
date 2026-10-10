"""evmax.nfl_projections touchdowns: red-zone usage, expected-TD shares, Poisson TD probabilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from evmax.nfl_projections import td_model
from evmax.nfl_projections.player_model import (
    PlayerModelConfig, fit_player_state, project_players, team_volume_rows,
)


def _pbp():
    rows = [
        # (game, play_type, rusher, receiver, yardline_100, rush_td, pass_td, two_point)
        ("g1", "run", "rb", None, 1, 1, 0, 0),
        ("g1", "run", "rb", None, 2, 0, 0, 0),
        ("g1", "run", "rb", None, 40, 0, 0, 0),
        ("g1", "pass", None, "wr", 8, 0, 1, 0),
        ("g1", "pass", None, "wr", 15, 0, 0, 0),
        ("g1", "pass", None, "wr", 2, 0, 1, 1),        # two-point attempt: excluded
        ("g1", "pass", None, None, 30, 0, 0, 0),       # throwaway: no receiver
        ("g1", "punt", None, None, 60, 0, 0, 0),
    ]
    return pd.DataFrame(rows, columns=["game_id", "play_type", "rusher_player_id", "receiver_player_id",
                                       "yardline_100", "rush_touchdown", "pass_touchdown", "two_point_attempt"])


def test_build_rz_usage_buckets_and_excludes_two_point_tries():
    rz = td_model.build_rz_usage(_pbp()).set_index("player_id")
    assert rz.loc["rb", "rush_b0"] == 2 and rz.loc["rb", "rush_td_b0"] == 1     # yardline 1-2
    assert rz.loc["rb", "rush_b4"] == 1                                         # 21+
    assert rz.loc["wr", "tgt_b2"] == 1 and rz.loc["wr", "rec_td_b2"] == 1       # 6-10
    assert rz.loc["wr", "tgt_b3"] == 1 and rz.loc["wr", "tgt_b0"] == 0          # the 2-pt try is gone
    assert rz.loc["wr", "rush_b0"] == 0 and rz.loc["rb", "tgt_b2"] == 0


def test_bucket_rates_and_expected_tds():
    rz = td_model.build_rz_usage(_pbp())
    r_rush, r_tgt = td_model.bucket_td_rates(rz)
    assert r_rush[0] == pytest.approx(0.5) and r_tgt[2] == pytest.approx(1.0) and r_tgt[3] == 0.0
    x_rush, x_rec = td_model.expected_tds(rz.set_index("player_id").loc[["rb", "wr"]], r_rush, r_tgt)
    assert x_rush[0] == pytest.approx(2 * 0.5) and x_rec[1] == pytest.approx(1.0)


def test_poisson_at_least_matches_scipy():
    lam = np.array([0.0, 0.3, 1.2])
    assert np.allclose(td_model.poisson_at_least(lam, 1), 1 - np.exp(-lam))
    assert np.allclose(td_model.poisson_at_least(lam, 2), stats.poisson.sf(1, lam))
    assert np.allclose(td_model.poisson_at_least(lam, 3), stats.poisson.sf(2, lam))


def _td_league(n_games=24, seed=1):
    """Two teams; on each, a goal-line back gets every carry inside the 5 and scores at
    the league rate, and a WR gets the red-zone targets."""
    rng = np.random.default_rng(seed)
    day = pd.Timestamp("2025-09-07")
    pg, tg, rz = [], [], []
    for i in range(n_games):
        gid = f"2025_{i:02d}_AAA_BBB"
        for team, opp, home in (("AAA", "BBB", 1), ("BBB", "AAA", 0)):
            tg.append({"game_id": gid, "team": team, "opp": opp, "home": home, "gameday": day})
            gl_carries, rz_tgts = rng.poisson(3), rng.poisson(4)
            rush_td, rec_td = rng.binomial(gl_carries, 0.4), rng.binomial(rz_tgts, 0.3)
            for pid, pos, car, tgt, rtd, ctd, rzc, rzt in (
                (f"{team}-RB", "RB", 18, 3, rush_td, 0, gl_carries, 0),
                (f"{team}-WR", "WR", 0, 9, 0, rec_td, 0, rz_tgts),
                (f"{team}-RB2", "RB", 6, 2, 0, 0, 0, 0),
            ):
                pg.append({"game_id": gid, "season": 2025, "week": i + 1, "season_type": "REG", "gameday": day,
                           "team": team, "opp": opp, "player_id": pid, "player_display_name": pid, "position": pos,
                           "targets": tgt, "receptions": tgt * 0.6, "receiving_yards": tgt * 8.0, "carries": car,
                           "rushing_yards": car * 4.0, "attempts": 0, "passing_yards": 0.0, "passing_tds": 0,
                           "rushing_tds": rtd, "receiving_tds": ctd,
                           "team_targets": 14, "team_carries": 24, "team_attempts": 34})
                row = {"game_id": gid, "player_id": pid, **{c: 0.0 for c in td_model.RZ_COLS}}
                row["rush_b1"], row["rush_td_b1"] = rzc, rtd
                row["tgt_b2"], row["rec_td_b2"] = rzt, ctd
                rz.append(row)
        day += pd.Timedelta(days=7)
    return pd.DataFrame(pg), pd.DataFrame(tg), pd.DataFrame(rz)


def test_xtd_shares_and_anytime_probabilities():
    pg, tg, rz = _td_league()
    st = fit_player_state(pg, team_volume_rows(pg, tg), pd.Timestamp("2026-06-01"),
                          PlayerModelConfig(td_share_prior_games=0.0, td_actual_weight=0.0), rz=rz)
    u = st.usage
    assert u.loc["AAA-RB", "rush_xtd_share"] == pytest.approx(1.0)      # every goal-line carry
    assert u.loc["AAA-RB2", "rush_xtd_share"] == pytest.approx(0.0)
    assert u.loc["AAA-WR", "rec_xtd_share"] == pytest.approx(1.0)
    roster = pd.DataFrame([{"game_id": "X", "team": "AAA", "opp": "BBB", "home": 1, "player_id": pid,
                            "position": pos, "is_starting_qb": False}
                           for pid, pos in (("AAA-RB", "RB"), ("AAA-RB2", "RB"), ("AAA-WR", "WR"))])
    out = project_players(st, roster).set_index("player_id")
    lam = out["proj_tds"]
    assert lam["AAA-RB"] > lam["AAA-RB2"] and lam["AAA-RB2"] == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(out["p_anytime_td"], 1 - np.exp(-lam))
    assert (out["p_two_plus_td"] < out["p_anytime_td"]).all() or (lam == 0).any()
    # the team's projected rushing TDs all land on the goal-line back
    assert lam["AAA-RB"] == pytest.approx(st.fits["team_rush_tds"].expect("AAA", "BBB", 1), rel=1e-6)


def test_actual_td_share_blend_moves_share_toward_realized_tds():
    pg, tg, rz = _td_league()
    # RB2 never touches the red zone but is credited with every rushing TD in this variant
    pg = pg.copy()
    rz = rz.copy()
    moved = pg["player_id"].str.endswith("-RB") & (pg["rushing_tds"] > 0)
    pg.loc[pg.index[moved].map(lambda i: i + 2), "rushing_tds"] = pg.loc[moved, "rushing_tds"].to_numpy()
    pg.loc[moved, "rushing_tds"] = 0
    for pid_from, pid_to in (("-RB", "-RB2"),):
        src = rz["player_id"].str.endswith(pid_from)
        dst = rz["player_id"].str.endswith(pid_to)
        rz.loc[dst, "rush_td_b1"] = rz.loc[src, "rush_td_b1"].to_numpy()
        rz.loc[src, "rush_td_b1"] = 0.0
    vol = team_volume_rows(pg, tg)
    cut = pd.Timestamp("2026-06-01")
    x_only = fit_player_state(pg, vol, cut, PlayerModelConfig(td_share_prior_games=0.0, td_actual_weight=0.0), rz=rz)
    blend = fit_player_state(pg, vol, cut, PlayerModelConfig(td_share_prior_games=0.0, td_actual_weight=0.25), rz=rz)
    assert x_only.usage.loc["AAA-RB2", "rush_xtd_share"] == pytest.approx(0.0)
    assert blend.usage.loc["AAA-RB2", "rush_xtd_share"] > 0.1
    assert blend.usage.loc["AAA-RB", "rush_xtd_share"] < x_only.usage.loc["AAA-RB", "rush_xtd_share"]
