"""evmax.nfl_projections player model: player-game table, usage/efficiency state,
distribution fits, medians and ranges, script-conditioned volume, live roster."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from evmax.nfl_projections import live, player_games
from evmax.nfl_projections.player_model import (
    PlayerModelConfig, curve_quantile, fit_dispersion, fit_negbin_k, fit_pass_sd, fit_player_state,
    fit_quantile_curve, gamma_median, negbin_median, predict_volume, project_players,
    team_volume_rows, volume_combiners, walk_forward,
)

TEAMS = ["AAA", "BBB", "CCC", "DDD"]
# per-team skill players: (suffix, position, target share, carry share, ypt, ypc)
ROLES = [("WR1", "WR", 0.30, 0.0, 9.0, 0.0), ("WR2", "WR", 0.15, 0.0, 7.0, 0.0),
         ("RB1", "RB", 0.10, 0.60, 6.0, 4.5), ("TE1", "TE", 0.20, 0.0, 7.5, 0.0),
         ("QB1", "QB", 0.0, 0.10, 0.0, 5.0)]


def _synthetic(seasons=(2020, 2021, 2022), seed=0):
    """Round-robin league with team-game rows (for the game model) and player-game rows."""
    rng = np.random.default_rng(seed)
    off = {"AAA": 4.0, "BBB": 1.0, "CCC": -1.0, "DDD": -4.0}
    deff = {"AAA": -3.0, "BBB": 0.0, "CCC": 1.0, "DDD": 2.0}
    tg, games, pg = [], [], []
    for season in seasons:
        day = pd.Timestamp(f"{season}-09-10")
        week = 1
        for h in TEAMS:
            for a in TEAMS:
                if h == a:
                    continue
                gid = f"{season}_{week:02d}_{a}_{h}"
                hp = 22 + 2 + off[h] + deff[a] + 3 * rng.normal()
                ap = 22 + off[a] + deff[h] + 3 * rng.normal()
                for team, opp, home, pts, pa in ((h, a, 1, hp, ap), (a, h, 0, ap, hp)):
                    tg.append({"game_id": gid, "season": season, "week": week, "season_type": "REG",
                               "gameday": day, "team": team, "opp": opp, "home": home, "neutral": 0,
                               "points_for": pts, "points_against": pa, "plays": 60, "comp_plays": 50,
                               "epa_pp": pts / 100.0, "sr": 0.4 + pts / 1000.0, "starter_id": f"{team}-QB1",
                               "starter_epa": 0.05, "starter_dropbacks": 35, "first_qb_id": f"{team}-QB1"})
                    team_tgt, team_car = 34 + rng.integers(-4, 5), 26 + rng.integers(-4, 5)
                    for suffix, pos, ts, cs, ypt, ypc in ROLES:
                        tgt = int(round(team_tgt * ts)); car = int(round(team_car * cs))
                        att = 36 if pos == "QB" else 0
                        pg.append({"game_id": gid, "season": season, "week": week, "season_type": "REG",
                                   "gameday": day, "team": team, "opp": opp, "player_id": f"{team}-{suffix}",
                                   "player_display_name": f"{team} {suffix}", "position": pos,
                                   "offense_snaps": 50.0, "offense_pct": 0.8, "targets": tgt,
                                   "receptions": round(tgt * 0.65), "receiving_yards": tgt * ypt,
                                   "receiving_tds": 0, "receiving_air_yards": 0, "carries": car,
                                   "rushing_yards": car * ypc, "rushing_tds": 0, "attempts": att,
                                   "completions": att * 0.65, "passing_yards": att * 7.0, "passing_tds": 0,
                                   "sacks_suffered": 0, "team_targets": team_tgt, "team_carries": team_car,
                                   "team_attempts": 36})
                games.append({"game_id": gid, "season": season, "week": week, "game_type": "REG",
                              "gameday": day, "home_team": h, "away_team": a, "home_score": hp,
                              "away_score": ap, "location": "Home", "roof": "outdoors", "wind": 5.0,
                              "temp": 60.0, "div_game": 0, "spread_line": hp - ap, "total_line": hp + ap})
                day += pd.Timedelta(days=7)
                week += 1
    return pd.DataFrame(tg), pd.DataFrame(games), pd.DataFrame(pg)


# ── player-game table ─────────────────────────────────────────────────────────

def test_build_player_games_keeps_zero_stat_players_and_team_totals():
    weekly = pd.DataFrame([
        {"game_id": "G1", "player_id": "p1", "player_display_name": "One", "position": "WR", "team": "AAA",
         **{s: 0.0 for s in player_games.STATS}, "targets": 8.0, "receptions": 5.0, "receiving_yards": 60.0},
        {"game_id": "G1", "player_id": "d1", "player_display_name": "Def", "position": "LB", "team": "AAA",
         **{s: 0.0 for s in player_games.STATS}},                    # defender: no offensive stat
    ])
    snaps = pd.DataFrame([
        {"game_id": "G1", "player_id": "p1", "player": "One", "position": "WR", "team": "AAA",
         "offense_snaps": 50.0, "offense_pct": 0.8},
        {"game_id": "G1", "player_id": "p2", "player": "Two", "position": "TE", "team": "AAA",
         "offense_snaps": 20.0, "offense_pct": 0.3},                   # played, recorded nothing
        {"game_id": "G1", "player_id": None, "player": "Unmapped", "position": "WR", "team": "AAA",
         "offense_snaps": 10.0, "offense_pct": 0.1},
    ])
    games = pd.DataFrame([{"game_id": "G1", "season": 2024, "week": 1, "game_type": "REG",
                           "gameday": pd.Timestamp("2024-09-08"), "home_team": "AAA", "away_team": "BBB"}])
    pg = player_games.build_player_games(weekly, snaps, games).set_index("player_id")
    assert set(pg.index) == {"p1", "p2"}
    assert pg.loc["p2", "targets"] == 0 and pg.loc["p2", "offense_snaps"] == 20
    assert pg.loc["p1", "team_targets"] == 8 and pg.loc["p1", "opp"] == "BBB"


# ── state, distributions, projection ─────────────────────────────────────────

def test_fit_player_state_recovers_stable_usage():
    tg, games, pg = _synthetic(seasons=(2021,))
    vol = team_volume_rows(pg, tg)
    st = fit_player_state(pg, vol, pd.Timestamp("2022-01-01"), PlayerModelConfig())
    u = st.usage
    assert u.loc["AAA-WR1", "tgt_share"] == pytest.approx(0.30, abs=0.02)
    assert u.loc["AAA-WR1", "tgt_share"] > u.loc["AAA-WR2", "tgt_share"] > 0
    assert u.loc["AAA-RB1", "car_share"] == pytest.approx(0.60, abs=0.03)
    assert u.loc["AAA-QB1", "att_share"] == pytest.approx(1.0, abs=0.01)
    assert u.loc["AAA-WR1", "ypt"] == pytest.approx(9.0, rel=0.1)


def _players_from(dist_samples):
    rows = []
    for pid, values in dist_samples.items():
        for v in values:
            rows.append({"player_id": pid, "receiving_yards": v, "rushing_yards": v, "receptions": v,
                         "attempts": 30.0, "passing_yards": v})
    return pd.DataFrame(rows)


def test_dispersion_fits_recover_generating_parameters():
    rng = np.random.default_rng(3)
    theta = 20.0
    gam = {f"g{i}": rng.gamma(shape=mu / theta, scale=theta, size=400) for i, mu in enumerate([20, 40, 60, 80])}
    assert fit_dispersion(_players_from(gam))["receiving_yards"] == pytest.approx(theta, rel=0.1)
    k = 8.0
    nb = {f"n{i}": rng.negative_binomial(k, k / (k + mu), size=400).astype(float) for i, mu in enumerate([2, 4, 6])}
    assert fit_negbin_k(_players_from(nb)) == pytest.approx(k, rel=0.25)
    sd = {f"q{i}": rng.normal(250, 70, size=300) for i in range(3)}
    assert fit_pass_sd(_players_from(sd)) == pytest.approx(70, rel=0.1)


def test_quantile_curve_brackets_the_mean():
    rng = np.random.default_rng(4)
    data = {f"g{i}": rng.gamma(shape=mu / 20, scale=20, size=60) for i, mu in enumerate(np.linspace(10, 90, 60))}
    pg = _players_from(data)
    lo, hi = fit_quantile_curve(pg, "receiving_yards", 0.1), fit_quantile_curve(pg, "receiving_yards", 0.9)
    m = np.array([30.0, 70.0])
    assert np.all(curve_quantile(m, lo) < m) and np.all(curve_quantile(m, hi) > m)
    assert curve_quantile(np.array([0.0]), lo)[0] == 0.0


def test_medians_match_scipy_and_handle_zero():
    assert gamma_median(np.array([60.0]), 20.0)[0] == pytest.approx(stats.gamma.ppf(0.5, 3.0, scale=20.0))
    assert gamma_median(np.array([60.0]), 20.0)[0] < 60.0          # right-skewed: median below mean
    assert gamma_median(np.array([0.0]), 20.0)[0] == 0.0
    med = negbin_median(np.array([4.6]), 8.0)[0]
    assert med == int(med) and med == stats.nbinom.ppf(0.5, 8.0, 8.0 / 12.6)


def test_project_players_uses_volume_override_and_gates_passing():
    tg, games, pg = _synthetic(seasons=(2021,))
    vol = team_volume_rows(pg, tg)
    st = fit_player_state(pg, vol, pd.Timestamp("2022-01-01"), PlayerModelConfig())
    roster = pd.DataFrame([
        {"game_id": "X", "team": "AAA", "opp": "BBB", "home": 1, "player_id": "AAA-WR1", "position": "WR",
         "is_starting_qb": False},
        {"game_id": "X", "team": "AAA", "opp": "BBB", "home": 1, "player_id": "AAA-QB1", "position": "QB",
         "is_starting_qb": True},
        {"game_id": "X", "team": "AAA", "opp": "BBB", "home": 1, "player_id": "rookie", "position": "WR",
         "is_starting_qb": False},
    ])
    base = project_players(st, roster).set_index("player_id")
    assert "rookie" not in base.index                                # no history -> not projected
    assert base.loc["AAA-QB1", "proj_passing_yards"] > 0 and base.loc["AAA-WR1", "proj_passing_yards"] == 0
    assert base.loc["AAA-WR1", "p10_receiving_yards"] < base.loc["AAA-WR1", "proj_receiving_yards"] \
        < base.loc["AAA-WR1", "p90_receiving_yards"]
    tv = pd.DataFrame({"team_targets": [68.0], "team_carries": [26.0], "team_attempts": [36.0]},
                      index=pd.MultiIndex.from_tuples([("X", "AAA")], names=["game_id", "team"]))
    boosted = project_players(st, roster, tv).set_index("player_id")
    assert boosted.loc["AAA-WR1", "proj_targets"] == pytest.approx(
        base.loc["AAA-WR1", "proj_targets"] * 68.0 / st.fits["team_targets"].expect("AAA", "BBB", 1), rel=1e-6)


def test_volume_combiners_recover_linear_script_effect():
    rng = np.random.default_rng(5)
    n = 400
    vft = pd.DataFrame({"game_id": [f"g{i}" for i in range(n)], "team": "AAA",
                        "proj_margin": rng.normal(0, 6, n), "proj_total": rng.normal(45, 5, n)})
    for m in ("team_targets", "team_carries", "team_attempts"):
        vft["exp_" + m] = rng.normal(30, 3, n)
        vft[m] = 2.0 + 0.9 * vft["exp_" + m] + 0.1 * vft["proj_margin"] + 0.05 * vft["proj_total"]
    coefs = volume_combiners(vft)
    assert coefs["team_targets"] == pytest.approx([2.0, 0.9, 0.1, 0.05], abs=1e-6)
    pred = predict_volume(vft.head(3), coefs)
    assert np.allclose(pred["team_carries"].to_numpy(), vft["team_carries"].head(3).to_numpy())


def test_player_walk_forward_is_leak_free():
    tg, games, pg = _synthetic()
    base = walk_forward(tg, pg, games, [2022])
    target_week = 6
    g = games[(games["season"] == 2022) & (games["week"] == target_week)].iloc[0]
    pg2 = pg.copy()
    mask = pg2["game_id"] == g.game_id
    pg2.loc[mask, ["targets", "receiving_yards", "carries", "rushing_yards"]] = 50.0
    after = walk_forward(tg, pg2, games, [2022])
    key = ["game_id", "player_id"]
    m = base.merge(after, on=key, suffixes=("", "_p"))
    early = m["week"] <= target_week
    assert np.allclose(m.loc[early, "proj_receiving_yards"], m.loc[early, "proj_receiving_yards_p"])
    assert not np.allclose(m.loc[~early, "proj_receiving_yards"], m.loc[~early, "proj_receiving_yards_p"])


# ── live roster ──────────────────────────────────────────────────────────────

def test_active_roster_drops_out_players_and_adds_schedule_qb():
    pg = pd.DataFrame([
        {"game_id": "a", "team": "AAA", "player_id": "wr", "player_display_name": "WR", "position": "WR",
         "gameday": pd.Timestamp("2026-09-20")},
        {"game_id": "a", "team": "AAA", "player_id": "hurt", "player_display_name": "Hurt", "position": "WR",
         "gameday": pd.Timestamp("2026-09-20")},
        {"game_id": "old", "team": "AAA", "player_id": "gone", "player_display_name": "Gone", "position": "TE",
         "gameday": pd.Timestamp("2026-08-01")},
    ] + [{"game_id": f"r{i}", "team": "AAA", "player_id": "wr", "player_display_name": "WR", "position": "WR",
          "gameday": pd.Timestamp("2026-09-21") + pd.Timedelta(days=i)} for i in range(3)])
    inj = pd.DataFrame([{"week": 5, "team": "AAA", "gsis_id": "hurt", "report_status": "Out"}])
    wk = pd.DataFrame([{"game_id": "g5", "home_team": "AAA", "away_team": "BBB", "location": "Home",
                        "home_qb_id": "newqb", "away_qb_id": None}])
    r = live.active_roster(pg, inj, wk, pd.Timestamp("2026-10-11"), 5)
    aaa = r[r["team"] == "AAA"]
    assert set(aaa["player_id"]) == {"wr", "newqb"}               # injured out, old player aged out
    assert aaa.set_index("player_id").loc["newqb", "is_starting_qb"]
