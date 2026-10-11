"""evmax.nfl_projections.simulate: joint box-score simulation identities, means and parameter fits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evmax.nfl_projections import simulate


def _team():
    return pd.DataFrame([
        {"player_id": "qb", "tgt_share": 0.0, "car_share": 0.10, "catch_rate": 0.5, "ypt": 0.0, "ypc": 5.0,
         "rush_xtd_share": 0.10, "rec_xtd_share": 0.0, "is_starting_qb": True},
        {"player_id": "wr1", "tgt_share": 0.30, "car_share": 0.0, "catch_rate": 0.65, "ypt": 9.0, "ypc": 0.0,
         "rush_xtd_share": 0.0, "rec_xtd_share": 0.35, "is_starting_qb": False},
        {"player_id": "wr2", "tgt_share": 0.20, "car_share": 0.0, "catch_rate": 0.6, "ypt": 8.0, "ypc": 0.0,
         "rush_xtd_share": 0.0, "rec_xtd_share": 0.25, "is_starting_qb": False},
        {"player_id": "rb", "tgt_share": 0.10, "car_share": 0.60, "catch_rate": 0.8, "ypt": 6.0, "ypc": 4.3,
         "rush_xtd_share": 0.65, "rec_xtd_share": 0.10, "is_starting_qb": False},
    ])


def test_simulate_team_identities_and_means():
    players = _team()
    params = simulate.SimParams(target_kappa=50, carry_kappa=20, team_vol_cv=0.1, rec_shape=1.5, rush_sd=5.0,
                                pass_eff_sd=0.05, rush_eff_sd=0.05)
    s = simulate.simulate_team(players, 34.0, 26.0, 0.9, 1.5, params, n=20000, rng=np.random.default_rng(1))
    assert s["receptions"].shape == (20000, 4)
    qb_pass = s["passing_yards"][:, 0]
    assert np.all(qb_pass + 1e-9 >= s["receiving_yards"].sum(axis=1))          # QB = receivers + unprojected
    assert np.all(s["passing_yards"][:, 1:] == 0)
    assert s["passing_tds"][:, 0].mean() == pytest.approx(1.5, rel=0.05)         # team receiving TDs
    assert np.all(s["receptions"] <= 34 * 3)                                     # sanity
    # means: WR1 targets ~ 34 * 0.30 (shares renormalized with a 0.40 "other" slot)
    wr1_rec = s["receptions"][:, 1].mean()
    assert wr1_rec == pytest.approx(34 * 0.30 * 0.65, rel=0.05)
    assert s["receiving_yards"][:, 1].mean() == pytest.approx(34 * 0.30 * 9.0, rel=0.06)
    assert s["rushing_yards"][:, 3].mean() == pytest.approx(26 * 0.60 * 4.3, rel=0.05)
    # QB and his top receiver are positively correlated; a non-QB has no passing yards
    assert np.corrcoef(qb_pass, s["receiving_yards"][:, 1])[0, 1] > 0.3
    summ = simulate.summarize(players, s).set_index("player_id")
    assert summ.loc["wr1", "sim_p10_receiving_yards"] < summ.loc["wr1", "sim_receiving_yards"] < \
        summ.loc["wr1", "sim_p90_receiving_yards"]
    assert 0 < summ.loc["rb", "sim_p_anytime_td"] < 1


def test_joint_probability_exceeds_independence_for_a_stack():
    players = _team()
    s = simulate.simulate_team(players, 34.0, 26.0, 0.9, 1.5, simulate.SimParams(), n=20000,
                               rng=np.random.default_rng(2))
    q = float(np.median(s["passing_yards"][:, 0]))
    w = float(np.median(s["receiving_yards"][:, 1]))
    joint = simulate.joint_probability(s, [("passing_yards", 0, q), ("receiving_yards", 1, w)])
    indep = (s["passing_yards"][:, 0] >= q).mean() * (s["receiving_yards"][:, 1] >= w).mean()
    assert joint > indep + 0.03


def test_fit_sim_params_recovers_dirichlet_concentration():
    rng = np.random.default_rng(3)
    rows = []
    kappa = 30.0
    shares = np.array([0.3, 0.2, 0.15, 0.35])
    for season in (2023, 2024):
        for g in range(17):
            T = 35
            p = rng.dirichlet(kappa * shares)
            tg = rng.multinomial(T, p)
            for i in range(3):
                rows.append({"season": season, "game_id": f"{season}_{g}", "team": "AAA", "player_id": f"p{i}",
                             "position": "WR", "targets": tg[i], "team_targets": T, "carries": 0,
                             "team_carries": 25, "receptions": tg[i] * 0.6, "receiving_yards": tg[i] * 8.0,
                             "rushing_yards": 0.0})
    params = simulate.fit_sim_params(pd.DataFrame(rows))
    assert params.target_kappa == pytest.approx(kappa, rel=0.5)


def test_qb_stacks_pairs_the_qb_with_his_top_targets():
    players = _team().assign(
        player_display_name=["QB One", "WR One", "WR Two", "RB One"],
        proj_targets=[0.0, 10.0, 7.0, 3.0],
    )
    s = simulate.simulate_team(players, 34.0, 26.0, 0.9, 1.5, simulate.SimParams(), n=20000,
                               rng=np.random.default_rng(4))
    stacks = simulate.qb_stacks(players, s, receivers=2)
    assert [k["receiver"] for k in stacks] == ["WR One", "WR Two"]          # by projected targets, QB excluded
    top = stacks[0]
    assert top["qb"] == "QB One"
    assert top["qb_passing_yards"] == pytest.approx(float(np.median(s["passing_yards"][:, 0])))
    assert top["receiver_receiving_yards"] == pytest.approx(float(np.median(s["receiving_yards"][:, 1])))
    assert top["joint"] == pytest.approx(simulate.joint_probability(
        s, [("passing_yards", 0, top["qb_passing_yards"]), ("receiving_yards", 1, top["receiver_receiving_yards"])]))
    assert top["joint"] > top["independent"] + 0.03                         # the stack is positively correlated


def test_qb_stacks_empty_without_a_starting_qb():
    players = _team().assign(is_starting_qb=False, player_display_name="x", proj_targets=1.0)
    s = simulate.simulate_team(players, 30.0, 25.0, 0.8, 1.2, simulate.SimParams(), n=500,
                               rng=np.random.default_rng(5))
    assert simulate.qb_stacks(players, s) == []


# ── participation: who plays is drawn per simulation ─────────────────────────

def test_participation_draw_splits_usage_among_the_players_who_play():
    players = _team().assign(p_active=[1.0, 1.0, 0.4, 1.0])            # WR2 plays 40% of the time
    s = simulate.simulate_team(players, 34.0, 26.0, 0.9, 1.5, simulate.SimParams(), n=20000,
                               rng=np.random.default_rng(6))
    act = s["active"]
    assert act[:, [0, 1, 3]].all() and act[:, 2].mean() == pytest.approx(0.4, abs=0.02)
    sits = ~act[:, 2]
    assert (s["receptions"][sits, 2] == 0).all() and (s["receiving_yards"][sits, 2] == 0).all()
    assert (s["tds"][sits, 2] == 0).all()
    # WR1 gets more targets when WR2 sits (shares renormalized over the players who play)
    assert s["receptions"][sits, 1].mean() > s["receptions"][~sits, 1].mean()
    summ = simulate.summarize(players, s).set_index("player_id")
    assert summ.loc["wr2", "sim_p_active"] == pytest.approx(act[:, 2].mean())
    # WR2's line is over the games he plays in, not diluted by the games he sits
    assert summ.loc["wr2", "sim_receiving_yards"] == pytest.approx(np.median(s["receiving_yards"][~sits, 2]))
    assert summ.loc["wr2", "sim_mean_receiving_yards"] == pytest.approx(s["receiving_yards"][~sits, 2].mean())
    assert summ.loc["wr2", "sim_p_anytime_td"] == pytest.approx((s["tds"][~sits, 2] >= 1).mean())
    assert np.all(s["passing_yards"][:, 0] + 1e-9 >= s["receiving_yards"].sum(axis=1))   # identity still holds


def test_without_p_active_everyone_plays_and_nan_counts_as_playing():
    s = simulate.simulate_team(_team(), 34.0, 26.0, 0.9, 1.5, simulate.SimParams(), n=2000,
                               rng=np.random.default_rng(7))
    assert s["active"].all()
    s = simulate.simulate_team(_team().assign(p_active=[np.nan, 1.0, 1.0, 1.0]), 34.0, 26.0, 0.9, 1.5,
                               simulate.SimParams(), n=2000, rng=np.random.default_rng(7))
    assert s["active"][:, 0].all()


def test_full_participation_keeps_the_expected_shares():
    # every slot plays: expected targets match the pre-participation simulation (share / (sum + other))
    s = simulate.simulate_team(_team().assign(p_active=1.0), 34.0, 26.0, 0.9, 1.5, simulate.SimParams(),
                               n=20000, rng=np.random.default_rng(8))
    assert s["receptions"][:, 1].mean() == pytest.approx(34 * 0.30 * 0.65, rel=0.05)


def test_team_totals_include_unprojected_players():
    players = _team()
    s = simulate.simulate_team(players, 34.0, 26.0, 0.9, 1.5, simulate.SimParams(), n=20000,
                               rng=np.random.default_rng(9))
    assert np.allclose(s["team_passing_yards"], s["passing_yards"][:, 0])     # the QB throws to everyone
    assert np.all(s["team_receptions"] >= s["receptions"].sum(axis=1))
    # 30% of carries go to unprojected players at the share-weighted yards per carry
    assert s["team_rushing_yards"].mean() > s["rushing_yards"].sum(axis=1).mean() * 1.2
    assert s["team_tds"].mean() == pytest.approx(0.9 + 1.5, rel=0.05)
    tot = simulate.summarize_team(s)
    med, lo, hi = tot["passing_yards"]
    assert lo < med < hi and med == pytest.approx(float(np.median(s["team_passing_yards"])))
    assert set(tot) == {"receptions", "passing_yards", "rushing_yards", "tds"}


def test_joint_probability_and_stacks_condition_on_the_receiver_playing():
    players = _team().assign(p_active=[1.0, 0.5, 1.0, 1.0], player_display_name=["Q", "W1", "W2", "R"],
                             proj_targets=[0.0, 10.0, 7.0, 3.0])
    s = simulate.simulate_team(players, 34.0, 26.0, 0.9, 1.5, simulate.SimParams(), n=20000,
                               rng=np.random.default_rng(10))
    on = s["active"][:, 1]
    w = float(np.median(s["receiving_yards"][on, 1]))
    j = simulate.joint_probability(s, [("receiving_yards", 1, w)])
    assert j == pytest.approx((s["receiving_yards"][on, 1] >= w).mean())      # sits do not count as misses
    top = simulate.qb_stacks(players, s, receivers=1)[0]
    assert top["receiver"] == "W1" and top["receiver_receiving_yards"] == pytest.approx(w)
    assert 0.25 < top["joint"] < 0.5 and top["joint"] > top["independent"]   # both above median, correlated
