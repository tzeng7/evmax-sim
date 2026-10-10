#!/usr/bin/env python3
"""Validate the NFL joint game simulation (evmax.nfl_projections.simulate) on a held-out season.

Parameters are fitted on --fit seasons (default 2019-2024) only. The player
model projects --season (default 2025) walk-forward; each team-game is then
simulated --sims times and compared with the official box scores:

  marginals   : share of results below the simulated p10 / at or below p90
                (ideal 10% / 90%) and median MAE next to the per-player model's
                medians, per stat, on the yardage harness's populations
  identity    : the starting QB's passing yards equal his receivers' yards
                (plus unprojected receivers) in every simulation
  correlation : across team-games, the realized correlation of residuals for
                QB passing yards vs the top projected receiver's yards, the top
                two receivers, and QB passing vs the lead back's rushing yards,
                next to the simulation's average within-game correlation

Data: EVMAX_NFL_PROJ_DATA=<cache dir>.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evmax.nfl_projections import data, player_games, simulate, td_model, team_games  # noqa: E402
from evmax.nfl_projections.player_model import PlayerModelConfig  # noqa: E402
from evmax.nfl_projections.player_model import walk_forward as model_walk_forward  # noqa: E402

SEASONS_LOADED = range(2013, 2027)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--fit", default="2019-2024")
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--save", type=Path, help="write the per-player sim summary parquet here")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.fit.split("-"))

    tg = team_games.load_team_games(SEASONS_LOADED)
    pg = player_games.load_player_games(SEASONS_LOADED)
    games = data.load_games()
    injuries = pd.concat([data.load_injuries(s) for s in SEASONS_LOADED], ignore_index=True)
    rz = td_model.load_rz_usage(SEASONS_LOADED)

    fit_pg = pg[pg["season"].between(lo, hi) & (pg["season_type"] == "REG")]
    params = simulate.fit_sim_params(fit_pg)
    print(f"params (fit {lo}-{hi}): {params}")

    proj = model_walk_forward(tg, pg, games, [args.season], PlayerModelConfig(), injuries=injuries, rz=rz)
    proj = proj[proj["season_type"] == "REG"].copy()
    rng = np.random.default_rng(7)
    rows, pairs, identity_ok = [], [], True
    for (gid, team), grp in proj.groupby(["game_id", "team"]):
        grp = grp.reset_index(drop=True)
        f = grp.iloc[0]
        sims = simulate.simulate_team(grp, f["exp_team_targets"], f["exp_team_carries"], f["exp_team_rush_tds"],
                                      f["exp_team_rec_tds"], params, n=args.sims, rng=rng)
        qb = grp.index[grp["is_starting_qb"]].tolist()
        if qb:
            # identity: QB yards >= every receiver total (equality includes the unprojected slot)
            identity_ok &= bool(np.all(sims["passing_yards"][:, qb[0]] + 1e-6 >= sims["receiving_yards"].sum(axis=1)))
        summ = simulate.summarize(grp, sims)
        rows.append(pd.concat([grp, summ.drop(columns="player_id")], axis=1))
        rec_order = grp["proj_targets"].sort_values(ascending=False).index.tolist()
        rb = grp["proj_carries"].idxmax() if (grp["proj_carries"] > 0).any() else None
        if qb and len(rec_order) >= 2:
            q, w1, w2 = qb[0], rec_order[0], rec_order[1]
            sc = {
                "qb_wr1": _corr(sims["passing_yards"][:, q], sims["receiving_yards"][:, w1]),
                "wr1_wr2": _corr(sims["receiving_yards"][:, w1], sims["receiving_yards"][:, w2]),
                "qb_rb1": _corr(sims["passing_yards"][:, q], sims["rushing_yards"][:, rb]) if rb is not None else np.nan,
            }
            res = {
                "qb": grp.at[q, "passing_yards"] - summ.at[q, "sim_passing_yards"],
                "wr1": grp.at[w1, "receiving_yards"] - summ.at[w1, "sim_receiving_yards"],
                "wr2": grp.at[w2, "receiving_yards"] - summ.at[w2, "sim_receiving_yards"],
                "rb1": (grp.at[rb, "rushing_yards"] - summ.at[rb, "sim_rushing_yards"]) if rb is not None else np.nan,
            }
            pairs.append({**{f"sim_{k}": v for k, v in sc.items()}, **{f"res_{k}": v for k, v in res.items()}})
    r = pd.concat(rows, ignore_index=True)
    pr = pd.DataFrame(pairs)

    # populations: same as the yardage harness (naive = last 8 played games before the week)
    hist = pg.sort_values("gameday")
    cut = games.groupby(["season", "week"])["gameday"].min().rename("cutoff")
    r = r.join(cut, on=["season", "week"])
    nv = []
    for cutoff, wk in r.groupby("cutoff"):
        h = hist[hist["gameday"] < cutoff].groupby("player_id").tail(8).groupby("player_id")[["targets", "carries"]].mean()
        nv.append(wk.join(h.add_prefix("nv_"), on="player_id"))
    r = pd.concat(nv, ignore_index=True)
    if args.save:
        r.to_parquet(args.save, index=False)
    pops = {"receptions": r["nv_targets"] >= 1.5, "receiving_yards": r["nv_targets"] >= 1.5,
            "rushing_yards": r["nv_carries"] >= 2.0, "passing_yards": r["is_starting_qb"]}
    print(f"\nmarginals, {args.season} (n team-games {r.groupby(['game_id', 'team']).ngroups}):")
    for st, m in pops.items():
        d = r[m]
        below = (d[st] < d[f"sim_p10_{st}"]).mean() * 100
        upto = (d[st] <= d[f"sim_p90_{st}"]).mean() * 100
        mae_sim = (d[f"sim_{st}"] - d[st]).abs().mean()
        mae_ana = (d[f"proj_{st}"] - d[st]).abs().mean()
        print(f"  {st:16s} n={len(d):5d}  <p10 {below:5.1f}%  <=p90 {upto:5.1f}%  "
              f"median MAE sim {mae_sim:6.2f} vs per-player model {mae_ana:6.2f}")
    if "sim_p_anytime_td" in r:
        a = r[pops["receiving_yards"] | pops["rushing_yards"]]
        y = ((a["rushing_tds"] + a["receiving_tds"]) >= 1).astype(float)
        print(f"  anytime TD brier sim {((a['sim_p_anytime_td'] - y) ** 2).mean():.4f} vs model "
              f"{((a['p_anytime_td'] - y) ** 2).mean():.4f}")
    print(f"\nidentity (QB passing yards >= sum of projected receivers in every sim): {identity_ok}")
    print(f"\ncorrelations across {len(pr)} team-games (realized residual corr vs mean simulated corr):")
    for name, a, b in (("QB pass yds ~ WR1 rec yds", "qb", "wr1"), ("WR1 ~ WR2 rec yds", "wr1", "wr2"),
                       ("QB pass yds ~ RB1 rush yds", "qb", "rb1")):
        d = pr[[f"res_{a}", f"res_{b}"]].dropna()
        key = {"qb|wr1": "sim_qb_wr1", "wr1|wr2": "sim_wr1_wr2", "qb|rb1": "sim_qb_rb1"}[f"{a}|{b}"]
        realized = _corr(d[f"res_{a}"].to_numpy(), d[f"res_{b}"].to_numpy())
        se = (1 - realized ** 2) / np.sqrt(max(len(d) - 3, 1))
        print(f"  {name:28s} realized {realized:+.3f} (±{1.96 * se:.3f})  simulated {pr[key].mean():+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
