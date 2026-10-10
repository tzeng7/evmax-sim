#!/usr/bin/env python3
"""Walk-forward evaluation of NFL touchdown projections (evmax.nfl_projections.td_model).

For every week, projects each player who played using only earlier games (the
player model with red-zone usage + the pre-game injury report) and scores:

  anytime  : P(rushing + receiving TDs >= 1) — binary log loss
  passing  : the starting QB's passing TDs — Poisson log loss

Populations (fixed here, from PRE-game information only, same as the yardage
harness): anytime = naive targets/game >= 1.5 OR naive carries/game >= 2;
passing = the team's pre-game starting QB (first-dropback passer).

Baseline (fixed here; a USAGE baseline, 2026-10-10): a Poisson rate =
  anytime: the player's targets + carries per game over his last 8 played
           games x his position's TDs per touch over the prior 730 days
  passing: the QB's attempts per start over his last 8 starts (>= 10
           attempts) x the league's passing TDs per attempt over the prior 730 days
(The first version compared against last-8 TD counts with a pseudo-game; the
reviewer showed that lost even to a position constant, so it overstated skill.)

  dev     = 2019-2024 (optimized by the improvement loop)
  holdout = 2025      (reported only with --holdout; never tuned on)

Loop signal: ``td_dev_score=<mean of the anytime and passing log-loss ratios vs naive>`` (lower is better).
Data: EVMAX_NFL_PROJ_DATA=<cache dir> (see evmax/nfl_projections/data.py).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evmax.nfl_projections import data, player_games, td_model, team_games  # noqa: E402
from evmax.nfl_projections.player_model import PlayerModelConfig  # noqa: E402
from evmax.nfl_projections.player_model import walk_forward as model_walk_forward  # noqa: E402

DEV = list(range(2019, 2025))
HOLDOUT = [2025]
SEASONS_LOADED = range(2013, 2027)
NAIVE_GAMES = 8
EPS = 1e-4


def naive_rates(pg: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    hist = pg[(pg["gameday"] < cutoff) & (pg["gameday"] >= cutoff - pd.Timedelta(days=730))].sort_values("gameday")
    hist = hist.assign(tds=hist["rushing_tds"] + hist["receiving_tds"], touches=hist["targets"] + hist["carries"])
    pos = hist.groupby("position")[["tds", "touches"]].sum()
    td_per_touch = pos["tds"] / pos["touches"].clip(lower=1)
    starts = hist[hist["attempts"] >= 10]
    td_per_att = float(starts["passing_tds"].sum() / max(starts["attempts"].sum(), 1))
    last = hist.groupby("player_id").tail(NAIVE_GAMES)
    agg = last.groupby("player_id").agg(touches=("touches", "mean"), targets=("targets", "mean"),
                                        carries=("carries", "mean"), position=("position", "last"))
    rate = agg["position"].map(td_per_touch).fillna(float(hist["tds"].sum() / max(hist["touches"].sum(), 1)))
    agg["naive_lam"] = agg["touches"] * rate
    qb = starts.groupby("player_id").tail(NAIVE_GAMES).groupby("player_id")["attempts"].mean()
    agg["naive_pass_lam"] = qb.reindex(agg.index) * td_per_att
    return agg[["naive_lam", "naive_pass_lam", "targets", "carries"]].add_prefix("nv_")


def walk_forward(seasons: list[int], cfg: PlayerModelConfig) -> pd.DataFrame:
    tg = team_games.load_team_games(SEASONS_LOADED)
    pg = player_games.load_player_games(SEASONS_LOADED)
    games = data.load_games()
    injuries = pd.concat([data.load_injuries(s) for s in SEASONS_LOADED], ignore_index=True)
    rz = td_model.load_rz_usage(SEASONS_LOADED)
    proj = model_walk_forward(tg, pg, games, seasons, cfg, injuries=injuries, rz=rz)
    cut = games.groupby(["season", "week"])["gameday"].min().rename("cutoff")
    proj = proj.join(cut, on=["season", "week"])
    out = [wk.join(naive_rates(pg, cutoff), on="player_id") for cutoff, wk in proj.groupby("cutoff", sort=True)]
    r = pd.concat(out, ignore_index=True)
    r["pop_any"] = (r["nv_targets"] >= 1.5) | (r["nv_carries"] >= 2.0)
    r["pop_pass"] = r["is_starting_qb"] & r["nv_naive_pass_lam"].notna()  # QBs with a prior start
    r["scored"] = ((r["rushing_tds"] + r["receiving_tds"]) >= 1).astype(float)
    return r


def _binary_ll(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def metrics(r: pd.DataFrame) -> dict[str, dict[str, float]]:
    a = r[r["pop_any"]]
    y = a["scored"].to_numpy()
    p_model = a["p_anytime_td"].to_numpy()
    p_naive = 1 - np.exp(-a["nv_naive_lam"].to_numpy())
    any_m = {"n": float(len(a)), "ll": _binary_ll(p_model, y), "ll_naive": _binary_ll(p_naive, y),
             "brier": float(((p_model - y) ** 2).mean()), "brier_naive": float(((p_naive - y) ** 2).mean()),
             "mean_p": float(p_model.mean()), "rate": float(y.mean())}
    any_m["ratio"] = any_m["ll"] / any_m["ll_naive"]
    q = r[r["pop_pass"]]
    k = q["passing_tds"].to_numpy()
    lam_m = np.clip(q["proj_passing_tds"].to_numpy(), EPS, None)
    lam_n = np.clip(q["nv_naive_pass_lam"].to_numpy(), EPS, None)
    pass_m = {"n": float(len(q)), "ll": float(-stats.poisson.logpmf(k, lam_m).mean()),
              "ll_naive": float(-stats.poisson.logpmf(k, lam_n).mean()),
              "mean_lam": float(lam_m.mean()), "mean_actual": float(k.mean())}
    pass_m["ratio"] = pass_m["ll"] / pass_m["ll_naive"]
    return {"anytime": any_m, "passing": pass_m}


def calibration(r: pd.DataFrame) -> str:
    a = r[r["pop_any"]]
    b = pd.cut(a["p_anytime_td"], [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0])
    g = a.groupby(b, observed=True).agg(n=("scored", "size"), p=("p_anytime_td", "mean"), y=("scored", "mean"))
    return " | ".join(f"{iv}: {row.p:.3f}->{row.y:.3f} (n={int(row.n)})" for iv, row in g.iterrows())


def fmt(name: str, m: dict) -> str:
    a, p = m["anytime"], m["passing"]
    return (f"{name:8s} anytime: ll {a['ll']:.4f} vs naive {a['ll_naive']:.4f} ({a['ratio']:.4f}), "
            f"brier {a['brier']:.4f} vs {a['brier_naive']:.4f}, mean p {a['mean_p']:.3f} vs rate {a['rate']:.3f}, "
            f"n={int(a['n'])} | passing: ll {p['ll']:.4f} vs naive {p['ll_naive']:.4f} ({p['ratio']:.4f}), "
            f"mean lam {p['mean_lam']:.2f} vs {p['mean_actual']:.2f}, n={int(p['n'])}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--save", type=Path)
    args = ap.parse_args()
    cfg = PlayerModelConfig()
    seasons = DEV + (HOLDOUT if args.holdout else [])
    r = walk_forward(seasons, cfg)
    reg = r[r["season_type"] == "REG"]
    if args.save:
        r.to_parquet(args.save, index=False)
    dev = metrics(reg[reg["season"].isin(DEV)])
    print(fmt("dev", dev))
    print("  dev calibration " + calibration(reg[reg["season"].isin(DEV)]))
    if args.holdout:
        ho = reg[reg["season"].isin(HOLDOUT)]
        print(fmt("holdout", metrics(ho)))
        print("  holdout calibration " + calibration(ho))
    score = float(np.mean([dev["anytime"]["ratio"], dev["passing"]["ratio"]]))
    print(f"td_dev_score={score:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
