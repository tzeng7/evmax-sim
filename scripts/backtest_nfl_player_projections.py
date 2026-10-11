#!/usr/bin/env python3
"""Walk-forward evaluation of NFL player stat-line projections (evmax.nfl_projections.player_model).

For every week, projects each player who played (offensive snaps or a stat)
using only earlier games, and scores four stats against the official result:
receiving yards, receptions, rushing yards, passing yards.

Populations (fixed here, from PRE-game information only):
  receiving  : naive targets/game >= 1.5        (naive = player's last 8 played games)
  rushing    : naive carries/game >= 2
  passing    : the team's pre-game starting QB (first-dropback passer)
Naive baseline (fixed here): the player's plain mean over his last 8 played games.

  dev      = 2019-2024 (optimized by the improvement loop)
  holdout  = 2025      (reported only with --holdout; never tuned on)

Teammates of players ruled Out/Doubtful on the week's pre-game injury report
take part of their usage. --redistribute-roster-out also frees the usage of
recent players the weekly roster rules out (reserve lists, released;
``PlayerModelConfig.roster_out_redistribution``) — rejected 2026-10-10
(dev_score 0.92456 -> 0.93185).

Loop signal: ``dev_score=<mean over the 4 stats of MAE_model / MAE_naive>`` (lower is better).
Data: EVMAX_NFL_PROJ_DATA=<cache dir> (see evmax/nfl_projections/data.py).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import replace  # noqa: E402

from evmax.nfl_projections import data, player_games, team_games  # noqa: E402
from evmax.nfl_projections.player_model import PlayerModelConfig  # noqa: E402
from evmax.nfl_projections.player_model import walk_forward as model_walk_forward  # noqa: E402

DEV = list(range(2019, 2025))
HOLDOUT = [2025]
SEASONS_LOADED = range(2013, 2027)
NAIVE_GAMES = 8
STATS = {  # stat -> (projection column, population column)
    "receiving_yards": ("proj_receiving_yards", "pop_rec"),
    "receptions": ("proj_receptions", "pop_rec"),
    "rushing_yards": ("proj_rushing_yards", "pop_rush"),
    "passing_yards": ("proj_passing_yards", "pop_pass"),
}


def naive_means(pg: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    hist = pg[pg["gameday"] < cutoff].sort_values("gameday")
    last = hist.groupby("player_id").tail(NAIVE_GAMES)
    return last.groupby("player_id")[["targets", "receptions", "receiving_yards", "carries",
                                      "rushing_yards", "passing_yards"]].mean().add_prefix("naive_")


def walk_forward(seasons: list[int], cfg: PlayerModelConfig) -> pd.DataFrame:
    """Model projections (from the package) + the harness's own naive baseline."""
    tg = team_games.load_team_games(SEASONS_LOADED)
    pg = player_games.load_player_games(SEASONS_LOADED)
    games = data.load_games()
    injuries = pd.concat([data.load_injuries(s) for s in SEASONS_LOADED], ignore_index=True)
    ros = None
    if cfg.roster_out_redistribution:
        for s in seasons:
            data.ensure_rosters(s, max_age_hours=None)
        ros = data.load_rosters(seasons)
    proj = model_walk_forward(tg, pg, games, seasons, cfg, injuries=injuries, rosters=ros)
    cut = games.groupby(["season", "week"])["gameday"].min().rename("cutoff")
    proj = proj.join(cut, on=["season", "week"])
    out = []
    for cutoff, wk in proj.groupby("cutoff", sort=True):
        out.append(wk.join(naive_means(pg, cutoff), on="player_id"))
    r = pd.concat(out, ignore_index=True)
    r["pop_rec"] = r["naive_targets"] >= 1.5
    r["pop_rush"] = r["naive_carries"] >= 2.0
    r["pop_pass"] = r["is_starting_qb"] & r["naive_passing_yards"].notna()
    return r


def metrics(r: pd.DataFrame) -> dict[str, dict[str, float]]:
    m = {}
    for stat, (pcol, popcol) in STATS.items():
        d = r[r[popcol]]
        m[stat] = {"n": float(len(d)),
                   "mae": float((d[pcol] - d[stat]).abs().mean()),
                   "naive": float((d["naive_" + stat] - d[stat]).abs().mean()),
                   "bias": float((d[pcol] - d[stat]).mean())}
        m[stat]["ratio"] = m[stat]["mae"] / m[stat]["naive"]
    return m


def coverage(r: pd.DataFrame) -> str:
    """Calibration of the reported 10th/90th percentiles (ideal: 10% below p10, 90% at/below p90).
    Reporting only — not part of the loop signal."""
    parts = []
    for stat, (_, popcol) in STATS.items():
        d = r[r[popcol]]
        if f"p10_{stat}" not in d:
            continue
        below = (d[stat] < d[f"p10_{stat}"]).mean() * 100
        upto = (d[stat] <= d[f"p90_{stat}"]).mean() * 100
        parts.append(f"{stat}: <p10 {below:.1f}% / <=p90 {upto:.1f}%")
    return "coverage " + " | ".join(parts)


def fmt(name: str, m: dict) -> str:
    parts = [f"{s}: {v['mae']:.3f} vs naive {v['naive']:.3f} ({v['ratio']:.4f}, bias {v['bias']:+.2f}, n={int(v['n'])})"
             for s, v in m.items()]
    return f"{name:8s} " + " | ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--save", type=Path)
    ap.add_argument("--redistribute-roster-out", action="store_true",
                    help="also free the usage of roster-ruled-out players (rejected arm)")
    args = ap.parse_args()
    cfg = replace(PlayerModelConfig(), roster_out_redistribution=args.redistribute_roster_out)
    seasons = DEV + (HOLDOUT if args.holdout else [])
    r = walk_forward(seasons, cfg)
    reg = r[r["season_type"] == "REG"]
    if args.save:
        r.to_parquet(args.save, index=False)
    dev = metrics(reg[reg["season"].isin(DEV)])
    print(fmt("dev", dev))
    print("  dev " + coverage(reg[reg["season"].isin(DEV)]))
    if args.holdout:
        print(fmt("holdout", metrics(reg[reg["season"].isin(HOLDOUT)])))
        print("  holdout " + coverage(reg[reg["season"].isin(HOLDOUT)]))
    score = float(np.mean([v["ratio"] for v in dev.values()]))
    print(f"dev_score={score:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
