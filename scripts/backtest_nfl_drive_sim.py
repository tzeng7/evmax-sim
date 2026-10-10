#!/usr/bin/env python3
"""Walk-forward evaluation of the NFL drive simulator (evmax.nfl_projections.drive_model).

Two lenses on the same games (regular season):

  standalone : the simulator's mean home/away points -> margin and total MAE,
               next to the game model's (the walk-forward ridge + combiner) and
               the closing line's
  feature    : the game model with the simulator's points added as the
               ``drive`` combiner feature (GameModelConfig.features) vs without

  dev = 2019-2024; --holdout adds 2025 (report only).
Data: EVMAX_NFL_PROJ_DATA=<cache dir>.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evmax.nfl_projections import data, team_games  # noqa: E402
from evmax.nfl_projections.game_model import GameModelConfig, drive_points, walk_forward  # noqa: E402

DEV = list(range(2019, 2025))
HOLDOUT = [2025]


def standalone(tg: pd.DataFrame, games: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    sched = games[games["season"].isin(seasons) & games["home_score"].notna() & (games["game_type"] == "REG")]
    rows = []
    for (season, week), wk in sched.groupby(["season", "week"], sort=True):
        pts = drive_points(tg, games, wk["gameday"].min(), wk)
        for g in wk.itertuples():
            rows.append({"game_id": g.game_id, "season": season, "sim_home": pts[(g.game_id, g.home_team)],
                         "sim_away": pts[(g.game_id, g.away_team)]})
    return pd.DataFrame(rows)


def mae(df: pd.DataFrame, home: str, away: str) -> tuple[float, float]:
    m = (df[home] - df[away]) - (df["home_score"] - df["away_score"])
    t = (df[home] + df[away]) - (df["home_score"] + df["away_score"])
    return float(m.abs().mean()), float(t.abs().mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true")
    args = ap.parse_args()
    seasons = DEV + (HOLDOUT if args.holdout else [])
    tg = team_games.load_team_games(range(2013, 2027))
    games = data.load_games()
    base_cfg = GameModelConfig()
    drive_cfg = dataclasses.replace(base_cfg, features=tuple(base_cfg.features) + ("drive",)) \
        if "drive" not in base_cfg.features else base_cfg
    gm = walk_forward(tg, games, seasons, base_cfg)
    gd = walk_forward(tg, games, seasons, drive_cfg)
    sim = standalone(tg, games, seasons)
    reg = gm[gm["game_type"] == "REG"].merge(sim.drop(columns="season"), on="game_id").merge(
        gd[["game_id", "proj_home", "proj_away"]].rename(columns={"proj_home": "d_home", "proj_away": "d_away"}),
        on="game_id")
    reg["close_home"], reg["close_away"] = reg["spread_line"] / 2 + reg["total_line"] / 2, \
        reg["total_line"] / 2 - reg["spread_line"] / 2
    for label, sel in (("dev", reg["season"].isin(DEV)), ("holdout", reg["season"].isin(HOLDOUT))):
        d = reg[sel]
        if d.empty:
            continue
        rows = {"drive sim (standalone)": mae(d, "sim_home", "sim_away"),
                "game model": mae(d, "proj_home", "proj_away"),
                "game model + drive feature": mae(d, "d_home", "d_away"),
                "closing line": mae(d.dropna(subset=["spread_line", "total_line"]), "close_home", "close_away")}
        print(f"{label} (n={len(d)} games)")
        for k, (m, t) in rows.items():
            print(f"  {k:28s} margin MAE {m:6.3f}  total MAE {t:6.3f}  sum {m + t:7.3f}")
        per = d.groupby("season").apply(lambda x: sum(mae(x, "d_home", "d_away")) - sum(mae(x, "proj_home", "proj_away")),
                                        include_groups=False)
        print(f"  drive feature minus base, per season: {np.round(per.to_numpy(), 3)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
