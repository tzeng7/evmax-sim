#!/usr/bin/env python3
"""Fit player_model.PARTICIPATION: P(an expected player plays) by pre-game evidence.

Population: every completed week's pre-game roster (``player_model.pregame_rosters``
— the roster ``live.active_roster`` builds, also what ``walk_forward(roster="live")``
and ``scripts/eval_nfl_joint_sim.py --roster live`` simulate): the team's last-3-games
players minus the week's injury report (Out/Doubtful) and the weekly roster's
ruled-out players, backup QBs dropped; skill positions only (the players the
simulation gets) and the starting QB excluded (he always plays).
Cells: (games played of the team's last 3, played the team's last game,
Questionable on the week's report). ``played`` = offensive snaps or a stat.

Fit on --fit seasons, calibration reported on --holdout. Prints the table to paste
into PARTICIPATION. Data: EVMAX_NFL_PROJ_DATA=<cache dir> (downloads weekly rosters).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evmax.nfl_projections import data, player_games, team_games  # noqa: E402
from evmax.nfl_projections.player_model import (  # noqa: E402
    PARTICIPATION, SKILL, participation_probability, pregame_rosters,
)

KEYS = ["recent_games", "played_last", "questionable"]


def build(seasons: list[int]) -> pd.DataFrame:
    tg = team_games.load_team_games(range(min(seasons) - 1, max(seasons) + 1))
    pg = player_games.load_player_games(range(min(seasons) - 1, max(seasons) + 1))
    games = data.load_games()
    for s in seasons:
        data.ensure_rosters(s, max_age_hours=None)
    injuries = pd.concat([data.load_injuries(s) for s in seasons], ignore_index=True)
    r = pregame_rosters(tg, pg, games, seasons, injuries, data.load_rosters(seasons))
    # Skill players only: the roster also lists linemen (offensive snaps), but only
    # QB/RB/WR/TE have usage and reach the simulation (project_players).
    return r[~r["is_starting_qb"].astype(bool) & (r["recent_games"] > 0) & r["position"].isin(SKILL)].copy()


def calibration(d: pd.DataFrame, p: np.ndarray, const: float) -> str:
    d = d.assign(p=p)
    d["bin"] = pd.cut(d["p"], [0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0])
    g = d.groupby("bin", observed=True).agg(predicted=("p", "mean"), played=("played", "mean"), n=("p", "size"))
    brier = float(((d["p"] - d["played"]) ** 2).mean())
    base = float(((const - d["played"]) ** 2).mean())
    return (f"mean predicted {d['p'].mean():.3f} vs played {d['played'].mean():.3f}; Brier {brier:.4f} vs "
            f"{base:.4f} for a constant\n{g.round(3).to_string()}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="2019-2024")
    ap.add_argument("--holdout", type=int, default=2025)
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.fit.split("-"))
    d = build(list(range(lo, hi + 1)) + [args.holdout])
    fit, hold = d[d["season"].between(lo, hi)], d[d["season"] == args.holdout]
    t = fit.groupby(KEYS)["played"].agg(["mean", "size"])
    print(f"fit {lo}-{hi}: n={len(fit)}, played {fit['played'].mean():.3f}")
    print(t.round(3).to_string())
    table = {tuple(int(v) if i == 0 else bool(v) for i, v in enumerate(k)): round(float(m), 3)
             for k, m in t["mean"].items()}
    print("\nPARTICIPATION = {")
    for k, v in table.items():
        print(f"    {k}: {v},")
    print("}")
    p_new = hold.join(t["mean"].rename("p"), on=KEYS)["p"].fillna(1.0).to_numpy()
    p_shipped = participation_probability(hold["recent_games"], hold["played_last"], hold["questionable"],
                                          hold["is_starting_qb"])
    print(f"\nholdout {args.holdout} (n={len(hold)}), this fit: " + calibration(hold, p_new, fit["played"].mean()))
    if table != PARTICIPATION:
        print(f"\nholdout {args.holdout}, shipped PARTICIPATION: "
              + calibration(hold, p_shipped, fit["played"].mean()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
