#!/usr/bin/env python3
"""Walk-forward evaluation of the NFL game projection engine (evmax.nfl_projections).

Projects every completed game week by week with ratings fit only on earlier
games, then scores margin / total / winner against the actual result and
reports the Vegas closing line on the SAME games for reference.

  dev      = 2019-2024 (the improvement loop optimizes this)
  holdout  = 2025      (reported only with --holdout; never tuned on)

The loop signal is the single line ``dev_score=<margin MAE + total MAE>``.

Usage:
    python scripts/backtest_nfl_game_projections.py            # dev signal + breakdown
    python scripts/backtest_nfl_game_projections.py --holdout  # also print 2025 + 2020-25 gate
Data: EVMAX_NFL_PROJ_DATA=<dir with pbp/ and games.parquet> (see evmax/nfl_projections/data.py).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evmax.nfl_projections import data, team_games  # noqa: E402
from evmax.nfl_projections.game_model import GameModelConfig, walk_forward  # noqa: E402

DEV = list(range(2019, 2025))
HOLDOUT = [2025]
GATE = list(range(2020, 2026))
SEASONS_LOADED = range(2013, 2027)


def metrics(df: pd.DataFrame) -> dict[str, float]:
    margin = df["home_score"] - df["away_score"]
    total = df["home_score"] + df["away_score"]
    home_won = (margin > 0).astype(float)
    decided = margin != 0
    out = {
        "n": float(len(df)),
        "margin_mae": float((df["proj_margin"] - margin).abs().mean()),
        "total_mae": float((df["proj_total"] - total).abs().mean()),
        "margin_bias": float((df["proj_margin"] - margin).mean()),
        "total_bias": float((df["proj_total"] - total).mean()),
        "brier": float(((df["p_home"] - home_won)[decided] ** 2).mean()),
        "su": float(((df["proj_margin"] > 0) == (margin > 0))[decided].mean()),
    }
    lines = df["spread_line"].notna() & df["total_line"].notna()
    out["vegas_margin_mae"] = float((df.loc[lines, "spread_line"] - margin[lines]).abs().mean())
    out["vegas_total_mae"] = float((df.loc[lines, "total_line"] - total[lines]).abs().mean())
    out["score"] = out["margin_mae"] + out["total_mae"]
    return out


def fmt(name: str, m: dict[str, float]) -> str:
    return (f"{name:9s} n={int(m['n']):4d}  margin MAE {m['margin_mae']:.3f} (Vegas {m['vegas_margin_mae']:.3f}, "
            f"bias {m['margin_bias']:+.2f})  total MAE {m['total_mae']:.3f} (Vegas {m['vegas_total_mae']:.3f}, "
            f"bias {m['total_bias']:+.2f})  Brier {m['brier']:.4f}  SU {m['su']*100:.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true", help="also report 2025 and the 2020-25 gate")
    ap.add_argument("--by-season", action="store_true")
    ap.add_argument("--save", type=Path, help="write per-game projections to this parquet")
    args = ap.parse_args()

    tg = team_games.load_team_games(SEASONS_LOADED)
    games = data.load_games()
    seasons = DEV + (HOLDOUT if args.holdout else [])
    proj = walk_forward(tg, games, seasons, GameModelConfig())
    reg = proj[proj["game_type"] == "REG"]
    if args.save:
        proj.to_parquet(args.save, index=False)

    dev = metrics(reg[reg["season"].isin(DEV)])
    print(fmt("dev", dev))
    if args.by_season:
        for s, d in reg.groupby("season"):
            print(fmt(str(s), metrics(d)))
    if args.holdout:
        print(fmt("holdout", metrics(reg[reg["season"].isin(HOLDOUT)])))
        print(fmt("gate", metrics(reg[reg["season"].isin(GATE)])))
    print(f"dev_score={dev['score']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
