"""Empirical σ calibration for evmax/ev/prop_pricing.py.

The Normal-distribution stats (PRA, NFL passing/rushing/receiving yards) need a
σ to fit μ from a Pinnacle anchor. The starter values in
``_NORMAL_STAT_SIGMA`` are educated guesses; this script measures what σ the
data actually says.

For Poisson-distributed stats (NBA points, rebounds, assists, threes, steals,
blocks) the script also reports Mean / Variance ratio — Poisson assumes 1.0.
If empirical variance >> mean, the stat is overdispersed and Poisson will
underestimate spread; the recommendation column flags candidates for migration
to Normal.

Sources:
  NBA — data/historical_nba/player_game_logs_2024_25.parquet (shufinskiy)
  NFL — data/backtest/nfl_props/weekly_stats.parquet (nflverse, optional)

Run:
  .venv/bin/python scripts/calibrate_prop_sigma.py
  .venv/bin/python scripts/calibrate_prop_sigma.py --min-games 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
NBA_PATH = REPO_ROOT / "data" / "historical_nba" / "player_game_logs_2024_25.parquet"
NFL_PATH = REPO_ROOT / "data" / "backtest" / "nfl_props" / "weekly_stats.parquet"

# Map our canonical stat name → column in the source parquet
_NBA_COLUMNS: dict[str, str] = {
    "points":     "PTS",
    "rebounds":   "REB",
    "assists":    "AST",
    "threes":     "FG3M",
    "steals":     "STL",
    "blocks":     "BLK",
}
_NBA_PRA = ("points_rebounds_assists", ["PTS", "REB", "AST"])

_NFL_COLUMNS: dict[str, str] = {
    "passing_yards":   "passing_yards",
    "rushing_yards":   "rushing_yards",
    "receiving_yards": "receiving_yards",
}

# Distribution choice in evmax/ev/prop_pricing.py — for the recommendation column
from evmax.ev.prop_pricing import (  # noqa: E402
    _STAT_DISTRIBUTION,
    _NORMAL_STAT_SIGMA,
    _NEGBIN_STAT_K,
)

HIGH_VOLUME_QUANTILE = 0.75  # top 25% by per-player mean — Pinnacle's prop pool


def _per_player_stats(
    df: pd.DataFrame,
    stat_col: str,
    player_col: str,
    min_games: int,
) -> pd.DataFrame:
    """For each player with ≥ min_games entries, compute count, mean, σ."""
    g = df.groupby(player_col)[stat_col]
    out = pd.DataFrame({
        "n_games":  g.count(),
        "mean":     g.mean(),
        "std":      g.std(ddof=1),
    })
    return out[out["n_games"] >= min_games].dropna()


def _negbin_k_high_volume(per_player: pd.DataFrame) -> Optional[float]:
    """Return median NegBin k = μ²/(σ²−μ) over the high-volume cohort
    (top 25% of players by mean). This is the population Pinnacle quotes —
    bench players inflate dispersion artificially because their minutes vary
    so much, so we filter them out before reporting k."""
    if per_player.empty:
        return None
    per_player = per_player.copy()
    per_player["var_minus_mean"] = per_player["std"] ** 2 - per_player["mean"]
    valid = per_player[per_player["var_minus_mean"] > 0.01]
    if valid.empty:
        return None
    threshold = valid["mean"].quantile(HIGH_VOLUME_QUANTILE)
    high_vol = valid[valid["mean"] >= threshold]
    if high_vol.empty:
        return None
    k_vals = (high_vol["mean"] ** 2) / high_vol["var_minus_mean"]
    return float(k_vals.median())


def _sample_summary(values: np.ndarray) -> tuple[float, float, float]:
    """Return (median, p25, p75) of the array."""
    return (
        float(np.median(values)),
        float(np.percentile(values, 25)),
        float(np.percentile(values, 75)),
    )


def calibrate_nba(min_games: int) -> list[dict]:
    if not NBA_PATH.exists():
        print(f"NBA: missing parquet {NBA_PATH} — skipping. Run scripts/fetch_shufinskiy.py.")
        return []

    df = pd.read_parquet(NBA_PATH)
    df = df[df["MIN"] >= 12]  # exclude DNP / blowout cameos that skew σ low

    # Add a PRA column for the composite
    df["_PRA"] = df["PTS"].fillna(0) + df["REB"].fillna(0) + df["AST"].fillna(0)

    def _row_for(canonical: str, col: str) -> Optional[dict]:
        per_player = _per_player_stats(df, col, "PLAYER_NAME", min_games)
        if per_player.empty:
            return None
        med_mean, _, _ = _sample_summary(per_player["mean"].values)
        med_std,  p25,  p75 = _sample_summary(per_player["std"].values)
        # Poisson check across all players (median var/mean — should be ~1)
        med_var_over_mean = float(np.median(
            (per_player["std"] ** 2) / per_player["mean"].clip(lower=0.01)
        ))
        return {
            "stat": canonical,
            "sport": "nba",
            "n_players": len(per_player),
            "median_mean": med_mean,
            "median_sd":   med_std,
            "sd_p25":      p25,
            "sd_p75":      p75,
            "var_over_mean": med_var_over_mean,
            "negbin_k_hv": _negbin_k_high_volume(per_player),
        }

    rows: list[dict] = []
    for canonical, col in _NBA_COLUMNS.items():
        row = _row_for(canonical, col)
        if row:
            rows.append(row)
    pra_row = _row_for(_NBA_PRA[0], "_PRA")
    if pra_row:
        rows.append(pra_row)
    return rows


def calibrate_nfl(min_games: int) -> list[dict]:
    if not NFL_PATH.exists():
        print(f"NFL: missing parquet {NFL_PATH} — skipping. Run scripts/fetch_nfl_features.py.")
        return []

    df = pd.read_parquet(NFL_PATH)
    rows: list[dict] = []
    for canonical, col in _NFL_COLUMNS.items():
        if col not in df.columns:
            continue
        # Filter to active-volume rows: any non-zero / non-null stat
        active = df[df[col].notna() & (df[col] != 0)]
        per_player = _per_player_stats(active, col, "player_id", min_games)
        if per_player.empty:
            continue
        med_mean, _, _ = _sample_summary(per_player["mean"].values)
        med_std,  p25,  p75 = _sample_summary(per_player["std"].values)
        rows.append({
            "stat": canonical,
            "sport": "nfl",
            "n_players": len(per_player),
            "median_mean": med_mean,
            "median_sd":   med_std,
            "sd_p25":      p25,
            "sd_p75":      p75,
            "var_over_mean": float(np.median(
                (per_player["std"] ** 2) / per_player["mean"].clip(lower=0.01)
            )),
            "negbin_k_hv": _negbin_k_high_volume(per_player),
        })
    return rows


def _recommendation(row: dict) -> str:
    """Compare empirical to current config and flag mismatches."""
    stat = row["stat"]
    dist = _STAT_DISTRIBUTION.get(stat, "?")

    if dist == "negbin":
        current_k = _NEGBIN_STAT_K.get(stat)
        empirical_k = row.get("negbin_k_hv")
        if current_k is None:
            return "NegBin but no k in _NEGBIN_STAT_K"
        if empirical_k is None:
            return f"NegBin; current k={current_k:.1f}, empirical=?"
        delta = empirical_k - current_k
        flag = "⚠" if abs(delta) / current_k > 0.20 else " "
        return f"{flag} NegBin; current k={current_k:.1f}, empirical_hv={empirical_k:.1f} ({delta:+.1f})"
    if dist == "normal":
        sd = row["median_sd"]
        current = _NORMAL_STAT_SIGMA.get(stat)
        if current is None:
            return "Normal but no σ in _NORMAL_STAT_SIGMA"
        delta = sd - current
        flag = "⚠" if abs(delta) / current > 0.15 else " "
        return f"{flag} Normal; current σ={current:.1f}, empirical={sd:.1f} ({delta:+.1f})"
    if dist == "poisson":
        ratio = row["var_over_mean"]
        if ratio > 1.5:
            return f"⚠ Poisson; var/mean={ratio:.2f} (overdispersed)"
        return f"Poisson ok; var/mean={ratio:.2f}"
    return "?"


def print_table(rows: list[dict]) -> None:
    if not rows:
        return
    header = (
        f"{'sport':<5} {'stat':<26} {'n_pl':>5} {'mean':>7} {'sd':>7} "
        f"{'var/μ':>6} {'k_hv':>6}  recommendation"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        rec = _recommendation(r)
        k_hv = r.get("negbin_k_hv")
        k_str = f"{k_hv:.1f}" if k_hv is not None else "n/a"
        print(
            f"{r['sport']:<5} {r['stat']:<26} {r['n_players']:>5d} "
            f"{r['median_mean']:>7.2f} {r['median_sd']:>7.2f} "
            f"{r['var_over_mean']:>6.2f} {k_str:>6}  {rec}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-games", type=int, default=30,
                        help="Minimum games per player to include (default 30)")
    args = parser.parse_args()

    print(f"=== calibrate_prop_sigma  min_games={args.min_games} ===\n")

    nba = calibrate_nba(args.min_games)
    nfl = calibrate_nfl(args.min_games)
    rows = nba + nfl

    if not rows:
        print("No data available — nothing to calibrate.")
        return 1

    print_table(rows)

    # Suggested config tables the user can paste
    print("\n# Suggested _NORMAL_STAT_SIGMA (only Normal-priced stats):")
    print("_NORMAL_STAT_SIGMA: dict[str, float] = {")
    for r in rows:
        if _STAT_DISTRIBUTION.get(r["stat"]) == "normal":
            print(f"    {r['stat']!r:<28}: {r['median_sd']:.1f},   # n={r['n_players']}")
    print("}")

    print("\n# Suggested _NEGBIN_STAT_K (high-volume cohort, top 25% by mean):")
    print("_NEGBIN_STAT_K: dict[str, float] = {")
    for r in rows:
        if _STAT_DISTRIBUTION.get(r["stat"]) == "negbin" and r.get("negbin_k_hv") is not None:
            print(f"    {r['stat']!r:<10}: {r['negbin_k_hv']:.1f},   # n={r['n_players']}")
    print("}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
