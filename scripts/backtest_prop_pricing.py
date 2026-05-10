"""Walk-forward backtest of evmax/ev/prop_pricing.py distribution choices.

Tests the question: given a player's L20 rolling mean as the "Pinnacle-like
anchor", which parametric distribution best predicts P(X >= threshold) at
actual game-time outcomes?

Methodology — per (player, game G) with G > WARMUP:
  1. Compute μ_pre, σ_pre from games 1 .. G-1 (rolling L20).
  2. For each threshold offset Δ ∈ {-5, -2, 0, +2, +5}:
       threshold = round(μ_pre + Δ)
       actual    = 1 if game G's stat >= threshold else 0
       predict   = P(X >= threshold) under each distribution:
                     Poisson(λ=μ_pre)
                     NegBin(μ=μ_pre, k=_NEGBIN_STAT_K[stat])
                     Normal(μ=μ_pre, σ=σ_pre)
       brier     = (predict - actual)²
  3. Accumulate Brier per stat / per model / per offset.

Why offsets, not arbitrary thresholds: Pinnacle posts lines near the player's
mean, but Kalshi posts thresholds 5+ points away. Tail accuracy is what
matters for betting; offsets exercise the tail behavior of each distribution.

Lower Brier = better calibration. NegBin should beat Poisson at high offsets
(heavier tail captures Giannis-50 events). Normal vs NegBin is the close fight.

Run:
  .venv/bin/python scripts/backtest_prop_pricing.py
  .venv/bin/python scripts/backtest_prop_pricing.py --warmup 20 --stats points,rebounds
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sci_stats

REPO_ROOT = Path(__file__).resolve().parent.parent
NBA_PATH = REPO_ROOT / "data" / "historical_nba" / "player_game_logs_2024_25.parquet"

sys.path.insert(0, str(REPO_ROOT))
from evmax.ev.prop_pricing import _NEGBIN_STAT_K  # noqa: E402

_NBA_COLUMNS: dict[str, str] = {
    "points":   "PTS",
    "rebounds": "REB",
    "assists":  "AST",
    "threes":   "FG3M",
    "steals":   "STL",
    "blocks":   "BLK",
}

# Fixed population σ per stat — what a production "Normal everywhere"
# variant would use (no per-player adaptivity). Median per-player SD from the
# 2024-25 high-volume cohort, derived in scripts/calibrate_prop_sigma.py.
_NORMAL_FIXED_SIGMA: dict[str, float] = {
    "points":   5.9,
    "rebounds": 2.3,
    "assists":  1.6,
    "threes":   1.3,
    "steals":   0.9,
    "blocks":   0.7,
}

OFFSETS = [-5, -2, 0, 2, 5, 10, 15]
DEFAULT_WARMUP = 20
MIN_GAMES = 40


def _poisson_sf_at(mu: float, threshold: int) -> float:
    """P(X >= threshold) under Poisson(λ=μ)."""
    if threshold <= 0:
        return 1.0
    return float(sci_stats.poisson.sf(threshold - 1, mu))


def _negbin_sf_at(mu: float, k: float, threshold: int) -> float:
    """P(X >= threshold) under NegBin(μ, k)."""
    if threshold <= 0:
        return 1.0
    if mu <= 0 or k <= 0:
        return 0.0
    p = k / (k + mu)
    return float(sci_stats.nbinom.sf(threshold - 1, n=k, p=p))


def _normal_sf_at(mu: float, sigma: float, threshold: int) -> float:
    """P(X >= threshold) under Normal(μ, σ) with continuity correction."""
    if sigma <= 0:
        return 1.0 if mu >= threshold else 0.0
    return float(sci_stats.norm.sf(threshold - 0.5, loc=mu, scale=sigma))


def _walk_forward_one_player(
    games: np.ndarray,
    stat_name: str,
    k_negbin: float,
    sigma_fixed: float,
    warmup: int,
) -> dict[str, dict[int, list[float]]]:
    """Return {model_name: {offset: [brier_per_eligible_game, ...]}}.

    Models compared:
      - poisson       : Poisson(λ=μ_pre)
      - negbin        : NegBin(μ=μ_pre, k=fixed) — what production uses
      - normal_fixed  : Normal(μ=μ_pre, σ=fixed_population) — production
                        alternative if we replaced NegBin with Normal
      - normal_player : Normal(μ=μ_pre, σ=per-player L20 σ) — best-case
                        Normal, requires per-player σ tracking (legacy L15)
    """
    model_names = ("poisson", "negbin", "normal_fixed", "normal_player")
    results: dict[str, dict[int, list[float]]] = {
        m: {off: [] for off in OFFSETS} for m in model_names
    }
    n = len(games)
    if n <= warmup + 1:
        return results

    for g in range(warmup, n):
        history = games[:g]
        mu_pre = float(np.mean(history))
        sigma_player = float(np.std(history, ddof=1)) if len(history) > 1 else 0.0
        actual = float(games[g])

        for offset in OFFSETS:
            thresh = int(round(mu_pre + offset))
            if thresh < 0:
                continue
            outcome = 1.0 if actual >= thresh else 0.0
            pp = _poisson_sf_at(mu_pre, thresh)
            pn = _negbin_sf_at(mu_pre, k_negbin, thresh)
            pr_fixed  = _normal_sf_at(mu_pre, sigma_fixed,  thresh) if sigma_fixed  > 0 else None
            pr_player = _normal_sf_at(mu_pre, sigma_player, thresh) if sigma_player > 0 else None
            results["poisson"][offset].append((pp - outcome) ** 2)
            results["negbin"][offset].append((pn - outcome) ** 2)
            if pr_fixed is not None:
                results["normal_fixed"][offset].append((pr_fixed - outcome) ** 2)
            if pr_player is not None:
                results["normal_player"][offset].append((pr_player - outcome) ** 2)
    return results


_MODEL_NAMES = ("poisson", "negbin", "normal_fixed", "normal_player")


def backtest_stat(df: pd.DataFrame, stat_name: str, col: str, warmup: int) -> dict:
    k = _NEGBIN_STAT_K.get(stat_name)
    sigma = _NORMAL_FIXED_SIGMA.get(stat_name, 0.0)
    if k is None:
        return {}

    accum: dict[str, dict[int, list[float]]] = {
        m: {off: [] for off in OFFSETS} for m in _MODEL_NAMES
    }

    df_sorted = df.sort_values(["PLAYER_ID", "GAME_DATE"])
    for _, player_df in df_sorted.groupby("PLAYER_ID"):
        if len(player_df) < MIN_GAMES:
            continue
        games = player_df[col].fillna(0).to_numpy()
        per_player = _walk_forward_one_player(games, stat_name, k, sigma, warmup)
        for model, by_off in per_player.items():
            for off, briers in by_off.items():
                accum[model][off].extend(briers)

    summary: dict[str, dict] = {"stat": stat_name, "k_negbin": k, "sigma_fixed": sigma, "by_model": {}}
    for model, by_off in accum.items():
        per_off = {off: float(np.mean(briers)) if briers else float("nan")
                   for off, briers in by_off.items()}
        all_briers = [b for briers in by_off.values() for b in briers]
        per_off["all"] = float(np.mean(all_briers)) if all_briers else float("nan")
        per_off["n"] = len(all_briers)
        summary["by_model"][model] = per_off
    return summary


def print_summary(rows: list[dict]) -> None:
    if not rows:
        print("no results")
        return
    header = (f"{'stat':<12} {'model':<14} " +
              " ".join(f"Δ{o:+d}".rjust(8) for o in OFFSETS) + " " +
              f"{'all':>8}  {'n':>8}")
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        for model in _MODEL_NAMES:
            cells = r["by_model"][model]
            offs = " ".join(f"{cells[o]:.4f}".rjust(8) for o in OFFSETS)
            print(f"{r['stat']:<12} {model:<14} {offs} {cells['all']:>8.4f}  {cells['n']:>8d}")
        print()


def print_winner_table(rows: list[dict]) -> None:
    """Show each model's relative improvement vs the worst-performing model
    per stat. Higher = lower Brier = better calibration."""
    print("Brier improvement vs worst model per stat (positive = better):")
    header = (f"{'stat':<12} " + " ".join(f"{m:>14}" for m in _MODEL_NAMES))
    print(header)
    print("-" * len(header))
    for r in rows:
        b = {m: r["by_model"][m]["all"] for m in _MODEL_NAMES}
        worst = max(b.values())
        cells = {m: worst - v for m, v in b.items()}
        print(f"{r['stat']:<12} " + " ".join(f"{cells[m]:>14.5f}" for m in _MODEL_NAMES))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                        help=f"Warmup games before predicting (default {DEFAULT_WARMUP})")
    parser.add_argument("--stats", default=",".join(_NBA_COLUMNS.keys()),
                        help="Comma-separated stats to backtest")
    args = parser.parse_args()

    if not NBA_PATH.exists():
        print(f"missing {NBA_PATH} — run scripts/fetch_shufinskiy.py")
        return 1

    df = pd.read_parquet(NBA_PATH)
    df = df[df["MIN"] >= 12].copy()
    print(f"=== backtest_prop_pricing  warmup={args.warmup}  rows={len(df):,}  "
          f"players={df['PLAYER_ID'].nunique()} ===\n")

    requested_stats = [s.strip() for s in args.stats.split(",") if s.strip()]
    rows: list[dict] = []
    for stat in requested_stats:
        col = _NBA_COLUMNS.get(stat)
        if col is None:
            print(f"unknown stat {stat!r}, skipping")
            continue
        print(f"  walking forward {stat} ...")
        rows.append(backtest_stat(df, stat, col, args.warmup))

    print_summary(rows)
    print()
    print_winner_table(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
