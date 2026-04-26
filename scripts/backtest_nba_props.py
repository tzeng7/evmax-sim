"""Baseline NBA props Brier — replays the current `nba_props_cache` model
math on 2024-25 player game logs and scores it against actual outcomes.

This is the "baseline" measurement for the player-props v2 work (TODO #22).
Without it, we can't tell whether v2 actually moves the needle.

The production model lives in `evmax/clients/nba_props_cache.py` and is
tightly coupled to a global daily cache. To backtest, we extract the math
into a pure function that takes raw history arrays.

Methodology:
  1. Load 2024-25 player game logs (from scripts/fetch_nba_game_logs.py)
  2. Holdout window: 2025-01-01 → 2025-04-13 (gives players ≥15 prior
     games of 2024-25 history at evaluation time)
  3. For each (player, game) in holdout:
       - Build prior-15-games history dict from preceding rows
       - For each stat × candidate line, compute model prob using the
         exact production formula (ex-opponent adjustment — that requires
         team-stats history we don't have backfilled yet)
       - Compare to actual stat outcome
  4. Report Brier per (stat, line bucket) plus overall Brier and
     calibration tables.

The opponent-adjustment is omitted because reconstructing point-in-time
team defensive stats for every game date in 2024-25 is a separate project.
The production model's opponent adjustment caps at ±15% on probability
so its absence shifts the baseline by a bounded amount.

Usage:
    python scripts/backtest_nba_props.py
    python scripts/backtest_nba_props.py --season 2024-25 --start 2025-01-01
    python scripts/backtest_nba_props.py --stats points,rebounds
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "historical_nba"

# Mirror production constants from nba_props_cache.py
_LAST_N_GAMES = 15
_MIN_GAMES = 5
_DECAY = 0.85
_MIN_VOL_THRESHOLD = 1.5
_VOL_DOWNWEIGHT = 0.25
_BASE_RATE = 0.40
_SHRINKAGE = 0.20
_VOL_BASE = 0.25
_VOL_SHRINK = 0.40

# Map our stat names → game-log column names
STAT_COL: dict[str, str] = {
    "points": "PTS",
    "rebounds": "REB",
    "assists": "AST",
    "threes": "FG3M",
    "steals": "STL",
    "blocks": "BLK",
    "turnovers": "TOV",
    "points_rebounds_assists": "__pra__",
    "blocks_steals": "__bs__",
}

# Realistic prop-line grids per stat — covers the range Kalshi actually offers
LINE_GRIDS: dict[str, list[float]] = {
    "points": [9.5, 14.5, 19.5, 24.5, 29.5],
    "rebounds": [3.5, 5.5, 7.5, 9.5],
    "assists": [2.5, 4.5, 6.5, 8.5],
    "threes": [0.5, 1.5, 2.5, 3.5],
    "steals": [0.5, 1.5, 2.5],
    "blocks": [0.5, 1.5],
    "turnovers": [1.5, 2.5, 3.5],
    "points_rebounds_assists": [19.5, 29.5, 39.5],
    "blocks_steals": [1.5, 2.5, 3.5],
}


# ----------------------------------------------------------------------------
# Pure prob computation — mirrors evmax/clients/nba_props_cache.py
# ----------------------------------------------------------------------------

def compute_prop_prob_from_history(
    history: dict[str, list[float]],
    stat_type: str,
    threshold: float,
) -> float | None:
    """Pure version of compute_prop_prob_cached — operates on raw history.

    `history` is a dict with lists per stat (most-recent-first):
      {"PTS": [...], "REB": [...], ..., "MIN": [...]}

    Returns the model's predicted P(actual ≥ threshold), or None if the
    player has insufficient history. Mirrors the production formula
    exactly except for the opponent adjustment (which requires team
    stats not available in this backtest).
    """
    col = STAT_COL.get(stat_type)
    if col is None:
        return None

    n_games = len(history.get("PTS", []))
    if n_games < _MIN_GAMES:
        return None

    # Build raw stat series
    if col == "__pra__":
        pts = history.get("PTS", [])
        reb = history.get("REB", [])
        ast = history.get("AST", [])
        if not pts or not reb or not ast:
            return None
        n = min(len(pts), len(reb), len(ast), _LAST_N_GAMES)
        raw = np.array([pts[i] + reb[i] + ast[i] for i in range(n)])
    elif col == "__bs__":
        blk = history.get("BLK", [])
        stl = history.get("STL", [])
        if not blk or not stl:
            return None
        n = min(len(blk), len(stl), _LAST_N_GAMES)
        raw = np.array([blk[i] + stl[i] for i in range(n)])
    else:
        if col not in history:
            return None
        raw = np.array(history[col][:_LAST_N_GAMES], dtype=float)
        n = len(raw)

    if n < _MIN_GAMES:
        return None

    # Per-36-minute normalization
    minutes = np.array(history.get("MIN", [36.0] * n)[:n], dtype=float)
    avg_min = float(np.mean(minutes))
    min_std = float(np.std(minutes)) if n >= 3 else 0.0
    minutes_cv = min_std / avg_min if avg_min > 5.0 else 0.0

    # Detect minutes volatility
    volatile_mask = np.zeros(n, dtype=bool)
    if avg_min >= 5.0 and min_std > 1.0:
        volatile_mask = np.abs(minutes - avg_min) > (_MIN_VOL_THRESHOLD * min_std)
    volatile_games = int(np.sum(volatile_mask))
    minutes_volatile = volatile_games >= 2 or (volatile_games >= 1 and minutes_cv > 0.25)

    if avg_min >= 5.0:
        per36 = raw / np.maximum(minutes, 1.0) * 36.0
        eff_threshold = threshold / avg_min * 36.0
    else:
        per36 = raw
        eff_threshold = float(threshold)

    # Exponential decay weights with volatility downweighting
    weights = np.array([_DECAY ** i for i in range(n)])
    for i in range(n):
        if volatile_mask[i]:
            weights[i] *= _VOL_DOWNWEIGHT
    weights /= weights.sum()

    wmean = float(np.dot(weights, per36))
    wvar = float(np.dot(weights, (per36 - wmean) ** 2))
    wstd = max(float(np.sqrt(wvar)), 0.5)

    # Empirical hit rate + margin
    hits = raw >= threshold
    avg_margin = float(np.mean(raw - threshold))
    weighted_hit_rate = float(np.dot(weights, hits.astype(float)))

    # Last-5 streak detection
    last5 = raw[:5]
    last5_hits = int(np.sum(last5 >= threshold))

    # Base probability from normal distribution
    model_prob = float(1 - norm.cdf(eff_threshold, wmean, wstd))
    model_prob = max(0.01, min(0.99, model_prob))

    # 40/60 model/empirical blend
    blended_prob = 0.40 * model_prob + 0.60 * weighted_hit_rate

    # Margin adjustment (capped ±3%)
    margin_adj = float(np.clip(avg_margin * 0.003, -0.03, 0.03))
    blended_prob += margin_adj

    # Streak adjustment
    if last5_hits >= 4:
        streak_adj = 0.01 + 0.005 * (last5_hits - 4)
    elif last5_hits <= 1:
        streak_adj = -0.01 - 0.005 * (1 - last5_hits)
    else:
        streak_adj = 0.0
    blended_prob += streak_adj

    # Shrink toward base rate
    blended_prob = blended_prob * (1 - _SHRINKAGE) + _BASE_RATE * _SHRINKAGE

    # Volatility shrinkage
    if minutes_volatile:
        blended_prob = blended_prob * (1 - _VOL_SHRINK) + _VOL_BASE * _VOL_SHRINK

    return float(max(0.01, min(0.99, blended_prob)))


# ----------------------------------------------------------------------------
# Backtest
# ----------------------------------------------------------------------------

def build_history_from_prior_games(prior_df: pd.DataFrame) -> dict[str, list[float]]:
    """Convert a per-game-row dataframe into the history dict the model expects.

    Most-recent-first ordering, matches what compute_prop_prob_cached reads
    from the daily cache.
    """
    df = prior_df.sort_values("GAME_DATE", ascending=False).head(_LAST_N_GAMES)
    return {
        "PTS": df.PTS.tolist(),
        "REB": df.REB.tolist(),
        "AST": df.AST.tolist(),
        "FG3M": df.FG3M.tolist(),
        "STL": df.STL.tolist(),
        "BLK": df.BLK.tolist(),
        "TOV": df.TOV.tolist(),
        "MIN": df.MIN.tolist(),
    }


def actual_value(row: pd.Series, stat_type: str) -> float:
    if stat_type == "points_rebounds_assists":
        return float(row.PTS + row.REB + row.AST)
    if stat_type == "blocks_steals":
        return float(row.BLK + row.STL)
    col = STAT_COL[stat_type]
    return float(row[col])


def brier(preds: list[float], actuals: list[bool]) -> float:
    if not preds:
        return 0.0
    return sum((p - float(a)) ** 2 for p, a in zip(preds, actuals)) / len(preds)


def calibration_buckets(preds: list[float], actuals: list[bool]) -> list[dict]:
    buckets = []
    edges = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.001]
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        idx = [j for j, p in enumerate(preds) if lo <= p < hi]
        if not idx:
            continue
        m_pred = sum(preds[j] for j in idx) / len(idx)
        actual_rate = sum(1 for j in idx if actuals[j]) / len(idx)
        buckets.append({
            "lo": lo, "hi": hi, "n": len(idx),
            "mean_pred": m_pred, "actual": actual_rate,
            "delta": actual_rate - m_pred,
        })
    return buckets


def run_backtest(
    season: str = "2024-25",
    start_date: str = "2025-01-01",
    stats: list[str] | None = None,
) -> dict:
    season_key = season.replace("-", "_")
    parquet = DATA_DIR / f"player_game_logs_{season_key}.parquet"
    if not parquet.exists():
        raise FileNotFoundError(
            f"{parquet} not found. Run: "
            f"python scripts/fetch_nba_game_logs.py --season {season}"
        )

    print(f"Loading game logs from {parquet.name}...")
    df = pd.read_parquet(parquet)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"])

    holdout_start = pd.Timestamp(start_date)
    holdout = df[df.GAME_DATE >= holdout_start].copy()
    print(f"  Total player-games: {len(df):,}")
    print(f"  Holdout (≥ {start_date}): {len(holdout):,}")

    stat_list = stats or list(STAT_COL.keys())
    print(f"  Stats: {stat_list}")

    # Per-stat aggregations: list of (predicted_prob, actual_bool, line)
    by_stat: dict[str, dict] = {s: {"preds": [], "actuals": [], "lines": []} for s in stat_list}

    n_evaluated = 0
    n_skipped_history = 0

    for _, row in holdout.iterrows():
        prior = df[(df.PLAYER_ID == row.PLAYER_ID) & (df.GAME_DATE < row.GAME_DATE)]
        if len(prior) < _MIN_GAMES:
            n_skipped_history += 1
            continue

        history = build_history_from_prior_games(prior)
        n_evaluated += 1

        for stat in stat_list:
            actual = actual_value(row, stat)
            for line in LINE_GRIDS.get(stat, []):
                prob = compute_prop_prob_from_history(history, stat, line)
                if prob is None:
                    continue
                # Filter out lines outside [0.10, 0.90] predicted prob range —
                # those aren't realistic prop offerings (sportsbooks don't post
                # 95% favorites or 5% longshots as standard props).
                if prob < 0.10 or prob > 0.90:
                    continue
                outcome = actual >= line
                by_stat[stat]["preds"].append(prob)
                by_stat[stat]["actuals"].append(bool(outcome))
                by_stat[stat]["lines"].append(line)

    print(f"\n  Player-games evaluated: {n_evaluated:,}")
    print(f"  Skipped (insufficient history): {n_skipped_history:,}")

    # Aggregate report
    print(f"\n{'='*78}")
    print(f"  BASELINE — current nba_props_cache model on {season} holdout from {start_date}")
    print(f"{'='*78}")

    total_n = sum(len(by_stat[s]["preds"]) for s in stat_list)
    total_brier = sum(
        brier(by_stat[s]["preds"], by_stat[s]["actuals"]) * len(by_stat[s]["preds"])
        for s in stat_list
    ) / max(total_n, 1)

    print(f"\n{'Stat':<28s} {'N':>7s} {'Brier':>8s} {'Acc':>7s} {'Mean pred':>10s} {'Actual %':>10s}")
    print("-" * 78)
    for stat in stat_list:
        preds = by_stat[stat]["preds"]
        actuals = by_stat[stat]["actuals"]
        if not preds:
            print(f"{stat:<28s} {'0':>7s}  no eligible lines")
            continue
        b = brier(preds, actuals)
        acc = sum(1 for p, a in zip(preds, actuals) if (p >= 0.5) == a) / len(preds)
        mean_p = sum(preds) / len(preds)
        actual_rate = sum(actuals) / len(actuals)
        print(f"{stat:<28s} {len(preds):>7,d} {b:>8.4f} {acc:>6.1%} {mean_p:>9.1%} {actual_rate:>9.1%}")
    print("-" * 78)
    print(f"{'POOLED':<28s} {total_n:>7,d} {total_brier:>8.4f}")

    # Calibration table
    print(f"\nCalibration (pooled across all stats × lines):")
    all_preds = [p for s in stat_list for p in by_stat[s]["preds"]]
    all_actuals = [a for s in stat_list for a in by_stat[s]["actuals"]]
    print(f"\n{'Bucket':<14s} {'N':>7s} {'Mean pred':>10s} {'Actual %':>10s} {'Delta':>10s}")
    print("-" * 56)
    for b in calibration_buckets(all_preds, all_actuals):
        bucket = f"{int(b['lo']*100):>3}–{int(b['hi']*100 + 0.5):>3}%"
        print(f"{bucket:<14s} {b['n']:>7,d} {b['mean_pred']:>9.1%} {b['actual']:>9.1%} {b['delta']:>+9.1%}")
    print()

    return {
        "by_stat": by_stat,
        "pooled_brier": total_brier,
        "n": total_n,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--season", default="2024-25")
    p.add_argument("--start", default="2025-01-01",
                   help="Holdout start date (default: 2025-01-01 — gives ~6 weeks of in-season history first)")
    p.add_argument("--stats", default=None,
                   help="Comma-separated stat list (default: all)")
    args = p.parse_args()

    stats = [s.strip() for s in args.stats.split(",")] if args.stats else None
    run_backtest(season=args.season, start_date=args.start, stats=stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
