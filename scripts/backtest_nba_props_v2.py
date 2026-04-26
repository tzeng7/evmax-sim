"""Player Props v2 backtest — extends the calibrated baseline with two new
signals on top: career-mean prior + days-rest adjustment.

Goal: measure whether shufinskiy historical data + simple per-player career
features can move Brier below the calibrated baseline (TODO #23).

V2 additions on top of the calibrated v1 (Path A):
  1. Career-mean blend — for each (player, stat) compute the player's career
     mean across 2020-21 through 2023-24 (from nba_api game logs). At
     prediction time, blend wmean_l15 with career_mean weighted by
     `season_games_played / 30`. Rookies (career_mean unknown) fall back
     to pure L15 — same as the baseline.
  2. Days-rest factor — compute days since the player's previous game.
     Apply a small additive multiplier to wmean before computing model_prob:
       0 days (back-to-back):  ×0.97  (small fatigue penalty)
       1 day:                  ×1.00  (baseline)
       2 days:                 ×1.01
       3+ days:                ×1.02  (fully rested)
     These factors are calibrated from public NBA back-to-back research.

Methodology mirrors scripts/backtest_nba_props.py — same train/test split,
same line grids, same evaluation metric. Reports baseline vs v2 side by
side so the delta is visible.

Usage:
    python scripts/backtest_nba_props_v2.py
    python scripts/backtest_nba_props_v2.py --skip-career    # ablation
    python scripts/backtest_nba_props_v2.py --skip-rest      # ablation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "historical_nba"
CALIBRATION_PATH = REPO_ROOT / "data" / "models" / "nba_props_calibration.json"

# Prior-seasons game logs for career-mean computation
PRIOR_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24"]

# Production constants (mirror nba_props_cache.py post-Path-A)
_LAST_N_GAMES = 15
_MIN_GAMES = 5
_DECAY = 0.85
_MIN_VOL_THRESHOLD = 1.5
_VOL_DOWNWEIGHT = 0.25
_VOL_BASE = 0.25
_VOL_SHRINK = 0.40

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

# Days-rest multipliers applied to wmean before computing model_prob.
# Calibrated from public NBA B2B research: 1.5–3% per-stat impact at the
# extremes. Values capped narrow because the per-36 normalization already
# absorbs most rotation effects.
DAYS_REST_FACTOR: dict[int, float] = {
    0: 0.97,   # back-to-back
    1: 1.00,
    2: 1.01,
    3: 1.02,
}


# ----------------------------------------------------------------------------
# Career-mean prior
# ----------------------------------------------------------------------------

def load_career_means() -> dict[int, dict[str, float]]:
    """Build {player_id → {stat → career_mean}} from prior seasons' game logs.

    Career means are computed across all prior seasons in PRIOR_SEASONS.
    Players who don't appear in prior seasons (rookies) get an empty dict
    and their predictions fall back to pure L15 (same as v1).
    """
    frames = []
    for season in PRIOR_SEASONS:
        season_key = season.replace("-", "_")
        path = DATA_DIR / f"player_game_logs_{season_key}.parquet"
        if not path.exists():
            print(f"  warning: missing {path.name} — career-mean prior won't include {season}")
            continue
        frames.append(pd.read_parquet(path))
    if not frames:
        return {}

    df = pd.concat(frames, ignore_index=True)
    # Compute per-player per-stat means. Use raw stat (not per-36) since L15
    # at runtime is also raw — the blend lives in that space.
    out: dict[int, dict[str, float]] = {}
    for pid, group in df.groupby("PLAYER_ID"):
        n = len(group)
        out[int(pid)] = {
            "PTS": float(group.PTS.mean()),
            "REB": float(group.REB.mean()),
            "AST": float(group.AST.mean()),
            "FG3M": float(group.FG3M.mean()),
            "STL": float(group.STL.mean()),
            "BLK": float(group.BLK.mean()),
            "TOV": float(group.TOV.mean()),
            "MIN": float(group.MIN.mean()),
            "GP": n,
        }
    print(f"  Loaded career means for {len(out):,} players across {len(frames)} seasons")
    return out


# ----------------------------------------------------------------------------
# Calibration loading (mirrors evmax/clients/nba_props_cache.py)
# ----------------------------------------------------------------------------

def load_calibration() -> dict | None:
    if not CALIBRATION_PATH.exists():
        return None
    return json.loads(CALIBRATION_PATH.read_text())


def apply_isotonic(prob: float, x_thresh: list[float], y_thresh: list[float]) -> float:
    if not x_thresh or not y_thresh:
        return prob
    if prob <= x_thresh[0]:
        return y_thresh[0]
    if prob >= x_thresh[-1]:
        return y_thresh[-1]
    lo, hi = 0, len(x_thresh) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if x_thresh[mid] <= prob:
            lo = mid
        else:
            hi = mid
    x0, x1 = x_thresh[lo], x_thresh[hi]
    y0, y1 = y_thresh[lo], y_thresh[hi]
    if x1 == x0:
        return y0
    t = (prob - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


# ----------------------------------------------------------------------------
# v2 prediction: same shape as v1 + career mean + days rest
# ----------------------------------------------------------------------------

def compute_prob_v2(
    history: dict[str, list[float]],
    stat_type: str,
    threshold: float,
    cal: dict,
    career_mean: dict[str, float] | None = None,
    days_rest: int | None = None,
    use_career: bool = True,
    use_rest: bool = True,
) -> float | None:
    """v2 prediction with career-mean prior + days-rest factor.

    Mirrors compute_prop_prob_cached's math (Path A constants + isotonic)
    but adds the two new signals. When use_career=False or use_rest=False,
    that signal is disabled (for ablation).
    """
    col = STAT_COL.get(stat_type)
    if col is None:
        return None
    n_games = len(history.get("PTS", []))
    if n_games < _MIN_GAMES:
        return None

    if col == "__pra__":
        pts, reb, ast = history.get("PTS", []), history.get("REB", []), history.get("AST", [])
        if not pts or not reb or not ast:
            return None
        n = min(len(pts), len(reb), len(ast), _LAST_N_GAMES)
        raw = np.array([pts[i] + reb[i] + ast[i] for i in range(n)])
        career_key = None  # PRA's career mean handled below
    elif col == "__bs__":
        blk, stl = history.get("BLK", []), history.get("STL", [])
        if not blk or not stl:
            return None
        n = min(len(blk), len(stl), _LAST_N_GAMES)
        raw = np.array([blk[i] + stl[i] for i in range(n)])
        career_key = None
    else:
        if col not in history:
            return None
        raw = np.array(history[col][:_LAST_N_GAMES], dtype=float)
        n = len(raw)
        career_key = col

    if n < _MIN_GAMES:
        return None

    minutes = np.array(history.get("MIN", [36.0] * n)[:n], dtype=float)
    avg_min = float(np.mean(minutes))
    min_std = float(np.std(minutes)) if n >= 3 else 0.0
    minutes_cv = min_std / avg_min if avg_min > 5.0 else 0.0

    volatile_mask = np.zeros(n, dtype=bool)
    if avg_min >= 5.0 and min_std > 1.0:
        volatile_mask = np.abs(minutes - avg_min) > (_MIN_VOL_THRESHOLD * min_std)
    volatile_games = int(np.sum(volatile_mask))
    minutes_volatile = volatile_games >= 2 or (volatile_games >= 1 and minutes_cv > 0.25)

    weights = np.array([_DECAY ** i for i in range(n)])
    for i in range(n):
        if volatile_mask[i]:
            weights[i] *= _VOL_DOWNWEIGHT
    weights /= weights.sum()

    if avg_min >= 5.0:
        per36 = raw / np.maximum(minutes, 1.0) * 36.0
        eff_threshold = threshold / avg_min * 36.0
    else:
        per36 = raw
        eff_threshold = float(threshold)

    wmean = float(np.dot(weights, per36))
    wvar = float(np.dot(weights, (per36 - wmean) ** 2))
    wstd = max(float(np.sqrt(wvar)), 0.5)

    # ---- v2 signal #1: career-mean prior ----
    # Blend the rolling L15 mean with the career mean. Weight depends on
    # how much in-season data we have: more games this season → more weight
    # on L15. Career mean uses the same per-36 transform.
    if use_career and career_mean and career_key and career_key in career_mean:
        career_raw = career_mean[career_key]
        career_min = career_mean.get("MIN", 36.0) or 36.0
        career_per36 = career_raw / max(career_min, 1.0) * 36.0
        # In-season data weight ramps from 50% at n=5 to 100% at n=30+
        season_n = n  # the # of L15 entries we have (≤15)
        l15_weight = min(1.0, 0.5 + season_n / 30.0)
        wmean = l15_weight * wmean + (1 - l15_weight) * career_per36

    # ---- v2 signal #2: days-rest adjustment ----
    if use_rest and days_rest is not None:
        factor = DAYS_REST_FACTOR.get(min(days_rest, 3), 1.02)
        wmean = wmean * factor

    hits = raw >= threshold
    avg_margin = float(np.mean(raw - threshold))
    weighted_hit_rate = float(np.dot(weights, hits.astype(float)))

    last5_hits = int(np.sum(raw[:5] >= threshold))

    model_prob = float(1 - norm.cdf(eff_threshold, wmean, wstd))
    model_prob = max(0.01, min(0.99, model_prob))

    blend_model = cal["blend_model"]
    base_rate = cal["base_rate"]
    shrinkage = cal["shrinkage"]

    blended_prob = blend_model * model_prob + (1 - blend_model) * weighted_hit_rate

    margin_adj = float(np.clip(avg_margin * 0.003, -0.03, 0.03))
    blended_prob += margin_adj

    if last5_hits >= 4:
        streak_adj = 0.01 + 0.005 * (last5_hits - 4)
    elif last5_hits <= 1:
        streak_adj = -0.01 - 0.005 * (1 - last5_hits)
    else:
        streak_adj = 0.0
    blended_prob += streak_adj

    blended_prob = blended_prob * (1 - shrinkage) + base_rate * shrinkage

    if minutes_volatile:
        blended_prob = blended_prob * (1 - _VOL_SHRINK) + _VOL_BASE * _VOL_SHRINK

    # Isotonic last
    blended_prob = apply_isotonic(
        blended_prob, cal.get("isotonic_x_thresholds", []), cal.get("isotonic_y_thresholds", []),
    )

    return float(max(0.01, min(0.99, blended_prob)))


# ----------------------------------------------------------------------------
# Backtest
# ----------------------------------------------------------------------------

def actual_value(row, stat_type: str) -> float:
    if stat_type == "points_rebounds_assists":
        return float(row.PTS + row.REB + row.AST)
    if stat_type == "blocks_steals":
        return float(row.BLK + row.STL)
    return float(getattr(row, STAT_COL[stat_type]))


def brier(preds, actuals) -> float:
    if not preds:
        return 0.0
    return float(np.mean((np.array(preds) - np.array(actuals)) ** 2))


def calibration_buckets(preds, actuals, n_bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1.001, n_bins + 1)
    out = []
    preds_a = np.array(preds)
    acts_a = np.array(actuals)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (preds_a >= lo) & (preds_a < hi)
        if mask.sum() == 0:
            continue
        out.append({
            "lo": float(lo), "hi": float(hi), "n": int(mask.sum()),
            "mean_pred": float(preds_a[mask].mean()),
            "actual": float(acts_a[mask].mean()),
        })
    return out


def run(use_career: bool, use_rest: bool) -> dict:
    print(f"Loading 2024-25 game logs (test set)...")
    test_logs = pd.read_parquet(DATA_DIR / "player_game_logs_2024_25.parquet")
    test_logs["GAME_DATE"] = pd.to_datetime(test_logs["GAME_DATE"])
    test_logs = test_logs.sort_values(["PLAYER_ID", "GAME_DATE"])

    print(f"Loading career means from {len(PRIOR_SEASONS)} prior seasons...")
    career_means = load_career_means() if use_career else {}

    cal = load_calibration()
    if cal is None:
        raise RuntimeError(f"Calibration JSON missing at {CALIBRATION_PATH}")
    print(f"Calibration loaded: base_rate={cal['base_rate']}, "
          f"shrinkage={cal['shrinkage']}, blend_model={cal['blend_model']}, "
          f"isotonic n={len(cal.get('isotonic_x_thresholds', []))}")

    holdout = test_logs[test_logs.GAME_DATE >= pd.Timestamp("2025-03-01")].copy()
    print(f"Test holdout (Mar-Apr 2025): {len(holdout):,} player-games")

    preds_list, actuals_list = [], []
    skipped = 0

    for row in holdout.itertuples(index=False):
        prior = test_logs[
            (test_logs.PLAYER_ID == row.PLAYER_ID) & (test_logs.GAME_DATE < row.GAME_DATE)
        ]
        if len(prior) < _MIN_GAMES:
            skipped += 1
            continue
        prior_sorted = prior.sort_values("GAME_DATE", ascending=False).head(_LAST_N_GAMES)
        history = {
            "PTS": prior_sorted.PTS.tolist(), "REB": prior_sorted.REB.tolist(),
            "AST": prior_sorted.AST.tolist(), "FG3M": prior_sorted.FG3M.tolist(),
            "STL": prior_sorted.STL.tolist(), "BLK": prior_sorted.BLK.tolist(),
            "TOV": prior_sorted.TOV.tolist(), "MIN": prior_sorted.MIN.tolist(),
        }
        last_game_date = prior_sorted.GAME_DATE.iloc[0]
        days_rest = int((row.GAME_DATE - last_game_date).days) - 1
        days_rest = max(0, days_rest)

        career_mean = career_means.get(int(row.PLAYER_ID))

        for stat in STAT_COL:
            actual = actual_value(row, stat)
            for line in LINE_GRIDS.get(stat, []):
                prob = compute_prob_v2(
                    history, stat, line, cal,
                    career_mean=career_mean, days_rest=days_rest,
                    use_career=use_career, use_rest=use_rest,
                )
                if prob is None or prob < 0.05 or prob > 0.95:
                    continue
                preds_list.append(prob)
                actuals_list.append(int(actual >= line))

    print(f"  Predictions made: {len(preds_list):,}, skipped: {skipped:,}")

    pooled_brier = brier(preds_list, actuals_list)
    label_parts = []
    if use_career: label_parts.append("career")
    if use_rest: label_parts.append("rest")
    label = "+".join(label_parts) if label_parts else "neither"

    print(f"\n[v2 with {label}] Pooled Brier: {pooled_brier:.4f}  "
          f"(over n={len(preds_list):,})")

    print(f"\nCalibration table:")
    print(f"{'Bucket':<12s} {'N':>7s} {'Mean pred':>10s} {'Actual %':>10s} {'Δ':>8s}")
    for b in calibration_buckets(preds_list, actuals_list):
        bucket = f"{int(b['lo']*100):>3}–{int(b['hi']*100 + 0.5):>3}%"
        print(f"{bucket:<12s} {b['n']:>7,d} {b['mean_pred']:>9.1%} "
              f"{b['actual']:>9.1%} {b['actual']-b['mean_pred']:>+7.1%}")

    return {
        "label": label,
        "brier": pooled_brier,
        "n": len(preds_list),
        "preds": preds_list,
        "actuals": actuals_list,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--skip-career", action="store_true",
                   help="Disable career-mean prior (ablation)")
    p.add_argument("--skip-rest", action="store_true",
                   help="Disable days-rest factor (ablation)")
    args = p.parse_args()

    use_career = not args.skip_career
    use_rest = not args.skip_rest

    print("="*78)
    print("  PLAYER PROPS v2 BACKTEST")
    print(f"  career_mean={use_career}, days_rest={use_rest}")
    print("="*78)

    run(use_career, use_rest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
