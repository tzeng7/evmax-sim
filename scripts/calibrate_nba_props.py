"""Calibrate NBA prop probability constants + fit an isotonic layer.

Path A of the player-props improvement plan (TODO #24): the baseline
backtest (scripts/backtest_nba_props.py) showed the production model is
under-confident in the 50-70% range — predicted 65% actually realizes 82%.
That's a calibration problem, not a data problem, fixable by re-fitting
the manual constants and adding a post-hoc isotonic layer.

Step 1: For each (player-game, stat, line) eligible row, precompute the
        intermediate model values (raw model_prob, weighted_hit_rate,
        avg_margin, last5_hits, minutes_volatile) and the actual outcome.
        Save to data/historical_nba/nba_props_intermediates_2024_25.parquet.

Step 2: Train/test split — Jan 1-Feb 28 2025 as train, Mar 1-Apr 13 2025
        as test. Both halves of the post-2025-01-01 holdout that the
        baseline used.

Step 3: Grid-search _BASE_RATE × _SHRINKAGE × blend_ratio on TRAIN.
        Pick the combo that minimizes train Brier.

Step 4: Fit IsotonicRegression on the grid-search-best raw probs (TRAIN
        only) and apply on TEST.

Step 5: Report: baseline (current production constants) vs grid-tuned vs
        grid-tuned + isotonic, all evaluated on TEST.

Output: prints the table, saves the chosen constants + isotonic mapping
to data/models/nba_props_calibration.json so production can load it.

Usage:
    python scripts/calibrate_nba_props.py                  # full pipeline
    python scripts/calibrate_nba_props.py --recompute      # rebuild parquet
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "historical_nba"
MODELS_DIR = REPO_ROOT / "data" / "models"
INTERMEDIATES_PARQUET = DATA_DIR / "nba_props_intermediates_2024_25.parquet"
CALIBRATION_OUT = MODELS_DIR / "nba_props_calibration.json"

# Production constants we're going to vary
PROD_BASE_RATE = 0.40
PROD_SHRINKAGE = 0.20
PROD_BLEND_MODEL = 0.40
PROD_VOL_BASE = 0.25
PROD_VOL_SHRINK = 0.40

# Other production constants we keep fixed
_LAST_N_GAMES = 15
_MIN_GAMES = 5
_DECAY = 0.85
_MIN_VOL_THRESHOLD = 1.5
_VOL_DOWNWEIGHT = 0.25

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


# ----------------------------------------------------------------------------
# Step 1: precompute intermediates
# ----------------------------------------------------------------------------

def compute_intermediates(history: dict[str, list[float]], stat_type: str, threshold: float) -> dict | None:
    """Return the model's intermediate quantities — without applying the
    blend / shrinkage / volatility steps. Caller can recombine those fast
    for grid search.
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
    elif col == "__bs__":
        blk, stl = history.get("BLK", []), history.get("STL", [])
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

    minutes = np.array(history.get("MIN", [36.0] * n)[:n], dtype=float)
    avg_min = float(np.mean(minutes))
    min_std = float(np.std(minutes)) if n >= 3 else 0.0
    minutes_cv = min_std / avg_min if avg_min > 5.0 else 0.0

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

    weights = np.array([_DECAY ** i for i in range(n)])
    for i in range(n):
        if volatile_mask[i]:
            weights[i] *= _VOL_DOWNWEIGHT
    weights /= weights.sum()

    wmean = float(np.dot(weights, per36))
    wvar = float(np.dot(weights, (per36 - wmean) ** 2))
    wstd = max(float(np.sqrt(wvar)), 0.5)

    hits = raw >= threshold
    avg_margin = float(np.mean(raw - threshold))
    weighted_hit_rate = float(np.dot(weights, hits.astype(float)))

    last5_hits = int(np.sum(raw[:5] >= threshold))

    model_prob = float(1 - norm.cdf(eff_threshold, wmean, wstd))
    model_prob = max(0.01, min(0.99, model_prob))

    return {
        "model_prob": model_prob,
        "weighted_hit_rate": weighted_hit_rate,
        "avg_margin": avg_margin,
        "last5_hits": last5_hits,
        "minutes_volatile": minutes_volatile,
    }


def actual_value(row, stat_type: str) -> float:
    """Read the actual stat value off a namedtuple row from itertuples()."""
    if stat_type == "points_rebounds_assists":
        return float(row.PTS + row.REB + row.AST)
    if stat_type == "blocks_steals":
        return float(row.BLK + row.STL)
    return float(getattr(row, STAT_COL[stat_type]))


def build_intermediates_table(season: str = "2024-25") -> pd.DataFrame:
    """Walk 2024-25 holdout and precompute intermediates for every eligible row."""
    season_key = season.replace("-", "_")
    parquet = DATA_DIR / f"player_game_logs_{season_key}.parquet"
    if not parquet.exists():
        raise FileNotFoundError(parquet)

    print(f"Loading {parquet.name}...")
    df = pd.read_parquet(parquet)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"])

    holdout = df[df.GAME_DATE >= pd.Timestamp("2025-01-01")].copy()
    print(f"  Holdout rows: {len(holdout):,}")

    rows = []
    for i, row in enumerate(holdout.itertuples(index=False)):
        if i % 2000 == 0 and i > 0:
            print(f"  ... {i:,} / {len(holdout):,} player-games processed")
        prior = df[(df.PLAYER_ID == row.PLAYER_ID) & (df.GAME_DATE < row.GAME_DATE)]
        if len(prior) < _MIN_GAMES:
            continue
        prior_sorted = prior.sort_values("GAME_DATE", ascending=False).head(_LAST_N_GAMES)
        history = {
            "PTS": prior_sorted.PTS.tolist(), "REB": prior_sorted.REB.tolist(),
            "AST": prior_sorted.AST.tolist(), "FG3M": prior_sorted.FG3M.tolist(),
            "STL": prior_sorted.STL.tolist(), "BLK": prior_sorted.BLK.tolist(),
            "TOV": prior_sorted.TOV.tolist(), "MIN": prior_sorted.MIN.tolist(),
        }

        for stat in STAT_COL:
            actual = actual_value(row, stat)
            for line in LINE_GRIDS.get(stat, []):
                inter = compute_intermediates(history, stat, line)
                if inter is None:
                    continue
                # Filter to realistic-prop range for parity with the baseline
                # backtest (sportsbooks don't post 95% favorites or 5% longshots)
                if inter["model_prob"] < 0.05 or inter["model_prob"] > 0.95:
                    continue
                rows.append({
                    "game_date": pd.Timestamp(row.GAME_DATE).strftime("%Y-%m-%d"),
                    "player_id": row.PLAYER_ID,
                    "stat": stat,
                    "line": line,
                    "model_prob": inter["model_prob"],
                    "weighted_hit_rate": inter["weighted_hit_rate"],
                    "avg_margin": inter["avg_margin"],
                    "last5_hits": inter["last5_hits"],
                    "minutes_volatile": inter["minutes_volatile"],
                    "outcome": int(actual >= line),
                })

    out = pd.DataFrame(rows)
    out.to_parquet(INTERMEDIATES_PARQUET, index=False)
    print(f"\nWrote {INTERMEDIATES_PARQUET.name} — {len(out):,} rows")
    return out


# ----------------------------------------------------------------------------
# Step 2: vectorized blend recombination
# ----------------------------------------------------------------------------

def apply_blend(
    df: pd.DataFrame,
    base_rate: float,
    shrinkage: float,
    blend_model: float,
    vol_base: float = PROD_VOL_BASE,
    vol_shrink: float = PROD_VOL_SHRINK,
) -> np.ndarray:
    """Vectorized recombination of intermediate values into final blended_prob.

    Mirrors compute_prop_prob_cached's blend/shrinkage/streak/volatility
    chain but operates on whole columns at once. ~1 ms for 400k rows.
    """
    model_prob = df["model_prob"].to_numpy()
    weighted_hit_rate = df["weighted_hit_rate"].to_numpy()
    avg_margin = df["avg_margin"].to_numpy()
    last5_hits = df["last5_hits"].to_numpy()
    volatile = df["minutes_volatile"].to_numpy()

    blended = blend_model * model_prob + (1 - blend_model) * weighted_hit_rate

    margin_adj = np.clip(avg_margin * 0.003, -0.03, 0.03)
    blended = blended + margin_adj

    streak_adj = np.where(
        last5_hits >= 4, 0.01 + 0.005 * (last5_hits - 4),
        np.where(last5_hits <= 1, -0.01 - 0.005 * (1 - last5_hits), 0.0),
    )
    blended = blended + streak_adj

    blended = blended * (1 - shrinkage) + base_rate * shrinkage

    blended = np.where(
        volatile,
        blended * (1 - vol_shrink) + vol_base * vol_shrink,
        blended,
    )

    return np.clip(blended, 0.01, 0.99)


def brier(preds: np.ndarray, actuals: np.ndarray) -> float:
    return float(np.mean((preds - actuals) ** 2))


def calibration_bins(preds: np.ndarray, actuals: np.ndarray, n_bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1.001, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (preds >= lo) & (preds < hi)
        if mask.sum() == 0:
            continue
        out.append({
            "lo": lo, "hi": hi, "n": int(mask.sum()),
            "mean_pred": float(preds[mask].mean()),
            "actual": float(actuals[mask].mean()),
        })
    return out


# ----------------------------------------------------------------------------
# Step 3-5: tuning pipeline
# ----------------------------------------------------------------------------

def fit_one(train: pd.DataFrame, test: pd.DataFrame, label: str) -> dict:
    """Grid-search constants on TRAIN, fit isotonic, return everything needed
    to serialize and a per-strategy test Brier. Returns None if either split
    is too small (<200 rows) — per-stat fitting on tiny stats is meaningless.
    """
    if len(train) < 200 or len(test) < 200:
        print(f"  [{label}] skipped — train={len(train)} test={len(test)} (need ≥200 each)")
        return None

    actuals_train = train.outcome.to_numpy(dtype=float)
    actuals_test = test.outcome.to_numpy(dtype=float)

    base_test = apply_blend(test, PROD_BASE_RATE, PROD_SHRINKAGE, PROD_BLEND_MODEL)
    base_brier = brier(base_test, actuals_test)

    base_rates = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    shrinkages = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    blend_models = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    best = None
    for br, sh, bm in product(base_rates, shrinkages, blend_models):
        preds = apply_blend(train, br, sh, bm)
        b = brier(preds, actuals_train)
        if best is None or b < best["brier"]:
            best = {"brier": b, "base_rate": br, "shrinkage": sh, "blend_model": bm}

    tuned_test = apply_blend(test, best["base_rate"], best["shrinkage"], best["blend_model"])
    tuned_brier = brier(tuned_test, actuals_test)

    tuned_train = apply_blend(train, best["base_rate"], best["shrinkage"], best["blend_model"])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
    iso.fit(tuned_train, actuals_train)
    iso_test = iso.predict(tuned_test)
    iso_brier = brier(iso_test, actuals_test)

    print(f"  [{label}] n={len(train):,}/{len(test):,} | "
          f"baseline {base_brier:.4f} → tuned {tuned_brier:.4f} → iso {iso_brier:.4f} "
          f"(Δ {iso_brier - base_brier:+.4f})")

    return {
        "base_rate": best["base_rate"],
        "shrinkage": best["shrinkage"],
        "blend_model": best["blend_model"],
        "isotonic_x_thresholds": iso.X_thresholds_.tolist(),
        "isotonic_y_thresholds": iso.y_thresholds_.tolist(),
        "test_brier": {
            "baseline": base_brier,
            "tuned": tuned_brier,
            "isotonic": iso_brier,
        },
        "n_train": len(train),
        "n_test": len(test),
    }


def run_pipeline(intermediates: pd.DataFrame) -> dict:
    intermediates = intermediates.copy()
    intermediates["game_date"] = pd.to_datetime(intermediates["game_date"])

    train_mask = intermediates.game_date < pd.Timestamp("2025-03-01")
    test_mask = ~train_mask
    train = intermediates[train_mask].reset_index(drop=True)
    test = intermediates[test_mask].reset_index(drop=True)
    actuals_train = train.outcome.to_numpy(dtype=float)
    actuals_test = test.outcome.to_numpy(dtype=float)
    print(f"\nTrain (Jan-Feb 2025): {len(train):,} rows | "
          f"Test (Mar-Apr 2025): {len(test):,} rows")

    # Global fit (all stats pooled) — keeps a single fallback calibration
    # for any stat that's too small to refit on its own.
    print("\nFitting GLOBAL calibration (all stats pooled)…")
    global_fit = fit_one(train, test, "global")

    # Per-stat fits — one isotonic per stat type, on rows for that stat only.
    # Each stat has its own distribution shape and bias, so a stat-specific
    # mapping should outperform the global one wherever sample is sufficient.
    print("\nFitting PER-STAT calibrations…")
    per_stat: dict[str, dict] = {}
    for stat in sorted(intermediates["stat"].unique()):
        train_stat = train[train["stat"] == stat].reset_index(drop=True)
        test_stat = test[test["stat"] == stat].reset_index(drop=True)
        fit = fit_one(train_stat, test_stat, stat)
        if fit is not None:
            per_stat[stat] = fit

    # Compare global vs per-stat on the test split using each strategy's
    # best (tuned + isotonic) — this is the number that production sees.
    actuals_test = test.outcome.to_numpy(dtype=float)
    if global_fit is not None:
        global_iso_brier = global_fit["test_brier"]["isotonic"]
        # Compute per-stat ensemble Brier — for each test row, use that stat's
        # isotonic if available, else global.
        per_stat_preds = np.empty(len(test), dtype=float)
        for stat, fit in per_stat.items():
            mask = (test["stat"] == stat).to_numpy()
            if not mask.any():
                continue
            tuned = apply_blend(
                test[mask], fit["base_rate"], fit["shrinkage"], fit["blend_model"]
            )
            per_stat_preds[mask] = _apply_isotonic_vec(
                tuned, fit["isotonic_x_thresholds"], fit["isotonic_y_thresholds"]
            )
        # Stats that didn't refit (e.g. too small) fall back to global
        unfilled = ~test["stat"].isin(per_stat).to_numpy()
        if unfilled.any():
            tuned = apply_blend(
                test[unfilled], global_fit["base_rate"],
                global_fit["shrinkage"], global_fit["blend_model"],
            )
            per_stat_preds[unfilled] = _apply_isotonic_vec(
                tuned, global_fit["isotonic_x_thresholds"],
                global_fit["isotonic_y_thresholds"],
            )
        per_stat_brier = brier(per_stat_preds, actuals_test)
        print(f"\nGlobal-only iso Brier:  {global_iso_brier:.4f}")
        print(f"Per-stat ensemble Brier: {per_stat_brier:.4f}  "
              f"(Δ {per_stat_brier - global_iso_brier:+.4f})")

    return {
        "global": global_fit,
        "per_stat": per_stat,
        "n_train": len(train),
        "n_test": len(test),
    }


def _apply_isotonic_vec(probs: np.ndarray, x_thresh: list[float], y_thresh: list[float]) -> np.ndarray:
    """Vectorized version of nba_props_cache._apply_isotonic — for the
    pipeline test-Brier comparison only. Production code uses the scalar
    version that ships in nba_props_cache.py.
    """
    if not x_thresh or not y_thresh:
        return probs
    return np.interp(probs, x_thresh, y_thresh)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--recompute", action="store_true",
                   help="Rebuild the intermediates parquet from scratch")
    p.add_argument("--season", default="2024-25")
    p.add_argument("--save", action="store_true",
                   help="Save tuned constants + isotonic mapping to "
                        f"{CALIBRATION_OUT.relative_to(REPO_ROOT)}")
    args = p.parse_args()

    if args.recompute or not INTERMEDIATES_PARQUET.exists():
        df = build_intermediates_table(season=args.season)
    else:
        print(f"Loading existing {INTERMEDIATES_PARQUET.name} (use --recompute to rebuild)")
        df = pd.read_parquet(INTERMEDIATES_PARQUET)
    print(f"Intermediates: {len(df):,} rows")

    result = run_pipeline(df)

    if args.save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        global_fit = result["global"]
        if global_fit is None:
            print("\nGlobal fit produced no result — refusing to save.")
            return 1
        payload = {
            "fitted_on": args.season,
            "schema_version": 2,
            # Top-level keys preserved for backwards compatibility with
            # nba_props_cache._load_calibration. New code should read from
            # `per_stat[stat]` first and fall back to `global`.
            "base_rate": global_fit["base_rate"],
            "shrinkage": global_fit["shrinkage"],
            "blend_model": global_fit["blend_model"],
            "isotonic_x_thresholds": global_fit["isotonic_x_thresholds"],
            "isotonic_y_thresholds": global_fit["isotonic_y_thresholds"],
            "global": global_fit,
            "per_stat": result["per_stat"],
            "test_brier": global_fit["test_brier"],
            "n_train": result["n_train"],
            "n_test": result["n_test"],
        }
        CALIBRATION_OUT.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved calibration to {CALIBRATION_OUT.relative_to(REPO_ROOT)}")
        print(f"  global + {len(result['per_stat'])} per-stat blocks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
