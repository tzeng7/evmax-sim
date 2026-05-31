"""Sweep WNBA ensemble weights against the 2025 walk-forward season.

Runs `run_walkforward("wnba", ...)` once to materialise per-game per-model
probabilities (cached to a pickle), then re-blends offline across a grid of
weight combinations and reports Brier / accuracy / log-loss for each.

Why offline re-blend: each walk-forward call hits ESPN's scoreboard for ~600
WNBA games and re-trains Elo/Form/Efficiency from scratch. Model probs don't
change with the *blend*, so we cache them once and only the weighted sum
changes per combo.

Usage:
    python scripts/sweep_wnba_weights.py
    python scripts/sweep_wnba_weights.py --seasons 2425        # 2025 only (default)
    python scripts/sweep_wnba_weights.py --seasons 2324,2425   # 2024+2025
    python scripts/sweep_wnba_weights.py --refresh             # ignore cached pickle
    python scripts/sweep_wnba_weights.py --top 20              # print top-N combos
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
from itertools import product
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from evmax.backtest.engine import WALKFORWARD_MONTHS
from evmax.backtest.sources.espn_walkforward import (
    WalkForwardResult,
    run_walkforward,
)

CACHE_DIR = _REPO_ROOT / "data" / "backtest" / "wnba_sweep"

# Live blend (mirrors WNBA_WEIGHT_OVERRIDES + ensemble_agent.py)
LIVE_BLEND = {
    "elo": 0.15,
    "form": 0.0,
    "wnba_efficiency": 0.40,
    "wnba_possession_sim": 0.45,
}

# Grid: each weight independently scanned; combos normalized to sum 1.
# poisson held at 0 (matches live override).
GRID = {
    "elo":                 [0.10, 0.20, 0.30, 0.40, 0.50],
    "form":                [0.00, 0.10, 0.15, 0.20, 0.25],
    "wnba_efficiency":     [0.10, 0.20, 0.30, 0.40],
    "wnba_possession_sim": [0.10, 0.20, 0.30, 0.40, 0.50],
}


def cache_path(seasons: list[str]) -> Path:
    return CACHE_DIR / f"results_{'_'.join(seasons)}.pkl"


def load_or_run(seasons: list[str], refresh: bool) -> list[WalkForwardResult]:
    """Run walk-forward for each season and cache to pickle."""
    cp = cache_path(seasons)
    if cp.exists() and not refresh:
        print(f"[cache] loading {cp}")
        with open(cp, "rb") as f:
            return pickle.load(f)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    all_results: list[WalkForwardResult] = []
    for season in seasons:
        months = WALKFORWARD_MONTHS[season]["wnba"]
        print(f"[run] season {season} → months {months[0]}..{months[-1]}")
        report = run_walkforward("wnba", months)
        print(f"  {report.n_games} games, {report.n_predicted} predicted")
        all_results.extend(report.results)

    with open(cp, "wb") as f:
        pickle.dump(all_results, f)
    print(f"[cache] wrote {cp} ({len(all_results)} games)")
    return all_results


def blend_one(r: WalkForwardResult, weights: dict[str, float]) -> float | None:
    """Weighted average of available model probs. Skips models with None."""
    pieces = [
        (r.elo_prob_home, weights.get("elo", 0.0)),
        (r.form_prob_home, weights.get("form", 0.0)),
        (r.efficiency_prob_home, weights.get("wnba_efficiency", 0.0)),
        (r.possession_sim_prob_home, weights.get("wnba_possession_sim", 0.0)),
    ]
    total_w = 0.0
    weighted_sum = 0.0
    for p, w in pieces:
        if p is None or w <= 0:
            continue
        total_w += w
        weighted_sum += p * w
    if total_w <= 0:
        return None
    return weighted_sum / total_w


def evaluate(results: list[WalkForwardResult], weights: dict[str, float]) -> dict:
    n = 0
    brier_sum = 0.0
    correct = 0
    log_loss_sum = 0.0
    eps = 1e-12
    for r in results:
        p = blend_one(r, weights)
        if p is None:
            continue
        y = 1.0 if r.home_won else 0.0
        n += 1
        brier_sum += (p - y) ** 2
        if (p >= 0.5) == r.home_won:
            correct += 1
        p_clip = max(min(p, 1 - eps), eps)
        log_loss_sum += -(y * math.log(p_clip) + (1 - y) * math.log(1 - p_clip))
    if n == 0:
        return {"n": 0, "brier": float("nan"), "accuracy": float("nan"), "logloss": float("nan")}
    return {
        "n": n,
        "brier": brier_sum / n,
        "accuracy": correct / n,
        "logloss": log_loss_sum / n,
    }


def normalize(weights: dict[str, float]) -> dict[str, float]:
    s = sum(weights.values())
    if s <= 0:
        return weights
    return {k: v / s for k, v in weights.items()}


def grid_combos():
    """Cartesian product over GRID, dropping all-zero combos and combos whose
    pre-norm sum is wildly off (helps trim degenerate combos)."""
    keys = list(GRID.keys())
    vals = [GRID[k] for k in keys]
    out = []
    for combo in product(*vals):
        d = dict(zip(keys, combo))
        s = sum(d.values())
        if s <= 0:
            continue
        # Skip combos whose unnormalized sum is too far from 1 — keeps the
        # set within "reasonable" relative-weight space and avoids blowing
        # up small models.
        if s < 0.5 or s > 1.6:
            continue
        out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2425", help="comma-separated season codes")
    ap.add_argument("--refresh", action="store_true", help="ignore cached pickle")
    ap.add_argument("--top", type=int, default=15, help="print top-N combos by Brier")
    args = ap.parse_args()

    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    results = load_or_run(seasons, args.refresh)

    # Filter to games where at least one WNBA-specific model fired — that's
    # the realistic "live" subset (Elo/Form alone can predict day-1 games but
    # the live blend gates on confidence ≥ 0.45 anyway).
    usable = [r for r in results if r.efficiency_prob_home is not None or r.possession_sim_prob_home is not None]
    print(f"\nWNBA games eval pool: {len(usable)} / {len(results)} (have at least one WNBA-specific model)\n")

    # Baseline: live blend
    live_norm = normalize(LIVE_BLEND)
    live_eval = evaluate(usable, live_norm)
    print("=" * 78)
    print(f"LIVE blend  {live_norm}")
    print(f"  N={live_eval['n']}  Brier={live_eval['brier']:.4f}  Acc={live_eval['accuracy']:.3f}  LogLoss={live_eval['logloss']:.4f}")
    print("=" * 78)

    # Sweep
    rows = []
    for w in grid_combos():
        wn = normalize(w)
        ev = evaluate(usable, wn)
        if ev["n"] == 0:
            continue
        rows.append((ev["brier"], ev["accuracy"], ev["logloss"], ev["n"], wn))

    rows.sort(key=lambda x: x[0])  # ascending Brier (lower is better)
    print(f"\nTop {args.top} weight combos by Brier (lower = better):\n")
    header = f"{'Brier':>7}  {'Acc':>5}  {'LogLoss':>7}  {'N':>4}   weights"
    print(header)
    print("-" * len(header))
    for brier, acc, ll, n, w in rows[: args.top]:
        wstr = "  ".join(f"{k}={v:.2f}" for k, v in w.items())
        print(f"{brier:.4f}  {acc:.3f}  {ll:.4f}  {n:>4}   {wstr}")

    print("\nBaseline (live):")
    wstr = "  ".join(f"{k}={v:.2f}" for k, v in live_norm.items())
    print(f"{live_eval['brier']:.4f}  {live_eval['accuracy']:.3f}  {live_eval['logloss']:.4f}  {live_eval['n']:>4}   {wstr}")


if __name__ == "__main__":
    main()
