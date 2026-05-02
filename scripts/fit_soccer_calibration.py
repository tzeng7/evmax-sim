"""Fit per-sector isotonic calibration for the soccer ensemble.

Symptom (live predictions.db, n=89 resolved bets through 2026-04-30):
  bucket 0.35–0.50: predicted 42.4%, actual 30.0%   (gap −12.4pp, n=30)
  bucket 0.65+   :  predicted 69.6%, actual 57.1%   (gap −12.4pp, n=7)
  bucket 0.00–0.20: predicted 15.2%, actual  6.3%   (gap  −8.9pp, n=16)
The blended ensemble systematically pushes team-win probabilities up. Draws
are calibrated (8 bets, +0.4pp), so the bias is on the team-win legs only.

This script:
  1. Loads BacktestRows from football-data.co.uk CSVs (the same source the
     walk-forward backtest uses) for all five top-tier European leagues
     across the configured train and validation seasons.
  2. Walks them chronologically through fresh Elo/Form/Poisson agents,
     producing the SAME blended_ensemble_3way prediction the live
     EnsembleModelAgent emits for soccer (modulo xG, which isn't in the
     walk-forward data — the bias signal is the same).
  3. Collects (model_team_win_prob, team_won) pairs from BOTH legs of each
     game (home leg + away leg) — draws are dropped from the fit because
     they're already calibrated and a single curve can't carry mismatched
     bias.
  4. Fits isotonic regression via the existing ModelCalibrator under the
     key "soccer_ensemble" — picked up automatically by
     EnsembleModelAgent._apply_sector_calibration which now handles 3-way
     by calibrating home and away legs, leaving draw untouched, and
     renormalizing.
  5. Validates on the held-out season(s): reports Brier on the team-win
     leg before/after.

Promotion rule: ≥0.001 Brier improvement on the held-out years to keep the
fitted calibration. Below that, the script restores the previous state.

Usage:
    .venv/bin/python scripts/fit_soccer_calibration.py
    .venv/bin/python scripts/fit_soccer_calibration.py --train 2324,2425 --validate 2526
    .venv/bin/python scripts/fit_soccer_calibration.py --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evmax.agents.models.calibration import CALIBRATION_PATH, ModelCalibrator
from evmax.backtest.loader import SOCCER_LEAGUES, fetch_soccer_csv
from evmax.backtest.models import BacktestRow
from evmax.backtest.sources.soccer_csv import parse_soccer_csv
from evmax.backtest.sources.soccer_walkforward import (
    blended_ensemble_3way,
    walk_forward_predictions,
)

CALIBRATION_KEY = "soccer_ensemble"
# Tennis used 0.001 (3× the observed +0.00154 Brier delta on its holdout).
# We keep the same bar — soccer's per-row Brier is ~0.21 with sample size
# in the same order of magnitude, so the standard error and signal strength
# scale similarly.
PROMOTION_BAR = 0.001


def _load_rows(seasons: list[str]) -> list[BacktestRow]:
    """Fetch + parse all top-5 league CSVs for the given seasons."""
    rows: list[BacktestRow] = []
    for season in seasons:
        for league in SOCCER_LEAGUES:
            try:
                path = fetch_soccer_csv(season, league)
            except Exception as e:
                print(f"  skip {season}/{league}: {e}")
                continue
            rows.extend(parse_soccer_csv(path, league, season))
    return rows


def _collect_team_win_pairs(
    rows: list[BacktestRow],
) -> tuple[list[float], list[int]]:
    """Walk-forward predict, then emit (prob, won) pairs for the home and
    away legs of each game. Draws are dropped from the fit — calibrated."""
    predictions = walk_forward_predictions(rows)
    probs: list[float] = []
    outcomes: list[int] = []
    for r in predictions:
        ph, _pd, pa = blended_ensemble_3way(r)
        if ph is None or pa is None:
            continue
        # Home leg: prob_h vs (result == "H")
        probs.append(ph)
        outcomes.append(1 if r.result == "H" else 0)
        # Away leg: prob_a vs (result == "A")
        probs.append(pa)
        outcomes.append(1 if r.result == "A" else 0)
    return probs, outcomes


def _brier(probs: list[float], outcomes: list[int]) -> float:
    return sum((p - o) ** 2 for p, o in zip(probs, outcomes)) / max(len(probs), 1)


def _apply_calibration(probs: list[float], cal: ModelCalibrator) -> list[float]:
    return [cal.calibrate(CALIBRATION_KEY, p) for p in probs]


_BUCKETS = [(0.0, 0.20), (0.20, 0.35), (0.35, 0.50), (0.50, 0.65), (0.65, 1.00)]


def _bucket_calibration(
    probs: list[float], outcomes: list[int]
) -> list[tuple[str, int, float, float, float]]:
    """Return (label, n, mean_pred, actual_rate, gap) per probability bucket."""
    out = []
    for lo, hi in _BUCKETS:
        idx = [i for i, p in enumerate(probs) if lo <= p < hi]
        if not idx:
            continue
        n = len(idx)
        mean_pred = sum(probs[i] for i in idx) / n
        actual_rate = sum(outcomes[i] for i in idx) / n
        out.append(
            (f"{lo:.2f}-{hi:.2f}", n, mean_pred, actual_rate, actual_rate - mean_pred)
        )
    return out


def _print_bucket_table(label: str, probs: list[float], outcomes: list[int]) -> None:
    print(f"  {label}")
    print(f"  {'bucket':<11} {'n':>6} {'pred':>7} {'actual':>7} {'gap':>7}")
    for tag, n, p, a, g in _bucket_calibration(probs, outcomes):
        print(f"  {tag:<11} {n:>6d} {p:>7.3f} {a:>7.3f} {g:>+7.3f}")


def main(train_seasons: list[str], val_seasons: list[str], dry_run: bool) -> None:
    print("=== fit_soccer_calibration ===")
    print(f"train seasons: {train_seasons}")
    print(f"val   seasons: {val_seasons}\n")

    # Stash existing calibration so we can revert on a bad fit
    prev_state = json.loads(CALIBRATION_PATH.read_text()) if CALIBRATION_PATH.exists() else {}
    prev_entry = prev_state.get(CALIBRATION_KEY)

    print("loading training rows ...")
    train_rows = _load_rows(train_seasons)
    print(f"  {len(train_rows)} rows from {train_seasons}")
    print("collecting training predictions (walk-forward) ...")
    train_probs, train_outcomes = _collect_team_win_pairs(train_rows)
    print(f"  {len(train_probs)} team-win pairs collected\n")

    print("loading validation rows ...")
    val_rows = _load_rows(val_seasons)
    print(f"  {len(val_rows)} rows from {val_seasons}")
    print("collecting validation predictions (walk-forward) ...")
    val_probs, val_outcomes = _collect_team_win_pairs(val_rows)
    print(f"  {len(val_probs)} team-win pairs collected\n")

    base_brier = _brier(val_probs, val_outcomes)
    print(f"baseline (uncalibrated) holdout Brier: {base_brier:.5f}")
    _print_bucket_table("baseline buckets:", val_probs, val_outcomes)
    print()

    if dry_run:
        print("[dry-run] skipping fit + write")
        return

    cal = ModelCalibrator()
    # Strip any prior soccer_ensemble entry first so the fit sees the raw
    # input (calibrate() returns identity when no entry is registered).
    cal._calibrations.pop(CALIBRATION_KEY, None)
    ok = cal.retrain(CALIBRATION_KEY, train_probs, train_outcomes)
    if not ok:
        print("retrain returned False (insufficient data or sklearn missing). Aborting.")
        return

    summary = cal.summary().get(CALIBRATION_KEY, {})
    print(f"fit summary: {summary}\n")

    # Reload from disk so calibrate() picks up the new mapping
    cal_eval = ModelCalibrator()
    cal_probs = _apply_calibration(val_probs, cal_eval)
    cal_brier = _brier(cal_probs, val_outcomes)
    delta = base_brier - cal_brier
    print(f"calibrated holdout Brier: {cal_brier:.5f}")
    print(f"baseline   holdout Brier: {base_brier:.5f}")
    print(
        f"Δ:                        {delta:+.5f}  "
        f"({'BETTER' if delta > 0 else 'WORSE'} by {abs(delta):.5f})\n"
    )
    _print_bucket_table("calibrated buckets:", cal_probs, val_outcomes)
    print()

    if delta >= PROMOTION_BAR:
        print(f"PROMOTE: Δ ≥ {PROMOTION_BAR}. Calibration kept.")
    else:
        print(
            f"REVERT: Δ < {PROMOTION_BAR} threshold "
            f"(actual {delta:+.5f}). Restoring previous calibration state."
        )
        new_state = json.loads(CALIBRATION_PATH.read_text())
        if prev_entry is None:
            new_state.pop(CALIBRATION_KEY, None)
        else:
            new_state[CALIBRATION_KEY] = prev_entry
        CALIBRATION_PATH.write_text(json.dumps(new_state, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        type=lambda s: [t.strip() for t in s.split(",")],
        default=["2324", "2425"],
        help="comma-separated season codes (e.g. 2324,2425)",
    )
    parser.add_argument(
        "--validate",
        type=lambda s: [t.strip() for t in s.split(",")],
        default=["2526"],
        help="comma-separated season codes",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(args.train, args.validate, args.dry_run)
