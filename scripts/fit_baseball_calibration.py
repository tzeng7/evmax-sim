"""Fit per-sector isotonic calibration for the baseball ensemble.

Symptom (multi-season backtest after pitcher integration):
  2025 ensemble Brier: 0.2433  vs baseline 0.2473  (BEATS by 0.004)
  2024 ensemble Brier: 0.2552  vs baseline 0.2498  (LOSES  by 0.005)
The model is over-confident in years with low home-field advantage
(2024 home WP was 51.6% — model was tuned for ~54%). Per-bucket gaps in
2024 reach ±10-19% in the high-probability ranges. Isotonic calibration
fits a monotonic correction that compresses these systematic biases.

This script:
  1. Runs walk-forward backtest on the training year (default: 2324 = 2024)
  2. Collects (ensemble_prob_home, home_won) pairs
  3. Fits isotonic regression via the existing ModelCalibrator under the key
     "baseball_ensemble" — picked up automatically by EnsembleModelAgent._blend
  4. Validates on holdout year (default: 2425 = 2025): reports Brier
     before/after on BOTH years (training year shows fit quality;
     holdout shows generalization)

Promotion rule: ≥0.001 holdout Brier improvement to keep the fitted
calibration. Below that, the previous state is restored. This protects
against overfit-on-train-but-broken-on-holdout outcomes.

Usage:
    # Single-season train + holdout (recommended once you have 3+ seasons)
    .venv/bin/python scripts/fit_baseball_calibration.py --train 2324 --validate 2425

    # Multi-season combined fit + report on each component year
    # (recommended now while we only have 2 seasons of data — single-year
    # fits don't generalize because each year has idiosyncratic biases)
    .venv/bin/python scripts/fit_baseball_calibration.py --train 2324+2425 --report-each

    .venv/bin/python scripts/fit_baseball_calibration.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running the script directly (no package install path manipulation needed
# in editable installs, but defensive for foreign environments).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evmax.agents.models.calibration import CALIBRATION_PATH, ModelCalibrator  # noqa: E402
from evmax.backtest.engine import WALKFORWARD_MONTHS  # noqa: E402
from evmax.backtest.sources.espn_walkforward import run_walkforward  # noqa: E402

CALIBRATION_KEY = "baseball_ensemble"
PROMOTION_BAR = 0.001


def _collect(season_code: str) -> tuple[list[float], list[int]]:
    """Run walk-forward on a season and return (ensemble_probs, home_won) pairs.

    Only games where the ensemble produced a prediction are included — these
    are the data points the calibration will actually transform at runtime.
    """
    months = WALKFORWARD_MONTHS.get(season_code, {}).get("baseball")
    if not months:
        raise ValueError(
            f"No baseball months mapped for season code {season_code!r}. "
            f"Add it to WALKFORWARD_MONTHS in evmax/backtest/engine.py."
        )

    print(f"  fetching games + boxscores for season {season_code} (months: {months[0]}-{months[-1]}) ...")
    report = run_walkforward("baseball", months)

    probs: list[float] = []
    outcomes: list[int] = []
    for r in report.results:
        if r.ensemble_prob_home is None:
            continue
        probs.append(r.ensemble_prob_home)
        outcomes.append(1 if r.home_won else 0)
    print(f"  {len(probs)} predictions collected")
    return probs, outcomes


def _brier(probs: list[float], outcomes: list[int]) -> float:
    if not probs:
        return float("nan")
    return sum((p - o) ** 2 for p, o in zip(probs, outcomes)) / len(probs)


def _accuracy(probs: list[float], outcomes: list[int]) -> float:
    if not probs:
        return float("nan")
    correct = sum(1 for p, o in zip(probs, outcomes) if (p >= 0.5) == bool(o))
    return correct / len(probs)


def _restore_prev_state(prev_entry):
    current_state = json.loads(CALIBRATION_PATH.read_text())
    if prev_entry is None:
        current_state.pop(CALIBRATION_KEY, None)
    else:
        current_state[CALIBRATION_KEY] = prev_entry
    CALIBRATION_PATH.write_text(json.dumps(current_state, indent=2))


def main(train_seasons: list[str], val_season: str, report_each: bool, dry_run: bool) -> int:
    multi_train = len(train_seasons) > 1

    print("=== fit_baseball_calibration ===")
    print(f"train seasons: {train_seasons}")
    if not multi_train:
        print(f"val   season:  {val_season}")
    print()

    # Stash previous calibration so we can revert on bad fit.
    prev_state = (
        json.loads(CALIBRATION_PATH.read_text())
        if CALIBRATION_PATH.exists()
        else {}
    )
    prev_baseball_entry = prev_state.get(CALIBRATION_KEY)

    # Collect uncalibrated predictions from each training season, concat them.
    print(f"[1/3] collecting training predictions")
    train_probs: list[float] = []
    train_outcomes: list[int] = []
    per_season_data: dict[str, tuple[list[float], list[int]]] = {}
    for season in train_seasons:
        print(f"  walk-forward {season} ...")
        probs, outcomes = _collect(season)
        per_season_data[season] = (probs, outcomes)
        train_probs.extend(probs)
        train_outcomes.extend(outcomes)
        seg_brier = _brier(probs, outcomes)
        print(f"    {season}: n={len(probs)}  uncalibrated_Brier={seg_brier:.5f}")
    print(f"  combined training set: n={len(train_probs)}\n")

    # Validation set:
    # - For multi-season training, we don't have a true holdout; instead we
    #   verify that the fit doesn't make any individual training year worse.
    #   The promotion bar is applied to the WORST per-season improvement.
    # - For single-season training, run the holdout walk-forward.
    if multi_train:
        val_probs = train_probs  # used only for sanity printout below
        val_outcomes = train_outcomes
        val_baseline_brier = _brier(val_probs, val_outcomes)
    else:
        print(f"[2/3] validation walk-forward ({val_season})")
        val_probs, val_outcomes = _collect(val_season)
        val_baseline_brier = _brier(val_probs, val_outcomes)
        val_baseline_acc = _accuracy(val_probs, val_outcomes)
        print(f"  baseline Brier: {val_baseline_brier:.5f}  Acc: {val_baseline_acc:.1%}\n")

    if dry_run:
        print("[dry-run] skipping fit + write\n")
        return 0

    # Fit isotonic on combined training set.
    print(f"[3/3] fitting isotonic calibration (key={CALIBRATION_KEY!r})")
    cal = ModelCalibrator()
    cal._calibrations.pop(CALIBRATION_KEY, None)
    ok = cal.retrain(CALIBRATION_KEY, train_probs, train_outcomes)
    if not ok:
        print("  retrain returned False (insufficient data or sklearn missing). Aborting.")
        return 1
    summary = cal.summary().get(CALIBRATION_KEY, {})
    print(f"  fit summary: {summary}\n")

    # Per-season report: how does the fit affect each year individually?
    if report_each or multi_train:
        print("Per-season impact of the fitted calibration:")
        worst_delta = float("inf")
        for season, (probs, outcomes) in per_season_data.items():
            raw = _brier(probs, outcomes)
            cal_p = [cal.calibrate(CALIBRATION_KEY, p) for p in probs]
            cal_b = _brier(cal_p, outcomes)
            delta = raw - cal_b
            worst_delta = min(worst_delta, delta)
            mark = "✓" if delta >= PROMOTION_BAR else "✗"
            print(
                f"  [{mark}] {season}: raw={raw:.5f}  calibrated={cal_b:.5f}  Δ={delta:+.5f}"
            )
        print()
        promote_delta = worst_delta
    else:
        cal_val_probs = [cal.calibrate(CALIBRATION_KEY, p) for p in val_probs]
        cal_val_brier = _brier(cal_val_probs, val_outcomes)
        promote_delta = val_baseline_brier - cal_val_brier
        cal_val_acc = _accuracy(cal_val_probs, val_outcomes)
        print(f"  uncalibrated val Brier: {val_baseline_brier:.5f}")
        print(f"  calibrated   val Brier: {cal_val_brier:.5f}  Acc: {cal_val_acc:.1%}")
        print(f"  Δ: {promote_delta:+.5f}\n")

    # Promotion gate. For multi-season we use the WORST per-season improvement —
    # we want the calibration to help every year, not on average.
    if promote_delta >= PROMOTION_BAR:
        print(
            f"PROMOTE: {'worst-season' if multi_train else 'holdout'} Δ "
            f"{promote_delta:+.5f} ≥ +{PROMOTION_BAR} bar. Calibration kept."
        )
        print(
            f"  Live ensemble will apply this calibration to baseball blends "
            f"automatically (key {CALIBRATION_KEY!r} read on init)."
        )
        return 0
    print(
        f"REVERT: Δ {promote_delta:+.5f} < +{PROMOTION_BAR} bar. "
        f"Restoring previous calibration state."
    )
    _restore_prev_state(prev_baseball_entry)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", default="2324+2425",
        help="Training season(s); use '+' for multi-season combined fit "
        "(default: 2324+2425). Single season: '2324'.",
    )
    parser.add_argument("--validate", default="2425", help="Holdout season (only used when --train is a single season)")
    parser.add_argument(
        "--report-each", action="store_true",
        help="Always show per-season impact, even for single-season training.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    train_seasons = [s.strip() for s in args.train.split("+") if s.strip()]
    sys.exit(main(train_seasons, args.validate, args.report_each, args.dry_run))
