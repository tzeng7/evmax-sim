"""Fit per-sector isotonic calibration for the NBA ensemble.

Symptom (4w live metrics, 153 resolved bets):
  <20% bucket (n=11): predicted 16.6%, actual 27.3%   (gap −10.7pp)
  >80% bucket (n=17): predicted 90.7%, actual 82.4%   (gap +8.3pp)
The blend is over-confident at BOTH tails — classic calibration miss that
isotonic regression should compress monotonically toward 50%.

This script:
  1. Walk-forward backtest on training season(s) (default 2425 = 2024-25)
  2. Collect (model-only ensemble_prob_home, home_won) pairs
  3. Fit isotonic regression via ModelCalibrator under key "nba_ensemble"
     — picked up automatically by EnsembleModelAgent._apply_sector_calibration
  4. Validate on holdout season (default 2526 = 2025-26): reports Brier
     before / after, with per-bucket bias

Promotion rule: ≥0.001 holdout Brier improvement to keep the fitted
calibration. Below that, the previous state is restored.

Usage:
    .venv/bin/python scripts/fit_nba_calibration.py
    .venv/bin/python scripts/fit_nba_calibration.py --train 2425 --validate 2526
    .venv/bin/python scripts/fit_nba_calibration.py --train 2324+2425 --report-each
    .venv/bin/python scripts/fit_nba_calibration.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evmax.agents.models.calibration import CALIBRATION_PATH, ModelCalibrator  # noqa: E402
from evmax.backtest.engine import WALKFORWARD_MONTHS  # noqa: E402
from evmax.backtest.sources.espn_walkforward import run_walkforward  # noqa: E402

CALIBRATION_KEY = "nba_ensemble"
PROMOTION_BAR = 0.001
BUCKETS = [(0.0, 0.20, "<20%"), (0.20, 0.40, "20-40%"),
           (0.40, 0.60, "40-60%"), (0.60, 0.80, "60-80%"),
           (0.80, 1.01, ">80%")]


def _collect(season_code: str) -> tuple[list[float], list[int]]:
    months = WALKFORWARD_MONTHS.get(season_code, {}).get("nba")
    if not months:
        raise ValueError(
            f"No nba months mapped for season code {season_code!r}. "
            f"Add it to WALKFORWARD_MONTHS in evmax/backtest/engine.py."
        )

    print(f"  walk-forward {season_code} (months {months[0]}-{months[-1]}) ...")
    report = run_walkforward("nba", months)

    probs: list[float] = []
    outcomes: list[int] = []
    for r in report.results:
        if r.ensemble_prob_home is None:
            continue
        probs.append(r.ensemble_prob_home)
        outcomes.append(1 if r.home_won else 0)
    print(f"    {len(probs)} predictions collected")
    return probs, outcomes


def _brier(probs: list[float], outcomes: list[int]) -> float:
    if not probs:
        return float("nan")
    return sum((p - o) ** 2 for p, o in zip(probs, outcomes)) / len(probs)


def _accuracy(probs: list[float], outcomes: list[int]) -> float:
    if not probs:
        return float("nan")
    return sum(1 for p, o in zip(probs, outcomes) if (p >= 0.5) == bool(o)) / len(probs)


def _bucket_report(probs: list[float], outcomes: list[int], label: str) -> None:
    print(f"  {label} per-bucket bias:")
    for lo, hi, blabel in BUCKETS:
        sub = [(p, o) for p, o in zip(probs, outcomes) if lo <= p < hi]
        if not sub:
            continue
        n = len(sub)
        avg_p = sum(p for p, _ in sub) / n
        avg_o = sum(o for _, o in sub) / n
        bias = avg_p - avg_o
        marker = "  ⚠" if abs(bias) >= 0.05 else ""
        print(f"    {blabel:<8} N={n:<4} pred={avg_p:.1%}  actual={avg_o:.1%}  bias={bias*100:+.1f}pp{marker}")


def _restore_prev_state(prev_entry):
    current_state = json.loads(CALIBRATION_PATH.read_text())
    if prev_entry is None:
        current_state.pop(CALIBRATION_KEY, None)
    else:
        current_state[CALIBRATION_KEY] = prev_entry
    CALIBRATION_PATH.write_text(json.dumps(current_state, indent=2))


def main(train_seasons: list[str], val_season: str, report_each: bool, dry_run: bool) -> int:
    multi_train = len(train_seasons) > 1

    print("=== fit_nba_calibration ===")
    print(f"train seasons: {train_seasons}")
    if not multi_train:
        print(f"val   season:  {val_season}")
    print()

    prev_state = (
        json.loads(CALIBRATION_PATH.read_text())
        if CALIBRATION_PATH.exists()
        else {}
    )
    prev_nba_entry = prev_state.get(CALIBRATION_KEY)

    print(f"[1/3] collecting training predictions")
    train_probs: list[float] = []
    train_outcomes: list[int] = []
    per_season_data: dict[str, tuple[list[float], list[int]]] = {}
    for season in train_seasons:
        probs, outcomes = _collect(season)
        per_season_data[season] = (probs, outcomes)
        train_probs.extend(probs)
        train_outcomes.extend(outcomes)
        seg_brier = _brier(probs, outcomes)
        print(f"    {season}: n={len(probs)}  uncalibrated_Brier={seg_brier:.5f}")
    print(f"  combined training set: n={len(train_probs)}\n")

    # Always run held-out validation (separate from training years) — the
    # NBA model's calibration profile shifted meaningfully between 2425
    # and 2526 after the efficiency/possession-sim upgrades, so a fit
    # that improves training years can still hurt the current holdout.
    # Promotion requires BOTH per-season improvement AND held-out gain.
    holdout_in_train = val_season in train_seasons
    val_probs: list[float] = []
    val_outcomes: list[int] = []
    val_baseline_brier = float("nan")
    if not holdout_in_train:
        print(f"[2/3] held-out validation walk-forward ({val_season})")
        val_probs, val_outcomes = _collect(val_season)
        val_baseline_brier = _brier(val_probs, val_outcomes)
        print(f"  baseline Brier: {val_baseline_brier:.5f}  Acc: {_accuracy(val_probs, val_outcomes):.1%}")
        _bucket_report(val_probs, val_outcomes, "baseline")
        print()
    else:
        print(f"[2/3] holdout {val_season} is in training set — skipping separate holdout")
        print()

    if dry_run:
        print("[dry-run] skipping fit + write\n")
        return 0

    print(f"[3/3] fitting isotonic calibration (key={CALIBRATION_KEY!r})")
    cal = ModelCalibrator()
    cal._calibrations.pop(CALIBRATION_KEY, None)
    ok = cal.retrain(CALIBRATION_KEY, train_probs, train_outcomes)
    if not ok:
        print("  retrain returned False (insufficient data or sklearn missing). Aborting.")
        return 1
    summary = cal.summary().get(CALIBRATION_KEY, {})
    print(f"  fit summary: {summary}\n")

    deltas: list[float] = []
    if report_each or multi_train:
        print("Per-season impact of the fitted calibration:")
        for season, (probs, outcomes) in per_season_data.items():
            raw = _brier(probs, outcomes)
            cal_p = [cal.calibrate(CALIBRATION_KEY, p) for p in probs]
            cal_b = _brier(cal_p, outcomes)
            delta = raw - cal_b
            deltas.append(delta)
            mark = "✓" if delta >= PROMOTION_BAR else "✗"
            print(f"  [{mark}] {season}: raw={raw:.5f}  calibrated={cal_b:.5f}  Δ={delta:+.5f}")
        print()

    if val_probs:
        cal_val_probs = [cal.calibrate(CALIBRATION_KEY, p) for p in val_probs]
        cal_val_brier = _brier(cal_val_probs, val_outcomes)
        holdout_delta = val_baseline_brier - cal_val_brier
        deltas.append(holdout_delta)
        mark = "✓" if holdout_delta >= PROMOTION_BAR else "✗"
        print(f"Held-out validation ({val_season}):")
        print(f"  [{mark}] uncalibrated val Brier: {val_baseline_brier:.5f}")
        print(f"        calibrated   val Brier: {cal_val_brier:.5f}  Acc: {_accuracy(cal_val_probs, val_outcomes):.1%}")
        print(f"        Δ: {holdout_delta:+.5f}")
        _bucket_report(cal_val_probs, val_outcomes, "calibrated")
        print()

    # Promote only when EVERY measured slice (training years + holdout)
    # clears the bar — guards against fits that win on training data but
    # hurt the current model state on held-out games.
    promote_delta = min(deltas) if deltas else 0.0

    if promote_delta >= PROMOTION_BAR:
        print(
            f"PROMOTE: worst slice (train + holdout) Δ "
            f"{promote_delta:+.5f} ≥ +{PROMOTION_BAR} bar. Calibration kept."
        )
        print(
            f"  Live ensemble will apply this calibration to NBA blends "
            f"automatically (key {CALIBRATION_KEY!r} read on init)."
        )
        return 0
    print(
        f"REVERT: Δ {promote_delta:+.5f} < +{PROMOTION_BAR} bar. "
        f"Restoring previous calibration state."
    )
    _restore_prev_state(prev_nba_entry)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", default="2425",
        help="Training season(s); use '+' for multi-season combined fit (default: 2425).",
    )
    parser.add_argument("--validate", default="2526", help="Holdout season (single-season training only)")
    parser.add_argument("--report-each", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    train_seasons = [s.strip() for s in args.train.split("+") if s.strip()]
    sys.exit(main(train_seasons, args.validate, args.report_each, args.dry_run))
