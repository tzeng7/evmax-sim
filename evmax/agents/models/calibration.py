"""Model calibration via isotonic regression.

Each model agent's raw probability output is passed through a learned
monotonic mapping (isotonic regression) that corrects systematic biases
like compression toward 50%.

Training:
  Collect (model_predicted_prob, actual_outcome) pairs from resolved predictions,
  fit sklearn IsotonicRegression, store the breakpoints in calibration.json.

Usage:
  calibrator = ModelCalibrator()
  calibrated_prob = calibrator.calibrate("elo", raw_prob)

Retraining:
  Call calibrator.retrain(model_name, probs, outcomes) with historical data, or
  `calibrator.retrain_all_from_db()` to refit every sector's `{sector}_ensemble`
  curve from resolved live predictions. The CLI entry point is
  `evmax cleanup recalibrate [--sector S] [--dry-run]`. For thin/early sectors,
  prefer the leakage-guarded point-in-time scripts (scripts/fit_<sector>_calibration.py).

  NOTE: the applied key MUST be `{sector}_ensemble` — the exact key
  EnsembleModelAgent._apply_sector_calibration looks up. A curve stored under any
  other key (e.g. a global "ensemble") is loaded but never applied.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

CALIBRATION_PATH = Path(__file__).resolve().parents[3] / "data" / "models" / "calibration.json"

# Minimum samples to train calibration
MIN_SAMPLES = 30


class ModelCalibrator:
    """Calibrates model probabilities using isotonic regression breakpoints."""

    def __init__(self) -> None:
        self._calibrations: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if CALIBRATION_PATH.exists():
            try:
                self._calibrations = json.loads(CALIBRATION_PATH.read_text())
                logger.info("calibration_loaded", models=list(self._calibrations.keys()))
            except Exception as e:
                logger.warning("calibration_load_failed", error=str(e))
                self._calibrations = {}

    def _save(self) -> None:
        CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        CALIBRATION_PATH.write_text(json.dumps(self._calibrations, indent=2))

    def calibrate(self, model_name: str, prob: float) -> float:
        """Apply calibration to a raw probability. Returns unchanged if no calibration exists."""
        cal = self._calibrations.get(model_name)
        if not cal:
            return prob

        x_breaks = cal["x"]
        y_breaks = cal["y"]

        if not x_breaks or len(x_breaks) < 2:
            return prob

        # Interpolate using breakpoints
        prob = max(x_breaks[0], min(x_breaks[-1], prob))
        return float(np.interp(prob, x_breaks, y_breaks))

    def _fit(self, probs: list[float], outcomes: list[int]) -> Optional[dict]:
        """Fit an isotonic curve and return its serialized form, or None.

        Pure: does NOT mutate state or touch disk. Returns None when there are
        fewer than MIN_SAMPLES points or sklearn is unavailable. Used by both
        ``retrain`` (which persists) and the ``dry_run`` path of
        ``retrain_all_from_db`` (which does not).
        """
        if len(probs) < MIN_SAMPLES:
            return None
        try:
            from sklearn.isotonic import IsotonicRegression
        except ImportError:
            logger.warning("calibration_skip", reason="sklearn not installed")
            return None

        X = np.array(probs)
        y = np.array(outcomes, dtype=float)

        iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
        iso.fit(X, y)

        # Store breakpoints for fast interpolation without sklearn at runtime
        x_breaks = iso.X_thresholds_.tolist()
        y_breaks = iso.y_thresholds_.tolist()

        # Before/after Brier on the training rows (in-sample; report only).
        brier_before = float(np.mean((X - y) ** 2))
        calibrated = np.interp(X, x_breaks, y_breaks)
        brier_after = float(np.mean((calibrated - y) ** 2))

        return {
            "x": [round(v, 5) for v in x_breaks],
            "y": [round(v, 5) for v in y_breaks],
            "n_samples": len(probs),
            "brier_before": round(brier_before, 5),
            "brier_after": round(brier_after, 5),
        }

    def retrain(self, model_name: str, probs: list[float], outcomes: list[int]) -> bool:
        """Retrain calibration for a model from historical predictions.

        Args:
            model_name: Calibration key. For the game ensemble this MUST be
                ``"{sector}_ensemble"`` — the exact key
                EnsembleModelAgent._apply_sector_calibration looks up — or the
                fitted curve is stored but never applied.
            probs: List of predicted probabilities
            outcomes: List of actual outcomes (1=correct, 0=incorrect)

        Returns:
            True if calibration was updated, False if insufficient data.
        """
        fit = self._fit(probs, outcomes)
        if fit is None:
            logger.info("calibration_skip", model=model_name, n=len(probs),
                        reason=f"need {MIN_SAMPLES}+ samples")
            return False

        self._calibrations[model_name] = fit
        self._save()

        logger.info(
            "calibration_trained",
            model=model_name,
            n=fit["n_samples"],
            breakpoints=len(fit["x"]),
            brier_before=fit["brier_before"],
            brier_after=fit["brier_after"],
            improvement=round(fit["brier_before"] - fit["brier_after"], 4),
        )
        return True

    def retrain_all_from_db(
        self,
        *,
        min_per_sector: int = MIN_SAMPLES,
        sector: Optional[str] = None,
        dry_run: bool = False,
        db_path: Optional[Path] = None,
    ) -> dict[str, dict]:
        """Refit PER-SECTOR ensemble calibration from resolved LIVE predictions.

        Fits one isotonic curve per sector under the key ``"{sector}_ensemble"`` —
        the SAME key EnsembleModelAgent._apply_sector_calibration reads — so a
        refit here is actually applied to future predictions. (The previous
        implementation trained a single global ``"ensemble"`` key that the
        ensemble never looks up, so it was a silent no-op.)

        Each ``(blended_true_prob, outcome)`` pair is a genuine out-of-sample
        forecast: the blend was computed at scan time from past-only model
        state. Fitting a monotone recalibration on resolved rows and applying it
        FORWARD is standard post-hoc recalibration, not lookahead. For thin or
        early-season sectors, prefer the leakage-guarded point-in-time scripts
        (``scripts/fit_{sector}_calibration.py``), which add a walk-forward split
        and a promotion gate this in-sample refit does not.

        Rows are deduped to the latest scan per market and restricted to
        ``mode='live'`` and non-void markets.

        Returns one report per sector key::

            {"{sector}_ensemble": {"updated": bool, "n": int,
                                   "brier_before": float|None,
                                   "brier_after": float|None,
                                   "reason": str|None}}
        """
        import sqlite3

        if db_path is None:
            db_path = Path(__file__).resolve().parents[3] / "data" / "predictions.db"
        if not db_path.exists():
            logger.warning("calibration_skip", reason="predictions.db not found")
            return {}

        db = sqlite3.connect(str(db_path))
        db.row_factory = sqlite3.Row

        query = """
            SELECT LOWER(p.sector) AS sector, p.blended_true_prob AS prob, o.outcome AS outcome
            FROM ev_outcomes o
            JOIN ev_predictions p ON o.market_id = p.market_id
            INNER JOIN (
                SELECT market_id, MAX(scan_date) AS latest_scan
                FROM ev_predictions WHERE voided = 0 GROUP BY market_id
            ) latest ON p.market_id = latest.market_id
                    AND p.scan_date = latest.latest_scan
            WHERE o.outcome IS NOT NULL
              AND p.blended_true_prob IS NOT NULL
              AND p.mode = 'live'
        """
        params: tuple = ()
        if sector:
            query += " AND LOWER(p.sector) = ?"
            params = (sector.lower(),)
        rows = db.execute(query, params).fetchall()
        db.close()

        by_sector: dict[str, tuple[list[float], list[int]]] = {}
        for r in rows:
            sec = r["sector"] or ""
            if not sec:
                continue
            probs, outs = by_sector.setdefault(sec, ([], []))
            probs.append(r["prob"])
            outs.append(int(r["outcome"]))

        report: dict[str, dict] = {}
        for sec, (probs, outcomes) in sorted(by_sector.items()):
            key = f"{sec}_ensemble"
            if len(probs) < min_per_sector:
                report[key] = {
                    "updated": False, "n": len(probs),
                    "brier_before": None, "brier_after": None,
                    "reason": f"need {min_per_sector}+ resolved live rows",
                }
                continue
            fit = self._fit(probs, outcomes)
            if fit is None:
                report[key] = {
                    "updated": False, "n": len(probs),
                    "brier_before": None, "brier_after": None,
                    "reason": "fit failed (sklearn unavailable?)",
                }
                continue
            if not dry_run:
                self._calibrations[key] = fit
                self._save()
            report[key] = {
                "updated": not dry_run, "n": fit["n_samples"],
                "brier_before": fit["brier_before"], "brier_after": fit["brier_after"],
                "reason": None,
            }
            logger.info(
                "calibration_refit", key=key, n=fit["n_samples"], dry_run=dry_run,
                improvement=round(fit["brier_before"] - fit["brier_after"], 4),
            )
        return report

    def summary(self) -> dict[str, dict]:
        """Return calibration summary for each model."""
        summary = {}
        for name, cal in self._calibrations.items():
            summary[name] = {
                "n_samples": cal.get("n_samples", 0),
                "brier_before": cal.get("brier_before"),
                "brier_after": cal.get("brier_after"),
                "breakpoints": len(cal.get("x", [])),
            }
        return summary
