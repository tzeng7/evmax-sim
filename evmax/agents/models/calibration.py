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
  Call calibrator.retrain(model_name, probs, outcomes) with historical data.
  Or use `evmax cleanup adjust` which retrains all models.
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

    def retrain(self, model_name: str, probs: list[float], outcomes: list[int]) -> bool:
        """Retrain calibration for a model from historical predictions.

        Args:
            model_name: Model agent name (e.g. "elo", "form", "efficiency")
            probs: List of predicted probabilities
            outcomes: List of actual outcomes (1=correct, 0=incorrect)

        Returns:
            True if calibration was updated, False if insufficient data.
        """
        if len(probs) < MIN_SAMPLES:
            logger.info("calibration_skip", model=model_name, n=len(probs),
                        reason=f"need {MIN_SAMPLES}+ samples")
            return False

        try:
            from sklearn.isotonic import IsotonicRegression
        except ImportError:
            logger.warning("calibration_skip", reason="sklearn not installed")
            return False

        X = np.array(probs)
        y = np.array(outcomes, dtype=float)

        iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
        iso.fit(X, y)

        # Store breakpoints for fast interpolation without sklearn at runtime
        x_breaks = iso.X_thresholds_.tolist()
        y_breaks = iso.y_thresholds_.tolist()

        # Compute before/after Brier score
        brier_before = float(np.mean((X - y) ** 2))
        calibrated = np.interp(X, x_breaks, y_breaks)
        brier_after = float(np.mean((calibrated - y) ** 2))

        self._calibrations[model_name] = {
            "x": [round(v, 5) for v in x_breaks],
            "y": [round(v, 5) for v in y_breaks],
            "n_samples": len(probs),
            "brier_before": round(brier_before, 5),
            "brier_after": round(brier_after, 5),
        }
        self._save()

        logger.info(
            "calibration_trained",
            model=model_name,
            n=len(probs),
            breakpoints=len(x_breaks),
            brier_before=round(brier_before, 4),
            brier_after=round(brier_after, 4),
            improvement=round(brier_before - brier_after, 4),
        )
        return True

    def retrain_all_from_db(self) -> dict[str, bool]:
        """Retrain calibration for all models from predictions.db.

        Requires resolved predictions with per-model probability data stored
        in the ensemble's per_model field. Falls back to blended_true_prob
        as a single "ensemble" calibration if per-model data isn't available.
        """
        import sqlite3

        db_path = Path(__file__).resolve().parents[3] / "data" / "predictions.db"
        if not db_path.exists():
            logger.warning("calibration_skip", reason="predictions.db not found")
            return {}

        db = sqlite3.connect(str(db_path))
        db.row_factory = sqlite3.Row

        rows = db.execute("""
            SELECT p.blended_true_prob, p.sharp_true_prob, o.outcome
            FROM ev_predictions p
            JOIN ev_outcomes o ON p.market_id = o.market_id
            WHERE o.outcome IS NOT NULL
        """).fetchall()

        results = {}

        if len(rows) >= MIN_SAMPLES:
            # Calibrate the ensemble output
            probs = [r["blended_true_prob"] for r in rows]
            outcomes = [r["outcome"] for r in rows]
            results["ensemble"] = self.retrain("ensemble", probs, outcomes)

            # Also calibrate sharp as a baseline comparison
            sharp_probs = [r["sharp_true_prob"] for r in rows]
            results["sharp"] = self.retrain("sharp", sharp_probs, outcomes)

        db.close()
        return results

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
