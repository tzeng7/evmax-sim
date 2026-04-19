"""MetaModel — trained combiner that replaces fixed-weight ensemble blending.

Instead of hardcoded weights (Elo 0.35, Form 0.25, Poisson 0.30, sharp 0.85),
a logistic regression learns the optimal combination from historical
predictions vs outcomes.

Features per prediction:
  - Each model agent's probability output (logit-transformed)
  - Sharp probability (logit-transformed)
  - Efficiency differential (ORTG gap)
  - Shot quality regression signal
  - Matchup margin

Training:
  Requires ≥50 resolved predictions with per-model outputs stored.
  Until trained, falls back to the fixed-weight EnsembleModelAgent.

State: data/models/meta_model.json
  {
    "coefficients": [w0, w1, ...],
    "intercept": float,
    "feature_names": ["sharp_logit", "elo_logit", ...],
    "brier_score": float,
    "n_samples": int,
    "trained_at": "2026-04-18"
  }

Usage:
  meta = MetaModel()
  if meta.is_trained:
      prob = meta.predict(feature_dict)
  else:
      # fall back to EnsembleModelAgent
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

STATE_PATH = Path(__file__).resolve().parents[3] / "data" / "models" / "meta_model.json"
MIN_TRAINING_SAMPLES = 50


def _logit(p: float) -> float:
    """Log-odds transform: logit(p) = log(p / (1-p))."""
    p = max(0.001, min(0.999, p))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    """Inverse logit: sigmoid(x) = 1 / (1 + exp(-x))."""
    if x > 500:
        return 1.0
    if x < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


class MetaModel:
    """Logistic regression meta-model for combining prediction signals."""

    FEATURE_NAMES = [
        "sharp_logit",
        "elo_logit",
        "form_logit",
        "poisson_logit",
        "efficiency_logit",
        "shot_quality_logit",
        "matchup_logit",
        "sharp_prob",  # raw (for interaction effects in logit space)
    ]

    def __init__(self) -> None:
        self._coefficients: Optional[np.ndarray] = None
        self._intercept: float = 0.0
        self._trained: bool = False
        self._feature_names: Optional[list[str]] = None
        self._load()

    @property
    def is_trained(self) -> bool:
        return self._trained

    def _load(self) -> None:
        if not STATE_PATH.exists():
            return
        try:
            data = json.loads(STATE_PATH.read_text())
            self._coefficients = np.array(data["coefficients"])
            self._intercept = data["intercept"]
            self._feature_names = data.get("feature_names")
            self._trained = True
            logger.info(
                "meta_model_loaded",
                features=data.get("feature_names"),
                brier=data.get("brier_score"),
                n=data.get("n_samples"),
            )
        except Exception as e:
            logger.warning("meta_model_load_failed", error=str(e))

    def _save(self, feature_names: list[str], brier: float, n: int) -> None:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps({
            "coefficients": self._coefficients.tolist(),
            "intercept": self._intercept,
            "feature_names": feature_names,
            "brier_score": round(brier, 5),
            "n_samples": n,
            "trained_at": date.today().isoformat(),
        }, indent=2))

    def predict(self, features: dict[str, float]) -> float:
        """Predict win probability from a feature dict.

        Missing features are filled with 0 (neutral in logit space ≈ 50%).
        Uses the feature names stored during training, not the class default.
        """
        if not self._trained or self._coefficients is None:
            raise RuntimeError("MetaModel not trained — use EnsembleModelAgent instead")

        trained_features = self._feature_names or self.FEATURE_NAMES
        x = np.array([features.get(f, 0.0) for f in trained_features])
        logit = float(np.dot(self._coefficients, x)) + self._intercept
        return _sigmoid(logit)

    def build_features(
        self,
        sharp_prob: float,
        model_probs: dict[str, float],
    ) -> dict[str, float]:
        """Build feature dict from model probabilities.

        Args:
            sharp_prob: Devigged Pinnacle probability
            model_probs: {model_name: probability} from individual agents
        """
        features = {
            "sharp_logit": _logit(sharp_prob),
            "sharp_prob": sharp_prob,
        }

        for model_name in ["elo", "form", "poisson", "efficiency", "shot_quality", "matchup"]:
            prob = model_probs.get(model_name)
            if prob is not None:
                features[f"{model_name}_logit"] = _logit(prob)
            else:
                features[f"{model_name}_logit"] = 0.0

        return features

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> float:
        """Train logistic regression from feature matrix and outcomes.

        Returns Brier score of the trained model.
        """
        from sklearn.linear_model import LogisticRegression

        if len(y) < MIN_TRAINING_SAMPLES:
            raise ValueError(f"Need {MIN_TRAINING_SAMPLES}+ samples, got {len(y)}")

        model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
        )
        model.fit(X, y)

        self._coefficients = model.coef_[0]
        self._intercept = float(model.intercept_[0])
        self._trained = True

        # Compute Brier score
        probs = model.predict_proba(X)[:, 1]
        brier = float(np.mean((probs - y) ** 2))

        self._save(feature_names, brier, len(y))

        logger.info(
            "meta_model_trained",
            n=len(y),
            brier=round(brier, 4),
            coefficients={f: round(c, 4) for f, c in zip(feature_names, self._coefficients)},
        )
        return brier

    def train_from_db(self) -> Optional[float]:
        """Train from predictions.db resolved outcomes.

        Since we don't store per-model probabilities in the DB yet,
        this trains on sharp_prob + blended_prob as features.
        A future version will log per-model outputs for richer training.
        """
        import sqlite3

        db_path = Path(__file__).resolve().parents[3] / "data" / "predictions.db"
        if not db_path.exists():
            return None

        db = sqlite3.connect(str(db_path))
        db.row_factory = sqlite3.Row

        rows = db.execute("""
            SELECT p.blended_true_prob, p.sharp_true_prob, p.kalshi_yes_price,
                   p.ev_pct, o.outcome, p.sector
            FROM ev_predictions p
            JOIN ev_outcomes o ON p.market_id = o.market_id
            WHERE o.outcome IS NOT NULL
        """).fetchall()
        db.close()

        if len(rows) < MIN_TRAINING_SAMPLES:
            logger.info("meta_model_skip", n=len(rows), reason="insufficient data")
            return None

        # Build feature matrix from available data
        feature_names = ["sharp_logit", "blended_logit", "kalshi_logit", "ev_pct"]
        X = []
        y = []

        for r in rows:
            features = [
                _logit(r["sharp_true_prob"]),
                _logit(r["blended_true_prob"]),
                _logit(r["kalshi_yes_price"]),
                r["ev_pct"],
            ]
            X.append(features)
            y.append(r["outcome"])

        X = np.array(X)
        y = np.array(y)

        # Temporarily override FEATURE_NAMES for this simpler model
        saved_names = self.FEATURE_NAMES
        self.FEATURE_NAMES = feature_names

        brier = self.train(X, y, feature_names)

        self.FEATURE_NAMES = saved_names
        return brier
