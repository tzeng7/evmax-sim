"""ModelBase ABC — pluggable probability model interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from evmax.models.market import PredictionMarket
from evmax.models.odds import SharpOdds


@dataclass
class ModelPrediction:
    """Output from a probability model for a two-way market."""

    true_prob_a: float  # Probability of outcome A (YES side)
    true_prob_b: float  # Probability of outcome B (NO side)
    true_prob_draw: Optional[float] = None  # For soccer 3-way markets
    confidence: float = 1.0  # Model confidence 0.0–1.0
    model_name: str = ""


class ModelBase(ABC):
    """
    Abstract base class for probability prediction models.

    Phase 1: SharpBooksModel (devigged sharp lines only)
    Phase 2+: ML models (Elo, regression, neural nets)

    Each model returns probabilities and a weight/confidence.
    The pipeline blends multiple models' outputs.
    """

    name: str
    weight: float  # Relative weight in ensemble blend (0.0–1.0)

    @abstractmethod
    async def predict(
        self,
        market: PredictionMarket,
        sharp_odds: SharpOdds,
    ) -> Optional[ModelPrediction]:
        """
        Generate probability prediction for a market.

        Args:
            market: The prediction market to evaluate.
            sharp_odds: Associated sharp sportsbook odds.

        Returns:
            ModelPrediction or None if model cannot evaluate this market.
        """
        ...


def blend_predictions(
    predictions: list[tuple[ModelPrediction, float]],  # (prediction, weight)
) -> Optional[ModelPrediction]:
    """
    Weighted blend of multiple model predictions.

    Args:
        predictions: List of (ModelPrediction, weight) tuples.

    Returns:
        Blended ModelPrediction, or None if no predictions.
    """
    if not predictions:
        return None

    total_weight = sum(w for _, w in predictions)
    if total_weight <= 0:
        return None

    prob_a = sum(p.true_prob_a * w for p, w in predictions) / total_weight
    prob_b = sum(p.true_prob_b * w for p, w in predictions) / total_weight

    draw_weighted = [(p.true_prob_draw or 0.0) * w for p, w in predictions]
    prob_draw = sum(draw_weighted) / total_weight if any(p.true_prob_draw for p, _ in predictions) else None

    # Normalize
    total_prob = prob_a + prob_b + (prob_draw or 0.0)
    if total_prob > 0:
        prob_a /= total_prob
        prob_b /= total_prob
        if prob_draw is not None:
            prob_draw /= total_prob

    avg_confidence = sum(p.confidence * w for p, w in predictions) / total_weight

    model_names = "+".join(p.model_name for p, _ in predictions if p.model_name)

    return ModelPrediction(
        true_prob_a=prob_a,
        true_prob_b=prob_b,
        true_prob_draw=prob_draw,
        confidence=avg_confidence,
        model_name=model_names,
    )
