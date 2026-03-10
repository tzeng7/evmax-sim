"""SharpBooksModel — Phase 1 probability model.

Uses devigged sharp sportsbook probabilities as the sole probability estimate.
No additional modeling applied. Weight=1.0.

This is the Phase 1 implementation. Phase 2+ will blend additional ML models
(Elo ratings, regression, neural nets) alongside this.
"""

from __future__ import annotations

from typing import Optional

import structlog

from evmax.models.market import PredictionMarket
from evmax.models.odds import SharpOdds
from evmax.models_ml.base import ModelBase, ModelPrediction

logger = structlog.get_logger(__name__)


class SharpBooksModel(ModelBase):
    """
    Probability model that directly uses devigged sharp book odds.

    Phase 1: Full weight. The devigged Pinnacle/Betfair probabilities
    ARE the true probabilities.

    Phase 2+: Reduced weight as statistical models are added to the ensemble.
    """

    name = "sharp_books"
    weight = 1.0  # Full weight in Phase 1

    async def predict(
        self,
        market: PredictionMarket,
        sharp_odds: SharpOdds,
    ) -> Optional[ModelPrediction]:
        """Return devigged sharp book probabilities."""
        if sharp_odds.true_prob_a <= 0 or sharp_odds.true_prob_b <= 0:
            logger.debug("sharp_only_skip", reason="no valid probs", event_id=sharp_odds.event_id)
            return None

        # Confidence is inversely related to vig — lower vig = higher confidence
        # Pinnacle typically has 2-4% vig; Betfair even less
        confidence = max(0.5, 1.0 - sharp_odds.margin * 10)

        return ModelPrediction(
            true_prob_a=sharp_odds.true_prob_a,
            true_prob_b=sharp_odds.true_prob_b,
            true_prob_draw=sharp_odds.true_prob_draw,
            confidence=confidence,
            model_name=self.name,
        )
