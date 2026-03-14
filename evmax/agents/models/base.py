"""ModelAgent base class — bridges Agent (bus communication) with ModelBase (probability prediction).

Every model agent:
  1. Implements Agent.run(request) — batch-predict for a list of (market, sharp_odds) pairs
  2. Implements ModelBase.predict(market, sharp_odds) — single prediction for pipeline use
  3. Persists its model state to data/models/{name}_state.json

Batch run() flow:
  request.params["pairs"] = list of {"market": PredictionMarket, "sharp": SharpOdds}
  Returns AgentResponse.data = dict[event_id, ModelAgentPrediction]
  Publishes to "model.{agent_name}.{sector}"
"""

from __future__ import annotations

import json
from abc import abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from evmax.agents.base import Agent, AgentRequest, AgentResponse
from evmax.models.market import PredictionMarket
from evmax.models.odds import SharpOdds
from evmax.models_ml.base import ModelBase, ModelPrediction

# Where model state (Elo ratings, form records, etc.) is persisted
STATE_DIR = Path(__file__).resolve().parents[3] / "data" / "models"


@dataclass
class ModelAgentPrediction:
    """Richer prediction output from a ModelAgent (superset of ModelPrediction)."""

    event_id: str
    model_name: str
    true_prob_a: float   # P(outcome_a wins / covers)
    true_prob_b: float
    true_prob_draw: Optional[float] = None
    confidence: float = 0.5    # 0.0–1.0; lower when data is sparse
    weight: float = 1.0        # blending weight assigned by ensemble
    sample_size: int = 0       # games used to compute this prediction
    notes: str = ""            # human-readable explanation

    def to_model_prediction(self) -> ModelPrediction:
        return ModelPrediction(
            true_prob_a=self.true_prob_a,
            true_prob_b=self.true_prob_b,
            true_prob_draw=self.true_prob_draw,
            confidence=self.confidence,
            model_name=self.model_name,
        )


class ModelAgent(Agent, ModelBase):
    """
    Abstract base for all statistical model agents.

    Subclasses must implement:
      - predict_pair(market, sharp_odds) -> Optional[ModelAgentPrediction]
      - load_state() / save_state()

    State is stored as JSON in data/models/{name}_state.json so ratings
    and form records survive between process restarts.
    """

    # Set in subclass
    name: str
    weight: float = 1.0

    def __init__(self) -> None:
        Agent.__init__(self)
        self._state: dict[str, Any] = {}
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        self._state_path = STATE_DIR / f"{self.name}_state.json"
        self.load_state()

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def load_state(self) -> None:
        """Load model state from JSON file (no-op if file missing)."""
        if self._state_path.exists():
            try:
                self._state = json.loads(self._state_path.read_text())
                self.log.info("state_loaded", model=self.name, keys=len(self._state))
            except Exception as exc:
                self.log.warning("state_load_failed", model=self.name, error=str(exc))
                self._state = {}
        else:
            self._state = {}

    def save_state(self) -> None:
        """Persist current model state to JSON."""
        try:
            self._state_path.write_text(json.dumps(self._state, indent=2, default=str))
        except Exception as exc:
            self.log.warning("state_save_failed", model=self.name, error=str(exc))

    # ------------------------------------------------------------------
    # Agent.run() — batch prediction
    # ------------------------------------------------------------------

    async def run(self, request: AgentRequest) -> AgentResponse:
        sector = request.sector
        pairs: list[dict] = request.params.get("pairs", [])

        predictions: dict[str, ModelAgentPrediction] = {}
        for pair in pairs:
            market: PredictionMarket = pair["market"]
            sharp: SharpOdds = pair["sharp"]
            pred = await self.predict_pair(market, sharp)
            if pred is not None:
                predictions[sharp.event_id] = pred

        self.log.info(
            "model_batch_done",
            model=self.name,
            sector=sector,
            predicted=len(predictions),
            total_pairs=len(pairs),
        )

        await self.publish(
            f"model.{self.name}.{sector}",
            predictions,
            request.correlation_id,
        )

        return AgentResponse(
            agent_name=self.name,
            sector=sector,
            data=predictions,
        )

    # ------------------------------------------------------------------
    # ModelBase.predict() — single prediction (pipeline compatibility)
    # ------------------------------------------------------------------

    async def predict(
        self,
        market: PredictionMarket,
        sharp_odds: SharpOdds,
    ) -> Optional[ModelPrediction]:
        pred = await self.predict_pair(market, sharp_odds)
        if pred is None:
            return None
        return pred.to_model_prediction()

    # ------------------------------------------------------------------
    # Abstract interface for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    async def predict_pair(
        self,
        market: PredictionMarket,
        sharp_odds: SharpOdds,
    ) -> Optional[ModelAgentPrediction]:
        """
        Produce a probability prediction for a single matched pair.

        Return None if the model has insufficient data for this matchup.
        """
        ...

    @abstractmethod
    def update(
        self,
        team_a: str,
        team_b: str,
        score_a: float,
        score_b: float,
        sector: str,
        event_date: Optional[str] = None,
    ) -> None:
        """
        Update model state with a known game result.

        Args:
            team_a: Home/outcome_a team name (normalized).
            team_b: Away/outcome_b team name.
            score_a: Final score / goals / points for team_a.
            score_b: Final score / goals / points for team_b.
            sector: Sport sector.
            event_date: ISO date string, for ordering updates.
        """
        ...
