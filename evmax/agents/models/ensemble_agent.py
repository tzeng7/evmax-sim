"""EnsembleModelAgent — blends multiple model agents into a single probability estimate.

Blending approach:
  - Each model produces a ModelAgentPrediction with a weight and confidence score.
  - Effective weight = model.weight × prediction.confidence
  - Final prob_a = Σ(prob_a_i × eff_weight_i) / Σ(eff_weight_i)
  - When only SharpBooksModel is available (Phase 1), just returns sharp probs.

Configuration (request.params):
  pairs         : list of {"market": PredictionMarket, "sharp": SharpOdds}
  sharp_weight  : float  — weight given to devigged Pinnacle in the blend (default 0.40)

Output (AgentResponse.data):
  {
    event_id: {
      "true_prob_a": float,
      "true_prob_b": float,
      "true_prob_draw": float | None,
      "confidence": float,
      "model_sources": str,   # e.g. "sharp+elo+form"
    }
  }

Published topic: "model.ensemble.{sector}"
Also sets request context for EVGapAgent via "model_probs" and "model_sources" dicts.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

from evmax.agents.base import Agent, AgentBus, AgentRequest, AgentResponse
from evmax.agents.models.base import ModelAgent, ModelAgentPrediction
from evmax.models.market import PredictionMarket
from evmax.models.odds import SharpOdds


@dataclass
class BlendedPrediction:
    event_id: str
    true_prob_a: float
    true_prob_b: float
    true_prob_draw: Optional[float]
    confidence: float
    model_sources: str
    per_model: dict[str, ModelAgentPrediction]   # model_name → prediction


class EnsembleModelAgent(Agent):
    """
    Runs all registered model agents in parallel, then blends their outputs.

    Usage:
        ensemble = EnsembleModelAgent(models=[elo_agent, form_agent, poisson_agent])
        response = await ensemble(request)  # data = dict[event_id, BlendedPrediction]
    """

    name = "ensemble"
    description = (
        "Runs all model agents in parallel, blends predictions by "
        "confidence-weighted average, outputs event_id → BlendedPrediction."
    )

    def __init__(
        self,
        models: list[ModelAgent],
        sharp_weight: float = 0.40,
    ) -> None:
        super().__init__()
        self._models = models
        self._sharp_weight = sharp_weight

    def attach_bus(self, bus: AgentBus) -> None:
        super().attach_bus(bus)
        for model in self._models:
            model.attach_bus(bus)

    async def run(self, request: AgentRequest) -> AgentResponse:
        sector = request.sector
        pairs: list[dict] = request.params.get("pairs", [])
        sharp_weight = request.params.get("sharp_weight", self._sharp_weight)

        if not pairs:
            return AgentResponse(agent_name=self.name, sector=sector, data={})

        # Run all models concurrently
        model_requests = [
            AgentRequest(
                sector=sector,
                params={"pairs": pairs},
                correlation_id=request.correlation_id,
            )
            for _ in self._models
        ]

        model_responses = await asyncio.gather(
            *(model(req) for model, req in zip(self._models, model_requests)),
            return_exceptions=True,
        )

        # Collect predictions per event_id from each model
        per_event: dict[str, dict[str, ModelAgentPrediction]] = {}
        for model, resp in zip(self._models, model_responses):
            if isinstance(resp, Exception):
                self.log.warning("model_failed", model=model.name, error=str(resp))
                continue
            if resp.status != "ok" or not resp.data:
                continue
            for event_id, pred in resp.data.items():
                if event_id not in per_event:
                    per_event[event_id] = {}
                per_event[event_id][model.name] = pred

        # Blend with sharp probs
        sharp_by_id = {pair["sharp"].event_id: pair["sharp"] for pair in pairs}

        blended: dict[str, BlendedPrediction] = {}
        for event_id, model_preds in per_event.items():
            sharp = sharp_by_id.get(event_id)
            blend = self._blend(event_id, model_preds, sharp, sharp_weight)
            if blend is not None:
                blended[event_id] = blend

        # For events with no model predictions, use sharp only
        for pair in pairs:
            eid = pair["sharp"].event_id
            if eid not in blended:
                sharp = pair["sharp"]
                blended[eid] = BlendedPrediction(
                    event_id=eid,
                    true_prob_a=sharp.true_prob_a,
                    true_prob_b=sharp.true_prob_b,
                    true_prob_draw=sharp.true_prob_draw,
                    confidence=max(0.5, 1.0 - sharp.margin * 10),
                    model_sources="sharp",
                    per_model={},
                )

        self.log.info(
            "ensemble_done",
            sector=sector,
            blended=len(blended),
            models_used=[m.name for m in self._models],
        )

        await self.publish(f"model.ensemble.{sector}", blended, request.correlation_id)

        return AgentResponse(
            agent_name=self.name,
            sector=sector,
            data=blended,
        )

    # ------------------------------------------------------------------
    # Blending logic
    # ------------------------------------------------------------------

    def _blend(
        self,
        event_id: str,
        model_preds: dict[str, ModelAgentPrediction],
        sharp: Optional[SharpOdds],
        sharp_weight: float,
    ) -> Optional[BlendedPrediction]:
        """Blend model predictions + sharp book.

        sharp_weight is a TRUE fraction [0,1]: the final probability is
          prob = sharp_weight × pinnacle_prob + (1 - sharp_weight) × model_prob

        Model predictions are confidence-weighted among themselves first,
        then combined with Pinnacle at the sharp_weight ratio.
        """
        # --- Step 1: Confidence-weighted average of model predictions ---
        model_contribs: list[tuple[float, float, float, Optional[float]]] = []
        for pred in model_preds.values():
            if pred.confidence < 0.45:
                continue
            eff_w = pred.weight * pred.confidence
            if eff_w <= 0:
                continue
            model_contribs.append((eff_w, pred.true_prob_a, pred.true_prob_b, pred.true_prob_draw))

        has_models = len(model_contribs) > 0
        has_sharp = sharp is not None

        if not has_models and not has_sharp:
            return None

        if not has_models:
            # No models — just use sharp directly
            prob_a = sharp.true_prob_a  # type: ignore[union-attr]
            prob_b = sharp.true_prob_b  # type: ignore[union-attr]
            prob_draw = sharp.true_prob_draw  # type: ignore[union-attr]
        elif not has_sharp:
            # No sharp — just use model average
            total_w = sum(c[0] for c in model_contribs)
            prob_a = sum(c[0] * c[1] for c in model_contribs) / total_w
            prob_b = sum(c[0] * c[2] for c in model_contribs) / total_w
            prob_draw = (sum(c[0] * (c[3] or 0.0) for c in model_contribs) / total_w
                         if any(c[3] is not None for c in model_contribs) else None)
        else:
            # Blend: true sharp_weight fraction from Pinnacle, rest from models
            total_mw = sum(c[0] for c in model_contribs)
            model_a = sum(c[0] * c[1] for c in model_contribs) / total_mw
            model_b = sum(c[0] * c[2] for c in model_contribs) / total_mw
            has_draw = any(c[3] is not None for c in model_contribs) or sharp.true_prob_draw is not None
            model_draw = (sum(c[0] * (c[3] or 0.0) for c in model_contribs) / total_mw
                          if any(c[3] is not None for c in model_contribs) else 0.0)

            model_weight = 1.0 - sharp_weight
            prob_a = sharp_weight * sharp.true_prob_a + model_weight * model_a
            prob_b = sharp_weight * sharp.true_prob_b + model_weight * model_b
            prob_draw = (sharp_weight * (sharp.true_prob_draw or 0.0) + model_weight * model_draw
                         if has_draw else None)

        # Normalize to sum to 1
        total_p = prob_a + prob_b + (prob_draw or 0.0)
        if total_p > 1e-9:
            prob_a /= total_p
            prob_b /= total_p
            if prob_draw is not None:
                prob_draw /= total_p

        if model_contribs:
            pred_list = list(model_preds.values())
            avg_conf = sum(
                c[0] * (pred_list[i].confidence if i < len(pred_list) else 0.5)
                for i, c in enumerate(model_contribs)
            ) / max(sum(c[0] for c in model_contribs), 1e-9)
        else:
            avg_conf = max(0.5, 1.0 - (sharp.margin * 10 if sharp else 0.5))

        model_sources = "+".join(sorted(model_preds.keys()) + (["sharp"] if sharp else []))

        return BlendedPrediction(
            event_id=event_id,
            true_prob_a=round(prob_a, 5),
            true_prob_b=round(prob_b, 5),
            true_prob_draw=round(prob_draw, 5) if prob_draw is not None else None,
            confidence=round(avg_conf, 3),
            model_sources=model_sources,
            per_model=model_preds,
        )
