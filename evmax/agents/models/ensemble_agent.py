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
from evmax.agents.models.calibration import ModelCalibrator
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

    # Per-sector weight overrides for individual models.  When a sector
    # is listed here the model's class-level `weight` attribute is replaced
    # during blending.  This lets us lean on Poisson (goal-matrix derived
    # draws) for soccer while down-weighting Elo/Form whose fixed draw
    # allocations were historically overconfident.
    SECTOR_WEIGHT_OVERRIDES: dict[str, dict[str, float]] = {
        "soccer": {"elo": 0.15, "form": 0.10, "poisson": 0.40, "xg": 0.25},
        "nba": {
            "efficiency": 0.30, "possession_sim": 0.30,
            "elo": 0.10, "form": 0.10,
            "shot_quality": 0.10, "matchup": 0.10,
            "poisson": 0.0,
        },
        # WNBA uses its own dedicated efficiency + possession-sim agents
        # (wnba_efficiency, wnba_possession_sim) driven by ESPN box scores.
        # See evmax/agents/models/wnba_efficiency_agent.py and
        # evmax/agents/models/wnba_possession_sim_agent.py.
        # Poisson zeroed for the same reason as NBA — basketball scoring is
        # not a Poisson process and the bucketed matrix was distorting margin
        # variance. Shot_quality / matchup still not ported (future work).
        "wnba": {
            "wnba_efficiency":     0.25,
            "wnba_possession_sim": 0.25,
            "elo":                 0.30,
            "form":                0.15,
            "poisson":             0.0,
        },
        # Tennis: Surface Elo + recency-weighted serve/return are primary signals.
        # Form agent replaces ranking trend as the recency/momentum voice.
        # H2H stays small. Advanced stats are a supporting signal.
        # Ranking trend is vestigial (kept at 0.05 as tiebreaker).
        "tennis": {
            "tennis_surface": 0.30,
            "tennis_serve_return": 0.25,
            "tennis_form": 0.20,
            "tennis_advanced": 0.15,
            "tennis_h2h": 0.05,
            "tennis_ranking_trend": 0.05,
        },
        # Baseball: starter FIP/ERA matchup (Pythag) is the dominant non-sharp
        # signal — one player drives ~50% of game outcome. Elo + form are
        # supporting talent/recency voices. Poisson is intentionally absent
        # from baseball's YAML model list (overdispersed runs distribution +
        # bullpen-leverage shifts violate the independence assumption); the
        # ensemble never instantiates it for baseball, so no override needed.
        "baseball": {
            "pitcher": 0.50,
            "elo":     0.25,
            "form":    0.25,
        },
        # NFL ensemble — per-model accuracy is clustered 63-65% so the blend
        # benefits from spreading weight rather than concentrating on one
        # signal. nfl_qb_elo overlaps with generic elo (both are Elo-shaped
        # team strength), so generic elo is down-weighted when nfl_qb_elo
        # is present. Phase 2 initial weights (subject to backtest sweep):
        #   nfl_efficiency 0.25 · nfl_qb_elo 0.25 · elo 0.20 · form 0.30
        # Poisson zeroed because NFL scoring is drive-based, not minute-
        # uniform — see scripts/backtest_nfl_efficiency.py for the EPA sweep.
        "nfl": {
            "nfl_efficiency": 0.25,
            "nfl_qb_elo":     0.25,
            "elo":            0.20,
            "form":           0.30,
            "poisson":        0.0,
        },
    }

    def __init__(
        self,
        models: list[ModelAgent],
        sharp_weight: float = 0.40,
    ) -> None:
        super().__init__()
        self._models = models
        self._sharp_weight = sharp_weight
        # Per-sector isotonic calibration applied to the model-side blend
        # before sharp blending. Calibrations are stored under the key
        # "{sector}_ensemble" — fit by scripts/fit_tennis_calibration.py
        # (and analogues for other sports) on a held-out backtest year.
        # Missing entries → identity (no-op), so the ensemble still works
        # for sports without a fitted calibration.
        self._calibrator = ModelCalibrator()

    def attach_bus(self, bus: AgentBus) -> None:
        super().attach_bus(bus)
        for model in self._models:
            model.attach_bus(bus)

    async def run(self, request: AgentRequest) -> AgentResponse:
        sector = request.sector
        pairs: list[dict] = request.params.get("pairs", [])
        sharp_weight = request.params.get("sharp_weight", self._sharp_weight)
        # Per-event override: callers can pass a dict of event_id → sharp_weight
        # to differentiate within a sector. Used for the soccer league tier
        # system (top-5 European leagues get ~0.85, secondary leagues keep
        # the default). Missing events fall through to `sharp_weight`.
        sharp_weight_by_event: dict[str, float] = request.params.get("sharp_weight_by_event") or {}

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
            event_sharp_weight = sharp_weight_by_event.get(event_id, sharp_weight)
            blend = self._blend(event_id, model_preds, sharp, event_sharp_weight, sector)
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

    def _apply_sector_calibration(
        self,
        sector: str,
        prob_a: float,
        prob_b: float,
        prob_draw: Optional[float],
    ) -> tuple[float, float, Optional[float]]:
        """Apply isotonic calibration for sector_ensemble; identity when missing.

        2-way markets: calibrate prob_a, recover prob_b = 1 - prob_a.

        3-way markets (soccer): the bias signal lives on the team-win legs
        (live data: 81 team-win bets at −6.9pp gap; 8 draw bets at +0.4pp),
        so the calibration curve is fit on team-win pairs only. We apply it
        symmetrically to home and away legs, leave draw untouched, and
        renormalize so the three legs sum to 1. This composes correctly:
        the curve is monotonic, so home > away ordering is preserved.
        """
        key = f"{sector.lower()}_ensemble"
        if prob_draw is None:
            calibrated_a = self._calibrator.calibrate(key, prob_a)
            if calibrated_a == prob_a:
                return prob_a, prob_b, prob_draw
            return calibrated_a, 1.0 - calibrated_a, prob_draw

        cal_a = self._calibrator.calibrate(key, prob_a)
        cal_b = self._calibrator.calibrate(key, prob_b)
        if cal_a == prob_a and cal_b == prob_b:
            return prob_a, prob_b, prob_draw
        total = cal_a + cal_b + prob_draw
        if total <= 1e-9:
            return prob_a, prob_b, prob_draw
        return cal_a / total, cal_b / total, prob_draw / total

    # Per-sector ramp params for _disagreement_sharp_weight. Sectors not
    # listed fall through to the global defaults (0.10 / 0.30 / 0.95).
    #
    # Soccer override (2026-04-30): the walk-forward disagreement profile
    # (n=4406, 2324–2526) shows that across all five disagreement buckets
    # sharp is 2-3× closer to the realized outcome than the stat blend:
    #   model > sharp by 7+pts  (n=921):  model 40.7%, sharp 29.5%, actual 24.2%
    #   model > sharp by 4-7pts (n=585):  model 39.8%, sharp 34.3%, actual 29.4%
    #   model < sharp by 4-7pts (n=442):  model 50.2%, sharp 55.6%, actual 53.2%
    #   model < sharp by 7+pts  (n=753):  model 51.4%, sharp 63.4%, actual 64.0%
    # The default ramp (threshold 0.10, sat 0.30, cap 0.95) only kicks in
    # at 10pt disagreement and tops out at 95% sharp — so on the 4-7pt
    # subset where sharp is provably right, the blend keeps the model's
    # bad view at full weight. The override starts the ramp at 4pts,
    # saturates at 10pts (V4: more aggressive than the prior V3 sat=15),
    # and goes to 100% sharp at saturation. Validated via the per-bucket
    # diagnostic (scripts/per_bucket_disagreement_perf.py) with xG wired
    # into the walk-forward: 2526 holdout binary Brier 0.19523 → 0.19506
    # (Δ −0.00017 vs sharp-only ceiling of 0.19499). V4 narrows the gap
    # to sharp-only further than V3 (0.04/0.15/0.99) did.
    # Tennis override (2026-05-01): production resolved bets (n=113 after
    # excluding $0.01 dead-market rows) show the same pattern as soccer but
    # at a tighter threshold. Bucketed by signed gap (blended − sharp) on
    # the YES leg:
    #   model > sharp by 2-5pp (n=10): hit 0.0%, sharp_pct 18.5%, model_pct 21.2%
    #     → Brier(blended)=0.0509, Brier(sharp)=0.0400 (sharp +0.011)
    #     → ROI proxy −19.2% (this single bucket accounts for ~all tennis bleed)
    #   within 2pp (n=102):           hit 35.3%, Brier delta ≈ 0 (sharp +0.0004)
    # Production data is selection-biased (only EV≥2% YES rows logged) so we
    # have no rows in `model < sharp` direction; saturate_at and cap mirror
    # soccer V4 because we lack the data to tune them. Validate via a
    # tennis_walkforward.py rebuild before tightening further.
    DISAGREEMENT_OVERRIDES: dict[str, tuple[float, float, float]] = {
        # sector → (threshold, saturate_at, cap)
        "soccer": (0.04, 0.10, 1.00),
        "tennis": (0.02, 0.10, 1.00),
    }

    @staticmethod
    def _disagreement_sharp_weight(
        model_a: float,
        model_b: float,
        model_draw: Optional[float],
        sharp_a: float,
        sharp_b: float,
        sharp_draw: Optional[float],
        base_sharp_weight: float,
        threshold: float = 0.10,
        saturate_at: float = 0.30,
        cap: float = 0.95,
        sector: Optional[str] = None,
    ) -> float:
        """Return a boosted sharp_weight when the stat blend disagrees with sharp.

        Rationale: soccer walk-forward (2026-04-23, n=2606) showed that when the
        stat ensemble's argmax differed from sharp's, sharp was right 41.6% of
        the time vs stat's 30.6% — an 11pp gap on 13% of the corpus. Rather
        than try to fix the stat models on those specific games, this helper
        defers to sharp when the two sides visibly disagree.

        Mechanism: compute the max per-leg |model - sharp| gap. Below
        `threshold` the base weight is unchanged (normal blend). Above, we
        linearly ramp toward `cap` so that at `saturate_at` disagreement the
        blend is almost pure sharp. FLB correction still runs afterward and
        composes with this.

        Defaults: threshold=0.10, saturate_at=0.30, cap=0.95.
          - diff=0.10 → no adjustment
          - diff=0.20 → halfway to cap
          - diff=0.30+ → saturated at cap

        When `sector` is provided and listed in DISAGREEMENT_OVERRIDES, the
        override values replace the defaults. Explicit threshold/saturate_at/
        cap kwargs always win over the override (used by the A/B harness).
        """
        if sector is not None:
            override = EnsembleModelAgent.DISAGREEMENT_OVERRIDES.get(sector.lower())
            if override is not None:
                # Only apply override values where the caller didn't pass an
                # explicit non-default — explicit kwargs take precedence so
                # the A/B script can sweep params without our overrides
                # silently overriding the sweep.
                if threshold == 0.10:
                    threshold = override[0]
                if saturate_at == 0.30:
                    saturate_at = override[1]
                if cap == 0.95:
                    cap = override[2]
        diffs = [abs(model_a - sharp_a), abs(model_b - sharp_b)]
        if model_draw is not None and sharp_draw is not None:
            diffs.append(abs(model_draw - sharp_draw))
        max_diff = max(diffs)
        if max_diff <= threshold:
            return base_sharp_weight
        span = max(saturate_at - threshold, 1e-9)
        fraction = min(1.0, (max_diff - threshold) / span)
        return base_sharp_weight + fraction * (cap - base_sharp_weight)

    @staticmethod
    def _flb_correct(
        prob_a: float,
        prob_b: float,
        prob_draw: float | None,
        sharp: SharpOdds,
        sharp_weight: float,
    ) -> tuple[float, float, float | None]:
        """Favorite-longshot bias correction.

        When sharp probability is far from 50%, models trained on regular-season
        data systematically compress toward 50/50 — inflating underdogs and
        deflating favorites. This re-blends toward sharp at the extremes.

        The effective sharp weight increases quadratically as sharp prob moves
        away from 50%:
            extra = FLB_STRENGTH × (sharp_prob - 0.5)²
            effective_sharp = sharp_weight + extra × (1 - sharp_weight)

        At sharp=90%, extra ≈ 0.6 × 0.16 = 0.096, so effective sharp goes
        from 0.85 to ~0.99 — almost purely sharp at the extremes. At sharp=60%,
        extra ≈ 0.006 — negligible. This only affects extreme probabilities.
        """
        FLB_STRENGTH = 1.5

        for sharp_p, blended_p, label in [
            (sharp.true_prob_a, prob_a, "a"),
            (sharp.true_prob_b, prob_b, "b"),
        ]:
            deviation = (sharp_p - 0.5) ** 2
            extra = FLB_STRENGTH * deviation
            effective_sharp = sharp_weight + extra * (1.0 - sharp_weight)
            effective_sharp = min(effective_sharp, 0.99)

            corrected = effective_sharp * sharp_p + (1.0 - effective_sharp) * blended_p
            if label == "a":
                prob_a = corrected
            else:
                prob_b = corrected

        if prob_draw is not None and sharp.true_prob_draw is not None:
            sharp_draw = sharp.true_prob_draw
            deviation = (sharp_draw - 0.5) ** 2
            extra = FLB_STRENGTH * deviation
            effective_sharp = sharp_weight + extra * (1.0 - sharp_weight)
            effective_sharp = min(effective_sharp, 0.99)
            prob_draw = effective_sharp * sharp_draw + (1.0 - effective_sharp) * prob_draw

        return prob_a, prob_b, prob_draw

    def _blend(
        self,
        event_id: str,
        model_preds: dict[str, ModelAgentPrediction],
        sharp: Optional[SharpOdds],
        sharp_weight: float,
        sector: str = "",
    ) -> Optional[BlendedPrediction]:
        """Blend model predictions + sharp book.

        sharp_weight is a TRUE fraction [0,1]: the final probability is
          prob = sharp_weight × pinnacle_prob + (1 - sharp_weight) × model_prob

        Model predictions are confidence-weighted among themselves first,
        then combined with Pinnacle at the sharp_weight ratio.
        """
        # --- Step 1: Confidence-weighted average of model predictions ---
        weight_overrides = self.SECTOR_WEIGHT_OVERRIDES.get(sector.lower(), {})
        model_contribs: list[tuple[float, float, float, Optional[float]]] = []
        for pred in model_preds.values():
            if pred.confidence <= 0.45:
                continue
            w = weight_overrides.get(pred.model_name, pred.weight)
            eff_w = w * pred.confidence
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
            # Apply per-sector isotonic calibration to the model output
            # (no-op when no calibration is fitted for this sector).
            prob_a, prob_b, prob_draw = self._apply_sector_calibration(
                sector, prob_a, prob_b, prob_draw,
            )
        else:
            # Blend: true sharp_weight fraction from Pinnacle, rest from models
            total_mw = sum(c[0] for c in model_contribs)
            model_a = sum(c[0] * c[1] for c in model_contribs) / total_mw
            model_b = sum(c[0] * c[2] for c in model_contribs) / total_mw
            has_draw = any(c[3] is not None for c in model_contribs) or sharp.true_prob_draw is not None
            model_draw_val = (sum(c[0] * (c[3] or 0.0) for c in model_contribs) / total_mw
                              if any(c[3] is not None for c in model_contribs) else None)
            # Calibrate the model-side blend BEFORE combining with sharp.
            # Sharp lines are already well-calibrated; running them through
            # a model-side correction would double-correct and add noise.
            model_a, model_b, model_draw_val = self._apply_sector_calibration(
                sector, model_a, model_b, model_draw_val,
            )

            # Defer to sharp when the stat blend visibly disagrees with it —
            # walk-forward showed sharp wins 41.6% vs stat 30.6% on argmax
            # disagreements, so the blend should lean on sharp when the two
            # sides pull apart.
            effective_sharp_weight = self._disagreement_sharp_weight(
                model_a, model_b, model_draw_val,
                sharp.true_prob_a, sharp.true_prob_b, sharp.true_prob_draw,
                base_sharp_weight=sharp_weight,
                sector=sector,
            )
            model_weight = 1.0 - effective_sharp_weight
            prob_a = effective_sharp_weight * sharp.true_prob_a + model_weight * model_a
            prob_b = effective_sharp_weight * sharp.true_prob_b + model_weight * model_b
            prob_draw = (effective_sharp_weight * (sharp.true_prob_draw or 0.0)
                         + model_weight * (model_draw_val or 0.0)
                         if has_draw else None)

        # Favorite-longshot bias correction: at extreme probabilities,
        # models systematically compress toward 50/50 vs sharp. Scale model
        # weight down when sharp prob is far from 50% so the blend trusts
        # sharp more at the tails.
        # Skip for NBA: our NBA-specific models (efficiency, possession_sim,
        # shot_quality, matchup) use actual ORTG/DRTG ratings and produce
        # well-calibrated extreme probabilities — FLB suppresses legitimate
        # model signal on favorites and biases edge-finding toward underdogs.
        if has_models and has_sharp and sector.lower() != "nba":
            prob_a, prob_b, prob_draw = self._flb_correct(
                prob_a, prob_b, prob_draw, sharp, sharp_weight,
            )

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

        # Only include models that actually contributed (passed confidence threshold)
        contributing_models = [
            name for name, pred in model_preds.items()
            if pred.confidence > 0.45 and pred.weight * pred.confidence > 0
        ]
        model_sources = "+".join(sorted(contributing_models) + (["sharp"] if sharp else []))

        return BlendedPrediction(
            event_id=event_id,
            true_prob_a=round(prob_a, 5),
            true_prob_b=round(prob_b, 5),
            true_prob_draw=round(prob_draw, 5) if prob_draw is not None else None,
            confidence=round(avg_conf, 3),
            model_sources=model_sources,
            per_model=model_preds,
        )
