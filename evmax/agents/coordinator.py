"""AgentCoordinator — orchestrates the full agent pipeline for one sector.

Execution order per cycle:
  1.  KalshiOddsAgent    — fetch live Kalshi markets
  2.  SharpOddsAgent     — fetch devigged Pinnacle odds
  3.  InjuryReportAgent  — fetch ESPN injury data
      (steps 1-3 run concurrently)
  4.  MatchingEngine     — match Kalshi markets → sharp events
  5.  EnsembleModelAgent — run Elo + Form + Poisson in parallel, blend predictions
  6.  Injury adjustment  — shift blended probs based on injury impact
  7.  EVGapAgent         — compute EV gaps with injury-adjusted blended probs
  8.  Publish summary    — "coordinator.cycle.done" with all gaps

All agents share one AgentBus so any subscriber (dashboard, Slack notifier,
simulation engine) can react to results without modifying this class.

Usage:
    coordinator = AgentCoordinator(
        sectors=["nba", "soccer"],
        bankroll=250.0,
        kelly_fraction=0.5,   # half Kelly
    )
    cycle = await coordinator.run_cycle()
    for gap in cycle.ev_gaps:
        print(gap.summary())
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import structlog

from evmax.agents.base import Agent, AgentBus, AgentMessage, AgentRequest, AgentResponse
from evmax.agents.odds.kalshi_agent import KalshiOddsAgent
from evmax.agents.odds.sharp_agent import SharpOddsAgent
from evmax.agents.odds.ev_gap_agent import EVGapAgent, EVGap
from evmax.agents.models.elo_agent import EloModelAgent
from evmax.agents.models.form_agent import FormModelAgent
from evmax.agents.models.poisson_agent import PoissonModelAgent
from evmax.agents.models.ensemble_agent import EnsembleModelAgent, BlendedPrediction
from evmax.agents.intelligence.injury_agent import InjuryReportAgent, InjuryReport
from evmax.models.market import PredictionMarket
from evmax.models.odds import SharpOdds
from evmax.matching.engine import MatchingEngine

logger = structlog.get_logger(__name__)


@dataclass
class CycleResult:
    """Results from a single coordinator cycle."""

    sectors_scanned: list[str] = field(default_factory=list)
    markets_fetched: int = 0
    markets_matched: int = 0
    ev_gaps: list[EVGap] = field(default_factory=list)
    blended_predictions: dict[str, BlendedPrediction] = field(default_factory=dict)
    injury_reports: dict[str, dict[str, InjuryReport]] = field(default_factory=dict)  # sector → team → report
    cycle_duration_s: float = 0.0
    errors: list[str] = field(default_factory=list)
    bankroll: float = 250.0
    kelly_fraction: float = 0.5

    @property
    def top_gaps(self) -> list[EVGap]:
        return sorted(self.ev_gaps, key=lambda g: g.ev_pct, reverse=True)

    def stake_for(self, gap: EVGap) -> float:
        """Dollar stake for a gap given the bankroll and Kelly fraction."""
        return round(self.bankroll * gap.kelly_fraction, 2)

    def print_plays(self, min_ev: float = 0.02, max_plays: int = 30) -> None:
        """Pretty-print +EV plays."""
        gaps = [g for g in self.top_gaps if g.ev_pct >= min_ev][:max_plays]
        if not gaps:
            print("No +EV plays found.")
            return
        print(f"\n{'='*80}")
        print(f"  +EV PLAYS  |  Bankroll: ${self.bankroll:.0f}  |  Kelly: {self.kelly_fraction:.0%}")
        print(f"{'='*80}")
        for i, g in enumerate(gaps, 1):
            stake = self.stake_for(g)
            print(
                f"{i:>2}. [{g.edge_label:<8}] {g.sector.upper():<6} "
                f"{g.display_label:<26} "
                f"Kalshi={g.kalshi_yes_price:.2f}  TrueP={g.blended_true_prob:.3f}  "
                f"EV={g.ev_pct*100:+.1f}%  Kelly={g.kelly_fraction*100:.2f}%  "
                f"Stake=${stake:.2f}  [{g.model_sources}]"
            )
        print(f"{'='*80}")
        total_stake = sum(self.stake_for(g) for g in gaps)
        print(f"  Total at risk: ${total_stake:.2f} / ${self.bankroll:.0f} ({total_stake/self.bankroll*100:.1f}%)")
        print()


class AgentCoordinator:
    """
    Top-level orchestrator for the evmax agent pipeline.

    Args:
        sectors:        List of sector keys to scan (default: all).
        enable_models:  If False, skip model agents and use sharp probs only.
        sharp_weight:   Weight given to Pinnacle in ensemble blend (0–1).
        enable_injuries: If True, fetch ESPN injury reports and adjust probs.
        bankroll:       Current bankroll in USD (default $250).
        kelly_fraction: Kelly multiplier (0.5 = half Kelly, 0.25 = quarter Kelly).
    """

    def __init__(
        self,
        sectors: Optional[list[str]] = None,
        enable_models: bool = True,
        sharp_weight: float = 0.40,
        enable_injuries: bool = True,
        bankroll: float = 250.0,
        kelly_fraction: float = 0.5,
    ) -> None:
        from evmax.sectors.registry import ALL_SECTORS
        self._sectors = [s.lower() for s in (sectors or ALL_SECTORS)]
        self._enable_models = enable_models
        self._sharp_weight = sharp_weight
        self._enable_injuries = enable_injuries
        self._bankroll = bankroll
        self._kelly_fraction = kelly_fraction

        self.bus = AgentBus()

        # Odds checker agents
        self.kalshi_agent = KalshiOddsAgent()
        self.sharp_agent = SharpOddsAgent()
        self.ev_gap_agent = EVGapAgent()

        # Intelligence agents
        self.injury_agent = InjuryReportAgent()

        # Statistical model agents
        self.elo_agent = EloModelAgent()
        self.form_agent = FormModelAgent()
        self.poisson_agent = PoissonModelAgent()
        self.ensemble_agent = EnsembleModelAgent(
            models=[self.elo_agent, self.form_agent, self.poisson_agent],
            sharp_weight=sharp_weight,
        )

        self._matching = MatchingEngine()

        for agent in self._all_agents():
            agent.attach_bus(self.bus)

        self.log = structlog.get_logger(__name__)

    def _all_agents(self) -> list[Agent]:
        return [
            self.kalshi_agent, self.sharp_agent, self.ev_gap_agent,
            self.injury_agent,
            self.elo_agent, self.form_agent, self.poisson_agent, self.ensemble_agent,
        ]

    # ------------------------------------------------------------------
    # Main cycle
    # ------------------------------------------------------------------

    async def run_cycle(self) -> CycleResult:
        t0 = time.perf_counter()
        correlation_id = str(uuid.uuid4())[:8]
        result = CycleResult(bankroll=self._bankroll, kelly_fraction=self._kelly_fraction)

        self.log.info("cycle_start", correlation_id=correlation_id, sectors=self._sectors)

        sector_results = await asyncio.gather(
            *(self._run_sector(sector, correlation_id) for sector in self._sectors),
            return_exceptions=True,
        )

        for sector, sr in zip(self._sectors, sector_results):
            if isinstance(sr, Exception):
                self.log.error("sector_failed", sector=sector, error=str(sr))
                result.errors.append(f"{sector}: {sr}")
                continue
            result.sectors_scanned.append(sector)
            result.markets_fetched += sr.get("markets_fetched", 0)
            result.markets_matched += sr.get("markets_matched", 0)
            result.ev_gaps.extend(sr.get("ev_gaps", []))
            result.blended_predictions.update(sr.get("blended_predictions", {}))
            if sr.get("injuries"):
                result.injury_reports[sector] = sr["injuries"]

        result.cycle_duration_s = time.perf_counter() - t0

        self.log.info(
            "cycle_done",
            correlation_id=correlation_id,
            sectors=len(result.sectors_scanned),
            markets=result.markets_fetched,
            matched=result.markets_matched,
            ev_gaps=len(result.ev_gaps),
            duration_s=round(result.cycle_duration_s, 2),
        )

        await self.bus.publish(AgentMessage(
            topic="coordinator.cycle.done",
            payload=result,
            sender="coordinator",
            correlation_id=correlation_id,
        ))

        return result

    # ------------------------------------------------------------------
    # Single-sector cycle
    # ------------------------------------------------------------------

    async def _run_sector(self, sector: str, correlation_id: str) -> dict:
        req = AgentRequest(sector=sector, correlation_id=correlation_id)

        # Steps 1-3: Fetch Kalshi + sharp + injuries concurrently
        fetch_tasks = [self.kalshi_agent(req), self.sharp_agent(req)]
        if self._enable_injuries:
            fetch_tasks.append(self.injury_agent(req))

        fetch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        kalshi_resp = fetch_results[0]
        sharp_resp = fetch_results[1]
        injury_resp = fetch_results[2] if self._enable_injuries else None

        markets: list[PredictionMarket] = (
            kalshi_resp.data if not isinstance(kalshi_resp, Exception) else []
        ) or []
        sharp_odds: list[SharpOdds] = (
            sharp_resp.data if not isinstance(sharp_resp, Exception) else []
        ) or []
        injuries: dict[str, InjuryReport] = (
            injury_resp.data if injury_resp and not isinstance(injury_resp, Exception) else {}
        ) or {}

        if isinstance(kalshi_resp, Exception):
            self.log.error("kalshi_failed", sector=sector, error=str(kalshi_resp))
        if isinstance(sharp_resp, Exception):
            self.log.error("sharp_failed", sector=sector, error=str(sharp_resp))

        if not markets or not sharp_odds:
            return {
                "markets_fetched": len(markets),
                "markets_matched": 0,
                "ev_gaps": [],
                "blended_predictions": {},
                "injuries": injuries,
            }

        # Step 4: Match markets → sharp events
        matched_pairs = self._matching.match_all(markets, sharp_odds)
        pairs = [{"market": m, "sharp": s} for m, s, _ in matched_pairs]

        # Step 5: Ensemble model predictions
        blended: dict[str, BlendedPrediction] = {}
        model_probs: dict[str, float] = {}
        model_sources: dict[str, str] = {}

        blended_preds: dict = {}  # event_id → BlendedPrediction
        if self._enable_models and pairs:
            ensemble_req = AgentRequest(
                sector=sector,
                params={"pairs": pairs, "sharp_weight": self._sharp_weight},
                correlation_id=correlation_id,
            )
            ensemble_resp = await self.ensemble_agent(ensemble_req)
            blended = ensemble_resp.data or {}
            blended_preds = blended

            for event_id, blend in blended.items():
                # Store full objects; EVGapAgent will pick correct side per YES alignment
                model_sources[event_id] = blend.model_sources

        # Step 7: EV gap analysis — pass full blended predictions for correct side alignment
        ev_req = AgentRequest(
            sector=sector,
            params={
                "kalshi_markets": markets,
                "sharp_odds": sharp_odds,
                "blended_preds": blended_preds,
                "injuries": injuries,
                "model_sources": model_sources,
                "kelly_base_fraction": self._kelly_fraction,
            },
            correlation_id=correlation_id,
        )
        ev_resp = await self.ev_gap_agent(ev_req)
        ev_gaps: list[EVGap] = ev_resp.data or []

        return {
            "markets_fetched": len(markets),
            "markets_matched": len(matched_pairs),
            "ev_gaps": ev_gaps,
            "blended_predictions": blended,
            "injuries": injuries,
        }

    # ------------------------------------------------------------------
    # Model training helpers
    # ------------------------------------------------------------------

    def update_models(
        self,
        team_a: str,
        team_b: str,
        score_a: float,
        score_b: float,
        sector: str,
        event_date: Optional[str] = None,
    ) -> None:
        """Feed a completed game result into all model agents."""
        for model in [self.elo_agent, self.form_agent, self.poisson_agent]:
            model.update(team_a, team_b, score_a, score_b, sector, event_date)
        self.elo_agent.save_state()
        self.form_agent.save_state()
        self.poisson_agent.save_state()

    def subscribe(self, topic: str, handler) -> None:
        self.bus.subscribe(topic, handler)
