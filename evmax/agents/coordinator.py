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
import dataclasses
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import structlog

from evmax.agents.base import Agent, AgentBus, AgentMessage, AgentRequest, AgentResponse
from evmax.archiver import DataArchiver
from evmax.agents.odds.kalshi_agent import KalshiOddsAgent
from evmax.agents.odds.sharp_agent import SharpOddsAgent
from evmax.agents.odds.ev_gap_agent import EVGapAgent, EVGap
from evmax.agents.models.elo_agent import EloModelAgent
from evmax.agents.models.form_agent import FormModelAgent
from evmax.agents.models.poisson_agent import PoissonModelAgent
from evmax.agents.models.ensemble_agent import EnsembleModelAgent, BlendedPrediction
from evmax.agents.models.tennis_model_agent import TennisModelAgent
from evmax.agents.models.tennis_serve_return_agent import TennisServeReturnAgent
from evmax.agents.models.tennis_h2h_agent import TennisH2HAgent
from evmax.agents.models.tennis_ranking_trend_agent import TennisRankingTrendAgent
from evmax.agents.models.pitcher_agent import PitcherModelAgent
from evmax.agents.intelligence.injury_agent import InjuryReportAgent, InjuryReport
from evmax.agents.intelligence.standings_agent import StandingsAgent, TeamStanding
from evmax.models.market import PredictionMarket
from evmax.models.odds import SharpOdds, SharpBook
from evmax.matching.engine import MatchingEngine

logger = structlog.get_logger(__name__)

_STEAM_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "steam_cache.json"
_STEAM_THRESHOLD = 0.02  # 2 percentage points


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
    exposure_guard_dropped: int = 0
    errors: list[str] = field(default_factory=list)
    bankroll: float = 250.0
    kelly_fraction: float = 0.5
    # Raw prop (SharpOdds, PredictionMarket) pairs for calibration logging
    # when _evaluate_prop is disabled and no EVGaps are created.
    prop_sharp_pairs: list[tuple[SharpOdds, PredictionMarket]] = field(default_factory=list)

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


def _load_steam_cache() -> dict[str, float]:
    """Load previous scan's sharp probabilities from disk."""
    try:
        if _STEAM_CACHE_PATH.exists():
            return json.loads(_STEAM_CACHE_PATH.read_text())
    except Exception as exc:
        logger.warning("steam_cache_load_failed", path=str(_STEAM_CACHE_PATH), error=str(exc))
    return {}


def _save_steam_cache(probs: dict[str, float]) -> None:
    try:
        _STEAM_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _STEAM_CACHE_PATH.write_text(json.dumps(probs))
    except Exception as exc:
        logger.error("steam_cache_save_failed", path=str(_STEAM_CACHE_PATH), error=str(exc))


def _detect_steam(
    sharp_odds: list,
    prev_probs: dict[str, float],
    threshold: float = _STEAM_THRESHOLD,
) -> set[str]:
    """Return event_ids where Pinnacle true_prob_a moved >= threshold since last scan."""
    steam: set[str] = set()
    for so in sharp_odds:
        prev = prev_probs.get(so.event_id)
        if prev is not None and abs(so.true_prob_a - prev) >= threshold:
            steam.add(so.event_id)
            logger.info(
                "steam_move_detected",
                event_id=so.event_id,
                prev=round(prev, 3),
                current=round(so.true_prob_a, 3),
                delta=round(so.true_prob_a - prev, 3),
            )
    return steam


def _apply_exposure_guard(
    gaps: list[EVGap],
    max_event_exposure: float = 0.08,
) -> list[EVGap]:
    """Cap total Kelly allocation per game to max_event_exposure (default 8%).

    Multiple markets on the same underlying game (ML + spread + total) are
    correlated — betting all at full Kelly compounds risk beyond the intended
    exposure. Best plays (by EV) consume budget first; lower-EV plays are
    scaled down or dropped when the cap is hit.
    """
    # Budget per base event (strip ::spread / ::total / ::spread::... suffixes)
    def _base_event(event_id: str) -> str:
        parts = event_id.split("::")
        # Prop events: "nba::2026-03-24::prop::player::stat::line"
        # Group by player (first 4 parts) so each player has its own budget
        if len(parts) > 3 and parts[2] == "prop":
            return "::".join(parts[:4])
        # Game events: "nba::2026-03-24::team_vs_team[::spread|total|...]"
        # Keep sector::date::matchup (first 3 parts)
        return "::".join(parts[:3])

    event_budget: dict[str, float] = {}
    guarded: list[EVGap] = []

    for gap in sorted(gaps, key=lambda g: g.ev_pct, reverse=True):
        base = _base_event(gap.event_id)
        used = event_budget.get(base, 0.0)
        remaining = max_event_exposure - used

        if remaining <= 0.005:  # < 0.5% left — skip
            logger.debug(
                "exposure_guard_dropped",
                event_id=gap.event_id,
                used=round(used, 4),
                cap=max_event_exposure,
            )
            continue

        if gap.kelly_fraction <= remaining:
            event_budget[base] = used + gap.kelly_fraction
            guarded.append(gap)
        else:
            # Scale down to fit remaining budget
            logger.debug(
                "exposure_guard_capped",
                event_id=gap.event_id,
                original=round(gap.kelly_fraction, 4),
                capped=round(remaining, 4),
            )
            capped = dataclasses.replace(gap, kelly_fraction=round(remaining, 4))
            event_budget[base] = max_event_exposure
            guarded.append(capped)

    return guarded


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
        self.standings_agent = StandingsAgent()

        # Statistical model agents
        self.elo_agent = EloModelAgent()
        self.form_agent = FormModelAgent()
        self.poisson_agent = PoissonModelAgent()
        self.tennis_agent = TennisModelAgent()
        self.tennis_serve_agent = TennisServeReturnAgent()
        self.tennis_h2h_agent = TennisH2HAgent()
        self.tennis_trend_agent = TennisRankingTrendAgent()
        self.pitcher_agent = PitcherModelAgent()
        self.ensemble_agent = EnsembleModelAgent(
            models=[
                self.elo_agent, self.form_agent, self.poisson_agent,
                self.tennis_agent, self.tennis_serve_agent,
                self.tennis_h2h_agent, self.tennis_trend_agent,
                self.pitcher_agent,
            ],
            sharp_weight=sharp_weight,
        )

        self._matching = MatchingEngine()
        self._archiver = DataArchiver()

        from evmax.notifications import Notifier
        self._notifier = Notifier.from_settings()

        for agent in self._all_agents():
            agent.attach_bus(self.bus)

        self.log = structlog.get_logger(__name__)

    def _all_agents(self) -> list[Agent]:
        return [
            self.kalshi_agent, self.sharp_agent, self.ev_gap_agent,
            self.injury_agent, self.standings_agent,
            self.elo_agent, self.form_agent, self.poisson_agent,
            self.tennis_agent, self.tennis_serve_agent,
            self.tennis_h2h_agent, self.tennis_trend_agent,
            self.pitcher_agent, self.ensemble_agent,
        ]

    # ------------------------------------------------------------------
    # Main cycle
    # ------------------------------------------------------------------

    async def run_cycle(self) -> CycleResult:
        t0 = time.perf_counter()
        correlation_id = str(uuid.uuid4())[:8]
        result = CycleResult(bankroll=self._bankroll, kelly_fraction=self._kelly_fraction)

        self.log.info("cycle_start", correlation_id=correlation_id, sectors=self._sectors)
        self._archiver.open_session(correlation_id, self._sectors, "agents")

        # All sectors run in parallel — Kalshi rate limiting is handled by the
        # module-level AsyncLimiter token bucket in kalshi.py (8 req/s).
        # Hard 45s timeout per sector: prevents a single hanging API call from
        # stalling the entire cycle indefinitely.
        async def _run_sector_with_timeout(sector: str) -> dict:
            try:
                return await asyncio.wait_for(
                    self._run_sector(sector, correlation_id),
                    timeout=45.0,
                )
            except asyncio.TimeoutError:
                self.log.warning("sector_timeout", sector=sector, timeout_s=45)
                raise asyncio.TimeoutError(f"{sector} timed out after 45s")

        sector_results = await asyncio.gather(
            *(_run_sector_with_timeout(sector) for sector in self._sectors),
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
            result.prop_sharp_pairs.extend(sr.get("prop_sharp_pairs", []))

        pre_guard = len(result.ev_gaps)
        result.ev_gaps = _apply_exposure_guard(result.ev_gaps)
        dropped = pre_guard - len(result.ev_gaps)
        if dropped > 0:
            self.log.info("exposure_guard_applied", dropped=dropped, remaining=len(result.ev_gaps))
        result.exposure_guard_dropped = dropped
        result.cycle_duration_s = time.perf_counter() - t0

        # Archive all raw fetched data for historical analysis
        k_total = s_total = 0
        for sector, sr in zip(self._sectors, sector_results):
            if isinstance(sr, Exception):
                continue
            k_total += self._archiver.archive_kalshi_markets(
                correlation_id, sector, sr.get("markets", [])
            )
            s_total += self._archiver.archive_sharp_odds(
                correlation_id, sector, sr.get("sharp_odds", [])
            )
        self._archiver.close_session(
            correlation_id,
            int(result.cycle_duration_s * 1000),
            k_total,
            s_total,
        )

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

        # Fire notifications (sync, non-blocking — uses urllib internally)
        try:
            self._notifier.notify_cycle(result)
        except Exception:
            pass

        return result

    # ------------------------------------------------------------------
    # Single-sector cycle
    # ------------------------------------------------------------------

    # Sectors that have Kalshi player prop series + TheOddsAPI prop coverage
    _PROP_SECTORS = {"nba", "nfl"}

    async def _run_sector(self, sector: str, correlation_id: str) -> dict:
        req = AgentRequest(sector=sector, correlation_id=correlation_id)

        # Steps 1-3: Fetch Kalshi + sharp + injuries concurrently
        # Also fetch player props for supported sectors
        fetch_tasks = [self.kalshi_agent(req), self.sharp_agent(req)]
        if self._enable_injuries:
            fetch_tasks.append(self.injury_agent(req))
        fetch_tasks.append(self.standings_agent(req))

        prop_task = None
        if sector.lower() in self._PROP_SECTORS:
            prop_task = asyncio.create_task(self._fetch_props(sector))

        fetch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        kalshi_resp = fetch_results[0]
        sharp_resp = fetch_results[1]
        _idx = 2
        injury_resp = None
        if self._enable_injuries:
            injury_resp = fetch_results[_idx]
            _idx += 1
        standings_resp = fetch_results[_idx]

        markets: list[PredictionMarket] = (
            kalshi_resp.data if not isinstance(kalshi_resp, Exception) else []
        ) or []
        sharp_odds: list[SharpOdds] = (
            sharp_resp.data if not isinstance(sharp_resp, Exception) else []
        ) or []

        # Merge prop markets and prop odds if available. Props run concurrently
        # with the main fetch (create_task above), so this timeout only gates
        # how long we wait *after* Kalshi/Pinnacle/injuries finish. 20s is
        # enough for stats.nba.com without blocking other sectors.
        prop_sharp_pairs: list[tuple[SharpOdds, PredictionMarket]] = []
        if prop_task is not None:
            try:
                prop_markets, prop_sharp = await asyncio.wait_for(prop_task, timeout=20.0)
                # Build (SharpOdds, PredictionMarket) pairs for calibration logging.
                # Match by player+stat+threshold since _fetch_props deduplicates markets.
                _pm_by_key = {
                    (m.player_name, m.stat_type, m.threshold): m
                    for m in prop_markets if m.player_name and m.stat_type
                }
                for so in prop_sharp:
                    key = (so.prop_player_name, so.prop_stat_type, so.total_line)
                    pm = _pm_by_key.get(key)
                    if pm is not None:
                        prop_sharp_pairs.append((so, pm))
                markets = markets + prop_markets
                sharp_odds = sharp_odds + prop_sharp
            except (asyncio.TimeoutError, Exception) as e:
                self.log.debug("prop_fetch_skipped", sector=sector, reason=str(e))
        injuries: dict[str, InjuryReport] = (
            injury_resp.data if injury_resp and not isinstance(injury_resp, Exception) else {}
        ) or {}
        standings: dict[str, TeamStanding] = (
            standings_resp.data if not isinstance(standings_resp, Exception) else {}
        ) or {}

        if isinstance(kalshi_resp, Exception):
            self.log.error("kalshi_failed", sector=sector, error=str(kalshi_resp))
        if isinstance(sharp_resp, Exception):
            self.log.error("sharp_failed", sector=sector, error=str(sharp_resp))

        # Steam move detection: compare current sharp probs to last scan's probs
        prev_probs = _load_steam_cache()
        steam_events = _detect_steam(sharp_odds, prev_probs)
        new_probs = {so.event_id: so.true_prob_a for so in sharp_odds}
        _save_steam_cache({**prev_probs, **new_probs})

        if not markets or not sharp_odds:
            return {
                "markets_fetched": len(markets),
                "markets_matched": 0,
                "ev_gaps": [],
                "blended_predictions": {},
                "injuries": injuries,
                "markets": markets,
                "sharp_odds": sharp_odds,
                "prop_sharp_pairs": prop_sharp_pairs,
            }

        # Step 4: Match non-prop markets → sharp events (for model ensemble)
        from evmax.models.market import MarketType as _MT
        regular_markets = [m for m in markets if m.market_type != _MT.player_prop]
        regular_sharp = [s for s in sharp_odds if s.prop_player_name is None]
        matched_pairs = self._matching.match_all(regular_markets, regular_sharp)
        pairs = [{"market": m, "sharp": s} for m, s, _ in matched_pairs]

        # Step 5: Ensemble model predictions
        blended: dict[str, BlendedPrediction] = {}
        model_probs: dict[str, float] = {}
        model_sources: dict[str, str] = {}

        blended_preds: dict = {}  # event_id → BlendedPrediction
        if self._enable_models and pairs:
            # Per-sector sharp_weight override comes from data/model_config.json's
            # `sharp_weight_by_sector` map (see evmax/agents/cleanup/metrics.py).
            # Tennis / baseball live there as defaults because our models are thin
            # and we want to trust Pinnacle more heavily on those sports.
            from evmax.agents.cleanup.metrics import load_config as _load_cfg
            cfg = _load_cfg()
            by_sector = cfg.get("sharp_weight_by_sector") or {}
            sector_sharp_weight = float(by_sector.get(sector.lower(), self._sharp_weight))
            ensemble_req = AgentRequest(
                sector=sector,
                params={
                    "pairs": pairs,
                    "sharp_weight": sector_sharp_weight,
                },
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
                "standings": standings,
                "model_sources": model_sources,
                "kelly_base_fraction": self._kelly_fraction,
                "steam_events": steam_events,
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
            "markets": markets,
            "sharp_odds": sharp_odds,
            "prop_sharp_pairs": prop_sharp_pairs,
        }

    async def _fetch_props(
        self,
        sector: str,
    ) -> tuple[list[PredictionMarket], list[SharpOdds]]:
        """Fetch Kalshi player prop markets and compute true probabilities.

        Uses a daily-refreshed local cache of player L15 game logs + team
        defensive stats. The cache is populated from stats.nba.com once per
        day (auto-refreshed on first scan if stale). During scans, all prop
        probabilities are computed from cached data with zero API calls.
        """
        from evmax.clients.kalshi import KalshiClient
        from evmax.clients.nba_props_cache import (
            PropResult,
            compute_prop_prob_cached,
            is_cache_fresh,
            refresh_props_cache,
        )

        prop_sector = f"{sector}_props"
        async with KalshiClient() as kalshi:
            raw = await kalshi.get_markets(prop_sector)

        prop_markets: list[PredictionMarket] = raw if not isinstance(raw, Exception) else []
        if not prop_markets:
            self.log.debug("props_fetched", sector=sector, prop_markets=0, prop_sharp=0)
            return [], []

        # Deduplicate (player, stat, threshold) so we only compute each prob once
        seen: set[tuple] = set()
        unique: list[PredictionMarket] = []
        for m in prop_markets:
            if m.player_name and m.stat_type and m.threshold is not None:
                key = (m.player_name, m.stat_type, m.threshold)
                if key not in seen:
                    seen.add(key)
                    unique.append(m)

        # Auto-refresh cache with only the players who have Kalshi prop markets
        if sector == "nba" and not is_cache_fresh():
            player_names = list({m.player_name for m in unique if m.player_name})
            self.log.info("props_cache_refreshing", players=len(player_names))
            await refresh_props_cache(force=True, player_names=player_names)

        # Compute probabilities from cached data (instant, no API calls)
        prop_sharp: list[SharpOdds] = []
        for market in unique:
            game_date = market.event_date.strftime("%Y-%m-%d") if market.event_date else None

            if sector == "nba":
                result = compute_prop_prob_cached(
                    market.player_name, market.stat_type, market.threshold, game_date,
                )
            else:
                result = None

            if result is None:
                continue

            date_str = game_date or "unknown"
            event_id = (
                f"{sector}::{date_str}::prop"
                f"::{market.player_name}::{market.stat_type}::{market.threshold}"
            )
            prop_sharp.append(SharpOdds(
                event_id=event_id,
                book=SharpBook.pinnacle,
                sector=sector,
                outcome_a_label="over",
                outcome_b_label="under",
                outcome_a_decimal=1.0,
                outcome_b_decimal=1.0,
                true_prob_a=0.0,
                true_prob_b=0.0,
                true_prob_over=result.prob,
                true_prob_under=1.0 - result.prob,
                total_line=market.threshold,
                margin=0.0,
                event_date=market.event_date,
                prop_player_name=market.player_name,
                prop_stat_type=market.stat_type,
                prop_l15_games=result.n_games,
                prop_minutes_volatile=result.minutes_volatile,
                prop_minutes_cv=result.minutes_cv,
            ))

        self.log.debug(
            "props_fetched",
            sector=sector,
            prop_markets=len(prop_markets),
            prop_sharp=len(prop_sharp),
        )
        return prop_markets, prop_sharp

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
        surface: str = "overall",
    ) -> None:
        """Feed a completed game result into all model agents."""
        for model in [self.elo_agent, self.form_agent, self.poisson_agent]:
            model.update(team_a, team_b, score_a, score_b, sector, event_date)
        self.elo_agent.save_state()
        self.form_agent.save_state()
        self.poisson_agent.save_state()
        if sector == "tennis":
            self.tennis_agent.update(team_a, team_b, score_a, score_b, sector, event_date, surface=surface)

    def next_scan_interval_seconds(self, result: "CycleResult") -> int:
        """Compute adaptive scan interval based on time until soonest game.

        Uses event_date from EV gaps found in the most recent cycle result.

        Tier logic (time to soonest kickoff):
          Live / already started: 90 sec
          < 60 min:               3 min
          60 min – 4 hours:       10 min
          > 4 hours / no games:   30 min
        """
        now_utc = datetime.now(timezone.utc)
        soonest_delta: float | None = None

        for gap in result.ev_gaps:
            ed = gap.event_date
            if ed is None:
                continue
            # Normalise to offset-aware UTC
            if not hasattr(ed, "tzinfo") or ed.tzinfo is None:
                ed = ed.replace(tzinfo=timezone.utc)
            delta = (ed - now_utc).total_seconds()
            if soonest_delta is None or delta < soonest_delta:
                soonest_delta = delta

        if soonest_delta is None:
            return 1800  # 30 min — no games found

        if soonest_delta <= 0:
            return 90   # game is live
        if soonest_delta < 60 * 60:
            return 180  # < 1 hour to kickoff
        if soonest_delta < 4 * 60 * 60:
            return 600  # 1–4 hours
        return 1800  # > 4 hours

    def subscribe(self, topic: str, handler) -> None:
        self.bus.subscribe(topic, handler)
