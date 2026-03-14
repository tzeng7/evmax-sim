"""Pipeline runner — async orchestrator for the full scan cycle.

Execution order per cycle:
  1.  Concurrent fetch: all sectors × (Kalshi + Polymarket + Pinnacle/TheOddsAPI)
  2.  Parse via sector handlers
  3.  Match PM markets to sharp events
  4.  Devig Pinnacle odds (done in client)
  5.  Blend probabilities via SharpBooksModel
  6.  Compute EV for all matched markets
  7.  Size via Kelly
  8.  Filter to EV >= 2%
  9.  Pass to SimulationEngine (paper bets)
  10. Log BankrollSnapshot

Target: <8s per cycle via full concurrency.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from evmax.clients.kalshi import KalshiClient
from evmax.clients.polymarket import PolymarketClient
from evmax.clients.pinnacle import PinnacleClient
from evmax.db import AsyncSessionLocal
from evmax.matching.normalizer import NameNormalizer
from evmax.ev.calculator import evaluate_market
from evmax.ev.kelly import compute_kelly
from evmax.matching.engine import MatchingEngine
from evmax.models.ev_bet import EVBet, EVBetORM
from evmax.models.market import PredictionMarket
from evmax.models.odds import SharpOdds
from evmax.models.market import MarketType
from evmax.models_ml.sharp_only import SharpBooksModel
from evmax.models_ml.spread_distribution import SpreadDistributionModel
from evmax.models_ml.ncaab_efficiency import NCAABEfficiencyModel
from evmax.models_ml.base import blend_predictions
from evmax.sectors.registry import ALL_SECTORS, get_handler
from evmax.settings import get_settings
from evmax.simulation.engine import SimulationEngine

logger = structlog.get_logger(__name__)


@dataclass
class CycleResult:
    """Results from a single scan cycle."""

    sectors_scanned: list[str] = field(default_factory=list)
    markets_fetched: int = 0
    markets_matched: int = 0
    ev_bets_found: int = 0
    cycle_duration_s: float = 0.0
    ev_bets: list[EVBet] = field(default_factory=list)


class PipelineRunner:
    """
    Async orchestrator for the full EV scanning pipeline.
    """

    def __init__(
        self,
        sectors: Optional[list[str]] = None,
        simulate: bool = True,
        dry_run: bool = False,
    ) -> None:
        self._settings = get_settings()
        self._sectors = [s.lower() for s in (sectors or ALL_SECTORS)]
        self._simulate = simulate
        self._dry_run = dry_run
        self._matching_engine = MatchingEngine()
        self._model = SharpBooksModel()
        self._spread_model = SpreadDistributionModel()
        self._ncaab_eff_model = NCAABEfficiencyModel()
        self._sim_engine = SimulationEngine()

    async def run_cycle(self) -> CycleResult:
        """Execute one full scan-match-EV-size cycle."""
        start = time.monotonic()
        result = CycleResult(sectors_scanned=self._sectors)

        logger.info("pipeline_cycle_start", sectors=self._sectors)

        # Step 1: Concurrent data fetch across all sectors and sources
        fetch_tasks = [self._fetch_sector(sector) for sector in self._sectors]
        sector_data_list = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        all_ev_bets: list[EVBet] = []

        for sector, sector_data in zip(self._sectors, sector_data_list):
            if isinstance(sector_data, Exception):
                logger.warning("pipeline_sector_error", sector=sector, error=str(sector_data))
                continue

            markets, sharp_odds_list = sector_data
            result.markets_fetched += len(markets)

            if not markets or not sharp_odds_list:
                logger.info("pipeline_no_data", sector=sector, markets=len(markets), sharp=len(sharp_odds_list))
                continue

            # Step 2: Enrich markets via sector handler
            handler = get_handler(sector)
            markets = [handler.enrich_market(m) for m in markets]

            # Step 3: Match markets to sharp events
            matched = self._matching_engine.match_all(markets, sharp_odds_list)
            result.markets_matched += len(matched)

            logger.info(
                "pipeline_matched",
                sector=sector,
                markets=len(markets),
                sharp=len(sharp_odds_list),
                matched=len(matched),
            )

            # Steps 4-8: EV pipeline for matched pairs
            sector_ev_bets = await self._evaluate_matches(matched, sector)
            all_ev_bets.extend(sector_ev_bets)

        result.ev_bets = all_ev_bets
        result.ev_bets_found = len(all_ev_bets)

        # Step 9: Save EV bets and place simulated bets (skipped in dry_run mode)
        if all_ev_bets and not self._dry_run:
            await self._process_ev_bets(all_ev_bets)

        result.cycle_duration_s = time.monotonic() - start
        logger.info(
            "pipeline_cycle_done",
            markets_fetched=result.markets_fetched,
            matched=result.markets_matched,
            ev_bets=result.ev_bets_found,
            duration_s=round(result.cycle_duration_s, 2),
        )
        return result

    async def _fetch_sector(
        self,
        sector: str,
    ) -> tuple[list[PredictionMarket], list[SharpOdds]]:
        """Concurrently fetch PM markets + sharp odds for one sector."""
        handler = get_handler(sector)

        async with KalshiClient() as kalshi, PolymarketClient() as poly:
            pm_tasks = [
                kalshi.get_markets(sector),
                poly.get_markets(sector),
            ]
            pm_results = await asyncio.gather(*pm_tasks, return_exceptions=True)

        markets: list[PredictionMarket] = []
        for r in pm_results:
            if isinstance(r, list):
                markets.extend(r)
            elif isinstance(r, Exception):
                logger.warning("pm_fetch_error", sector=sector, error=str(r))

        # Sharp odds: all sectors use Pinnacle via TheOddsAPI
        sharp_odds: list[SharpOdds] = []
        async with PinnacleClient() as pinnacle:
            try:
                odds = await pinnacle.get_odds(sector)
                sharp_odds.extend(odds)
            except Exception as e:
                logger.warning("pinnacle_error", sector=sector, error=str(e))

        return markets, sharp_odds

    async def _evaluate_matches(
        self,
        matched: list[tuple[PredictionMarket, SharpOdds, float]],
        sector: str,
    ) -> list[EVBet]:
        """Run EV calculation + Kelly sizing for all matched pairs."""
        ev_bets: list[EVBet] = []

        now_utc = datetime.now(timezone.utc)
        # Kalshi encodes dates as midnight UTC of the ET game date (no time component).
        # Games actually happen in the evening ET, so add a 24h buffer before filtering:
        # only skip a market if event_date is more than 24h in the past.
        stale_threshold = now_utc - timedelta(hours=24)
        odds_ttl = timedelta(seconds=self._settings.max_odds_age_s)

        for market, sharp_odds, match_confidence in matched:
            # Skip markets whose event ended more than 24h ago
            if market.event_date is not None and market.event_date < stale_threshold:
                logger.debug("skip_past_event", market_id=market.id, event_date=market.event_date)
                continue

            # Skip live games — Pinnacle pre-game odds are no longer a valid true probability
            # once the game has started. sharp_odds.event_date comes from TheOddsAPI commence_time
            # which is the actual tipoff/kickoff datetime (not just date).
            if sharp_odds.event_date is not None and sharp_odds.event_date < now_utc:
                logger.debug(
                    "skip_live_game",
                    market_id=market.id,
                    commence_time=sharp_odds.event_date.isoformat(),
                )
                continue

            # Skip effectively-resolved in-play markets (price <4% or >96%)
            if market.yes_price < 0.04 or market.yes_price > 0.96:
                logger.debug("skip_extreme_price", market_id=market.id, yes_price=market.yes_price)
                continue

            # Skip if sharp odds are stale (older than max_odds_age_s)
            sharp_age = now_utc - sharp_odds.fetched_at.replace(tzinfo=timezone.utc) if sharp_odds.fetched_at.tzinfo is None else now_utc - sharp_odds.fetched_at
            if sharp_age > odds_ttl:
                logger.warning(
                    "skip_stale_sharp_odds",
                    market_id=market.id,
                    age_s=round(sharp_age.total_seconds()),
                    max_age_s=self._settings.max_odds_age_s,
                )
                continue

            # Skip if PM market price is stale
            market_age = now_utc - market.fetched_at.replace(tzinfo=timezone.utc) if market.fetched_at.tzinfo is None else now_utc - market.fetched_at
            if market_age > odds_ttl:
                logger.warning(
                    "skip_stale_market_price",
                    market_id=market.id,
                    age_s=round(market_age.total_seconds()),
                    max_age_s=self._settings.max_odds_age_s,
                )
                continue

            # Step 5: Get true probability from model
            if market.market_type == MarketType.spread and market.line is not None:
                # Skip alternative lines — only evaluate lines within tolerance of Pinnacle's
                if sharp_odds.spread_line is not None:
                    line_diff = abs(abs(market.line) - abs(sharp_odds.spread_line))
                    if line_diff > self._settings.spread_line_tolerance:
                        logger.debug(
                            "skip_alt_spread",
                            market_id=market.id,
                            market_line=market.line,
                            pinnacle_line=sharp_odds.spread_line,
                            diff=line_diff,
                        )
                        continue

                # Detect whether YES side is the underdog (outcome_b)
                normalizer = NameNormalizer(sector)
                norm_b = normalizer.normalize(sharp_odds.outcome_b_label)
                yes_is_underdog = (market.yes_team == norm_b)

                # Spread distribution model: computes P(YES side covers market.line)
                # directly, so no further alignment swap is needed.
                spread_pred = self._spread_model.predict(
                    sharp_odds, market.line, sector, yes_is_underdog
                )
                if not spread_pred:
                    continue
                blended_odds = sharp_odds.model_copy(
                    update={
                        "true_prob_a": spread_pred.true_prob,
                        "true_prob_b": 1.0 - spread_pred.true_prob,
                    }
                )
            else:
                # For NCAAB moneylines, blend SharpBooksModel (60%) with
                # the efficiency model (40%) for an independent probability check.
                if sector == "ncaab" and market.market_type == MarketType.moneyline:
                    sharp_pred = await self._model.predict(market, sharp_odds)
                    eff_pred = await self._ncaab_eff_model.predict(market, sharp_odds)
                    if sharp_pred and eff_pred:
                        prediction = blend_predictions([(sharp_pred, 0.6), (eff_pred, 0.4)])
                    else:
                        prediction = sharp_pred or eff_pred
                else:
                    prediction = await self._model.predict(market, sharp_odds)
                if not prediction:
                    continue
                blended_odds = sharp_odds.model_copy(
                    update={
                        "true_prob_a": prediction.true_prob_a,
                        "true_prob_b": prediction.true_prob_b,
                    }
                )

                # Align probabilities based on which team/outcome the YES side represents.
                if market.yes_team is not None:
                    # Draw/TIE market: YES = draw → use true_prob_draw
                    if market.yes_team in ("tie", "draw", "x"):
                        if blended_odds.true_prob_draw is None:
                            logger.debug("skip_draw_no_prob", market_id=market.id)
                            continue
                        blended_odds = blended_odds.model_copy(
                            update={"true_prob_a": blended_odds.true_prob_draw}
                        )
                    else:
                        # YES = away team → swap a/b so evaluate_market sees correct probs
                        normalizer = NameNormalizer(sector)
                        norm_b = normalizer.normalize(sharp_odds.outcome_b_label)
                        if market.yes_team == norm_b:
                            blended_odds = blended_odds.model_copy(
                                update={
                                    "true_prob_a": blended_odds.true_prob_b,
                                    "true_prob_b": blended_odds.true_prob_a,
                                }
                            )
                            logger.debug(
                                "yes_team_swap",
                                market_id=market.id,
                                yes_team=market.yes_team,
                                outcome_b=norm_b,
                            )

            # Step 6: Compute EV
            ev_results = evaluate_market(
                market,
                blended_odds,
                ev_threshold=self._settings.ev_threshold,
            )

            for ev_result in ev_results:
                # Step 7: Kelly sizing
                kelly_result = compute_kelly(
                    true_prob=ev_result.true_prob,
                    payout_decimal=ev_result.payout_decimal,
                    edge_pct=ev_result.edge_pct,
                    spread_pct=market.spread_pct,
                    max_kelly=self._settings.max_kelly_fraction,
                    min_kelly=self._settings.min_kelly_fraction,
                )

                if kelly_result.suggested_units <= 0:
                    continue

                ev_bet = EVBet(
                    market_id=market.id,
                    event_id=sharp_odds.event_id,
                    sector=sector,
                    outcome=ev_result.outcome,
                    market_implied_prob=ev_result.market_implied_prob,
                    true_prob=ev_result.true_prob,
                    payout_decimal=ev_result.payout_decimal,
                    ev=ev_result.ev,
                    edge_pct=ev_result.edge_pct,
                    kelly_full=kelly_result.kelly_full,
                    kelly_fraction=kelly_result.kelly_fraction,
                    suggested_units=kelly_result.suggested_units,
                    volume_usd=market.volume_usd,
                    open_interest_usd=market.open_interest_usd,
                    spread_pct=market.spread_pct,
                    event_date=market.event_date,
                )
                ev_bets.append(ev_bet)

        return ev_bets

    async def place_bets(self, ev_bets: list[EVBet]) -> None:
        """Public entry point: save EV bets to DB and place simulated bets."""
        await self._process_ev_bets(ev_bets)

    async def _process_ev_bets(self, ev_bets: list[EVBet]) -> None:
        """Save EV bets to DB and optionally place simulated bets."""
        async with AsyncSessionLocal() as session:
            async with KalshiClient() as kalshi:
                for ev_bet in ev_bets:
                    orm = ev_bet.to_orm()
                    session.add(orm)
                    await session.flush()

                    if self._simulate:
                        # Gate on minimum volume — zero-volume markets are stale prices
                        if ev_bet.volume_usd < self._settings.min_volume_usd:
                            logger.debug(
                                "sim_skip_low_volume",
                                market_id=ev_bet.market_id,
                                volume_usd=ev_bet.volume_usd,
                                min=self._settings.min_volume_usd,
                            )
                            continue

                        # Live price check: re-fetch Kalshi price and recompute EV
                        if ev_bet.market_id.startswith("kalshi:"):
                            ticker = ev_bet.market_id.removeprefix("kalshi:")
                            live_price = await kalshi.get_market_price(ticker)
                            if live_price is None:
                                logger.warning("live_price_check_failed", market_id=ev_bet.market_id)
                                continue
                            live_payout = 1.0 / live_price
                            live_ev = ev_bet.true_prob * live_payout - 1.0
                            if live_ev < self._settings.ev_threshold:
                                logger.info(
                                    "skip_live_price_drift",
                                    market_id=ev_bet.market_id,
                                    scan_ev=round(ev_bet.ev, 4),
                                    live_ev=round(live_ev, 4),
                                    scan_price=round(ev_bet.market_implied_prob, 3),
                                    live_price=round(live_price, 3),
                                )
                                continue
                            # Update bet fields to reflect live price
                            ev_bet = ev_bet.model_copy(update={
                                "market_implied_prob": live_price,
                                "payout_decimal": live_payout,
                                "ev": live_ev,
                            })
                            logger.debug(
                                "live_price_confirmed",
                                market_id=ev_bet.market_id,
                                live_price=round(live_price, 3),
                                live_ev=round(live_ev, 4),
                            )

                        await self._sim_engine.place_bet(ev_bet, orm.id, session)

            await session.commit()
