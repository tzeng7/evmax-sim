"""Live EV scanner — finds +EV opportunities on in-progress games.

Uses Pinnacle's current live moneyline as the true probability anchor,
comparing against live Kalshi prices. Returns EVBet objects with full
Kelly sizing so they can be treated identically to pre-game bets.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

import structlog

from evmax.clients.kalshi import KalshiClient
from evmax.clients.pinnacle import PinnacleClient
from evmax.ev.calculator import calculate_ev
from evmax.ev.kelly import compute_kelly
from evmax.matching.engine import MatchingEngine
from evmax.matching.normalizer import NameNormalizer
from evmax.models.ev_bet import EVBet
from evmax.models.market import MarketType
from evmax.models_ml.spread_distribution import SpreadDistributionModel
from evmax.sectors.registry import get_handler
from evmax.settings import get_settings

logger = structlog.get_logger(__name__)


class LiveScanner:
    """Scans in-progress games for live +EV opportunities on Kalshi."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._matching_engine = MatchingEngine()
        self._spread_model = SpreadDistributionModel()

    async def scan(
        self,
        sectors: list[str],
        threshold: Optional[float] = None,
    ) -> list[EVBet]:
        """
        Scan all live (in-progress) games for +EV opportunities.

        Returns EVBet objects with live Kalshi prices and Kelly sizing.
        Does NOT save to DB or place bets.
        """
        if threshold is None:
            threshold = self._settings.ev_threshold

        now_utc = datetime.now(timezone.utc)
        results: list[EVBet] = []

        for sector in sectors:
            async with KalshiClient() as kalshi, PinnacleClient() as pinnacle:
                pm_markets, sharp_odds_list = await asyncio.gather(
                    kalshi.get_markets(sector),
                    pinnacle.get_odds(sector),
                    return_exceptions=True,
                )

            if isinstance(pm_markets, Exception) or isinstance(sharp_odds_list, Exception):
                logger.warning("live_scan_fetch_error", sector=sector)
                continue

            # Only consider games that have already started
            live_sharp = [
                so for so in sharp_odds_list
                if so.event_date is not None and so.event_date < now_utc
            ]
            if not live_sharp or not pm_markets:
                continue

            handler = get_handler(sector)
            pm_markets = [handler.enrich_market(m) for m in pm_markets]

            matched = self._matching_engine.match_all(pm_markets, live_sharp)

            for market, sharp_odds, _ in matched:
                if market.yes_price < 0.04 or market.yes_price > 0.96:
                    continue

                true_prob = self._resolve_true_prob(market, sharp_odds, sector)
                if true_prob is None:
                    continue

                # Fetch real-time Kalshi price for this market
                ticker = market.ticker or market.id.removeprefix("kalshi:")
                async with KalshiClient() as kalshi:
                    live_price = await kalshi.get_market_price(ticker)

                if live_price is None or live_price <= 0.04 or live_price >= 0.96:
                    continue

                payout = 1.0 / live_price
                ev, edge_pct = calculate_ev(live_price, true_prob)

                if ev < threshold:
                    continue

                kelly_result = compute_kelly(
                    true_prob=true_prob,
                    payout_decimal=payout,
                    edge_pct=edge_pct,
                    spread_pct=market.spread_pct,
                    max_kelly=self._settings.max_kelly_fraction,
                    min_kelly=self._settings.min_kelly_fraction,
                )
                if kelly_result.suggested_units <= 0:
                    continue

                results.append(EVBet(
                    market_id=market.id,
                    event_id=sharp_odds.event_id,
                    sector=sector,
                    outcome="yes",
                    market_implied_prob=live_price,
                    true_prob=true_prob,
                    payout_decimal=payout,
                    ev=ev,
                    edge_pct=edge_pct,
                    kelly_full=kelly_result.kelly_full,
                    kelly_fraction=kelly_result.kelly_fraction,
                    suggested_units=kelly_result.suggested_units,
                    volume_usd=market.volume_usd,
                    open_interest_usd=market.open_interest_usd,
                    spread_pct=market.spread_pct,
                    event_date=market.event_date,
                ))

        results.sort(key=lambda b: b.ev, reverse=True)
        return results

    def _resolve_true_prob(self, market, sharp_odds, sector: str) -> Optional[float]:
        """Resolve the correct true probability for the YES side."""
        if market.market_type == MarketType.spread and market.line is not None:
            if sharp_odds.spread_line is None:
                return None
            line_diff = abs(abs(market.line) - abs(sharp_odds.spread_line))
            if line_diff > self._settings.spread_line_tolerance:
                return None
            normalizer = NameNormalizer(sector)
            norm_b = normalizer.normalize(sharp_odds.outcome_b_label)
            yes_is_underdog = (market.yes_team == norm_b)
            pred = self._spread_model.predict(sharp_odds, market.line, sector, yes_is_underdog)
            return pred.true_prob if pred else None

        true_prob = sharp_odds.true_prob_a
        if market.yes_team is not None:
            if market.yes_team in ("tie", "draw", "x"):
                return sharp_odds.true_prob_draw
            normalizer = NameNormalizer(sector)
            norm_b = normalizer.normalize(sharp_odds.outcome_b_label)
            if market.yes_team == norm_b:
                true_prob = sharp_odds.true_prob_b
        return true_prob
