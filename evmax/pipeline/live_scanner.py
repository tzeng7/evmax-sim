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

from evmax.agents.cleanup.resolver import fetch_live_scores_for_sector, _fuzzy_team_match
from evmax.archiver import DataArchiver
from evmax.clients.kalshi import KalshiClient
from evmax.clients.pinnacle import PinnacleClient
from evmax.ev.calculator import calculate_ev
from evmax.ev.kelly import compute_kelly
from evmax.matching.engine import MatchingEngine
from evmax.matching.normalizer import NameNormalizer
from evmax.models.ev_bet import EVBet
from evmax.models.market import MarketType
from evmax.models_ml.live_win_prob import LiveWinProbModel, parse_time_elapsed
from evmax.models_ml.spread_distribution import SpreadDistributionModel
from evmax.sectors.registry import get_handler
from evmax.settings import get_settings

logger = structlog.get_logger(__name__)

_live_model = LiveWinProbModel()


class LiveScanner:
    """Scans in-progress games for live +EV opportunities on Kalshi."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._matching_engine = MatchingEngine()
        self._spread_model = SpreadDistributionModel()
        self._archiver = DataArchiver()

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
                pm_markets, sharp_odds_list, espn_live = await asyncio.gather(
                    kalshi.get_markets(sector),
                    pinnacle.get_odds(sector),
                    fetch_live_scores_for_sector(sector),
                    return_exceptions=True,
                )

            if isinstance(pm_markets, Exception) or isinstance(sharp_odds_list, Exception):
                logger.warning("live_scan_fetch_error", sector=sector)
                continue

            if isinstance(espn_live, Exception):
                espn_live = []

            # Only consider games that have already started
            live_sharp = [
                so for so in sharp_odds_list
                if so.event_date is not None and so.event_date < now_utc
            ]

            # Pinnacle closes pre-game lines once a game starts — fall back to
            # the most recent archived snapshot for today's games.
            if not live_sharp and pm_markets:
                live_sharp = self._load_archived_sharp_odds(sector, now_utc)
                if live_sharp:
                    logger.info(
                        "live_scan_using_archived_odds",
                        sector=sector,
                        count=len(live_sharp),
                    )

            if not live_sharp or not pm_markets:
                continue

            handler = get_handler(sector)
            pm_markets = [handler.enrich_market(m) for m in pm_markets]

            matched = self._matching_engine.match_all(pm_markets, live_sharp)

            # Compute true probabilities for all matched markets first,
            # then batch-fetch live prices via WebSocket in one connection.
            candidates = []
            for market, sharp_odds, _ in matched:
                if market.yes_price < 0.04 or market.yes_price > 0.96:
                    continue
                true_prob = self._resolve_live_prob(market, sharp_odds, sector, espn_live)
                if true_prob is None:
                    continue
                ticker = market.ticker or market.id.removeprefix("kalshi:")
                candidates.append((market, sharp_odds, true_prob, ticker))

            if not candidates:
                continue

            # Single WebSocket session for all tickers in this sector
            tickers = [ticker for _, _, _, ticker in candidates]
            async with KalshiClient() as kalshi:
                live_prices = await kalshi.get_market_asks_batch(tickers)

            for market, sharp_odds, true_prob, ticker in candidates:
                live_price = live_prices.get(ticker)
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

    def _load_archived_sharp_odds(self, sector: str, now_utc: datetime) -> list:
        """Load today's pre-game Pinnacle lines from archive.db.

        Called when Pinnacle returns no live odds (market closed after game start).
        Returns SharpOdds objects for games that started today (event_date = today).
        """
        from evmax.models.odds import SharpOdds, SharpBook

        today_str = now_utc.date().isoformat()
        rows = self._archiver.get_latest_sharp_odds(sector, today_str)
        if not rows:
            return []

        result = []
        for r in rows:
            try:
                # Reconstruct event_date as tz-aware datetime
                ed_raw = r.get("event_date")
                if ed_raw:
                    try:
                        ed = datetime.fromisoformat(ed_raw)
                        if ed.tzinfo is None:
                            ed = ed.replace(tzinfo=timezone.utc)
                    except ValueError:
                        ed = None
                else:
                    ed = None

                # Only include games that have already started
                if ed is None or ed >= now_utc:
                    continue

                so = SharpOdds(
                    event_id=r["event_id"],
                    book=SharpBook.pinnacle,
                    sector=sector,
                    outcome_a_label=r.get("outcome_a_label") or "",
                    outcome_b_label=r.get("outcome_b_label") or "",
                    outcome_a_decimal=r.get("outcome_a_decimal") or 2.0,
                    outcome_b_decimal=r.get("outcome_b_decimal") or 2.0,
                    outcome_draw_decimal=r.get("outcome_draw_decimal"),
                    true_prob_a=r.get("true_prob_a") or 0.5,
                    true_prob_b=r.get("true_prob_b") or 0.5,
                    true_prob_draw=r.get("true_prob_draw"),
                    margin=r.get("margin") or 0.0,
                    spread_line=r.get("spread_line"),
                    event_date=ed,
                )
                result.append(so)
            except Exception as e:
                logger.debug("archived_odds_parse_error", error=str(e))
                continue
        return result

    def _resolve_live_prob(
        self,
        market,
        sharp_odds,
        sector: str,
        espn_live: list[dict],
    ) -> Optional[float]:
        """Resolve true probability for the YES side using live game state.

        If ESPN live data contains a matching game, uses LiveWinProbModel to
        adjust the pre-game Pinnacle probability based on current score + time.
        Falls back to _resolve_true_prob (pre-game anchor) on any failure.
        """
        # Don't apply live adjustment to draw markets or spread markets
        if market.market_type == MarketType.spread:
            return self._resolve_true_prob(market, sharp_odds, sector)

        if market.yes_team in ("tie", "draw", "x"):
            return self._resolve_true_prob(market, sharp_odds, sector)

        # Try to find a matching ESPN live game
        live_state = self._find_espn_live_game(sharp_odds, espn_live)
        if live_state is None:
            return self._resolve_true_prob(market, sharp_odds, sector)

        # Compute pre-game probability for team_a (always outcome_a perspective)
        prior_prob_a = sharp_odds.true_prob_a
        if prior_prob_a is None or not (0 < prior_prob_a < 1):
            return self._resolve_true_prob(market, sharp_odds, sector)

        score_diff = live_state["score_a"] - live_state["score_b"]
        time_elapsed_pct = parse_time_elapsed(
            sector,
            live_state["period"],
            live_state["clock_secs"],
        )
        live_prob_a = _live_model.predict(sector, prior_prob_a, score_diff, time_elapsed_pct)

        # Map to YES side
        normalizer = NameNormalizer(sector)
        norm_b = normalizer.normalize(sharp_odds.outcome_b_label or "")
        if market.yes_team == norm_b:
            return 1.0 - live_prob_a
        return live_prob_a

    def _find_espn_live_game(
        self,
        sharp_odds,
        espn_live: list[dict],
    ) -> Optional[dict]:
        """Find the ESPN live entry matching sharp_odds team names.

        Returns dict with {score_a, score_b, period, clock_secs} where
        score_a corresponds to sharp_odds.outcome_a_label (home/team_a).
        Returns None if no confident match found.
        """
        MATCH_THRESHOLD = 65  # lenient for live matching
        team_a_raw = (sharp_odds.outcome_a_label or "").lower()
        team_b_raw = (sharp_odds.outcome_b_label or "").lower()

        for game in espn_live:
            home = game["home_name"]
            away = game["away_name"]

            score_home = game["score_home"]
            score_away = game["score_away"]

            # Check if team_a = home, team_b = away
            a_home = _fuzzy_team_match(team_a_raw, home) >= MATCH_THRESHOLD
            b_away = _fuzzy_team_match(team_b_raw, away) >= MATCH_THRESHOLD
            if a_home and b_away:
                return {
                    "score_a": score_home,
                    "score_b": score_away,
                    "period": game["period"],
                    "clock_secs": game["clock_secs"],
                }

            # Check if team_a = away, team_b = home
            a_away = _fuzzy_team_match(team_a_raw, away) >= MATCH_THRESHOLD
            b_home = _fuzzy_team_match(team_b_raw, home) >= MATCH_THRESHOLD
            if a_away and b_home:
                return {
                    "score_a": score_away,
                    "score_b": score_home,
                    "period": game["period"],
                    "clock_secs": game["clock_secs"],
                }

        return None

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
