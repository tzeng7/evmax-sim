"""Matching engine: links prediction market events to sharp odds events.

Matching priority:
  1. Exact canonical key match
  2. Date-windowed fuzzy match (rapidfuzz, threshold=88)
  3. Manual override JSON
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import structlog

from evmax.matching.fuzzy import fuzzy_match_event_keys
from evmax.matching.normalizer import NameNormalizer
from evmax.models.market import MarketType, PredictionMarket
from evmax.models.odds import SharpOdds
from evmax.settings import get_settings

logger = structlog.get_logger(__name__)

OVERRIDES_PATH = Path(__file__).parent / "overrides.json"


class MatchingEngine:
    """
    Links prediction markets to their corresponding sharp odds events.

    Usage:
        engine = MatchingEngine()
        result = engine.match(market, sharp_odds_list)
        if result:
            sharp_odds, confidence = result
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._overrides = self._load_overrides()
        self._normalizers: dict[str, NameNormalizer] = {}

    def _load_overrides(self) -> dict[str, str]:
        """Load manual market_id → event_id overrides."""
        if OVERRIDES_PATH.exists():
            with open(OVERRIDES_PATH) as f:
                return json.load(f)
        return {}

    def _get_normalizer(self, sector: str) -> NameNormalizer:
        if sector not in self._normalizers:
            self._normalizers[sector] = NameNormalizer(sector)
        return self._normalizers[sector]

    def build_market_key(self, market: PredictionMarket) -> Optional[str]:
        """Build canonical event key from a prediction market."""
        if not market.team_home or not market.team_away:
            return None
        if not market.event_date:
            return None

        normalizer = self._get_normalizer(market.sector)
        date_str = market.event_date.strftime("%Y-%m-%d")
        base_key = normalizer.normalize_event_key(
            market.team_home,
            market.team_away,
            date_str,
            market.sector,
        )

        # Spread markets: line handled by SpreadDistributionModel downstream
        if market.market_type == MarketType.spread:
            return f"{base_key}::spread"

        # Total markets: embed the line so each total line is a distinct sharp record
        if market.market_type == MarketType.total:
            if market.line is not None:
                return f"{base_key}::total::{market.line}"
            return f"{base_key}::total"

        # Advance markets (knockout "to advance", winner incl. ET/pens): keyed
        # apart from the 3-way regulation moneyline record of the same game.
        if market.market_type == MarketType.advance:
            return f"{base_key}::advance"

        return base_key

    def match(
        self,
        market: PredictionMarket,
        sharp_odds_list: list[SharpOdds],
    ) -> Optional[tuple[SharpOdds, float]]:
        """
        Match a prediction market to the best sharp odds.

        Args:
            market: PredictionMarket to match.
            sharp_odds_list: Available sharp odds (for this sector).

        Returns:
            (SharpOdds, confidence_score) or None if no match found.
        """
        if not sharp_odds_list:
            return None

        # Advance markets may only match advance sharp records and vice versa.
        # Advance and regulation-ML probs differ materially (P(advance) >
        # P(win in 90') whenever a draw is possible), and the fuzzy fallback
        # below scores the ::advance suffix close enough to the bare key that
        # cross-type matches would otherwise slip through.
        is_advance = market.market_type == MarketType.advance
        sharp_odds_list = [
            so for so in sharp_odds_list
            if so.event_id.endswith("::advance") == is_advance
        ]
        if not sharp_odds_list:
            return None

        # Check manual override first
        override_event_id = self._overrides.get(market.id)
        if override_event_id:
            for so in sharp_odds_list:
                if so.event_id == override_event_id:
                    logger.info("match_override", market_id=market.id, event_id=override_event_id)
                    return so, 100.0

        # Build canonical key for the market
        market_key = self.build_market_key(market)
        if not market_key:
            logger.debug("match_no_key", market_id=market.id, reason="missing teams/date")
            return None

        sharp_keys = [so.event_id for so in sharp_odds_list]

        # 1. Exact match
        for so in sharp_odds_list:
            if so.event_id == market_key:
                logger.debug("match_exact", market_id=market.id, event_id=so.event_id)
                return so, 100.0

        # 2. Fuzzy match — disabled for spread/total markets (line must match exactly)
        if market.market_type not in (MarketType.spread, MarketType.total):
            fuzzy_result = fuzzy_match_event_keys(
                market_key,
                sharp_keys,
                threshold=self._settings.fuzzy_threshold,
            )
            if fuzzy_result:
                matched_key, score = fuzzy_result
                for so in sharp_odds_list:
                    if so.event_id == matched_key:
                        logger.debug(
                            "match_fuzzy",
                            market_id=market.id,
                            event_id=matched_key,
                            score=score,
                        )
                        return so, score

        # 3. Nearest-line fallback for totals — find closest Pinnacle line within 2 pts
        if market.market_type == MarketType.total and market.line is not None:
            nearest = self._find_nearest_total(market, sharp_odds_list)
            if nearest:
                so, score = nearest
                logger.debug(
                    "match_total_nearest_line",
                    market_id=market.id,
                    kalshi_line=market.line,
                    sharp_line=so.total_line,
                    event_id=so.event_id,
                )
                return so, score

        logger.debug("match_failed", market_id=market.id, market_key=market_key)
        return None

    def _find_nearest_total(
        self,
        market: PredictionMarket,
        sharp_odds_list: list[SharpOdds],
        tolerance: float = 2.0,
    ) -> Optional[tuple[SharpOdds, float]]:
        """Find the Pinnacle total line closest to the Kalshi line within tolerance."""
        if not market.team_home or not market.team_away or not market.event_date:
            return None

        normalizer = self._get_normalizer(market.sector)
        date_str = market.event_date.strftime("%Y-%m-%d")
        base_key = normalizer.normalize_event_key(
            market.team_home, market.team_away, date_str, market.sector
        )
        prefix = f"{base_key}::total::"

        best_so: Optional[SharpOdds] = None
        best_dist = float("inf")

        for so in sharp_odds_list:
            if not so.event_id.startswith(prefix):
                continue
            try:
                sharp_line = float(so.event_id.split(prefix, 1)[1])
            except (ValueError, IndexError):
                continue
            dist = abs((market.line or 0.0) - sharp_line)
            if dist <= tolerance and dist < best_dist:
                best_dist = dist
                best_so = so

        if best_so is None:
            return None

        # Confidence penalty: 5 points per unit of line distance
        confidence = max(50.0, 95.0 - best_dist * 5.0)
        return best_so, confidence

    def match_all(
        self,
        markets: list[PredictionMarket],
        sharp_odds_list: list[SharpOdds],
    ) -> list[tuple[PredictionMarket, SharpOdds, float]]:
        """
        Match a list of markets to sharp odds.

        When multiple markets match the same sharp event (e.g. playoff
        series games on Kalshi), keeps the one whose date is closest to
        the sharp event date.

        Returns:
            List of (market, sharp_odds, confidence) tuples.
        """
        raw: list[tuple[PredictionMarket, SharpOdds, float]] = []
        for market in markets:
            result = self.match(market, sharp_odds_list)
            if result:
                so, confidence = result
                raw.append((market, so, confidence))

        best: dict[tuple, tuple[PredictionMarket, SharpOdds, float]] = {}
        for m, s, conf in raw:
            mt = m.market_type.value if hasattr(m.market_type, "value") else str(m.market_type)
            # For moneyline/spread, each team's YES market is a distinct bet
            # (different Kalshi price, different edge). Include yes_team so
            # both sides survive dedup. For totals, over/under are distinct.
            # Venue is part of the key: the same game listed on Kalshi AND
            # Polymarket US is two independent books — this dedup only
            # collapses duplicate listings WITHIN a venue (e.g. Kalshi
            # playoff-series tickers).
            yes_side = (m.yes_team or "").lower().strip()
            venue = m.source.value if hasattr(m.source, "value") else str(m.source)
            key = (s.event_id, mt, yes_side, venue)
            existing = best.get(key)
            if existing is None:
                best[key] = (m, s, conf)
                continue
            em = existing[0]
            if s.event_date and m.event_date and em.event_date:
                new_delta = abs((m.event_date - s.event_date).total_seconds())
                old_delta = abs((em.event_date - s.event_date).total_seconds())
                if new_delta < old_delta:
                    best[key] = (m, s, conf)
            elif conf > existing[2]:
                best[key] = (m, s, conf)

        if len(best) < len(raw):
            logger.info(
                "match_dedup_playoff",
                raw=len(raw),
                deduped=len(best),
                dropped=len(raw) - len(best),
            )

        return list(best.values())
