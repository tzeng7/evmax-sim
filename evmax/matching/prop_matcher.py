"""PropMatcher: links Kalshi player prop markets to Pinnacle sharp prop odds.

Matching logic:
  1. Exact stat_type match
  2. Exact or near-exact player name match (rapidfuzz >= 90)
  3. Threshold within tolerance (±0.5)
"""

from __future__ import annotations

from typing import Optional

import structlog
from rapidfuzz import fuzz

from evmax.models.market import PredictionMarket
from evmax.models.odds import SharpOdds

logger = structlog.get_logger(__name__)

PLAYER_MATCH_THRESHOLD = 85   # rapidfuzz score
THRESHOLD_TOLERANCE = 0.5     # how far apart O/U lines can be (e.g. 24.5 vs 24.5)


class PropMatcher:
    """Matches Kalshi player prop markets to Pinnacle sharp prop odds.

    A match requires:
      - Same stat_type (e.g. "points")
      - Player name fuzzy match >= PLAYER_MATCH_THRESHOLD
      - Threshold within THRESHOLD_TOLERANCE
    """

    def match(
        self,
        market: PredictionMarket,
        sharp_props: list[SharpOdds],
    ) -> Optional[tuple[SharpOdds, float]]:
        """Find the best matching SharpOdds for a prop PredictionMarket.

        Returns (sharp_odds, confidence) or None if no match found.
        """
        if market.player_name is None or market.stat_type is None or market.threshold is None:
            return None

        best: Optional[tuple[SharpOdds, float]] = None
        best_score = 0.0

        for so in sharp_props:
            if so.prop_stat_type != market.stat_type:
                continue
            if so.prop_player_name is None:
                continue
            if so.total_line is None:
                continue

            # Threshold check
            if abs(so.total_line - market.threshold) > THRESHOLD_TOLERANCE:
                continue

            # Player name fuzzy match
            score = fuzz.token_sort_ratio(
                market.player_name.lower(),
                so.prop_player_name.lower(),
            )
            if score < PLAYER_MATCH_THRESHOLD:
                continue

            if score > best_score:
                best_score = score
                best = (so, score / 100.0)

        return best

    def match_all(
        self,
        markets: list[PredictionMarket],
        sharp_props: list[SharpOdds],
    ) -> list[tuple[PredictionMarket, SharpOdds, float]]:
        """Match all prop markets against available sharp prop odds.

        Returns list of (market, sharp_odds, confidence) tuples.
        """
        results = []
        for market in markets:
            match = self.match(market, sharp_props)
            if match:
                so, conf = match
                results.append((market, so, conf))
                logger.debug(
                    "prop_matched",
                    player=market.player_name,
                    stat=market.stat_type,
                    threshold=market.threshold,
                    confidence=round(conf, 2),
                )
        return results
