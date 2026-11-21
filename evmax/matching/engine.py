"""Matching engine: links prediction market events to sharp odds events.

Matching priority:
  1. Exact canonical key match
  2. Fuzzy match on the TEAM part only (rapidfuzz, threshold=88), with the
     date as a hard filter: the same ET game day for ET-dated sectors
     (``uses_et_game_day``), a ±1-day window elsewhere — see
     ``fuzzy_match_event_keys``
  3. Manual override JSON
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Optional

import structlog

from evmax.clients.time_util import kalshi_game_day
from evmax.matching.alignment import align_yes_side, spread_rung_prices_yes
from evmax.matching.fuzzy import FuzzyKeyIndex, fuzzy_match_event_keys_indexed
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

    def normalizer_for(self, sector: str) -> NameNormalizer:
        """Sector normalizer (cached) — shared with the YES-side alignment step."""
        return self._get_normalizer(sector)

    def _get_normalizer(self, sector: str) -> NameNormalizer:
        if sector not in self._normalizers:
            self._normalizers[sector] = NameNormalizer(sector)
        return self._normalizers[sector]

    def build_market_key(self, market: PredictionMarket) -> Optional[str]:
        """Build canonical event key from a prediction market."""
        base_key = self._base_key(market)
        if base_key is None:
            return None
        return self._typed_key(market, base_key)

    def _base_key(self, market: PredictionMarket) -> Optional[str]:
        """``sector::game-day::teams`` for a market, or None without teams/date."""
        if not market.team_home or not market.team_away:
            return None
        if not market.event_date:
            return None

        normalizer = self._get_normalizer(market.sector)
        date_str = kalshi_game_day(market.event_date, market.sector)
        return normalizer.normalize_event_key(
            market.team_home,
            market.team_away,
            date_str,
            market.sector,
        )

    @staticmethod
    def _typed_key(market: PredictionMarket, base_key: str) -> str:
        """The market's key: the base key plus its market-type suffix."""
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
        return self._match_indexed(market, _SharpIndex(sharp_odds_list))

    def _match_indexed(
        self,
        market: PredictionMarket,
        index: "_SharpIndex",
    ) -> Optional[tuple[SharpOdds, float]]:
        """``match`` against a prebuilt index of the sharp records.

        ``match_all`` builds the index ONCE per batch. Each lookup below is the
        index form of a scan the per-market path used to do over the whole
        sharp list, and returns the same record: the first one in list order.
        """
        # Advance markets may only match advance sharp records and vice versa.
        # Advance and regulation-ML probs differ materially (P(advance) >
        # P(win in 90') whenever a draw is possible), and the fuzzy fallback
        # below scores the ::advance suffix close enough to the bare key that
        # cross-type matches would otherwise slip through.
        is_advance = market.market_type == MarketType.advance
        part = index.partition(advance=is_advance)
        if not part.odds:
            return None

        # Check manual override first
        override_event_id = self._overrides.get(market.id)
        if override_event_id:
            so = part.first_by_id.get(override_event_id)
            if so is not None:
                logger.info("match_override", market_id=market.id, event_id=override_event_id)
                return so, 100.0

        # Build canonical key for the market
        base_key = self._base_key(market)
        market_key = self._typed_key(market, base_key) if base_key else None
        if not market_key:
            logger.debug("match_no_key", market_id=market.id, reason="missing teams/date")
            return None

        # 0. Alt-spread ladder: prefer the Pinnacle spread rung whose line matches
        # this market's line (main OR alternate) over the bare main-line exact
        # match below. A hit lets ev_gap price off the book's own devigged cover
        # prob for that exact line instead of extrapolating a normal CDF. When the
        # nearest rung is farther than the tolerance, fall through to the main-line
        # exact match → the CDF gap-filler (today's behaviour). Inert when the
        # ladder flag is off (no alt rungs are emitted, so this only ever finds
        # the main line, which the exact match would return anyway).
        from evmax.models_ml.spread_distribution import spread_ladder_enabled
        if (
            spread_ladder_enabled(market.sector)
            and market.market_type == MarketType.spread
            and market.line is not None
        ):
            nearest = self._find_nearest_spread(
                market, part.spreads_by_base.get(base_key, []), base_key=base_key,
            )
            if nearest:
                so, score = nearest
                logger.debug(
                    "match_spread_ladder", market_id=market.id,
                    kalshi_line=market.line, sharp_line=so.spread_line,
                    event_id=so.event_id,
                )
                return so, score

        # 1. Exact match
        so = part.first_by_id.get(market_key)
        if so is not None:
            logger.debug("match_exact", market_id=market.id, event_id=so.event_id)
            return so, 100.0

        # 2. Fuzzy match — disabled for spread/total markets (line must match exactly)
        if market.market_type not in (MarketType.spread, MarketType.total):
            fuzzy_result = fuzzy_match_event_keys_indexed(
                market_key,
                part.fuzzy_index(),
                threshold=self._settings.fuzzy_threshold,
            )
            if fuzzy_result:
                matched_key, score = fuzzy_result
                so = part.first_by_id.get(matched_key)
                if so is not None:
                    logger.debug(
                        "match_fuzzy",
                        market_id=market.id,
                        event_id=matched_key,
                        score=score,
                    )
                    return so, score

        # 3. Nearest-line fallback for totals — find closest Pinnacle line within 2 pts
        if market.market_type == MarketType.total and market.line is not None:
            nearest = self._find_nearest_total(
                market, part.totals_by_base.get(base_key, []), base_key=base_key,
            )
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
        base_key: Optional[str] = None,
    ) -> Optional[tuple[SharpOdds, float]]:
        """Find the Pinnacle total line closest to the Kalshi line within tolerance.

        ``base_key`` (the market's ``_base_key``) skips recomputing it; the
        indexed path passes it together with only this game's total records.
        """
        if base_key is None:
            base_key = self._base_key(market)
        if base_key is None:
            return None
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

    def _find_nearest_spread(
        self,
        market: PredictionMarket,
        sharp_odds_list: list[SharpOdds],
        base_key: Optional[str] = None,
    ) -> Optional[tuple[SharpOdds, float]]:
        """Find the Pinnacle spread rung whose |line| is nearest the market's
        |line|, within SPREAD_LADDER_LINE_TOLERANCE. Compares absolute lines so
        it is side-agnostic — a Kalshi "Pats -16.5" and "Seahawks +16.5" both
        match the Pinnacle 16.5 rung; align_yes_side orients the side downstream.
        Considers the main line (event_id ::spread) and every alternate rung
        (::spread::<line>) for this game."""
        from evmax.models_ml.spread_distribution import spread_ladder_tolerance

        tolerance = spread_ladder_tolerance(market.sector)

        if base_key is None:
            base_key = self._base_key(market)
        if base_key is None:
            return None
        normalizer = self._get_normalizer(market.sector)
        main_id = f"{base_key}::spread"
        alt_prefix = f"{base_key}::spread::"
        target = abs(market.line or 0.0)

        best_so: Optional[SharpOdds] = None
        best_dist = float("inf")
        for so in sharp_odds_list:
            if not (so.event_id == main_id or so.event_id.startswith(alt_prefix)):
                continue
            if so.spread_line is None:
                continue
            dist = abs(target - abs(so.spread_line))
            if not (dist <= tolerance and dist < best_dist):
                continue
            # Same |line| is not the same bet: the YES team must be the one
            # that holds this handicap on the rung (see spread_rung_prices_yes).
            if not spread_rung_prices_yes(
                market, so, align_yes_side(market, so, normalizer), tolerance,
            ):
                continue
            best_dist = dist
            best_so = so
        if best_so is None:
            return None
        # Exact line → 98; a half-point off → 93.
        confidence = max(90.0, 98.0 - best_dist * 10.0)
        return best_so, confidence

    def match_all(
        self,
        markets: list[PredictionMarket],
        sharp_odds_list: list[SharpOdds],
    ) -> list[tuple[PredictionMarket, SharpOdds, float]]:
        """
        Match a list of markets to sharp odds.

        When multiple markets match the same sharp event (e.g. playoff
        series games on Kalshi), keeps the one on the sharp event's game
        day, then the one closest in time (``_date_rank``).

        Returns:
            List of (market, sharp_odds, confidence) tuples.
        """
        raw: list[tuple[PredictionMarket, SharpOdds, float]] = []
        if not sharp_odds_list:
            return []
        # One index for the whole batch: the per-market path rescanned every
        # sharp record 5-6 times per market (O(markets x records)).
        index = _SharpIndex(sharp_odds_list)
        for market in markets:
            result = self._match_indexed(market, index)
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
            # The LINE is part of the key for spread/total. A single Pinnacle
            # spread record (::spread, the main line) is matched by EVERY Kalshi
            # alternate-spread rung of the same game/side, so without the line
            # in the key all but one rung collapse — the 2026-09-06 NFL scan
            # dropped 874 of 1137 matched markets this way, and each dropped rung
            # is a market that never gets an EV evaluation. The rungs are still
            # priced off that one Pinnacle record via SpreadDistributionModel
            # (gated to ±1σ from the main line in ev_gap_agent), so keeping every
            # rung is exactly the intended behavior. Totals already embed the
            # line in s.event_id, but a nearest-line fallback can map several
            # Kalshi total lines onto one Pinnacle record — the line in the key
            # protects those too. The playoff-duplicate collapse this dedup was
            # built for is preserved: true duplicate listings share a line.
            line_key = m.line if m.market_type in (MarketType.spread, MarketType.total) else None
            key = (s.event_id, mt, yes_side, venue, line_key)
            existing = best.get(key)
            if existing is None:
                best[key] = (m, s, conf)
                continue
            em = existing[0]
            if s.event_date and m.event_date and em.event_date:
                new_rank = _date_rank(m, s)
                old_rank = _date_rank(em, s)
                if new_rank < old_rank or (new_rank == old_rank and conf > existing[2]):
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


def _date_rank(market: PredictionMarket, sharp: SharpOdds) -> tuple[int, float]:
    """Dedup distance of a market from the sharp event it matched: (game-day
    gap, seconds gap). The game day decides first.

    Raw seconds alone favor the WRONG game for a late start: Kalshi dates
    anchor at noon UTC, so for a 20:40 ET first pitch (00:40Z next day) the
    NEXT day's anchor is 11h20m away and the correct one is 12h40m away. That
    kept Kalshi's next-game market and dropped the right one.
    """
    m_day = date.fromisoformat(kalshi_game_day(market.event_date, market.sector))
    s_day = date.fromisoformat(kalshi_game_day(sharp.event_date, sharp.sector))
    seconds = abs((market.event_date - sharp.event_date).total_seconds())
    return abs((m_day - s_day).days), seconds


class _SharpPartition:
    """Lookups over one partition (advance or non-advance) of the sharp records.

    Every list keeps the original record order, and ``first_by_id`` keeps the
    FIRST record per event_id, which is what the old linear scans returned.
    """

    def __init__(self, odds: list[SharpOdds]) -> None:
        self.odds = odds
        self.first_by_id: dict[str, SharpOdds] = {}
        # base game key -> records whose id starts with f"{base}::total::"
        self.totals_by_base: dict[str, list[SharpOdds]] = {}
        # base game key -> records whose id is f"{base}::spread" or starts with
        # f"{base}::spread::" (main line + alternate rungs, interleaved in order)
        self.spreads_by_base: dict[str, list[SharpOdds]] = {}
        for so in odds:
            eid = so.event_id
            self.first_by_id.setdefault(eid, so)
            if "::total::" in eid:
                self.totals_by_base.setdefault(eid.split("::total::", 1)[0], []).append(so)
            if eid.endswith("::spread"):
                self.spreads_by_base.setdefault(eid[: -len("::spread")], []).append(so)
            elif "::spread::" in eid:
                self.spreads_by_base.setdefault(eid.split("::spread::", 1)[0], []).append(so)
        self._fuzzy: Optional[FuzzyKeyIndex] = None

    def fuzzy_index(self) -> FuzzyKeyIndex:
        """Built on first use: most batches match exactly and never need it."""
        if self._fuzzy is None:
            self._fuzzy = FuzzyKeyIndex([so.event_id for so in self.odds])
        return self._fuzzy


class _SharpIndex:
    """Sharp records split by the advance rule, each half indexed lazily."""

    def __init__(self, odds: list[SharpOdds]) -> None:
        self._odds = odds
        self._parts: dict[bool, _SharpPartition] = {}

    def partition(self, advance: bool) -> _SharpPartition:
        part = self._parts.get(advance)
        if part is None:
            part = _SharpPartition([
                so for so in self._odds
                if so.event_id.endswith("::advance") == advance
            ])
            self._parts[advance] = part
        return part
