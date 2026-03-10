"""Polymarket Gamma API client (no authentication required).

Uses the /events endpoint (not /markets) since Polymarket's tag parameter on
/markets does not filter correctly. Instead we fetch active events and filter
client-side by slug prefix — Polymarket consistently names sports events with
sector-specific slug prefixes (e.g. "nba-", "nfl-", "cs2-").

Markets are embedded inside each event object, so one /events call returns
both event metadata and all child markets in a single round-trip.

Gamma API docs: https://docs.polymarket.com/
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import structlog

from evmax.clients.base import BaseAPIClient
from evmax.models.market import MarketSource, MarketType, PredictionMarket

logger = structlog.get_logger(__name__)

GAMMA_BASE_URL = "https://gamma-api.polymarket.com"

# Slug prefixes and title keywords that identify each sector's events.
# Polymarket event slugs reliably start with these strings for game-level markets.
SECTOR_SLUG_PREFIXES: dict[str, list[str]] = {
    "nfl": ["nfl-"],
    "nba": ["nba-"],
    "ncaab": ["ncaa-", "ncaab-", "college-basketball-", "march-madness-"],
    "soccer": [
        "epl-", "premier-league-", "mls-", "champions-league-", "ucl-",
        "la-liga-", "bundesliga-", "serie-a-", "ligue-1-", "europa-league-",
        "soccer-", "football-match-",
    ],
    "lol": ["lol-", "league-of-legends-", "lck-", "lec-", "lcs-", "worlds-"],
    "cs2": ["cs2-", "csgo-", "counter-strike-", "cs-go-"],
}

# Title keyword fallback when no slug prefix matches
SECTOR_TITLE_KEYWORDS: dict[str, list[str]] = {
    "nfl": ["nfl:", "nfl "],
    "nba": ["nba:", "nba "],
    "ncaab": ["ncaa", "college basketball"],
    "soccer": ["premier league", "champions league", "mls", "la liga", "bundesliga", "serie a"],
    "lol": ["league of legends", "lol:", "lck:", "lec:", "lcs:"],
    "cs2": ["cs2:", "csgo:", "counter-strike"],
}


class PolymarketClient(BaseAPIClient):
    """Client for Polymarket's Gamma API (no auth).

    Queries /events with active=true and filters by slug prefix to get
    sports-specific events. Each event embeds its child markets.
    """

    def __init__(self) -> None:
        super().__init__(
            base_url=GAMMA_BASE_URL,
            concurrency=3,
            timeout=20.0,
        )

    async def get_markets(
        self,
        sector: str,
        limit: int = 200,
    ) -> list[PredictionMarket]:
        """Fetch active markets for a sector by fetching events and filtering by slug."""
        slug_prefixes = SECTOR_SLUG_PREFIXES.get(sector.lower(), [])
        title_keywords = SECTOR_TITLE_KEYWORDS.get(sector.lower(), [])

        all_markets: list[PredictionMarket] = []
        seen_ids: set[str] = set()

        try:
            data = await self._get(
                "/events",
                params={
                    "active": "true",
                    "closed": "false",
                    "limit": limit,
                },
            )
            events = data if isinstance(data, list) else data.get("data", [])
        except Exception as e:
            logger.warning("polymarket_events_fetch_failed", sector=sector, error=str(e))
            return []

        for event in events:
            if not self._event_matches_sector(event, slug_prefixes, title_keywords):
                continue

            for raw_market in event.get("markets", []):
                parsed = self._parse_market(raw_market, event, sector)
                if parsed and parsed.id not in seen_ids:
                    all_markets.append(parsed)
                    seen_ids.add(parsed.id)

        logger.debug("polymarket_fetched", sector=sector, count=len(all_markets))
        return all_markets

    def _event_matches_sector(
        self,
        event: dict[str, Any],
        slug_prefixes: list[str],
        title_keywords: list[str],
    ) -> bool:
        """Return True if this event belongs to the target sector."""
        slug = (event.get("slug") or "").lower()
        title = (event.get("title") or "").lower()

        if any(slug.startswith(p) for p in slug_prefixes):
            return True
        if any(kw in title for kw in title_keywords):
            return True
        return False

    def _parse_market(
        self,
        raw: dict[str, Any],
        event: dict[str, Any],
        sector: str,
    ) -> Optional[PredictionMarket]:
        """Parse a Gamma market (embedded inside an event) into PredictionMarket."""
        try:
            condition_id = raw.get("conditionId") or raw.get("condition_id", "")
            slug = raw.get("slug") or event.get("slug", "")
            question = raw.get("question") or raw.get("title") or event.get("title", "")

            if not question:
                return None

            # Extract YES/NO prices from tokens or outcomePrices
            yes_price, no_price = self._extract_prices(raw)

            if not (0.01 <= yes_price <= 0.99):
                return None

            volume = float(raw.get("volume") or raw.get("volume24hr") or 0)

            # Event date: prefer market endDate, fall back to event endDate
            event_date = self._parse_date(
                raw.get("endDate") or event.get("endDate") or ""
            )

            team_home, team_away = self._extract_teams(question)

            return PredictionMarket(
                id=f"polymarket:{condition_id or slug}",
                source=MarketSource.polymarket,
                sector=sector,
                market_type=self._infer_market_type(question),
                title=question,
                ticker=slug,
                yes_price=yes_price,
                no_price=no_price,
                volume_usd=volume,
                open_interest_usd=0.0,
                team_home=team_home,
                team_away=team_away,
                event_date=event_date,
            )
        except Exception as e:
            logger.warning("polymarket_parse_failed", error=str(e))
            return None

    def _extract_prices(self, raw: dict[str, Any]) -> tuple[float, float]:
        """Return (yes_price, no_price) from a market dict."""
        yes_price = 0.5
        no_price = 0.5

        tokens = raw.get("tokens", [])
        if tokens:
            for token in tokens:
                outcome = (token.get("outcome") or "").upper()
                try:
                    price = float(token.get("price", 0.5))
                except (ValueError, TypeError):
                    continue
                if outcome == "YES":
                    yes_price = price
                elif outcome == "NO":
                    no_price = price
            return yes_price, no_price

        outcomes = raw.get("outcomes", [])
        prices = raw.get("outcomePrices", [])
        if outcomes and prices:
            for outcome, price in zip(outcomes, prices):
                try:
                    p = float(price)
                except (ValueError, TypeError):
                    continue
                if str(outcome).upper() == "YES":
                    yes_price = p
                elif str(outcome).upper() == "NO":
                    no_price = p

        return yes_price, no_price

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except ValueError:
            return None

    def _extract_teams(self, title: str) -> tuple[Optional[str], Optional[str]]:
        # Strip leading sport prefixes like "NBA: " or "NFL: "
        clean = title
        for prefix in ["NBA: ", "NFL: ", "NCAA: ", "EPL: ", "MLS: ", "LoL: ", "CS2: "]:
            if clean.upper().startswith(prefix.upper()):
                clean = clean[len(prefix):]
                break

        separators = [" vs ", " vs. ", " @ ", " at "]
        for sep in separators:
            if sep.lower() in clean.lower():
                idx = clean.lower().find(sep.lower())
                team_a = clean[:idx].strip().rstrip("?")
                team_b = clean[idx + len(sep):].strip().rstrip("?")
                for suffix in [" to win", " moneyline", " ML", " wins"]:
                    team_a = team_a.replace(suffix, "").replace(suffix.lower(), "")
                    team_b = team_b.replace(suffix, "").replace(suffix.lower(), "")
                if "@" in sep or sep.strip().lower() == "at":
                    return team_b.strip(), team_a.strip()
                return team_a.strip(), team_b.strip()
        return None, None

    def _infer_market_type(self, title: str) -> MarketType:
        tl = title.lower()
        if any(kw in tl for kw in ["over", "under", "total", "more than", "fewer than"]):
            return MarketType.total
        if any(kw in tl for kw in ["map", "round"]):
            return MarketType.map_handicap
        if any(kw in tl for kw in ["series", "advance", "championship", "champion", "win the"]):
            return MarketType.series_winner
        return MarketType.moneyline
