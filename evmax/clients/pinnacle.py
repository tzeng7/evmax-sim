"""Pinnacle odds via TheOddsAPI client.

TheOddsAPI commercially licenses Pinnacle lines.
Docs: https://the-odds-api.com/lol-api/

Key sports IDs on TheOddsAPI:
  americanfootball_nfl, basketball_nba, basketball_ncaab,
  soccer_usa_mls, soccer_epl, soccer_spain_la_liga, soccer_uefa_champs_league,
  esports_lol (League of Legends), esports_csgo (CS2/CS:GO)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import structlog

from evmax.clients.base import BaseAPIClient
from evmax.ev.devig import devig_two_way, devig_three_way, american_to_decimal
from evmax.models.odds import SharpBook, SharpOdds
from evmax.settings import get_settings

logger = structlog.get_logger(__name__)

# Map our sector names → TheOddsAPI sport keys
SECTOR_SPORT_KEYS: dict[str, list[str]] = {
    "nfl": ["americanfootball_nfl"],
    "nba": ["basketball_nba"],
    "ncaab": ["basketball_ncaab"],
    "soccer": [
        "soccer_usa_mls",
        "soccer_epl",
        "soccer_spain_la_liga",
        "soccer_germany_bundesliga",
        "soccer_italy_serie_a",
        "soccer_uefa_champs_league",
        "soccer_france_ligue_one",
    ],
    "lol": ["esports_lol"],
    "cs2": ["esports_csgo"],
}


class PinnacleClient(BaseAPIClient):
    """Fetches Pinnacle odds via TheOddsAPI."""

    def __init__(self) -> None:
        settings = get_settings()
        super().__init__(
            base_url=settings.the_odds_api_base_url,
            concurrency=3,
            timeout=15.0,
        )
        self._api_key = settings.the_odds_api_key

    async def get_odds(self, sector: str) -> list[SharpOdds]:
        """Fetch devigged Pinnacle odds (moneylines + spreads) for a sector."""
        sport_keys = SECTOR_SPORT_KEYS.get(sector.lower(), [])
        if not sport_keys:
            return []

        if not self._api_key:
            logger.warning("pinnacle_no_api_key", sector=sector)
            return []

        all_odds: list[SharpOdds] = []
        for sport_key in sport_keys:
            # Fetch moneylines and spreads in parallel
            import asyncio
            results = await asyncio.gather(
                self._fetch_market_type(sport_key, "h2h", sector),
                self._fetch_market_type(sport_key, "spreads", sector),
                return_exceptions=True,
            )
            for r in results:
                if isinstance(r, list):
                    all_odds.extend(r)
                elif isinstance(r, Exception):
                    logger.warning("pinnacle_fetch_failed", sport_key=sport_key, error=str(r))

        return all_odds

    async def _fetch_market_type(self, sport_key: str, market_type: str, sector: str) -> list[SharpOdds]:
        """Fetch a specific market type (h2h or spreads) from TheOddsAPI."""
        try:
            data = await self._get(
                f"/sports/{sport_key}/odds",
                params={
                    "apiKey": self._api_key,
                    "regions": "us",
                    "markets": market_type,
                    "bookmakers": "pinnacle",
                    "oddsFormat": "decimal",
                },
            )
            odds = []
            if isinstance(data, list):
                for event in data:
                    if market_type == "h2h":
                        parsed = self._parse_event(event, sector)
                        if parsed:
                            odds.append(parsed)
                    elif market_type == "spreads":
                        parsed_list = self._parse_spread_event(event, sector)
                        odds.extend(parsed_list)
            return odds
        except Exception as e:
            logger.warning("pinnacle_fetch_failed", sport_key=sport_key, market_type=market_type, error=str(e))
            return []

    def _parse_event(self, event: dict[str, Any], sector: str) -> Optional[SharpOdds]:
        """Parse a TheOddsAPI event into SharpOdds."""
        try:
            event_id_raw = event.get("id", "")
            home_team = event.get("home_team", "")
            away_team = event.get("away_team", "")
            sport = event.get("sport_key", "")

            # Find Pinnacle bookmaker
            pinnacle_data = None
            for bm in event.get("bookmakers", []):
                if bm.get("key") == "pinnacle":
                    pinnacle_data = bm
                    break

            if not pinnacle_data:
                return None

            # Find h2h market
            h2h_market = None
            for mkt in pinnacle_data.get("markets", []):
                if mkt.get("key") == "h2h":
                    h2h_market = mkt
                    break

            if not h2h_market:
                return None

            outcomes = h2h_market.get("outcomes", [])
            if len(outcomes) < 2:
                return None

            # Build outcome map
            outcome_map: dict[str, float] = {}
            for o in outcomes:
                outcome_map[o["name"]] = float(o["price"])

            # Event date
            event_date = None
            commence_time = event.get("commence_time", "")
            if commence_time:
                try:
                    event_date = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
                except ValueError:
                    pass

            date_str = event_date.strftime("%Y-%m-%d") if event_date else "unknown"
            canonical_event_id = (
                f"{sector}::{date_str}::{self._normalize(home_team, sector)}_vs_{self._normalize(away_team, sector)}"
            )

            # Three-way (soccer with draw)
            draw_decimal = outcome_map.get("Draw")
            home_decimal = outcome_map.get(home_team, list(outcome_map.values())[0])
            away_decimal = outcome_map.get(away_team, list(outcome_map.values())[1] if len(outcome_map) > 1 else 2.0)

            if draw_decimal:
                prob_a, prob_b, prob_draw, margin = devig_three_way(
                    home_decimal, away_decimal, draw_decimal
                )
            else:
                prob_a, prob_b, margin = devig_two_way(home_decimal, away_decimal)
                prob_draw = None

            return SharpOdds(
                event_id=canonical_event_id,
                book=SharpBook.pinnacle,
                sector=sector,
                outcome_a_label=home_team,
                outcome_b_label=away_team,
                outcome_a_decimal=home_decimal,
                outcome_b_decimal=away_decimal,
                outcome_draw_decimal=draw_decimal,
                true_prob_a=prob_a,
                true_prob_b=prob_b,
                true_prob_draw=prob_draw,
                margin=margin,
                event_date=event_date,
            )
        except Exception as e:
            logger.warning("pinnacle_parse_failed", error=str(e))
            return None

    def _parse_spread_event(self, event: dict[str, Any], sector: str) -> list[SharpOdds]:
        """
        Parse a TheOddsAPI event into SharpOdds objects for spread markets.

        Creates one SharpOdds per spread line where the covering team has a
        negative spread (i.e. the favorite). Event ID format:
          "{sector}::{date}::{home}_vs_{away}::spread::{covering_team}{line_int}"
        """
        try:
            home_team = event.get("home_team", "")
            away_team = event.get("away_team", "")

            commence_time = event.get("commence_time", "")
            event_date = None
            if commence_time:
                try:
                    event_date = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
                except ValueError:
                    pass

            date_str = event_date.strftime("%Y-%m-%d") if event_date else "unknown"
            home_norm = self._normalize(home_team, sector)
            away_norm = self._normalize(away_team, sector)
            base_event_id = f"{sector}::{date_str}::{home_norm}_vs_{away_norm}"

            # Find Pinnacle spreads market
            pinnacle_data = next(
                (bm for bm in event.get("bookmakers", []) if bm.get("key") == "pinnacle"),
                None,
            )
            if not pinnacle_data:
                return []

            spreads_market = next(
                (m for m in pinnacle_data.get("markets", []) if m.get("key") == "spreads"),
                None,
            )
            if not spreads_market:
                return []

            outcomes = spreads_market.get("outcomes", [])
            if len(outcomes) != 2:
                return []

            # Build outcome map: name → (price, point)
            outcome_map = {o["name"]: (float(o["price"]), float(o["point"])) for o in outcomes}

            # Identify covering team (negative spread = favorite)
            covering_team = next(
                (name for name, (_, pt) in outcome_map.items() if pt < 0), None
            )
            if not covering_team:
                return []

            cover_price, cover_point = outcome_map[covering_team]
            other_team = next(t for t in outcome_map if t != covering_team)
            other_price, _ = outcome_map[other_team]

            prob_cover, prob_other, margin = devig_two_way(cover_price, other_price)

            # Event ID at game level — line is stored in spread_line field
            event_id = f"{base_event_id}::spread"

            return [SharpOdds(
                event_id=event_id,
                book=SharpBook.pinnacle,
                sector=sector,
                outcome_a_label=covering_team,
                outcome_b_label=other_team,
                outcome_a_decimal=cover_price,
                outcome_b_decimal=other_price,
                true_prob_a=prob_cover,
                true_prob_b=prob_other,
                spread_line=cover_point,  # e.g. -7.5
                margin=margin,
                event_date=event_date,
            )]
        except Exception as e:
            logger.warning("pinnacle_spread_parse_failed", error=str(e))
            return []

    @staticmethod
    def _normalize(name: str, sector: str = "") -> str:
        """Normalize team name via sector alias, then slugify."""
        from evmax.matching.normalizer import NameNormalizer
        normalized = NameNormalizer(sector).normalize(name) if sector else name.lower().strip()
        return normalized.replace(" ", "_").replace(".", "")
