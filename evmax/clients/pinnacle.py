"""Pinnacle odds via TheOddsAPI client.

TheOddsAPI commercially licenses Pinnacle lines.
Docs: https://the-odds-api.com/lol-api/

Key sports IDs on TheOddsAPI:
  americanfootball_nfl, basketball_nba, basketball_ncaab,
  soccer_usa_mls, soccer_epl, soccer_spain_la_liga, soccer_uefa_champs_league,
  esports_lol (League of Legends), esports_csgo (CS2/CS:GO)
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Optional

import structlog

from evmax.clients.base import BaseAPIClient
from evmax.ev.devig import devig_two_way, devig_three_way, american_to_decimal
from evmax.models.odds import SharpBook, SharpOdds
from evmax.settings import get_settings

logger = structlog.get_logger(__name__)

# In-memory cache: sector → (fetched_at_timestamp, results)
_CACHE: dict[str, tuple[float, list[SharpOdds]]] = {}
_CACHE_TTL = 300.0  # seconds

# Map our sector names → TheOddsAPI sport keys
SECTOR_SPORT_KEYS: dict[str, list[str]] = {
    "nfl": ["americanfootball_nfl"],
    "nba": ["basketball_nba"],
    "ncaab": ["basketball_ncaab"],
    "ncaaw": ["basketball_wncaab"],
    "soccer": [
        "soccer_usa_mls",
        "soccer_epl",
        "soccer_spain_la_liga",
        "soccer_germany_bundesliga",
        "soccer_italy_serie_a",
        "soccer_uefa_champs_league",
        "soccer_france_ligue_one",
    ],
    "baseball": ["baseball_mlb"],
    # lol / cs2 / valorant use EsportsPinnacleClient (Pinnacle guest API) — not TheOddsAPI
    "tennis": [
        "tennis_atp_miami_open",
        "tennis_wta_miami_open",
        "tennis_atp_french_open",
        "tennis_atp_wimbledon",
        "tennis_atp_australian_open",
        "tennis_atp_us_open",
        "tennis_wta_french_open",
        "tennis_wta_wimbledon",
        "tennis_wta_australian_open",
        "tennis_wta_us_open",
    ],
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
        """Fetch devigged Pinnacle odds (moneylines + spreads) for a sector.

        Results are cached in-process for _CACHE_TTL seconds to avoid
        redundant TheOddsAPI calls when multiple scan cycles run close together.
        """
        sector_key = sector.lower()
        cached = _CACHE.get(sector_key)
        if cached is not None:
            age = time.monotonic() - cached[0]
            if age < _CACHE_TTL:
                logger.debug("pinnacle_cache_hit", sector=sector_key, age_s=round(age, 1))
                return cached[1]

        sport_keys = SECTOR_SPORT_KEYS.get(sector_key, [])
        if not sport_keys:
            return []

        if not self._api_key:
            logger.warning("pinnacle_no_api_key", sector=sector)
            return []

        logger.debug("pinnacle_cache_miss", sector=sector_key)
        import asyncio
        # Fire ALL sport_key × market_type requests concurrently (was sequential per sport_key)
        tasks = [
            self._fetch_market_type(sport_key, market_type, sector)
            for sport_key in sport_keys
            for market_type in ("h2h", "spreads")
        ]
        all_results = await asyncio.gather(*tasks, return_exceptions=True)
        all_odds: list[SharpOdds] = []
        for r in all_results:
            if isinstance(r, list):
                all_odds.extend(r)
            elif isinstance(r, Exception):
                logger.warning("pinnacle_fetch_error", sector=sector_key, error=str(r))

        _CACHE[sector_key] = (time.monotonic(), all_odds)
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

    async def get_prop_odds(self, sector: str) -> list[SharpOdds]:
        """Fetch Pinnacle player prop lines via TheOddsAPI event-player-props endpoint.

        Supports nba and nfl. Returns SharpOdds with prop_player_name, prop_stat_type,
        and devigged true_prob_over/true_prob_under.
        """
        import asyncio as _asyncio

        base_sector = sector.lower().replace("_props", "")
        sport_keys = SECTOR_SPORT_KEYS.get(base_sector, [])
        if not sport_keys or not self._api_key:
            return []

        # Stat markets to fetch from TheOddsAPI
        prop_markets_by_sector: dict[str, list[str]] = {
            "nba": ["player_points", "player_rebounds", "player_assists",
                    "player_threes", "player_steals", "player_blocks",
                    "player_points_rebounds_assists"],
            "nfl": ["player_pass_yds", "player_rush_yds", "player_reception_yds", "player_anytime_td"],
        }
        stat_markets = prop_markets_by_sector.get(base_sector, [])
        if not stat_markets:
            return []

        # Player props require the event-level TheOddsAPI endpoint which is a premium feature.
        # Gracefully return empty if the plan doesn't support it.
        all_props: list[SharpOdds] = []
        _warned = False
        for sport_key in sport_keys:
            tasks = [
                self._fetch_player_props(sport_key, stat_mkt, base_sector)
                for stat_mkt in stat_markets
            ]
            results = await _asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, list):
                    all_props.extend(r)
                elif isinstance(r, Exception) and not _warned:
                    logger.warning(
                        "pinnacle_props_unavailable",
                        reason="Player props require a premium TheOddsAPI plan",
                        sport_key=sport_key,
                    )
                    _warned = True

        return all_props

    async def _fetch_player_props(
        self,
        sport_key: str,
        stat_market: str,
        sector: str,
        bookmakers: str = "pinnacle",
    ) -> list[SharpOdds]:
        """Fetch a single player-prop market type from TheOddsAPI."""
        try:
            data = await self._get(
                f"/sports/{sport_key}/odds",
                params={
                    "apiKey": self._api_key,
                    "regions": "us",
                    "markets": stat_market,
                    "bookmakers": bookmakers,
                    "oddsFormat": "decimal",
                },
            )
            props = []
            if not isinstance(data, list):
                return []

            for event in data:
                parsed = self._parse_prop_event(event, stat_market, sector, bookmakers)
                props.extend(parsed)
            return props
        except Exception as e:
            logger.debug("pinnacle_player_props_failed", sport_key=sport_key, stat_market=stat_market, error=str(e))
            return []

    async def get_us_book_prop_odds(self, sector: str, bookmakers: list[str] | None = None) -> list[SharpOdds]:
        """Fetch player prop lines from US sportsbooks (FanDuel, DraftKings, etc.) via TheOddsAPI.

        Uses the event-level endpoint (required for player props — the bulk endpoint
        doesn't support prop markets). Batches all stat markets into one call per event
        to minimise credit usage (~4 credits × n_bookmakers per event).

        Useful as a consensus reference for NBA props where Pinnacle has no coverage.
        """
        import asyncio as _asyncio

        base_sector = sector.lower().replace("_props", "")
        sport_keys = SECTOR_SPORT_KEYS.get(base_sector, [])
        if not sport_keys or not self._api_key:
            return []

        books = ",".join(bookmakers or ["fanduel", "draftkings"])
        prop_markets_by_sector: dict[str, list[str]] = {
            "nba": ["player_points", "player_rebounds", "player_assists",
                    "player_threes", "player_steals", "player_blocks",
                    "player_points_rebounds_assists"],
            "nfl": ["player_pass_yds", "player_rush_yds", "player_reception_yds", "player_anytime_td"],
        }
        stat_markets = prop_markets_by_sector.get(base_sector, [])
        if not stat_markets:
            return []

        all_props: list[SharpOdds] = []
        for sport_key in sport_keys:
            # Step 1: get event IDs for today's slate
            event_ids = await self._fetch_event_ids(sport_key)
            if not event_ids:
                continue

            # Step 2: one event-level call per event with ALL stat markets batched
            # Cost: 4 credits × n_bookmakers per event (not per market when batched)
            markets_param = ",".join(stat_markets)
            tasks = [
                self._fetch_event_props(sport_key, eid, markets_param, books, base_sector)
                for eid in event_ids
            ]
            results = await _asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, list):
                    all_props.extend(r)

        return all_props

    async def _fetch_event_ids(self, sport_key: str) -> list[str]:
        """Return event IDs for upcoming games in a sport."""
        try:
            data = await self._get(
                f"/sports/{sport_key}/events",
                params={"apiKey": self._api_key},
            )
            if isinstance(data, list):
                return [e["id"] for e in data if "id" in e]
        except Exception as e:
            logger.debug("pinnacle_event_ids_failed", sport_key=sport_key, error=str(e))
        return []

    async def _fetch_event_props(
        self,
        sport_key: str,
        event_id: str,
        markets_param: str,
        bookmakers: str,
        sector: str,
    ) -> list[SharpOdds]:
        """Fetch player props for a single event via the event-level endpoint.

        This is the correct endpoint for player props — the bulk /sports/{sport}/odds
        endpoint does not support player prop market types.
        Credit cost: 4 credits × number_of_bookmakers (independent of market count
        when batched in one call).
        """
        try:
            data = await self._get(
                f"/sports/{sport_key}/events/{event_id}/odds",
                params={
                    "apiKey": self._api_key,
                    "markets": markets_param,
                    "bookmakers": bookmakers,
                    "oddsFormat": "decimal",
                },
            )
            if not isinstance(data, dict):
                return []

            # Wrap as a single-event list so _parse_prop_event can process each market
            props: list[SharpOdds] = []
            event_wrapper = {
                "id": data.get("id", ""),
                "home_team": data.get("home_team", ""),
                "away_team": data.get("away_team", ""),
                "commence_time": data.get("commence_time", ""),
                "bookmakers": data.get("bookmakers", []),
            }
            # Parse each stat market from the batched response
            for stat_mkt in markets_param.split(","):
                parsed = self._parse_prop_event(event_wrapper, stat_mkt.strip(), sector, bookmakers)
                props.extend(parsed)
            return props
        except Exception as e:
            logger.debug("pinnacle_event_props_failed", event_id=event_id, error=str(e))
            return []

    def _parse_prop_event(
        self,
        event: dict[str, Any],
        stat_market: str,
        sector: str,
        bookmakers: str = "pinnacle",
    ) -> list[SharpOdds]:
        """Parse a TheOddsAPI event for player props.

        TheOddsAPI player prop format:
          market.key = "player_points", outcomes = [
            {"name": "LeBron James", "description": "Over", "price": 1.90, "point": 24.5},
            {"name": "LeBron James", "description": "Under", "price": 1.95, "point": 24.5},
          ]
        """
        try:
            from evmax.players import normalize_player_name

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
            game_key = f"{sector}::{date_str}::{self._normalize(home_team, sector)}_vs_{self._normalize(away_team, sector)}"

            # Collect outcomes from all bookmakers in the response, grouped by player.
            # When multiple books are requested, merge their lines (first seen wins per player).
            player_outcomes: dict[str, dict] = {}
            book_key_used = "unknown"
            for bm in event.get("bookmakers", []):
                book_key_used = bm.get("key", "unknown")
                target_market = next(
                    (m for m in bm.get("markets", []) if m.get("key") == stat_market),
                    None,
                )
                if not target_market:
                    continue
                for o in target_market.get("outcomes", []):
                    player = o.get("name", "")
                    desc = (o.get("description", "") or "").lower()
                    price = float(o.get("price", 2.0))
                    point = o.get("point")

                    if player not in player_outcomes:
                        player_outcomes[player] = {}
                    if "over" in desc:
                        player_outcomes[player].setdefault("over_price", price)
                        player_outcomes[player].setdefault("threshold", float(point) if point is not None else None)
                    elif "under" in desc:
                        player_outcomes[player].setdefault("under_price", price)

            # Build SharpOdds per player
            # Map TheOddsAPI stat_market key → canonical stat type
            stat_type_map = {
                "player_points": "points",
                "player_rebounds": "rebounds",
                "player_assists": "assists",
                "player_threes": "threes",
                "player_steals": "steals",
                "player_blocks": "blocks",
                "player_points_rebounds_assists": "points_rebounds_assists",
                "player_pass_yds": "passing_yards",
                "player_rush_yds": "rushing_yards",
                "player_reception_yds": "receiving_yards",
                "player_anytime_td": "touchdowns",
            }
            stat_type = stat_type_map.get(stat_market, stat_market)

            results = []
            for player_name, data in player_outcomes.items():
                over_price = data.get("over_price")
                under_price = data.get("under_price")
                threshold = data.get("threshold")

                if not over_price or not under_price or threshold is None:
                    continue

                try:
                    prob_over, prob_under, margin = devig_two_way(over_price, under_price)
                except Exception:
                    continue

                player_norm = normalize_player_name(player_name, sector)
                event_id = f"{game_key}::prop::{player_norm}::{stat_type}::{threshold}"

                try:
                    book_enum = SharpBook(book_key_used)
                except ValueError:
                    book_enum = SharpBook.pinnacle

                results.append(SharpOdds(
                    event_id=event_id,
                    book=book_enum,
                    sector=sector,
                    outcome_a_label=player_name,
                    outcome_b_label=player_name,
                    outcome_a_decimal=over_price,
                    outcome_b_decimal=under_price,
                    true_prob_a=prob_over,
                    true_prob_b=prob_under,
                    total_line=threshold,
                    true_prob_over=prob_over,
                    true_prob_under=prob_under,
                    prop_player_name=player_norm,
                    prop_stat_type=stat_type,
                    margin=margin,
                    event_date=event_date,
                ))
            return results
        except Exception as e:
            logger.warning("pinnacle_prop_event_parse_failed", error=str(e))
            return []

    @staticmethod
    def _normalize(name: str, sector: str = "") -> str:
        """Normalize team name via sector alias, then slugify."""
        from evmax.matching.normalizer import NameNormalizer
        normalized = NameNormalizer(sector).normalize(name) if sector else name.lower().strip()
        return normalized.replace(" ", "_").replace(".", "")
