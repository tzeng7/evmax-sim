"""Pinnacle odds via the guest Arcadia API — covers all sectors.

No credentials required. Replaces TheOddsAPI for Pinnacle sharp lines.

Sport IDs:
  4  = Basketball (NBA id=487, NCAA id=493)
  12 = E Sports   (CS2, LoL, Valorant)
  15 = Football   (NFL id=258)
  29 = Soccer     (EPL id=1980, La Liga id=2196, Bundesliga id=1842,
                   Serie A id=2436, Ligue 1 id=2036, UCL id=2186)

Odds are American format; we convert via american_to_decimal() + devig_two_way().
Soccer three-way (draw) odds use devig_three_way().
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Optional

import structlog

from evmax.clients.base import BaseAPIClient
from evmax.ev.devig import devig_two_way, devig_three_way, american_to_decimal
from evmax.models.odds import SharpBook, SharpOdds

logger = structlog.get_logger(__name__)

GUEST_API_BASE = "https://guest.api.arcadia.pinnacle.com/0.1"

# sport_id → list of league_ids we care about
SECTOR_SPORT_LEAGUES: dict[str, tuple[int, list[int]]] = {
    "nba":      (4,  [487]),
    "ncaab":    (4,  [493]),
    "nfl":      (15, [258]),
    "baseball": (3,  [246]),    # MLB
    "ufc":      (22, []),       # Mixed Martial Arts — matched by league name
    "f1":       (44, []),       # Formula 1 — matched by league name
    "soccer":   (29, [1980, 2196, 1842, 2436, 2036, 2186]),  # EPL, La Liga, Bundesliga, Serie A, Ligue1, UCL
    "cs2":      (12, []),       # Esports — matched by league name
    "lol":      (12, []),
    "valorant": (12, []),
    "tennis":   (33, []),       # ATP/WTA — matched by league name (ATP Miami, WTA Miami, etc.)
}

# Sectors that use league-name matching instead of league IDs
NAME_MATCHED_LEAGUE_MAP: list[tuple[str, str]] = [
    ("CS2", "cs2"),
    ("Counter-Strike", "cs2"),
    ("League of Legends", "lol"),
    ("Valorant", "valorant"),
    ("UFC", "ufc"),
    ("Bellator", "ufc"),      # Bellator MMA also captured under ufc sector
    ("Formula 1", "f1"),
    ("Formula One", "f1"),
    ("ATP", "tennis"),
    ("WTA", "tennis"),
]

# All sectors this client can handle
ALL_SECTORS = set(SECTOR_SPORT_LEAGUES.keys())
# Sectors resolved by league name (not league ID list)
NAME_MATCHED_SECTORS = {"cs2", "lol", "valorant", "ufc", "f1", "tennis"}
ESPORTS_SECTORS = {"cs2", "lol", "valorant"}  # kept for backward compat

# Soccer leagues that have draws (all of them)
SOCCER_DRAW_LEAGUES = {1980, 2196, 1842, 2436, 2036, 2186}


def _name_matched_sector(league_name: str) -> Optional[str]:
    """Return sector for a league resolved by name matching (esports, UFC, F1)."""
    for keyword, sector in NAME_MATCHED_LEAGUE_MAP:
        if keyword.lower() in league_name.lower():
            return sector
    return None


class PinnacleGuestClient(BaseAPIClient):
    """
    Fetches sharp Pinnacle odds from the public guest Arcadia API.
    Covers NBA, NCAAB, NFL, Baseball, Soccer, UFC, F1, CS2, LoL, Valorant.
    No API key or account required.
    """

    def __init__(self) -> None:
        super().__init__(
            base_url=GUEST_API_BASE,
            concurrency=8,
            timeout=20.0,
            headers={
                "Accept": "application/json",
                "X-Api-Key": "CmX2KcMrXuFmNg6YFbmTxE0y9CIrOi0R",
            },
        )

    async def get_prop_odds(self, sector: str) -> list[SharpOdds]:
        """Fetch devigged Pinnacle player prop lines from the guest Arcadia API.

        Pinnacle posts player props as 'special' matchups with
        special.category == 'Player Props' and description like 'Luka Doncic (Points)'.
        Each prop has one over/under market at a specific line.

        Supports: nba, nfl, ncaab, baseball.
        """
        sector = sector.lower()
        _PROP_SECTORS = {"nba", "nfl", "ncaab", "baseball"}
        if sector not in _PROP_SECTORS or sector not in SECTOR_SPORT_LEAGUES:
            return []

        sport_id, league_ids = SECTOR_SPORT_LEAGUES[sector]

        try:
            all_matchups = await self._get(
                f"/sports/{sport_id}/matchups",
                params={"withSpecials": "true"},
            )
        except Exception as e:
            logger.warning("pinnacle_guest_props_matchups_failed", sector=sector, error=str(e))
            return []

        if not isinstance(all_matchups, list):
            return []

        # Filter to player prop specials for the right leagues
        prop_matchups = [
            m for m in all_matchups
            if m.get("type") == "special"
            and m.get("league", {}).get("id") in league_ids
            and (m.get("special") or {}).get("category", "").lower() == "player props"
        ]

        if not prop_matchups:
            logger.info("pinnacle_guest_no_props", sector=sector)
            return []

        results = await asyncio.gather(
            *(self._fetch_prop_matchup(m, sector) for m in prop_matchups),
            return_exceptions=True,
        )

        props: list[SharpOdds] = []
        for r in results:
            if isinstance(r, SharpOdds):
                props.append(r)
            elif isinstance(r, list):
                props.extend(r)

        logger.info("pinnacle_props_fetched", sector=sector, count=len(props))
        return props

    async def _fetch_prop_matchup(self, matchup: dict, sector: str) -> Optional[SharpOdds]:
        """Fetch and parse a single player prop special matchup."""
        matchup_id = matchup.get("id")
        special = matchup.get("special") or {}
        description = special.get("description", "")  # e.g. "Luka Doncic (Points)"

        # Parse "Player Name (Stat Type)"
        import re as _re
        m = _re.match(r"^(.+?)\s*\((.+?)\)$", description.strip())
        if not m:
            return None
        player_raw = m.group(1).strip()
        stat_raw = m.group(2).strip().lower()  # "Points", "Rebounds", etc.

        # Map Pinnacle stat label → canonical stat_type
        _STAT_MAP = {
            "points": "points", "pts": "points",
            "rebounds": "rebounds", "reb": "rebounds",
            "assists": "assists", "ast": "assists",
            "threes": "threes", "3-pointers": "threes", "3 pointers made": "threes",
            "steals": "steals", "stl": "steals",
            "blocks": "blocks", "blk": "blocks",
            "pts+reb+ast": "points_rebounds_assists",
            "points + rebounds + assists": "points_rebounds_assists",
            "pra": "points_rebounds_assists",
            "passing yards": "pass_yds", "passing yds": "pass_yds",
            "rushing yards": "rush_yds", "rushing yds": "rush_yds",
            "receiving yards": "reception_yds", "receiving yds": "reception_yds",
        }
        stat_type = _STAT_MAP.get(stat_raw)
        if stat_type is None:
            logger.debug("pinnacle_prop_unknown_stat", description=description)
            return None

        # Normalize player name
        from evmax.players import normalize_player_name
        player_norm = normalize_player_name(player_raw, sector)

        # Fetch markets for this prop matchup
        start_time = matchup.get("startTime", "")
        event_date: Optional[datetime] = None
        if start_time:
            try:
                event_date = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            except ValueError:
                pass

        try:
            markets_data = await self._get(f"/matchups/{matchup_id}/markets/related/straight")
        except Exception as e:
            logger.debug("pinnacle_prop_markets_failed", matchup_id=matchup_id, error=str(e))
            return None

        if not isinstance(markets_data, list):
            return None

        # Find the over/under market — prop markets have status=None (not "open")
        ou_market = next(
            (mk for mk in markets_data if mk.get("type") == "total"),
            None,
        )
        if not ou_market:
            return None

        prices = ou_market.get("prices", [])
        # Prop market prices have designation=None; positional: index 0 = over, index 1 = under
        over_entry  = next((p for p in prices if p.get("designation") == "over"),  None)
        under_entry = next((p for p in prices if p.get("designation") == "under"), None)
        if not over_entry or not under_entry:
            # Fall back to positional indexing for prop markets
            if len(prices) >= 2:
                over_entry, under_entry = prices[0], prices[1]
            else:
                return None

        raw_line = over_entry.get("points") or over_entry.get("handicap")
        if raw_line is None:
            return None
        total_line = float(raw_line)

        try:
            over_dec  = american_to_decimal(int(over_entry["price"]))
            under_dec = american_to_decimal(int(under_entry["price"]))
            prob_over, prob_under, margin = devig_two_way(over_dec, under_dec)
        except Exception as e:
            logger.debug("pinnacle_prop_devig_failed", error=str(e))
            return None

        date_str = event_date.strftime("%Y-%m-%d") if event_date else "unknown"
        event_id = f"{sector}::{date_str}::prop::{player_norm}::{stat_type}::{total_line}"

        return SharpOdds(
            event_id=event_id,
            book=SharpBook.pinnacle,
            sector=sector,
            outcome_a_label="over",
            outcome_b_label="under",
            outcome_a_decimal=over_dec,
            outcome_b_decimal=under_dec,
            true_prob_a=0.0,
            true_prob_b=0.0,
            true_prob_over=prob_over,
            true_prob_under=prob_under,
            total_line=total_line,
            margin=margin,
            event_date=event_date,
            prop_player_name=player_norm,
            prop_stat_type=stat_type,
        )

    async def get_odds(self, sector: str) -> list[SharpOdds]:
        """Fetch devigged Pinnacle moneyline odds for a sector."""
        sector = sector.lower()
        if sector not in ALL_SECTORS:
            return []

        sport_id, league_ids = SECTOR_SPORT_LEAGUES[sector]

        try:
            all_matchups = await self._get(
                f"/sports/{sport_id}/matchups",
                params={"withSpecials": "false"},
            )
        except Exception as e:
            logger.warning("pinnacle_guest_matchups_failed", sector=sector, error=str(e))
            return []

        if not isinstance(all_matchups, list):
            return []

        # Filter to parent matchups for this sector
        if sector in NAME_MATCHED_SECTORS:
            matchups = [
                m for m in all_matchups
                if m.get("parentId") is None
                and m.get("type") == "matchup"
                and _name_matched_sector(m.get("league", {}).get("name", "")) == sector
            ]
        else:
            matchups = [
                m for m in all_matchups
                if m.get("parentId") is None
                and m.get("type") == "matchup"
                and m.get("league", {}).get("id") in league_ids
            ]

        if not matchups:
            logger.info("pinnacle_guest_no_matchups", sector=sector)
            return []

        results = await asyncio.gather(
            *(self._fetch_matchup_odds(m, sector) for m in matchups),
            return_exceptions=True,
        )

        odds: list[SharpOdds] = []
        for r in results:
            if isinstance(r, SharpOdds):
                odds.append(r)
            elif isinstance(r, list):
                odds.extend(r)
            elif isinstance(r, Exception):
                logger.warning("pinnacle_guest_odds_error", error=str(r))

        logger.info("sharp_fetched", sector=sector, count=len(odds),
                    avg_margin=round(sum(o.margin for o in odds) / len(odds), 4) if odds else 0.0)
        return odds

    async def _fetch_matchup_odds(self, matchup: dict, sector: str) -> Optional[SharpOdds] | list[SharpOdds]:
        """Fetch moneyline (and spread for non-soccer) for one matchup."""
        matchup_id = matchup.get("id")
        if not matchup_id:
            return None

        participants = matchup.get("participants", [])
        home = next((p["name"] for p in participants if p.get("alignment") == "home"), None)
        away = next((p["name"] for p in participants if p.get("alignment") == "away"), None)
        if not home or not away:
            return None

        start_time = matchup.get("startTime", "")
        event_date: Optional[datetime] = None
        if start_time:
            try:
                event_date = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            except ValueError:
                pass

        date_str = event_date.strftime("%Y-%m-%d") if event_date else "unknown"
        home_norm = self._normalize(home, sector)
        away_norm = self._normalize(away, sector)
        base_event_id = f"{sector}::{date_str}::{home_norm}_vs_{away_norm}"

        try:
            markets_data = await self._get(f"/matchups/{matchup_id}/markets/related/straight")
        except Exception as e:
            logger.debug("pinnacle_guest_markets_failed", matchup_id=matchup_id, error=str(e))
            return None

        if not isinstance(markets_data, list):
            return None

        results: list[SharpOdds] = []

        # --- Moneyline ---
        ml_market = next(
            (m for m in markets_data
             if m.get("matchupId") == matchup_id
             and m.get("type") == "moneyline"
             and m.get("period") == 0
             and not m.get("isAlternate", False)
             and m.get("status") == "open"),
            None,
        )
        if ml_market:
            ml_odds = self._parse_moneyline(ml_market, base_event_id, sector, home, away, event_date)
            if ml_odds:
                results.append(ml_odds)

        # --- Spread (only for team sports with meaningful point spreads) ---
        if sector not in NAME_MATCHED_SECTORS and sector != "soccer":
            spread_market = next(
                (m for m in markets_data
                 if m.get("matchupId") == matchup_id
                 and m.get("type") == "spread"
                 and m.get("period") == 0
                 and not m.get("isAlternate", False)
                 and m.get("status") == "open"),
                None,
            )
            if spread_market:
                spread_odds = self._parse_spread(spread_market, base_event_id, sector, home, away, event_date)
                if spread_odds:
                    results.append(spread_odds)

        # --- Totals (scoring team sports only — excludes esports, UFC, F1, tennis) ---
        _TOTALS_SECTORS = {"nba", "nfl", "ncaab", "soccer", "baseball"}
        if sector in _TOTALS_SECTORS:
            totals_market = next(
                (m for m in markets_data
                 if m.get("matchupId") == matchup_id
                 and m.get("type") == "total"
                 and m.get("period") == 0
                 and not m.get("isAlternate", False)
                 and m.get("status") == "open"),
                None,
            )
            if totals_market:
                totals_odds = self._parse_totals(totals_market, base_event_id, sector, event_date)
                if totals_odds:
                    results.append(totals_odds)

        return results if results else None

    def _parse_moneyline(
        self, market: dict, base_event_id: str, sector: str,
        home: str, away: str, event_date: Optional[datetime],
    ) -> Optional[SharpOdds]:
        prices = market.get("prices", [])

        home_price = next((p["price"] for p in prices if p.get("designation") == "home"), None)
        away_price = next((p["price"] for p in prices if p.get("designation") == "away"), None)
        draw_price = next((p["price"] for p in prices if p.get("designation") == "draw"), None)

        if home_price is None or away_price is None:
            return None

        try:
            home_dec = american_to_decimal(int(home_price))
            away_dec = american_to_decimal(int(away_price))

            if draw_price is not None:
                draw_dec = american_to_decimal(int(draw_price))
                prob_a, prob_b, prob_draw, margin = devig_three_way(home_dec, away_dec, draw_dec)
            else:
                draw_dec = None
                prob_draw = None
                prob_a, prob_b, margin = devig_two_way(home_dec, away_dec)
        except Exception as e:
            logger.debug("pinnacle_guest_devig_failed", error=str(e))
            return None

        return SharpOdds(
            event_id=base_event_id,
            book=SharpBook.pinnacle,
            sector=sector,
            outcome_a_label=home,
            outcome_b_label=away,
            outcome_a_decimal=home_dec,
            outcome_b_decimal=away_dec,
            outcome_draw_decimal=draw_dec,
            true_prob_a=prob_a,
            true_prob_b=prob_b,
            true_prob_draw=prob_draw,
            margin=margin,
            event_date=event_date,
        )

    def _parse_spread(
        self, market: dict, base_event_id: str, sector: str,
        home: str, away: str, event_date: Optional[datetime],
    ) -> Optional[SharpOdds]:
        prices = market.get("prices", [])
        if len(prices) < 2:
            return None

        home_entry = next((p for p in prices if p.get("designation") == "home"), None)
        away_entry = next((p for p in prices if p.get("designation") == "away"), None)
        if not home_entry or not away_entry:
            return None

        # Pinnacle guest API uses "points" for the handicap value (not "handicap")
        home_hdp = home_entry.get("points") if "points" in home_entry else home_entry.get("handicap", 0.0)
        away_hdp = away_entry.get("points") if "points" in away_entry else away_entry.get("handicap", 0.0)

        # Covering team = negative handicap (favorite)
        if home_hdp < 0:
            covering_team, other_team = home, away
            cover_price, other_price = home_entry["price"], away_entry["price"]
            cover_point = float(home_hdp)
        elif away_hdp < 0:
            covering_team, other_team = away, home
            cover_price, other_price = away_entry["price"], home_entry["price"]
            cover_point = float(away_hdp)
        else:
            return None

        try:
            cover_dec = american_to_decimal(int(cover_price))
            other_dec = american_to_decimal(int(other_price))
            prob_cover, prob_other, margin = devig_two_way(cover_dec, other_dec)
        except Exception as e:
            logger.debug("pinnacle_guest_spread_devig_failed", error=str(e))
            return None

        return SharpOdds(
            event_id=f"{base_event_id}::spread",
            book=SharpBook.pinnacle,
            sector=sector,
            outcome_a_label=covering_team,
            outcome_b_label=other_team,
            outcome_a_decimal=cover_dec,
            outcome_b_decimal=other_dec,
            true_prob_a=prob_cover,
            true_prob_b=prob_other,
            spread_line=cover_point,
            margin=margin,
            event_date=event_date,
        )

    def _parse_totals(
        self, market: dict, base_event_id: str, sector: str,
        event_date: Optional[datetime],
    ) -> Optional[SharpOdds]:
        prices = market.get("prices", [])
        over_entry  = next((p for p in prices if p.get("designation") == "over"),  None)
        under_entry = next((p for p in prices if p.get("designation") == "under"), None)
        if not over_entry or not under_entry:
            return None

        # "points" field holds the total line (e.g. 220.5)
        raw_line = over_entry.get("points") or over_entry.get("handicap")
        if raw_line is None:
            return None
        total_line = float(raw_line)

        try:
            over_dec  = american_to_decimal(int(over_entry["price"]))
            under_dec = american_to_decimal(int(under_entry["price"]))
            prob_over, prob_under, margin = devig_two_way(over_dec, under_dec)
        except Exception as e:
            logger.debug("pinnacle_guest_totals_devig_failed", error=str(e))
            return None

        return SharpOdds(
            event_id=f"{base_event_id}::total::{total_line}",
            book=SharpBook.pinnacle,
            sector=sector,
            outcome_a_label="over",
            outcome_b_label="under",
            outcome_a_decimal=over_dec,
            outcome_b_decimal=under_dec,
            true_prob_a=0.0,
            true_prob_b=0.0,
            true_prob_over=prob_over,
            true_prob_under=prob_under,
            total_line=total_line,
            margin=margin,
            event_date=event_date,
        )

    @staticmethod
    def _normalize(name: str, sector: str) -> str:
        from evmax.matching.normalizer import NameNormalizer
        normalized = NameNormalizer(sector).normalize(name)
        return normalized.replace(" ", "_").replace(".", "")


# Keep old name as alias for backward compatibility
EsportsPinnacleClient = PinnacleGuestClient
