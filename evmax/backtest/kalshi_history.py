"""Fetch resolved Kalshi markets for backtest Kalshi-vs-Pinnacle EV analysis."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)

# Series to fetch per sector
SECTOR_SERIES = {
    "soccer": [
        "KXEPLGAME", "KXUCLGAME", "KXLALIGAGAME",
        "KXBUNDESLIGAGAME", "KXSERIEAGAME", "KXLIGUE1GAME",
    ],
    "tennis": ["KXATPMATCH", "KXWTAMATCH"],
}


@dataclass
class ResolvedMarket:
    ticker: str
    title: str
    sector: str
    series: str
    result: str           # "yes" or "no"
    last_price: float     # last_price_dollars before resolution (proxy, biased near resolution)
    event_date: Optional[date]
    team_home: Optional[str]
    team_away: Optional[str]
    yes_team: Optional[str]


async def _fetch_series(client, series: str, sector: str) -> list[ResolvedMarket]:
    """Fetch all finalized markets for one series, paginating through all pages."""
    from evmax.clients.kalshi import KalshiClient

    markets: list[ResolvedMarket] = []
    cursor = None
    page = 0

    while True:
        params: dict = {
            "series_ticker": series,
            "status": "finalized",
            "limit": 200,
        }
        if cursor:
            params["cursor"] = cursor

        try:
            resp = await client._get("/markets", params=params)
        except Exception as e:
            logger.warning("kalshi_history_fetch_failed", series=series, error=str(e))
            break

        if not isinstance(resp, dict):
            break

        raw_markets = resp.get("markets", [])
        if not raw_markets:
            break

        for raw in raw_markets:
            ticker = raw.get("ticker", "")
            result = raw.get("result", "")
            if result not in ("yes", "no"):
                continue

            last_price = float(raw.get("last_price_dollars") or 0)
            if last_price <= 0:
                # Fall back: result determines price
                last_price = 1.0 if result == "yes" else 0.0

            title = raw.get("title", "")

            # Parse date from ticker
            from evmax.clients.kalshi import KalshiClient as _KC
            # Use a temporary KalshiClient instance just to call parsing methods
            _kc = _KC.__new__(_KC)
            event_date_dt = _kc._parse_ticker_date(ticker)
            event_date = event_date_dt.date() if event_date_dt else None

            # Parse team names
            if sector == "tennis":
                team_home, team_away = _kc._extract_teams_from_title(title)
                yes_team = _kc._extract_tennis_yes_player(title, sector)
            else:
                team_home, team_away = _kc._extract_teams_from_ticker(ticker)
                if not team_home:
                    team_home, team_away = _kc._extract_teams_from_title(title)
                yes_team = _kc._extract_yes_team(ticker, sector)

            markets.append(ResolvedMarket(
                ticker=ticker,
                title=title,
                sector=sector,
                series=series,
                result=result,
                last_price=last_price,
                event_date=event_date,
                team_home=team_home,
                team_away=team_away,
                yes_team=yes_team,
            ))

        cursor = resp.get("cursor")
        page += 1
        if not cursor or page > 50:  # safety limit
            break

    logger.info("kalshi_history_fetched", series=series, count=len(markets))
    return markets


async def fetch_resolved_markets(sector: str) -> list[ResolvedMarket]:
    """Fetch all finalized Kalshi markets for a sector."""
    from evmax.clients.kalshi import KalshiClient

    series_list = SECTOR_SERIES.get(sector.lower(), [])
    if not series_list:
        logger.warning("no_series_for_sector", sector=sector)
        return []

    all_markets: list[ResolvedMarket] = []
    async with KalshiClient() as client:
        results = await asyncio.gather(
            *[_fetch_series(client, s, sector) for s in series_list],
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, list):
                all_markets.extend(r)
            elif isinstance(r, Exception):
                logger.warning("kalshi_series_error", error=str(r))

    return all_markets
