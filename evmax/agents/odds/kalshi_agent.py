"""KalshiOddsAgent — fetches live Kalshi markets for a sector.

Publishes to topic: "odds.kalshi.{sector}"
"""

from __future__ import annotations

from evmax.agents.base import Agent, AgentRequest, AgentResponse
from evmax.clients.kalshi import KalshiClient
from evmax.models.market import MarketType, PredictionMarket


def disabled_market_types(sector: str) -> frozenset[MarketType]:
    """Market types whose effective mode is ``disabled`` for ``sector``.

    Uses ``get_mode(sector, market_type=...)``, so runtime / env overrides win
    exactly as they do at persistence (``--shadow ncaaf`` re-enables every
    type). A sector whose WHOLE mode is ``disabled`` returns nothing: it only
    runs when forced with ``--sectors`` for debugging, and a forced scan should
    see every series. An unknown category (latent sector) disables nothing.
    """
    from evmax.categories import get_category
    from evmax.modes import get_mode

    key = sector.lower()
    try:
        spec = get_category(key)
    except KeyError:
        return frozenset()
    if get_mode(key) == "disabled":
        return frozenset()
    out: set[MarketType] = set()
    for mt in spec.market_types:
        try:
            if get_mode(key, market_type=mt.value) == "disabled":
                out.add(mt)
        except KeyError:
            continue
    return frozenset(out)


class KalshiOddsAgent(Agent):
    """
    Fetches all live Kalshi prediction markets for a given sector.

    Series whose market type is ``disabled`` for the sector are not fetched:
    their rows are dropped before persistence, and on 2026-11-20 they were
    93% of NCAAF's 4,808 markets. A partial fetch (some series failed) returns
    the markets it got with ``status="partial"`` and the failed prefixes in
    ``error``, so the coordinator can record it in ``scan_sector_stats``.

    Published topic payload: list[PredictionMarket]
    """

    name = "kalshi_odds"
    description = (
        "Fetches live Kalshi prediction market prices for a sector, "
        "parses team names and YES prices, publishes to odds.kalshi.{sector}."
    )

    async def run(self, request: AgentRequest) -> AgentResponse:
        sector = request.sector
        skip = disabled_market_types(sector)
        self.log.info(
            "fetching_kalshi", sector=sector,
            skipped_market_types=sorted(mt.value for mt in skip),
        )

        async with KalshiClient() as client:
            markets: list[PredictionMarket] = await client.get_markets(
                sector, skip_market_types=skip or None,
            )
            failed_series = list(client.last_failed_series)
            failed_events = list(client.last_failed_events)

        self.log.info("kalshi_fetched", sector=sector, count=len(markets))

        # Publish so downstream agents (EVGapAgent) can react
        await self.publish(f"odds.kalshi.{sector}", markets, request.correlation_id)

        error = None
        if failed_series or failed_events:
            parts = []
            if failed_series:
                parts.append("markets:" + ",".join(failed_series))
            if failed_events:
                parts.append("events:" + ",".join(failed_events))
            error = "kalshi series failed " + " ".join(parts)
            self.log.warning(
                "kalshi_partial_fetch", sector=sector,
                failed_series=failed_series, failed_events=failed_events,
            )

        return AgentResponse(
            agent_name=self.name,
            sector=sector,
            data=markets,
            status="partial" if error else "ok",
            error=error,
        )
