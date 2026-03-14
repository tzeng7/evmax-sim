"""KalshiOddsAgent — fetches live Kalshi markets for a sector.

Publishes to topic: "odds.kalshi.{sector}"
"""

from __future__ import annotations

from evmax.agents.base import Agent, AgentRequest, AgentResponse
from evmax.clients.kalshi import KalshiClient
from evmax.models.market import PredictionMarket


class KalshiOddsAgent(Agent):
    """
    Fetches all live Kalshi prediction markets for a given sector.

    Published topic payload: list[PredictionMarket]
    """

    name = "kalshi_odds"
    description = (
        "Fetches live Kalshi prediction market prices for a sector, "
        "parses team names and YES prices, publishes to odds.kalshi.{sector}."
    )

    async def run(self, request: AgentRequest) -> AgentResponse:
        sector = request.sector
        self.log.info("fetching_kalshi", sector=sector)

        async with KalshiClient() as client:
            markets: list[PredictionMarket] = await client.get_markets(sector)

        self.log.info("kalshi_fetched", sector=sector, count=len(markets))

        # Publish so downstream agents (EVGapAgent) can react
        await self.publish(f"odds.kalshi.{sector}", markets, request.correlation_id)

        return AgentResponse(
            agent_name=self.name,
            sector=sector,
            data=markets,
        )
