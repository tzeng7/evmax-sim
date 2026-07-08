"""PolymarketUSOddsAgent — fetches live Polymarket US markets for a sector.

Publishes to topic: "odds.polymarket_us.{sector}"

Mirror of KalshiOddsAgent for the second prediction-market venue. Sectors
without a Polymarket US product (see POLYMARKET_US_LEAGUE_MAP) return an
empty list, so it is safe to run for every sector in the scan fan-out.
"""

from __future__ import annotations

from evmax.agents.base import Agent, AgentRequest, AgentResponse
from evmax.clients.polymarket_us import PolymarketUSClient
from evmax.models.market import PredictionMarket


class PolymarketUSOddsAgent(Agent):
    """
    Fetches all live Polymarket US markets for a given sector.

    Published topic payload: list[PredictionMarket]
    """

    name = "polymarket_us_odds"
    description = (
        "Fetches live Polymarket US market prices for a sector via the "
        "gateway leagues/events API, publishes to odds.polymarket_us.{sector}."
    )

    async def run(self, request: AgentRequest) -> AgentResponse:
        sector = request.sector
        self.log.info("fetching_polymarket_us", sector=sector)

        async with PolymarketUSClient() as client:
            markets: list[PredictionMarket] = await client.get_markets(sector)

        self.log.info("polymarket_us_fetched", sector=sector, count=len(markets))

        await self.publish(
            f"odds.polymarket_us.{sector}", markets, request.correlation_id
        )

        return AgentResponse(
            agent_name=self.name,
            sector=sector,
            data=markets,
        )
