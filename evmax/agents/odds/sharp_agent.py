"""SharpOddsAgent — fetches devigged Pinnacle odds for a sector.

Uses Pinnacle's public guest API (no credentials required) for all sectors.
"""

from __future__ import annotations

from evmax.agents.base import Agent, AgentRequest, AgentResponse
from evmax.clients.esports_pinnacle import PinnacleGuestClient
from evmax.models.odds import SharpOdds


class SharpOddsAgent(Agent):
    """
    Fetches devigged Pinnacle odds (moneylines + spreads) for a sector
    via the Pinnacle guest Arcadia API.

    Published topic payload: list[SharpOdds]
    """

    name = "sharp_odds"
    description = (
        "Fetches Pinnacle lines via guest API, power-method deviggs them, "
        "publishes devigged SharpOdds to odds.sharp.{sector}."
    )

    async def run(self, request: AgentRequest) -> AgentResponse:
        sector = request.sector
        self.log.info("fetching_pinnacle", sector=sector)

        async with PinnacleGuestClient() as client:
            sharp_odds: list[SharpOdds] = await client.get_odds(sector)

        await self.publish(f"odds.sharp.{sector}", sharp_odds, request.correlation_id)

        return AgentResponse(
            agent_name=self.name,
            sector=sector,
            data=sharp_odds,
        )
