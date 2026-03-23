"""UFC / MMA sector handler.

Sharp source: Pinnacle guest API (sport_id=22, Mixed Martial Arts).
Market types: moneyline (fight winner). No draws (extremely rare in MMA).
No meaningful home advantage — fights are held at neutral venues.
"""

from evmax.models.market import MarketType, PredictionMarket
from evmax.sectors.base import SectorHandler


class UFCHandler(SectorHandler):
    name = "ufc"
    sharp_source = "pinnacle"

    def enrich_market(self, market: PredictionMarket) -> PredictionMarket:
        updates = {}
        if market.team_home:
            updates["team_home"] = self.normalize_team(market.team_home)
        if market.team_away:
            updates["team_away"] = self.normalize_team(market.team_away)
        return market.model_copy(update=updates) if updates else market

    def market_types_supported(self) -> list[str]:
        return [MarketType.moneyline]
