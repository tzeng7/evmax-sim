"""Formula 1 sector handler.

Sharp source: Pinnacle guest API (sport_id=44, Formula 1).
Pinnacle lists F1 as head-to-head driver matchups (who finishes higher in the race).
Market types: moneyline (driver A finishes ahead of driver B).
No home advantage — races held at different circuits worldwide.
"""

from evmax.models.market import MarketType, PredictionMarket
from evmax.sectors.base import SectorHandler


class F1Handler(SectorHandler):
    name = "f1"
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
