"""Soccer sector handler.

Soccer uses three-way markets (home win / draw / away win).
The prediction market YES typically maps to "home team wins".
"""

from evmax.models.market import MarketType, PredictionMarket
from evmax.sectors.base import SectorHandler


class SoccerHandler(SectorHandler):
    name = "soccer"
    sharp_source = "pinnacle"

    def enrich_market(self, market: PredictionMarket) -> PredictionMarket:
        """Normalize soccer club names."""
        updates = {}
        if market.team_home:
            updates["team_home"] = self.normalize_team(market.team_home)
        if market.team_away:
            updates["team_away"] = self.normalize_team(market.team_away)
        return market.model_copy(update=updates) if updates else market

    def market_types_supported(self) -> list[str]:
        return [
            MarketType.moneyline,  # home/away win (ignoring draw)
            MarketType.total,
            MarketType.series_winner,
        ]
