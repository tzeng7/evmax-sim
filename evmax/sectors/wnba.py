"""WNBA sector handler."""

from evmax.models.market import MarketType, PredictionMarket
from evmax.sectors.base import SectorHandler


class WNBAHandler(SectorHandler):
    name = "wnba"
    sharp_source = "pinnacle"

    def enrich_market(self, market: PredictionMarket) -> PredictionMarket:
        """Normalize WNBA team names."""
        updates = {}
        if market.team_home:
            updates["team_home"] = self.normalize_team(market.team_home)
        if market.team_away:
            updates["team_away"] = self.normalize_team(market.team_away)
        return market.model_copy(update=updates) if updates else market

    def market_types_supported(self) -> list[str]:
        return [
            MarketType.moneyline,
            MarketType.spread,
            MarketType.total,
        ]
