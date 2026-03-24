"""NCAAW sector handler (Women's College Basketball)."""

from evmax.models.market import MarketType, PredictionMarket
from evmax.sectors.base import SectorHandler


class NCAAWHandler(SectorHandler):
    name = "ncaaw"
    sharp_source = "pinnacle"

    def enrich_market(self, market: PredictionMarket) -> PredictionMarket:
        """Normalize NCAAW team names (strip mascots, normalize school names)."""
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
            MarketType.series_winner,
        ]
