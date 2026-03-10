"""NFL sector handler."""

from evmax.models.market import MarketType, PredictionMarket
from evmax.sectors.base import SectorHandler


class NFLHandler(SectorHandler):
    name = "nfl"
    sharp_source = "pinnacle"

    def enrich_market(self, market: PredictionMarket) -> PredictionMarket:
        """Normalize NFL team names."""
        if market.team_home:
            market = market.model_copy(
                update={"team_home": self.normalize_team(market.team_home)}
            )
        if market.team_away:
            market = market.model_copy(
                update={"team_away": self.normalize_team(market.team_away)}
            )
        return market

    def market_types_supported(self) -> list[str]:
        return [
            MarketType.moneyline,
            MarketType.spread,
            MarketType.total,
            MarketType.series_winner,
        ]
