"""MLB baseball sector handler.

Sharp source: Pinnacle guest API (sport_id=3, league_id=246).
Market types: moneyline, run line (-1.5 spread), total runs (over/under).
No draws — baseball goes to extras.
"""

from evmax.models.market import MarketType, PredictionMarket
from evmax.sectors.base import SectorHandler


class BaseballHandler(SectorHandler):
    name = "baseball"
    sharp_source = "pinnacle"

    def enrich_market(self, market: PredictionMarket) -> PredictionMarket:
        updates = {}
        if market.team_home:
            updates["team_home"] = self.normalize_team(market.team_home)
        if market.team_away:
            updates["team_away"] = self.normalize_team(market.team_away)
        return market.model_copy(update=updates) if updates else market

    def market_types_supported(self) -> list[str]:
        return [
            MarketType.moneyline,
            MarketType.spread,   # run line (-1.5)
            MarketType.total,    # over/under total runs
        ]
