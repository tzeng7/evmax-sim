"""CS2 sector handler.

Sharp source: Pinnacle guest API (via PinnacleGuestClient).
Market types include match winner, map handicap, series winner.
"""

from evmax.models.market import MarketType, PredictionMarket
from evmax.sectors.base import SectorHandler


class CS2Handler(SectorHandler):
    name = "cs2"
    sharp_source = "pinnacle"
    # Kalshi prints team names with accents and dots ("Movistar KOI Fénix",
    # "Gen.G", "Virtus.pro"); Pinnacle drops the accents and its event keys
    # drop the dots. Esports is sharp-only (no model state keyed on names),
    # so folding both costs nothing and lets the two venues' keys agree.
    fold_accents = True
    strip_dots = True

    def enrich_market(self, market: PredictionMarket) -> PredictionMarket:
        """Normalize CS2 team names."""
        updates = {}
        if market.team_home:
            updates["team_home"] = self.normalize_team(market.team_home)
        if market.team_away:
            updates["team_away"] = self.normalize_team(market.team_away)
        return market.model_copy(update=updates) if updates else market

    def market_types_supported(self) -> list[str]:
        return [
            MarketType.moneyline,
            MarketType.map_handicap,
            MarketType.series_winner,
        ]
