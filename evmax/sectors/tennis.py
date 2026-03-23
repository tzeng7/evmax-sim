"""Tennis sector handler."""

from evmax.models.market import MarketType, PredictionMarket
from evmax.sectors.base import SectorHandler


class TennisHandler(SectorHandler):
    name = "tennis"
    sharp_source = "pinnacle"

    def normalize_team(self, name: str) -> str:
        """Normalize player name to last name.

        Tries alias first (handles compound surnames like "de minaur").
        Falls back to taking the last word as surname.
        """
        if not name:
            return ""
        cleaned = name.strip().lower()
        # Check full-name alias first
        if cleaned in self._aliases:
            return self._aliases[cleaned]
        # Take the last word as the surname (covers "Alex Michelsen" → "michelsen")
        parts = cleaned.split()
        return parts[-1] if parts else cleaned

    def enrich_market(self, market: PredictionMarket) -> PredictionMarket:
        updates = {}
        if market.team_home:
            updates["team_home"] = self.normalize_team(market.team_home)
        if market.team_away:
            updates["team_away"] = self.normalize_team(market.team_away)
        return market.model_copy(update=updates) if updates else market

    def market_types_supported(self) -> list[str]:
        return [MarketType.moneyline]
