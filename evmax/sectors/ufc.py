"""UFC / MMA sector handler.

Sharp source: Pinnacle guest API (sport_id=22, Mixed Martial Arts —
name-matched by league, "UFC"/"Bellator" → ufc; see esports_pinnacle.py).
Prediction market: Kalshi series KXUFCFIGHT (one YES market per fighter,
tennis-style).

Market types: moneyline (fight winner). Draws are extremely rare in MMA and
Kalshi voids/scalar-settles cancelled bouts — no draw outcome is modeled.
No home advantage — neutral venues.

Name normalization mirrors tennis: alias map first (compound / reordered
names), then last-word surname fallback. Hyphenated surnames survive as one
token ("Benoit Saint-Denis" → "saint-denis"), which matches both Kalshi's
title spelling and Pinnacle's.
"""

from evmax.models.market import MarketType, PredictionMarket
from evmax.sectors.base import SectorHandler


class UFCHandler(SectorHandler):
    name = "ufc"
    sharp_source = "pinnacle"

    def normalize_team(self, name: str) -> str:
        """Normalize a fighter name to the canonical surname.

        Tries the alias map first (handles name-order variants like
        "Zhang Weili" vs "Weili Zhang"), then falls back to the last word
        ("Conor McGregor" → "mcgregor", "Ian Machado Garry" → "garry").
        """
        if not name:
            return ""
        cleaned = name.strip().lower()
        if cleaned in self._aliases:
            return self._aliases[cleaned]
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
