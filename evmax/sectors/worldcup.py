"""World Cup (national-team) sector handler.

Structurally identical to club soccer — three-way markets (team A win /
draw / team B win), YES typically maps to a named nation, and TIE markets
carry the draw outcome. It is a SEPARATE sector from `soccer` on purpose:

  - its own Elo namespace (national teams never appear in the club ratings,
    and club Elo is meaningless for them — different rosters, ~10 games/year),
  - its own alias map (FIFA 3-letter codes + national-team name variants),
  - shadow mode + a national-team-only model list.

Kalshi series KXWCGAME encodes teams as 3-letter FIFA codes in the ticker
(e.g. KXWCGAME-26JUN27CODUZB-COD), which the shared ticker parser already
splits; `worldcup.yaml` maps those codes — and the Pinnacle/ESPN full-name
spellings — onto one canonical per nation.
"""

from evmax.models.market import MarketType, PredictionMarket
from evmax.sectors.base import SectorHandler


class WorldCupHandler(SectorHandler):
    name = "worldcup"
    sharp_source = "pinnacle"

    def enrich_market(self, market: PredictionMarket) -> PredictionMarket:
        """Normalize national-team names (FIFA codes or full names) to canonical."""
        updates = {}
        if market.team_home:
            updates["team_home"] = self.normalize_team(market.team_home)
        if market.team_away:
            updates["team_away"] = self.normalize_team(market.team_away)
        return market.model_copy(update=updates) if updates else market

    def market_types_supported(self) -> list[str]:
        # v1 bets the 3-way match winner only. Pinnacle exposes WC spreads
        # (Asian handicap) and totals, but Kalshi WC spread/total series are
        # not wired into SECTOR_SERIES_MAP yet and totals are stale-line prone
        # on a night-before workflow (see the baseball totals notes).
        return [MarketType.moneyline]
