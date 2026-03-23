"""TennisModelAgent — surface-aware Elo ratings for tennis match prediction.

Tennis differs from team sports in key ways:
  - Players not teams (no home advantage in the traditional sense)
  - Court surface dramatically affects player performance
  - ATP/WTA ranking serves as a strong prior
  - Serve statistics, recent form, H2H matter more per-matchup

Model design:
  - Maintains separate Elo ratings per player per surface (hard/clay/grass/indoor)
  - Falls back to overall Elo if surface-specific data is sparse
  - Uses ATP ranking as initial Elo prior for unseeded players
  - Surface bonus adjustments applied on top of Elo for known specialists

State file: data/models/tennis_surface_state.json
  {
    "ratings": {
      "hard":    {"djokovic": 1860.0, "alcaraz": 1820.0, ...},
      "clay":    {"nadal": 1910.0, "alcaraz": 1840.0, ...},
      "grass":   {"djokovic": 1830.0, ...},
      "indoor":  {"djokovic": 1870.0, ...},
      "overall": {"djokovic": 1855.0, ...}
    },
    "game_counts": {
      "djokovic": {"hard": 80, "clay": 70, "grass": 50, "indoor": 30, "overall": 230}
    },
    "atp_rankings": {
      "sinner": 1, "alcaraz": 2, "djokovic": 3, ...
    }
  }
"""

from __future__ import annotations

import re
from typing import Optional

from evmax.agents.models.base import ModelAgent, ModelAgentPrediction
from evmax.models.market import PredictionMarket
from evmax.models.odds import SharpOdds

# Surface detection: tournament keywords → surface
SURFACE_KEYWORDS: dict[str, list[str]] = {
    "clay": [
        "french open", "roland garros", "monte carlo", "monte-carlo",
        "madrid", "barcelona", "rome", "roma", "hamburg",
        "munich", "estoril", "marrakech", "bucharest", "geneva",
        "lyon", "houston", "rio", "buenos aires", "santiago",
        "cordoba", "bastad", "umag", "gstaad", "kitzbuhel", "winston-salem",
    ],
    "grass": [
        "wimbledon", "queen's", "queens club", "halle", "s-hertogenbosch",
        "hertogenbosch", "eastbourne", "nottingham", "newport",
        "birmingham",  # WTA grass
    ],
    "indoor": [
        "paris masters", "paris bercy", "nitto atp finals", "atp finals",
        "nitto finals", "wta finals", "rotterdam", "marseille",
        "montpellier", "sofia", "st. petersburg", "st petersburg",
        "cologne", "vienna", "basel", "stockholm", "moscow", "antwerp",
        "metz", "gijon", "nur-sultan",
    ],
}
# Default surface for all others (Australian Open, US Open, Miami, Cincinnati, etc.)
DEFAULT_SURFACE = "hard"

# K-factor for tennis Elo updates
K_FACTOR = 24.0

# ATP ranking → approximate starting Elo
# Based on empirical calibration: rank 1 ≈ 1900, rank 100 ≈ 1550
def ranking_to_elo(rank: Optional[int]) -> float:
    if rank is None:
        return 1500.0
    if rank <= 5:
        return 1900.0 - (rank - 1) * 10.0   # 1900, 1890, 1880, 1870, 1860
    if rank <= 20:
        return 1860.0 - (rank - 5) * 10.0   # 1860..1710
    if rank <= 50:
        return 1710.0 - (rank - 20) * 5.0   # 1710..1560
    if rank <= 100:
        return 1560.0 - (rank - 50) * 1.2   # 1560..1500
    return max(1350.0, 1500.0 - (rank - 100) * 1.0)


DEFAULT_ELO = 1500.0
MIN_SURFACE_GAMES = 8   # need this many on-surface results before trusting surface Elo
MIN_OVERALL_GAMES = 5   # need this many total results before trusting overall Elo


class TennisModelAgent(ModelAgent):
    """
    Surface-aware Elo model for ATP/WTA tennis match prediction.

    Only activates for sector == "tennis". Returns None for all other sectors
    so the EnsembleModelAgent safely ignores it.
    """

    name = "tennis_surface"
    weight = 0.45   # Higher weight in ensemble vs generic elo/form/poisson

    def _ratings(self, surface: str) -> dict[str, float]:
        return self._state.setdefault("ratings", {}).setdefault(surface, {})

    def _counts(self, player: str) -> dict[str, int]:
        return self._state.setdefault("game_counts", {}).setdefault(player, {})

    def _get_rating(self, player: str, surface: str) -> float:
        """Get surface-specific Elo, falling back to overall, then ATP ranking prior."""
        surface_ratings = self._ratings(surface)
        if player in surface_ratings:
            return surface_ratings[player]
        overall = self._ratings("overall")
        if player in overall:
            return overall[player]
        # Fall back to ATP ranking prior
        rank = self._state.get("atp_rankings", {}).get(player)
        return ranking_to_elo(rank)

    def _get_count(self, player: str, surface: str) -> int:
        return self._counts(player).get(surface, 0)

    def _get_overall_count(self, player: str) -> int:
        return self._counts(player).get("overall", 0)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    async def predict_pair(
        self,
        market: PredictionMarket,
        sharp_odds: SharpOdds,
    ) -> Optional[ModelAgentPrediction]:
        sector = (market.sector or "").lower()
        if sector != "tennis":
            return None

        player_a = (sharp_odds.outcome_a_label or market.team_home or "").lower().strip()
        player_b = (sharp_odds.outcome_b_label or market.team_away or "").lower().strip()

        if not player_a or not player_b:
            return None

        # Detect surface from event title
        surface = self._detect_surface(market.title or "")

        elo_a = self._get_rating(player_a, surface)
        elo_b = self._get_rating(player_b, surface)

        prob_a = 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))
        prob_b = 1.0 - prob_a

        # Confidence: how much surface-specific data do we have?
        count_a_surf = self._get_count(player_a, surface)
        count_b_surf = self._get_count(player_b, surface)
        count_a_all = self._get_overall_count(player_a)
        count_b_all = self._get_overall_count(player_b)
        min_surf = min(count_a_surf, count_b_surf)
        min_all = min(count_a_all, count_b_all)

        if min_surf >= MIN_SURFACE_GAMES:
            confidence = min(0.80, 0.55 + 0.005 * min_surf)
        elif min_all >= MIN_OVERALL_GAMES:
            confidence = 0.50   # Have overall data but not surface-specific
        else:
            confidence = 0.30   # Sparse data — mostly relying on ranking prior

        return ModelAgentPrediction(
            event_id=sharp_odds.event_id,
            model_name=self.name,
            true_prob_a=prob_a,
            true_prob_b=prob_b,
            true_prob_draw=None,
            confidence=confidence,
            weight=self.weight,
            sample_size=min_surf,
            notes=(
                f"surface={surface} "
                f"elo_a={elo_a:.0f} elo_b={elo_b:.0f} "
                f"n_surf={min_surf} n_all={min_all}"
            ),
        )

    # ------------------------------------------------------------------
    # Surface detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_surface(event_title: str) -> str:
        """Detect court surface from event/tournament title."""
        title_lower = event_title.lower()
        for surface, keywords in SURFACE_KEYWORDS.items():
            if any(kw in title_lower for kw in keywords):
                return surface
        return DEFAULT_SURFACE

    # ------------------------------------------------------------------
    # Update from result
    # ------------------------------------------------------------------

    def update(
        self,
        team_a: str,
        team_b: str,
        score_a: float,
        score_b: float,
        sector: str,
        event_date: Optional[str] = None,
        surface: str = "overall",
    ) -> None:
        """Update Elo ratings from a completed match result."""
        if sector != "tennis":
            return

        player_a = team_a.lower().strip()
        player_b = team_b.lower().strip()
        surf = surface.lower().strip() if surface else "overall"

        elo_a = self._get_rating(player_a, surf)
        elo_b = self._get_rating(player_b, surf)

        # score_a > 0 means player_a won (sets won: e.g. 2-0, 2-1, 3-0, 3-1, 3-2)
        actual_a = 1.0 if score_a > score_b else 0.0

        expected_a = 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))
        delta = K_FACTOR * (actual_a - expected_a)

        # Update surface-specific rating
        surf_ratings = self._ratings(surf)
        surf_ratings[player_a] = round(elo_a + delta, 2)
        surf_ratings[player_b] = round(elo_b - delta, 2)

        # Also update overall rating (if surface != overall to avoid double update)
        if surf != "overall":
            overall_a = self._get_rating(player_a, "overall")
            overall_b = self._get_rating(player_b, "overall")
            overall_ratings = self._ratings("overall")
            # Smaller K for overall (mixture of surfaces)
            delta_overall = (K_FACTOR * 0.6) * (actual_a - expected_a)
            overall_ratings[player_a] = round(overall_a + delta_overall, 2)
            overall_ratings[player_b] = round(overall_b - delta_overall, 2)

        # Increment game counts
        counts_a = self._counts(player_a)
        counts_b = self._counts(player_b)
        counts_a[surf] = counts_a.get(surf, 0) + 1
        counts_b[surf] = counts_b.get(surf, 0) + 1
        counts_a["overall"] = counts_a.get("overall", 0) + 1
        counts_b["overall"] = counts_b.get("overall", 0) + 1

        self.save_state()
        self.log.debug(
            "tennis_elo_updated",
            player_a=player_a, player_b=player_b,
            surface=surf, delta=round(delta, 2),
            new_elo_a=surf_ratings[player_a],
            new_elo_b=surf_ratings[player_b],
        )

    # ------------------------------------------------------------------
    # Seeding helpers
    # ------------------------------------------------------------------

    def seed_rankings(self, rankings: dict[str, int]) -> None:
        """
        Seed ATP/WTA rankings from an external source.

        Args:
            rankings: {player_name (lowercase) → rank_integer}
                      e.g. {"sinner": 1, "alcaraz": 2, "djokovic": 3}
        """
        atp = self._state.setdefault("atp_rankings", {})
        for player, rank in rankings.items():
            atp[player.lower().strip()] = int(rank)
        self.save_state()
        self.log.info("tennis_rankings_seeded", count=len(rankings))

    def seed_surface_ratings(self, surface: str, ratings: dict[str, float]) -> None:
        """
        Seed surface-specific Elo ratings.

        Args:
            surface: "hard", "clay", "grass", or "indoor"
            ratings: {player_name → elo_rating}
        """
        surface = surface.lower().strip()
        surf_ratings = self._ratings(surface)
        for player, elo in ratings.items():
            surf_ratings[player.lower().strip()] = float(elo)
        self.save_state()
        self.log.info("tennis_surface_ratings_seeded", surface=surface, count=len(ratings))

    def get_rating(self, player: str, surface: str = "overall") -> float:
        return self._get_rating(player.lower().strip(), surface)

    def all_ratings(self, surface: str = "overall") -> dict[str, float]:
        return dict(self._ratings(surface))
