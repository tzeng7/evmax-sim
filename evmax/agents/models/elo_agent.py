"""EloModelAgent — Elo rating system for all sectors.

Elo is a well-validated method for estimating relative team strength from
head-to-head results.  Each team starts at 1500.  After each game:
  - The winner gains K × (1 - expected_win_prob) Elo points
  - The loser loses the same amount

Win probability formula:
  P(A beats B) = 1 / (1 + 10^((Elo_B - Elo_A) / 400))

Home advantage:
  Added as a fixed bonus to the home team's effective Elo before computing
  win probability.  Values are sector-specific.

K-factor (controls how fast ratings update after each result):
  NBA: 20  |  NFL: 25  |  Soccer: 30  |  NCAAB: 20  |  LoL: 20  |  CS2: 20

State file: data/models/elo_state.json
  {
    "nba": {"lakers": 1547.3, "celtics": 1601.2, ...},
    "soccer": {"manchester city": 1632.1, ...},
    ...
  }

Seeding:
  Pre-populate the state file with known ratings from any external source.
  The agent will load them automatically on startup and refine from there.
"""

from __future__ import annotations

import math
from typing import Optional

from evmax.agents.models.base import ModelAgent, ModelAgentPrediction
from evmax.models.market import PredictionMarket
from evmax.models.odds import SharpOdds

# K-factor per sector
K_FACTORS: dict[str, float] = {
    "nfl": 25.0,
    "nba": 20.0,
    "ncaab": 20.0,
    "soccer": 30.0,
    "baseball": 6.0,   # 162-game season → smaller K to avoid overreaction per game
    "ufc": 32.0,       # fighters have few bouts/year → larger K for faster updates
    "f1": 16.0,        # ~24 races/season, pairwise head-to-head updates per race
    "lol": 20.0,
    "cs2": 20.0,
}

# Home Elo advantage in Elo points (added to home team effective rating)
HOME_ADVANTAGE_ELO: dict[str, float] = {
    "nfl": 48.0,      # ~3 pts / ~55% win rate
    "nba": 100.0,     # ~6 pts / ~60% win rate
    "ncaab": 70.0,
    "soccer": 60.0,
    "baseball": 32.0, # ~54% home win rate historically
    "ufc": 0.0,       # neutral venue
    "f1": 0.0,        # different circuit each race
    "lol": 0.0,
    "cs2": 0.0,
}

DEFAULT_ELO = 1500.0
# Confidence caps based on number of games seen for a team
LOW_DATA_THRESHOLD = 5    # fewer games → low confidence
MED_DATA_THRESHOLD = 15   # moderate confidence above this


class EloModelAgent(ModelAgent):
    """
    Predicts win probability via Elo ratings.

    State structure:
      _state = {
        "{sector}": {
          "ratings": {"{team}": float, ...},
          "game_counts": {"{team}": int, ...},   # # of games used to build rating
        },
        ...
      }
    """

    name = "elo"
    weight = 0.35   # weight in EnsembleModelAgent blend

    def _sector_state(self, sector: str) -> dict:
        if sector not in self._state:
            self._state[sector] = {"ratings": {}, "game_counts": {}}
        return self._state[sector]

    def _get_rating(self, sector: str, team: str) -> float:
        return self._sector_state(sector)["ratings"].get(team, DEFAULT_ELO)

    def _get_count(self, sector: str, team: str) -> int:
        return self._sector_state(sector)["game_counts"].get(team, 0)

    def _set_rating(self, sector: str, team: str, rating: float) -> None:
        self._sector_state(sector)["ratings"][team] = round(rating, 2)

    def _increment_count(self, sector: str, team: str) -> None:
        gc = self._sector_state(sector)["game_counts"]
        gc[team] = gc.get(team, 0) + 1

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    async def predict_pair(
        self,
        market: PredictionMarket,
        sharp_odds: SharpOdds,
    ) -> Optional[ModelAgentPrediction]:
        sector = (market.sector or "").lower()
        team_a = (sharp_odds.outcome_a_label or market.team_home or "").lower().strip()
        team_b = (sharp_odds.outcome_b_label or market.team_away or "").lower().strip()

        if not team_a or not team_b:
            return None

        prob_a, prob_b, prob_draw = self._win_probs(sector, team_a, team_b)

        count_a = self._get_count(sector, team_a)
        count_b = self._get_count(sector, team_b)
        min_count = min(count_a, count_b)

        # Confidence scales with data availability
        if min_count == 0:
            confidence = 0.3   # brand new teams — default Elo, low trust
        elif min_count < LOW_DATA_THRESHOLD:
            confidence = 0.45
        elif min_count < MED_DATA_THRESHOLD:
            confidence = 0.60
        else:
            confidence = 0.80

        return ModelAgentPrediction(
            event_id=sharp_odds.event_id,
            model_name=self.name,
            true_prob_a=prob_a,
            true_prob_b=prob_b,
            true_prob_draw=prob_draw,
            confidence=confidence,
            weight=self.weight,
            sample_size=min_count,
            notes=(
                f"Elo_a={self._get_rating(sector, team_a):.0f} "
                f"Elo_b={self._get_rating(sector, team_b):.0f} "
                f"n={min_count}"
            ),
        )

    def _win_probs(
        self, sector: str, team_a: str, team_b: str
    ) -> tuple[float, float, Optional[float]]:
        """
        Compute head-to-head win probabilities with home advantage.
        team_a is treated as home (outcome_a_label from Pinnacle = home team).
        """
        elo_a = self._get_rating(sector, team_a)
        elo_b = self._get_rating(sector, team_b)
        home_bonus = HOME_ADVANTAGE_ELO.get(sector, 0.0)

        expected_a = 1.0 / (1.0 + 10.0 ** ((elo_b - (elo_a + home_bonus)) / 400.0))
        expected_b = 1.0 - expected_a

        # Soccer: allocate draw probability from the expected margin
        if sector == "soccer":
            # Approx: draw likelihood is highest when teams are balanced.
            # Use Bradley-Terry-inspired draw fraction: flat 25% adjusted toward 50/50.
            closeness = 1.0 - abs(expected_a - 0.5) * 2.0  # 1=even, 0=one-sided
            draw_base = 0.22
            draw_prob = draw_base * (0.5 + 0.5 * closeness)
            # Rescale win probs to leave room for draw
            scale = (1.0 - draw_prob) / (expected_a + expected_b)
            return expected_a * scale, expected_b * scale, draw_prob

        return expected_a, expected_b, None

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
    ) -> None:
        """Update Elo ratings from a completed game result."""
        team_a = team_a.lower().strip()
        team_b = team_b.lower().strip()

        elo_a = self._get_rating(sector, team_a)
        elo_b = self._get_rating(sector, team_b)
        home_bonus = HOME_ADVANTAGE_ELO.get(sector, 0.0)
        k = K_FACTORS.get(sector, 20.0)

        # Actual score: 1=A wins, 0=B wins, 0.5=draw
        if score_a > score_b:
            actual_a = 1.0
        elif score_b > score_a:
            actual_a = 0.0
        else:
            actual_a = 0.5

        expected_a = 1.0 / (1.0 + 10.0 ** ((elo_b - (elo_a + home_bonus)) / 400.0))

        delta = k * (actual_a - expected_a)
        self._set_rating(sector, team_a, elo_a + delta)
        self._set_rating(sector, team_b, elo_b - delta)
        self._increment_count(sector, team_a)
        self._increment_count(sector, team_b)

        self.log.debug(
            "elo_updated",
            team_a=team_a,
            team_b=team_b,
            delta=round(delta, 2),
            new_elo_a=self._get_rating(sector, team_a),
            new_elo_b=self._get_rating(sector, team_b),
        )

    # ------------------------------------------------------------------
    # Utility: seed ratings from external dict
    # ------------------------------------------------------------------

    def seed_ratings(self, sector: str, ratings: dict[str, float]) -> None:
        """
        Bulk-load Elo ratings from an external source (e.g. 538, espn, manual).

        Args:
            sector: Sport sector key.
            ratings: {team_name (lowercase) → elo_rating}
        """
        state = self._sector_state(sector)
        for team, rating in ratings.items():
            state["ratings"][team.lower().strip()] = float(rating)
        self.save_state()
        self.log.info("elo_seeded", sector=sector, teams=len(ratings))

    def get_rating(self, sector: str, team: str) -> float:
        return self._get_rating(sector, team.lower().strip())

    def all_ratings(self, sector: str) -> dict[str, float]:
        return dict(self._sector_state(sector).get("ratings", {}))
