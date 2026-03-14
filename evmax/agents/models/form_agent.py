"""FormModelAgent — recent-form probability model.

Uses each team's last N games to compute a form-adjusted win probability.

Algorithm:
  1. Retrieve last N results for team_a and team_b.
  2. Compute exponentially-weighted win rates:
       w_i = decay^(N-1-i)   (most-recent game weighted highest)
       form_rate = Σ(result_i × w_i) / Σ w_i
  3. Convert to head-to-head probability via log5 formula:
       P(A beats B) = (form_a - form_a×form_b) / (form_a + form_b - 2×form_a×form_b)
  4. Apply home-court/field adjustment (additive on probability).

State file: data/models/form_state.json
  {
    "nba": {
      "lakers": [
        {"date": "2026-02-10", "won": true, "opp": "bulls", "home": true},
        ...
      ],
      ...
    }
  }

Seeding:
  Import recent box-score data as a list of game dicts.
  Use seed_results() to bulk load.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from evmax.agents.models.base import ModelAgent, ModelAgentPrediction
from evmax.models.market import PredictionMarket
from evmax.models.odds import SharpOdds

# Exponential decay factor (per game going back in time)
DECAY: float = 0.85

# Window of recent games to consider
WINDOW: int = 10

# Home-side probability bonus (additive)
HOME_ADJ: dict[str, float] = {
    "nfl": 0.03,
    "nba": 0.04,
    "ncaab": 0.05,
    "soccer": 0.04,
    "lol": 0.0,
    "cs2": 0.0,
}

# Minimum games to produce any prediction
MIN_GAMES: int = 3


@dataclass
class GameRecord:
    date: str    # ISO date string
    won: bool    # did this team win?
    opp: str     # opponent name (normalized)
    home: bool   # was this team the home team?


class FormModelAgent(ModelAgent):
    """
    Recent-form probability model.

    Weights the last WINDOW games with exponential decay so the most
    recent result counts more than a game from 10 matches ago.
    """

    name = "form"
    weight = 0.25   # blending weight in ensemble

    def _team_records(self, sector: str, team: str) -> list[GameRecord]:
        team_data = self._state.get(sector, {}).get(team, [])
        return [GameRecord(**r) for r in team_data]

    def _form_rate(self, records: list[GameRecord]) -> float:
        """Exponentially-weighted win rate from most recent WINDOW games."""
        recent = sorted(records, key=lambda r: r.date, reverse=True)[:WINDOW]
        if not recent:
            return 0.5  # unknown → assume 50/50

        total_weight = 0.0
        weighted_wins = 0.0
        for i, rec in enumerate(recent):
            w = DECAY ** i  # most recent = decay^0 = 1.0
            weighted_wins += w * (1.0 if rec.won else 0.0)
            total_weight += w

        return weighted_wins / total_weight if total_weight > 0 else 0.5

    @staticmethod
    def _log5(p_a: float, p_b: float) -> float:
        """
        Bill James Log5 formula: head-to-head win probability for team A.

        P(A beats B) = (p_a - p_a*p_b) / (p_a + p_b - 2*p_a*p_b)
        """
        denom = p_a + p_b - 2.0 * p_a * p_b
        if denom < 1e-9:
            return 0.5
        return (p_a - p_a * p_b) / denom

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

        recs_a = self._team_records(sector, team_a)
        recs_b = self._team_records(sector, team_b)

        if len(recs_a) < MIN_GAMES or len(recs_b) < MIN_GAMES:
            return None   # insufficient data — let ensemble skip this model

        form_a = self._form_rate(recs_a)
        form_b = self._form_rate(recs_b)

        prob_a = self._log5(form_a, form_b)
        home_adj = HOME_ADJ.get(sector, 0.03)
        prob_a = min(0.95, max(0.05, prob_a + home_adj))  # team_a is home
        prob_b = 1.0 - prob_a
        prob_draw: Optional[float] = None

        if sector == "soccer":
            # Same draw allocation logic as Elo model
            closeness = 1.0 - abs(prob_a - 0.5) * 2.0
            draw_base = 0.22
            prob_draw = draw_base * (0.5 + 0.5 * closeness)
            scale = (1.0 - prob_draw)
            prob_a = prob_a * scale
            prob_b = prob_b * scale

        sample = min(len(recs_a), len(recs_b))
        confidence = 0.5 + 0.3 * min(1.0, (sample - MIN_GAMES) / (WINDOW - MIN_GAMES))

        return ModelAgentPrediction(
            event_id=sharp_odds.event_id,
            model_name=self.name,
            true_prob_a=prob_a,
            true_prob_b=prob_b,
            true_prob_draw=prob_draw,
            confidence=confidence,
            weight=self.weight,
            sample_size=sample,
            notes=(
                f"form_a={form_a:.3f} form_b={form_b:.3f} "
                f"log5={prob_a:.3f} n={sample}"
            ),
        )

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
        date_str = event_date or datetime.utcnow().date().isoformat()
        sector = sector.lower()
        team_a = team_a.lower().strip()
        team_b = team_b.lower().strip()

        a_won = score_a > score_b
        b_won = score_b > score_a

        if sector not in self._state:
            self._state[sector] = {}

        def _add_record(team: str, won: bool, opp: str, home: bool) -> None:
            if team not in self._state[sector]:
                self._state[sector][team] = []
            self._state[sector][team].append(
                {"date": date_str, "won": won, "opp": opp, "home": home}
            )
            # Keep only the most recent 2×WINDOW entries to control file size
            self._state[sector][team] = sorted(
                self._state[sector][team], key=lambda r: r["date"], reverse=True
            )[: WINDOW * 2]

        _add_record(team_a, a_won, team_b, home=True)
        _add_record(team_b, b_won, team_a, home=False)

        self.log.debug("form_updated", team_a=team_a, team_b=team_b, a_won=a_won)

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def seed_results(self, sector: str, results: list[dict]) -> None:
        """
        Bulk-load historical results.

        Each result dict:
          {
            "date": "2026-01-15",
            "home": "lakers",
            "away": "celtics",
            "score_home": 112,
            "score_away": 108,
          }
        """
        for r in results:
            self.update(
                team_a=r["home"],
                team_b=r["away"],
                score_a=r["score_home"],
                score_b=r["score_away"],
                sector=sector,
                event_date=r.get("date"),
            )
        self.save_state()
        self.log.info("form_seeded", sector=sector, records=len(results))
