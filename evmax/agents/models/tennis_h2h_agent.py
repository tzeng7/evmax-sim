"""TennisH2HAgent — head-to-head record nudge.

Tennis matchups carry stylistic effects that don't show up in surface Elo or
serve stats: lefty vs righty, flat ball vs heavy spin, return-position
preferences. Players with a clear H2H edge tend to repeat that edge.

The agent is intentionally a small voice (weight=0.10) — it nudges the
probability away from 0.5 by an amount proportional to the H2H imbalance and
the size of the sample. Below 3 meetings, returns None so the ensemble drops
this model from the blend.

State file: data/models/tennis_h2h_state.json
  {
    "h2h": {
      "alcaraz c.|sinner j.": {"wins_a": 4, "wins_b": 5},   # key sorted alphabetically
      ...
    }
  }
"""

from __future__ import annotations

from typing import Optional

from evmax.agents.models.base import ModelAgent, ModelAgentPrediction
from evmax.agents.models.tennis_common import resolve_player
from evmax.models.market import PredictionMarket
from evmax.models.odds import SharpOdds

_MIN_MEETINGS = 3
_MAX_NUDGE = 0.18    # ±18 percentage points from 0.5 even with a perfect record


def _pair_key(a: str, b: str) -> tuple[str, bool]:
    """Return (sorted_key, swapped) where swapped=True if a came after b alphabetically."""
    if a <= b:
        return f"{a}|{b}", False
    return f"{b}|{a}", True


class TennisH2HAgent(ModelAgent):
    name = "tennis_h2h"
    weight = 0.10

    def _h2h_store(self) -> dict[str, dict]:
        return self._state.setdefault("h2h", {})

    def _resolve_pair(self, player_a: str, player_b: str) -> Optional[tuple[int, int]]:
        """Return (wins_a, wins_b) from the H2H store, accounting for key ordering."""
        store = self._h2h_store()
        # Build a player → canonical-key lookup so we can resolve fuzzy names
        all_players: set[str] = set()
        for k in store:
            left, right = k.split("|", 1)
            all_players.add(left)
            all_players.add(right)
        player_index = {p: p for p in all_players}

        ra = resolve_player(player_a, player_index)
        rb = resolve_player(player_b, player_index)
        if ra is None or rb is None:
            return None

        key, swapped = _pair_key(ra, rb)
        rec = store.get(key)
        if rec is None:
            return None
        wins_a = int(rec.get("wins_a", 0))
        wins_b = int(rec.get("wins_b", 0))
        if swapped:
            wins_a, wins_b = wins_b, wins_a
        return wins_a, wins_b

    async def predict_pair(
        self,
        market: PredictionMarket,
        sharp_odds: SharpOdds,
    ) -> Optional[ModelAgentPrediction]:
        if (market.sector or "").lower() != "tennis":
            return None

        player_a = (sharp_odds.outcome_a_label or market.team_home or "").lower().strip()
        player_b = (sharp_odds.outcome_b_label or market.team_away or "").lower().strip()
        if not player_a or not player_b:
            return None

        rec = self._resolve_pair(player_a, player_b)
        if rec is None:
            return None

        wins_a, wins_b = rec
        total = wins_a + wins_b
        if total < _MIN_MEETINGS:
            return None

        # Symmetric Laplace smoothing prevents 100/0 predictions from a few games
        prob_a_raw = (wins_a + 1.0) / (total + 2.0)
        # Pull toward 0.5 by sample-size factor so small H2H sets stay modest
        shrinkage = total / (total + 4.0)
        prob_a = 0.5 + (prob_a_raw - 0.5) * shrinkage
        prob_a = max(0.5 - _MAX_NUDGE, min(0.5 + _MAX_NUDGE, prob_a))
        prob_b = 1.0 - prob_a

        confidence = min(0.80, 0.50 + 0.04 * total)

        return ModelAgentPrediction(
            event_id=sharp_odds.event_id,
            model_name=self.name,
            true_prob_a=prob_a,
            true_prob_b=prob_b,
            true_prob_draw=None,
            confidence=confidence,
            weight=self.weight,
            sample_size=total,
            notes=f"h2h {wins_a}-{wins_b} prob_a={prob_a:.3f}",
        )

    def update(
        self,
        team_a: str,
        team_b: str,
        score_a: float,
        score_b: float,
        sector: str,
        event_date: Optional[str] = None,
    ) -> None:
        if (sector or "").lower() != "tennis":
            return
        a = team_a.lower().strip()
        b = team_b.lower().strip()
        key, swapped = _pair_key(a, b)
        store = self._h2h_store()
        rec = store.setdefault(key, {"wins_a": 0, "wins_b": 0})
        a_won = score_a > score_b
        if swapped:
            a_won = not a_won
        if a_won:
            rec["wins_a"] = int(rec.get("wins_a", 0)) + 1
        else:
            rec["wins_b"] = int(rec.get("wins_b", 0)) + 1
        self.save_state()

    def seed_h2h(self, records: list[dict]) -> None:
        """Bulk-load H2H records.

        Each record: {"player_a": str, "player_b": str, "wins_a": int, "wins_b": int}
        Re-seeds overwrite existing entries.
        """
        store = self._h2h_store()
        for r in records:
            a = r["player_a"].lower().strip()
            b = r["player_b"].lower().strip()
            key, swapped = _pair_key(a, b)
            wins_a = int(r["wins_a"])
            wins_b = int(r["wins_b"])
            if swapped:
                wins_a, wins_b = wins_b, wins_a
            store[key] = {"wins_a": wins_a, "wins_b": wins_b}
        self.save_state()
        self.log.info("tennis_h2h_seeded", count=len(records))
