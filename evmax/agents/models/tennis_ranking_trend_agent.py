"""TennisRankingTrendAgent — rolling ATP/WTA ranking momentum.

Players whose rank is *improving* over the last ~12 weeks systematically
beat their static rating; players whose rank is *deteriorating* underperform.
The signal is independent of Elo and serve stats — it captures form not yet
priced into the long-running models.

State file: data/models/tennis_ranking_trend_state.json
  {
    "history": {
      "alcaraz c.": [
        {"date": "2026-01-06", "rank": 2},
        {"date": "2026-02-03", "rank": 1},
        ...
      ],
      ...
    }
  }

Each entry is one weekly snapshot. The agent computes a 12-week rank delta
per player, converts it to a small log-odds shift, and combines the two
players' momenta into a probability.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

from evmax.agents.models.base import ModelAgent, ModelAgentPrediction
from evmax.agents.models.tennis_common import resolve_player
from evmax.models.market import PredictionMarket
from evmax.models.odds import SharpOdds

_TREND_WEEKS = 12
_MIN_SNAPSHOTS = 4   # need at least this many weekly snapshots per player
# Each rank-spot improvement adds this much log-odds; capped to keep the
# nudge modest. Calibrated so a 50-spot jump (clear breakout) yields ~+8pp.
_LOGIT_PER_SPOT = 0.007
_MAX_LOGIT = 0.40


class TennisRankingTrendAgent(ModelAgent):
    name = "tennis_ranking_trend"
    weight = 0.10

    def _history_store(self) -> dict[str, list[dict]]:
        return self._state.setdefault("history", {})

    def _resolve(self, player: str) -> Optional[list[dict]]:
        store = self._history_store()
        key = resolve_player(player, store, weight_fn=lambda k: len(store.get(k, [])))
        if key is None:
            return None
        return store[key]

    @staticmethod
    def _trend_delta(history: list[dict]) -> Optional[float]:
        """Return (older_rank - recent_rank) over ~12 weeks. Positive = improving."""
        if len(history) < _MIN_SNAPSHOTS:
            return None
        sorted_hist = sorted(history, key=lambda r: r["date"])
        latest = sorted_hist[-1]
        cutoff = date.fromisoformat(latest["date"]) - timedelta(weeks=_TREND_WEEKS)
        # Find the snapshot closest to (but not after) the cutoff
        baseline = sorted_hist[0]
        for snap in sorted_hist:
            if date.fromisoformat(snap["date"]) <= cutoff:
                baseline = snap
            else:
                break
        return float(baseline["rank"]) - float(latest["rank"])

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

        hist_a = self._resolve(player_a)
        hist_b = self._resolve(player_b)
        if hist_a is None or hist_b is None:
            return None

        delta_a = self._trend_delta(hist_a)
        delta_b = self._trend_delta(hist_b)
        if delta_a is None or delta_b is None:
            return None

        # Differential momentum. A positive value means A is rising faster.
        diff = delta_a - delta_b
        logit = max(-_MAX_LOGIT, min(_MAX_LOGIT, diff * _LOGIT_PER_SPOT))
        # Convert log-odds shift to absolute prob centered at 0.5
        from math import exp
        prob_a = 1.0 / (1.0 + exp(-logit))
        prob_b = 1.0 - prob_a

        sample = min(len(hist_a), len(hist_b))
        confidence = min(0.75, 0.48 + 0.02 * sample)

        return ModelAgentPrediction(
            event_id=sharp_odds.event_id,
            model_name=self.name,
            true_prob_a=prob_a,
            true_prob_b=prob_b,
            true_prob_draw=None,
            confidence=confidence,
            weight=self.weight,
            sample_size=sample,
            notes=f"trend_a={delta_a:+.0f} trend_b={delta_b:+.0f} prob_a={prob_a:.3f}",
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
        # Ranking history isn't derivable from match results — seed from
        # ATP/WTA weekly ranking releases via seed_history().
        return

    def seed_history(self, history: dict[str, list[dict]]) -> None:
        """Bulk-load ranking history.

        Args:
            history: {player → [{"date": "YYYY-MM-DD", "rank": int}, ...]}
        """
        store = self._history_store()
        for player, snaps in history.items():
            key = player.lower().strip()
            store[key] = [
                {"date": s["date"], "rank": int(s["rank"])} for s in snaps
            ]
        self.save_state()
        self.log.info("tennis_ranking_history_seeded", count=len(history))

    def append_snapshot(self, player: str, snap_date: str, rank: int) -> None:
        """Append one weekly ranking snapshot for a player. Idempotent on date."""
        store = self._history_store()
        key = player.lower().strip()
        snaps = store.setdefault(key, [])
        if any(s["date"] == snap_date for s in snaps):
            return
        snaps.append({"date": snap_date, "rank": int(rank)})
        snaps.sort(key=lambda s: s["date"])
        self.save_state()
