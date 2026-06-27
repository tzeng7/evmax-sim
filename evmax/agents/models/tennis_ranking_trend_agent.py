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
# Logit = _RANK_ALPHA * (ln rank_b - ln rank_a)            [absolute ranking]
#       + clamp(_MOMENTUM_K * (Δlog_a - Δlog_b), ±_MAX_LOGIT) [12-wk momentum]
# Momentum is measured in log-rank space so 100→50 counts the same as 4→2;
# the old raw-spots formula treated 100→50 as a 25× bigger move and scored
# coin-flip accuracy (49.8%) on the 2025+2026 walk-forward. Params swept on
# 2024 and frozen before eval — see scripts/experiment_ranking_trend.py
# (eval standalone Brier 0.2530 → 0.2249, accuracy 64.2%).
_RANK_ALPHA = 0.40
_MOMENTUM_K = 0.15
_MAX_LOGIT = 0.40
# Latest usable snapshot must be within this many weeks of the match date,
# otherwise the "trend" is measuring a window that ended long before the
# match (the state was once 18 months stale and the model was pure noise).
_STALE_WEEKS = 6


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
    def _anchored_snapshots(
        history: list[dict], as_of: Optional[date] = None
    ) -> Optional[list[dict]]:
        """Snapshots usable for a match on `as_of`, oldest → newest.

        `as_of` anchors the window at the match date: snapshots after it are
        ignored (no lookahead in backtests), and if the newest usable snapshot
        is more than _STALE_WEEKS old relative to it, the history is unusable
        (an expired window measures nothing) and we return None.
        """
        sorted_hist = sorted(history, key=lambda r: r["date"])
        if as_of is not None:
            sorted_hist = [s for s in sorted_hist
                           if date.fromisoformat(s["date"]) <= as_of]
        if len(sorted_hist) < _MIN_SNAPSHOTS:
            return None
        latest_date = date.fromisoformat(sorted_hist[-1]["date"])
        if as_of is not None and (as_of - latest_date) > timedelta(weeks=_STALE_WEEKS):
            return None
        return sorted_hist

    @staticmethod
    def _log_momentum(snaps: list[dict]) -> float:
        """ln(baseline_rank) - ln(latest_rank) over ~_TREND_WEEKS. Positive =
        improving. Log space: halving your rank counts the same at 100→50
        as at 4→2."""
        from math import log
        latest = snaps[-1]
        cutoff = date.fromisoformat(latest["date"]) - timedelta(weeks=_TREND_WEEKS)
        baseline = snaps[0]
        for snap in snaps:
            if date.fromisoformat(snap["date"]) <= cutoff:
                baseline = snap
            else:
                break
        return log(float(baseline["rank"])) - log(float(latest["rank"]))

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

        # Anchor the trend window at the match date (same pattern as the
        # form agent): live scans default to today, backtests pass the
        # historical date and get a leak-free window.
        ref_date = date.today()
        if market.event_date:
            try:
                ref_date = (market.event_date.date()
                            if hasattr(market.event_date, "date")
                            else date.fromisoformat(str(market.event_date)[:10]))
            except (ValueError, AttributeError):
                pass

        snaps_a = self._anchored_snapshots(hist_a, as_of=ref_date)
        snaps_b = self._anchored_snapshots(hist_b, as_of=ref_date)
        if snaps_a is None or snaps_b is None:
            return None

        from math import exp, log

        # Absolute ranking differential + capped momentum differential.
        rank_a = float(snaps_a[-1]["rank"])
        rank_b = float(snaps_b[-1]["rank"])
        rank_logit = _RANK_ALPHA * (log(rank_b) - log(rank_a))
        mom_diff = self._log_momentum(snaps_a) - self._log_momentum(snaps_b)
        mom_logit = max(-_MAX_LOGIT, min(_MAX_LOGIT, _MOMENTUM_K * mom_diff))
        logit = rank_logit + mom_logit
        prob_a = 1.0 / (1.0 + exp(-logit))
        prob_b = 1.0 - prob_a

        sample = min(len(snaps_a), len(snaps_b))
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
            notes=(f"rank_a={rank_a:.0f} rank_b={rank_b:.0f} "
                   f"mom_diff={mom_diff:+.2f} prob_a={prob_a:.3f}"),
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
        # A score pair carries no ranking — the rank series is seeded in bulk
        # from Tennis Abstract matchmx (winner/loser rank per match) via
        # seed_history(), so resolve-time score updates are a no-op here.
        return

    def seed_history(self, history: dict[str, list[dict]]) -> None:
        """Bulk-load ranking history, fully replacing the existing store.

        This is the authoritative reseed from the matchmx-derived rank series
        (scripts/seed_tennis_models.py): it REPLACES the store rather than
        merging, so players who have aged out of Tennis Abstract's coverage
        don't linger with stale snapshots. Use append_snapshot() for
        incremental single-week additions.

        Args:
            history: {player → [{"date": "YYYY-MM-DD", "rank": int}, ...]}
        """
        self._state["history"] = {
            player.lower().strip(): [
                {"date": s["date"], "rank": int(s["rank"])} for s in snaps
            ]
            for player, snaps in history.items()
        }
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
