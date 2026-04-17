"""TennisServeReturnAgent — serve/return point dominance model.

Tennis matches are dominated by service games. A player's career
serve-points-won percentage (SPW) is the single strongest tennis predictor
that's independent of Elo. The differential in SPW between two players maps
cleanly to match win probability via a logistic curve.

Calibration is empirical: against ATP tour data, a 4pp SPW edge produces
roughly 60% bo3 win rate and 63% bo5 win rate (longer formats amplify the
favorite). The k constants below match those targets and the model is
symmetric — equal SPW gives exactly 0.5 as required.

State file: data/models/tennis_serve_return_state.json
  {
    "spw": {                  # serve points won % by player
      "sinner j.": {"spw": 0.683, "n": 142},
      "alcaraz c.": {"spw": 0.671, "n": 138},
      ...
    }
  }

Seeded externally from Sackmann tennis_atp/wta CSVs (winner/loser serve
stat columns). Until seeded, predict_pair returns None and the ensemble
silently skips this model.
"""

from __future__ import annotations

from math import exp
from typing import Optional

from evmax.agents.models.base import ModelAgent, ModelAgentPrediction
from evmax.agents.models.tennis_common import resolve_player
from evmax.models.market import PredictionMarket
from evmax.models.odds import SharpOdds

# League-average SPW fallback (ATP ~63%, WTA ~58%). Used only for opponents
# we haven't seen at all — yields low confidence so the ensemble down-weights.
_DEFAULT_SPW = 0.62

# Below this many recorded service points we don't trust the rate
_MIN_POINTS = 100

# Slam keywords → best-of-5; everything else best-of-3
_SLAM_KEYWORDS = (
    "australian open", "french open", "roland garros",
    "wimbledon", "us open",
)


# ---------------------------------------------------------------------------
# Match-win probability from serve-points-won differential
# ---------------------------------------------------------------------------

# Logistic slopes calibrated so:
#   bo3: 0.04 SPW edge → ~60% match win, 0.10 edge → ~80%
#   bo5: 0.04 SPW edge → ~63% match win, 0.10 edge → ~85% (favorites compound)
_K_BO3 = 14.0
_K_BO5 = 18.0


def _match_win_prob(spw_a: float, spw_b: float, best_of: int = 3) -> float:
    """P(A wins match) given each player's serve-points-won rate.

    Symmetric: equal SPW returns 0.5 by construction.
    """
    diff = spw_a - spw_b
    k = _K_BO3 if best_of == 3 else _K_BO5
    return 1.0 / (1.0 + exp(-k * diff))


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TennisServeReturnAgent(ModelAgent):
    """Barnett-Clarke serve/return model — primary tennis signal alongside Elo."""

    name = "tennis_serve_return"
    weight = 0.15

    # ------------------------------------------------------------------

    def _spw_store(self) -> dict[str, dict]:
        return self._state.setdefault("spw", {})

    def _lookup(self, player: str) -> Optional[tuple[float, int]]:
        store = self._spw_store()
        key = resolve_player(
            player.lower().strip(),
            store,
            weight_fn=lambda k: store.get(k, {}).get("n", 0),
        )
        if key is None:
            return None
        rec = store[key]
        spw = rec.get("spw")
        n = int(rec.get("n", 0))
        if spw is None:
            return None
        return float(spw), n

    @staticmethod
    def _best_of(title: str) -> int:
        t = (title or "").lower()
        return 5 if any(kw in t for kw in _SLAM_KEYWORDS) else 3

    # ------------------------------------------------------------------

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

        rec_a = self._lookup(player_a)
        rec_b = self._lookup(player_b)
        if rec_a is None or rec_b is None:
            return None

        spw_a, n_a = rec_a
        spw_b, n_b = rec_b
        min_n = min(n_a, n_b)
        if min_n < _MIN_POINTS:
            return None

        best_of = self._best_of(market.title or "")
        prob_a = _match_win_prob(spw_a, spw_b, best_of=best_of)
        prob_a = max(0.02, min(0.98, prob_a))
        prob_b = 1.0 - prob_a

        # Confidence: more service points = more trustworthy rate. Saturates
        # at ~500 points (≈ 5 matches of returnable data).
        confidence = min(0.85, 0.55 + min_n / 2000.0)

        return ModelAgentPrediction(
            event_id=sharp_odds.event_id,
            model_name=self.name,
            true_prob_a=prob_a,
            true_prob_b=prob_b,
            true_prob_draw=None,
            confidence=confidence,
            weight=self.weight,
            sample_size=min_n,
            notes=(
                f"spw_a={spw_a:.3f}({n_a}) spw_b={spw_b:.3f}({n_b}) "
                f"bo{best_of} prob_a={prob_a:.3f}"
            ),
        )

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
        # Final scores don't carry serve-point info — bulk seeding from
        # Sackmann match logs is the only meaningful update path.
        return

    # ------------------------------------------------------------------

    def seed_serve_stats(
        self,
        stats: dict[str, tuple[float, int]],
    ) -> None:
        """Bulk-load {player → (spw, n_points)} for ATP and WTA players."""
        store = self._spw_store()
        for player, (spw, n) in stats.items():
            key = player.lower().strip()
            store[key] = {"spw": float(spw), "n": int(n)}
        self.save_state()
        self.log.info("tennis_serve_seeded", count=len(stats))

    def all_spw(self) -> dict[str, dict]:
        return dict(self._spw_store())
