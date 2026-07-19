"""TennisAdvancedStatsAgent — logistic model on break points, return %, UE, winners.

Predicts tennis match outcomes using four advanced stat differentials:
  1. Break point conversion rate (bp_won / bp_faced)
  2. Return points won % (return_pts_won / return_pts)
  3. Unforced error rate (unforced / total_pts)
  4. Winners-to-UE ratio (winners / unforced)

Uses a pre-fitted logistic regression (trained on 2023-2024 Sackmann + MCP data,
see scripts/fit_tennis_advanced.py). Falls back to a 2-feature model (BP conv +
RPW only) when a player has no MCP coverage (no winners/UE data).

State file: data/models/tennis_advanced_state.json
  {
    "stats": {
      "sinner j.": {
        "bp_won": 320, "bp_faced": 800,
        "return_pts": 4200, "return_pts_won": 1650,
        "unforced": 280, "winners": 410,
        "total_pts": 9000, "matches": 85
      }
    }
  }
"""

from __future__ import annotations

import math
from typing import Optional

import structlog

from evmax.agents.models.base import ModelAgent, ModelAgentPrediction
from evmax.agents.models.tennis_common import resolve_player
from evmax.models.market import PredictionMarket
from evmax.models.odds import SharpOdds

logger = structlog.get_logger(__name__)

# Logistic coefficients fitted on 2023-2024 ATP+WTA via scripts/fit_tennis_advanced.py.
# Features: [bp_conv_diff, rpw_diff, ue_rate_diff, w_ue_ratio_diff]
_COEFS_FULL = [0.8268, 12.3467, 6.3898, 1.3847]
_INTERCEPT_FULL = 0.0405

# Reduced model for players without MCP data (UE/winners unavailable).
# RPW-only outperforms BP+RPW (58.1% vs 57.8%) — BP adds multicollinearity
# noise when RPW is already present.
_COEFS_REDUCED = [13.2333]
_INTERCEPT_REDUCED = 0.0512

_MIN_MATCHES = 10
_MIN_BP_FACED = 20
_MIN_RETURN_PTS = 100


class TennisAdvancedStatsAgent(ModelAgent):
    """Advanced tennis stats model — logistic on BP conversion, RPW, UE, W/UE."""

    name = "tennis_advanced"
    weight = 0.25

    def _stats_store(self) -> dict[str, dict]:
        return self._state.get("stats", {})

    def _lookup(self, player: str) -> Optional[dict]:
        """Resolve via the shared tennis resolver (2026-07-19).

        The old endswith-surname scan matched the FIRST store key ending in
        the surname — no dehyphenation, no given-name narrowing, and on
        same-surname collisions it silently returned a wrong player's stats.
        ``resolve_player`` handles hyphenated surnames, narrows by given
        name, returns None on genuine ambiguity (wrong-player data is worse
        than no data), and breaks same-person duplicates by match count.
        """
        store = self._stats_store()
        rec = store.get(player)
        if rec:
            return rec
        resolved = resolve_player(
            player, store, weight_fn=lambda k: store[k].get("matches", 0) or 0
        )
        if resolved is not None:
            return store[resolved]
        return None

    @staticmethod
    def _compute_features(rec: dict) -> tuple[list[float], bool]:
        """Returns (feature_vector, has_extended) where has_extended means
        UE/winners data is available for the full 4-feature model."""
        bp_won = rec.get("bp_won", 0)
        bp_faced = rec.get("bp_faced", 0)
        rpts = rec.get("return_pts", 0)
        rpts_won = rec.get("return_pts_won", 0)
        ue = rec.get("unforced", 0)
        winners = rec.get("winners", 0)
        total_pts = rec.get("total_pts", 0)

        if bp_faced < _MIN_BP_FACED or rpts < _MIN_RETURN_PTS:
            return [], False

        bp_conv = bp_won / bp_faced
        rpw = rpts_won / rpts

        has_extended = ue > 0 and total_pts > 0
        if has_extended:
            ue_rate = ue / total_pts
            w_ue = winners / max(ue, 1)
            return [bp_conv, rpw, ue_rate, w_ue], True
        else:
            return [bp_conv, rpw], False

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

        matches_a = rec_a.get("matches", 0)
        matches_b = rec_b.get("matches", 0)
        if matches_a < _MIN_MATCHES or matches_b < _MIN_MATCHES:
            return None

        feats_a, ext_a = self._compute_features(rec_a)
        feats_b, ext_b = self._compute_features(rec_b)
        if not feats_a or not feats_b:
            return None

        use_full = ext_a and ext_b
        if use_full:
            diff = [fa - fb for fa, fb in zip(feats_a, feats_b)]
            coefs = _COEFS_FULL
            intercept = _INTERCEPT_FULL
            model_variant = "full"
        else:
            # RPW-only: index 1 in the feature vector is return points won %
            diff = [feats_a[1] - feats_b[1]]
            coefs = _COEFS_REDUCED
            intercept = _INTERCEPT_REDUCED
            model_variant = "reduced"

        z = sum(c * d for c, d in zip(coefs, diff)) + intercept
        prob_a = 1.0 / (1.0 + math.exp(-z))
        prob_a = max(0.02, min(0.98, prob_a))
        prob_b = 1.0 - prob_a

        min_matches = min(matches_a, matches_b)
        confidence = min(0.80, 0.50 + min_matches / 500.0)

        return ModelAgentPrediction(
            event_id=sharp_odds.event_id,
            model_name=self.name,
            true_prob_a=prob_a,
            true_prob_b=prob_b,
            true_prob_draw=None,
            confidence=confidence,
            weight=self.weight,
            sample_size=min_matches,
            notes=(
                f"variant={model_variant} "
                f"n_a={matches_a} n_b={matches_b} "
                f"bp_conv_a={feats_a[0]:.3f} rpw_a={feats_a[1]:.3f} "
                f"bp_conv_b={feats_b[0]:.3f} rpw_b={feats_b[1]:.3f} "
                f"prob_a={prob_a:.3f}"
            ),
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
        return

    def seed_stats(self, stats: dict[str, dict]) -> None:
        """Bulk-load {player → {bp_won, bp_faced, return_pts, return_pts_won,
        unforced, winners, total_pts, matches}}."""
        self._state["stats"] = stats
        self.save_state()
        logger.info("advanced_stats_seeded", n_players=len(stats))
