"""TennisServeReturnAgent — serve/return point dominance model.

Tennis matches are dominated by service games. A player's serve-points-won
percentage (SPW) is the single strongest tennis predictor independent of Elo.
The differential in SPW between two players maps cleanly to match win
probability via a logistic curve.

v2 changes (recency + surface):
  - State stores per-match entries: {date, surface, spw_won, svpt}
  - At prediction time, computes exponentially-weighted SPW favoring recent
    matches (half-life = 180 days ≈ 6 months)
  - Uses surface-specific SPW when enough on-surface data exists (≥ 200 pts),
    falling back to overall
  - Retains backward compatibility: old flat {spw, n} entries are treated as
    undated overall data with low recency weight

State file: data/models/tennis_serve_return_state.json
  {
    "matches": {
      "sinner j.": [
        {"date": "2024-06-15", "surface": "grass", "won": 48, "svpt": 72},
        {"date": "2024-07-01", "surface": "hard",  "won": 55, "svpt": 80},
        ...
      ]
    },
    "spw": { ... }   # legacy flat format, still readable
  }

Seeded from Sackmann tennis_atp/wta CSVs via scripts/seed_tennis_models.py.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Optional

from evmax.agents.models.base import ModelAgent, ModelAgentPrediction
from evmax.agents.models.tennis_common import resolve_player
from evmax.agents.models.tennis_model_agent import TennisModelAgent
from evmax.models.market import PredictionMarket
from evmax.models.odds import SharpOdds

# Recency half-life in days: matches 6 months ago get weight 0.5
_HALF_LIFE_DAYS = 180.0
_DECAY = math.log(2) / _HALF_LIFE_DAYS

# Minimum service points (weighted) to trust the rate
_MIN_WEIGHTED_PTS = 80
# Minimum surface-specific points to prefer surface SPW over overall
_MIN_SURFACE_PTS = 200

# Slam keywords → best-of-5; everything else best-of-3
_SLAM_KEYWORDS = (
    "australian open", "french open", "roland garros",
    "wimbledon", "us open",
)

# Logistic slopes calibrated so:
#   bo3: 0.04 SPW edge → ~60% match win, 0.10 edge → ~80%
#   bo5: 0.04 SPW edge → ~63% match win, 0.10 edge → ~85%
_K_BO3 = 14.0
_K_BO5 = 18.0


def _match_win_prob(spw_a: float, spw_b: float, best_of: int = 3) -> float:
    """P(A wins match) given each player's serve-points-won rate."""
    diff = spw_a - spw_b
    k = _K_BO3 if best_of == 3 else _K_BO5
    return 1.0 / (1.0 + math.exp(-k * diff))


def _weighted_spw(
    entries: list[dict],
    surface: Optional[str],
    ref_date: date,
) -> Optional[tuple[float, float]]:
    """Compute recency-weighted SPW from match entries.

    Args:
        entries: list of {"date": str, "surface": str, "won": int, "svpt": int}
        surface: target surface (None = overall)
        ref_date: date to measure recency from

    Returns:
        (weighted_spw, weighted_pts) or None if insufficient data.
    """
    total_w_won = 0.0
    total_w_pts = 0.0

    for e in entries:
        if surface and e.get("surface", "").lower() != surface.lower():
            continue
        svpt = e.get("svpt", 0)
        won = e.get("won", 0)
        if svpt <= 0:
            continue

        match_date = e.get("date")
        if match_date:
            try:
                d = date.fromisoformat(match_date)
                days_ago = max(0, (ref_date - d).days)
            except (ValueError, TypeError):
                days_ago = 365  # undated → treat as ~1 year old
        else:
            days_ago = 365

        weight = math.exp(-_DECAY * days_ago)
        total_w_won += won * weight
        total_w_pts += svpt * weight

    if total_w_pts < _MIN_WEIGHTED_PTS:
        return None

    return total_w_won / total_w_pts, total_w_pts


class TennisServeReturnAgent(ModelAgent):
    """Barnett-Clarke serve/return model with recency weighting + surface split."""

    name = "tennis_serve_return"
    weight = 0.15

    def _matches_store(self) -> dict[str, list[dict]]:
        return self._state.setdefault("matches", {})

    def _legacy_store(self) -> dict[str, dict]:
        """Old flat {spw, n} format for backward compatibility."""
        return self._state.get("spw", {})

    def _lookup_entries(self, player: str) -> Optional[list[dict]]:
        """Look up match entries for a player, with surname fallback."""
        store = self._matches_store()
        key = resolve_player(
            player.lower().strip(),
            store,
            weight_fn=lambda k: len(store.get(k, [])),
        )
        if key is not None:
            return store[key]

        # Fall back to legacy flat format → convert to single undated entry
        legacy = self._legacy_store()
        lkey = resolve_player(
            player.lower().strip(),
            legacy,
            weight_fn=lambda k: legacy.get(k, {}).get("n", 0),
        )
        if lkey is not None:
            rec = legacy[lkey]
            spw = rec.get("spw")
            n = int(rec.get("n", 0))
            if spw is not None and n > 0:
                won = int(round(spw * n))
                return [{"date": None, "surface": "overall", "won": won, "svpt": n}]

        return None

    @staticmethod
    def _best_of(title: str) -> int:
        t = (title or "").lower()
        return 5 if any(kw in t for kw in _SLAM_KEYWORDS) else 3

    def _get_spw(
        self,
        player: str,
        surface: Optional[str],
        ref_date: date,
    ) -> Optional[tuple[float, float]]:
        """Get recency-weighted SPW for a player, preferring surface-specific."""
        entries = self._lookup_entries(player)
        if entries is None:
            return None

        # Try surface-specific first
        if surface:
            result = _weighted_spw(entries, surface, ref_date)
            if result and result[1] >= _MIN_SURFACE_PTS:
                return result

        # Fall back to overall (all surfaces)
        return _weighted_spw(entries, None, ref_date)

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

        # Resolve surface from market
        surface, _ = TennisModelAgent._resolve_surface(
            competition=market.competition,
            title=market.title,
        )

        ref_date = date.today()
        if market.event_date:
            try:
                ref_date = market.event_date.date() if hasattr(market.event_date, 'date') else date.fromisoformat(str(market.event_date)[:10])
            except (ValueError, AttributeError):
                pass

        rec_a = self._get_spw(player_a, surface, ref_date)
        rec_b = self._get_spw(player_b, surface, ref_date)
        if rec_a is None or rec_b is None:
            return None

        spw_a, wpts_a = rec_a
        spw_b, wpts_b = rec_b

        best_of = self._best_of(market.title or "")
        prob_a = _match_win_prob(spw_a, spw_b, best_of=best_of)
        prob_a = max(0.02, min(0.98, prob_a))
        prob_b = 1.0 - prob_a

        # Confidence: weighted points → more recent data = more trustworthy
        min_wpts = min(wpts_a, wpts_b)
        confidence = min(0.85, 0.55 + min_wpts / 2000.0)

        # Note whether we used surface-specific or overall
        used_surface = False
        if surface:
            entries_a = self._lookup_entries(player_a)
            if entries_a:
                surf_result = _weighted_spw(entries_a, surface, ref_date)
                used_surface = surf_result is not None and surf_result[1] >= _MIN_SURFACE_PTS

        return ModelAgentPrediction(
            event_id=sharp_odds.event_id,
            model_name=self.name,
            true_prob_a=prob_a,
            true_prob_b=prob_b,
            true_prob_draw=None,
            confidence=confidence,
            weight=self.weight,
            sample_size=int(min_wpts),
            notes=(
                f"spw_a={spw_a:.3f}(w{wpts_a:.0f}) spw_b={spw_b:.3f}(w{wpts_b:.0f}) "
                f"bo{best_of} surface={'yes' if used_surface else 'no'}({surface}) "
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
        # Match scores don't carry serve-point breakdown — seeding from
        # Sackmann is the primary update path.
        return

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def seed_match_serve_stats(
        self,
        match_stats: dict[str, list[dict]],
    ) -> None:
        """Bulk-load per-match serve stats.

        Args:
            match_stats: {player → [{date, surface, won, svpt}, ...]}
        """
        store = self._matches_store()
        for player, entries in match_stats.items():
            key = player.lower().strip()
            store[key] = entries
        self.save_state()
        self.log.info("tennis_serve_match_seeded", count=len(match_stats))

    # Legacy interface for backward compatibility
    def seed_serve_stats(
        self,
        stats: dict[str, tuple[float, int]],
    ) -> None:
        """Bulk-load flat {player → (spw, n_points)}. Legacy — prefer seed_match_serve_stats."""
        spw_store = self._state.setdefault("spw", {})
        for player, (spw, n) in stats.items():
            key = player.lower().strip()
            spw_store[key] = {"spw": float(spw), "n": int(n)}
        self.save_state()
        self.log.info("tennis_serve_seeded_legacy", count=len(stats))

    def all_spw(self) -> dict[str, dict]:
        return dict(self._legacy_store())
