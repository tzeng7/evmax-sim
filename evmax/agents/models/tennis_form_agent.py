"""TennisFormAgent — opponent-adjusted recent form + fatigue signal.

Replaces the ranking-trend agent with a richer model that uses actual recent
match results instead of crude ranking deltas. Two signals:

1. **Opponent-adjusted form** (last 15 matches):
   Each win/loss is weighted by (a) recency (exponential decay, half-life 60 days)
   and (b) opponent quality (beating rank #5 counts more than beating rank #150).
   The quality-weighted win rate is converted to a logistic probability.

2. **Fatigue penalty**:
   Players who have played many matches or long matches in the last 7 days
   get a small probability discount. Empirically, a player coming off a
   5-setter yesterday underperforms their rating by 2-4%.

State file: data/models/tennis_form_state.json
  {
    "matches": {
      "sinner j.": [
        {
          "date": "2026-04-15",
          "won": true,
          "opp_rank": 12,
          "surface": "clay",
          "minutes": 95,
          "tourney_level": "M"   # G=Grand Slam, M=Masters, A=ATP/WTA, D=250
        },
        ...
      ]
    }
  }
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Optional

from evmax.agents.models.base import ModelAgent, ModelAgentPrediction
from evmax.agents.models.tennis_common import resolve_player
from evmax.models.market import PredictionMarket
from evmax.models.odds import SharpOdds

# Form window: consider last N matches
_FORM_WINDOW = 15
# Recency half-life for form weighting (days)
_FORM_HALF_LIFE = 60.0
_FORM_DECAY = math.log(2) / _FORM_HALF_LIFE

# Opponent quality: rank → quality weight
# Beating a top-10 player is worth ~3x beating a rank 100+ player
def _opp_quality(rank: Optional[int]) -> float:
    if rank is None or rank <= 0:
        return 0.5  # unknown opponent
    if rank <= 10:
        return 2.0
    if rank <= 30:
        return 1.5
    if rank <= 50:
        return 1.2
    if rank <= 100:
        return 1.0
    return 0.7

# Logistic slope for form differential
_FORM_K = 3.5

# Fatigue: matches and minutes in last 7 days
_FATIGUE_WINDOW_DAYS = 7
# Each match in last 7 days → small penalty. Extra penalty per 60 min above 120.
_FATIGUE_PER_MATCH = 0.008     # 0.8% per recent match
_FATIGUE_PER_LONG_MIN = 0.0002  # extra per minute above 120 (long matches)
_MAX_FATIGUE_PENALTY = 0.04    # cap at 4% total discount

# Minimum matches to produce a prediction
_MIN_RECENT_MATCHES = 5


class TennisFormAgent(ModelAgent):
    """Opponent-adjusted recent form + fatigue for tennis."""

    name = "tennis_form"
    weight = 0.15

    def _matches_store(self) -> dict[str, list[dict]]:
        return self._state.setdefault("matches", {})

    def _lookup(self, player: str) -> Optional[list[dict]]:
        store = self._matches_store()
        key = resolve_player(
            player.lower().strip(),
            store,
            weight_fn=lambda k: len(store.get(k, [])),
        )
        if key is None:
            return None
        return store[key]

    def _compute_form(
        self,
        matches: list[dict],
        ref_date: date,
        surface: Optional[str] = None,
    ) -> Optional[float]:
        """Compute quality-weighted recent form score in [0, 1].

        Returns None if fewer than _MIN_RECENT_MATCHES are available.
        """
        # Sort by date descending, take last N
        sorted_matches = sorted(matches, key=lambda m: m.get("date", ""), reverse=True)

        # Prefer surface-specific matches but fall back to all if too few
        if surface:
            surface_matches = [m for m in sorted_matches if m.get("surface", "").lower() == surface.lower()]
            if len(surface_matches) >= _MIN_RECENT_MATCHES:
                sorted_matches = surface_matches

        recent = sorted_matches[:_FORM_WINDOW]
        if len(recent) < _MIN_RECENT_MATCHES:
            return None

        weighted_wins = 0.0
        total_weight = 0.0

        for m in recent:
            match_date = m.get("date")
            if match_date:
                try:
                    d = date.fromisoformat(match_date)
                    days_ago = max(0, (ref_date - d).days)
                except (ValueError, TypeError):
                    days_ago = 90
            else:
                days_ago = 90

            recency_w = math.exp(-_FORM_DECAY * days_ago)
            quality_w = _opp_quality(m.get("opp_rank"))
            w = recency_w * quality_w

            if m.get("won"):
                weighted_wins += w
            total_weight += w

        if total_weight < 1e-9:
            return None

        return weighted_wins / total_weight

    def _compute_fatigue(
        self,
        matches: list[dict],
        ref_date: date,
    ) -> float:
        """Compute fatigue penalty in [0, _MAX_FATIGUE_PENALTY].

        Higher = more fatigued = worse expected performance.
        """
        cutoff = ref_date - timedelta(days=_FATIGUE_WINDOW_DAYS)
        recent_matches = []
        for m in matches:
            match_date = m.get("date")
            if not match_date:
                continue
            try:
                d = date.fromisoformat(match_date)
            except (ValueError, TypeError):
                continue
            if d >= cutoff and d < ref_date:
                recent_matches.append(m)

        if not recent_matches:
            return 0.0

        n_matches = len(recent_matches)
        penalty = n_matches * _FATIGUE_PER_MATCH

        # Extra penalty for long matches
        for m in recent_matches:
            minutes = m.get("minutes", 0) or 0
            if minutes > 120:
                penalty += (minutes - 120) * _FATIGUE_PER_LONG_MIN

        return min(penalty, _MAX_FATIGUE_PENALTY)

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

        matches_a = self._lookup(player_a)
        matches_b = self._lookup(player_b)
        if matches_a is None or matches_b is None:
            return None

        # Resolve surface
        from evmax.agents.models.tennis_model_agent import TennisModelAgent
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

        form_a = self._compute_form(matches_a, ref_date, surface)
        form_b = self._compute_form(matches_b, ref_date, surface)
        if form_a is None or form_b is None:
            return None

        # Form differential → probability via logistic
        diff = form_a - form_b
        logit = _FORM_K * diff
        prob_a = 1.0 / (1.0 + math.exp(-logit))

        # Apply fatigue penalties
        fatigue_a = self._compute_fatigue(matches_a, ref_date)
        fatigue_b = self._compute_fatigue(matches_b, ref_date)
        # Net fatigue: if A is more fatigued, reduce A's probability
        net_fatigue = fatigue_a - fatigue_b
        prob_a -= net_fatigue
        prob_a = max(0.05, min(0.95, prob_a))
        prob_b = 1.0 - prob_a

        min_n = min(len(matches_a), len(matches_b))
        confidence = min(0.75, 0.50 + min_n / 200.0)

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
                f"form_a={form_a:.3f} form_b={form_b:.3f} "
                f"fatigue_a={fatigue_a:.3f} fatigue_b={fatigue_b:.3f} "
                f"surface={surface} prob_a={prob_a:.3f}"
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
        # Match results without opponent rank/minutes aren't useful for
        # quality-weighted form. Seeding from Sackmann is the primary path.
        return

    def seed_match_history(self, history: dict[str, list[dict]]) -> None:
        """Bulk-load match history.

        Args:
            history: {player → [{date, won, opp_rank, surface, minutes, tourney_level}, ...]}
        """
        store = self._matches_store()
        for player, matches in history.items():
            key = player.lower().strip()
            store[key] = sorted(matches, key=lambda m: m.get("date", ""))
        self.save_state()
        self.log.info("tennis_form_seeded", count=len(history))
