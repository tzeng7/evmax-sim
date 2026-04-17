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

Recency weighting:
  When event_date is provided during seeding, the K-factor scales up for
  more recent games. Games in the last 14 days get up to 2× the base K,
  while older games use the base K. This lets the model react faster to
  recent form changes (e.g. key players returning from injury).

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

import json
import math
from datetime import date, datetime
from pathlib import Path
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
    "wnba": 20.0,     # 40-game season, fewer teams → each game informative
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
    "wnba": 60.0,    # ~54% home win rate (weaker than NBA's 60%)
    "ufc": 0.0,       # neutral venue
    "f1": 0.0,        # different circuit each race
    "lol": 0.0,
    "cs2": 0.0,
}

DEFAULT_ELO = 1500.0
# Confidence caps based on number of games seen for a team
LOW_DATA_THRESHOLD = 5    # fewer games → low confidence
MED_DATA_THRESHOLD = 15   # moderate confidence above this

# Head-to-head adjustment: max probability shift from H2H record
H2H_MAX_ADJ = 0.05       # cap at ±5% probability shift
H2H_MIN_GAMES = 2        # need at least 2 H2H meetings to apply

# Rest-day adjustment in Elo points (applied at prediction time, not stored)
# Back-to-back is the biggest factor — worth ~3 pts in NBA
REST_ELO_ADJ: dict[str, dict[int, float]] = {
    "nba": {0: -75.0, 1: 0.0, 2: 15.0, 3: 25.0},   # 0=B2B, 1=normal, 2+=extra rest
    "nfl": {0: -30.0, 1: 0.0, 2: 10.0, 3: 10.0},    # NFL rarely has B2B but short weeks matter
    "ncaab": {0: -50.0, 1: 0.0, 2: 10.0, 3: 15.0},
    "ncaaw": {0: -50.0, 1: 0.0, 2: 10.0, 3: 15.0},
    "wnba": {0: -50.0, 1: 0.0, 2: 10.0, 3: 15.0},
    "soccer": {0: -40.0, 1: -15.0, 2: 0.0, 3: 10.0},  # soccer baseline is 2 days
}
FORM_STATE_PATH = Path(__file__).resolve().parents[3] / "data" / "models" / "form_state.json"


class EloModelAgent(ModelAgent):
    """
    Predicts win probability via Elo ratings.

    State structure:
      _state = {
        "{sector}": {
          "ratings": {"{team}": float, ...},
          "game_counts": {"{team}": int, ...},
          "h2h": {"{team_a}::{team_b}": {"a_wins": int, "b_wins": int, "games": int}, ...},
        },
        ...
      }
    """

    name = "elo"
    weight = 0.35   # weight in EnsembleModelAgent blend

    def _sector_state(self, sector: str) -> dict:
        if sector not in self._state:
            self._state[sector] = {"ratings": {}, "game_counts": {}, "h2h": {}}
        state = self._state[sector]
        if "h2h" not in state:
            state["h2h"] = {}
        return state

    def _resolve_team(self, sector: str, team: str, store: dict) -> str:
        """Resolve team name via normalizer + last-word/prefix fallbacks.

        Routes through NameNormalizer first so sector aliases (e.g.
        "Karmine Corp" → "kc", "LGD Gaming" → "lgd") resolve against stored
        state keys. Falls through to legacy last-word / prefix matching.
        """
        if team in store:
            return team
        # Normalizer-driven resolution: strip noise words + apply alias map
        try:
            from evmax.matching.normalizer import NameNormalizer
            normed = NameNormalizer(sector).normalize(team)
            if normed and normed != team and normed in store:
                return normed
        except Exception:
            pass
        if " " in team:
            last = team.rsplit(" ", 1)[-1]
            if last in store:
                return last
        for key in store:
            if (team.startswith(key + " ") or key.startswith(team + " ")
                    or team.endswith(key) or key.endswith(team)):
                return key
        return team  # not found — will use default

    def _get_rating(self, sector: str, team: str) -> float:
        ratings = self._sector_state(sector)["ratings"]
        team = self._resolve_team(sector, team, ratings)
        return ratings.get(team, DEFAULT_ELO)

    def _get_count(self, sector: str, team: str) -> int:
        counts = self._sector_state(sector)["game_counts"]
        team = self._resolve_team(sector, team, counts)
        return counts.get(team, 0)

    def _set_rating(self, sector: str, team: str, rating: float) -> None:
        self._sector_state(sector)["ratings"][team] = round(rating, 2)

    # ------------------------------------------------------------------
    # Head-to-head tracking
    # ------------------------------------------------------------------

    @staticmethod
    def _h2h_key(team_a: str, team_b: str) -> tuple[str, str, bool]:
        """Return canonical H2H key (alphabetical order) and whether teams were swapped."""
        if team_a <= team_b:
            return f"{team_a}::{team_b}", team_a, False
        return f"{team_b}::{team_a}", team_b, True

    def _record_h2h(self, sector: str, team_a: str, team_b: str, a_won: bool) -> None:
        """Record a head-to-head result."""
        h2h = self._sector_state(sector)["h2h"]
        key, first_team, swapped = self._h2h_key(team_a, team_b)
        if key not in h2h:
            h2h[key] = {"a_wins": 0, "b_wins": 0, "games": 0}
        rec = h2h[key]
        rec["games"] += 1
        # "a" in the record = the alphabetically first team
        if swapped:
            a_won = not a_won
        if a_won:
            rec["a_wins"] += 1
        else:
            rec["b_wins"] += 1

    def _h2h_adjustment(self, sector: str, team_a: str, team_b: str) -> float:
        """Return probability adjustment for team_a based on H2H record.

        Positive = team_a has H2H edge, negative = team_b has edge.
        Capped at ±H2H_MAX_ADJ.
        """
        h2h = self._sector_state(sector).get("h2h", {})
        key, _, swapped = self._h2h_key(team_a, team_b)
        rec = h2h.get(key)
        if not rec or rec["games"] < H2H_MIN_GAMES:
            return 0.0

        # Win rate for alphabetically-first team
        first_wr = rec["a_wins"] / rec["games"]
        # Convert to team_a's perspective
        a_wr = (1.0 - first_wr) if swapped else first_wr

        # Adjustment: deviation from 50% scaled by confidence (more games = more weight)
        confidence = min(1.0, rec["games"] / 6.0)  # full confidence at 6+ meetings
        raw_adj = (a_wr - 0.5) * confidence * 2 * H2H_MAX_ADJ
        return max(-H2H_MAX_ADJ, min(H2H_MAX_ADJ, raw_adj))

    def get_h2h(self, sector: str, team_a: str, team_b: str) -> Optional[dict]:
        """Get H2H record between two teams. Returns None if no meetings."""
        h2h = self._sector_state(sector).get("h2h", {})
        key, _, swapped = self._h2h_key(team_a.lower().strip(), team_b.lower().strip())
        rec = h2h.get(key)
        if not rec:
            return None
        if swapped:
            return {"team_a_wins": rec["b_wins"], "team_b_wins": rec["a_wins"], "games": rec["games"]}
        return {"team_a_wins": rec["a_wins"], "team_b_wins": rec["b_wins"], "games": rec["games"]}

    def _increment_count(self, sector: str, team: str) -> None:
        gc = self._sector_state(sector)["game_counts"]
        gc[team] = gc.get(team, 0) + 1

    # ------------------------------------------------------------------
    # Rest-day adjustment
    # ------------------------------------------------------------------

    def _days_of_rest(self, sector: str, team: str) -> Optional[int]:
        """Get days since last game for a team from form_state.json."""
        try:
            if not FORM_STATE_PATH.exists():
                return None
            form = json.loads(FORM_STATE_PATH.read_text())
            games = form.get(sector, {}).get(team, [])
            if not games:
                return None
            last_date_str = games[0].get("date")
            if not last_date_str:
                return None
            last_date = datetime.strptime(last_date_str[:10], "%Y-%m-%d").date()
            return (date.today() - last_date).days
        except Exception:
            return None

    def _games_in_last_n_days(self, sector: str, team: str, days: int = 7) -> int:
        """Count games played in the last N days from form_state."""
        try:
            if not FORM_STATE_PATH.exists():
                return 0
            form = json.loads(FORM_STATE_PATH.read_text())
            games = form.get(sector, {}).get(team, [])
            if not games:
                return 0
            cutoff = date.today() - __import__("datetime").timedelta(days=days)
            count = 0
            for g in games:
                d_str = g.get("date", "")[:10]
                if not d_str:
                    continue
                try:
                    gd = datetime.strptime(d_str, "%Y-%m-%d").date()
                except ValueError:
                    continue
                if gd < cutoff:
                    break
                count += 1
            return count
        except Exception:
            return 0

    def _congestion_penalty(self, sector: str, team: str) -> float:
        """Elo penalty for fixture congestion (soccer: UCL midweek + league weekend).

        Teams playing 3+ games in 7 days get a meaningful fatigue penalty.
        """
        if sector != "soccer":
            return 0.0
        games = self._games_in_last_n_days(sector, team, days=7)
        if games >= 3:
            return -40.0
        if games >= 2:
            return -15.0
        return 0.0

    def _rest_elo_bonus(self, sector: str, team: str) -> float:
        """Return Elo bonus/penalty based on days of rest + fixture congestion."""
        days = self._days_of_rest(sector, team)
        base = 0.0
        if days is not None and days <= 7:
            adj_table = REST_ELO_ADJ.get(sector)
            if adj_table:
                clamped = min(days, max(adj_table.keys()))
                base = adj_table.get(clamped, 0.0)

        congestion = self._congestion_penalty(sector, team)
        return base + congestion

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
            notes=self._prediction_notes(sector, team_a, team_b, min_count),
        )

    def _prediction_notes(self, sector: str, team_a: str, team_b: str, min_count: int) -> str:
        base = (
            f"Elo_a={self._get_rating(sector, team_a):.0f} "
            f"Elo_b={self._get_rating(sector, team_b):.0f} "
            f"n={min_count}"
        )
        h2h = self.get_h2h(sector, team_a, team_b)
        if h2h and h2h["games"] >= H2H_MIN_GAMES:
            base += f" H2H={h2h['team_a_wins']}-{h2h['team_b_wins']}"
        rest_a = self._days_of_rest(sector, team_a)
        rest_b = self._days_of_rest(sector, team_b)
        if rest_a is not None or rest_b is not None:
            base += f" rest={rest_a or '?'}d/{rest_b or '?'}d"
        return base

    def _win_probs(
        self, sector: str, team_a: str, team_b: str
    ) -> tuple[float, float, Optional[float]]:
        """
        Compute head-to-head win probabilities with home advantage, H2H, and rest adjustments.
        team_a is treated as home (outcome_a_label from Pinnacle = home team).
        """
        elo_a = self._get_rating(sector, team_a)
        elo_b = self._get_rating(sector, team_b)
        home_bonus = HOME_ADVANTAGE_ELO.get(sector, 0.0)

        # Rest-day adjustments (temporary Elo bonus/penalty)
        rest_a = self._rest_elo_bonus(sector, team_a)
        rest_b = self._rest_elo_bonus(sector, team_b)

        effective_a = elo_a + home_bonus + rest_a
        effective_b = elo_b + rest_b

        expected_a = 1.0 / (1.0 + 10.0 ** ((effective_b - effective_a) / 400.0))

        # Apply head-to-head adjustment
        h2h_adj = self._h2h_adjustment(sector, team_a, team_b)
        expected_a = max(0.01, min(0.99, expected_a + h2h_adj))
        expected_b = 1.0 - expected_a

        # Soccer: allocate draw probability from the expected margin.
        # Draw rate is strength-dependent: evenly matched teams draw ~26%,
        # heavily mismatched teams draw ~10%.  Calibrated against EPL/La Liga
        # historical averages (overall ~25%, but conditional on strength gap).
        if sector == "soccer":
            gap = abs(expected_a - 0.5) * 2.0  # 0 = even, 1 = total mismatch
            # Quadratic decay: 0.26 at gap=0, ~0.10 at gap=0.6
            draw_prob = max(0.08, 0.26 - 0.45 * gap * gap)
            scale = (1.0 - draw_prob) / (expected_a + expected_b)
            return expected_a * scale, expected_b * scale, draw_prob

        return expected_a, expected_b, None

    # ------------------------------------------------------------------
    # Recency-weighted K-factor
    # ------------------------------------------------------------------

    @staticmethod
    def _recency_k(base_k: float, event_date: Optional[str] = None) -> float:
        """Scale K-factor by recency: recent games get up to 2× base K.

        Decay curve:
          - Last 14 days: K multiplier = 2.0 (full recency boost)
          - 14–60 days ago: linearly decays from 2.0 → 1.0
          - 60+ days ago: K multiplier = 1.0 (base K, no boost)
        """
        if not event_date:
            return base_k
        try:
            game_date = datetime.strptime(event_date[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return base_k

        days_ago = (date.today() - game_date).days
        if days_ago < 0:
            days_ago = 0

        if days_ago <= 14:
            multiplier = 2.0
        elif days_ago <= 60:
            # Linear decay from 2.0 at day 14 to 1.0 at day 60
            multiplier = 2.0 - (days_ago - 14) / (60 - 14)
        else:
            multiplier = 1.0

        return base_k * multiplier

    def _sos_multiplier(self, sector: str, opponent: str) -> float:
        """Strength-of-schedule K multiplier.

        Beating a strong opponent (above-average Elo) gives a bigger K boost,
        while beating a weak opponent yields a smaller update.

        Range: [0.85, 1.15] — mild effect to avoid over-amplifying.
        """
        ratings = self._sector_state(sector).get("ratings", {})
        if len(ratings) < 4:
            return 1.0  # not enough teams to compute league average
        opp_elo = self._get_rating(sector, opponent)
        avg_elo = sum(ratings.values()) / len(ratings)
        # Scale: +200 Elo above avg → 1.15x, -200 below → 0.85x
        diff = opp_elo - avg_elo
        mult = 1.0 + diff / (200.0 / 0.15)
        return max(0.85, min(1.15, mult))

    @staticmethod
    def _mov_multiplier(margin: float, elo_diff: float) -> float:
        """Margin-of-victory K multiplier (FiveThirtyEight-inspired).

        Blowouts move ratings more than squeakers, but the effect is dampened
        when the winner was already heavily favored (prevents autocorrelation).

        Calibrated so:
        - 1-point win  → ~0.85x (close games slightly dampened)
        - 5-point win  → ~1.0x  (baseline)
        - 10-point win → ~1.15x
        - 20-point win → ~1.35x
        - 30-point win → ~1.45x (capped at 1.5)
        """
        if margin <= 0:
            return 1.0
        # Logarithmic scaling normalized so margin=5 ≈ 1.0
        raw = math.log(margin + 1) / math.log(6)  # log(6) ≈ 1.79
        # Autocorrelation dampener: shrink when elo_diff is large
        dampener = 2.2 / (abs(elo_diff) * 0.001 + 2.2)
        mov = raw * dampener
        return max(0.75, min(1.5, mov))  # clamp to [0.75, 1.5]

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
        base_k = K_FACTORS.get(sector, 20.0)
        k = self._recency_k(base_k, event_date)

        # Actual score: 1=A wins, 0=B wins, 0.5=draw
        if score_a > score_b:
            actual_a = 1.0
        elif score_b > score_a:
            actual_a = 0.0
        else:
            actual_a = 0.5

        expected_a = 1.0 / (1.0 + 10.0 ** ((elo_b - (elo_a + home_bonus)) / 400.0))

        # Margin-of-victory adjustment: blowouts move ratings more
        margin = abs(score_a - score_b)
        elo_diff = (elo_a + home_bonus) - elo_b if score_a > score_b else (elo_b) - (elo_a + home_bonus)
        mov = self._mov_multiplier(margin, elo_diff)
        k *= mov

        # Strength-of-schedule: games against strong opponents move ratings more
        sos_a = self._sos_multiplier(sector, team_b)  # team_a's opponent
        sos_b = self._sos_multiplier(sector, team_a)  # team_b's opponent
        k *= (sos_a + sos_b) / 2.0

        delta = k * (actual_a - expected_a)
        self._set_rating(sector, team_a, elo_a + delta)
        self._set_rating(sector, team_b, elo_b - delta)
        self._increment_count(sector, team_a)
        self._increment_count(sector, team_b)

        # Record H2H result (skip draws)
        if score_a != score_b:
            self._record_h2h(sector, team_a, team_b, a_won=(score_a > score_b))

        self.log.debug(
            "elo_updated",
            team_a=team_a,
            team_b=team_b,
            delta=round(delta, 2),
            mov=round(mov, 3),
            sos=round((sos_a + sos_b) / 2.0, 3),
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
