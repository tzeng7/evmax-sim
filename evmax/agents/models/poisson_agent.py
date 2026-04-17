"""PoissonModelAgent — Poisson goal model for soccer (and totals for other sports).

Methodology (Dixon-Coles inspired):
  - Each team has an attack_strength and defense_strength.
  - Expected goals for a match:
        lambda_home = home_attack × away_defense × league_avg_home
        lambda_away = away_attack × home_defense × league_avg_away
  - Win probabilities derived by computing P(home goals > away goals) over
    a simulated score matrix (goals 0–8 each).
  - Dixon-Coles correction applied for low-scoring games (0-0, 1-0, 0-1, 1-1).

Works for soccer and can adapt to NBA/NFL totals if attack/defense stats are loaded.

State file: data/models/poisson_state.json
  {
    "soccer": {
      "league_avg": {"home": 1.55, "away": 1.15},
      "teams": {
        "manchester city": {"attack": 1.42, "defense": 0.65, "games": 28},
        ...
      }
    }
  }

Seeding:
  Use seed_team_stats() to load attack/defense strengths from any source
  (Opta, FBref, WhoScored, etc.).
"""

from __future__ import annotations

import math
from typing import Optional

from evmax.agents.models.base import ModelAgent, ModelAgentPrediction
from evmax.models.market import PredictionMarket
from evmax.models.odds import SharpOdds

# Default league-average goals per game (home / away)
LEAGUE_AVG_DEFAULTS: dict[str, dict[str, float]] = {
    "soccer":   {"home": 1.55, "away": 1.15},
    "nba":      {"home": 113.5, "away": 111.0},
    "nfl":      {"home": 23.5,  "away": 21.5},
    "ncaab":    {"home": 72.0,  "away": 70.0},
    "baseball": {"home": 4.65,  "away": 4.37},  # MLB 2024 averages (runs/game)
    "wnba":     {"home": 83.0,  "away": 81.0},   # WNBA 2023-2024 averages
}

# Supported sectors (Poisson makes most sense for goal/point-scoring games)
SUPPORTED_SECTORS = {"soccer", "nba", "nfl", "ncaab", "baseball", "wnba"}

# Maximum *bucket* count in the score matrix. Units depend on BUCKET_SIZE.
MAX_SCORE: dict[str, int] = {
    "soccer":   8,
    "nba":      25,   # with bucket=5 → effectively 0-125 points
    "nfl":      20,   # with bucket=4 → effectively 0-80 points
    "ncaab":    20,   # with bucket=5 → effectively 0-100 points
    "baseball": 15,   # covers >99% of MLB games
    "wnba":     20,   # with bucket=5 → effectively 0-100 points
}

# Score-bucket size per sector. Basketball/football goals-scored rates are far
# above 1/possession, so a raw Poisson over integer points collapses around the
# mean. Bucketing scales lambda down to a regime where the Poisson mass fits in
# the MAX_SCORE window: win is decided by comparing *buckets scored*.
BUCKET_SIZE: dict[str, int] = {
    "soccer":   1,
    "nba":      5,
    "nfl":      4,
    "ncaab":    5,
    "baseball": 1,
    "wnba":     5,
}

# Dixon-Coles rho correction factor (τ parameter)
DC_RHO = 0.1

# Min games before model is trusted
MIN_GAMES = 5


def _poisson_pmf(lam: float, k: int) -> float:
    """Poisson probability P(X = k) = e^(-λ) × λ^k / k!"""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _dc_correction(lam_h: float, lam_a: float, h: int, a: int, rho: float) -> float:
    """Dixon-Coles low-score correction factor τ."""
    if h == 0 and a == 0:
        return 1.0 - lam_h * lam_a * rho
    elif h == 1 and a == 0:
        return 1.0 + lam_a * rho
    elif h == 0 and a == 1:
        return 1.0 + lam_h * rho
    elif h == 1 and a == 1:
        return 1.0 - rho
    return 1.0


def _score_matrix(lam_h: float, lam_a: float, max_g: int = 8, rho: float = DC_RHO) -> list[list[float]]:
    """Build matrix[home_goals][away_goals] = joint probability."""
    matrix = []
    for h in range(max_g + 1):
        row = []
        for a in range(max_g + 1):
            p = (
                _poisson_pmf(lam_h, h)
                * _poisson_pmf(lam_a, a)
                * _dc_correction(lam_h, lam_a, h, a, rho)
            )
            row.append(p)
        matrix.append(row)
    return matrix


def _win_draw_probs(matrix: list[list[float]]) -> tuple[float, float, float]:
    """Compute P(home win), P(draw), P(away win) from score matrix."""
    p_home = p_draw = p_away = 0.0
    for h, row in enumerate(matrix):
        for a, p in enumerate(row):
            if h > a:
                p_home += p
            elif h == a:
                p_draw += p
            else:
                p_away += p
    total = p_home + p_draw + p_away
    if total < 1e-9:
        return 1 / 3, 1 / 3, 1 / 3
    return p_home / total, p_draw / total, p_away / total


class PoissonModelAgent(ModelAgent):
    """
    Poisson goal model — most applicable to soccer.

    Attack/defense strengths can be seeded from Opta, FBref, or similar.
    Without seeding, the model falls back to league-average (neutral prediction).
    """

    name = "poisson"
    weight = 0.30  # blending weight in ensemble (higher for soccer)

    def _league_avg(self, sector: str) -> dict[str, float]:
        return (
            self._state.get(sector, {}).get("league_avg")
            or LEAGUE_AVG_DEFAULTS.get(sector, {"home": 1.5, "away": 1.2})
        )

    def _team_stats(self, sector: str, team: str) -> Optional[dict]:
        teams = self._state.get(sector, {}).get("teams", {})
        stats = teams.get(team)
        # Fallback: try last word (e.g. "new york knicks" → "knicks")
        if not stats and " " in team:
            last = team.rsplit(" ", 1)[-1]
            stats = teams.get(last)
        # Fallback: prefix/suffix/substring match
        # Handles "duke blue devils" → "duke" and "st. johns red storm" → "st. johns red storm"
        if not stats:
            for key, val in teams.items():
                if (team.startswith(key + " ") or key.startswith(team + " ")
                        or team.endswith(key) or key.endswith(team)):
                    stats = val
                    break
        return stats

    def _expected_goals(
        self,
        sector: str,
        home_team: str,
        away_team: str,
    ) -> tuple[float, float]:
        """Compute expected goals (λ_home, λ_away)."""
        avg = self._league_avg(sector)
        avg_h = avg["home"]
        avg_a = avg["away"]

        home_stats = self._team_stats(sector, home_team)
        away_stats = self._team_stats(sector, away_team)

        home_attack = home_stats["attack"] if home_stats else 1.0
        home_defense = home_stats["defense"] if home_stats else 1.0
        away_attack = away_stats["attack"] if away_stats else 1.0
        away_defense = away_stats["defense"] if away_stats else 1.0

        lam_h = home_attack * away_defense * avg_h
        lam_a = away_attack * home_defense * avg_a
        return lam_h, lam_a

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    async def predict_pair(
        self,
        market: PredictionMarket,
        sharp_odds: SharpOdds,
    ) -> Optional[ModelAgentPrediction]:
        sector = (market.sector or "").lower()
        if sector not in SUPPORTED_SECTORS:
            return None

        home = (sharp_odds.outcome_a_label or market.team_home or "").lower().strip()
        away = (sharp_odds.outcome_b_label or market.team_away or "").lower().strip()

        if not home or not away:
            return None

        home_stats = self._team_stats(sector, home)
        away_stats = self._team_stats(sector, away)

        # Low data → low confidence, but still produce a (league-average) estimate
        has_home = home_stats and home_stats.get("games", 0) >= MIN_GAMES
        has_away = away_stats and away_stats.get("games", 0) >= MIN_GAMES

        lam_h, lam_a = self._expected_goals(sector, home, away)
        max_g = MAX_SCORE.get(sector, 8)
        bucket = BUCKET_SIZE.get(sector, 1)

        # Bucket lambda so the Poisson mass fits within max_g. For soccer/MLB
        # this is a no-op (bucket=1); for basketball/football it maps raw point
        # totals into "buckets scored" so the matrix isn't truncated.
        lam_h_bucketed = lam_h / bucket
        lam_a_bucketed = lam_a / bucket

        matrix = _score_matrix(
            lam_h_bucketed,
            lam_a_bucketed,
            max_g=max_g,
            rho=DC_RHO if sector == "soccer" else 0.0,
        )
        p_home, p_draw, p_away = _win_draw_probs(matrix)

        # For non-soccer: merge draw into proportional split
        if sector != "soccer":
            p_home += p_draw * (p_home / (p_home + p_away + 1e-9))
            p_away += p_draw * (p_away / (p_home + p_away + 1e-9))
            p_draw = None  # type: ignore

        games_h = home_stats.get("games", 0) if home_stats else 0
        games_a = away_stats.get("games", 0) if away_stats else 0
        min_games = min(games_h, games_a)

        if has_home and has_away:
            confidence = 0.65 + 0.15 * min(1.0, (min_games - MIN_GAMES) / 25)
        elif has_home or has_away:
            confidence = 0.40  # one team unseeded — not reliable enough for blend
        else:
            confidence = 0.30   # league-average fallback

        return ModelAgentPrediction(
            event_id=sharp_odds.event_id,
            model_name=self.name,
            true_prob_a=p_home,
            true_prob_b=p_away,
            true_prob_draw=p_draw,
            confidence=confidence,
            weight=self.weight,
            sample_size=min_games,
            notes=(
                f"λ_h={lam_h:.2f} λ_a={lam_a:.2f} "
                f"P(h/d/a)={p_home:.3f}/{(p_draw or 0):.3f}/{p_away:.3f}"
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
        """Update attack/defense strengths via simple moving average of per-game rates."""
        sector = sector.lower()
        if sector not in self._state:
            self._state[sector] = {"league_avg": dict(LEAGUE_AVG_DEFAULTS.get(sector, {"home": 1.5, "away": 1.2})), "teams": {}}

        avg = self._league_avg(sector)
        teams = self._state[sector]["teams"]

        def _update_team(team: str, scored: float, conceded: float, is_home: bool) -> None:
            team = team.lower().strip()
            if team not in teams:
                teams[team] = {"attack": 1.0, "defense": 1.0, "games": 0}
            t = teams[team]
            n = t["games"]
            avg_for = avg["home"] if is_home else avg["away"]
            avg_against = avg["away"] if is_home else avg["home"]

            # Incremental average of strength ratios
            new_attack = scored / max(avg_for, 0.1)
            new_defense = conceded / max(avg_against, 0.1)

            t["attack"] = (t["attack"] * n + new_attack) / (n + 1)
            t["defense"] = (t["defense"] * n + new_defense) / (n + 1)
            t["games"] = n + 1

        _update_team(team_a, score_a, score_b, is_home=True)
        _update_team(team_b, score_b, score_a, is_home=False)

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def seed_team_stats(
        self,
        sector: str,
        team_stats: dict[str, dict],
        league_avg: Optional[dict[str, float]] = None,
    ) -> None:
        """
        Bulk-load team attack/defense statistics.

        Args:
            sector: Sport sector.
            team_stats: {team_name: {"attack": float, "defense": float, "games": int}}
            league_avg: {"home": float, "away": float} — defaults to LEAGUE_AVG_DEFAULTS.
        """
        if sector not in self._state:
            self._state[sector] = {}
        if league_avg:
            self._state[sector]["league_avg"] = league_avg
        self._state[sector]["teams"] = {
            k.lower().strip(): v for k, v in team_stats.items()
        }
        self.save_state()
        self.log.info("poisson_seeded", sector=sector, teams=len(team_stats))
