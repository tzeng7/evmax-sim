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
    # National-team World Cup: SYMMETRIC home/away avg because WC matches are
    # at neutral venues (no home edge — mirrors the worldcup Elo's
    # HOME_ADVANTAGE_ELO=0). ~1.30 each ≈ the ~2.6 goals/game seen across modern
    # international competition. The home/away *labels* are just side A / side B
    # here, so a flat average makes the score matrix venue-neutral.
    "worldcup": {"home": 1.30, "away": 1.30},
    "nba":      {"home": 113.5, "away": 111.0},
    "nfl":      {"home": 23.5,  "away": 21.5},
    "ncaab":    {"home": 72.0,  "away": 70.0},
    "baseball": {"home": 4.65,  "away": 4.37},  # MLB 2024 averages (runs/game)
    "wnba":     {"home": 83.0,  "away": 81.0},   # WNBA 2023-2024 averages
}

# Supported sectors. Poisson is intentionally SOCCER-ONLY: a low-count
# goal-scoring distribution with weak inter-event dependence is exactly what the
# Poisson / Dixon-Coles model is built for (soccer weight 0.40). Every other
# sector was either explicitly zeroed in the ensemble override (nba/wnba/nfl) or
# is a poor fit for the independence assumption (baseball run overdispersion +
# bullpen leverage; ncaab possession-level correlation).
#
# Membership here — not the override weight — is what truly keeps poisson out:
# the ensemble's per-sector override only re-weights *listed* models, so an
# unlisted model silently falls back to its 0.30 class weight
# (ensemble_agent.py:469). Excluding a sector here means predict_pair returns
# None, so poisson never enters model_preds and never shows in `model_sources`.
#
# `worldcup` (national-team World Cup) is included for the SAME reason soccer is:
# international football is the canonical low-count, weakly-dependent goal process
# the Poisson / Dixon-Coles model is built for. It uses its OWN namespace
# (poisson_state.json['worldcup'], seeded from international results — never the
# club `soccer` pool) and symmetric neutral-venue league averages.
SUPPORTED_SECTORS = {"soccer", "worldcup"}

# Sectors whose goal process is soccer-like (low count, draws are real outcomes):
# apply the Dixon-Coles low-score correction and KEEP the draw mass for the
# 3-way market instead of merging it into home/away.
SOCCER_LIKE_SECTORS = {"soccer", "worldcup"}

# Maximum *bucket* count in the score matrix. Units depend on BUCKET_SIZE.
MAX_SCORE: dict[str, int] = {
    "soccer":   8,
    "worldcup": 8,
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
    "worldcup": 1,
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

# Bayesian shrinkage prior — virtual games pulling attack/defense ratios toward
# 1.0 (league-average). Rationale: ratios computed from <15 games are noisy
# and can reach the [0.3, 2.5] clip rails spuriously (one 5-goal game swings
# attack_ratio +1.5 on a soccer team's third match). Walk-forward exposed
# Poisson's 80-100% P(home) bucket at 87.2% predicted / 72.4% actual — a
# −14.8pp overconfidence that originates from too-wide ratios on low-N teams.
# With PRIOR=8 and a mid-season team at 15 games, shrinkage=15/23≈0.65 toward
# data; a cold-start 3-game team gets 3/11≈0.27 (mostly prior).
POISSON_PRIOR_GAMES: float = 8.0
POISSON_BASELINE_RATIO: float = 1.0

# Per-game implied attack/defense are clamped to this band before folding into
# the moving average. The opponent-adjusted single-game value can spike when the
# opponent's rating is extreme (scoring 3 on a def≈0.3 elite defense implies an
# attack ≈ 7); the clamp keeps one freak result from dominating a team's rating
# while preserving the directional signal.
IMPLIED_RATING_CLAMP: tuple[float, float] = (0.2, 3.0)


def _shrink(ratio: float, games: int, prior: float = POISSON_PRIOR_GAMES) -> float:
    """Bayesian shrinkage of a team rate ratio toward 1.0 (league-average)."""
    if games <= 0:
        return POISSON_BASELINE_RATIO
    w = games / (games + prior)
    return ratio * w + POISSON_BASELINE_RATIO * (1.0 - w)


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

        home_games = home_stats.get("games", 0) if home_stats else 0
        away_games = away_stats.get("games", 0) if away_stats else 0

        home_attack = _shrink(home_stats["attack"], home_games) if home_stats else 1.0
        home_defense = _shrink(home_stats["defense"], home_games) if home_stats else 1.0
        away_attack = _shrink(away_stats["attack"], away_games) if away_stats else 1.0
        away_defense = _shrink(away_stats["defense"], away_games) if away_stats else 1.0

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
            rho=DC_RHO if sector in SOCCER_LIKE_SECTORS else 0.0,
        )
        p_home, p_draw, p_away = _win_draw_probs(matrix)

        # For non-soccer-like sectors: merge draw into proportional split
        if sector not in SOCCER_LIKE_SECTORS:
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
        """Update attack/defense strengths, adjusting for opponent quality.

        Strengths are latent rates in the Poisson model:
            goals = own_attack × opp_defense × league_avg
        so the strength *implied* by a single result must divide out the
        opponent's rating, not just the flat league average:
            implied_attack  = scored   / (league_avg_for     × opp_defense)
            implied_defense = conceded / (league_avg_against  × opp_attack)

        Without the opponent term, beating a weak side inflates a team's
        attack/defense identically to beating a strong one — harmless for a
        balanced round-robin (club leagues, where every team faces the same
        opponents) but badly broken for national teams, whose schedules are
        split by confederation strength. That SOS-blindness ranked AFC/friendly
        minnows (Japan, Russia, Iran) above Brazil/France and pushed underdog
        win probabilities — hence +EV flags — systematically too high. The
        opponent's PRE-game rating is snapshotted first so the two sides of one
        match don't contaminate each other's adjustment.
        """
        sector = sector.lower()
        if sector not in self._state:
            self._state[sector] = {"league_avg": dict(LEAGUE_AVG_DEFAULTS.get(sector, {"home": 1.5, "away": 1.2})), "teams": {}}

        avg = self._league_avg(sector)
        teams = self._state[sector]["teams"]

        def _rating(team: str, key: str) -> float:
            t = teams.get(team.lower().strip())
            return t.get(key, POISSON_BASELINE_RATIO) if t else POISSON_BASELINE_RATIO

        # Snapshot both opponents' pre-game ratings before either side updates.
        a_attack_pre, a_defense_pre = _rating(team_a, "attack"), _rating(team_a, "defense")
        b_attack_pre, b_defense_pre = _rating(team_b, "attack"), _rating(team_b, "defense")

        lo, hi = IMPLIED_RATING_CLAMP

        def _update_team(
            team: str, scored: float, conceded: float, is_home: bool,
            opp_attack: float, opp_defense: float,
        ) -> None:
            team = team.lower().strip()
            if team not in teams:
                teams[team] = {"attack": 1.0, "defense": 1.0, "games": 0}
            t = teams[team]
            n = t["games"]
            avg_for = avg["home"] if is_home else avg["away"]
            avg_against = avg["away"] if is_home else avg["home"]

            # Opponent-adjusted implied strengths (see docstring), clamped so a
            # single freak result against an extreme opponent can't dominate.
            new_attack = scored / max(avg_for * opp_defense, 0.1)
            new_defense = conceded / max(avg_against * opp_attack, 0.1)
            new_attack = max(lo, min(hi, new_attack))
            new_defense = max(lo, min(hi, new_defense))

            t["attack"] = (t["attack"] * n + new_attack) / (n + 1)
            t["defense"] = (t["defense"] * n + new_defense) / (n + 1)
            t["games"] = n + 1

        _update_team(
            team_a, score_a, score_b, is_home=True,
            opp_attack=b_attack_pre, opp_defense=b_defense_pre,
        )
        _update_team(
            team_b, score_b, score_a, is_home=False,
            opp_attack=a_attack_pre, opp_defense=a_defense_pre,
        )

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
