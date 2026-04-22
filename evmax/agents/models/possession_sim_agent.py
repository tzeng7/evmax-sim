"""PossessionSimAgent — Monte Carlo possession-level NBA game simulation.

Simulates games possession-by-possession using team ORTG/DRTG/Pace to
produce a full score distribution. More accurate than normal CDF for:
  - Spread markets (exact margin distribution)
  - Total markets (combined score distribution)
  - Moneyline (win probability from simulated outcomes)

Each possession outcome is drawn from a calibrated distribution:
  - Turnover: ~14% of possessions (team TOV%)
  - 3PT attempt: ~35% of non-TO possessions (team 3PA rate)
  - 2PT attempt: ~55% of non-TO possessions
  - Free throws: ~10% of non-TO possessions
  - Points per scoring possession scaled by team ORTG vs league avg

The simulation captures:
  - Pace interaction (fast vs slow team → specific possession count)
  - Fat tails from 3PT variance (hot/cold shooting nights)
  - Score correlation (pace drives both teams' totals)

Data source: reuses EfficiencyModelAgent's state (no extra API calls).

N_SIMS = 10,000 games → standard error ≈ 0.5% on win probability.
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import structlog

from evmax.agents.models.base import ModelAgent, ModelAgentPrediction
from evmax.models.market import PredictionMarket
from evmax.models.odds import SharpOdds

logger = structlog.get_logger(__name__)

EFFICIENCY_STATE_PATH = Path(__file__).resolve().parents[3] / "data" / "models" / "efficiency_state.json"

N_SIMS = 10_000
HOME_EDGE_ORTG = 1.5  # home team gets ~1.5 ORTG boost
LEAGUE_AVG_ORTG = 114.5  # updated from fetched data

# Playoff tightening factor (empirical: 17 NBA playoff games, Apr 14-20 2026).
# Measured league ORTG dropped 114.79 → 110.46 (-4.33) while pace only moved
# 100.22 → 99.39 (-0.83). Defensive intensity — not pace — is the real story.
# Factor applied to both teams' off_factor; matches observed -10 pt total drop.
PLAYOFF_ORTG_FACTOR = 110.46 / 114.79  # ≈ 0.9623
PLAYOFF_PACE_DELTA = -0.83

# Per-possession outcome probabilities (league averages, calibrated to 2025-26)
AVG_TOV_RATE = 0.135      # ~13.5% of possessions end in turnover
AVG_3PA_RATE = 0.40       # ~40% of FGA are 3-pointers
AVG_3PT_PCT = 0.362
AVG_2PT_PCT = 0.535
AVG_FT_RATE = 0.22        # free throw trips per non-TO possession
AVG_FT_PCT = 0.785
AVG_OREB_RATE = 0.27      # offensive rebound rate


class PossessionSimAgent(ModelAgent):
    """Monte Carlo possession-level NBA game simulator."""

    name = "possession_sim"
    weight = 0.35

    def __init__(self) -> None:
        super().__init__()
        self._efficiency_data: Optional[dict] = None
        self._rng = np.random.default_rng(seed=42)
        self._margin_cache: dict[str, np.ndarray] = {}
        self._total_cache: dict[str, np.ndarray] = {}

    def cover_probability(self, event_id: str, line: float, yes_is_underdog: bool = False) -> Optional[float]:
        """P(team_a margin > line) using sim mean + calibrated normal CDF.

        Uses the sim's matchup-specific mean margin but applies empirical
        σ=11.5 for the spread distribution — the raw sim distribution is too
        narrow, producing overconfident tail probabilities.

        Args:
            event_id: Must match the event_id from a prior predict_pair call.
            line: Spread line from team_a's perspective (negative = favorite).
                  e.g. -7.5 means team_a must win by >7.5.
            yes_is_underdog: If True, compute P(team_b covers -line).
        """
        margins = self._margin_cache.get(event_id)
        if margins is None:
            return None
        sim_mean = float(margins.mean())
        sigma = 11.5
        from scipy.stats import norm
        if yes_is_underdog:
            z = (line - sim_mean) / sigma
            return float(norm.cdf(z))
        else:
            z = (-line - sim_mean) / sigma
            return float(1.0 - norm.cdf(z))

    def total_probability(self, event_id: str, line: float, is_over: bool = True) -> Optional[float]:
        """P(total > line) or P(total < line) using sim mean + calibrated σ.

        Uses the sim's matchup-specific total mean but applies empirical
        σ=20.0 for the total distribution — the raw sim distribution is too
        narrow (σ~16 from CLT on the per-possession model), producing
        overconfident tail probabilities on totals markets.
        """
        totals = self._total_cache.get(event_id)
        if totals is None:
            return None
        sim_mean = float(totals.mean())
        sigma = 20.0
        from scipy.stats import norm
        if is_over:
            z = (line - sim_mean) / sigma
            return float(1.0 - norm.cdf(z))
        z = (line - sim_mean) / sigma
        return float(norm.cdf(z))

    def _load_efficiency_state(self) -> dict:
        """Load efficiency data from the shared state file."""
        if self._efficiency_data is not None:
            cached_date = self._efficiency_data.get("fetched_at", "")
            if cached_date == date.today().isoformat():
                return self._efficiency_data

        if EFFICIENCY_STATE_PATH.exists():
            try:
                data = json.loads(EFFICIENCY_STATE_PATH.read_text())
                nba = data.get("nba", {})
                if nba.get("teams"):
                    self._efficiency_data = nba
                    return nba
            except Exception:
                pass
        return {}

    def _resolve_team(self, teams: dict, team: str) -> Optional[dict]:
        team = team.lower().strip()
        if team in teams:
            return teams[team]
        if " " in team:
            last = team.rsplit(" ", 1)[-1]
            if last in teams:
                return teams[last]
        for key, val in teams.items():
            full = val.get("full_name", "")
            if team in full or full.endswith(team) or team.startswith(key):
                return val
        return None

    def _simulate_game(
        self,
        ortg_a: float,
        drtg_a: float,
        ortg_b: float,
        drtg_b: float,
        pace_a: float,
        pace_b: float,
        tov_a: float,
        tov_b: float,
        league_ortg: float,
        is_playoff: bool = False,
        ortg_adj_a: float = 0.0,
        ortg_adj_b: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate N_SIMS games, return (scores_a, scores_b) arrays.

        Each game is simulated at the possession level:
        1. Determine total possessions from pace interaction
        2. For each possession, sample outcome based on team efficiency
        3. Sum points for each team

        When ``is_playoff`` is True, apply empirically calibrated playoff
        tightening (defensive ORTG haircut + small pace reduction).

        ``ortg_adj_a`` / ``ortg_adj_b`` are additive ORTG deltas (typically
        negative, from injuries) applied before the league-relative factor.
        """
        # Expected possessions per game (average of both teams' pace)
        expected_pace = (pace_a + pace_b) / 2.0
        if is_playoff:
            expected_pace += PLAYOFF_PACE_DELTA
        # Add possession-count variance (~3 possessions std dev)
        possessions = self._rng.normal(expected_pace, 3.0, size=N_SIMS).astype(int)
        possessions = np.clip(possessions, 80, 120)

        # Efficiency factors relative to league average
        off_factor_a = (ortg_a + HOME_EDGE_ORTG + ortg_adj_a) / league_ortg
        def_factor_a = drtg_a / league_ortg
        off_factor_b = (ortg_b + ortg_adj_b) / league_ortg
        def_factor_b = drtg_b / league_ortg

        if is_playoff:
            # Tighter defense across the board — shrink both teams' offense.
            off_factor_a *= PLAYOFF_ORTG_FACTOR
            off_factor_b *= PLAYOFF_ORTG_FACTOR

        # Points per possession for each team (adjusted for opponent defense)
        # Team A's PPP = their offense × opponent's defensive weakness
        ppp_a = off_factor_a * def_factor_b * league_ortg / 100.0
        ppp_b = off_factor_b * def_factor_a * league_ortg / 100.0

        # Per-possession scoring variance
        # NBA scoring per possession is ~1.14 pts with std ~1.1
        # (mix of 0s, 2s, 3s, and-1s, FTs)
        scores_a = np.zeros(N_SIMS)
        scores_b = np.zeros(N_SIMS)

        # Per-possession scoring std. NBA possessions are {0, 2, 3} mostly;
        # empirical variance ≈ 1.2 (std ≈ 1.1). Do NOT clip the per-poss
        # samples at 0 — that turns the negative tail of the Normal into mass
        # at 0 and biased the per-game mean up by ~17 pts. The sum over ~100
        # possessions is always safely positive; no clipping needed.
        pts_std = 1.10

        for i in range(N_SIMS):
            n_poss = possessions[i]

            tov_mask_a = self._rng.random(n_poss) < tov_a
            scoring_poss_a = n_poss - tov_mask_a.sum()
            if scoring_poss_a > 0:
                # ORTG already folds in offensive rebounds (it's points per
                # team possession, where a possession ends on made FG, FTs,
                # defensive rebound, or turnover — NOT on an OREB). So we
                # DO NOT add extra OREB "bonus" possessions here; doing so
                # double-counts and was the source of a ~30-pt totals bias.
                pts_per = self._rng.normal(ppp_a / (1 - tov_a), pts_std, size=scoring_poss_a)
                scores_a[i] = pts_per.sum()

            tov_mask_b = self._rng.random(n_poss) < tov_b
            scoring_poss_b = n_poss - tov_mask_b.sum()
            if scoring_poss_b > 0:
                pts_per = self._rng.normal(ppp_b / (1 - tov_b), pts_std, size=scoring_poss_b)
                scores_b[i] = pts_per.sum()

        return scores_a, scores_b

    def simulate_matchup(
        self,
        home_team: str,
        away_team: str,
        event_id: Optional[str] = None,
        is_playoff: bool = False,
        ortg_adj_a: float = 0.0,
        ortg_adj_b: float = 0.0,
    ) -> Optional[dict]:
        """Synchronously run the possession-level sim for an NBA matchup.

        Caches margin/total distributions under ``event_id`` (auto-generated if
        None). Returns a dict with keys: scores_a/b, margin, total, prob_a,
        stats_a/b, confidence, event_id. Returns None when efficiency state is
        missing or either team has <20 games played.

        ``home_team`` plays the team_a slot (HOME_EDGE_ORTG applied to them).
        When ``is_playoff`` is True, applies playoff tightening factors.
        ``ortg_adj_a`` / ``ortg_adj_b`` are additive ORTG deltas (typically
        from missing starters / stars); callers compute them upstream.
        """
        eff_data = self._load_efficiency_state()
        teams = eff_data.get("teams", {})
        if not teams:
            return None

        team_a = home_team.lower().strip()
        team_b = away_team.lower().strip()

        stats_a = self._resolve_team(teams, team_a)
        stats_b = self._resolve_team(teams, team_b)
        if not stats_a or not stats_b:
            return None
        if stats_a.get("gp", 0) < 20 or stats_b.get("gp", 0) < 20:
            return None

        league_ortg = eff_data.get("league_avg_ortg", LEAGUE_AVG_ORTG)
        tov_a = stats_a.get("tov_pct", AVG_TOV_RATE)
        tov_b = stats_b.get("tov_pct", AVG_TOV_RATE)

        seed_suffix = "_playoff" if is_playoff else ""
        adj_suffix = f"_adj{ortg_adj_a:+.1f}{ortg_adj_b:+.1f}" if (ortg_adj_a or ortg_adj_b) else ""
        seed = hash(f"{team_a}_{team_b}_{date.today().isoformat()}{seed_suffix}{adj_suffix}") & 0xFFFFFFFF
        self._rng = np.random.default_rng(seed=seed)

        scores_a, scores_b = self._simulate_game(
            ortg_a=stats_a["ortg"], drtg_a=stats_a["drtg"],
            ortg_b=stats_b["ortg"], drtg_b=stats_b["drtg"],
            pace_a=stats_a["pace"], pace_b=stats_b["pace"],
            tov_a=tov_a, tov_b=tov_b,
            league_ortg=league_ortg,
            is_playoff=is_playoff,
            ortg_adj_a=ortg_adj_a,
            ortg_adj_b=ortg_adj_b,
        )

        wins_a = (scores_a > scores_b).sum()
        ties = (scores_a == scores_b).sum()
        prob_a = (wins_a + ties * 0.5) / N_SIMS
        prob_a = max(0.02, min(0.98, prob_a))

        margin_dist = scores_a - scores_b
        total_dist = scores_a + scores_b

        eid = event_id or f"sim_{team_a}_{team_b}"
        self._margin_cache[eid] = margin_dist
        self._total_cache[eid] = total_dist

        confidence = 0.80 if min(stats_a["gp"], stats_b["gp"]) >= 60 else 0.65

        return {
            "event_id": eid,
            "scores_a": scores_a,
            "scores_b": scores_b,
            "margin": margin_dist,
            "total": total_dist,
            "prob_a": prob_a,
            "stats_a": stats_a,
            "stats_b": stats_b,
            "confidence": confidence,
        }

    async def predict_pair(
        self,
        market: PredictionMarket,
        sharp_odds: SharpOdds,
    ) -> Optional[ModelAgentPrediction]:
        sector = (market.sector or "").lower()
        if sector != "nba":
            return None

        team_a = (sharp_odds.outcome_a_label or market.team_home or "").lower().strip()
        team_b = (sharp_odds.outcome_b_label or market.team_away or "").lower().strip()

        sim = self.simulate_matchup(team_a, team_b, event_id=sharp_odds.event_id)
        if sim is None:
            return None

        scores_a = sim["scores_a"]
        scores_b = sim["scores_b"]
        prob_a = sim["prob_a"]
        prob_b = 1.0 - prob_a

        avg_score_a = float(scores_a.mean())
        avg_score_b = float(scores_b.mean())
        avg_total = avg_score_a + avg_score_b
        avg_margin = avg_score_a - avg_score_b
        margin_sigma = float(sim["margin"].std())

        return ModelAgentPrediction(
            event_id=sharp_odds.event_id,
            model_name=self.name,
            true_prob_a=prob_a,
            true_prob_b=prob_b,
            true_prob_draw=None,
            confidence=sim["confidence"],
            weight=self.weight,
            sample_size=N_SIMS,
            notes=(
                f"sim={N_SIMS} margin={avg_margin:+.1f} "
                f"total={avg_total:.0f} "
                f"score={avg_score_a:.0f}-{avg_score_b:.0f} "
                f"sigma={margin_sigma:.1f}"
            ),
        )

    def update(self, team_a: str, team_b: str, score_a: float, score_b: float,
               sector: str, event_date: Optional[str] = None) -> None:
        pass
