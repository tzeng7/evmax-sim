"""WNBAPossessionSimAgent — Monte Carlo possession-level WNBA game simulation.

Standalone WNBA version of the NBA possession sim. Reads team ORTG/DRTG/
Pace/TOV% from data/models/wnba_efficiency_state.json (seeded by
scripts/seed_wnba_efficiency.py) and runs N_SIMS possession-by-possession
simulations per matchup.

Architecture mirror of evmax/agents/models/possession_sim_agent.py but
entirely separate — never touches NBA state, never fires for non-WNBA
sectors. NBA's agent is preserved exactly as-is.

Data flow:
  wnba_efficiency_state.json
    → team stats: {ortg, drtg, pace, tov_pct, gp, ...}
    → league averages: {league_avg_ortg, league_avg_pace}
    → feeds _simulate_game(N_SIMS=10,000) per matchup
    → caches margin/total distributions per event_id for spread/totals markets

WNBA-specific calibration (vs NBA):
  - LEAGUE_AVG_ORTG = 100.6 (NBA: 114.5) — measured from 2025 seed
  - Possessions clipped to [65, 100] (NBA: [80, 120]) — WNBA pace 78-84
  - Margin σ = 12.5 (NBA: 11.5) — matches WNBA efficiency's SCORE_STDEV
  - Total σ = 18.0 (NBA: 20.0) — 40-min game → lower total variance
  - MIN_GAMES = 4 (NBA: 20) — lowered from 12 once empirical-Bayes shrinkage
    landed; shrinkage handles the small-sample noise a hard floor used to
  - Playoff tightening disabled — WNBA playoffs not calibrated yet

Returns None for any sector other than "wnba". WNBA moneyline is live
(2026-05-26); spread / total are disabled_market_types for the scanner and
owned by the anchored-entry trigger, so the cover_probability /
total_probability methods stay ported but unused on the live path.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import structlog

from evmax.agents.models.base import ModelAgent, ModelAgentPrediction
from evmax.agents.models.wnba_efficiency_agent import (
    shrink_team_stats,
    smooth_confidence,
    state_is_stale_for_today,
)
from evmax.models.market import PredictionMarket
from evmax.models.odds import SharpOdds

logger = structlog.get_logger(__name__)

WNBA_EFFICIENCY_STATE_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "models" / "wnba_efficiency_state.json"
)

N_SIMS = 10_000

# Home-court advantage in ORTG points. NBA uses 1.5; WNBA home edge is
# weaker per base-rate (home WR 56.7% vs NBA 59%) but the ORTG→margin
# conversion also differs due to pace. 1.5 is a defensible starting
# point; tune via walk-forward after first 60 games of 2026 data.
HOME_EDGE_ORTG = 1.5

# Fallback only — real value comes from the state file. Matches the 2025
# seeded league average so predictions are sensible even if state is missing.
LEAGUE_AVG_ORTG = 100.6

# Per-team fallback when a team entry has no TOV%. WNBA average is higher
# than NBA's 13.5% — set to match the 2025 seeded league mean.
AVG_TOV_RATE = 0.17

# Games-played gate — minimum per team before we trust the simulation.
# Lowered 12 → 4 on 2026-05-30: empirical-Bayes shrinkage handles small-
# sample noise (see shrink_team_stats() in wnba_efficiency_agent). With
# shrinkage active the sim contributes lightly from week 1 instead of
# going dark for 3+ weeks at each season start.
MIN_GAMES = 4

# Possession count clipping: WNBA pace is 78-84, so clip to [65, 100] to
# allow ±15 variance without unreasonable tails.
POSSESSIONS_CLIP_MIN = 65
POSSESSIONS_CLIP_MAX = 100

# Margin σ for cover_probability (spread markets) and total σ for
# total_probability. Matches WNBA efficiency agent's SCORE_STDEV = 12.5
# and an 18.0 total-scoreline std (NBA's raw sim is too narrow; WNBA
# inherits the same calibration gap).
SPREAD_SIGMA = 12.5
TOTAL_SIGMA = 18.0

# Per-possession scoring std dev. WNBA possessions look like NBA in
# shape (mix of 0s, 2s, 3s, and-1s, FTs) so 1.10 is a reasonable start.
# Small effect on Brier; not worth over-tuning up front.
PTS_STD = 1.10


class WNBAPossessionSimAgent(ModelAgent):
    """Monte Carlo possession-level WNBA game simulator.

    Fires only when `market.sector == "wnba"`. Reads its own efficiency
    state file; never touches NBA data.
    """

    name = "wnba_possession_sim"
    weight = 0.25   # Matches NBA possession_sim's effective contribution
                    # after blending (NBA class weight 0.35, sector override 0.30).

    def __init__(self) -> None:
        super().__init__()
        self._efficiency_data: Optional[dict] = None
        self._rng = np.random.default_rng(seed=42)
        self._margin_cache: dict[str, np.ndarray] = {}
        self._total_cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # State loading
    # ------------------------------------------------------------------

    def _load_efficiency_state(self) -> dict:
        """Read data/models/wnba_efficiency_state.json lazily with daily caching.

        Returns the full state dict (teams + league averages) or {} when the
        file is missing. Mirrors NBA's pattern but points at the WNBA file.
        """
        if self._efficiency_data is not None:
            cached_date = self._efficiency_data.get("fetched_at", "")
            if cached_date == date.today().isoformat():
                return self._efficiency_data

        if WNBA_EFFICIENCY_STATE_PATH.exists():
            try:
                data = json.loads(WNBA_EFFICIENCY_STATE_PATH.read_text())
                if data.get("teams"):
                    self._efficiency_data = data
                    return data
            except Exception:
                pass
        return {}

    @staticmethod
    def _resolve_team(teams: dict, team: str) -> Optional[dict]:
        """Resolve a team name / alias to the stats dict."""
        team = team.lower().strip()
        if team in teams:
            return teams[team]
        if " " in team:
            last = team.rsplit(" ", 1)[-1]
            if last in teams:
                return teams[last]
        for key, val in teams.items():
            full = val.get("full_name", "")
            if team in full or (full and full.endswith(team)) or team.startswith(key):
                return val
        return None

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

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
        ortg_adj_a: float = 0.0,
        ortg_adj_b: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run N_SIMS possession-level game simulations.

        For each simulated game:
          1. Draw possession count from Normal(avg_pace, σ=3), clip to [65, 100]
          2. For each possession, coin-flip turnover vs scoring outcome
          3. For scoring possessions, sample points per poss ~ Normal(ppp, PTS_STD)
          4. Sum possession outcomes per team

        `ortg_adj_{a,b}` are additive ORTG deltas (e.g. from injury hits);
        callers compute them upstream. Default 0 means no adjustment.
        """
        expected_pace = (pace_a + pace_b) / 2.0
        possessions = self._rng.normal(expected_pace, 3.0, size=N_SIMS).astype(int)
        possessions = np.clip(possessions, POSSESSIONS_CLIP_MIN, POSSESSIONS_CLIP_MAX)

        # Offensive / defensive factors relative to league average
        off_factor_a = (ortg_a + HOME_EDGE_ORTG + ortg_adj_a) / league_ortg
        def_factor_a = drtg_a / league_ortg
        off_factor_b = (ortg_b + ortg_adj_b) / league_ortg
        def_factor_b = drtg_b / league_ortg

        # Points per possession, adjusted for opponent defense
        ppp_a = off_factor_a * def_factor_b * league_ortg / 100.0
        ppp_b = off_factor_b * def_factor_a * league_ortg / 100.0

        scores_a = np.zeros(N_SIMS)
        scores_b = np.zeros(N_SIMS)

        for i in range(N_SIMS):
            n_poss = possessions[i]

            tov_mask_a = self._rng.random(n_poss) < tov_a
            scoring_poss_a = n_poss - tov_mask_a.sum()
            if scoring_poss_a > 0:
                # ppp is per-team-possession; scale up for scoring (non-TO) posses.
                # Same no-clip rationale as NBA: sum over ~80 posses is always
                # positive, clipping the negative tail would bias mean up.
                pts_per = self._rng.normal(ppp_a / (1 - tov_a), PTS_STD, size=scoring_poss_a)
                scores_a[i] = pts_per.sum()

            tov_mask_b = self._rng.random(n_poss) < tov_b
            scoring_poss_b = n_poss - tov_mask_b.sum()
            if scoring_poss_b > 0:
                pts_per = self._rng.normal(ppp_b / (1 - tov_b), PTS_STD, size=scoring_poss_b)
                scores_b[i] = pts_per.sum()

        return scores_a, scores_b

    def simulate_matchup(
        self,
        home_team: str,
        away_team: str,
        event_id: Optional[str] = None,
        ortg_adj_a: float = 0.0,
        ortg_adj_b: float = 0.0,
    ) -> Optional[dict]:
        """Run one matchup's possession sim synchronously; cache margin/total."""
        eff_data = self._load_efficiency_state()
        if state_is_stale_for_today(eff_data):
            logger.warning(
                "wnba_possession_sim_stale_source_season",
                source_season=eff_data.get("source_season"),
                current_year=date.today().year,
                hint="re-run scripts/seed_wnba_efficiency.py --year <current>",
            )
            return None
        teams = eff_data.get("teams", {})
        if not teams:
            return None

        team_a = home_team.lower().strip()
        team_b = away_team.lower().strip()

        stats_a = self._resolve_team(teams, team_a)
        stats_b = self._resolve_team(teams, team_b)
        if not stats_a or not stats_b:
            return None
        if stats_a.get("gp", 0) < MIN_GAMES or stats_b.get("gp", 0) < MIN_GAMES:
            return None

        league_ortg = eff_data.get("league_avg_ortg", LEAGUE_AVG_ORTG)
        # Use the same league dict shape the efficiency agent uses so
        # shrink_team_stats can be shared. league_avg_pace falls back to
        # the average of seeded paces when missing.
        league = {
            "league_avg_ortg": league_ortg,
            "league_avg_drtg": eff_data.get("league_avg_drtg", league_ortg),
            "league_avg_pace": eff_data.get("league_avg_pace", 82.0),
            "league_avg_tov_pct": AVG_TOV_RATE,
        }
        # Empirical-Bayes shrinkage matches the efficiency agent so both
        # consumers of this state file see the same regularized inputs.
        stats_a = shrink_team_stats(stats_a, league)
        stats_b = shrink_team_stats(stats_b, league)
        tov_a = stats_a.get("tov_pct", AVG_TOV_RATE)
        tov_b = stats_b.get("tov_pct", AVG_TOV_RATE)

        adj_suffix = (
            f"_adj{ortg_adj_a:+.1f}{ortg_adj_b:+.1f}"
            if (ortg_adj_a or ortg_adj_b) else ""
        )
        seed = hash(
            f"wnba_{team_a}_{team_b}_{date.today().isoformat()}{adj_suffix}"
        ) & 0xFFFFFFFF
        self._rng = np.random.default_rng(seed=seed)

        scores_a, scores_b = self._simulate_game(
            ortg_a=stats_a["ortg"], drtg_a=stats_a["drtg"],
            ortg_b=stats_b["ortg"], drtg_b=stats_b["drtg"],
            pace_a=stats_a["pace"], pace_b=stats_b["pace"],
            tov_a=tov_a, tov_b=tov_b,
            league_ortg=league_ortg,
            ortg_adj_a=ortg_adj_a, ortg_adj_b=ortg_adj_b,
        )

        wins_a = (scores_a > scores_b).sum()
        ties = (scores_a == scores_b).sum()
        prob_a = (wins_a + ties * 0.5) / N_SIMS
        prob_a = max(0.02, min(0.98, prob_a))

        margin_dist = scores_a - scores_b
        total_dist = scores_a + scores_b

        eid = event_id or f"wnba_sim_{team_a}_{team_b}"
        self._margin_cache[eid] = margin_dist
        self._total_cache[eid] = total_dist

        confidence = smooth_confidence(min(stats_a["gp"], stats_b["gp"]))

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

    # ------------------------------------------------------------------
    # Spread / totals probabilities (for future promotion from shadow)
    # ------------------------------------------------------------------

    def cover_probability(
        self, event_id: str, line: float, yes_is_underdog: bool = False,
    ) -> Optional[float]:
        """P(team_a margin > line) using sim mean + calibrated σ=12.5."""
        margins = self._margin_cache.get(event_id)
        if margins is None:
            return None
        sim_mean = float(margins.mean())
        from scipy.stats import norm
        if yes_is_underdog:
            z = (line - sim_mean) / SPREAD_SIGMA
            return float(norm.cdf(z))
        z = (-line - sim_mean) / SPREAD_SIGMA
        return float(1.0 - norm.cdf(z))

    def total_probability(
        self, event_id: str, line: float, is_over: bool = True,
    ) -> Optional[float]:
        """P(total > line) or P(total < line) using sim mean + calibrated σ=18.0."""
        totals = self._total_cache.get(event_id)
        if totals is None:
            return None
        sim_mean = float(totals.mean())
        from scipy.stats import norm
        if is_over:
            z = (line - sim_mean) / TOTAL_SIGMA
            return float(1.0 - norm.cdf(z))
        z = (line - sim_mean) / TOTAL_SIGMA
        return float(norm.cdf(z))

    # ------------------------------------------------------------------
    # ModelAgent interface
    # ------------------------------------------------------------------

    async def predict_pair(
        self,
        market: PredictionMarket,
        sharp_odds: SharpOdds,
    ) -> Optional[ModelAgentPrediction]:
        sector = (market.sector or "").lower()
        if sector != "wnba":
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
                f"wnba_sim={N_SIMS} margin={avg_margin:+.1f} "
                f"total={avg_total:.0f} "
                f"score={avg_score_a:.0f}-{avg_score_b:.0f} "
                f"sigma={margin_sigma:.1f}"
            ),
        )

    def update(
        self, team_a: str, team_b: str, score_a: float, score_b: float,
        sector: str, event_date: Optional[str] = None,
    ) -> None:
        """No-op — efficiency stats refreshed by scripts/seed_wnba_efficiency.py."""
        return
