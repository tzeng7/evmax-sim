"""NcaawPossessionSimAgent — Monte Carlo possession-level NCAAW game simulation.

Women's college analogue of the NCAAB possession sim (see that module for the
architecture docs). Reads data/models/ncaaw_efficiency_state.json (seeded by
scripts/seed_ncaab_efficiency.py --league womens) and shares MIN_GAMES /
SHRINK_K with the ncaaw_efficiency agent so both state consumers see
identical regularized inputs.

NCAAW-specific calibration:
  - Possessions clipped to [55, 92] — women's college pace runs ~66-76 per 40
  - Spread/total sigmas sized for the women's game (wider margins, similar
    totals variance)

Parallel-stack rules: never touches NBA/WNBA code or state.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import structlog

from evmax.agents.models._college_efficiency import (
    resolve_team,
    shrink_team_stats,
    simulate_matchup_vectorized,
    smooth_confidence,
    state_is_stale_for_today,
)
from evmax.agents.models.ncaaw_efficiency_agent import (
    FULL_SEASON_GP,
    HCA_EFF_FALLBACK,
    MIN_GAMES,
    SHRINK_K,
)
from evmax.agents.models.base import ModelAgent, ModelAgentPrediction
from evmax.models.market import PredictionMarket
from evmax.models.odds import SharpOdds

logger = structlog.get_logger(__name__)

SECTOR = "ncaaw"

EFFICIENCY_STATE_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "models" / "ncaaw_efficiency_state.json"
)

N_SIMS = 10_000

POSSESSIONS_CLIP = (55.0, 92.0)

PTS_STD = 1.10
AVG_TOV_RATE = 0.18

SPREAD_SIGMA = 11.5
TOTAL_SIGMA = 16.0


class NcaawPossessionSimAgent(ModelAgent):
    """Monte Carlo possession-level NCAAW simulator (sector == "ncaaw" only)."""

    name = "ncaaw_possession_sim"
    weight = 0.25

    def __init__(self) -> None:
        super().__init__()
        self._efficiency_data: Optional[dict] = None
        self._margin_cache: dict[str, np.ndarray] = {}
        self._total_cache: dict[str, np.ndarray] = {}
        self._normalizer = None

    def _normalize(self, name: str) -> str:
        if self._normalizer is None:
            from evmax.matching.normalizer import NameNormalizer
            self._normalizer = NameNormalizer(SECTOR)
        return self._normalizer.normalize(name)

    def _load_efficiency_state(self) -> dict:
        if self._efficiency_data is not None:
            if self._efficiency_data.get("fetched_at", "") == date.today().isoformat():
                return self._efficiency_data
        if EFFICIENCY_STATE_PATH.exists():
            try:
                data = json.loads(EFFICIENCY_STATE_PATH.read_text())
                if data.get("teams"):
                    self._efficiency_data = data
                    return data
            except Exception:
                pass
        return {}

    def simulate_matchup(
        self,
        home_team: str,
        away_team: str,
        event_id: Optional[str] = None,
    ) -> Optional[dict]:
        eff = self._load_efficiency_state()
        if not eff:
            return None
        if state_is_stale_for_today(eff):
            logger.warning(
                "ncaaw_possession_sim_stale_source_season",
                source_season=eff.get("source_season"),
                hint="re-run scripts/seed_ncaab_efficiency.py --league womens",
            )
            return None
        teams = eff.get("teams", {})
        if not teams:
            return None

        stats_h = resolve_team(teams, home_team, normalize=self._normalize)
        stats_a = resolve_team(teams, away_team, normalize=self._normalize)
        if not stats_h or not stats_a:
            return None
        if stats_h.get("gp", 0) < MIN_GAMES or stats_a.get("gp", 0) < MIN_GAMES:
            return None

        league = {
            "league_avg_ortg": eff.get("league_avg_ortg", 95.0),
            "league_avg_drtg": eff.get("league_avg_drtg", 95.0),
            "league_avg_pace": eff.get("league_avg_pace", 70.0),
            "league_avg_tov_pct": eff.get("league_avg_tov_pct", AVG_TOV_RATE),
        }
        sh = shrink_team_stats(stats_h, league, k=SHRINK_K)
        sa = shrink_team_stats(stats_a, league, k=SHRINK_K)

        seed = hash(
            f"{SECTOR}_{home_team}_{away_team}_{date.today().isoformat()}"
        ) & 0xFFFFFFFF
        rng = np.random.default_rng(seed=seed)

        sim = simulate_matchup_vectorized(
            sh, sa, league,
            hca_eff=eff.get("hca_eff", HCA_EFF_FALLBACK),
            n_sims=N_SIMS, rng=rng,
            poss_clip=POSSESSIONS_CLIP,
            pts_std=PTS_STD, avg_tov_rate=AVG_TOV_RATE,
        )

        eid = event_id or f"{SECTOR}_sim_{home_team}_{away_team}"
        self._margin_cache[eid] = sim["margin"]
        self._total_cache[eid] = sim["total"]

        sim["event_id"] = eid
        sim["confidence"] = smooth_confidence(
            min(stats_h["gp"], stats_a["gp"]), full_gp=FULL_SEASON_GP,
        )
        sim["stats_home"] = sh
        sim["stats_away"] = sa
        return sim

    def cover_probability(
        self, event_id: str, line: float, yes_is_underdog: bool = False,
    ) -> Optional[float]:
        margins = self._margin_cache.get(event_id)
        if margins is None:
            return None
        sim_mean = float(margins.mean())
        from scipy.stats import norm
        if yes_is_underdog:
            return float(norm.cdf((line - sim_mean) / SPREAD_SIGMA))
        return float(1.0 - norm.cdf((-line - sim_mean) / SPREAD_SIGMA))

    def total_probability(
        self, event_id: str, line: float, is_over: bool = True,
    ) -> Optional[float]:
        totals = self._total_cache.get(event_id)
        if totals is None:
            return None
        sim_mean = float(totals.mean())
        from scipy.stats import norm
        z = (line - sim_mean) / TOTAL_SIGMA
        return float(1.0 - norm.cdf(z)) if is_over else float(norm.cdf(z))

    async def predict_pair(
        self,
        market: PredictionMarket,
        sharp_odds: SharpOdds,
    ) -> Optional[ModelAgentPrediction]:
        sector = (market.sector or "").lower()
        if sector != SECTOR:
            return None

        team_a = (sharp_odds.outcome_a_label or market.team_home or "").lower().strip()
        team_b = (sharp_odds.outcome_b_label or market.team_away or "").lower().strip()

        sim = self.simulate_matchup(team_a, team_b, event_id=sharp_odds.event_id)
        if sim is None:
            return None

        prob_a = sim["prob_home"]
        avg_margin = float(sim["margin"].mean())
        avg_total = float(sim["total"].mean())

        return ModelAgentPrediction(
            event_id=sharp_odds.event_id,
            model_name=self.name,
            true_prob_a=prob_a,
            true_prob_b=1.0 - prob_a,
            true_prob_draw=None,
            confidence=sim["confidence"],
            weight=self.weight,
            sample_size=N_SIMS,
            notes=(
                f"{SECTOR}_sim={N_SIMS} margin={avg_margin:+.1f} "
                f"total={avg_total:.0f} pace={sim['pace']:.1f}"
            ),
        )

    def update(
        self, team_a: str, team_b: str, score_a: float, score_b: float,
        sector: str, event_date: Optional[str] = None,
    ) -> None:
        """No-op — state refreshed by scripts/seed_ncaab_efficiency.py --league womens."""
        return
