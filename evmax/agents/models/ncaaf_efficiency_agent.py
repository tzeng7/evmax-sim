"""NcaafEfficiencyModelAgent — NCAA-football win probability from opponent-adjusted
EPA with a preseason-prior ramp.

The CFB-specific piece (vs the NFL efficiency agent this is modeled on) is the
RAMP: college football has 12–13 game seasons, near-total roster turnover year to
year, and enormous schedule-strength variance, so in-season efficiency is
untrustworthy for the first ~3 weeks. The model therefore blends the
opponent-adjusted in-season EPA rating with a regressed preseason prior by a
game-count weight w(gp)=gp/(gp+k): week-0 games are 100% prior, and the prior
phases out toward mid-season — the SP+/FPI shrinkage the market leans on early.

Margin model (per team net EPA/play):
  net(t)   = w·(off_epa_adj − def_epa_adj) + (1−w)·(off_epa_prior − def_epa_prior)
  margin   = (net_home − net_away) · PLAYS_PER_TEAM_GAME + HOME_EDGE_PTS(0 if neutral)
  P(home)  = Φ(margin / SCORE_STDEV)

State is seed-driven (scripts/seed_ncaaf_efficiency.py, weekly during the
season); update() is a no-op because per-game EPA cannot be reconstructed from a
final score — same pattern as NFL/WNBA. State file:
data/models/ncaaf_efficiency_state.json (see the seed script for the schema).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import structlog

from evmax.agents.models import _cfb_efficiency as E
from evmax.agents.models.base import ModelAgent, ModelAgentPrediction
from evmax.matching.normalizer import NameNormalizer
from evmax.models.market import PredictionMarket
from evmax.models.odds import SharpOdds

logger = structlog.get_logger(__name__)

STATE_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "models" / "ncaaf_efficiency_state.json"
)

# CFB-tuned constants. SCORE_STDEV is much wider than the NFL's 13.5 — college
# margins are far more variable (talent gaps, blowouts). All swept in
# scripts/backtest_ncaaf_efficiency.py; these are the pre-backtest defaults.
HOME_EDGE_PTS = 2.5           # ~+2.5 pts home in CFB
SCORE_STDEV = 16.5            # CFB margin σ (wider than the NFL)
PLAYS_PER_TEAM_GAME = 70.0    # offensive scrimmage plays per team per game
RAMP_K = 3.0                  # prior→in-season half-life in games (week 3 ≈ 50/50)
MIN_GP_FOR_INSEASON = 0       # even gp=0 predicts (100% prior); confidence scales

# Confidence ramps with the SMALLER team's game count. Below the ensemble's 0.45
# gate for the first game or two of teams with no usable prior, but a seeded
# prior keeps early-season confidence above the gate for established programs.
LOW_CONF_GP = 3
MID_CONF_GP = 6
HIGH_CONF_GP = 9


class NcaafEfficiencyModelAgent(ModelAgent):
    """Opponent-adjusted EPA + preseason-prior ramp → NCAAF win probability."""

    name = "ncaaf_efficiency"
    weight = 0.55  # ensemble weight (set per-sector in SECTOR_WEIGHT_OVERRIDES)

    def __init__(self) -> None:
        super().__init__()
        self._normalizer = NameNormalizer("ncaaf")

    def _sector_state(self) -> dict:
        return self._state.get("ncaaf", {})

    def _resolve_team(self, teams: dict, name: str) -> Optional[dict]:
        """Resolve a Pinnacle/Kalshi label to a team row.

        College-safe (no bare last-word/mascot fallback — 130+ FBS teams share
        mascots). Order: exact key, normalized key (alias map → ESPN location),
        then a longest-prefix match with ambiguity → None.
        """
        n = (name or "").lower().strip()
        if not n:
            return None
        if n in teams:
            return teams[n]
        try:
            normed = self._normalizer.normalize(n)
        except Exception:  # noqa: BLE001
            normed = None
        if normed and normed in teams:
            return teams[normed]
        if normed:
            n = normed
            if n in teams:
                return teams[n]
        # Longest-prefix disambiguation; refuse to guess on ties.
        cands = [k for k in teams if k.startswith(n + " ") or n.startswith(k + " ")]
        if len(cands) == 1:
            return teams[cands[0]]
        return None

    async def predict_pair(
        self,
        market: PredictionMarket,
        sharp_odds: SharpOdds,
    ) -> Optional[ModelAgentPrediction]:
        if (market.sector or "").lower() != "ncaaf":
            return None

        if E.state_is_stale_for_today(self._state):
            logger.warning(
                "ncaaf_efficiency_stale",
                source_season=self._sector_state().get("source_season"),
                hint="re-run scripts/seed_ncaaf_efficiency.py",
            )
            return None

        teams = self._sector_state().get("teams", {})
        if not teams:
            return None

        team_a = (sharp_odds.outcome_a_label or market.team_home or "").lower().strip()
        team_b = (sharp_odds.outcome_b_label or market.team_away or "").lower().strip()
        stats_a = self._resolve_team(teams, team_a)
        stats_b = self._resolve_team(teams, team_b)
        if not stats_a or not stats_b:
            return None

        # net EPA/play blends in-season with the regressed prior by the ramp.
        net_a = E.blended_net(stats_a, k=RAMP_K)
        net_b = E.blended_net(stats_b, k=RAMP_K)

        neutral = bool(getattr(market, "neutral_site", False))
        prob_a, margin = E.project_win_prob(
            net_a, net_b,
            plays_per_team=PLAYS_PER_TEAM_GAME,
            home_edge_pts=HOME_EDGE_PTS,
            score_stdev=SCORE_STDEV,
            neutral=neutral,
        )
        prob_b = 1.0 - prob_a

        gp_a = stats_a.get("gp", 0)
        gp_b = stats_b.get("gp", 0)
        min_gp = min(gp_a, gp_b)
        has_prior_a = stats_a.get("off_epa_prior", 0.0) or stats_a.get("def_epa_prior", 0.0)
        has_prior_b = stats_b.get("off_epa_prior", 0.0) or stats_b.get("def_epa_prior", 0.0)
        # A seeded prior alone is enough to clear the ensemble gate preseason;
        # a team with neither in-season games nor a prior is too thin to fire.
        if min_gp == 0 and not (has_prior_a and has_prior_b):
            return None

        if min_gp >= HIGH_CONF_GP:
            confidence = 0.85
        elif min_gp >= MID_CONF_GP:
            confidence = 0.72
        elif min_gp >= LOW_CONF_GP:
            confidence = 0.60
        else:
            # Preseason / early: prior-driven. Above the 0.45 gate but modest.
            confidence = 0.52

        w = E.prior_ramp_weight(min_gp, RAMP_K)
        return ModelAgentPrediction(
            event_id=sharp_odds.event_id,
            model_name=self.name,
            true_prob_a=prob_a,
            true_prob_b=prob_b,
            true_prob_draw=None,
            confidence=confidence,
            weight=self.weight,
            sample_size=min_gp,
            notes=(
                f"net={net_a:+.3f}/{net_b:+.3f} margin={margin:+.1f} "
                f"gp={gp_a}/{gp_b} w_inseason={w:.2f}"
                + (" neutral" if neutral else "")
            ),
        )

    def update(self, team_a, team_b, score_a, score_b, sector, event_date=None) -> None:
        """No-op: EPA is recomputed from play-by-play via the seed script.

        Per-game EPA cannot be reconstructed from a final score, so weekly state
        refresh runs `python scripts/seed_ncaaf_efficiency.py`. Same pattern as
        the NFL and WNBA efficiency agents.
        """
        return
