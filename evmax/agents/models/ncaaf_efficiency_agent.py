"""NcaafEfficiencyModelAgent (``ncaaf_efficiency_v2``) — NCAA-football win
probability from opponent-adjusted EPA + success rate with an FPI-mixed
preseason-prior ramp.

The CFB-specific piece (vs the NFL efficiency agent this is modeled on) is the
RAMP: college football has 12–13 game seasons, near-total roster turnover year to
year, and enormous schedule-strength variance, so in-season efficiency is
untrustworthy for the first ~3 weeks. The model therefore blends the
opponent-adjusted in-season rating with a preseason prior by a game-count
weight w(gp)=gp/(gp+k): week-0 games are 100% prior, and the prior phases out
toward mid-season — the SP+/FPI shrinkage the market leans on early.

v2 (2026-09-03) changed two things, both validated leak-free on three held-out
seasons (2023/2024/2025, scripts/backtest_ncaaf_v2.py; frozen constants fit on
2024 only):

  * The preseason prior is a 50/50 mix of the regressed prior-season EPA rating
    and ESPN's PRESEASON FPI (evmax/clients/cfb_fpi.py) — the market's edge
    over an EPA-only model sits in weeks 0–8 where the rating IS the prior, and
    play-by-play carries no offseason information. Teams without an FPI row
    fall back to the EPA-only prior.
  * Opponent-adjusted SUCCESS RATE is a second rating dimension. It was the
    only play-level signal that added held-out value on top of EPA;
    explosiveness, pass/rush splits, turnover splits, special teams, rest and
    tempo did not (see the memory/notes in CLAUDE.md).

  Standalone held-out Brier v1 → v2: 2023 0.1920 → 0.1853, 2024 0.1946 →
  0.1897, 2025 0.1917 → 0.1849. In the 85%-sharp blend the change is below
  the noise floor (≤0.4/1000) — the value is CLV-shaped (v2 predicts the
  open→close move better: slope +0.07 → +0.11), so promotion is judged on
  shadow CLV, not Brier. The model was renamed so the contamination filter
  dates every v1-priced row out of the promotion sample.

Margin model (per team, home minus away deltas of ramp-blended net ratings):
  Δepa     = net_epa(home) − net_epa(away)      [EPA/play, FPI-mixed prior]
  Δsr      = net_sr(home)  − net_sr(away)       [success rate, regressed prior]
  margin   = V2_EPA_PTS·Δepa + V2_SR_PTS·Δsr + V2_HOME_EDGE_PTS (0 if neutral)
  P(home)  = Φ(margin / SCORE_STDEV)

State is seed-driven (scripts/seed_ncaaf_efficiency.py, weekly during the
season); update() is a no-op because per-game EPA cannot be reconstructed from a
final score — same pattern as NFL/WNBA. A state file written by the v1 seed
(no ``schema_version: 2``) is refused fail-clear (returns None) rather than
silently priced with the v1 formula under the v2 name.
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
# margins are far more variable (talent gaps, blowouts).
SCORE_STDEV = 16.5            # CFB margin σ (wider than the NFL)
PLAYS_PER_TEAM_GAME = 70.0    # offensive scrimmage plays per team per game (FPI pts → EPA/play)
RAMP_K = 3.0                  # prior→in-season half-life in games (week 3 ≈ 50/50)
MIN_GP_FOR_INSEASON = 0       # even gp=0 predicts (100% prior); confidence scales

# v1 margin constants — kept for the v1 reference column in the backtests.
HOME_EDGE_PTS = 2.5           # v1: ~+2.5 pts home in CFB

# v2 margin constants — no-intercept OLS of actual margin on (home, Δepa, Δsr)
# over the 2024 walk-forward rows, rounded to the flat region of the fit
# (home 3.8 / epa 32.7 / sr 75.8 raw; every neighbour within 0.0005 Brier),
# then held out on 2023 + 2025. See scripts/backtest_ncaaf_v2.py --fit.
V2_HOME_EDGE_PTS = 3.5        # points of margin for the nominal home team
V2_EPA_PTS = 35.0             # points of margin per unit Δ net EPA/play
V2_SR_PTS = 70.0              # points of margin per unit Δ net success rate
FPI_PRIOR_SHARE = 0.5         # preseason prior = 0.5·regressed EPA + 0.5·FPI/plays

# Confidence ramps with the SMALLER team's game count. Below the ensemble's 0.45
# gate for the first game or two of teams with no usable prior, but a seeded
# prior keeps early-season confidence above the gate for established programs.
LOW_CONF_GP = 3
MID_CONF_GP = 6
HIGH_CONF_GP = 9


def _has_prior(stats: dict) -> bool:
    return bool(
        stats.get("off_epa_prior", 0.0)
        or stats.get("def_epa_prior", 0.0)
        or stats.get("fpi_prior") is not None
    )


class NcaafEfficiencyModelAgent(ModelAgent):
    """Opponent-adjusted EPA + success rate, FPI-mixed prior ramp → NCAAF win probability."""

    name = "ncaaf_efficiency_v2"
    weight = 0.55  # ensemble weight (set per-sector in SECTOR_WEIGHT_OVERRIDES)
    # The seed writes ncaaf_efficiency_state.json (STATE_PATH above); the model
    # NAME carries the version, the FILE does not. Without this the base class
    # would look for ncaaf_efficiency_v2_state.json and load nothing.
    state_filename = STATE_PATH.name

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

        sector_state = self._sector_state()
        if not E.state_has_v2_schema(sector_state):
            # A v1 seed has no success-rate ratings / FPI prior. Pricing it with
            # the v1 formula under the v2 name would contaminate the v2 shadow
            # sample, so fail clear (elo + form + sharp carry the blend).
            logger.warning(
                "ncaaf_efficiency_v2_state_schema_v1",
                schema_version=sector_state.get("schema_version", 1),
                hint="re-run scripts/seed_ncaaf_efficiency.py (writes schema_version 2)",
            )
            return None

        teams = sector_state.get("teams", {})
        if not teams:
            return None

        team_a = (sharp_odds.outcome_a_label or market.team_home or "").lower().strip()
        team_b = (sharp_odds.outcome_b_label or market.team_away or "").lower().strip()
        stats_a = self._resolve_team(teams, team_a)
        stats_b = self._resolve_team(teams, team_b)
        if not stats_a or not stats_b:
            return None

        gp_a = stats_a.get("gp", 0)
        gp_b = stats_b.get("gp", 0)
        min_gp = min(gp_a, gp_b)
        # A seeded prior alone is enough to clear the ensemble gate preseason;
        # a team with neither in-season games nor a prior is too thin to fire.
        if min_gp == 0 and not (_has_prior(stats_a) and _has_prior(stats_b)):
            return None

        # Ramp-blended component nets (in-season vs prior by each team's own gp).
        net_epa_a = E.blended_component(stats_a, RAMP_K, "epa", FPI_PRIOR_SHARE, PLAYS_PER_TEAM_GAME)
        net_epa_b = E.blended_component(stats_b, RAMP_K, "epa", FPI_PRIOR_SHARE, PLAYS_PER_TEAM_GAME)
        net_sr_a = E.blended_component(stats_a, RAMP_K, "sr")
        net_sr_b = E.blended_component(stats_b, RAMP_K, "sr")

        neutral = bool(getattr(market, "neutral_site", False))
        prob_a, margin = E.project_win_prob_v2(
            net_epa_a - net_epa_b,
            net_sr_a - net_sr_b,
            epa_pts=V2_EPA_PTS,
            sr_pts=V2_SR_PTS,
            home_edge_pts=V2_HOME_EDGE_PTS,
            score_stdev=SCORE_STDEV,
            neutral=neutral,
        )
        prob_b = 1.0 - prob_a

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
        fpi_flag = "".join(
            "y" if s.get("fpi_prior") is not None else "n" for s in (stats_a, stats_b)
        )
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
                f"epa={net_epa_a:+.3f}/{net_epa_b:+.3f} sr={net_sr_a:+.3f}/{net_sr_b:+.3f} "
                f"margin={margin:+.1f} gp={gp_a}/{gp_b} w_inseason={w:.2f} fpi={fpi_flag}"
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
