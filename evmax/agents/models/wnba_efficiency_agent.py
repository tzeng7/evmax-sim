"""WNBAEfficiencyModelAgent — WNBA win probability from team efficiency ratings.

Mirrors the structure of the NBA EfficiencyModelAgent but operates only on
WNBA data. The two agents never share code paths or state files — this is
intentional so NBA tuning never risks disturbing WNBA and vice versa.

Inputs per team (seeded + incrementally refreshed by scripts/seed_wnba_efficiency.py):
  - ORTG: points scored per 100 possessions
  - DRTG: points allowed per 100 possessions
  - Pace: possessions per 40 minutes (WNBA game length)
  - eFG%: (FGM + 0.5 × 3PM) / FGA
  - TOV%: turnovers per 100 possessions
  - OREB%: offensive rebounds / (own OREB + opponent DREB)
  - FTr: FTA / FGA

Projected margin:
  pace          = (pace_home + pace_away) / 2
  possessions   = pace
  home_pts      = ortg_home_factor × drtg_away_factor × league_avg_ortg × pace / 100
  away_pts      = ortg_away_factor × drtg_home_factor × league_avg_ortg × pace / 100
  margin        = home_pts − away_pts + HOME_EDGE_PTS

Win probability:
  P(home) = Φ(margin / SCORE_STDEV)

Constants are WNBA-specific (see below) — shorter games (40 min vs 48),
slower pace (~82 poss vs ~100), different home edge. Do not copy NBA values.

State file: data/models/wnba_efficiency_state.json (auto-resolved by base class
from the `name` attribute). Shape matches the NBA efficiency state for
operator familiarity, but the two files live independently on disk.

NBA is never read, never modified, never referenced by this agent.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Optional

import structlog

from evmax.agents.models.base import ModelAgent, ModelAgentPrediction
from evmax.models.market import PredictionMarket
from evmax.models.odds import SharpOdds

logger = structlog.get_logger(__name__)

# WNBA-specific constants — calibrated separately from NBA.
# Home edge: 2025 WNBA home win rate was 56.7% (weaker than NBA's ~59%).
#   56.7% ≈ ~2.6 pts of margin at σ=12.5.
# Score stdev: slightly higher than NBA because 40-game sample → noisier team
#   totals, and lower pace amplifies per-possession variance.
# Min games: 12 is ~30% of a 40-game WNBA season — the point at which ORTG/DRTG
#   estimates have stabilised enough to trust.
HOME_EDGE_PTS = 2.6
SCORE_STDEV = 12.5
# Lowered from 12 → 4 on 2026-05-30: empirical-Bayes shrinkage (see
# SHRINK_K below) handles the small-sample noise that MIN_GAMES used to
# guard against. With shrinkage active the model contributes lightly
# from week 1 instead of going dark for the first 3+ weeks of each season.
MIN_GAMES = 4

# Empirical-Bayes pseudo-count for shrinking team ORTG/DRTG/pace toward
# the league average. shrunk = (gp · raw + k · league_avg) / (gp + k).
# k=8 means at gp=8 we're 50/50 raw-vs-prior; at gp=24 raw dominates 75/25.
# Tuned to be aggressive early-season (when ratings are noisy from roster
# turnover + small samples) and naturally relax as the season progresses.
SHRINK_K = 8.0

# Months during which the WNBA regular season + playoffs can run.
# Outside this window (Nov-Apr) there are no games to predict, so the
# staleness guard relaxes and prior-year state is treated as fresh enough.
_WNBA_SEASON_MONTHS = range(5, 11)


def shrink_team_stats(stats: dict, league: dict, k: float = SHRINK_K) -> dict:
    """Empirical-Bayes shrinkage of team efficiency stats toward league means.

    With few games the team's raw ORTG/DRTG/pace are noisy; this pulls them
    toward the league average by a factor that decays as gp grows. Returns a
    new dict (does not mutate `stats`). Keys not in `league` pass through
    unchanged.

    league must carry: league_avg_ortg, league_avg_drtg, league_avg_pace, and
    optionally league_avg_tov_pct (defaults to 0.17 for WNBA when missing).
    """
    gp = stats.get("gp", 0)
    if gp <= 0:
        return dict(stats)
    weight = gp / (gp + k)   # weight on team's own data
    prior = 1.0 - weight     # weight on league prior

    def _shrunk(val: float, prior_val: float) -> float:
        return weight * val + prior * prior_val

    league_tov = league.get("league_avg_tov_pct", 0.17)
    out = dict(stats)
    out["ortg"] = _shrunk(stats.get("ortg", league["league_avg_ortg"]), league["league_avg_ortg"])
    out["drtg"] = _shrunk(stats.get("drtg", league["league_avg_drtg"]), league["league_avg_drtg"])
    out["pace"] = _shrunk(stats.get("pace", league["league_avg_pace"]), league["league_avg_pace"])
    if "tov_pct" in stats:
        out["tov_pct"] = _shrunk(stats["tov_pct"], league_tov)
    return out


def smooth_confidence(min_gp: int) -> float:
    """Ramp confidence from 0.40 (gp=0) to 0.80 (gp≥40).

    Replaces the prior step thresholds (>=30 → 0.80, >=20 → 0.70, else 0.55)
    so confidence tracks sample size smoothly instead of jumping at arbitrary
    boundaries. Clears the ensemble's 0.45 gate at gp >= 6.
    """
    return 0.40 + 0.40 * min(min_gp / 40.0, 1.0)


def state_is_stale_for_today(state: dict, today: Optional[date] = None) -> bool:
    """Return True when the state's source_season is older than the current WNBA season.

    Prevents the May-2026-using-2025-ratings failure mode that produced
    +24pp chalk bias on the 2026 opening month. During offseason (Nov-Apr)
    we treat last year's state as fresh, since there are no games to score.
    """
    today = today or date.today()
    if today.month not in _WNBA_SEASON_MONTHS:
        return False
    raw = state.get("source_season")
    if raw is None:
        return False
    try:
        source_year = int(raw)
    except (TypeError, ValueError):
        return False
    return source_year < today.year

# League-average fallbacks (used only when the state file has no seeded data).
# Real values get overwritten by the seed script from 2025 box scores.
LEAGUE_AVG_ORTG_FALLBACK = 104.0
LEAGUE_AVG_DRTG_FALLBACK = 104.0
LEAGUE_AVG_PACE_FALLBACK = 82.0


def _normal_cdf(x: float) -> float:
    """Standard normal CDF (Abramowitz & Stegun approximation)."""
    if x < -6:
        return 0.0
    if x > 6:
        return 1.0
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = 0.3989422804014327  # 1/sqrt(2π)
    p = d * math.exp(-x * x / 2.0) * t * (
        0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.8212560 + t * 1.3302744)))
    )
    return 1.0 - p if x > 0 else p


class WNBAEfficiencyModelAgent(ModelAgent):
    """Win probability from WNBA team efficiency ratings.

    Only fires for sector == "wnba". For all other sectors (including "nba")
    `predict_pair` returns None immediately — the NBA efficiency model is an
    entirely separate agent with its own state file.
    """

    name = "wnba_efficiency"
    weight = 0.30   # Slightly lower than NBA's 0.40 initially — tunable once
                    # walk-forward results confirm the model's contribution.

    async def predict_pair(
        self,
        market: PredictionMarket,
        sharp_odds: SharpOdds,
    ) -> Optional[ModelAgentPrediction]:
        sector = (market.sector or "").lower()
        if sector != "wnba":
            return None

        if state_is_stale_for_today(self._state):
            logger.warning(
                "wnba_efficiency_stale_source_season",
                source_season=self._state.get("source_season"),
                current_year=date.today().year,
                hint="re-run scripts/seed_wnba_efficiency.py --year <current>",
            )
            return None

        teams: dict = self._state.get("teams", {})
        if not teams:
            return None

        team_a = (sharp_odds.outcome_a_label or market.team_home or "").lower().strip()
        team_b = (sharp_odds.outcome_b_label or market.team_away or "").lower().strip()

        stats_a = self._resolve_team(teams, team_a)
        stats_b = self._resolve_team(teams, team_b)

        if not stats_a or not stats_b:
            return None

        if stats_a.get("gp", 0) < MIN_GAMES or stats_b.get("gp", 0) < MIN_GAMES:
            return None

        league_ortg = self._state.get("league_avg_ortg", LEAGUE_AVG_ORTG_FALLBACK)
        league_drtg = self._state.get("league_avg_drtg", LEAGUE_AVG_DRTG_FALLBACK)
        league_pace = self._state.get("league_avg_pace", LEAGUE_AVG_PACE_FALLBACK)
        league = {
            "league_avg_ortg": league_ortg,
            "league_avg_drtg": league_drtg,
            "league_avg_pace": league_pace,
        }
        # Empirical-Bayes shrinkage: at gp=8 both halves carry equal weight,
        # by gp=24 the team's own data dominates 75/25. Prevents the raw-ratings
        # explosion that drove +24pp chalk bias in May 2026 (see MEMORY).
        stats_a = shrink_team_stats(stats_a, league)
        stats_b = shrink_team_stats(stats_b, league)

        # Offensive / defensive efficiency factors relative to league average
        off_factor_a = stats_a["ortg"] / league_ortg
        off_factor_b = stats_b["ortg"] / league_ortg
        def_factor_a = stats_a["drtg"] / league_drtg
        def_factor_b = stats_b["drtg"] / league_drtg

        expected_pace = (stats_a["pace"] + stats_b["pace"]) / 2.0
        possessions = expected_pace   # pace is already per-40-min in WNBA

        proj_pts_a = off_factor_a * def_factor_b * league_ortg * possessions / 100.0
        proj_pts_b = off_factor_b * def_factor_a * league_ortg * possessions / 100.0

        margin = proj_pts_a - proj_pts_b + HOME_EDGE_PTS
        prob_a = _normal_cdf(margin / SCORE_STDEV)
        prob_a = max(0.02, min(0.98, prob_a))
        prob_b = 1.0 - prob_a

        min_gp = min(stats_a["gp"], stats_b["gp"])
        confidence = smooth_confidence(min_gp)

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
                f"ORTG={stats_a['ortg']:.1f}/{stats_b['ortg']:.1f} "
                f"DRTG={stats_a['drtg']:.1f}/{stats_b['drtg']:.1f} "
                f"eFG={stats_a.get('efg', 0):.3f}/{stats_b.get('efg', 0):.3f} "
                f"TOV={stats_a.get('tov_pct', 0):.3f}/{stats_b.get('tov_pct', 0):.3f} "
                f"margin={margin:+.1f} pace={expected_pace:.1f}"
            ),
        )

    @staticmethod
    def _resolve_team(teams: dict, team: str) -> Optional[dict]:
        """Resolve a team name (possibly aliased) to the stats dict.

        Handles: "aces", "las vegas aces", "Aces", "LV Aces" — all map to the
        canonical short key used in the alias YAML (e.g. "aces").
        """
        team = team.lower().strip()
        if team in teams:
            return teams[team]
        # Try last word
        if " " in team:
            last = team.rsplit(" ", 1)[-1]
            if last in teams:
                return teams[last]
        # Prefix / suffix / substring match against stored full names
        for key, val in teams.items():
            full = val.get("full_name", "")
            if (team.startswith(key) or key.startswith(team)
                    or team in full or (full and full.endswith(team))):
                return val
        return None

    def update(
        self,
        team_a: str,
        team_b: str,
        score_a: float,
        score_b: float,
        sector: str,
        event_date: Optional[str] = None,
    ) -> None:
        """No-op — efficiency stats are refreshed in bulk by the seed script.

        Reasoning: a single (score_a, score_b) tuple doesn't carry the FGA/
        FTA/TO/OREB box-score detail needed to update ORTG/DRTG/pace. The
        seed script re-walks ESPN summaries, so running it periodically
        (weekly, or after each day's games) is how WNBA data flows in.
        """
        return
