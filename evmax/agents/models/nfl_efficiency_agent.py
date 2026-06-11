"""NflEfficiencyModelAgent — NFL win probability from opponent-adjusted EPA.

Football is drive-based, not minute-uniform — Poisson is the wrong shape, and
raw W/L is noisy at 17 games/team/season. Expected Points Added per play
(EPA/play), opponent-adjusted, is the most predictive single signal in public
NFL analytics. This agent stores per-team opponent-adjusted EPA on offense
and defense (plus success rate, explosive rate, and red-zone TD rate as
v2-ready secondary signals) and converts the matchup differential into a
projected margin → win probability via a normal CDF.

State is seed-driven via scripts/seed_nfl_efficiency.py (weekly refresh during
the season). Same pattern as WNBA — `update()` is a no-op because per-game
EPA cannot be reconstructed from a final score; the seed script re-walks PBP.

Margin model:
  net_epa(team) = off_epa_adj − def_epa_adj
    (good offense + good defense → high net; def_epa_adj is "EPA allowed
     relative to league avg" — lower is better, so subtracting flips the sign)
  margin = (net_epa_home − net_epa_away) × PLAYS_PER_TEAM_GAME + HOME_EDGE_PTS
  P(home) = Φ(margin / SCORE_STDEV)

State file: data/models/nfl_efficiency_state.json
  {
    "nfl": {
      "league_avg_pts_per_game": 22.5,
      "teams": {
        "kansas city chiefs": {
          "off_epa_adj": 0.08, "def_epa_adj": -0.04,
          "off_success_rate": 0.48, "def_success_rate": 0.43,
          "off_explosive_rate": 0.12, "def_explosive_rate": 0.09,
          "rz_td_rate_off": 0.62, "rz_td_rate_def": 0.51,
          "gp": 17, "abbrev": "KC"
        },
        ...
      },
      "fetched_at": "2026-05-01"
    }
  }
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from evmax.agents.models.base import ModelAgent, ModelAgentPrediction
from evmax.models.market import PredictionMarket
from evmax.models.odds import SharpOdds

STATE_PATH = Path(__file__).resolve().parents[3] / "data" / "models" / "nfl_efficiency_state.json"

# NFL-tuned constants
HOME_EDGE_PTS = 2.0          # ~+2 pts home, consistent with the +48 Elo home advantage
SCORE_STDEV = 13.5           # NFL margin σ — empirical, used by FTE-style models
PLAYS_PER_TEAM_GAME = 64.0   # offensive plays per team per game (league avg)
MIN_GAMES = 6                # ~4 weeks; below this confidence collapses
LOW_CONF_GAMES = 9           # ~9 games → moderate confidence
HIGH_CONF_GAMES = 13         # ~13 games → full confidence


def _normal_cdf(x: float) -> float:
    """Standard normal CDF (Abramowitz & Stegun)."""
    import math
    if x < -6:
        return 0.0
    if x > 6:
        return 1.0
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = 0.3989422804014327
    p = d * math.exp(-x * x / 2.0) * t * (
        0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.8212560 + t * 1.3302744)))
    )
    return 1.0 - p if x > 0 else p


# NFL abbreviation → canonical lowercased name. Used by both seeding and
# resolution so the same key space is used in state and in market lookups.
NFL_ABBREV_TO_NAME: dict[str, str] = {
    "ARI": "arizona cardinals",
    "ATL": "atlanta falcons",
    "BAL": "baltimore ravens",
    "BUF": "buffalo bills",
    "CAR": "carolina panthers",
    "CHI": "chicago bears",
    "CIN": "cincinnati bengals",
    "CLE": "cleveland browns",
    "DAL": "dallas cowboys",
    "DEN": "denver broncos",
    "DET": "detroit lions",
    "GB": "green bay packers",
    "HOU": "houston texans",
    "IND": "indianapolis colts",
    "JAX": "jacksonville jaguars",
    "KC": "kansas city chiefs",
    "LV": "las vegas raiders",
    "LAC": "los angeles chargers",
    "LA": "los angeles rams",
    "MIA": "miami dolphins",
    "MIN": "minnesota vikings",
    "NE": "new england patriots",
    "NO": "new orleans saints",
    "NYG": "new york giants",
    "NYJ": "new york jets",
    "PHI": "philadelphia eagles",
    "PIT": "pittsburgh steelers",
    "SF": "san francisco 49ers",
    "SEA": "seattle seahawks",
    "TB": "tampa bay buccaneers",
    "TEN": "tennessee titans",
    "WAS": "washington commanders",
}

# Reverse map: last-word/nickname → canonical name for fuzzy resolution
NFL_NICKNAME_TO_NAME: dict[str, str] = {}
for _abbr, _full in NFL_ABBREV_TO_NAME.items():
    # Last word of full name (e.g. "chiefs", "bills", "49ers")
    NFL_NICKNAME_TO_NAME[_full.rsplit(" ", 1)[-1]] = _full


class NflEfficiencyModelAgent(ModelAgent):
    """Win probability from opponent-adjusted NFL EPA (off + def per play)."""

    name = "nfl_efficiency"
    weight = 0.35  # ensemble weight (overridden per-sector in SECTOR_WEIGHT_OVERRIDES)

    def _sector_state(self) -> dict:
        return self._state.get("nfl", {})

    def _resolve_team(self, teams: dict, team: str) -> Optional[dict]:
        """Resolve a team identifier (full name, last word, or abbreviation) to its stats dict."""
        if not team:
            return None
        team = team.lower().strip()
        if team in teams:
            return teams[team]

        # Abbreviation lookup
        upper = team.upper()
        if upper in NFL_ABBREV_TO_NAME:
            full = NFL_ABBREV_TO_NAME[upper]
            if full in teams:
                return teams[full]

        # Last-word / nickname lookup ("chiefs" → "kansas city chiefs")
        if " " in team:
            last = team.rsplit(" ", 1)[-1]
            if last in NFL_NICKNAME_TO_NAME:
                full = NFL_NICKNAME_TO_NAME[last]
                if full in teams:
                    return teams[full]
        elif team in NFL_NICKNAME_TO_NAME:
            full = NFL_NICKNAME_TO_NAME[team]
            if full in teams:
                return teams[full]

        # Substring fallback (handles "ny giants", "la rams", etc.)
        for key, val in teams.items():
            if team in key or key in team:
                return val
        return None

    async def predict_pair(
        self,
        market: PredictionMarket,
        sharp_odds: SharpOdds,
    ) -> Optional[ModelAgentPrediction]:
        sector = (market.sector or "").lower()
        if sector != "nfl":
            return None

        sector_state = self._sector_state()
        teams = sector_state.get("teams", {})
        if not teams:
            return None

        # team_a is the home side per Pinnacle convention (outcome_a_label)
        team_a = (sharp_odds.outcome_a_label or market.team_home or "").lower().strip()
        team_b = (sharp_odds.outcome_b_label or market.team_away or "").lower().strip()

        stats_a = self._resolve_team(teams, team_a)
        stats_b = self._resolve_team(teams, team_b)
        if not stats_a or not stats_b:
            return None

        gp_a = stats_a.get("gp", 0)
        gp_b = stats_b.get("gp", 0)
        if gp_a < MIN_GAMES or gp_b < MIN_GAMES:
            return None

        # Opponent-adjusted EPA already encodes performance vs league average.
        # Net = off − def: high net means good offense AND good defense
        # (def_epa_adj is "EPA allowed relative to league" — lower is better,
        # so subtracting flips the sign correctly).
        net_a = stats_a["off_epa_adj"] - stats_a["def_epa_adj"]
        net_b = stats_b["off_epa_adj"] - stats_b["def_epa_adj"]
        epa_diff_per_play = net_a - net_b

        # Convert per-play differential to projected point margin.
        # Each team gets PLAYS_PER_TEAM_GAME plays; the differential applies
        # to A's scoring on offense and to A's defense limiting B equally,
        # so we multiply by PLAYS_PER_TEAM_GAME (not 2x — symmetric).
        margin = epa_diff_per_play * PLAYS_PER_TEAM_GAME + HOME_EDGE_PTS

        prob_a = _normal_cdf(margin / SCORE_STDEV)
        prob_a = max(0.02, min(0.98, prob_a))
        prob_b = 1.0 - prob_a

        min_gp = min(gp_a, gp_b)
        if min_gp >= HIGH_CONF_GAMES:
            confidence = 0.85
        elif min_gp >= LOW_CONF_GAMES:
            confidence = 0.70
        else:
            confidence = 0.55

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
                f"net_epa={net_a:+.3f}/{net_b:+.3f} "
                f"margin={margin:+.1f} gp={gp_a}/{gp_b}"
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
        """No-op: NFL EPA stats are recomputed from PBP via the seed script.

        Per-game EPA cannot be reconstructed from just the final score, so
        weekly state refresh runs `python scripts/seed_nfl_efficiency.py`
        rather than incremental updates. Same pattern as WNBA efficiency.
        """
        return
