"""NhlXgModelAgent — NHL win probability from team 5v5 xG rates.

Hockey is a low-event sport (~6 goals/game) where actual goals are noisier than
the underlying chance creation. The most predictive single team-level signal in
public NHL analytics is 5v5 expected goals for/against per 60 minutes (xGF/60,
xGA/60), score-and-venue adjusted. MoneyPuck publishes these via its team CSVs
(situation=5on5).

This agent stores per-team xGF/60 and xGA/60 (and the actual goal rates as
secondary signals) and converts the matchup differential into a projected goal
margin → win probability via a normal CDF.

State is seed-driven via scripts/seed_nhl_xg.py — there is no incremental update
path because per-game xG cannot be reconstructed from a final score. Re-seed
weekly during the season (same pattern as NFL efficiency / WNBA efficiency).

Margin model:
  net_xg(team) = xGF_per_60 − xGA_per_60         (per-60-min net xG)
  margin = (net_xg_home − net_xg_away) × MIN_5V5_PER_GAME / 60 + HOME_EDGE_GOALS
  P(home) = Φ(margin / GOAL_STDEV)

5v5 only captures ~75-80% of goal scoring; special teams + empty net are
intentionally out of scope for v1. They land in v2 (nhl_special_teams_agent
+ goalie GSAx adjustment). Across an 82-game season opponent quality washes
out, so we use raw xG rates rather than running an explicit SoS subtraction.

State file: data/models/nhl_xg_state.json
  {
    "nhl": {
      "league_avg_xg_per_60": 2.50,
      "teams": {
        "boston bruins": {
          "abbrev": "BOS",
          "xgf_per_60": 2.71, "xga_per_60": 2.32,
          "gf_per_60": 2.85,  "ga_per_60":  2.20,
          "gp": 78
        },
        ...
      },
      "fetched_at": "2026-05-02",
      "season_start_year": 2025
    }
  }
"""

from __future__ import annotations

from typing import Optional

from evmax.agents.models.base import ModelAgent, ModelAgentPrediction
from evmax.models.market import PredictionMarket
from evmax.models.odds import SharpOdds

# NHL-tuned constants
HOME_EDGE_GOALS = 0.18       # ~+0.18 goal home advantage; NHL home WP ~55%
GOAL_STDEV = 2.05            # single-game goal-margin σ (empirical, NHL last 5 seasons)
MIN_5V5_PER_GAME = 50.0      # 5v5 minutes per team per game; remainder is special teams
MIN_GAMES = 10               # below this, xG rates haven't stabilized
LOW_CONF_GAMES = 25          # ~30% of season → moderate confidence
HIGH_CONF_GAMES = 50         # ~60% of season → full confidence


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


# NHL abbreviation → canonical lowercased team name. 32-team alignment as of
# the 2025-26 season (Arizona Coyotes relocated to Utah Mammoth for 2024-25;
# rebranded "Utah Mammoth" for 2025-26). MoneyPuck uses dotted variants for
# some teams (L.A, N.J, S.J, T.B); the seed script normalizes those to the
# standard 3-letter codes below before keying state.
NHL_ABBREV_TO_NAME: dict[str, str] = {
    "ANA": "anaheim ducks",
    "BOS": "boston bruins",
    "BUF": "buffalo sabres",
    "CAR": "carolina hurricanes",
    "CBJ": "columbus blue jackets",
    "CGY": "calgary flames",
    "CHI": "chicago blackhawks",
    "COL": "colorado avalanche",
    "DAL": "dallas stars",
    "DET": "detroit red wings",
    "EDM": "edmonton oilers",
    "FLA": "florida panthers",
    "LAK": "los angeles kings",
    "MIN": "minnesota wild",
    "MTL": "montreal canadiens",
    "NJD": "new jersey devils",
    "NSH": "nashville predators",
    "NYI": "new york islanders",
    "NYR": "new york rangers",
    "OTT": "ottawa senators",
    "PHI": "philadelphia flyers",
    "PIT": "pittsburgh penguins",
    "SEA": "seattle kraken",
    "SJS": "san jose sharks",
    "STL": "st. louis blues",
    "TBL": "tampa bay lightning",
    "TOR": "toronto maple leafs",
    "UTA": "utah mammoth",
    "VAN": "vancouver canucks",
    "VGK": "vegas golden knights",
    "WPG": "winnipeg jets",
    "WSH": "washington capitals",
}

# Reverse map: last word / nickname → canonical full name.
NHL_NICKNAME_TO_NAME: dict[str, str] = {}
for _abbr, _full in NHL_ABBREV_TO_NAME.items():
    NHL_NICKNAME_TO_NAME[_full.rsplit(" ", 1)[-1]] = _full
# A few multi-word nicknames Pinnacle/Kalshi sometimes ship verbatim
NHL_NICKNAME_TO_NAME["maple leafs"] = "toronto maple leafs"
NHL_NICKNAME_TO_NAME["red wings"] = "detroit red wings"
NHL_NICKNAME_TO_NAME["blue jackets"] = "columbus blue jackets"
NHL_NICKNAME_TO_NAME["golden knights"] = "vegas golden knights"

# MoneyPuck uses dotted abbrevs for two-word cities. Normalized to the
# 3-letter codes above by the seed script, but accept them at lookup time
# in case somebody runs against an old state file.
NHL_MONEYPUCK_ABBREV_ALIASES: dict[str, str] = {
    "L.A": "LAK",
    "N.J": "NJD",
    "S.J": "SJS",
    "T.B": "TBL",
}


class NhlXgModelAgent(ModelAgent):
    """Win probability from team 5v5 xGF/60 − xGA/60 differentials."""

    name = "nhl_xg"
    weight = 0.30  # base weight; overridden per-sector in SECTOR_WEIGHT_OVERRIDES

    def _sector_state(self) -> dict:
        return self._state.get("nhl", {})

    def _resolve_team(self, teams: dict, team: str) -> Optional[dict]:
        """Resolve a team identifier (full name, last word, abbrev) to its stats dict."""
        if not team:
            return None
        team = team.lower().strip()
        if team in teams:
            return teams[team]

        # Abbreviation lookup (also handles MoneyPuck dotted variants)
        upper = team.upper()
        if upper in NHL_MONEYPUCK_ABBREV_ALIASES:
            upper = NHL_MONEYPUCK_ABBREV_ALIASES[upper]
        if upper in NHL_ABBREV_TO_NAME:
            full = NHL_ABBREV_TO_NAME[upper]
            if full in teams:
                return teams[full]

        # Multi-word nickname check first ("maple leafs", "blue jackets")
        if team in NHL_NICKNAME_TO_NAME:
            full = NHL_NICKNAME_TO_NAME[team]
            if full in teams:
                return teams[full]

        # Last-word lookup ("bruins" → "boston bruins")
        if " " in team:
            last = team.rsplit(" ", 1)[-1]
            if last in NHL_NICKNAME_TO_NAME:
                full = NHL_NICKNAME_TO_NAME[last]
                if full in teams:
                    return teams[full]

        # Substring fallback (handles "ny rangers", "la kings", etc.)
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
        if sector != "nhl":
            return None

        sector_state = self._sector_state()
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
        if gp_a < MIN_GAMES or gp_b < MIN_GAMES:
            return None

        # Net xG/60 differential: high xGF + low xGA both push net up.
        net_a = stats_a["xgf_per_60"] - stats_a["xga_per_60"]
        net_b = stats_b["xgf_per_60"] - stats_b["xga_per_60"]
        diff_per_60 = net_a - net_b

        # Convert per-60 differential to projected goal margin over the 5v5
        # portion of a game; HOME_EDGE_GOALS captures the residual home effect
        # (rink familiarity, last change, no travel).
        margin = diff_per_60 * (MIN_5V5_PER_GAME / 60.0) + HOME_EDGE_GOALS

        prob_a = _normal_cdf(margin / GOAL_STDEV)
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
                f"net_xg/60={net_a:+.2f}/{net_b:+.2f} "
                f"margin={margin:+.2f}g gp={gp_a}/{gp_b}"
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
        """No-op: NHL xG stats are recomputed from MoneyPuck via the seed script.

        Per-game xG cannot be reconstructed from just the final score, so
        weekly state refresh runs `python scripts/seed_nhl_xg.py` rather than
        incremental updates. Same pattern as NFL efficiency and WNBA.
        """
        return
