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

Preseason-prior ramp (2026-09-22, the NCAAF-v2 idea applied to hockey):
each team rate is blended with a REGRESSED prior-season rate by games played,

  prior_reg = lg + PRIOR_REGRESS_RHO · (prior − lg)       (lg = prior-season league avg)
  rate      = (gp · in_season + PRIOR_RAMP_K · prior_reg) / (gp + PRIOR_RAMP_K)

with PRIOR_RAMP_K=20 and PRIOR_REGRESS_RHO=0.7. At gp=0 the rate IS the
regressed prior (the model fires on opening night instead of going dark for
the ~3 weeks MIN_GAMES used to blank it); at gp=20 it is 50/50; at gp=82 the
prior keeps ~20%. Confidence is 0.70 until min(gp) reaches HIGH_CONF_GAMES,
then 0.85. A point-in-time walk-forward over MoneyPuck game logs (weekly
reseed cutoffs, prior-season rates only) validated the ramp together with
the elo 0.15 / form 0 blend: model-side Brier +5.7/1000 on the 2014-21 fit
(z 7.7), +5.6/1000 on the 2022-24 confirm (z 5.1), +1.6/1000 on the 2025
holdout, and +16.4/1000 over the first six weeks of a season (z 6.7).

Staleness guard: the in-season block carries `season_start_year`. When it is
the PREVIOUS season relative to the game date (NHL seasons roll over on
1 September), the block is last season's final ratings — it is used only as
the regressed prior (gp treated as 0), never fired as current-season ratings
at full confidence. A block two or more seasons stale returns None.

State file: data/models/nhl_xg_state.json (written by scripts/seed_nhl_xg.py)
  {
    "nhl": {
      "schema_version": 2,
      "season_start_year": 2026,           # season of the in-season block
      "league_avg_xg_per_60": 2.50,        # None when prior-only
      "teams": {                           # {} before the season's first game
        "boston bruins": {
          "abbrev": "BOS",
          "xgf_per_60": 2.71, "xga_per_60": 2.32,
          "gf_per_60": 2.85,  "ga_per_60":  2.20,
          "gp": 12
        },
        ...
      },
      "prior": {                           # regressed PREVIOUS-season rates
        "season_start_year": 2025,
        "league_avg_xg_per_60": 2.48,
        "regress_rho": 0.7,
        "teams": {
          "boston bruins": {"xgf_per_60": ..., "xga_per_60": ...,   # regressed
                            "raw_xgf_per_60": ..., "raw_xga_per_60": ..., "gp": 82},
          ...
        }
      },
      "fetched_at": "2026-09-22"
    }
  }
A legacy state without a `prior` block (and not stale) keeps the original
behaviour: raw in-season rates, MIN_GAMES gate, 0.55/0.70/0.85 confidence.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from evmax.agents.models._team_lookup import identity_team_key, resolve_team_key
from evmax.agents.models.base import ModelAgent, ModelAgentPrediction
from evmax.models.market import PredictionMarket
from evmax.models.odds import SharpOdds

# NHL-tuned constants
HOME_EDGE_GOALS = 0.18       # ~+0.18 goal home advantage; NHL home WP ~55%
GOAL_STDEV = 2.05            # single-game goal-margin σ (empirical, NHL last 5 seasons)
MIN_5V5_PER_GAME = 50.0      # 5v5 minutes per team per game; remainder is special teams
MIN_GAMES = 10               # legacy (no-prior) path: below this, raw rates haven't stabilized
LOW_CONF_GAMES = 25          # ~30% of season → moderate confidence
HIGH_CONF_GAMES = 50         # ~60% of season → full confidence

# Preseason-prior ramp — the exact constants the walk-forward validated
# (see the module docstring). Changing either needs a new walk-forward.
PRIOR_RAMP_K = 20.0          # gp at which in-season and prior weigh 50/50
PRIOR_REGRESS_RHO = 0.7      # share of the prior season's deviation from the league mean kept
RAMP_CONFIDENCE = 0.70       # confidence while min(gp) < HIGH_CONF_GAMES on the ramp path

# NHL seasons are labelled by their START year and roll over on this month:
# a September game (the 2026-27 opener is 2026-09-29) belongs to the new season.
SEASON_ROLLOVER_MONTH = 9


def nhl_season_for(d: date) -> int:
    """Start year of the NHL season a game on `d` belongs to."""
    return d.year if d.month >= SEASON_ROLLOVER_MONTH else d.year - 1


def regress_prior_rate(prior: float, league_avg: float, rho: float = PRIOR_REGRESS_RHO) -> float:
    """Shrink a prior-season rate toward that season's league average."""
    return league_avg + rho * (prior - league_avg)


def ramp_rate(in_season: Optional[float], gp: int, prior_reg: float, k: float = PRIOR_RAMP_K) -> float:
    """Blend an in-season rate with the regressed prior by games played.

    gp=0 (or no in-season rate) returns the regressed prior exactly.
    """
    if gp <= 0 or in_season is None:
        return prior_reg
    return (gp * in_season + k * prior_reg) / (gp + k)


# Shared Abramowitz & Stegun normal CDF (single source in models_ml).
from evmax.models_ml._math import normal_cdf as _normal_cdf


# NHL abbreviation → canonical lowercased team name. 32-team alignment as of
# the 2026-27 season (Arizona Coyotes relocated to Utah as "Utah Hockey Club"
# for 2024-25; rebranded "Utah Mammoth" for 2025-26). MoneyPuck uses dotted
# variants for some teams (L.A, N.J, S.J, T.B); the seed script normalizes
# those to the standard 3-letter codes below before keying state. Values are
# the sector canonicals from evmax/sectors/aliases/nhl.yaml — St. Louis is the
# DOT-FREE "st louis blues" there (Pinnacle event keys strip dots).
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
    "STL": "st louis blues",
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
        """Resolve a team identifier (full name, nickname, abbrev) to its stats dict.

        Order: the identity tiers of the shared rule (alias canonical → exact →
        canonical equality, ``_team_lookup.identity_team_key`` — canonical
        equality also makes "st. louis blues" ≡ "st louis blues"); then the
        curated ``NHL_ABBREV_TO_NAME`` / ``NHL_NICKNAME_TO_NAME`` dictionaries
        ("L.A" → LAK, "ny rangers" → "new york rangers"); then the shared guarded
        word-boundary fallback. None when nothing qualifies.
        """
        if not team:
            return None
        key = identity_team_key("nhl", team, teams)
        if key is None:
            t = team.lower().strip()
            upper = NHL_MONEYPUCK_ABBREV_ALIASES.get(t.upper(), t.upper())
            full = NHL_ABBREV_TO_NAME.get(upper) or NHL_NICKNAME_TO_NAME.get(t)
            if full is None and " " in t:
                full = NHL_NICKNAME_TO_NAME.get(t.rsplit(" ", 1)[-1])
            if full:
                key = identity_team_key("nhl", full, teams)
        if key is None:
            key = resolve_team_key("nhl", team, teams)
        return teams[key] if key else None

    def _rating_context(
        self, market: PredictionMarket,
    ) -> Optional[tuple[dict, dict, str]]:
        """Return (in_season_teams, regressed_prior_teams, mode) for this game.

        mode is "ramp" (in-season blended with the prior), "prior_only"
        (the in-season block is last season's → it becomes the prior, gp=0)
        or "legacy" (no prior available → raw in-season rates, MIN_GAMES gate).
        None when the state is empty or two-plus seasons stale.
        """
        st = self._sector_state()
        teams = st.get("teams") or {}
        prior = st.get("prior") or {}
        prior_teams = prior.get("teams") or {}
        season = st.get("season_start_year")

        ref = market.event_date.date() if market.event_date else date.today()
        active = nhl_season_for(ref)

        if season is not None and int(season) < active:
            # Staleness guard: last season's FINAL ratings must never fire as
            # current-season ratings (gp=82 → confidence 0.85 on opening night).
            if int(season) != active - 1:
                return None
            lg = st.get("league_avg_xg_per_60")
            if lg is None or not teams:
                return None
            regressed = {
                name: {
                    "xgf_per_60": regress_prior_rate(s["xgf_per_60"], lg),
                    "xga_per_60": regress_prior_rate(s["xga_per_60"], lg),
                }
                for name, s in teams.items()
                if s.get("xgf_per_60") is not None and s.get("xga_per_60") is not None
            }
            return {}, regressed, "prior_only"

        prior_season = prior.get("season_start_year")
        if (
            prior_teams
            and season is not None
            and prior_season is not None
            and int(prior_season) != int(season) - 1
        ):
            prior_teams = {}  # a prior from any other season is not the validated prior
        if not teams and not prior_teams:
            return None
        if prior_teams:
            return teams, prior_teams, ("ramp" if teams else "prior_only")
        return teams, {}, "legacy"

    def _team_rates(
        self, in_teams: dict, prior_teams: dict, team: str,
    ) -> Optional[tuple[float, float, int, bool]]:
        """(xgf_per_60, xga_per_60, gp, used_prior) for one side, or None."""
        stats = self._resolve_team(in_teams, team) if in_teams else None
        pri = self._resolve_team(prior_teams, team) if prior_teams else None
        gp = int(stats.get("gp", 0) or 0) if stats else 0
        if pri is not None:
            # Never blank a team that has a prior: gp=0 → the regressed prior.
            xgf = ramp_rate(stats.get("xgf_per_60") if stats else None, gp, pri["xgf_per_60"])
            xga = ramp_rate(stats.get("xga_per_60") if stats else None, gp, pri["xga_per_60"])
            return xgf, xga, gp, True
        if stats is None or gp < MIN_GAMES:
            return None
        return stats["xgf_per_60"], stats["xga_per_60"], gp, False

    async def predict_pair(
        self,
        market: PredictionMarket,
        sharp_odds: SharpOdds,
    ) -> Optional[ModelAgentPrediction]:
        sector = (market.sector or "").lower()
        if sector != "nhl":
            return None

        ctx = self._rating_context(market)
        if ctx is None:
            return None
        in_teams, prior_teams, mode = ctx

        team_a = (sharp_odds.outcome_a_label or market.team_home or "").lower().strip()
        team_b = (sharp_odds.outcome_b_label or market.team_away or "").lower().strip()

        rates_a = self._team_rates(in_teams, prior_teams, team_a)
        rates_b = self._team_rates(in_teams, prior_teams, team_b)
        if rates_a is None or rates_b is None:
            return None
        xgf_a, xga_a, gp_a, ramped_a = rates_a
        xgf_b, xga_b, gp_b, ramped_b = rates_b

        # Net xG/60 differential: high xGF + low xGA both push net up.
        net_a = xgf_a - xga_a
        net_b = xgf_b - xga_b
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
        elif min_gp >= LOW_CONF_GAMES or (ramped_a and ramped_b):
            # The prior ramp keeps a regressed-prior floor under both sides, so
            # it earns the moderate tier from gp=0 (the validated setting).
            confidence = RAMP_CONFIDENCE
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
                f"{mode} net_xg/60={net_a:+.2f}/{net_b:+.2f} "
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
