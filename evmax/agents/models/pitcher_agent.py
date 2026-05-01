"""PitcherModelAgent — pitcher-matchup win probability for MLB games.

Uses the Pythagorean expectation formula (Pythagenpat, exponent=1.83):
    W% = RS^e / (RS^e + RA^e)

Each team scores at the rate the *opposing* starter allows and allows runs at
the rate its own starter does. The "rate" defaults to ERA, but if FIP
(Fielding Independent Pitching) is also seeded for a pitcher, the agent
blends 60% FIP + 40% ERA. FIP strips out defense + sequencing luck and is
more predictive of forward-looking run prevention than ERA — so the blend
favors it. ERA-only pitchers fall back to current behavior.

Live probable starters are fetched from ESPN's scoreboard API each scan
cycle, replacing the static team_starters map with actual game-day pitching.
ESPN's scoreboard provides ERA only — FIP must be seeded externally
(scripts/seed_espn.py::seed_pitchers, or any future Statcast/pybaseball
ingest path).

Only activates for sector == "baseball". Returns None for all other sectors.
"""

from __future__ import annotations

import time
from typing import Optional

import httpx
import structlog

from evmax.agents.models.base import ModelAgent, ModelAgentPrediction
from evmax.models.market import PredictionMarket
from evmax.models.odds import SharpOdds

logger = structlog.get_logger(__name__)

PYTHAG_EXP = 1.83
HOME_BONUS = 0.04  # Default ~54% baseline home win rate in MLB. Used when
                   # state["home_wp_running"] is absent. See _home_advantage().
HOME_BONUS_MIN = 0.01  # Floor — never go fully neutral or anti-home; even in
                       # low-HFA seasons, home teams sleep at home.
HOME_BONUS_MAX = 0.07  # Ceiling — caps over-aggressive correction in years
                       # with unusually high home WP, which often regress.
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"

# When both FIP and ERA are seeded, blend them. FIP is more predictive of
# future run-prevention because it strips defensive luck and sequencing,
# but ERA still carries real signal (a pitcher who consistently outperforms
# FIP via deception, weak contact, or pitching-from-the-stretch may have a
# repeatable skill ERA captures and FIP doesn't). 60/40 in FIP's favor is
# the textbook compromise — heavier than 50/50 because FIP wins on
# year-over-year correlation, but not pure FIP because we don't want to
# discard ERA's real-world result entirely.
FIP_BLEND_WEIGHT = 0.60
ERA_BLEND_WEIGHT = 0.40


def _effective_era(pitcher: dict, league_avg: float) -> float:
    """Compute the run-allowed rate to feed into Pythag.

    Preference order:
      1. Both FIP and ERA present → blend (FIP_BLEND_WEIGHT * FIP + ERA_BLEND_WEIGHT * ERA)
      2. FIP only → return FIP
      3. ERA only → return ERA
      4. Neither → league average
    """
    fip = pitcher.get("fip")
    era = pitcher.get("era")
    if fip is not None and era is not None:
        return FIP_BLEND_WEIGHT * float(fip) + ERA_BLEND_WEIGHT * float(era)
    if fip is not None:
        return float(fip)
    if era is not None:
        return float(era)
    return league_avg

# Cache ESPN probable starters for 30 minutes
_probable_cache: dict[str, dict] = {}
_probable_cache_ts: float = 0.0
_CACHE_TTL = 1800


async def _fetch_probable_starters() -> dict[str, dict]:
    """Fetch today's probable starters from ESPN.

    Returns: {team_short_name: {"name": str, "era": float, "ip": float}}
    """
    global _probable_cache, _probable_cache_ts
    if _probable_cache and (time.time() - _probable_cache_ts) < _CACHE_TTL:
        return _probable_cache

    result: dict[str, dict] = {}
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(ESPN_SCOREBOARD, follow_redirects=True)
            resp.raise_for_status()
            data = resp.json()

        for event in data.get("events", []):
            for comp in event.get("competitions", []):
                for team in comp.get("competitors", []):
                    team_name = team.get("team", {}).get("shortDisplayName", "").lower().strip()
                    if not team_name:
                        team_name = team.get("team", {}).get("displayName", "").lower().strip()
                        # Extract last word: "New York Yankees" → "yankees"
                        if team_name:
                            team_name = team_name.split()[-1]

                    probables = team.get("probables", [])
                    for prob in probables:
                        athlete = prob.get("athlete", {})
                        pitcher_name = athlete.get("fullName", "").lower().strip()
                        # Parse ERA from statistics or record
                        era = 4.08  # default
                        for stat in prob.get("statistics", []):
                            if stat.get("abbreviation") == "ERA":
                                try:
                                    era = float(stat["displayValue"])
                                except (ValueError, KeyError):
                                    pass
                        if team_name and pitcher_name:
                            result[team_name] = {
                                "name": pitcher_name,
                                "era": era,
                                "ip": 0,  # not available from scoreboard
                            }

        _probable_cache = result
        _probable_cache_ts = time.time()
        logger.info("probable_starters_fetched", teams=len(result))
    except Exception as e:
        logger.warning("probable_starters_fetch_failed", error=str(e))

    return result


class PitcherModelAgent(ModelAgent):
    name = "pitcher"
    weight = 0.50

    def _league_avg_era(self) -> float:
        return self._state.get("league_avg_era", 4.08)

    def _pitchers(self) -> dict[str, dict]:
        return self._state.get("pitchers", {})

    def _home_advantage(self) -> float:
        """Resolve the home-side probability bonus, preferring an adaptive
        running estimate when one is recorded in state.

        State shape (optional):
            {"home_wp_running": {"games": int, "home_wins": int}, ...}

        If at least 200 games of running data are available we compute
        HFA = home_wp - 0.50, clamped to [HOME_BONUS_MIN, HOME_BONUS_MAX].
        Below 200 games (very early-season), we fall back to the static
        HOME_BONUS to avoid noisy small-sample HFA. This number can be
        refreshed by `scripts/update_home_advantage.py` from resolved
        outcomes in predictions.db.
        """
        running = self._state.get("home_wp_running") or {}
        games = running.get("games", 0)
        home_wins = running.get("home_wins", 0)
        if games < 200:
            return HOME_BONUS
        wp = home_wins / games
        return max(HOME_BONUS_MIN, min(HOME_BONUS_MAX, wp - 0.50))

    def _find_starter(self, team: str, live_starters: dict[str, dict] | None = None) -> tuple[Optional[dict], bool]:
        """Find the probable starter for a team.

        Returns (pitcher_data, is_live) — is_live=True if from ESPN probables.
        """
        team = team.lower().strip()
        # Last word fallback: "new york yankees" → "yankees"
        team_short = team.split()[-1] if team else team

        # 1) Live ESPN probable starters (today's actual starter)
        if live_starters:
            for key in (team, team_short):
                if key in live_starters:
                    live = live_starters[key]
                    # Check if we have this pitcher in our ERA database
                    pitcher_name = live["name"]
                    stored = self._pitchers().get(pitcher_name)
                    if stored:
                        return stored, True
                    # Use ESPN's ERA directly
                    return live, True

        # 2) Static team_starters (fallback)
        team_starters = self._state.get("team_starters", {})
        for key in (team, team_short):
            pitcher_name = team_starters.get(key)
            if pitcher_name:
                data = self._pitchers().get(pitcher_name)
                if data:
                    return data, False

        # Substring match
        for stored_team, pitcher_name in team_starters.items():
            if stored_team in team or team in stored_team:
                data = self._pitchers().get(pitcher_name)
                if data:
                    return data, False

        return None, False

    async def predict_pair(
        self,
        market: PredictionMarket,
        sharp_odds: SharpOdds,
    ) -> Optional[ModelAgentPrediction]:
        sector = (market.sector or "").lower()
        if sector != "baseball":
            return None

        home = (sharp_odds.outcome_a_label or market.team_home or "").lower().strip()
        away = (sharp_odds.outcome_b_label or market.team_away or "").lower().strip()

        # Fetch live probable starters from ESPN
        live_starters = await _fetch_probable_starters()

        home_pitcher, home_live = self._find_starter(home, live_starters)
        away_pitcher, away_live = self._find_starter(away, live_starters)

        if not home_pitcher or not away_pitcher:
            return None

        league_avg = self._league_avg_era()
        home_rate = _effective_era(home_pitcher, league_avg)
        away_rate = _effective_era(away_pitcher, league_avg)

        # Pythagorean matchup: each team scores at the rate the opposing
        # starter gives up runs and allows at its own starter's rate.
        # The "rate" here is FIP-blended ERA when FIP data is available
        # (preferred — fielding-independent), else raw ERA.
        e = PYTHAG_EXP
        home_rs = away_rate  # we score at rate the opposing pitcher allows
        home_ra = home_rate  # we allow at rate our own pitcher allows
        away_rs = home_rate
        away_ra = away_rate

        home_wp = (home_rs ** e) / (home_rs ** e + home_ra ** e)
        away_wp = (away_rs ** e) / (away_rs ** e + away_ra ** e)

        # home_wp and away_wp are already complements by construction
        # (home_wp = away_era^e / (away_era^e + home_era^e) = 1 - away_wp),
        # but normalize defensively in case ERAs are degenerate.
        total = home_wp + away_wp
        prob_a = home_wp / total if total > 0 else 0.5

        # Apply home field advantage — adaptive when state carries a running
        # estimate, else falls back to the static HOME_BONUS default.
        prob_a = min(0.90, max(0.10, prob_a + self._home_advantage()))
        prob_b = 1.0 - prob_a

        # Confidence tiers — designed so the model fires (>= 0.45 gate) on
        # any pitcher with FIP data, even early-season when IP totals are low.
        # FIP itself is the data-quality signal; IP is a sample-size proxy
        # that mattered more when seeds were prior-season totals (~150+ IP).
        # With current-season FIP seeded, 30 IP of recent data is more
        # informative than 200 IP of last-year data, so the FIP path gets
        # a higher floor.
        home_ip = home_pitcher.get("ip", 0)
        away_ip = away_pitcher.get("ip", 0)
        min_ip = min(home_ip, away_ip)
        both_live = home_live and away_live
        both_fip = (
            home_pitcher.get("fip") is not None
            and away_pitcher.get("fip") is not None
        )

        if both_live and min_ip >= 150:
            confidence = 0.80  # Live + deep history
        elif both_live and min_ip >= 100:
            confidence = 0.70
        elif both_fip and min_ip >= 100:
            confidence = 0.75  # FIP-armed + good sample
        elif both_fip and min_ip >= 30:
            confidence = 0.60  # FIP-armed early-season
        elif both_fip and min_ip >= 15:
            confidence = 0.50  # FIP-armed thin sample — barely above 0.45 gate.
                               # Justified empirically: backtest shows pitcher
                               # is the best single MLB model where it fires,
                               # and a pitcher with FIP from 2-3 starts is more
                               # informative than no signal at all.
        elif both_live:
            confidence = 0.55  # Live but ESPN-only (no IP, no FIP)
        elif min_ip >= 150:
            confidence = 0.65  # ERA-only deep history
        elif min_ip >= 100:
            confidence = 0.55  # ERA-only moderate
        else:
            confidence = 0.35  # ERA-only thin — below gate, won't contribute

        pitcher_notes = []
        if home_live:
            pitcher_notes.append(f"home={home_pitcher.get('name', '?')}(live)")
        if away_live:
            pitcher_notes.append(f"away={away_pitcher.get('name', '?')}(live)")

        def _rate_label(p: dict) -> str:
            if p.get("fip") is not None and p.get("era") is not None:
                return f"fip={p['fip']:.2f}/era={p['era']:.2f}"
            if p.get("fip") is not None:
                return f"fip={p['fip']:.2f}"
            return f"era={p.get('era', league_avg):.2f}"

        notes = (
            f"home[{_rate_label(home_pitcher)}] "
            f"away[{_rate_label(away_pitcher)}] "
            f"effective_home={home_rate:.2f} effective_away={away_rate:.2f} "
            f"{' '.join(pitcher_notes)}"
        ).strip()

        return ModelAgentPrediction(
            event_id=sharp_odds.event_id,
            model_name=self.name,
            true_prob_a=round(prob_a, 4),
            true_prob_b=round(prob_b, 4),
            confidence=confidence,
            weight=self.weight,
            sample_size=int(min_ip),
            notes=notes,
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
        # Pitcher data is seeded externally, not updated from game results
        pass

    def seed_pitchers(self, pitchers: dict[str, dict], league_avg_era: float = 4.08) -> None:
        """Bulk-seed pitcher data.

        Args:
            pitchers: {name: {"era": float, "fip": float (optional), "ip": float, "team": str}}
                ERA is required as the baseline rate. FIP is optional but
                strongly preferred — when present, the agent blends 60% FIP
                + 40% ERA, which is more predictive of future run prevention.
            league_avg_era: league average ERA for normalizing fallback rates
        """
        self._state["league_avg_era"] = league_avg_era
        store = self._state.setdefault("pitchers", {})
        team_starters: dict[str, str] = self._state.setdefault("team_starters", {})

        for name, data in pitchers.items():
            key = name.lower().strip()
            store[key] = data
            team = data.get("team", "").lower()
            if team:
                # First pitcher encountered per team becomes the "starter"
                # (seed script should list #1 starter first)
                if team not in team_starters:
                    team_starters[team] = key

        self.save_state()
        self.log.info("pitchers_seeded", count=len(pitchers), league_avg=league_avg_era)
