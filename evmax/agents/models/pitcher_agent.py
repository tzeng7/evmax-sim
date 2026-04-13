"""PitcherModelAgent — ERA-based win probability for MLB games.

Uses the Pythagorean expectation formula (Pythagenpat, exponent=1.83):
    W% = RS^e / (RS^e + RA^e)

For each team, RS = league average (assumes average offense), RA = opposing
pitcher's ERA. This isolates the pitching matchup as the primary driver of
game-level win probability.

Live probable starters are fetched from ESPN's scoreboard API each scan
cycle, replacing the static team_starters map with actual game-day pitching.

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
HOME_BONUS = 0.04  # ~54% baseline home win rate in MLB
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"

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
    weight = 0.30

    def _league_avg_era(self) -> float:
        return self._state.get("league_avg_era", 4.08)

    def _pitchers(self) -> dict[str, dict]:
        return self._state.get("pitchers", {})

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
        home_era = home_pitcher.get("era", league_avg)
        away_era = away_pitcher.get("era", league_avg)

        # Pythagorean matchup: each team scores at the rate the opposing
        # starter gives up runs (opponent's ERA) and allows at its own
        # starter's rate. league_avg is *not* used in the matchup formula —
        # it only backstops pitchers with unknown ERA in the fallback above.
        e = PYTHAG_EXP
        home_rs = away_era  # we score at rate the opposing pitcher allows
        home_ra = home_era  # we allow at rate our own pitcher allows
        away_rs = home_era
        away_ra = away_era

        home_wp = (home_rs ** e) / (home_rs ** e + home_ra ** e)
        away_wp = (away_rs ** e) / (away_rs ** e + away_ra ** e)

        # home_wp and away_wp are already complements by construction
        # (home_wp = away_era^e / (away_era^e + home_era^e) = 1 - away_wp),
        # but normalize defensively in case ERAs are degenerate.
        total = home_wp + away_wp
        prob_a = home_wp / total if total > 0 else 0.5

        # Apply home field advantage
        prob_a = min(0.90, max(0.10, prob_a + HOME_BONUS))
        prob_b = 1.0 - prob_a

        # Confidence: live starters with known ERA from our DB get high confidence.
        # ESPN-only ERA (no IP context) gets moderate. Static fallback gets low.
        home_ip = home_pitcher.get("ip", 0)
        away_ip = away_pitcher.get("ip", 0)
        min_ip = min(home_ip, away_ip)
        both_live = home_live and away_live

        if both_live and min_ip >= 150:
            confidence = 0.80  # Live starters + deep ERA history
        elif both_live and min_ip >= 100:
            confidence = 0.70
        elif both_live:
            confidence = 0.55  # Live starters but ERA from ESPN only (no IP)
        elif min_ip >= 150:
            confidence = 0.65  # Static starters but good ERA data
        elif min_ip >= 100:
            confidence = 0.55
        else:
            confidence = 0.35  # Static + thin data — below gate, won't contribute

        pitcher_notes = []
        if home_live:
            pitcher_notes.append(f"home={home_pitcher.get('name', '?')}(live)")
        if away_live:
            pitcher_notes.append(f"away={away_pitcher.get('name', '?')}(live)")

        return ModelAgentPrediction(
            event_id=sharp_odds.event_id,
            model_name=self.name,
            true_prob_a=round(prob_a, 4),
            true_prob_b=round(prob_b, 4),
            confidence=confidence,
            weight=self.weight,
            sample_size=int(min_ip),
            notes=f"home_era={home_era:.2f} away_era={away_era:.2f} {' '.join(pitcher_notes)}",
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
        """Bulk-seed pitcher ERA data.

        Args:
            pitchers: {name: {"era": float, "ip": float, "team": str}}
            league_avg_era: league average ERA for normalizing
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
