"""PitcherModelAgent — ERA-based win probability for MLB games.

Uses the Pythagorean expectation formula (Pythagenpat, exponent=1.83):
    W% = RS^e / (RS^e + RA^e)

For each team, RS = league average (assumes average offense), RA = opposing
pitcher's ERA. This isolates the pitching matchup as the primary driver of
game-level win probability.

Only activates for sector == "baseball". Returns None for all other sectors.
"""

from __future__ import annotations

from typing import Optional

from evmax.agents.models.base import ModelAgent, ModelAgentPrediction
from evmax.models.market import PredictionMarket
from evmax.models.odds import SharpOdds

PYTHAG_EXP = 1.83
HOME_BONUS = 0.04  # ~54% baseline home win rate in MLB


class PitcherModelAgent(ModelAgent):
    name = "pitcher"
    weight = 0.30

    def _league_avg_era(self) -> float:
        return self._state.get("league_avg_era", 4.08)

    def _pitchers(self) -> dict[str, dict]:
        return self._state.get("pitchers", {})

    def _find_starter(self, team: str) -> Optional[dict]:
        """Find the probable starter for a team.

        Handles full names ("new york yankees") by checking if the stored
        short team name ("yankees") appears in the input, or vice versa.
        """
        team = team.lower().strip()
        team_starters = self._state.get("team_starters", {})

        # Exact match first
        pitcher_name = team_starters.get(team)
        if pitcher_name:
            return self._pitchers().get(pitcher_name)

        # Substring match: "yankees" in "new york yankees" or vice versa
        for stored_team, pitcher_name in team_starters.items():
            if stored_team in team or team in stored_team:
                return self._pitchers().get(pitcher_name)

        # Fallback: search pitchers by team field
        for name, data in self._pitchers().items():
            t = data.get("team", "").lower()
            if t == team or t in team or team in t:
                return data
        return None

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

        home_pitcher = self._find_starter(home)
        away_pitcher = self._find_starter(away)

        if not home_pitcher or not away_pitcher:
            return None

        league_avg = self._league_avg_era()
        home_era = home_pitcher.get("era", league_avg)
        away_era = away_pitcher.get("era", league_avg)

        # Pythagorean: home team scores at league avg, allows at away pitcher's rate
        # Away team scores at league avg, allows at home pitcher's rate
        home_rs = league_avg
        home_ra = away_era
        away_rs = league_avg
        away_ra = home_era

        e = PYTHAG_EXP
        home_wp = (home_rs ** e) / (home_rs ** e + home_ra ** e)
        away_wp = (away_rs ** e) / (away_rs ** e + away_ra ** e)

        # Normalize to sum to 1
        total = home_wp + away_wp
        prob_a = home_wp / total if total > 0 else 0.5

        # Apply home field advantage
        prob_a = min(0.90, max(0.10, prob_a + HOME_BONUS))
        prob_b = 1.0 - prob_a

        # Confidence based on innings pitched (more IP = more reliable ERA)
        home_ip = home_pitcher.get("ip", 0)
        away_ip = away_pitcher.get("ip", 0)
        min_ip = min(home_ip, away_ip)
        if min_ip >= 150:
            confidence = 0.75
        elif min_ip >= 100:
            confidence = 0.60
        else:
            confidence = 0.40

        return ModelAgentPrediction(
            event_id=sharp_odds.event_id,
            model_name=self.name,
            true_prob_a=round(prob_a, 4),
            true_prob_b=round(prob_b, 4),
            confidence=confidence,
            weight=self.weight,
            sample_size=int(min_ip),
            notes=f"home_era={home_era:.2f} away_era={away_era:.2f}",
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
