"""Point Projection Model — standalone team score predictions.

Combines Poisson attack/defense strengths with Elo-implied win probability
to produce projected scores, spread, and total for any matchup.

This is NOT part of the EV pipeline. It's a standalone conviction tool
for evaluating games independent of market pricing.

Methodology:
  1. Poisson base: λ_home = home_attack × away_defense × league_avg_home
                   λ_away = away_attack × home_defense × league_avg_away
  2. Elo adjustment: shift projected scores toward Elo-implied margin
     (blends Poisson raw projection with Elo-derived spread)
  3. Output: team_a_pts, team_b_pts, projected_spread, projected_total
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)

DATA_DIR = Path("data/models")

# Home advantage in points (used when Elo doesn't provide one)
HOME_ADVANTAGE: dict[str, float] = {
    "nba": 3.5,
    "nfl": 2.5,
    "ncaab": 3.0,
    "ncaaw": 3.0,
    "soccer": 0.4,  # goals
}

# Elo home advantage in Elo points (for win prob calculation)
ELO_HOME_ADV: dict[str, float] = {
    "nba": 100.0,
    "nfl": 48.0,
    "ncaab": 70.0,
    "ncaaw": 70.0,
    "soccer": 60.0,
}

# Default league averages (fallback if not in Poisson state)
LEAGUE_AVG_DEFAULTS: dict[str, dict[str, float]] = {
    "nba": {"home": 113.5, "away": 111.0},
    "nfl": {"home": 23.5, "away": 21.5},
    "ncaab": {"home": 72.0, "away": 70.0},
    "ncaaw": {"home": 68.0, "away": 65.0},
    "soccer": {"home": 1.55, "away": 1.15},
}

# Empirical scoring margin standard deviations
SECTOR_SIGMA: dict[str, float] = {
    "nba": 11.5,
    "nfl": 14.0,
    "ncaab": 12.5,
    "ncaaw": 12.5,
    "soccer": 1.9,
}

DEFAULT_ELO = 1500.0

# How much Elo adjustment influences the final projection (0=pure Poisson, 1=pure Elo)
ELO_BLEND_WEIGHT = 0.35

# City → team name mapping for common lookups (lowercase)
CITY_TO_TEAM: dict[str, str] = {
    # NBA
    "atlanta": "hawks", "boston": "celtics", "brooklyn": "nets",
    "charlotte": "hornets", "chicago": "bulls", "cleveland": "cavaliers",
    "dallas": "mavericks", "denver": "nuggets", "detroit": "pistons",
    "golden state": "warriors", "houston": "rockets", "indiana": "pacers",
    "memphis": "grizzlies", "miami": "heat", "milwaukee": "bucks",
    "minnesota": "timberwolves", "new orleans": "pelicans",
    "new york": "knicks", "oklahoma city": "thunder", "okc": "thunder",
    "orlando": "magic", "philadelphia": "76ers", "philly": "76ers",
    "phoenix": "suns", "portland": "trail blazers", "sacramento": "kings",
    "san antonio": "spurs", "toronto": "raptors", "utah": "jazz",
    "washington": "wizards", "la clippers": "la clippers",
    "los angeles lakers": "lakers", "los angeles clippers": "la clippers",
    # NFL
    "green bay": "packers", "kansas city": "chiefs",
    "las vegas": "raiders", "jacksonville": "jaguars",
    "pittsburgh": "steelers", "baltimore": "ravens",
    "cincinnati": "bengals", "buffalo": "bills",
    "seattle": "seahawks", "tampa bay": "buccaneers",
    "tennessee": "titans", "carolina": "panthers",
    "new england": "patriots", "arizona": "cardinals",
}


@dataclass
class GameProjection:
    """Projected scores and derived lines for a single game."""

    home_team: str
    away_team: str
    home_points: float
    away_points: float
    projected_spread: float  # negative = home favored
    projected_total: float
    home_elo: float
    away_elo: float
    elo_win_prob_home: float
    confidence: str  # "high", "medium", "low"
    sector: str

    @property
    def spread_pick(self) -> str:
        """Which side the model favors on the spread."""
        if abs(self.projected_spread) < 1.0:
            return "Pick"
        return self.home_team if self.projected_spread < 0 else self.away_team


class PointProjectionModel:
    """Produces point projections by combining Poisson and Elo state."""

    def __init__(self) -> None:
        self._poisson_state: dict = {}
        self._elo_state: dict = {}
        self._load_state()

    def _load_state(self) -> None:
        poisson_path = DATA_DIR / "poisson_state.json"
        elo_path = DATA_DIR / "elo_state.json"
        if poisson_path.exists():
            self._poisson_state = json.loads(poisson_path.read_text())
        if elo_path.exists():
            self._elo_state = json.loads(elo_path.read_text())

    def _get_league_avg(self, sector: str) -> dict[str, float]:
        return (
            self._poisson_state.get(sector, {}).get("league_avg")
            or LEAGUE_AVG_DEFAULTS.get(sector, {"home": 72.0, "away": 70.0})
        )

    def _resolve_key(self, store: dict, team: str) -> Optional[str]:
        """Resolve a team name against a dict using exact, last-word, suffix, and normalizer fallbacks."""
        if team in store:
            return team
        if " " in team:
            last = team.rsplit(" ", 1)[-1]
            if last in store:
                return last
        for key in store:
            if team.endswith(key) or key.endswith(team):
                return key
        return None

    def _normalize_team(self, sector: str, team: str) -> str:
        """Normalize a team name using city lookup and sector alias system."""
        team_lower = team.lower().strip()
        # Try city → team mapping first
        if team_lower in CITY_TO_TEAM:
            return CITY_TO_TEAM[team_lower]
        try:
            from evmax.matching.normalizer import NameNormalizer
            norm = NameNormalizer(sector)
            result = norm.normalize(team_lower)
            return result if result else team_lower
        except Exception:
            return team_lower

    def _get_poisson_stats(self, sector: str, team: str) -> Optional[dict]:
        teams = self._poisson_state.get(sector, {}).get("teams", {})
        key = self._resolve_key(teams, team)
        return teams.get(key) if key else None

    def _get_elo(self, sector: str, team: str) -> float:
        ratings = self._elo_state.get(sector, {}).get("ratings", {})
        key = self._resolve_key(ratings, team)
        return ratings.get(key, DEFAULT_ELO) if key else DEFAULT_ELO

    def _elo_win_prob(self, elo_a: float, elo_b: float, home_adv: float) -> float:
        """P(A wins) given Elo ratings and home advantage for A."""
        return 1.0 / (1.0 + 10 ** ((elo_b - elo_a - home_adv) / 400))

    def _elo_implied_margin(self, win_prob: float, sigma: float) -> float:
        """Convert win probability to implied point margin via normal inverse."""
        from scipy.stats import norm

        # Clamp to avoid infinite margins
        p = max(0.01, min(0.99, win_prob))
        return norm.ppf(p) * sigma

    def project(
        self,
        home_team: str,
        away_team: str,
        sector: str,
    ) -> Optional[GameProjection]:
        """Generate point projection for a matchup."""
        sector = sector.lower()
        home_key = self._normalize_team(sector, home_team.lower().strip())
        away_key = self._normalize_team(sector, away_team.lower().strip())

        avg = self._get_league_avg(sector)
        avg_h = avg["home"]
        avg_a = avg["away"]

        # --- Poisson base projection ---
        home_stats = self._get_poisson_stats(sector, home_key)
        away_stats = self._get_poisson_stats(sector, away_key)

        home_attack = home_stats["attack"] if home_stats else 1.0
        home_defense = home_stats["defense"] if home_stats else 1.0
        away_attack = away_stats["attack"] if away_stats else 1.0
        away_defense = away_stats["defense"] if away_stats else 1.0

        poisson_home = home_attack * away_defense * avg_h
        poisson_away = away_attack * home_defense * avg_a

        # --- Elo adjustment ---
        home_elo = self._get_elo(sector, home_key)
        away_elo = self._get_elo(sector, away_key)
        home_adv_elo = ELO_HOME_ADV.get(sector, 50.0)
        sigma = SECTOR_SIGMA.get(sector, 12.0)

        elo_wp = self._elo_win_prob(home_elo, away_elo, home_adv_elo)
        elo_margin = self._elo_implied_margin(elo_wp, sigma)

        # Poisson margin
        poisson_margin = poisson_home - poisson_away

        # Blend: shift Poisson scores toward Elo-implied margin
        blended_margin = (
            (1 - ELO_BLEND_WEIGHT) * poisson_margin
            + ELO_BLEND_WEIGHT * elo_margin
        )

        # Adjust scores symmetrically to match blended margin
        margin_shift = (blended_margin - poisson_margin) / 2
        home_pts = poisson_home + margin_shift
        away_pts = poisson_away - margin_shift

        # --- Confidence ---
        home_games = home_stats.get("games", 0) if home_stats else 0
        away_games = away_stats.get("games", 0) if away_stats else 0
        min_games = min(home_games, away_games)

        if min_games >= 15:
            confidence = "high"
        elif min_games >= 5:
            confidence = "medium"
        else:
            confidence = "low"

        return GameProjection(
            home_team=home_team,
            away_team=away_team,
            home_points=round(home_pts, 2),
            away_points=round(away_pts, 2),
            projected_spread=round(-(home_pts - away_pts), 2),  # negative = home fav
            projected_total=round(home_pts + away_pts, 2),
            home_elo=round(home_elo, 1),
            away_elo=round(away_elo, 1),
            elo_win_prob_home=round(elo_wp, 4),
            confidence=confidence,
            sector=sector,
        )

    def project_from_sharp(
        self,
        home_team: str,
        away_team: str,
        sector: str,
        book_spread: Optional[float] = None,
        book_total: Optional[float] = None,
    ) -> Optional[dict]:
        """Project scores and compare against sportsbook lines.

        Returns a dict with projection + edge analysis.
        """
        proj = self.project(home_team, away_team, sector)
        if proj is None:
            return None

        result = {
            "projection": proj,
            "book_spread": book_spread,
            "book_total": book_total,
            "spread_edge": None,
            "total_edge": None,
            "spread_play": None,
            "total_play": None,
        }

        if book_spread is not None:
            # book_spread is home spread (e.g. -7.5 means home favored by 7.5)
            spread_diff = proj.projected_spread - book_spread
            result["spread_edge"] = round(spread_diff, 2)
            # If our projected spread is more negative than book → home is undervalued
            if abs(spread_diff) >= 1.0:
                if spread_diff < 0:
                    result["spread_play"] = proj.home_team
                else:
                    result["spread_play"] = proj.away_team

        if book_total is not None:
            total_diff = proj.projected_total - book_total
            result["total_edge"] = round(total_diff, 2)
            if abs(total_diff) >= 1.5:
                result["total_play"] = "Over" if total_diff > 0 else "Under"

        return result

    def available_teams(self, sector: str) -> list[str]:
        """List teams with Poisson data for a sector."""
        return sorted(self._poisson_state.get(sector, {}).get("teams", {}).keys())

    def detect_sector(self, team: str) -> Optional[str]:
        """Auto-detect which sector a team belongs to by checking Poisson + Elo state."""
        team_lower = team.lower().strip()

        def _check_store(store: dict) -> bool:
            if self._resolve_key(store, team_lower):
                return True
            return False

        # Check Poisson teams first (more reliable — requires minimum games)
        for sector in self._poisson_state:
            teams = self._poisson_state[sector].get("teams", {})
            if _check_store(teams):
                return sector
            # Try normalizer: "boston" → "celtics" via alias lookup
            normalized = self._normalize_team(sector, team_lower)
            if normalized != team_lower and self._resolve_key(teams, normalized):
                return sector
        # Fallback: check Elo ratings
        for sector in self._elo_state:
            ratings = self._elo_state[sector].get("ratings", {})
            if _check_store(ratings):
                return sector
            normalized = self._normalize_team(sector, team_lower)
            if normalized != team_lower and self._resolve_key(ratings, normalized):
                return sector
        return None
