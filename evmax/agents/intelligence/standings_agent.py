"""StandingsAgent — detects playoff clinch status and rest risk from ESPN standings.

Teams that have clinched a playoff spot (or been eliminated) in the final week
of the regular season are high rest risks. Stars sit, rotations shorten, and
game-level probabilities shift significantly vs what sharp lines assume.

ESPN clinch codes:
  * — Clinched Best League Record
  z — Clinched Conference
  y — Clinched Division
  x — Clinched Playoff Berth
  pb — Clinched Play-in Berth
  e — Eliminated From Playoff
  (empty) — Still competing

Data source (public, no auth):
  https://site.api.espn.com/apis/v2/sports/basketball/nba/standings

Published topic: "intelligence.standings.{sector}"
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Optional

import httpx
import structlog

from evmax.agents.base import Agent, AgentRequest, AgentResponse

logger = structlog.get_logger(__name__)

ESPN_V2 = "https://site.api.espn.com/apis/v2/sports"
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports"

SECTOR_STANDINGS_URL: dict[str, str] = {
    "nba": f"{ESPN_V2}/basketball/nba/standings",
    "ncaab": f"{ESPN_V2}/basketball/mens-college-basketball/standings",
}

SECTOR_SCOREBOARD_URL: dict[str, str] = {
    "nba": f"{ESPN_SCOREBOARD}/basketball/nba/scoreboard",
    "ncaab": f"{ESPN_SCOREBOARD}/basketball/mens-college-basketball/scoreboard",
}

# Clinch codes that indicate playoff position is locked
_CLINCHED_CODES = {"*", "z", "y", "x"}
_ELIMINATED_CODE = "e"

# NBA regular season = 82 games
_REGULAR_SEASON_GAMES = {"nba": 82, "ncaab": 35}

# How many games remaining triggers "rest zone" (combined with clinch)
_REST_ZONE_GAMES = 3


@dataclass
class TeamStanding:
    """Parsed standings entry for one team."""

    team_name: str  # ESPN displayName, lowercased
    abbreviation: str
    wins: int
    losses: int
    clinch_code: str  # "", "x", "y", "z", "*", "e", "pb"
    clinch_desc: str
    playoff_seed: int
    games_remaining: int
    is_back_to_back: bool = False

    @property
    def is_clinched(self) -> bool:
        return self.clinch_code in _CLINCHED_CODES

    @property
    def is_eliminated(self) -> bool:
        return self.clinch_code == _ELIMINATED_CODE

    @property
    def rest_risk(self) -> str:
        """Categorize rest risk: 'high', 'medium', 'low', 'none'."""
        if self.is_clinched and self.games_remaining <= _REST_ZONE_GAMES:
            return "high"
        if self.is_clinched and self.games_remaining <= 5:
            return "medium"
        if self.is_eliminated and self.games_remaining <= _REST_ZONE_GAMES:
            return "high"  # Tank/development mode
        if self.is_back_to_back and self.is_clinched:
            return "medium"
        return "none"

    @property
    def probability_adjustment(self) -> float:
        """Negative adjustment to win probability due to rest risk.

        High rest risk: -5% (starters likely sit or play limited minutes)
        Medium risk:    -2% (load management likely for key players)
        """
        risk = self.rest_risk
        if risk == "high":
            return -0.05
        if risk == "medium":
            return -0.02
        return 0.0


class StandingsAgent(Agent):
    """Fetches ESPN standings to detect playoff clinch/elimination and rest risk."""

    name = "standings"
    description = "Detects playoff clinch status and rest risk from ESPN standings."

    def __init__(self, timeout: float = 8.0) -> None:
        super().__init__()
        self._timeout = timeout

    async def run(self, request: AgentRequest) -> AgentResponse:
        sector = request.sector
        url = SECTOR_STANDINGS_URL.get(sector)
        if not url:
            return AgentResponse(agent_name=self.name, sector=sector, data={})

        self.log.info("fetching_standings", sector=sector)

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                standings = await self._fetch_standings(url, sector, client)
            except Exception as e:
                self.log.warning("standings_fetch_failed", error=str(e))
                return AgentResponse(agent_name=self.name, sector=sector, data={})

            # Back-to-back detection: check yesterday + today
            b2b_teams = await self._detect_back_to_back(sector, client)
            for team_key, standing in standings.items():
                if standing.abbreviation in b2b_teams:
                    standing.is_back_to_back = True

        rest_teams = {k: v for k, v in standings.items() if v.rest_risk != "none"}
        if rest_teams:
            self.log.info(
                "rest_risk_detected",
                sector=sector,
                teams={k: v.rest_risk for k, v in rest_teams.items()},
            )

        await self.publish(f"intelligence.standings.{sector}", standings, request.correlation_id)
        return AgentResponse(agent_name=self.name, sector=sector, data=standings)

    async def _fetch_standings(
        self, url: str, sector: str, client: httpx.AsyncClient
    ) -> dict[str, TeamStanding]:
        resp = await client.get(url, follow_redirects=True)
        resp.raise_for_status()
        data = resp.json()

        total_games = _REGULAR_SEASON_GAMES.get(sector, 82)
        standings: dict[str, TeamStanding] = {}

        for conf in data.get("children", []):
            for entry in conf.get("standings", {}).get("entries", []):
                team = entry.get("team", {})
                stats = {s["name"]: s for s in entry.get("stats", [])}

                name = team.get("displayName", "").lower().strip()
                abbrev = team.get("abbreviation", "")
                wins = int(stats.get("wins", {}).get("value", 0))
                losses = int(stats.get("losses", {}).get("value", 0))
                clinch = stats.get("clincher", {})
                clinch_code = clinch.get("displayValue", "")
                clinch_desc = clinch.get("description", "")
                seed = int(stats.get("playoffSeed", {}).get("value", 0))
                games_remaining = max(0, total_games - wins - losses)

                standings[name] = TeamStanding(
                    team_name=name,
                    abbreviation=abbrev,
                    wins=wins,
                    losses=losses,
                    clinch_code=clinch_code,
                    clinch_desc=clinch_desc,
                    playoff_seed=seed,
                    games_remaining=games_remaining,
                )

        return standings

    async def _detect_back_to_back(
        self, sector: str, client: httpx.AsyncClient
    ) -> set[str]:
        """Return abbreviations of teams playing a back-to-back (played yesterday + today)."""
        scoreboard_url = SECTOR_SCOREBOARD_URL.get(sector)
        if not scoreboard_url:
            return set()

        today = date.today()
        yesterday = today - timedelta(days=1)

        try:
            resp_y, resp_t = await asyncio.gather(
                client.get(f"{scoreboard_url}?dates={yesterday:%Y%m%d}", follow_redirects=True),
                client.get(f"{scoreboard_url}?dates={today:%Y%m%d}", follow_redirects=True),
            )
        except Exception:
            return set()

        def _extract_teams(resp: httpx.Response) -> set[str]:
            teams: set[str] = set()
            try:
                for event in resp.json().get("events", []):
                    for comp in event.get("competitions", []):
                        for t in comp.get("competitors", []):
                            teams.add(t.get("team", {}).get("abbreviation", ""))
            except Exception:
                pass
            teams.discard("")
            return teams

        yesterday_teams = _extract_teams(resp_y)
        today_teams = _extract_teams(resp_t)
        return yesterday_teams & today_teams

    @staticmethod
    def apply_adjustments(
        standings: dict[str, TeamStanding],
        true_prob_a: float,
        true_prob_b: float,
        team_a: str,
        team_b: str,
    ) -> tuple[float, float, str]:
        """Apply rest-risk probability adjustments to a matched pair.

        Returns (adjusted_prob_a, adjusted_prob_b, notes_str).
        """
        notes: list[str] = []
        adj_a = 0.0
        adj_b = 0.0

        team_a_norm = team_a.lower().strip()
        team_b_norm = team_b.lower().strip()

        for team_key, standing in standings.items():
            adj = standing.probability_adjustment
            if adj == 0.0:
                continue
            if team_a_norm in team_key or team_key in team_a_norm:
                adj_a = adj
                risk = standing.rest_risk
                parts = [f"rest={risk}"]
                if standing.is_clinched:
                    parts.append(f"clinched(#{standing.playoff_seed})")
                if standing.is_eliminated:
                    parts.append("eliminated")
                if standing.is_back_to_back:
                    parts.append("b2b")
                parts.append(f"{standing.games_remaining}g left")
                notes.append(f"{team_a}:{adj:+.1%}({','.join(parts)})")
            elif team_b_norm in team_key or team_key in team_b_norm:
                adj_b = adj
                risk = standing.rest_risk
                parts = [f"rest={risk}"]
                if standing.is_clinched:
                    parts.append(f"clinched(#{standing.playoff_seed})")
                if standing.is_eliminated:
                    parts.append("eliminated")
                if standing.is_back_to_back:
                    parts.append("b2b")
                parts.append(f"{standing.games_remaining}g left")
                notes.append(f"{team_b}:{adj:+.1%}({','.join(parts)})")

        if adj_a == 0.0 and adj_b == 0.0:
            return true_prob_a, true_prob_b, ""

        new_a = max(0.02, min(0.98, true_prob_a + adj_a - adj_b))
        new_b = max(0.02, min(0.98, true_prob_b + adj_b - adj_a))

        total = new_a + new_b
        new_a /= total
        new_b /= total

        return new_a, new_b, "; ".join(notes)
