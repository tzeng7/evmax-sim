"""PlayerImpactAgent — adjusts win probability based on player availability.

Uses player on-court NET_RATING and minutes from stats.nba.com to estimate
each player's contribution to team strength. When a player is injured/out,
the team's projected strength is reduced proportionally.

Algorithm:
  1. Fetch all player advanced stats (NET_RATING, MIN) from nba_api.
  2. For each team, compute total "impact minutes" = Σ(player_min × player_net_rating).
  3. When InjuryReportAgent flags players as OUT/DOUBTFUL, subtract their
     impact from the team total.
  4. Convert the impact differential to a win probability adjustment.

The adjustment is additive on the probability scale and capped at ±15%.

Published topic: "intelligence.player_impact.{sector}"
"""

from __future__ import annotations

import asyncio
import json
from datetime import date
from pathlib import Path
from typing import Optional

import structlog

from evmax.agents.base import Agent, AgentRequest, AgentResponse

logger = structlog.get_logger(__name__)

STATE_PATH = Path(__file__).resolve().parents[3] / "data" / "models" / "player_impact_state.json"
CACHE_TTL_HOURS = 12

# Cap per-team adjustment at ±15%
MAX_ADJ = 0.15

# Minimum minutes per game to be considered a rotation player
MIN_MPG = 10.0

# NBA team abbreviation → canonical name mapping
TEAM_ABBREV: dict[str, str] = {
    "ATL": "hawks", "BOS": "celtics", "BKN": "nets", "CHA": "hornets",
    "CHI": "bulls", "CLE": "cavaliers", "DAL": "mavericks", "DEN": "nuggets",
    "DET": "pistons", "GSW": "warriors", "HOU": "rockets", "IND": "pacers",
    "LAC": "la clippers", "LAL": "lakers", "MEM": "grizzlies", "MIA": "heat",
    "MIL": "bucks", "MIN": "timberwolves", "NOP": "pelicans", "NYK": "knicks",
    "OKC": "thunder", "ORL": "magic", "PHI": "76ers", "PHX": "suns",
    "POR": "trail blazers", "SAC": "kings", "SAS": "spurs", "TOR": "raptors",
    "UTA": "jazz", "WAS": "wizards",
}


class PlayerImpactAgent(Agent):
    """Estimates team strength adjustment from player availability."""

    name = "player_impact"
    description = "Adjusts win probability based on which players are available."

    def __init__(self) -> None:
        super().__init__()
        self._player_data: dict[str, list[dict]] = {}  # team_key → [player stats]
        self._fetched_date: str = ""

    async def run(self, request: AgentRequest) -> AgentResponse:
        sector = request.sector
        if sector != "nba":
            return AgentResponse(agent_name=self.name, sector=sector, data={})

        await self._ensure_data()

        injuries: dict[str, list[str]] = request.params.get("injuries", {})

        adjustments: dict[str, dict] = {}
        pairs = request.params.get("pairs", [])

        for pair in pairs:
            market = pair.get("market")
            sharp = pair.get("sharp")
            if not market or not sharp:
                continue

            team_a = (sharp.outcome_a_label or market.team_home or "").lower().strip()
            team_b = (sharp.outcome_b_label or market.team_away or "").lower().strip()
            event_id = sharp.event_id

            adj_a = self._compute_injury_impact(team_a, injuries)
            adj_b = self._compute_injury_impact(team_b, injuries)

            if adj_a != 0.0 or adj_b != 0.0:
                adjustments[event_id] = {
                    "adj_a": adj_a,
                    "adj_b": adj_b,
                    "missing_a": self._missing_players(team_a, injuries),
                    "missing_b": self._missing_players(team_b, injuries),
                }

        self.log.info("player_impact_done", adjustments=len(adjustments))

        await self.publish(
            f"intelligence.player_impact.{sector}", adjustments, request.correlation_id
        )

        return AgentResponse(agent_name=self.name, sector=sector, data=adjustments)

    async def _ensure_data(self) -> None:
        """Fetch player stats if not cached for today."""
        if self._fetched_date == date.today().isoformat() and self._player_data:
            return

        # Try loading from disk cache first
        if STATE_PATH.exists():
            try:
                cached = json.loads(STATE_PATH.read_text())
                if cached.get("fetched_date") == date.today().isoformat():
                    self._player_data = cached.get("teams", {})
                    self._fetched_date = cached["fetched_date"]
                    return
            except Exception:
                pass

        try:
            await self._fetch_player_stats()
        except Exception as e:
            self.log.warning("player_impact_fetch_failed", error=str(e))

    async def _fetch_player_stats(self) -> None:
        """Fetch player advanced stats from stats.nba.com."""
        from nba_api.stats.endpoints import LeagueDashPlayerStats

        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            None,
            lambda: LeagueDashPlayerStats(
                season="2025-26",
                per_mode_detailed="PerGame",
                measure_type_detailed_defense="Advanced",
                timeout=15,
            ).get_dict(),
        )

        headers = data["resultSets"][0]["headers"]
        rows = data["resultSets"][0]["rowSet"]

        teams: dict[str, list[dict]] = {}

        for row in rows:
            d = dict(zip(headers, row))
            mpg = d.get("MIN", 0) or 0
            if mpg < MIN_MPG:
                continue

            abbrev = d.get("TEAM_ABBREVIATION", "")
            team_key = TEAM_ABBREV.get(abbrev, abbrev.lower())
            net_rating = d.get("NET_RATING", 0) or 0

            player = {
                "name": d["PLAYER_NAME"].lower(),
                "mpg": mpg,
                "net_rating": net_rating,
                "gp": d.get("GP", 0),
                "pie": d.get("PIE", 0),
                "usg": d.get("USG_PCT", 0),
            }

            if team_key not in teams:
                teams[team_key] = []
            teams[team_key].append(player)

        self._player_data = teams
        self._fetched_date = date.today().isoformat()

        # Cache to disk
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps({
            "fetched_date": self._fetched_date,
            "teams": teams,
        }, indent=2))

        total_players = sum(len(v) for v in teams.values())
        self.log.info("player_stats_fetched", teams=len(teams), players=total_players)

    def _resolve_team(self, team: str) -> Optional[list[dict]]:
        """Resolve team name to player list."""
        team = team.lower().strip()
        if team in self._player_data:
            return self._player_data[team]
        if " " in team:
            last = team.rsplit(" ", 1)[-1]
            if last in self._player_data:
                return self._player_data[last]
        for key in self._player_data:
            if team.endswith(key) or key.endswith(team) or team.startswith(key):
                return self._player_data[key]
        return None

    def _missing_players(self, team: str, injuries: dict[str, list[str]]) -> list[str]:
        """Get list of missing player names for a team."""
        team_injuries = injuries.get(team, [])
        if not team_injuries:
            for key, players in injuries.items():
                if team in key.lower() or key.lower() in team:
                    team_injuries = players
                    break
        return [p.lower() for p in team_injuries]

    def _compute_injury_impact(self, team: str, injuries: dict[str, list[str]]) -> float:
        """Compute win probability adjustment from missing players.

        Uses weighted impact: player_mpg × player_net_rating as a measure
        of how much value the team loses without them. Normalized by the
        team's total impact minutes.

        Returns a negative adjustment (team is weaker without the player).
        """
        players = self._resolve_team(team)
        if not players:
            return 0.0

        missing = self._missing_players(team, injuries)
        if not missing:
            return 0.0

        total_impact = sum(p["mpg"] * p["net_rating"] for p in players)
        if abs(total_impact) < 1e-6:
            return 0.0

        lost_impact = 0.0
        for p in players:
            p_name = p["name"].lower()
            for m in missing:
                if m in p_name or p_name in m:
                    lost_impact += p["mpg"] * p["net_rating"]
                    break

        if lost_impact == 0.0:
            return 0.0

        # Fraction of team impact lost → probability adjustment
        # A team losing 30% of its positive impact ≈ -6% win probability
        fraction_lost = lost_impact / total_impact if total_impact > 0 else 0.0
        adj = -fraction_lost * 0.20  # scale factor calibrated to historical data

        return max(-MAX_ADJ, min(MAX_ADJ, adj))
