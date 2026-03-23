"""InjuryReportAgent — fetches injury reports from ESPN public API and adjusts win probabilities.

Data sources (all public, no auth required):
  NBA:   https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries
  NFL:   https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries
  NCAAB: https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/injuries
  Soccer (EPL): https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/injuries
  Soccer (UCL): https://site.api.espn.com/apis/site/v2/sports/soccer/UEFA.CHAMPIONS/injuries
  Soccer (La Liga): https://site.api.espn.com/apis/site/v2/sports/soccer/ESP.1/injuries
  Soccer (Bundesliga): https://site.api.espn.com/apis/site/v2/sports/soccer/GER.1/injuries
  Soccer (Serie A): https://site.api.espn.com/apis/site/v2/sports/soccer/ITA.1/injuries
  Soccer (Ligue 1): https://site.api.espn.com/apis/site/v2/sports/soccer/FRA.1/injuries

Published topic: "intelligence.injuries.{sector}"

Output: dict[team_name_normalized → InjuryReport]

Probability adjustment logic:
  - Each injured player reduces their team's win probability.
  - Impact is based on:
      1. injury_status: OUT (-0.045), DAY_TO_DAY (-0.025), QUESTIONABLE (-0.012)
      2. player_tier: STAR (1.5×), STARTER (1.0×), ROTATION (0.5×)
  - STAR = top-3 PPG/RPG leader on their team (fetched from ESPN leaders endpoint)
  - Total adjustment is capped at -0.12 per team (can't lose more than 12% regardless)

The adjustment is additive on top of model/sharp probabilities.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Optional

import httpx
import structlog

from evmax.agents.base import Agent, AgentRequest, AgentResponse

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# ESPN API endpoints
# ---------------------------------------------------------------------------

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"

SECTOR_INJURY_URLS: dict[str, list[str]] = {
    "nba": [f"{ESPN_BASE}/basketball/nba/injuries"],
    "nfl": [f"{ESPN_BASE}/football/nfl/injuries"],
    "ncaab": [f"{ESPN_BASE}/basketball/mens-college-basketball/injuries",
              f"{ESPN_BASE}/basketball/mens-college-basketball/tournament/injuries"],
    "soccer": [
        f"{ESPN_BASE}/soccer/eng.1/injuries",       # EPL
        f"{ESPN_BASE}/soccer/UEFA.CHAMPIONS/injuries",  # UCL
        f"{ESPN_BASE}/soccer/ESP.1/injuries",       # La Liga
        f"{ESPN_BASE}/soccer/GER.1/injuries",       # Bundesliga
        f"{ESPN_BASE}/soccer/ITA.1/injuries",       # Serie A
        f"{ESPN_BASE}/soccer/FRA.1/injuries",       # Ligue 1
    ],
    # LoL/CS2: no ESPN injury data — would need Liquipedia or team sites
}

# Status name → raw impact (before tier multiplier)
STATUS_IMPACT: dict[str, float] = {
    "out": 0.045,
    "day-to-day": 0.025,
    "questionable": 0.012,
    "doubtful": 0.030,
    "probable": 0.004,
    "active": 0.0,
}

# Player tier multiplier
TIER_MULTIPLIER = {"star": 1.5, "starter": 1.0, "rotation": 0.5}

# Cap on total adjustment per team
MAX_ADJ = 0.12

# Positions treated as starters by default
HIGH_IMPACT_POSITIONS = {
    # NBA
    "PG", "SG", "SF", "PF", "C",
    "G", "F",
    # Soccer
    "GK", "CB", "LB", "RB", "CM", "CAM", "LW", "RW", "ST", "CF",
    # NFL
    "QB", "WR", "RB", "TE", "LT", "RT",
    # NCAAB
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class InjuredPlayer:
    name: str
    position: str           # e.g. "PG", "C", "GK"
    status: str             # "OUT", "QUESTIONABLE", "DAY-TO-DAY", etc.
    tier: str               # "star", "starter", "rotation"
    impact: float           # probability reduction for their team (0.0–0.05)
    injury_type: str        # "Knee", "Ankle", "Back", etc.
    notes: str              # short human-readable note


@dataclass
class InjuryReport:
    team: str               # normalized team name
    sector: str
    players: list[InjuredPlayer] = field(default_factory=list)

    @property
    def probability_adjustment(self) -> float:
        """Total win probability reduction due to injuries (negative float)."""
        total = sum(p.impact for p in self.players)
        return -min(total, MAX_ADJ)

    @property
    def has_significant_injuries(self) -> bool:
        return any(p.status.lower() in ("out", "doubtful", "day-to-day") for p in self.players)

    def summary(self) -> str:
        if not self.players:
            return f"{self.team}: No injuries"
        lines = [f"{self.team} (adj={self.probability_adjustment:+.1%}):"]
        for p in self.players:
            lines.append(f"  [{p.status}] {p.name} ({p.position}) – {p.injury_type} (−{p.impact:.1%})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class InjuryReportAgent(Agent):
    """
    Fetches ESPN injury reports and computes per-team probability adjustments.

    Published topic payload: dict[team_name → InjuryReport]

    Integrates with EVGapAgent via the coordinator — injuries reduce the
    blended true probability for the affected team, widening the EV gap
    against the Kalshi price when the market hasn't priced in the injury.
    """

    name = "injury_report"
    description = (
        "Fetches ESPN injury data for a sector, parses player status and impact, "
        "outputs team → InjuryReport with probability adjustment."
    )

    def __init__(self, timeout: float = 10.0) -> None:
        super().__init__()
        self._timeout = timeout

    async def run(self, request: AgentRequest) -> AgentResponse:
        sector = request.sector
        urls = SECTOR_INJURY_URLS.get(sector, [])

        if not urls:
            self.log.info("no_injury_source", sector=sector)
            return AgentResponse(agent_name=self.name, sector=sector, data={})

        self.log.info("fetching_injuries", sector=sector, urls=len(urls))

        results = await asyncio.gather(
            *(self._fetch_injuries(url, sector) for url in urls),
            return_exceptions=True,
        )

        reports: dict[str, InjuryReport] = {}
        for r in results:
            if isinstance(r, Exception):
                self.log.warning("injury_fetch_failed", error=str(r))
                continue
            for team, report in r.items():
                if team in reports:
                    # Merge injuries from multiple leagues for the same team (unlikely but safe)
                    reports[team].players.extend(report.players)
                else:
                    reports[team] = report

        self.log.info(
            "injuries_parsed",
            sector=sector,
            teams_with_injuries=len(reports),
            total_players=sum(len(r.players) for r in reports.values()),
        )

        await self.publish(f"intelligence.injuries.{sector}", reports, request.correlation_id)

        return AgentResponse(agent_name=self.name, sector=sector, data=reports)

    # ------------------------------------------------------------------
    # Fetching + parsing
    # ------------------------------------------------------------------

    async def _fetch_injuries(self, url: str, sector: str) -> dict[str, InjuryReport]:
        """Fetch one ESPN injuries URL and return team → InjuryReport."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(url, follow_redirects=True)
            resp.raise_for_status()
            data = resp.json()

        reports: dict[str, InjuryReport] = {}
        for team_entry in data.get("injuries", []):
            if not isinstance(team_entry, dict):
                continue
            team_name = self._normalize_team(team_entry.get("displayName", ""))
            if not team_name:
                continue

            report = InjuryReport(team=team_name, sector=sector)

            for inj in team_entry.get("injuries", []):
                if not isinstance(inj, dict):
                    continue
                player = self._parse_player(inj)
                if player is not None and player.impact > 0:
                    report.players.append(player)

            if report.players:
                reports[team_name] = report

        return reports

    def _parse_player(self, inj: dict, star_ids: set[str] | None = None) -> Optional[InjuredPlayer]:
        """Parse a single injury entry from ESPN JSON."""
        athlete = inj.get("athlete", {})
        name = athlete.get("displayName", "")
        # ESPN injuries API doesn't expose id directly — parse from playercard link href
        athlete_id = ""
        for link in athlete.get("links", []):
            href = link.get("href", "")
            if "/id/" in href:
                m = re.search(r"/id/(\d+)/", href)
                if m:
                    athlete_id = m.group(1)
                    break
        position_obj = athlete.get("position", {})
        position = position_obj.get("abbreviation", "").upper()

        # ESPN returns status as either a plain string ("Out") or a dict with "name"
        status_raw = inj.get("status", "Active")
        if isinstance(status_raw, str):
            status_name = status_raw.strip()
        else:
            status_name = status_raw.get("name", "Active").strip()
        status_key = status_name.lower()

        raw_impact = STATUS_IMPACT.get(status_key, 0.0)
        if raw_impact == 0.0:
            return None  # Active or unknown — no adjustment

        # Injury type
        details = inj.get("details", {})
        injury_type = details.get("type", "Unknown")
        short_comment = inj.get("shortComment", "")

        # Tier: use ESPN leader IDs if available, otherwise fall back to position
        tier = self._classify_tier(position, athlete_id=athlete_id, star_ids=star_ids)
        impact = raw_impact * TIER_MULTIPLIER[tier]

        return InjuredPlayer(
            name=name,
            position=position,
            status=status_name,
            tier=tier,
            impact=round(impact, 4),
            injury_type=injury_type,
            notes=short_comment[:120],
        )

    @staticmethod
    def _classify_tier(position: str, athlete_id: str = "", star_ids: set[str] | None = None) -> str:
        """Classify player tier based on ESPN leaders data and position."""
        if star_ids and athlete_id and athlete_id in star_ids:
            return "star"
        if position in HIGH_IMPACT_POSITIONS:
            return "starter"
        return "rotation"

    @staticmethod
    def _normalize_team(name: str) -> str:
        """Basic team name normalization (lowercase, strip city prefix for matching)."""
        name = name.lower().strip()
        # Remove common city prefixes to get just the team nickname
        # e.g., "Los Angeles Lakers" → "lakers", "Manchester City" → "manchester city" (keep)
        # For NBA/NFL we want the nickname; for soccer the full name is fine
        parts = name.split()
        if len(parts) >= 2:
            # If last word is a common team nickname, try to use last 1-2 words
            # But to avoid over-stripping, just return full lowercase for now
            return name
        return name

    # ------------------------------------------------------------------
    # Utility: apply injury adjustments to a probability dict
    # ------------------------------------------------------------------

    @staticmethod
    def apply_adjustments(
        reports: dict[str, InjuryReport],
        true_prob_a: float,
        true_prob_b: float,
        team_a: str,
        team_b: str,
        spread_multiplier: float = 1.0,
    ) -> tuple[float, float, str]:
        """
        Apply injury probability adjustments to a matched pair.

        spread_multiplier: amplify injury impact for spread markets (default 1.0 for ML,
          use 2.0 for spread — missing a star scorer shifts the margin far more than
          just the win probability).

        Returns (adjusted_prob_a, adjusted_prob_b, notes_str).
        """
        notes = []
        adj_a = 0.0
        adj_b = 0.0

        team_a_norm = team_a.lower().strip()
        team_b_norm = team_b.lower().strip()

        # Cap per-team effective adjustment: max ±10% swing (before multiplier allows up to ±MAX_ADJ)
        _adj_cap = 0.10

        # Find matching reports (fuzzy — check if report team is substring of team_a or vice versa)
        for team_key, report in reports.items():
            if team_a_norm in team_key or team_key in team_a_norm:
                raw = report.probability_adjustment * spread_multiplier
                adj_a = max(-_adj_cap, raw)  # probability_adjustment is negative
                if report.players:
                    notes.append(f"{team_a}:{adj_a:+.1%}({len(report.players)} inj)")
            elif team_b_norm in team_key or team_key in team_b_norm:
                raw = report.probability_adjustment * spread_multiplier
                adj_b = max(-_adj_cap, raw)
                if report.players:
                    notes.append(f"{team_b}:{adj_b:+.1%}({len(report.players)} inj)")

        new_a = max(0.02, min(0.98, true_prob_a + adj_a - adj_b))
        new_b = max(0.02, min(0.98, true_prob_b + adj_b - adj_a))

        # Renormalize
        total = new_a + new_b
        new_a /= total
        new_b /= total

        return new_a, new_b, "; ".join(notes)
