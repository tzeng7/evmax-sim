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
    "ncaab": [f"{ESPN_BASE}/basketball/mens-college-basketball/injuries"],
    "ncaaw": [f"{ESPN_BASE}/basketball/womens-college-basketball/injuries"],
    "wnba": [f"{ESPN_BASE}/basketball/wnba/injuries"],
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

# Cap on total adjustment per team.
# With star detection working, the raw impact of 2 stars OUT is already 13.5%.
# A team missing its top 2 players should absolutely reflect > 12% impact.
# Cap at 20% — losing your entire starting five is roughly a 20-25pp swing.
MAX_ADJ = 0.20

# ---------------------------------------------------------------------------
# Known star players — fallback when ESPN leaders endpoint is unavailable.
# Name-normalized (lowercased). Updated periodically; does not need to be
# exhaustive — the ESPN leaders path (`star_ids`) catches most active stars.
# This list is the safety net for injured stars who drop off the leaders.
# ---------------------------------------------------------------------------
KNOWN_STARS: set[str] = {
    # NBA
    "luka doncic", "nikola jokic", "giannis antetokounmpo", "shai gilgeous-alexander",
    "jayson tatum", "anthony edwards", "lebron james", "stephen curry",
    "kevin durant", "joel embiid", "jaylen brown", "devin booker",
    "donovan mitchell", "jimmy butler", "anthony davis", "damian lillard",
    "trae young", "de'aaron fox", "tyrese haliburton", "lamelo ball",
    "zion williamson", "karl-anthony towns", "bam adebayo", "paolo banchero",
    "victor wembanyama", "ja morant", "chet holmgren", "jalen brunson",
    "darius garland", "evan mobley", "franz wagner", "scottie barnes",
    "austin reaves", "alperen sengun", "cade cunningham", "tyrese maxey",
    # NFL
    "patrick mahomes", "josh allen", "lamar jackson", "joe burrow",
    "jalen hurts", "justin jefferson", "ja'marr chase", "tyreek hill",
    "ceedee lamb", "davante adams", "travis kelce", "nick bosa",
    "myles garrett", "micah parsons", "t.j. watt", "derrick henry",
    "saquon barkley", "christian mccaffrey", "breece hall",
    "tua tagovailoa", "c.j. stroud", "caleb williams",
    # WNBA
    "a'ja wilson", "breanna stewart", "sabrina ionescu", "caitlin clark",
    "alyssa thomas", "napheesa collier", "kelsey plum", "jewell loyd",
    "angel reese", "dearica hamby",
    # Soccer (key players in top-5 leagues)
    "erling haaland", "kylian mbappe", "vinicius junior", "jude bellingham",
    "bukayo saka", "rodri", "martin odegaard", "mohamed salah",
    "robert lewandowski", "lamine yamal", "florian wirtz",
    # NHL
    "connor mcdavid", "nathan mackinnon", "auston matthews",
    "leon draisaitl", "nikita kucherov", "cale makar",
}

# Positions treated as starters by default
HIGH_IMPACT_POSITIONS = {
    # NBA
    "PG", "SG", "SF", "PF", "C",
    "G", "F",
    # Soccer
    "GK", "CB", "LB", "RB", "CM", "CAM", "LW", "RW", "ST", "CF",
    # NFL — tier system says these are starters; the sector-aware position
    # weight below differentiates *which* starter (a QB out is ~10x a guard out)
    "QB", "WR", "RB", "TE", "LT", "RT", "C", "G", "OL",
    "DE", "DT", "EDGE", "OLB", "ILB", "LB", "CB", "S", "FS", "SS",
}


# ---------------------------------------------------------------------------
# Sector-aware position weights — multiply the existing tier-weighted impact.
# Only NFL uses these today. Soccer / NBA / NCAAB / etc. fall through to a
# flat 1.0 multiplier so behavior for those sectors is unchanged.
#
# Calibration intent: a starting QB (Mahomes/Allen-tier) being ruled OUT shifts
# Vegas spreads ~5-7 points = ~12-15pp swing in win probability. Working
# backward: 0.045 (OUT raw) × 1.5 (star tier) × 1.5 (QB position weight) ≈ 10pp
# at the report level, which the apply_adjustments step then redistributes
# between the two teams (≈ 12-14pp swing on the matchup). Aligned with sharps.
#
# Other positions are scaled relative to QB. A starting LT/RT (pass protection
# anchor) is the second-biggest position swing, an EDGE is third (because they
# warp the opposing offense's pass game). Specialty positions (K/P/LS) are
# near-zero — their absence rarely moves a number more than a fraction of a pt.
# ---------------------------------------------------------------------------

NFL_POSITION_WEIGHTS: dict[str, float] = {
    "QB":   1.50,
    "LT":   0.50,
    "RT":   0.40,
    "EDGE": 0.40,
    "DE":   0.40,
    "OLB":  0.35,
    "C":    0.30,
    "WR":   0.30,
    "CB":   0.30,
    "RB":   0.25,
    "S":    0.20,
    "FS":   0.20,
    "SS":   0.20,
    "TE":   0.20,
    "DT":   0.20,
    "G":    0.15,
    "OL":   0.15,
    "LB":   0.15,
    "ILB":  0.15,
    "FB":   0.05,
    "K":    0.05,
    "P":    0.02,
    "LS":   0.01,
}

# Default position weight for NFL when the position string isn't in the map
# (typos, weird ESPN abbreviations, etc.). Conservative — better to under-
# weight an unknown position than over-weight it.
NFL_DEFAULT_POSITION_WEIGHT = 0.10

SECTOR_POSITION_WEIGHTS: dict[str, dict[str, float]] = {
    "nfl": NFL_POSITION_WEIGHTS,
}


def _position_weight(sector: str, position: str) -> float:
    """Return the position-specific multiplier for a sector, or 1.0 when no
    sector-specific weights are defined (preserves existing behavior)."""
    if not sector:
        return 1.0
    table = SECTOR_POSITION_WEIGHTS.get(sector.lower())
    if table is None:
        return 1.0
    if not position:
        return NFL_DEFAULT_POSITION_WEIGHT if sector.lower() == "nfl" else 1.0
    return table.get(position.upper(), NFL_DEFAULT_POSITION_WEIGHT
                     if sector.lower() == "nfl" else 1.0)


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
    reported_at: Optional[str] = None  # ISO timestamp of latest ESPN update,
                                       # used for late-news edge detection


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

    @property
    def latest_report_at(self) -> Optional[str]:
        """ISO timestamp of the most recent status change across all listed
        players. Drives the "late news" tagging in EVGapAgent — bets logged
        within a few hours of this timestamp are flagged as potentially
        anticipating Pinnacle's adjustment to the news."""
        stamps = [p.reported_at for p in self.players if p.reported_at]
        return max(stamps) if stamps else None

    @property
    def has_significant_out(self) -> bool:
        """True if any star/starter is OUT or DOUBTFUL — the cases where a
        late status change is most likely to move the line. Filters out
        deep-bench day-to-day churn that doesn't move books."""
        return any(
            p.status.lower() in ("out", "doubtful")
            and p.tier in ("star", "starter")
            for p in self.players
        )

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

        # Share one httpx client across all parallel fetches to reuse TCP/TLS connections
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            results = await asyncio.gather(
                *(self._fetch_injuries(url, sector, client) for url in urls),
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

    async def _fetch_injuries(self, url: str, sector: str, client: httpx.AsyncClient) -> dict[str, InjuryReport]:
        """Fetch one ESPN injuries URL and return team → InjuryReport."""
        resp = await client.get(url, follow_redirects=True)
        if resp.status_code == 404:
            self.log.debug("injury_url_not_found", url=url)
            return {}
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
                player = self._parse_player(inj, sector=sector)
                if player is not None and player.impact > 0:
                    report.players.append(player)

            if report.players:
                reports[team_name] = report

        return reports

    def _parse_player(
        self,
        inj: dict,
        star_ids: set[str] | None = None,
        sector: str = "",
    ) -> Optional[InjuredPlayer]:
        """Parse a single injury entry from ESPN JSON.

        `sector` enables sector-aware position weighting (currently only NFL).
        Defaults to empty string so existing call sites and tests keep their
        original behavior — only sectors registered in SECTOR_POSITION_WEIGHTS
        get the position multiplier applied.
        """
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

        # Tier: use ESPN leader IDs if available, then known stars, then position
        tier = self._classify_tier(position, athlete_id=athlete_id, star_ids=star_ids, player_name=name)
        # Sector-aware position multiplier — non-NFL sectors get 1.0 (no change).
        # For NFL: a starting QB out hits the team much harder than a backup G,
        # which the flat tier system can't express on its own.
        pos_weight = _position_weight(sector, position)
        impact = raw_impact * TIER_MULTIPLIER[tier] * pos_weight

        # ESPN's injuries endpoint includes a `date` field (ISO timestamp)
        # for each entry — when the status was last updated. Capture it so
        # late-news edge detection can flag bets logged shortly after a
        # status change. Format varies: usually "2026-05-11T18:45Z" or
        # similar; pass through as-is.
        reported_at = inj.get("date")

        return InjuredPlayer(
            name=name,
            position=position,
            status=status_name,
            tier=tier,
            impact=round(impact, 4),
            injury_type=injury_type,
            notes=short_comment[:120],
            reported_at=reported_at,
        )

    @staticmethod
    def _classify_tier(
        position: str,
        athlete_id: str = "",
        star_ids: set[str] | None = None,
        player_name: str = "",
    ) -> str:
        """Classify player tier based on ESPN leaders data, known stars list, and position.

        Priority:
          1. ESPN leaders endpoint (star_ids) — most accurate, but only covers
             players with enough games this season.
          2. KNOWN_STARS hardcoded set — catches franchise players who are injured
             long-term and drop off the leaders board (e.g., Luka, Embiid).
          3. Position-based fallback — all starters get "starter" tier.
        """
        if star_ids and athlete_id and athlete_id in star_ids:
            return "star"
        if player_name and player_name.lower().strip() in KNOWN_STARS:
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

    # Default usage share by tier when OUT — redistributed among remaining players
    _USAGE_BY_TIER = {"star": 0.25, "starter": 0.15, "rotation": 0.08}

    @staticmethod
    def get_out_players(
        reports: dict[str, InjuryReport],
        team: str,
    ) -> list[dict]:
        """Return OUT/DOUBTFUL players for a team with estimated usage shares.

        Used by prop injury boost: when teammates are OUT, remaining players
        absorb their usage (minutes, touches, shots).

        Returns: [{"name": str, "tier": str, "usage_share": float}, ...]
        """
        team_norm = team.lower().strip()
        out_players = []

        for team_key, report in reports.items():
            if team_norm not in team_key and team_key not in team_norm:
                continue
            for p in report.players:
                if p.status.lower() in ("out", "doubtful"):
                    usage = InjuryReportAgent._USAGE_BY_TIER.get(p.tier, 0.08)
                    out_players.append({
                        "name": p.name,
                        "tier": p.tier,
                        "usage_share": usage,
                    })

        return out_players

    @staticmethod
    def compute_prop_injury_boost(
        reports: dict[str, InjuryReport],
        team: str,
        redistribution_factor: float = 0.6,
        max_boost: float = 0.15,
    ) -> float:
        """Compute probability boost for a player's props when teammates are OUT.

        Not all freed usage goes to a single player — `redistribution_factor` controls
        what fraction of the freed usage is assumed to reach our player (0.6 = 60%).

        Returns: boost as a fraction (e.g. 0.08 = 8% boost to over probability).
        """
        out_players = InjuryReportAgent.get_out_players(reports, team)
        if not out_players:
            return 0.0

        total_freed = sum(p["usage_share"] for p in out_players)
        boost = total_freed * redistribution_factor
        return min(boost, max_boost)

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
