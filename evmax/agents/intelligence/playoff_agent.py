"""PlayoffAgent — detects NBA/NHL playoff context and applies series-aware adjustments.

During playoffs, game dynamics shift significantly:
- Elimination games: trailing team plays with desperation (historically ~48% win rate
  vs ~40% expected, a +8pp boost)
- Closeout games: leading team slightly under-performs expectations (complacency)
- Game 7: massive home court advantage (~78% home win rate vs ~64% regular season)
- Playoff home court: stronger than regular season across all games
- Pace: playoff games are slower, affecting totals markets

Data source (public, no auth):
  https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard

Published topic: "intelligence.playoff.{sector}"
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import httpx
import structlog

from evmax.agents.base import Agent, AgentRequest, AgentResponse

logger = structlog.get_logger(__name__)

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports"

SECTOR_SCOREBOARD_URL: dict[str, str] = {
    "nba": f"{ESPN_SCOREBOARD}/basketball/nba/scoreboard",
    "nhl": f"{ESPN_SCOREBOARD}/hockey/nhl/scoreboard",
    "wnba": f"{ESPN_SCOREBOARD}/basketball/wnba/scoreboard",
}

# Series length per sport (best-of-7 for NBA/NHL playoffs)
SERIES_LENGTH: dict[str, int] = {"nba": 7, "nhl": 7, "wnba": 5}
WINS_TO_CLINCH: dict[str, int] = {"nba": 4, "nhl": 4, "wnba": 3}


@dataclass
class PlayoffSeries:
    """Parsed playoff series context for a single game."""

    team_a: str  # home team (lowercased)
    team_b: str  # away team (lowercased)
    team_a_abbrev: str
    team_b_abbrev: str
    series_wins_a: int  # home team's series wins
    series_wins_b: int  # away team's series wins
    round_num: int  # 1=first round, 2=semis, 3=conf finals, 4=finals
    round_name: str  # e.g. "NBA Finals", "Conference Semifinals"
    sector: str

    @property
    def _clinch(self) -> int:
        return WINS_TO_CLINCH.get(self.sector, 4)

    @property
    def is_elimination_for_a(self) -> bool:
        """Home team faces elimination (away leads clinch-1 to something)."""
        return self.series_wins_b == self._clinch - 1 and self.series_wins_a < self._clinch - 1

    @property
    def is_elimination_for_b(self) -> bool:
        """Away team faces elimination."""
        return self.series_wins_a == self._clinch - 1 and self.series_wins_b < self._clinch - 1

    @property
    def is_game_7(self) -> bool:
        """Both teams one win from elimination (Game 7 / Game 5 in WNBA)."""
        c = self._clinch - 1
        return self.series_wins_a == c and self.series_wins_b == c

    @property
    def is_closeout_for_a(self) -> bool:
        """Home team can close out the series."""
        return self.series_wins_a == self._clinch - 1 and self.series_wins_b < self._clinch - 1

    @property
    def is_closeout_for_b(self) -> bool:
        """Away team can close out the series."""
        return self.series_wins_b == self._clinch - 1 and self.series_wins_a < self._clinch - 1

    @property
    def is_finals(self) -> bool:
        return self.round_num >= 4 or "final" in self.round_name.lower()

    @property
    def game_number(self) -> int:
        return self.series_wins_a + self.series_wins_b + 1

    @property
    def is_playoff(self) -> bool:
        return True


class PlayoffAgent(Agent):
    """Fetches ESPN playoff scoreboard to detect series context."""

    name = "playoff"
    description = "Detects NBA/NHL playoff series context for probability adjustments."

    def __init__(self, timeout: float = 8.0) -> None:
        super().__init__()
        self._timeout = timeout

    async def run(self, request: AgentRequest) -> AgentResponse:
        sector = request.sector.lower()
        url = SECTOR_SCOREBOARD_URL.get(sector)
        if not url:
            return AgentResponse(agent_name=self.name, sector=sector, data={})

        self.log.info("fetching_playoff_context", sector=sector)

        try:
            series_map = await self._fetch_playoff_data(url, sector)
        except Exception as e:
            self.log.warning("playoff_fetch_failed", error=str(e))
            return AgentResponse(agent_name=self.name, sector=sector, data={})

        if series_map:
            self.log.info(
                "playoff_series_detected",
                sector=sector,
                games=len(series_map),
                series=[
                    f"{s.team_a_abbrev} {s.series_wins_a}-{s.series_wins_b} {s.team_b_abbrev}"
                    for s in series_map.values()
                ],
            )

        await self.publish(f"intelligence.playoff.{sector}", series_map, request.correlation_id)
        return AgentResponse(agent_name=self.name, sector=sector, data=series_map)

    async def _fetch_playoff_data(
        self, url: str, sector: str
    ) -> dict[str, PlayoffSeries]:
        """Fetch today's scoreboard and extract playoff series data.

        Returns a dict keyed by "{team_a_lower}_vs_{team_b_lower}" for easy
        lookup during EV gap evaluation.
        """
        today = date.today()
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(
                f"{url}?dates={today:%Y%m%d}", follow_redirects=True,
            )
            resp.raise_for_status()
            data = resp.json()

        series_map: dict[str, PlayoffSeries] = {}

        for event in data.get("events", []):
            season_type = event.get("season", {}).get("type", 0)
            # ESPN season type 3 = postseason/playoffs
            if season_type != 3:
                continue

            for comp in event.get("competitions", []):
                series_info = comp.get("series", {})
                if not series_info:
                    continue

                # Extract series summary (e.g. "Series tied 1-1", "BOS leads 3-2")
                competitors = comp.get("competitors", [])
                if len(competitors) < 2:
                    continue

                # ESPN lists home first, away second
                home = competitors[0] if competitors[0].get("homeAway") == "home" else competitors[1]
                away = competitors[1] if competitors[0].get("homeAway") == "home" else competitors[0]

                home_name = home.get("team", {}).get("displayName", "").lower().strip()
                away_name = away.get("team", {}).get("displayName", "").lower().strip()
                home_abbrev = home.get("team", {}).get("abbreviation", "")
                away_abbrev = away.get("team", {}).get("abbreviation", "")

                # Series wins from the series object or competitor records
                home_series_wins = 0
                away_series_wins = 0

                # Try series.competitors first (more reliable)
                for sc in series_info.get("competitors", []):
                    sc_id = sc.get("id", "")
                    wins = int(sc.get("wins", 0))
                    if sc_id == home.get("id", ""):
                        home_series_wins = wins
                    elif sc_id == away.get("id", ""):
                        away_series_wins = wins

                # Fallback: parse series summary text
                if home_series_wins == 0 and away_series_wins == 0:
                    summary = series_info.get("summary", "")
                    home_series_wins, away_series_wins = _parse_series_summary(
                        summary, home_abbrev, away_abbrev,
                    )

                # Determine round number from notes or series type
                round_name = ""
                round_num = 1
                for note in event.get("competitions", [{}])[0].get("notes", []):
                    headline = note.get("headline", "")
                    if headline:
                        round_name = headline
                        break
                if not round_name:
                    round_name = series_info.get("title", "")

                round_num = _infer_round_number(round_name, sector)

                key = f"{home_name}_vs_{away_name}"
                series_map[key] = PlayoffSeries(
                    team_a=home_name,
                    team_b=away_name,
                    team_a_abbrev=home_abbrev,
                    team_b_abbrev=away_abbrev,
                    series_wins_a=home_series_wins,
                    series_wins_b=away_series_wins,
                    round_num=round_num,
                    round_name=round_name,
                    sector=sector,
                )

        return series_map

    @staticmethod
    def apply_adjustments(
        playoff_data: dict[str, PlayoffSeries],
        true_prob_a: float,
        true_prob_b: float,
        team_a: str,
        team_b: str,
        is_total: bool = False,
    ) -> tuple[float, float, str, bool]:
        """Apply playoff-aware probability adjustments.

        Returns (adjusted_prob_a, adjusted_prob_b, notes_str, is_playoff_game).

        Adjustment magnitudes (empirically grounded):
        - Elimination game boost:       +3.0% to trailing team
        - Closeout discount:            -1.5% to leading team
        - Game 7 home boost:            +2.0% to home team (on top of regular HCA)
        - Playoff home court bump:      +1.5% to home team (playoffs > regular season)
        - Finals intensity:             +0.5% to home team
        - Totals pace adjustment:       applied separately via is_total flag
        """
        if not playoff_data:
            return true_prob_a, true_prob_b, "", False

        team_a_norm = team_a.lower().strip()
        team_b_norm = team_b.lower().strip()

        # Find matching series
        series: Optional[PlayoffSeries] = None
        for key, s in playoff_data.items():
            if (team_a_norm in s.team_a or s.team_a in team_a_norm or
                team_a_norm in s.team_b or s.team_b in team_a_norm):
                if (team_b_norm in s.team_b or s.team_b in team_b_norm or
                    team_b_norm in s.team_a or s.team_a in team_b_norm):
                    series = s
                    break

        if series is None:
            return true_prob_a, true_prob_b, "", False

        # Determine which team in the series is team_a and team_b in our eval
        a_is_home = team_a_norm in series.team_a or series.team_a in team_a_norm

        notes: list[str] = []
        adj_a = 0.0
        adj_b = 0.0

        series_str = f"G{series.game_number} ({series.series_wins_a}-{series.series_wins_b})"

        # --- Playoff home court bump (applies to all playoff games) ---
        # NBA playoff HCA is ~67% vs ~60% regular season = +3.5pp over baseline
        # Our Elo model already has regular season HCA baked in, so add the
        # incremental playoff premium of ~1.5pp
        if a_is_home:
            adj_a += 0.015
            notes.append(f"playoff_hca:+1.5%→{team_a[:12]}")
        else:
            adj_b += 0.015
            notes.append(f"playoff_hca:+1.5%→{team_b[:12]}")

        # --- Elimination game boost ---
        if series.is_game_7:
            # Game 7: both teams desperate, home team gets massive edge
            # Historical Game 7 home win rate: ~78% vs ~64% regular season
            # Additional +2% to home team (on top of playoff HCA above)
            if a_is_home:
                adj_a += 0.020
                notes.append(f"game7_home:+2.0%→{team_a[:12]}")
            else:
                adj_b += 0.020
                notes.append(f"game7_home:+2.0%→{team_b[:12]}")
        elif series.is_elimination_for_a if a_is_home else series.is_elimination_for_b:
            # Team A faces elimination
            adj_a += 0.030
            notes.append(f"elim_boost:+3.0%→{team_a[:12]}")
        elif series.is_elimination_for_b if a_is_home else series.is_elimination_for_a:
            # Team B faces elimination
            adj_b += 0.030
            notes.append(f"elim_boost:+3.0%→{team_b[:12]}")
        elif series.is_closeout_for_a if a_is_home else series.is_closeout_for_b:
            # Team A can close out — slight complacency discount
            adj_a -= 0.015
            notes.append(f"closeout:-1.5%→{team_a[:12]}")
        elif series.is_closeout_for_b if a_is_home else series.is_closeout_for_a:
            # Team B can close out
            adj_b -= 0.015
            notes.append(f"closeout:-1.5%→{team_b[:12]}")

        # --- Finals intensity ---
        if series.is_finals:
            if a_is_home:
                adj_a += 0.005
            else:
                adj_b += 0.005
            notes.append(f"finals:{series.round_name[:20]}")

        if adj_a == 0.0 and adj_b == 0.0:
            return true_prob_a, true_prob_b, f"playoff:{series_str}", True

        # Apply adjustments and renormalize
        new_a = max(0.02, min(0.98, true_prob_a + adj_a - adj_b))
        new_b = max(0.02, min(0.98, true_prob_b + adj_b - adj_a))

        total = new_a + new_b
        new_a /= total
        new_b /= total

        note_str = f"playoff:{series_str}; " + "; ".join(notes)
        return new_a, new_b, note_str, True

    @staticmethod
    def pace_adjustment(sector: str) -> float:
        """Return multiplicative pace factor for totals in playoff games.

        NBA playoffs historically see ~3-5% slower pace than regular season
        due to tighter defensive schemes and longer possessions.
        Returns a multiplier < 1.0 to reduce total projections.
        """
        pace_factors = {
            "nba": 0.95,   # -5% pace
            "nhl": 0.97,   # -3% pace (tighter checking)
            "wnba": 0.96,  # -4% pace
        }
        return pace_factors.get(sector, 1.0)


def _parse_series_summary(summary: str, home_abbrev: str, away_abbrev: str) -> tuple[int, int]:
    """Parse ESPN series summary like 'BOS leads 3-2' or 'Series tied 1-1'."""
    import re
    # Pattern: "TEAM leads X-Y" or "Series tied X-X"
    tied = re.search(r"tied\s+(\d+)-(\d+)", summary, re.IGNORECASE)
    if tied:
        w = int(tied.group(1))
        return w, w

    leads = re.search(r"(\w+)\s+leads\s+(\d+)-(\d+)", summary, re.IGNORECASE)
    if leads:
        leader = leads.group(1).upper()
        w1 = int(leads.group(2))
        w2 = int(leads.group(3))
        if leader == home_abbrev.upper():
            return w1, w2
        else:
            return w2, w1

    return 0, 0


def _infer_round_number(round_name: str, sector: str) -> int:
    """Infer playoff round number from ESPN round/note name."""
    name = round_name.lower()

    # WNBA has fewer rounds (semis → finals = rounds 2, 3)
    if sector == "wnba":
        if "final" in name and "semi" not in name:
            return 3
        if "semi" in name:
            return 2
        return 1

    if "final" in name and "conference" not in name and "semi" not in name:
        return 4  # NBA Finals / Stanley Cup Final
    if "conference final" in name or "conf. final" in name:
        return 3
    if "semifinal" in name or "semi-final" in name or "conference semi" in name:
        return 2
    if "first round" in name or "1st round" in name or "round 1" in name:
        return 1
    return 1
