"""Resolve player prop outcomes using ESPN boxscore data.

Fetches actual player stats for a given date and fills in
prop_observations.actual_value + outcome (1=over, 0=under).

ESPN endpoints used:
  Scoreboard: https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard?dates=YYYYMMDD
  Summary:    https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/summary?event={event_id}

Stat column indices in ESPN boxscore athlete.stats array (NBA):
  0=MIN  1=FG(m-a)  2=3PT(m-a)  3=FT(m-a)  4=OREB  5=DREB  6=REB
  7=AST  8=STL  9=BLK  10=TO  11=PF  12=+/-  13=PTS
"""

from __future__ import annotations

import re
from datetime import date
from typing import Optional

import httpx
import structlog

logger = structlog.get_logger(__name__)

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"

_SECTOR_ESPN = {
    "nba": ("basketball", "nba"),
    "nfl": ("football", "nfl"),
}

# stat_type → how to extract a numeric value from ESPN stats array
# Each entry is either an int (direct index) or a callable(stats_list) -> float
_STAT_EXTRACTORS: dict[str, int | str] = {
    "points": 13,
    "rebounds": 6,
    "assists": 7,
    "steals": 8,
    "blocks": 9,
    "threes": "3pt_made",   # special: parse "2-5" → 2
    "points_rebounds_assists": "pra",
}


def _extract_stat(stats: list[str], stat_type: str) -> Optional[float]:
    extractor = _STAT_EXTRACTORS.get(stat_type)
    if extractor is None:
        return None
    if isinstance(extractor, int):
        try:
            return float(stats[extractor])
        except (IndexError, ValueError):
            return None
    if extractor == "3pt_made":
        try:
            made = stats[2].split("-")[0]
            return float(made)
        except (IndexError, ValueError):
            return None
    if extractor == "pra":
        try:
            return float(stats[13]) + float(stats[6]) + float(stats[7])
        except (IndexError, ValueError):
            return None
    return None


def fetch_player_stats(sector: str, game_date: date) -> dict[str, dict[str, float]]:
    """
    Fetch all player stats for a sector on a given date.

    Returns: dict[normalized_player_name → dict[stat_type → value]]
    e.g. {"lebron james": {"points": 28.0, "rebounds": 7.0, "assists": 9.0, ...}}
    """
    sport_league = _SECTOR_ESPN.get(sector.lower())
    if not sport_league:
        return {}

    sport, league = sport_league
    date_str = game_date.strftime("%Y%m%d")

    try:
        with httpx.Client(timeout=15.0) as client:
            sb_url = f"{ESPN_BASE}/{sport}/{league}/scoreboard"
            sb = client.get(sb_url, params={"dates": date_str}).json()
    except Exception as e:
        logger.warning("prop_resolver_scoreboard_failed", sector=sector, date=game_date, error=str(e))
        return {}

    event_ids = [ev["id"] for ev in sb.get("events", [])]
    if not event_ids:
        return {}

    player_stats: dict[str, dict[str, float]] = {}

    with httpx.Client(timeout=15.0) as client:
        for event_id in event_ids:
            try:
                summary = client.get(
                    f"{ESPN_BASE}/{sport}/{league}/summary",
                    params={"event": event_id},
                ).json()
            except Exception as e:
                logger.warning("prop_resolver_summary_failed", event_id=event_id, error=str(e))
                continue

            for team_entry in summary.get("boxscore", {}).get("players", []):
                for stat_group in team_entry.get("statistics", []):
                    for athlete in stat_group.get("athletes", []):
                        name = athlete.get("athlete", {}).get("displayName", "").lower()
                        stats_list = athlete.get("stats", [])
                        if not name or not stats_list:
                            continue
                        row: dict[str, float] = {}
                        for stat_type in _STAT_EXTRACTORS:
                            val = _extract_stat(stats_list, stat_type)
                            if val is not None:
                                row[stat_type] = val
                        if row:
                            player_stats[name] = row

    logger.info("prop_resolver_fetched", sector=sector, date=str(game_date), players=len(player_stats))
    return player_stats


def _normalize_for_match(name: str) -> str:
    """Normalize a player name for matching: lowercase, strip accents, underscores→spaces."""
    import unicodedata
    name = name.lower().strip().replace("_", " ")
    # Strip accents: dončić → doncic, müller → muller
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    # Strip suffixes like Jr., Sr., III, II
    name = re.sub(r"\s+(jr\.?|sr\.?|iii|ii|iv)$", "", name, flags=re.IGNORECASE)
    return name.strip()


def resolve_prop_observations(sector: str, game_date: date) -> dict[str, int]:
    """
    Fetch ESPN boxscores for game_date and update prop_observations rows
    with actual_value + outcome.

    Returns: {"resolved": N, "unmatched": M}
    """
    from evmax.agents.cleanup.db import get_connection

    player_stats = fetch_player_stats(sector, game_date)
    if not player_stats:
        return {"resolved": 0, "unmatched": 0}

    # Build normalized lookup: "lebron james" → stats dict
    # Handles underscores, accents, suffixes
    norm_stats: dict[str, dict[str, float]] = {}
    for raw_name, stats_dict in player_stats.items():
        norm_stats[_normalize_for_match(raw_name)] = stats_dict

    conn = get_connection()
    rows = conn.execute(
        """SELECT id, player_name, stat_type, line
           FROM prop_observations
           WHERE sector = ? AND event_date = ? AND outcome IS NULL""",
        (sector, game_date.isoformat()),
    ).fetchall()

    resolved = unmatched = 0
    for row in rows:
        player_key = _normalize_for_match(row["player_name"])

        # 1. Exact normalized match
        stats = norm_stats.get(player_key)

        # 2. Last-name fallback (only if unique)
        if stats is None:
            last_name = player_key.split()[-1]
            candidates = [v for k, v in norm_stats.items() if k.split()[-1] == last_name]
            if len(candidates) == 1:
                stats = candidates[0]

        # 3. First-initial + last-name fallback (e.g. "s gilgeous-alexander" matches "shai gilgeous-alexander")
        if stats is None and len(player_key.split()) >= 2:
            first_initial = player_key[0]
            last_parts = " ".join(player_key.split()[1:])
            stats = next(
                (v for k, v in norm_stats.items()
                 if k.split()[0][0] == first_initial
                 and " ".join(k.split()[1:]) == last_parts),
                None,
            )

        if stats is None:
            unmatched += 1
            continue

        actual = stats.get(row["stat_type"])
        if actual is None:
            unmatched += 1
            continue

        outcome = 1 if actual > row["line"] else 0
        conn.execute(
            """UPDATE prop_observations
               SET actual_value = ?, outcome = ?, resolved_at = datetime('now')
               WHERE id = ?""",
            (actual, outcome, row["id"]),
        )
        resolved += 1

    conn.commit()
    conn.close()

    logger.info(
        "prop_observations_resolved",
        sector=sector,
        date=str(game_date),
        resolved=resolved,
        unmatched=unmatched,
    )
    return {"resolved": resolved, "unmatched": unmatched}
