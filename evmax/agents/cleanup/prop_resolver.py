"""Resolve player prop outcomes using ESPN boxscore data.

Fetches actual player stats for a given date and fills in
prop_observations.actual_value + outcome (1=over, 0=under).

ESPN endpoints used:
  Scoreboard: https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard?dates=YYYYMMDD
  Summary:    https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/summary?event={event_id}

ESPN boxscore columns are identified by the per-group `keys` list (e.g.
['minutes', 'points', 'fieldGoalsMade-fieldGoalsAttempted', ...]). Indices
shift over time, so we look stat_type up by key rather than hardcoded index.
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

# stat_type → ESPN key name (from stat_group['keys']). "threes", "pra", and
# "anytime_td" are derived and handled in _extract_stat.
#
# NBA keys live in a single `statistics` group per team. NFL keys live across
# multiple groups (passing / rushing / receiving) — fetch_player_stats MERGES
# stats per athlete across groups so a QB picks up both passing and rushing
# rows.
_STAT_KEY: dict[str, str] = {
    # NBA
    "points": "points",
    "rebounds": "rebounds",
    "assists": "assists",
    "steals": "steals",
    "blocks": "blocks",
    # NFL — yardage + counting stats sit directly in group 'keys' arrays.
    # passing group
    "passing_yards": "passingYards",
    "passing_tds": "passingTouchdowns",
    # rushing group
    "rushing_yards": "rushingYards",
    "rushing_tds": "rushingTouchdowns",
    # receiving group
    "receiving_yards": "receivingYards",
    "receiving_tds": "receivingTouchdowns",
    "receptions": "receptions",
}


def _extract_stat(
    stats: list[str],
    key_index: dict[str, int],
    stat_type: str,
) -> Optional[float]:
    def _get(key: str) -> Optional[float]:
        idx = key_index.get(key)
        if idx is None or idx >= len(stats):
            return None
        try:
            return float(stats[idx])
        except (TypeError, ValueError):
            return None

    if stat_type == "threes":
        # ESPN stores "threePointFieldGoalsMade-threePointFieldGoalsAttempted"
        # as a single "m-a" string like "3-7".
        idx = key_index.get(
            "threePointFieldGoalsMade-threePointFieldGoalsAttempted"
        )
        if idx is None or idx >= len(stats):
            return None
        try:
            return float(str(stats[idx]).split("-")[0])
        except (TypeError, ValueError):
            return None

    if stat_type == "points_rebounds_assists":
        pts = _get("points")
        reb = _get("rebounds")
        ast = _get("assists")
        if pts is None or reb is None or ast is None:
            return None
        return pts + reb + ast

    # "anytime_td" is derived post-hoc in fetch_player_stats from merged
    # rushing_tds + receiving_tds — not extractable from a single stat
    # group.
    if stat_type == "anytime_td":
        return None

    key = _STAT_KEY.get(stat_type)
    if key is None:
        return None
    return _get(key)


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

            stat_types = list(_STAT_KEY.keys()) + [
                "threes",
                "points_rebounds_assists",
            ]
            for team_entry in summary.get("boxscore", {}).get("players", []):
                # Accumulate stats PER PLAYER across every stat_group in the
                # team entry. For NBA this is a single group per team so
                # the behavior is identical to the pre-refactor pass; for
                # NFL a player's passing / rushing / receiving numbers are
                # in three separate groups and must merge.
                for stat_group in team_entry.get("statistics", []):
                    keys = stat_group.get("keys") or []
                    key_index = {k: i for i, k in enumerate(keys)}
                    for athlete in stat_group.get("athletes", []):
                        name = athlete.get("athlete", {}).get("displayName", "").lower()
                        stats_list = athlete.get("stats", [])
                        if not name or not stats_list:
                            continue
                        row = player_stats.setdefault(name, {})
                        for stat_type in stat_types:
                            val = _extract_stat(stats_list, key_index, stat_type)
                            if val is not None:
                                row[stat_type] = val

            # NFL derived: anytime_td = rushing_tds + receiving_tds. Passing
            # TDs are the thrower's stat, not the scorer's. Computed after
            # the merge so QBs with rushing TDs and receivers with rec TDs
            # both pick up their scoring TDs correctly.
            for name, row in player_stats.items():
                if "rushing_tds" in row or "receiving_tds" in row:
                    row["anytime_td"] = float(
                        (row.get("rushing_tds") or 0) + (row.get("receiving_tds") or 0)
                    )

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
