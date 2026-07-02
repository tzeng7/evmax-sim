"""stats.nba.com helpers (nba_api, no auth required).

Player-ID resolution, L15 game-log fetch, and the Kalshi stat_type →
game-log column map. Consumers: the NBA props diagnostics cache
(`nba_props_cache.py`) and prop outcome resolution
(`agents/cleanup/resolver.py`).

The L15 prop *probability model* that used to live here (per-36
normalization, exponential recency decay, streak nudges, opponent
defensive adjustment) was superseded by Pinnacle anchor pricing
(`evmax/ev/prop_pricing.py`) in `edb3d7b` (2026-05-10) and removed in the
2026-07-01 drift-audit follow-up — it had no callers.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)

_NBA_API_TIMEOUT = 10  # seconds per request (nba_api timeout param)

_SEASON = "2025-26"
_LAST_N_GAMES = 15

# Kalshi stat_type → nba_api game log column (or virtual key)
STAT_COL: dict[str, str] = {
    "points":                   "PTS",
    "rebounds":                 "REB",
    "assists":                  "AST",
    "threes":                   "FG3M",
    "steals":                   "STL",
    "blocks":                   "BLK",
    "points_rebounds_assists":  "__pra__",
    "turnovers":                "TOV",
    "blocks_steals":            "__bs__",
}


# ---------------------------------------------------------------------------
# Player ID lookup
# ---------------------------------------------------------------------------

@lru_cache(maxsize=512)
def _find_player_id(player_name: str) -> Optional[int]:
    """Resolve normalized player name → NBA player ID. Cached.

    Tries full-name regex match first (avoids Mitchell vs Duncan Robinson),
    falls back to last-name-only when no first name is available.
    """
    try:
        from nba_api.stats.static import players as nba_players
    except ImportError:
        return None

    parts = player_name.split("_")
    last_name = parts[-1]

    if len(parts) >= 2:
        first_name = parts[0]
        by_full = nba_players.find_players_by_full_name(f"^{first_name}.*{last_name}$")
        active = [p for p in by_full if p["is_active"]]
        if active:
            return active[0]["id"]

    by_last = nba_players.find_players_by_last_name(f"^{last_name}$")
    active = [p for p in by_last if p["is_active"]]
    if len(active) == 1:
        return active[0]["id"]
    if len(active) > 1:
        logger.debug("nba_stats_ambiguous_player", player_name=player_name,
                     matches=[p["full_name"] for p in active])
        return active[0]["id"]
    return None


# ---------------------------------------------------------------------------
# Game log — LRU-cached, cleared daily via cache expiry key
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def _fetch_gamelog_sync(player_id: int) -> Optional[object]:
    """Fetch last-N-game log for a player. LRU-cached per process."""
    try:
        from nba_api.stats.endpoints import playergamelogs
        logs = playergamelogs.PlayerGameLogs(
            player_id_nullable=player_id,
            season_nullable=_SEASON,
            last_n_games_nullable=_LAST_N_GAMES,
            timeout=_NBA_API_TIMEOUT,
        )
        return logs.get_data_frames()[0]
    except Exception as exc:
        logger.debug("nba_stats_gamelog_failed", player_id=player_id, error=str(exc))
        return None
