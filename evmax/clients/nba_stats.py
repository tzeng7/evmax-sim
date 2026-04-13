"""NBA player prop probability model via stats.nba.com (nba_api, no auth required).

Model pipeline (per prop line):
  1. Fetch player's L15 game log — per-minute normalise stats to per-36 rate.
  2. Apply exponential decay so recent games count more (λ=0.85 → game 10
     has ≈ 0.20× the weight of last night's game).
  3. Streak bonus/penalty: if player has hit/missed the line in ≥4 of his last
     5 games, shift probability +5% / -5% (hot/cold streak signal).
  4. Fetch tonight's opponent from the ESPN scoreboard.
  5. Apply stat-specific opponent defensive adjustment:
       • points  → opponent defensive rating vs league average
       • rebounds → opponent rebounding rate vs league average
       • threes  → opponent 3-PT allowed rate vs league average
       • assists → pace adjustment only
     Adjustments are capped at ±15% so one great/terrible defense
     can't single-handedly swing the probability.

All remote calls are cached within the process — the nba_api LRU cache
covers game logs, and team stats are session-cached via _TEAM_STATS_CACHE.
"""

from __future__ import annotations

import asyncio
import time
from functools import lru_cache
from typing import Optional

import structlog
import numpy as np
from scipy.stats import norm

logger = structlog.get_logger(__name__)

# Cap concurrent nba_api calls — stats.nba.com throttles at ~3–5 req/s
_NBA_API_SEM = asyncio.Semaphore(3)
_NBA_API_TIMEOUT = 10  # seconds per request (nba_api timeout param)

_SEASON = "2025-26"
_LAST_N_GAMES = 15
_MIN_GAMES = 5           # need at least this many data points
_DECAY = 0.85            # exponential recency decay per game (most-recent = weight 1)
_MAX_OPP_ADJ = 0.15      # cap opponent adjustment at ±15%

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

# Session-level cache for league team stats (expires every 6 hours)
_TEAM_STATS_CACHE: dict[str, tuple[float, dict]] = {}
_TEAM_STATS_TTL = 6 * 3600

# ESPN scoreboard cache per date
_SCHEDULE_CACHE: dict[str, tuple[float, list[dict]]] = {}
_SCHEDULE_TTL = 300  # 5 min — games can be postponed


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


# ---------------------------------------------------------------------------
# League team defensive stats (pace, opp shooting, rebounding)
# ---------------------------------------------------------------------------

def _fetch_team_stats() -> dict[str, dict]:
    """Fetch all teams' season stats for pace and defensive context.

    Returns {team_abbreviation: {pace, def_rating, opp_fg3_pct, opp_reb_pct}}
    Session-cached with 6-hour TTL.
    """
    now = time.monotonic()
    cached = _TEAM_STATS_CACHE.get("all")
    if cached and now - cached[0] < _TEAM_STATS_TTL:
        return cached[1]

    try:
        from nba_api.stats.endpoints import leaguedashteamstats
        import pandas as pd

        # General stats: pace, defensive rating
        df_gen = leaguedashteamstats.LeagueDashTeamStats(
            season=_SEASON,
            per_mode_simple="PerGame",
            measure_type_simple_defense="Base",
            timeout=_NBA_API_TIMEOUT,
        ).get_data_frames()[0]

        # Opponent shooting stats: opp 3PT makes/attempts per game
        df_opp = leaguedashteamstats.LeagueDashTeamStats(
            season=_SEASON,
            per_mode_simple="PerGame",
            measure_type_simple_defense="Opponent",
            timeout=_NBA_API_TIMEOUT,
        ).get_data_frames()[0]

        result: dict[str, dict] = {}
        for _, row in df_gen.iterrows():
            abbrev = row.get("TEAM_ABBREVIATION", "")
            if not abbrev:
                continue
            result[abbrev] = {
                "def_rating": float(row.get("DEF_RATING", 110.0)),
                "pace": float(row.get("PACE", 98.0)),
            }

        # Merge opponent shooting
        for _, row in df_opp.iterrows():
            abbrev = row.get("TEAM_ABBREVIATION", "")
            if abbrev in result:
                result[abbrev]["opp_fg3m"] = float(row.get("OPP_FG3M", 12.0))
                result[abbrev]["opp_fg3a"] = float(row.get("OPP_FG3A", 34.0))
                result[abbrev]["opp_fg3_pct"] = (
                    float(row.get("OPP_FG3M", 12.0)) / max(float(row.get("OPP_FG3A", 34.0)), 1)
                )
                result[abbrev]["opp_reb"] = float(row.get("OPP_REB", 42.0))
                result[abbrev]["opp_oreb"] = float(row.get("OPP_OREB", 10.0))
                result[abbrev]["opp_pts"] = float(row.get("OPP_PTS", 110.0))

        if result:
            _TEAM_STATS_CACHE["all"] = (now, result)
            logger.debug("nba_stats_team_stats_fetched", n_teams=len(result))
        return result

    except Exception as exc:
        logger.debug("nba_stats_team_stats_failed", error=str(exc))
        return {}


def _league_averages(team_stats: dict[str, dict]) -> dict:
    """Compute league averages across all teams from the fetched stats."""
    if not team_stats:
        return {
            "def_rating": 112.0,
            "pace": 98.5,
            "opp_fg3_pct": 0.362,
            "opp_fg3m": 12.5,
            "opp_reb": 43.0,
            "opp_oreb": 10.5,
            "opp_pts": 112.0,
        }
    avgs: dict = {}
    for key in ("def_rating", "pace", "opp_fg3_pct", "opp_fg3m", "opp_reb", "opp_oreb", "opp_pts"):
        vals = [v[key] for v in team_stats.values() if key in v]
        avgs[key] = float(np.mean(vals)) if vals else 0.0
    return avgs


# ---------------------------------------------------------------------------
# Tonight's schedule lookup (ESPN, no auth)
# ---------------------------------------------------------------------------

def _fetch_todays_schedule_sync(game_date: str) -> list[dict]:
    """Fetch today's NBA schedule from ESPN. Returns list of matchup dicts.

    Each dict: {home_abbrev, away_abbrev, home_name, away_name}
    Session-cached per date with 5-min TTL.
    """
    now = time.monotonic()
    cached = _SCHEDULE_CACHE.get(game_date)
    if cached and now - cached[0] < _SCHEDULE_TTL:
        return cached[1]

    try:
        import httpx
        espn_date = game_date.replace("-", "")
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
        r = httpx.get(url, params={"dates": espn_date}, timeout=8.0)
        r.raise_for_status()
        data = r.json()

        matchups = []
        for event in data.get("events", []):
            comps = event.get("competitions", [])
            if not comps:
                continue
            competitors = comps[0].get("competitors", [])
            home = next((c for c in competitors if c.get("homeAway") == "home"), None)
            away = next((c for c in competitors if c.get("homeAway") == "away"), None)
            if home and away:
                matchups.append({
                    "home_abbrev": home.get("team", {}).get("abbreviation", ""),
                    "away_abbrev": away.get("team", {}).get("abbreviation", ""),
                    "home_name":   home.get("team", {}).get("displayName", ""),
                    "away_name":   away.get("team", {}).get("displayName", ""),
                })

        _SCHEDULE_CACHE[game_date] = (now, matchups)
        return matchups

    except Exception as exc:
        logger.debug("nba_stats_schedule_failed", date=game_date, error=str(exc))
        return []


def _player_team_abbrev(df) -> Optional[str]:
    """Extract the player's team abbreviation from the most recent MATCHUP.

    Format is 'NYK vs. MIL' (home) or 'NYK @ LAL' (away).
    The player's team is always the first part.
    """
    if df is None or df.empty or "MATCHUP" not in df.columns:
        return None
    matchup = str(df.iloc[0]["MATCHUP"])
    return matchup.split(" ")[0].upper()


def _find_opponent(player_team: str, matchups: list[dict]) -> Optional[str]:
    """Given a team abbreviation and tonight's matchups, return the opponent."""
    team = player_team.upper()
    for m in matchups:
        if m["home_abbrev"] == team:
            return m["away_abbrev"]
        if m["away_abbrev"] == team:
            return m["home_abbrev"]
    return None


# ---------------------------------------------------------------------------
# Opponent adjustment factor (capped at ±15%)
# ---------------------------------------------------------------------------

def _opponent_adjustment(
    stat_type: str,
    opp_stats: dict,
    league_avg: dict,
) -> float:
    """Return a multiplicative adjustment for opponent defensive strength.

    A value of 1.05 means this opponent makes the stat 5% easier to achieve.
    A value of 0.92 means this opponent makes it 8% harder.
    Capped at [1 - MAX_ADJ, 1 + MAX_ADJ].

    Adjustments:
      points   → opponent defensive rating (lower = harder to score)
      rebounds → opponent offensive rebounding rate (higher = fewer boards available)
      threes   → opponent 3PT allowed percentage vs league average
      assists  → opponent pace factor (more possessions = more assists)
    """
    adj = 1.0
    try:
        if stat_type == "points":
            # Lower def_rating = better defense → harder to score
            opp_def = opp_stats.get("def_rating", league_avg["def_rating"])
            lg_def  = league_avg["def_rating"]
            if lg_def > 0:
                # +1% probability per point worse than average defense
                adj = 1.0 + (opp_def - lg_def) / lg_def

        elif stat_type == "rebounds":
            # Higher opp_oreb = opponent crashes glass → fewer defensive boards for player
            opp_oreb = opp_stats.get("opp_oreb", league_avg["opp_oreb"])
            lg_oreb  = league_avg["opp_oreb"]
            if lg_oreb > 0:
                adj = 1.0 - (opp_oreb - lg_oreb) / lg_oreb

        elif stat_type == "threes":
            # Higher opp_fg3_pct allowed = porous 3PT defense → easier to make threes
            opp_3pct = opp_stats.get("opp_fg3_pct", league_avg["opp_fg3_pct"])
            lg_3pct  = league_avg["opp_fg3_pct"]
            if lg_3pct > 0:
                adj = 1.0 + (opp_3pct - lg_3pct) / lg_3pct

        elif stat_type == "assists":
            # Faster pace = more possessions = more assist opportunities
            opp_pace = opp_stats.get("pace", league_avg["pace"])
            lg_pace  = league_avg["pace"]
            if lg_pace > 0:
                adj = opp_pace / lg_pace

        elif stat_type == "points_rebounds_assists":
            # Blend def_rating + pace
            opp_def  = opp_stats.get("def_rating", league_avg["def_rating"])
            opp_pace = opp_stats.get("pace", league_avg["pace"])
            lg_def   = league_avg["def_rating"]
            lg_pace  = league_avg["pace"]
            pts_adj  = 1.0 + (opp_def - lg_def) / lg_def if lg_def > 0 else 1.0
            pace_adj = opp_pace / lg_pace if lg_pace > 0 else 1.0
            adj = (pts_adj + pace_adj) / 2

    except Exception:
        adj = 1.0

    # Cap adjustment
    return max(1.0 - _MAX_OPP_ADJ, min(1.0 + _MAX_OPP_ADJ, adj))


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _compute_prob_sync(
    player_id: int,
    stat_type: str,
    threshold: float,
    game_date: Optional[str] = None,
) -> Optional[tuple[float, int]]:
    """Compute P(stat >= threshold) with recency weighting and opponent context.

    Steps:
      1. Fetch L15 game log, build per-36 stat series.
      2. Apply exponential decay weights (recent games count more).
      3. Detect hot/cold streak vs this specific threshold.
      4. Look up tonight's opponent and apply defensive adjustment.
      5. Fit normal distribution, return (probability, n_games).
    """
    df = _fetch_gamelog_sync(player_id)
    if df is None or len(df) < _MIN_GAMES:
        return None

    col = STAT_COL.get(stat_type)
    if col is None:
        return None

    # --- Build raw stat series (cap to most recent _LAST_N_GAMES) ---
    # nba_api's last_n_games_nullable is advisory; the API sometimes returns
    # more rows (entire season). Index 0 is most recent, so slicing [:N]
    # keeps the newest games.
    df = df.head(_LAST_N_GAMES)
    if col == "__pra__":
        raw = (df["PTS"] + df["REB"] + df["AST"]).astype(float).values
    elif col == "__bs__":
        raw = (df["BLK"] + df["STL"]).astype(float).values
    else:
        if col not in df.columns:
            return None
        raw = df[col].astype(float).values

    n_games = len(raw)

    # --- Per-36-minute normalization ---
    # Kalshi "over N" resolves via strict inequality (actual > line), and all
    # NBA prop lines are integer. Continuity correction for strict-over on a
    # discrete stat is P(X >= N+1) ≈ P(Z > N + 0.5). We apply the +0.5 to the
    # raw threshold BEFORE per-36 scaling so it lives in the same space as
    # the sample distribution.
    cc_threshold = float(threshold) + 0.5
    if "MIN" in df.columns:
        minutes = df["MIN"].astype(float).values
        avg_min = float(np.mean(minutes))
        if avg_min >= 5.0:
            per36 = raw / np.maximum(minutes, 1.0) * 36.0
            eff_threshold = cc_threshold / avg_min * 36.0
        else:
            per36 = raw
            eff_threshold = cc_threshold
    else:
        per36 = raw
        eff_threshold = cc_threshold

    # --- Exponential decay weights (index 0 = most recent) ---
    weights = np.array([_DECAY ** i for i in range(n_games)], dtype=float)
    weights /= weights.sum()

    wmean = float(np.dot(weights, per36))
    wvar  = float(np.dot(weights, (per36 - wmean) ** 2))
    wstd  = max(float(np.sqrt(wvar)), 0.5)

    # --- Streak adjustment (last 5 games vs threshold) ---
    # Use RAW values (not per-36) to match the actual threshold the market uses
    last5 = raw[:5]
    last5_hits = int(np.sum(last5 >= threshold))
    if last5_hits >= 4:
        streak_adj = 0.05   # hot streak: nudge prob up
    elif last5_hits <= 1:
        streak_adj = -0.05  # cold streak: nudge prob down
    else:
        streak_adj = 0.0

    # --- Base probability (continuity correction already baked into eff_threshold) ---
    base_prob = float(1 - norm.cdf(eff_threshold, wmean, wstd))
    base_prob = max(0.01, min(0.99, base_prob))

    # --- Opponent defensive adjustment ---
    opp_adj = 1.0
    if game_date:
        try:
            matchups = _fetch_todays_schedule_sync(game_date)
            player_team = _player_team_abbrev(df)
            if player_team and matchups:
                opponent = _find_opponent(player_team, matchups)
                if opponent:
                    team_stats = _fetch_team_stats()
                    league_avg = _league_averages(team_stats)
                    opp_stats  = team_stats.get(opponent, {})
                    opp_adj    = _opponent_adjustment(stat_type, opp_stats, league_avg)
                    logger.debug(
                        "nba_stats_opp_adj",
                        player_team=player_team,
                        opponent=opponent,
                        stat=stat_type,
                        adj=round(opp_adj, 3),
                    )
        except Exception as exc:
            logger.debug("nba_stats_opp_lookup_failed", error=str(exc))

    # Combine: base × opponent adjustment + streak, then clamp
    prob = base_prob * opp_adj + streak_adj
    prob = max(0.01, min(0.99, prob))

    logger.debug(
        "nba_stats_prob_detail",
        stat=stat_type,
        threshold=threshold,
        wmean=round(wmean, 2),
        wstd=round(wstd, 2),
        base_prob=round(base_prob, 3),
        opp_adj=round(opp_adj, 3),
        streak_adj=streak_adj,
        final_prob=round(prob, 3),
        n_games=n_games,
        last5_hits=last5_hits,
    )

    return prob, n_games


async def get_prop_true_prob(
    player_name: str,
    stat_type: str,
    threshold: float,
    game_date: Optional[str] = None,
) -> Optional[tuple[float, int]]:
    """Async: compute P(player stat >= threshold) with recency + opponent context.

    Returns (probability, n_games_in_sample), or None if player/stat unavailable.
    Semaphore-limited to _NBA_API_SEM concurrent requests to avoid rate-limiting
    stats.nba.com. Each executor call has _NBA_API_TIMEOUT applied at the HTTP layer.
    """
    async with _NBA_API_SEM:
        loop = asyncio.get_event_loop()
        player_id = await loop.run_in_executor(None, _find_player_id, player_name)
        if player_id is None:
            logger.debug("nba_stats_player_not_found", player=player_name)
            return None
        result = await loop.run_in_executor(
            None, _compute_prob_sync, player_id, stat_type, threshold, game_date
        )
        return result
