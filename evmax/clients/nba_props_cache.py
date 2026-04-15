"""Daily NBA player stats cache for instant prop probability computation.

Pre-fetches L15 game logs + team defensive stats from stats.nba.com once per day,
stores to disk (data/nba_props_cache.json). During scans, prop probabilities are
computed from cached data with zero API calls — the full model (recency-weighted
mean, opponent adjustment, streak detection) runs locally in <1ms per prop.

Usage:
    # Daily refresh (run once, e.g. morning cron or first scan of the day):
    await refresh_props_cache()

    # During scan (instant):
    prob, n_games = compute_prop_prob_cached("lebron_james", "points", 25.5, "2026-04-10")
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import structlog
from scipy.stats import norm

logger = structlog.get_logger(__name__)

_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "nba_props_cache.json"
_CACHE_TTL = 8 * 3600  # 8 hours — refresh once per day is fine

# Model params (match nba_stats.py)
_LAST_N_GAMES = 15
_MIN_GAMES = 5
_DECAY = 0.85
_MAX_OPP_ADJ = 0.15

# Stat column mapping
_MIN_VOL_THRESHOLD = 1.5  # flag games with minutes > mean ± 1.5σ
_VOL_DOWNWEIGHT = 0.25     # volatile games get 25% of their normal weight


@dataclass
class PropResult:
    """Result from compute_prop_prob_cached with volatility metadata."""
    prob: float
    n_games: int
    minutes_volatile: bool = False     # True if recent games have abnormal minutes
    minutes_cv: float = 0.0            # coefficient of variation (std/mean) of minutes
    volatile_games: int = 0            # count of games flagged as outlier-minutes
    avg_minutes: float = 0.0           # L15 average minutes


STAT_COL: dict[str, str] = {
    "points": "PTS",
    "rebounds": "REB",
    "assists": "AST",
    "threes": "FG3M",
    "steals": "STL",
    "blocks": "BLK",
    "points_rebounds_assists": "__pra__",
    "turnovers": "TOV",
    "blocks_steals": "__bs__",
}

# In-memory cache (loaded from disk on first access)
_mem_cache: dict | None = None
_mem_cache_time: float = 0


def _load_cache() -> dict | None:
    """Load cache from disk into memory."""
    global _mem_cache, _mem_cache_time
    if _mem_cache is not None and (time.monotonic() - _mem_cache_time) < _CACHE_TTL:
        return _mem_cache
    try:
        if _CACHE_PATH.exists():
            data = json.loads(_CACHE_PATH.read_text())
            age = time.time() - data.get("fetched_at", 0)
            if age < _CACHE_TTL:
                _mem_cache = data
                _mem_cache_time = time.monotonic()
                return data
    except Exception as e:
        logger.debug("nba_props_cache_load_failed", error=str(e))
    return None


def _save_cache(data: dict) -> None:
    """Write cache to disk."""
    global _mem_cache, _mem_cache_time
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(data))
        _mem_cache = data
        _mem_cache_time = time.monotonic()
    except Exception as e:
        logger.warning("nba_props_cache_save_failed", error=str(e))


async def refresh_props_cache(
    force: bool = False,
    player_names: list[str] | None = None,
) -> int:
    """Fetch L15 game logs for NBA players + team stats. Save to disk.

    If player_names is provided, only fetch those players (used during scan
    to cache only players with active Kalshi prop markets). Otherwise fetches
    all players from the existing cache or skips if fresh.

    Only hits stats.nba.com if cache is stale (>8h) or force=True.
    Returns number of players cached.
    """
    if not force:
        existing = _load_cache()
        if existing is not None:
            n = len(existing.get("players", {}))
            logger.info("nba_props_cache_fresh", players=n,
                        age_h=round((time.time() - existing.get("fetched_at", 0)) / 3600, 1))
            return n

    logger.info("nba_props_cache_refreshing",
                targeted=len(player_names) if player_names else 0)

    # Fetch player game logs (targeted or all)
    players_data = await _fetch_player_stats_async(player_names)

    # Merge with existing cache if doing a targeted refresh
    existing = _load_cache()
    if existing and player_names:
        merged = existing.get("players", {})
        merged.update(players_data)
        players_data = merged

    loop = asyncio.get_event_loop()
    team_stats = await loop.run_in_executor(None, _fetch_team_stats_sync)
    league_avg = _league_averages(team_stats)

    from datetime import date
    today = date.today().isoformat()
    schedule = await loop.run_in_executor(None, _fetch_schedule_sync, today)

    cache_data = {
        "fetched_at": time.time(),
        "date": today,
        "players": players_data,
        "team_stats": team_stats,
        "league_avg": league_avg,
        "schedule": schedule,
    }
    _save_cache(cache_data)
    logger.info("nba_props_cache_saved", players=len(players_data))
    return len(players_data)


# Semaphore for stats.nba.com (3 concurrent to avoid rate-limiting)
_SEM = asyncio.Semaphore(3)
_SEASON = "2025-26"
_TIMEOUT = 10


async def _fetch_player_stats_async(
    player_names: list[str] | None = None,
) -> dict[str, dict]:
    """Fetch L15 game logs for players. Uses per-player endpoint (reliable).

    If player_names is given, only fetch those. Otherwise fetch all active
    players who have played recently.
    """
    try:
        from nba_api.stats.static import players as nba_players
    except ImportError:
        logger.warning("nba_api_not_installed")
        return {}

    # Resolve player name → ID
    if player_names:
        # Targeted: only players with Kalshi prop markets
        from evmax.clients.nba_stats import _find_player_id
        targets = []
        for name in set(player_names):
            pid = _find_player_id(name)
            if pid:
                targets.append((name, pid))
    else:
        # All active players
        all_active = [p for p in nba_players.get_active_players()]
        targets = [
            (p["full_name"].lower().replace(" ", "_"), p["id"])
            for p in all_active
        ]

    logger.info("nba_props_fetching_players", count=len(targets))

    # Fetch game logs concurrently (semaphore-limited)
    async def _fetch_one(name_key: str, player_id: int) -> tuple[str, dict | None]:
        async with _SEM:
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None, _fetch_single_player_sync, player_id,
            )
            return name_key, data

    results = await asyncio.gather(
        *(_fetch_one(name, pid) for name, pid in targets),
        return_exceptions=True,
    )

    players: dict[str, dict] = {}
    for r in results:
        if isinstance(r, Exception):
            continue
        name_key, data = r
        if data is not None:
            players[name_key] = data

    logger.info("nba_props_players_fetched", count=len(players))
    return players


def _fetch_single_player_sync(player_id: int) -> dict | None:
    """Fetch L15 game log for a single player. Returns parsed dict or None."""
    try:
        from nba_api.stats.endpoints import playergamelog

        logs = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=_SEASON,
            season_type_all_star="Regular Season",
            timeout=_TIMEOUT,
        )
        df = logs.get_data_frames()[0]

        if df is None or len(df) < _MIN_GAMES:
            return None

        # Take last N games (already sorted most recent first)
        df = df.head(_LAST_N_GAMES)

        stat_cols = ["PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV", "MIN"]
        stats: dict[str, list[float]] = {}
        for col in stat_cols:
            if col in df.columns:
                stats[col] = [float(v) for v in df[col].values]

        if not stats.get("PTS"):
            return None

        # Extract player name and team from MATCHUP
        player_name = ""
        if "PLAYER_NAME" in df.columns:
            player_name = str(df.iloc[0]["PLAYER_NAME"])

        matchup = str(df.iloc[0].get("MATCHUP", ""))
        team_abbrev = matchup.split(" ")[0].upper() if matchup else ""

        return {
            "player_name": player_name,
            "player_id": int(player_id),
            "team": team_abbrev,
            "n_games": len(df),
            "stats": stats,
        }
    except Exception as e:
        logger.debug("nba_props_player_fetch_failed", player_id=player_id, error=str(e))
        return None


def _fetch_team_stats_sync() -> dict[str, dict]:
    """Fetch team defensive stats for opponent adjustment."""
    try:
        from nba_api.stats.endpoints import leaguedashteamstats
        _SEASON = "2025-26"
        _TIMEOUT = 12

        df_gen = leaguedashteamstats.LeagueDashTeamStats(
            season=_SEASON,
            per_mode_simple="PerGame",
            measure_type_simple_defense="Base",
            timeout=_TIMEOUT,
        ).get_data_frames()[0]

        df_opp = leaguedashteamstats.LeagueDashTeamStats(
            season=_SEASON,
            per_mode_simple="PerGame",
            measure_type_simple_defense="Opponent",
            timeout=_TIMEOUT,
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

        return result
    except Exception as e:
        logger.warning("nba_props_team_stats_failed", error=str(e))
        return {}


def _fetch_schedule_sync(game_date: str) -> list[dict]:
    """Fetch NBA schedule from ESPN for a given date."""
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
                })
        return matchups
    except Exception as e:
        logger.debug("nba_props_schedule_failed", error=str(e))
        return []


def _league_averages(team_stats: dict[str, dict]) -> dict:
    """Compute league averages from team stats."""
    if not team_stats:
        return {
            "def_rating": 112.0, "pace": 98.5, "opp_fg3_pct": 0.362,
            "opp_fg3m": 12.5, "opp_reb": 43.0, "opp_oreb": 10.5, "opp_pts": 112.0,
        }
    avgs: dict = {}
    for key in ("def_rating", "pace", "opp_fg3_pct", "opp_fg3m", "opp_reb", "opp_oreb", "opp_pts"):
        vals = [v[key] for v in team_stats.values() if key in v]
        avgs[key] = float(np.mean(vals)) if vals else 0.0
    return avgs


def compute_prop_prob_cached(
    player_name: str,
    stat_type: str,
    threshold: float,
    game_date: str | None = None,
) -> PropResult | None:
    """Compute P(player stat >= threshold) from cached data. No API calls.

    Same model as nba_stats.py: recency-weighted mean + opponent adjustment + streak.
    Returns PropResult or None if player not in cache.

    Minutes volatility: games where the player's minutes deviate more than 1.5σ
    from their L15 average are downweighted to 25% of normal decay weight.  This
    prevents rest-day blowups (e.g. a bench player getting 35 min because starters
    sat) from inflating projected stats.
    """
    cache = _load_cache()
    if cache is None:
        return None

    players = cache.get("players", {})
    player = players.get(player_name)
    if player is None:
        return None

    col = STAT_COL.get(stat_type)
    if col is None:
        return None

    stats = player.get("stats", {})
    n_games = player.get("n_games", 0)
    if n_games < _MIN_GAMES:
        return None

    # Build raw stat series
    if col == "__pra__":
        pts = stats.get("PTS", [])
        reb = stats.get("REB", [])
        ast = stats.get("AST", [])
        if not pts or not reb or not ast:
            return None
        n = min(len(pts), len(reb), len(ast), _LAST_N_GAMES)
        raw = np.array([pts[i] + reb[i] + ast[i] for i in range(n)])
    elif col == "__bs__":
        blk = stats.get("BLK", [])
        stl = stats.get("STL", [])
        if not blk or not stl:
            return None
        n = min(len(blk), len(stl), _LAST_N_GAMES)
        raw = np.array([blk[i] + stl[i] for i in range(n)])
    else:
        if col not in stats:
            return None
        raw = np.array(stats[col][:_LAST_N_GAMES])
        n = len(raw)

    if n < _MIN_GAMES:
        return None

    # Per-36-minute normalization
    minutes = np.array(stats.get("MIN", [36.0] * n)[:n])
    avg_min = float(np.mean(minutes))
    min_std = float(np.std(minutes)) if n >= 3 else 0.0
    minutes_cv = min_std / avg_min if avg_min > 5.0 else 0.0

    # Detect minutes volatility: flag games where minutes deviate > 1.5σ
    volatile_mask = np.zeros(n, dtype=bool)
    if avg_min >= 5.0 and min_std > 1.0:
        volatile_mask = np.abs(minutes - avg_min) > (_MIN_VOL_THRESHOLD * min_std)
    volatile_games = int(np.sum(volatile_mask))
    minutes_volatile = volatile_games >= 2 or (volatile_games >= 1 and minutes_cv > 0.25)

    if avg_min >= 5.0:
        per36 = raw / np.maximum(minutes, 1.0) * 36.0
        eff_threshold = threshold / avg_min * 36.0
    else:
        per36 = raw
        eff_threshold = float(threshold)

    # Exponential decay weights — downweight volatile-minutes games
    weights = np.array([_DECAY ** i for i in range(n)])
    for i in range(n):
        if volatile_mask[i]:
            weights[i] *= _VOL_DOWNWEIGHT
    weights /= weights.sum()

    wmean = float(np.dot(weights, per36))
    wvar = float(np.dot(weights, (per36 - wmean) ** 2))
    wstd = max(float(np.sqrt(wvar)), 0.5)

    # ------------------------------------------------------------------
    # Line performance: empirical hit rate + margin over/under
    # ------------------------------------------------------------------
    hits = raw >= threshold
    hit_rate = float(np.mean(hits))
    avg_margin = float(np.mean(raw - threshold))  # positive = player beats line on avg

    # Recency-weighted hit rate (recent games count more)
    weighted_hit_rate = float(np.dot(weights, hits.astype(float)))

    # Streak detection (last 5 games vs threshold)
    last5 = raw[:5]
    last5_hits = int(np.sum(last5 >= threshold))

    # ------------------------------------------------------------------
    # Base probability from normal distribution
    # ------------------------------------------------------------------
    # No continuity correction — the old -0.5 inflated prob by ~5pp.
    model_prob = float(1 - norm.cdf(eff_threshold, wmean, wstd))
    model_prob = max(0.01, min(0.99, model_prob))

    # ------------------------------------------------------------------
    # Calibrated blend
    #
    # Calibration replay (n=143, Apr 2026) revealed the Kalshi over market
    # is systematically overpriced: avg implied 62%, actual hit rate 37%.
    # Our old model was EVEN WORSE — predicting 75% average, producing
    # illusory +EV signals.  Root causes:
    #   1. Normal CDF overestimates upper tail (stats are right-skewed)
    #   2. Per-36 normalization inflates short-minute games
    #   3. Margin/streak adjustments stacked +6-11pp upward bias
    #   4. No calibration against the base rate of prop overs (~37%)
    #
    # Fix: anchor on weighted empirical hit rate (most grounded signal),
    # blend in model CDF at low weight, apply conservative adjustments,
    # and shrink toward the empirical Kalshi base rate (~0.40).
    # ------------------------------------------------------------------

    # 40/60 model/empirical — empirical is the stronger anchor
    blended_prob = 0.40 * model_prob + 0.60 * weighted_hit_rate

    # Margin adjustment: conservative — +0.3% per point, capped ±3%
    margin_adj = np.clip(avg_margin * 0.003, -0.03, 0.03)
    blended_prob += margin_adj

    # Streak adjustment: very small nudge
    if last5_hits >= 4:
        streak_adj = 0.01 + 0.005 * (last5_hits - 4)  # +1% for 4/5, +1.5% for 5/5
    elif last5_hits <= 1:
        streak_adj = -0.01 - 0.005 * (1 - last5_hits)  # -1% for 1/5, -1.5% for 0/5
    else:
        streak_adj = 0.0
    blended_prob += streak_adj

    # Shrink toward the empirical Kalshi over base rate (~0.40).
    # The market-wide over hit rate is ~37%, but some of that is driven
    # by extreme mispricing at 80-95c.  Use 0.40 as a conservative anchor.
    # Shrinkage = 20% — strong enough to pull 80% down to 72%, mild enough
    # that genuine 60% stays at 56%.
    _BASE_RATE = 0.40
    _SHRINKAGE = 0.20
    blended_prob = blended_prob * (1 - _SHRINKAGE) + _BASE_RATE * _SHRINKAGE

    # ------------------------------------------------------------------
    # Opponent defensive adjustment
    # ------------------------------------------------------------------
    opp_adj = 1.0
    if game_date:
        team_stats = cache.get("team_stats", {})
        league_avg = cache.get("league_avg", {})
        schedule = cache.get("schedule", [])

        player_team = player.get("team", "")
        opponent = None
        for m in schedule:
            if m.get("home_abbrev") == player_team:
                opponent = m.get("away_abbrev")
                break
            if m.get("away_abbrev") == player_team:
                opponent = m.get("home_abbrev")
                break

        if opponent and opponent in team_stats:
            opp_stats = team_stats[opponent]
            opp_adj = _opponent_adjustment(stat_type, opp_stats, league_avg)

    prob = blended_prob * opp_adj

    # Apply volatility discount: volatile players hit overs only ~18% of the
    # time (calibration Apr 2026, n=28).  Shrink 40% toward the volatile base
    # rate (0.25) — strong enough to kill false +EV signals from rest-day
    # blowup games while still allowing genuinely high probs through.
    if minutes_volatile:
        _VOL_BASE = 0.25
        _VOL_SHRINK = 0.40
        prob = prob * (1 - _VOL_SHRINK) + _VOL_BASE * _VOL_SHRINK

    prob = max(0.01, min(0.99, prob))

    return PropResult(
        prob=prob,
        n_games=n,
        minutes_volatile=minutes_volatile,
        minutes_cv=round(minutes_cv, 3),
        volatile_games=volatile_games,
        avg_minutes=round(avg_min, 1),
    )


def _opponent_adjustment(stat_type: str, opp_stats: dict, league_avg: dict) -> float:
    """Opponent defensive adjustment factor (same logic as nba_stats.py)."""
    adj = 1.0
    try:
        if stat_type == "points":
            opp_def = opp_stats.get("def_rating", league_avg.get("def_rating", 112))
            lg_def = league_avg.get("def_rating", 112)
            if lg_def > 0:
                adj = 1.0 + (opp_def - lg_def) / lg_def
        elif stat_type == "rebounds":
            opp_oreb = opp_stats.get("opp_oreb", league_avg.get("opp_oreb", 10.5))
            lg_oreb = league_avg.get("opp_oreb", 10.5)
            if lg_oreb > 0:
                adj = 1.0 - (opp_oreb - lg_oreb) / lg_oreb
        elif stat_type == "threes":
            opp_3pct = opp_stats.get("opp_fg3_pct", league_avg.get("opp_fg3_pct", 0.362))
            lg_3pct = league_avg.get("opp_fg3_pct", 0.362)
            if lg_3pct > 0:
                adj = 1.0 + (opp_3pct - lg_3pct) / lg_3pct
        elif stat_type == "assists":
            opp_pace = opp_stats.get("pace", league_avg.get("pace", 98.5))
            lg_pace = league_avg.get("pace", 98.5)
            if lg_pace > 0:
                adj = opp_pace / lg_pace
        elif stat_type == "points_rebounds_assists":
            opp_def = opp_stats.get("def_rating", league_avg.get("def_rating", 112))
            opp_pace = opp_stats.get("pace", league_avg.get("pace", 98.5))
            lg_def = league_avg.get("def_rating", 112)
            lg_pace = league_avg.get("pace", 98.5)
            pts_adj = 1.0 + (opp_def - lg_def) / lg_def if lg_def > 0 else 1.0
            pace_adj = opp_pace / lg_pace if lg_pace > 0 else 1.0
            adj = (pts_adj + pace_adj) / 2
    except Exception:
        adj = 1.0

    return max(1.0 - _MAX_OPP_ADJ, min(1.0 + _MAX_OPP_ADJ, adj))


def is_cache_fresh() -> bool:
    """Check if the props cache exists and is within TTL."""
    return _load_cache() is not None


# Abbreviation → team nickname used by InjuryReportAgent (keys are lowercase
# full team names like "los angeles lakers", and substring matching reduces
# them via the nickname). Keeping a small local map avoids pulling in the
# sector alias loader just for injury boosts.
_NBA_ABBREV_TO_NICKNAME: dict[str, str] = {
    "atl": "hawks", "bos": "celtics", "bkn": "nets", "cha": "hornets",
    "chi": "bulls", "cle": "cavaliers", "dal": "mavericks", "den": "nuggets",
    "det": "pistons", "gsw": "warriors", "hou": "rockets", "ind": "pacers",
    "lac": "clippers", "lal": "lakers", "mem": "grizzlies", "mia": "heat",
    "mil": "bucks", "min": "timberwolves", "nop": "pelicans", "nyk": "knicks",
    "okc": "thunder", "orl": "magic", "phi": "76ers", "phx": "suns",
    "por": "trail blazers", "sac": "kings", "sas": "spurs", "tor": "raptors",
    "uta": "jazz", "was": "wizards",
}


def lookup_player_team(player_name: str) -> str | None:
    """Return the canonical team nickname for a cached NBA player, or None.

    Used by the prop-injury-boost logic so it can identify the player's team
    directly instead of trying to parse a game slug out of the prop event_id
    (which has no game slug).
    """
    cache = _load_cache()
    if cache is None:
        return None
    player = cache.get("players", {}).get(player_name)
    if not player:
        return None
    abbrev = (player.get("team") or "").lower().strip()
    return _NBA_ABBREV_TO_NICKNAME.get(abbrev)
