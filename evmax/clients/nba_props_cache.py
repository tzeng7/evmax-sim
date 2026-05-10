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
_CALIBRATION_PATH = Path(__file__).resolve().parents[2] / "data" / "models" / "nba_props_calibration.json"

# Model params (match nba_stats.py)
_LAST_N_GAMES = 15
_MIN_GAMES = 5
_DECAY = 0.85
_MAX_OPP_ADJ = 0.15

# Stat column mapping
_MIN_VOL_THRESHOLD = 1.5  # flag games with minutes > mean ± 1.5σ
_VOL_DOWNWEIGHT = 0.25     # volatile games get 25% of their normal weight

# Long-shot empirical replacement (see compute_prop_prob_cached).
_LOW_TAIL_PROB = 0.15           # below this, swap Normal-CDF for empirical
_LOW_TAIL_PRIOR_HITS = 1.0      # Beta α — 1 pseudo-hit
_LOW_TAIL_PRIOR_MISSES = 4.0    # Beta β — 4 pseudo-misses (prior mean 0.20, n=5)

# Calibration loaded once on first prop computation. The JSON is fitted by
# scripts/calibrate_nba_props.py against the 2024-25 holdout — see Path A
# of TODO #24. When absent, we fall back to the original hand-tuned constants.
_calibration: dict | None = None
_calibration_loaded = False


def _load_calibration() -> dict | None:
    """Load fitted calibration once and cache. Returns None if missing/broken."""
    global _calibration, _calibration_loaded
    if _calibration_loaded:
        return _calibration
    _calibration_loaded = True
    if not _CALIBRATION_PATH.exists():
        return None
    try:
        _calibration = json.loads(_CALIBRATION_PATH.read_text())
    except Exception as exc:
        logger.warning("nba_props_calibration_load_failed", error=str(exc))
        _calibration = None
    return _calibration


def _calibration_for_stat(stat_type: str) -> dict | None:
    """Return the calibration block for one stat type, with fallback chain.

    Schema-v2 calibrations have a `per_stat` map plus a `global` block.
    Each per-stat block holds the same fields as the legacy top-level
    schema (base_rate, shrinkage, blend_model, isotonic_x_thresholds,
    isotonic_y_thresholds), fitted on rows for that stat only.

    Resolution order:
      1. per_stat[stat_type]  — stat-specific fit
      2. global               — pooled fallback
      3. legacy top-level     — v1 schema (single calibration for all stats)
      4. None                 — falls through to hand-tuned constants
    """
    cal = _load_calibration()
    if cal is None:
        return None
    per_stat = cal.get("per_stat") or {}
    if stat_type in per_stat:
        return per_stat[stat_type]
    if "global" in cal and cal["global"] is not None:
        return cal["global"]
    # Legacy v1 schema: top-level dict already IS the calibration
    if "isotonic_x_thresholds" in cal:
        return cal
    return None


def _apply_isotonic(prob: float, x_thresh: list[float], y_thresh: list[float]) -> float:
    """Piecewise-linear interpolation through the isotonic regression fit.

    Replicates `sklearn.IsotonicRegression.predict([prob])` without dragging
    in a sklearn import at scan time. Out-of-bounds clipping matches the
    calibrator's `out_of_bounds="clip"` setting.
    """
    if not x_thresh or not y_thresh:
        return prob
    if prob <= x_thresh[0]:
        return y_thresh[0]
    if prob >= x_thresh[-1]:
        return y_thresh[-1]
    # Binary search for the segment
    lo, hi = 0, len(x_thresh) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if x_thresh[mid] <= prob:
            lo = mid
        else:
            hi = mid
    x0, x1 = x_thresh[lo], x_thresh[hi]
    y0, y1 = y_thresh[lo], y_thresh[hi]
    if x1 == x0:
        return y0
    t = (prob - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


@dataclass
class PropResult:
    """Result from compute_prop_prob_cached with volatility metadata."""
    prob: float
    n_games: int
    minutes_volatile: bool = False     # True if recent games have abnormal minutes
    minutes_cv: float = 0.0            # coefficient of variation (std/mean) of minutes
    volatile_games: int = 0            # count of games flagged as outlier-minutes
    avg_minutes: float = 0.0           # L15 average minutes


@dataclass
class PropDiagnostics:
    """Cheap diagnostic-only L15 lookup — no probability math.

    Production scan reads this via compute_prop_diagnostics() instead of
    compute_prop_prob_cached(). The latter still exists for backtest scripts
    and the `evmax cleanup replay-props` CLI command, which intentionally
    re-runs the legacy L15 model on resolved rows.

    Why split these: anchor-based pricing (evmax/ev/prop_pricing.py with a
    Pinnacle anchor) is now the production source of probability. The L15
    cache's projected probability is redundant with what line-setters already
    know — that's the failure mode that took nba_props to shadow on 2026-05-01
    (−223u, long-shot bias). We keep n_games and minutes_volatile because
    they're cheap data-quality signals, not predictions.
    """
    n_games: int
    minutes_volatile: bool = False
    minutes_cv: float = 0.0
    avg_minutes: float = 0.0


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


def _load_cache_any_age() -> dict | None:
    """Load cache regardless of TTL — used for fallback when refresh fails."""
    try:
        if _CACHE_PATH.exists():
            return json.loads(_CACHE_PATH.read_text())
    except Exception as e:
        logger.debug("nba_props_cache_load_any_failed", error=str(e))
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

    # Cache-safety: if the fetch produced zero players (network blocked,
    # ESPN+nba_api both down) AND we have an existing non-empty cache,
    # keep the prior cache instead of wiping it. Prevents a death-spiral
    # where one bad refresh clears the cache and every subsequent prop
    # scan returns empty.
    if not players_data:
        prior = existing or _load_cache_any_age()
        prior_players = (prior or {}).get("players") or {}
        if prior_players:
            logger.warning(
                "nba_props_cache_refresh_empty_keeping_prior",
                prior_count=len(prior_players),
            )
            return len(prior_players)

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

    # Resolve player name → ID. nba_api ID is optional now — when missing
    # (or stats.nba.com is in backoff), we fall through to ESPN using the
    # name_key directly. Targets carry pid=0 as the sentinel for "ESPN-only".
    if player_names:
        from evmax.clients.nba_stats import _find_player_id
        targets: list[tuple[str, int]] = []
        for name in set(player_names):
            pid = _find_player_id(name) or 0
            targets.append((name, pid))
    else:
        all_active = [p for p in nba_players.get_active_players()]
        targets = [
            (p["full_name"].lower().replace(" ", "_"), p["id"])
            for p in all_active
        ]

    # Skip nba_api entirely when the circuit breaker is tripped — every call
    # would just burn the per-player timeout (3 retries × 10s) before failing.
    from evmax.agents.models._nba_freshness import nba_api_in_backoff, mark_nba_api_failure
    nba_api_blocked = nba_api_in_backoff()
    if nba_api_blocked:
        logger.info("nba_props_skipping_nba_api_breaker_tripped", targets=len(targets))

    logger.info("nba_props_fetching_players", count=len(targets))

    nba_failures = 0  # track failures to trip the breaker on first sign of trouble

    async def _fetch_one(name_key: str, player_id: int) -> tuple[str, dict | None]:
        nonlocal nba_failures
        loop = asyncio.get_event_loop()

        if not nba_api_blocked and player_id:
            async with _SEM:
                data = await loop.run_in_executor(
                    None, _fetch_single_player_sync, player_id,
                )
            if data is not None:
                return name_key, data
            nba_failures += 1

        # ESPN fallback. No semaphore — ESPN handles concurrent requests fine
        # and we want to recover the cache fast when stats.nba.com is down.
        data = await loop.run_in_executor(None, _fetch_player_via_espn_sync, name_key)
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

    # If every nba_api call failed but ESPN saved us, trip the breaker so
    # the next refresh skips nba_api straight away.
    if not nba_api_blocked and nba_failures and nba_failures >= len(targets) // 2:
        mark_nba_api_failure()
        logger.warning("nba_props_nba_api_breaker_tripped", failures=nba_failures, total=len(targets))

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


# ESPN fallback path. stats.nba.com routinely blocks scraping during playoffs;
# ESPN's public gamelog API is unauthenticated, fast (~350ms), and exposes the
# same per-game stat columns we need. Used when nba_api returns None.
#
# Stat indices come from the response's `labels` array, fixed across requests:
#   ['MIN','FG','FG%','3PT','3P%','FT','FT%','REB','AST','BLK','STL','PF','TO','PTS']
_ESPN_STAT_IDX = {"MIN": 0, "REB": 7, "AST": 8, "BLK": 9, "STL": 10, "TOV": 12, "PTS": 13}
_ESPN_3PT_IDX = 3  # "made-attempted" string, e.g. "4-9"
_ESPN_SEARCH_URL = "https://site.web.api.espn.com/apis/common/v3/search"
_ESPN_GAMELOG_URL = "https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/{id}/gamelog"
_ESPN_SEASON_YEAR = 2026  # 2025-26 season; bump in offseason

# In-memory cache of name → ESPN athlete ID so repeated scans don't re-search
# (lookup costs ~300ms per player; cache survives the process lifetime).
_espn_id_cache: dict[str, str] = {}


def _resolve_espn_athlete_id(player_name: str) -> str | None:
    """Resolve a name like 'lebron_james' to ESPN athlete id via search."""
    cached = _espn_id_cache.get(player_name)
    if cached is not None:
        return cached or None
    query = player_name.replace("_", " ").strip()
    if not query:
        return None
    try:
        import httpx
        r = httpx.get(
            _ESPN_SEARCH_URL,
            params={"query": query, "limit": 5, "type": "player"},
            timeout=8.0,
        )
        r.raise_for_status()
        for item in (r.json() or {}).get("items", []):
            if item.get("league") == "nba" and item.get("sport") == "basketball":
                aid = str(item.get("id") or "")
                if aid:
                    _espn_id_cache[player_name] = aid
                    return aid
        _espn_id_cache[player_name] = ""  # negative-cache so we don't retry
        return None
    except Exception as e:
        logger.debug("nba_props_espn_search_failed", player=player_name, error=str(e))
        return None


def _parse_espn_stat(stats_row: list, idx: int) -> float:
    """Coerce ESPN stat-cell to float; '-' / '' → 0.0."""
    try:
        v = stats_row[idx]
    except (IndexError, TypeError):
        return 0.0
    if v in (None, "", "-"):
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _parse_espn_made(stats_row: list, idx: int) -> float:
    """Parse a 'made-attempted' cell like '4-9' → 4.0."""
    try:
        cell = str(stats_row[idx])
    except (IndexError, TypeError):
        return 0.0
    if "-" not in cell:
        return 0.0
    try:
        return float(cell.split("-", 1)[0])
    except (TypeError, ValueError):
        return 0.0


def _fetch_player_via_espn_sync(player_name: str) -> dict | None:
    """ESPN fallback. Returns same shape as _fetch_single_player_sync, or None.

    Walks seasonTypes → categories → events in the order ESPN returns them
    (postseason first, then regular season most-recent-month first), collects
    up to _LAST_N_GAMES, parses stats by fixed label index.
    """
    aid = _resolve_espn_athlete_id(player_name)
    if not aid:
        return None
    try:
        import httpx
        r = httpx.get(
            _ESPN_GAMELOG_URL.format(id=aid),
            params={"season": _ESPN_SEASON_YEAR},
            timeout=10.0,
        )
        r.raise_for_status()
        data = r.json() or {}
    except Exception as e:
        logger.debug("nba_props_espn_gamelog_failed", player=player_name, aid=aid, error=str(e))
        return None

    events_meta = data.get("events") or {}
    collected: list[tuple[str, list]] = []  # (event_id, stats_row)
    for st in data.get("seasonTypes") or []:
        for cat in st.get("categories") or []:
            for ev in cat.get("events") or []:
                eid = str(ev.get("eventId") or "")
                row = ev.get("stats") or []
                if eid and row:
                    collected.append((eid, row))

    # ESPN returns most-recent-first within each category, but categories
    # are ordered postseason→regular, recent→old. Sort by event gameDate
    # descending using the events meta map so playoff-and-RS interleave
    # is handled correctly.
    def _date_key(eid: str) -> str:
        return (events_meta.get(eid, {}) or {}).get("gameDate", "")
    collected.sort(key=lambda p: _date_key(p[0]), reverse=True)
    collected = collected[:_LAST_N_GAMES]

    if len(collected) < _MIN_GAMES:
        return None

    stats: dict[str, list[float]] = {k: [] for k in ("PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV", "MIN")}
    for _eid, row in collected:
        stats["PTS"].append(_parse_espn_stat(row, _ESPN_STAT_IDX["PTS"]))
        stats["REB"].append(_parse_espn_stat(row, _ESPN_STAT_IDX["REB"]))
        stats["AST"].append(_parse_espn_stat(row, _ESPN_STAT_IDX["AST"]))
        stats["STL"].append(_parse_espn_stat(row, _ESPN_STAT_IDX["STL"]))
        stats["BLK"].append(_parse_espn_stat(row, _ESPN_STAT_IDX["BLK"]))
        stats["TOV"].append(_parse_espn_stat(row, _ESPN_STAT_IDX["TOV"]))
        stats["MIN"].append(_parse_espn_stat(row, _ESPN_STAT_IDX["MIN"]))
        stats["FG3M"].append(_parse_espn_made(row, _ESPN_3PT_IDX))

    if not stats["PTS"]:
        return None

    # Pull team + display name from the most recent event meta.
    most_recent_eid = collected[0][0]
    meta = events_meta.get(most_recent_eid, {}) or {}
    team_abbrev = ((meta.get("team") or {}).get("abbreviation") or "").upper()
    display_name = player_name.replace("_", " ").title()

    return {
        "player_name": display_name,
        "player_id": 0,  # not an nba_api id; downstream only uses name keys
        "team": team_abbrev,
        "n_games": len(collected),
        "stats": stats,
        "source": "espn",
    }


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
    normal_prob = float(1 - norm.cdf(eff_threshold, wmean, wstd))
    normal_prob = max(0.01, min(0.99, normal_prob))

    # Long-shot empirical replacement. Counting stats (PTS/AST/3PM) are
    # right-skewed, so Normal-CDF systematically overstates the upper tail
    # at low probabilities. 2026-04→2026-05 production audit: <10c Kalshi
    # bucket hit 1.9% (n=212) while the calibrated model said 14.1% — a
    # −163u leak driven by Normal-CDF tail bias amplified by an isotonic
    # floor of 12.4%. When raw Normal predicts <15%, replace with a
    # Laplace-smoothed empirical CDF from the L15 game log: Beta(α=1, β=4)
    # prior (mean 0.20, effective n=5). 0/15 → 5%, 1/15 → 10%, 3/15 → 20%.
    if normal_prob < _LOW_TAIL_PROB:
        n_hits_total = float(np.sum(hits))
        n_games_total = float(len(hits))
        model_prob = (n_hits_total + _LOW_TAIL_PRIOR_HITS) / (
            n_games_total + _LOW_TAIL_PRIOR_HITS + _LOW_TAIL_PRIOR_MISSES
        )
        low_tail_empirical = True
    else:
        model_prob = normal_prob
        low_tail_empirical = False

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

    # Calibration loaded from data/models/nba_props_calibration.json (fitted
    # via scripts/calibrate_nba_props.py against 2024-25 holdout). When the
    # file is present, its base_rate / shrinkage / blend_model override the
    # legacy hand-tuned values, and an isotonic post-hoc layer applies at
    # the very end. When absent, we fall back to the original constants.
    #
    # Schema v2 (2026-05-03+): per-stat calibrations preferred over a single
    # global. _calibration_for_stat handles the per-stat → global → legacy
    # fallback chain. Each stat type sees its own isotonic mapping fitted on
    # ~10–20k 2024-25 rows of just-that-stat (threes, points, assists are
    # the biggest individual-fit improvements).
    cal = _calibration_for_stat(stat_type)
    if cal is not None:
        blend_model = cal.get("blend_model", 0.40)
        base_rate = cal.get("base_rate", 0.40)
        shrinkage = cal.get("shrinkage", 0.20)
    else:
        blend_model, base_rate, shrinkage = 0.40, 0.40, 0.20

    blended_prob = blend_model * model_prob + (1 - blend_model) * weighted_hit_rate

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

    # Shrink toward base rate. With the fitted calibration, shrinkage=0 and
    # the isotonic layer below handles the actual probability adjustment.
    blended_prob = blended_prob * (1 - shrinkage) + base_rate * shrinkage

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

    # Isotonic post-hoc calibration. Fitted on 2024-25 train half, validated
    # on test half — fixes the systematic +13pp under-prediction in the
    # 50-70% bucket that the hand-tuned shrinkage was masking. When the
    # calibration file is absent, this is a no-op.
    #
    # Skipped for low-tail empirical replacements: the 2024-25 isotonic
    # floors any input ≤12% at 12.4%, which would re-introduce the long-shot
    # bias the empirical replacement just removed.
    if cal is not None and not low_tail_empirical:
        x_thresh = cal.get("isotonic_x_thresholds") or []
        y_thresh = cal.get("isotonic_y_thresholds") or []
        if x_thresh and y_thresh:
            prob = _apply_isotonic(prob, x_thresh, y_thresh)

    prob = max(0.01, min(0.99, prob))

    return PropResult(
        prob=prob,
        n_games=n,
        minutes_volatile=minutes_volatile,
        minutes_cv=round(minutes_cv, 3),
        volatile_games=volatile_games,
        avg_minutes=round(avg_min, 1),
    )


def compute_prop_diagnostics(player_name: str) -> PropDiagnostics | None:
    """Cheap L15 diagnostic lookup — sample size + minutes volatility only.

    Production scan path. Returns None if the player isn't in cache or has
    fewer than _MIN_GAMES recent games. Otherwise returns a PropDiagnostics
    with n_games / minutes_volatile / minutes_cv / avg_minutes — no
    probability math, no opponent adjustment, no isotonic calibration.

    See PropDiagnostics docstring for the design rationale.
    """
    cache = _load_cache()
    if cache is None:
        return None
    player = (cache.get("players") or {}).get(player_name)
    if player is None:
        return None

    n_games = player.get("n_games", 0)
    if n_games < _MIN_GAMES:
        return None

    stats = player.get("stats", {})
    minutes_list = stats.get("MIN", [])
    if not minutes_list:
        # No minutes data — return what we can.
        return PropDiagnostics(n_games=n_games)

    n = min(len(minutes_list), _LAST_N_GAMES)
    minutes = np.array(minutes_list[:n])
    avg_min = float(np.mean(minutes))
    min_std = float(np.std(minutes)) if n >= 3 else 0.0
    minutes_cv = min_std / avg_min if avg_min > 5.0 else 0.0

    volatile_games = 0
    if avg_min >= 5.0 and min_std > 1.0:
        volatile_mask = np.abs(minutes - avg_min) > (_MIN_VOL_THRESHOLD * min_std)
        volatile_games = int(np.sum(volatile_mask))
    minutes_volatile = volatile_games >= 2 or (volatile_games >= 1 and minutes_cv > 0.25)

    return PropDiagnostics(
        n_games=n,
        minutes_volatile=minutes_volatile,
        minutes_cv=round(minutes_cv, 3),
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
