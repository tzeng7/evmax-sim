"""NFL player-prop probability model — pure compute layer.

Mirrors the NBA model in `nba_props_cache.py` but adapted for NFL's
weekly cadence and different stat distributions. Stage 4 of the NFL prop
backtest plan provides only the pure computation; Stage 5 will wrap this
with a disk-backed cache (`data/nfl_props_cache.json`) matching the NBA
schema, so the coordinator can call one helper regardless of data source.

Model stages for yardage props (passing/rushing/receiving yards):

  1. Filter the player's prior-week game log (point-in-time).
  2. Exponential decay weights (most recent game = 1.0, decay=0.80).
  3. Weighted mean + weighted std.
  4. Normal CDF with continuity correction → model probability.
  5. Empirical hit rate (60/40 blend with model).
  6. Margin adjustment (±8% cap).
  7. Streak adjustment from last 3 games.
  8. Multiplicative opponent adjustment (±15% cap).

Model for touchdown props (anytime TD, passing TDs 1.5+/2.5+):

  - Weighted-mean expected TDs (λ) then Poisson tail.
  - Opponent adjustment scales λ.
  - Streak/margin adjustments do not apply — noisy at low counts.
"""

from __future__ import annotations

import io
import re
import time
import unicodedata
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import structlog
from scipy.stats import norm, poisson

logger = structlog.get_logger(__name__)

# Constants — CLAUDE.md testing policy requires tests; tune values live in
# tests/test_nfl_props_cache.py which calls the pure functions directly.
LAST_N_GAMES = 8
MIN_GAMES = 4
DECAY = 0.80
MAX_OPP_ADJ = 0.15
MODEL_EMPIRICAL_BLEND = 0.60  # 60% model / 40% empirical hit rate
EMPIRICAL_DISAGREE_CAP = 0.15 # if empirical disagrees with Gaussian by more than
                              # this, the empirical signal is being driven by
                              # 1-2 outlier games. Cap the blend contribution.
MARGIN_SCALE = 0.01           # per-unit margin → probability nudge
MARGIN_CAP = 0.08
MIN_STD = 10.0                # floor on yard std to avoid overconfident Gaussians

YARDAGE_STATS = {"passing_yards", "rushing_yards", "receiving_yards"}
TD_STATS = {"anytime_td", "passing_tds", "receptions"}
# NOTE: receptions is not a TD stat — it's a count stat handled like yards
# but with integer thresholds. We override via the dispatch table below.

STAT_TO_BRANCH = {
    "passing_yards": "yardage",
    "rushing_yards": "yardage",
    "receiving_yards": "yardage",
    "receptions": "count",     # uses yardage branch with MIN_STD=1.5
    "passing_tds": "poisson",  # ≥ 1.5 or ≥ 2.5
    "anytime_td": "poisson",   # ≥ 0.5
}

MIN_STD_BY_STAT = {
    # Floors doubled from initial Stage 4 values — initial floors assumed
    # a tighter per-game distribution than NFL actually produces (game
    # script, weather, opponent pressure add variance the L8 rolling std
    # fails to capture). Raising these flattens the model's tails, which
    # is the dominant source of the Stage 4 ROI failure. MODEL-8 step 1.
    "passing_yards": 70.0,
    "rushing_yards": 30.0,
    "receiving_yards": 24.0,
    "receptions": 2.0,
}


def _exp_weights(n: int) -> np.ndarray:
    w = np.array([DECAY ** i for i in range(n)], dtype=float)
    return w / w.sum()


def _yardage_prob(
    threshold: float,
    values: np.ndarray,
    stat_type: str,
    opp_adj: float,
) -> float:
    """Normal-CDF model for a continuous yardage stat.

    `values` is ordered most-recent-first and has already been clipped to
    LAST_N_GAMES. Returns a probability in [0.01, 0.99].
    """
    n = len(values)
    weights = _exp_weights(n)

    wmean = float(np.dot(weights, values))
    wvar = float(np.dot(weights, (values - wmean) ** 2))
    wstd = max(float(np.sqrt(wvar)), MIN_STD_BY_STAT.get(stat_type, MIN_STD))

    # Continuity correction — mirror NBA (threshold - 0.5)
    model_prob = float(1.0 - norm.cdf(threshold - 0.5, wmean, wstd))
    model_prob = max(0.01, min(0.99, model_prob))

    # Empirical hit-rate blend — MODEL-8 step 3 caps disagreement.
    # When the empirical hit rate is far from the Gaussian prediction,
    # it's usually because 1-2 outlier games are dominating the small
    # sample. Clip the empirical signal to stay within ±EMPIRICAL_DISAGREE_CAP
    # of the Gaussian so it can't amplify the model into the miscalibrated tails.
    hits = (values >= threshold).astype(float)
    weighted_hit_rate = float(np.dot(weights, hits))
    clipped_empirical = float(
        np.clip(
            weighted_hit_rate,
            model_prob - EMPIRICAL_DISAGREE_CAP,
            model_prob + EMPIRICAL_DISAGREE_CAP,
        )
    )
    blended = MODEL_EMPIRICAL_BLEND * model_prob + (1 - MODEL_EMPIRICAL_BLEND) * clipped_empirical

    # Margin adjustment: average margin over the line tilts the probability
    avg_margin = float(np.mean(values - threshold))
    scale = MARGIN_SCALE / max(MIN_STD_BY_STAT.get(stat_type, MIN_STD) / 10.0, 0.1)
    margin_adj = float(np.clip(avg_margin * scale, -MARGIN_CAP, MARGIN_CAP))
    blended += margin_adj

    # MODEL-8 step 2: streak adjustment removed. 3-game streaks at NFL's
    # weekly cadence are mostly noise and they compound with the empirical
    # blend to push the model into the miscalibrated tail buckets.

    prob = blended * opp_adj
    return max(0.01, min(0.99, float(prob)))


def _poisson_prob(
    threshold: float,
    values: np.ndarray,
    opp_adj: float,
) -> float:
    """Poisson tail for count-based TD props.

    Uses weighted-mean expected value as λ. The standard NFL thresholds
    are 0.5 (anytime), 1.5 (passing 2+), 2.5 (passing 3+) so we evaluate
    P(X ≥ ceil(threshold)) = 1 - poisson.cdf(k-1, λ) where k = ceil.
    """
    n = len(values)
    weights = _exp_weights(n)
    lam = float(np.dot(weights, values)) * opp_adj
    lam = max(0.01, lam)

    # k = smallest integer strictly greater than threshold
    # For threshold=0.5 → k=1 → P(X≥1); for 1.5 → k=2; for 2.5 → k=3
    k = int(np.ceil(threshold))
    if k <= 0:
        k = 1

    prob = float(1.0 - poisson.cdf(k - 1, lam))
    return max(0.01, min(0.99, prob))


def compute_nfl_prop_prob(
    stat_type: str,
    threshold: float,
    values: np.ndarray | list[float],
    opp_adj: float = 1.0,
) -> Optional[tuple[float, int]]:
    """Compute P(stat ≥ threshold) for an NFL player prop.

    Args:
      stat_type: canonical stat key (see STAT_TO_BRANCH).
      threshold: the prop line (e.g. 249.5 for "250+ passing yards").
      values: per-game observed stat values, ordered most-recent-first.
              The function will use up to LAST_N_GAMES of these.
      opp_adj: multiplicative opponent adjustment factor. 1.0 = neutral.

    Returns:
      (probability, n_games_used) or None if sample too thin.
    """
    branch = STAT_TO_BRANCH.get(stat_type)
    if branch is None:
        return None

    arr = np.asarray(values, dtype=float)[:LAST_N_GAMES]
    n = len(arr)
    if n < MIN_GAMES:
        return None

    if branch in ("yardage", "count"):
        prob = _yardage_prob(threshold, arr, stat_type, opp_adj)
    elif branch == "poisson":
        prob = _poisson_prob(threshold, arr, opp_adj)
    else:
        return None

    return prob, n


def compute_opponent_adjustment(
    stat_type: str,
    opponent_allowed_pg: float,
    league_avg_allowed_pg: float,
) -> float:
    """Multiplicative adjustment factor based on opponent defense.

    A team allowing more than league average → higher player prob
    (factor > 1.0). A team allowing less → lower (factor < 1.0).
    Clipped to [1 - MAX_OPP_ADJ, 1 + MAX_OPP_ADJ].
    """
    if league_avg_allowed_pg <= 0:
        return 1.0
    ratio = opponent_allowed_pg / league_avg_allowed_pg
    return float(np.clip(ratio, 1.0 - MAX_OPP_ADJ, 1.0 + MAX_OPP_ADJ))


# ---------------------------------------------------------------------------
# Disk-backed cache layer (MODEL-9 / ARCH-4)
# ---------------------------------------------------------------------------
#
# Unlike the NBA cache, which fetches fresh data from stats.nba.com daily, the
# NFL cache reuses the nflverse parquets that PR #6 downloads into
# data/backtest/nfl_props/. The parquets are re-run by the user periodically
# during the season via scripts/fetch_nfl_features.py.
#
# Process-level memoization: the first call loads and prepares all tables
# (~40k rows), and every subsequent call hits the in-memory objects directly.
# Live scans call compute_nfl_prop_prob_cached(player_name, stat_type, line,
# game_date) — the same signature as the NBA cached function — so the
# coordinator can branch symmetrically on sector.


_NFL_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "backtest" / "nfl_props"
_WEEKLY_PATH = _NFL_DATA_DIR / "weekly_stats.parquet"
_ROSTERS_PATH = _NFL_DATA_DIR / "rosters.parquet"
_SCHEDULES_PATH = _NFL_DATA_DIR / "schedules.parquet"

_STALE_DAYS = 7  # re-download if parquets are older than this

_tables: Optional[dict] = None

# nflverse GitHub release URLs — same as scripts/fetch_nfl_features.py
_NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"


def _normalize_name(s: str) -> str:
    """Lowercase ASCII with suffixes stripped. Mirrors the backtest join logic."""
    s = unicodedata.normalize("NFKD", s or "")
    s = s.encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s)
    return s


def _load_tables() -> Optional[dict]:
    """Lazy-load and prepare all NFL prop feature tables.

    Reads four parquets — weekly stats, rosters, schedules — and builds the
    same derived structures (`_prepare_weekly`, `_build_defense_table`,
    `_league_averages`, per-player groupby) that the backtest loader uses.
    Returns None if any parquet is missing.
    """
    global _tables
    if _tables is not None:
        return _tables
    if not (_WEEKLY_PATH.exists() and _ROSTERS_PATH.exists() and _SCHEDULES_PATH.exists()):
        logger.warning(
            "nfl_props_cache_missing_parquets",
            weekly=_WEEKLY_PATH.exists(),
            rosters=_ROSTERS_PATH.exists(),
            schedules=_SCHEDULES_PATH.exists(),
            hint="run scripts/fetch_nfl_features.py",
        )
        return None

    import pandas as pd

    from evmax.backtest.sources.nfl_props import (
        _build_defense_table,
        _league_averages,
        _prepare_weekly,
    )

    weekly = _prepare_weekly(pd.read_parquet(_WEEKLY_PATH))
    rosters = pd.read_parquet(_ROSTERS_PATH)
    schedules = pd.read_parquet(_SCHEDULES_PATH)

    by_player: dict[str, "pd.DataFrame"] = {
        pid: grp for pid, grp in weekly.groupby("player_id")
    }
    defense = _build_defense_table(weekly)
    league_avg = _league_averages(defense)

    # Normalize name → gsis_id map, preferring the most recent season to
    # handle team changes and rookies. Fall back to older seasons so
    # veterans still resolve if the latest-season roster row is missing.
    name_to_id: dict[str, str] = {}
    if "season" in rosters.columns:
        seasons = sorted(rosters["season"].dropna().unique().tolist(), reverse=True)
    else:
        seasons = [None]
    for season in seasons:
        sub = rosters if season is None else rosters[rosters["season"] == season]
        for _, r in sub.iterrows():
            name = r.get("player_display_name") or r.get("full_name") or ""
            pid = r.get("gsis_id") or r.get("player_id") or ""
            if not name or not pid:
                continue
            key = _normalize_name(str(name))
            if key and key not in name_to_id:
                name_to_id[key] = str(pid)

    _tables = {
        "weekly": weekly,
        "by_player": by_player,
        "defense": defense,
        "league_avg": league_avg,
        "schedules": schedules,
        "name_to_id": name_to_id,
    }
    logger.info(
        "nfl_props_cache_loaded",
        players=len(name_to_id),
        weekly_rows=len(weekly),
        defense_keys=len(defense),
    )
    return _tables


def _parquets_are_stale() -> bool:
    """True if any of the three parquets is missing or older than _STALE_DAYS."""
    for p in (_WEEKLY_PATH, _ROSTERS_PATH, _SCHEDULES_PATH):
        if not p.exists():
            return True
        age_days = (time.time() - p.stat().st_mtime) / 86400
        if age_days > _STALE_DAYS:
            return True
    return False


def _current_seasons() -> list[int]:
    """Return the NFL seasons to fetch.

    During the season (Sep–Feb) we want the current season and the prior
    one for history depth. During the offseason we just want the most
    recently completed season.
    """
    from datetime import date
    today = date.today()
    year = today.year
    # NFL season year = the year it started. A season that starts in
    # Sep 2026 is "2026" even in Jan 2027.
    if today.month >= 9:
        return [year - 1, year]
    elif today.month <= 2:
        return [year - 2, year - 1]
    else:
        return [year - 2, year - 1]


def _download_parquets() -> bool:
    """Download fresh nflverse parquets to _NFL_DATA_DIR.

    Returns True on success, False on any download failure.
    """
    import pandas as pd

    seasons = _current_seasons()
    _NFL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("nfl_props_downloading_parquets", seasons=seasons)

    try:
        # Weekly stats — one file per season
        frames = []
        for y in seasons:
            url = f"{_NFLVERSE}/stats_player/stats_player_week_{y}.parquet"
            logger.debug("nfl_props_fetch", url=url)
            req = urllib.request.Request(url, headers={"User-Agent": "evmax-scan"})
            with urllib.request.urlopen(req, timeout=60) as r:
                blob = r.read()
            df = pd.read_parquet(io.BytesIO(blob))
            if "season" not in df.columns:
                df["season"] = y
            frames.append(df)
        weekly = pd.concat(frames, ignore_index=True)
        weekly.to_parquet(_WEEKLY_PATH, index=False)

        # Schedules — single file covers all seasons
        url = f"{_NFLVERSE}/schedules/games.parquet"
        req = urllib.request.Request(url, headers={"User-Agent": "evmax-scan"})
        with urllib.request.urlopen(req, timeout=60) as r:
            blob = r.read()
        schedules = pd.read_parquet(io.BytesIO(blob))
        schedules = schedules[schedules.season.isin(seasons)].reset_index(drop=True)
        schedules.to_parquet(_SCHEDULES_PATH, index=False)

        # Rosters — one file per season
        frames = []
        for y in seasons:
            url = f"{_NFLVERSE}/weekly_rosters/roster_weekly_{y}.parquet"
            req = urllib.request.Request(url, headers={"User-Agent": "evmax-scan"})
            with urllib.request.urlopen(req, timeout=60) as r:
                blob = r.read()
            frames.append(pd.read_parquet(io.BytesIO(blob)))
        rosters = pd.concat(frames, ignore_index=True)
        rosters.to_parquet(_ROSTERS_PATH, index=False)

        logger.info(
            "nfl_props_parquets_downloaded",
            weekly_rows=len(weekly),
            schedule_rows=len(schedules),
            roster_rows=len(rosters),
        )
        return True
    except Exception as e:
        logger.warning("nfl_props_download_failed", error=str(e))
        return False


def is_nfl_cache_fresh() -> bool:
    """True if the parquets exist, are not stale, and are loadable."""
    if _parquets_are_stale():
        return False
    return _load_tables() is not None


async def refresh_nfl_props_cache(
    force: bool = False,
    player_names: Optional[list[str]] = None,
) -> int:
    """Ensure parquets are fresh, then load into process memory.

    Auto-downloads from nflverse if parquets are missing or older than
    _STALE_DAYS. The download runs synchronously (~5 seconds, ~3MB total)
    via urllib — acceptable since it only triggers once per week at most.
    Returns the number of players whose game logs are available.
    """
    global _tables

    if force or _parquets_are_stale():
        _tables = None
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _download_parquets)

    t = _load_tables()
    return len(t["name_to_id"]) if t else 0


def _resolve_opponent_from_schedule(
    schedules, player_team: str, game_date_iso: str
) -> Optional[str]:
    """Look up the opponent for player_team on game_date via schedules.parquet.

    Matches against `gameday` (YYYY-MM-DD). Returns the opposing team
    abbreviation or None if no match.
    """
    try:
        import pandas as pd

        if "gameday" not in schedules.columns:
            return None
        sched = schedules[schedules["gameday"] == game_date_iso]
        for _, row in sched.iterrows():
            home = str(row.get("home_team") or "")
            away = str(row.get("away_team") or "")
            if home == player_team:
                return away
            if away == player_team:
                return home
    except Exception:
        pass
    return None


def compute_nfl_prop_prob_cached(
    player_name: str,
    stat_type: str,
    threshold: float,
    game_date: Optional[str],
):
    """Live-scan entry point — mirrors the NBA cached signature.

    Returns an NBA-compatible `PropResult` (reused so the coordinator's
    SharpOdds construction works unchanged), or None if the sample is too
    thin or the player can't be resolved.

    `game_date` is the ISO YYYY-MM-DD of the *target* game. The model looks
    at every weekly row strictly before the target week in the same season.
    If `game_date` is None we fall back to "next unplayed week for this
    player," which is the right answer during an active season.
    """
    from evmax.clients.nba_props_cache import PropResult

    t = _load_tables()
    if t is None:
        return None

    pid = t["name_to_id"].get(_normalize_name(player_name))
    if pid is None:
        return None

    player_df = t["by_player"].get(pid)
    if player_df is None or len(player_df) == 0:
        return None

    # Pick target (season, week). For a known game_date we use the season
    # and week from schedules.parquet if possible; otherwise we fall back
    # to "most recent season in the weekly data + next unplayed week".
    target_season: Optional[int] = None
    target_week: Optional[int] = None
    player_team: Optional[str] = None
    opponent: Optional[str] = None

    if game_date:
        try:
            sched = t["schedules"]
            if "gameday" in sched.columns and "week" in sched.columns:
                row = sched[sched["gameday"] == game_date].head(1)
                if len(row) > 0:
                    target_season = int(row.iloc[0]["season"])
                    target_week = int(row.iloc[0]["week"])
        except Exception:
            pass

    if target_season is None or target_week is None:
        seasons = player_df["season"].dropna().unique().tolist()
        if not seasons:
            return None
        target_season = int(max(seasons))
        same_season = player_df[player_df["season"] == target_season]
        if len(same_season) == 0:
            return None
        target_week = int(same_season["week"].max()) + 1

    # Resolve team + opponent for the opponent adjustment.
    # Prefer the player's most recent team in the target season.
    try:
        last_row = (
            player_df[player_df["season"] == target_season]
            .sort_values("week")
            .tail(1)
        )
        if len(last_row) > 0:
            player_team = str(last_row.iloc[0].get("team") or "")
    except Exception:
        player_team = None

    if game_date and player_team:
        opponent = _resolve_opponent_from_schedule(t["schedules"], player_team, game_date)

    opp_adj = 1.0
    if opponent:
        from evmax.backtest.sources.nfl_props import OPP_STAT_PER_PROP

        opp_key = (target_season, target_week, opponent)
        league_key = (target_season, target_week)
        opp_stat_name = OPP_STAT_PER_PROP.get(stat_type, "pass_yards_allowed")
        opp_allowed = t["defense"].get(opp_key, {}).get(opp_stat_name, 0.0)
        lg_allowed = t["league_avg"].get(league_key, {}).get(opp_stat_name, 0.0)
        opp_adj = compute_opponent_adjustment(stat_type, opp_allowed, lg_allowed)

    # Player history (strictly before target_week in target_season).
    from evmax.backtest.sources.nfl_props import STAT_TO_COLUMN, _player_history

    col = STAT_TO_COLUMN.get(stat_type)
    if col is None:
        return None
    history_col = col if col != "__anytime_td__" else "anytime_td"
    hist = _player_history(t["by_player"], pid, target_season, target_week, history_col)
    if len(hist) < MIN_GAMES:
        return None

    result = compute_nfl_prop_prob(
        stat_type=stat_type,
        threshold=threshold,
        values=hist,
        opp_adj=opp_adj,
    )
    if result is None:
        return None
    prob, n_used = result

    return PropResult(
        prob=prob,
        n_games=n_used,
        minutes_volatile=False,
        minutes_cv=0.0,
        volatile_games=0,
        avg_minutes=0.0,
    )


def _reset_cache_for_tests() -> None:
    """Drop the in-memory cache so a test can reload from disk."""
    global _tables
    _tables = None
