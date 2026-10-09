"""nflverse data access for the NFL projection engine (download + local parquet cache).

Files live under ``data_dir()``: ``pbp/play_by_play_<season>.parquet`` and
``games.parquet`` (schedules with scores, closing lines, rest, roof, weather).
The directory defaults to ``<repo>/data/backtest/nfl_projections`` (gitignored)
and can be pointed elsewhere with ``EVMAX_NFL_PROJ_DATA`` — useful from a git
worktree, whose own ``data/`` starts empty.

Historical franchise moves are folded onto the current abbreviation
(``OAK→LV``, ``SD→LAC``, ``STL→LA``) so a team's rating history is continuous.
"""
from __future__ import annotations

import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"
REPO_ROOT = Path(__file__).resolve().parents[2]

TEAM_ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

# The play-by-play columns the engine reads (the full file has ~370).
PBP_COLUMNS = [
    "game_id", "season", "week", "season_type", "posteam", "defteam", "home_team", "away_team",
    "play_type", "pass", "rush", "qb_dropback", "epa", "success", "wp", "pass_oe", "down",
    "yards_gained", "fixed_drive", "fixed_drive_result", "drive_time_of_possession",
    "drive_inside20", "interception", "fumble_lost", "passer_player_id", "passer_player_name",
    "qb_epa",
]


def data_dir(path: Optional[str | Path] = None) -> Path:
    """Resolve the cache directory: explicit path > $EVMAX_NFL_PROJ_DATA > repo default."""
    if path is not None:
        return Path(path)
    env = os.environ.get("EVMAX_NFL_PROJ_DATA")
    if env:
        return Path(env)
    return REPO_ROOT / "data" / "backtest" / "nfl_projections"


def pbp_file(season: int, d: Optional[Path] = None) -> Path:
    return data_dir(d) / "pbp" / f"play_by_play_{season}.parquet"


def games_file(d: Optional[Path] = None) -> Path:
    return data_dir(d) / "games.parquet"


def _download(url: str, dest: Path, timeout: int = 600) -> bool:
    """Download ``url`` to ``dest`` atomically. False on 404 (season not published)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "evmax-nfl-projections"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    except urllib.error.HTTPError as e:
        tmp.unlink(missing_ok=True)
        if e.code == 404:
            logger.warning("nfl_proj_not_published", url=url)
            return False
        raise
    tmp.replace(dest)
    return True


def _stale(path: Path, max_age_hours: Optional[float]) -> bool:
    if not path.exists():
        return True
    if max_age_hours is None:
        return False
    return (time.time() - path.stat().st_mtime) > max_age_hours * 3600


def ensure_pbp(
    seasons: Iterable[int],
    d: Optional[Path] = None,
    refresh_seasons: Iterable[int] = (),
    max_age_hours: float = 12.0,
) -> list[int]:
    """Make sure each season's play-by-play is cached; return the seasons available.

    Completed seasons are downloaded once. Seasons in ``refresh_seasons`` (the
    current one) are re-downloaded when older than ``max_age_hours``.
    """
    refresh = set(refresh_seasons)
    have = []
    for s in seasons:
        f = pbp_file(s, d)
        if _stale(f, max_age_hours if s in refresh else None):
            if not _download(f"{NFLVERSE}/pbp/play_by_play_{s}.parquet", f) and not f.exists():
                continue
        have.append(s)
    return have


def load_pbp(seasons: Iterable[int], d: Optional[Path] = None,
             columns: list[str] = PBP_COLUMNS) -> pd.DataFrame:
    """Concatenate cached play-by-play for ``seasons`` (only ``columns``), teams normalized."""
    frames = []
    for s in seasons:
        f = pbp_file(s, d)
        if f.exists():
            frames.append(pd.read_parquet(f, columns=columns))
    if not frames:
        return pd.DataFrame(columns=columns)
    p = pd.concat(frames, ignore_index=True)
    for c in ("posteam", "defteam", "home_team", "away_team"):
        p[c] = p[c].replace(TEAM_ALIASES)
    return p


def ensure_games(d: Optional[Path] = None, max_age_hours: float = 12.0) -> bool:
    f = games_file(d)
    if _stale(f, max_age_hours):
        return _download(f"{NFLVERSE}/schedules/games.parquet", f) or f.exists()
    return True


def load_games(d: Optional[Path] = None) -> pd.DataFrame:
    """Schedules (one row per game) with scores, closing lines, rest, roof and weather."""
    g = pd.read_parquet(games_file(d))
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(TEAM_ALIASES)
    g["gameday"] = pd.to_datetime(g["gameday"])
    return g
