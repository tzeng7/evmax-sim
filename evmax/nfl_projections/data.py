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
    "game_id", "play_id", "season", "week", "season_type", "posteam", "defteam", "home_team", "away_team",
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


# ── player-level sources (Phase 2) ───────────────────────────────────────────

def player_week_file(season: int, d: Optional[Path] = None) -> Path:
    return data_dir(d) / "player_week" / f"stats_player_week_{season}.parquet"


def snaps_file(season: int, d: Optional[Path] = None) -> Path:
    return data_dir(d) / "snaps" / f"snap_counts_{season}.parquet"


def players_file(d: Optional[Path] = None) -> Path:
    return data_dir(d) / "players.parquet"


def ensure_player_sources(seasons: Iterable[int], d: Optional[Path] = None,
                          refresh_seasons: Iterable[int] = (), max_age_hours: float = 12.0) -> list[int]:
    """Cache official weekly player stats + PFR snap counts per season and the
    players id map; return the seasons with both files available."""
    refresh = set(refresh_seasons)
    have = []
    for s in seasons:
        ok = True
        for f, url in ((player_week_file(s, d), f"{NFLVERSE}/stats_player/stats_player_week_{s}.parquet"),
                       (snaps_file(s, d), f"{NFLVERSE}/snap_counts/snap_counts_{s}.parquet")):
            if _stale(f, max_age_hours if s in refresh else None):
                ok = (_download(url, f) or f.exists()) and ok
        if ok:
            have.append(s)
    pf = players_file(d)
    if _stale(pf, 24.0 * 7):
        _download(f"{NFLVERSE}/players/players.parquet", pf)
    return have


def load_player_week(seasons: Iterable[int], d: Optional[Path] = None) -> pd.DataFrame:
    """Official per-player game stats (gsis ``player_id``), teams normalized."""
    frames = [pd.read_parquet(player_week_file(s, d)) for s in seasons if player_week_file(s, d).exists()]
    if not frames:
        return pd.DataFrame()
    w = pd.concat(frames, ignore_index=True)
    for c in ("team", "opponent_team"):
        w[c] = w[c].replace(TEAM_ALIASES)
    return w


def load_snaps(seasons: Iterable[int], d: Optional[Path] = None) -> pd.DataFrame:
    """PFR snap counts with the gsis ``player_id`` attached (via the nflverse players table)."""
    frames = [pd.read_parquet(snaps_file(s, d)) for s in seasons if snaps_file(s, d).exists()]
    if not frames:
        return pd.DataFrame()
    s = pd.concat(frames, ignore_index=True)
    s["team"] = s["team"].replace(TEAM_ALIASES)
    ids = pd.read_parquet(players_file(d), columns=["gsis_id", "pfr_id"]).dropna()
    ids = ids.drop_duplicates("pfr_id")
    return s.merge(ids, left_on="pfr_player_id", right_on="pfr_id", how="left").rename(
        columns={"gsis_id": "player_id"})


def injuries_file(season: int, d: Optional[Path] = None) -> Path:
    return data_dir(d) / "injuries" / f"injuries_{season}.parquet"


def ensure_injuries(season: int, d: Optional[Path] = None, max_age_hours: float = 6.0) -> bool:
    f = injuries_file(season, d)
    if _stale(f, max_age_hours):
        return _download(f"{NFLVERSE}/injuries/injuries_{season}.parquet", f) or f.exists()
    return True


def load_injuries(season: int, d: Optional[Path] = None) -> pd.DataFrame:
    """Weekly official injury reports (gsis_id, week, team, report_status)."""
    f = injuries_file(season, d)
    if not f.exists():
        return pd.DataFrame(columns=["season", "week", "team", "gsis_id", "report_status"])
    i = pd.read_parquet(f)
    i["team"] = i["team"].replace(TEAM_ALIASES)
    return i


# ── historical opening lines (ESPN) — model-pick backtests only ──────────────

ESPN_ODDS = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{eid}/competitions/{eid}/odds"


def espn_open_file(d: Optional[Path] = None) -> Path:
    return data_dir(d) / "espn_open_lines.parquet"


def _espn_num(x) -> Optional[float]:
    if isinstance(x, dict):
        x = x.get("american", x.get("value"))
    if x is None:
        return None
    s = str(x).strip().upper()
    if s in ("EVEN", "PK", "PICK"):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return None


def parse_espn_open(doc: Optional[dict]) -> tuple[Optional[float], Optional[float], Optional[str]]:
    """(opening HOME HANDICAP, opening total, provider) from an ESPN competition-odds document.

    ESPN carries openers in two shapes: 2024+ books expose ``homeTeamOdds.open``
    / ``open.total`` (ESPN BET from late 2023, DraftKings in 2026); 2014-16
    list a provider literally named "Opening". Live in-game providers are
    skipped. Pre-2014 and 2017 to mid-2023 games have neither.
    """
    if not doc:
        return None, None, None
    items = [i for i in doc.get("items", []) if "live" not in str(i.get("provider", {}).get("name", "")).lower()]
    for i in items:
        hs = _espn_num(((i.get("homeTeamOdds") or {}).get("open") or {}).get("pointSpread"))
        if hs is not None:
            return hs, _espn_num((i.get("open") or {}).get("total")), i.get("provider", {}).get("name")
    for i in items:
        if i.get("provider", {}).get("name") == "Opening" and i.get("spread") is not None:
            return float(i["spread"]), _espn_num(i.get("overUnder")), "Opening"
    return None, None, None


def load_espn_open_lines(games: pd.DataFrame, d: Optional[Path] = None, fetch: bool = True,
                         workers: int = 8) -> pd.DataFrame:
    """Opening lines per finished game: game_id, open_line_margin (expected HOME margin,
    nflverse ``spread_line`` sign), open_total, open_source.

    Cached in ``espn_open_lines.parquet``; with ``fetch`` the missing finished
    games are requested from ESPN's public odds API (one call per game).
    """
    import json
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    f = espn_open_file(d)
    have = pd.read_parquet(f) if f.exists() else pd.DataFrame(
        columns=["game_id", "open_line_margin", "open_total", "open_source"])
    todo = games[games["home_score"].notna() & games["espn"].notna() & ~games["game_id"].isin(have["game_id"])]
    if fetch and not todo.empty:
        def one(eid: str) -> Optional[dict]:
            req = urllib.request.Request(ESPN_ODDS.format(eid=eid), headers={"User-Agent": "evmax-nfl-projections"})
            try:
                with urllib.request.urlopen(req, timeout=20) as r:
                    return json.load(r)
            except Exception:  # noqa: BLE001 — a missing game stays uncached and is retried next run
                return None
        with ThreadPoolExecutor(workers) as ex:
            docs = list(ex.map(one, todo["espn"].astype(str)))
        new = []
        for gid, doc in zip(todo["game_id"], docs):
            if doc is None:
                continue
            hs, tot, src = parse_espn_open(doc)
            new.append({"game_id": gid, "open_line_margin": None if hs is None else -hs,
                        "open_total": tot, "open_source": src})
        if new:
            have = pd.concat([have, pd.DataFrame(new)], ignore_index=True)
            f.parent.mkdir(parents=True, exist_ok=True)
            have.to_parquet(f, index=False)
    return have
