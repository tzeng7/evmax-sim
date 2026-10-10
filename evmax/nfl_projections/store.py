"""Persist, grade and track NFL projections in ``data/projections.db``.

Two tables live beside the NBA ``projections`` table (same gitignored file):

* ``nfl_game_projections`` — one row per game (``game_id``, nflverse id).
* ``nfl_player_projections`` — one row per (game, player).

Lifecycle:

1. ``log_games`` / ``log_players`` write the week's projections. A row is
   rewritten on every run until its game kicks off (the Sunday-morning refresh
   picks up late inactives), then it is frozen: the graded projection is the
   last pre-kickoff one. A player dropped from a later pre-kickoff run (ruled
   out) is deleted from that game.
2. ``resolve`` fills actual results from nflverse once a game is final:
   scores and the closing consensus line for games, official stat lines for
   players (``played = 0`` when a player has no row for a finished game).
3. ``accuracy`` summarizes graded rows: game margin / total MAE next to the
   closing line's, and player MAE plus 10th/90th-percentile coverage.

Line convention (explicit, tested): ``market_home_margin`` and
``close_home_margin`` are nflverse ``spread_line`` values — the expected HOME
margin, positive when the home team is favored. A home team laying 3 has +3.0.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = REPO_ROOT / "data" / "projections.db"
ET = ZoneInfo("America/New_York")

PLAYER_STATS = ("receptions", "receiving_yards", "rushing_yards", "passing_yards")

# Tracked populations: the players the CLI / dashboard actually show.
TRACK_MIN_TARGETS = 3.0      # receiving stats
TRACK_MIN_CARRIES = 5.0      # rushing yards

_SCHEMA = """
CREATE TABLE IF NOT EXISTS nfl_game_projections (
    game_id            TEXT PRIMARY KEY,
    season             INTEGER NOT NULL,
    week               INTEGER NOT NULL,
    gameday            TEXT NOT NULL,
    gametime           TEXT,
    kickoff_utc        TEXT,
    home_team          TEXT NOT NULL,
    away_team          TEXT NOT NULL,
    neutral            INTEGER NOT NULL DEFAULT 0,
    roof               TEXT,
    proj_home          REAL NOT NULL,
    proj_away          REAL NOT NULL,
    proj_margin        REAL NOT NULL,   -- home - away
    proj_total         REAL NOT NULL,
    p_home_win         REAL NOT NULL,
    home_qb_id         TEXT,
    away_qb_id         TEXT,
    home_qb_name       TEXT,
    away_qb_name       TEXT,
    market_home_margin REAL,            -- nflverse spread_line at log time (+ = home favored)
    market_total       REAL,
    model_version      TEXT,
    logged_at          TEXT NOT NULL,
    updated_at         TEXT NOT NULL,
    actual_home        REAL,
    actual_away        REAL,
    close_home_margin  REAL,            -- nflverse spread_line after the game (closing consensus)
    close_total        REAL,
    resolved_at        TEXT
);
CREATE INDEX IF NOT EXISTS idx_nfl_gp_week ON nfl_game_projections(season, week);

CREATE TABLE IF NOT EXISTS nfl_player_projections (
    game_id                 TEXT NOT NULL,
    player_id               TEXT NOT NULL,
    season                  INTEGER NOT NULL,
    week                    INTEGER NOT NULL,
    gameday                 TEXT NOT NULL,
    kickoff_utc             TEXT,
    team                    TEXT NOT NULL,
    opp                     TEXT NOT NULL,
    player_name             TEXT,
    position                TEXT,
    is_starting_qb          INTEGER NOT NULL DEFAULT 0,
    proj_targets            REAL,
    proj_carries            REAL,
    proj_receptions         REAL, mean_receptions      REAL, p10_receptions      REAL, p90_receptions      REAL,
    proj_receiving_yards    REAL, mean_receiving_yards REAL, p10_receiving_yards REAL, p90_receiving_yards REAL,
    proj_rushing_yards      REAL, mean_rushing_yards   REAL, p10_rushing_yards   REAL, p90_rushing_yards   REAL,
    proj_passing_yards      REAL, p10_passing_yards    REAL, p90_passing_yards   REAL,
    model_version           TEXT,
    logged_at               TEXT NOT NULL,
    updated_at              TEXT NOT NULL,
    played                  INTEGER,
    actual_receptions       REAL,
    actual_receiving_yards  REAL,
    actual_rushing_yards    REAL,
    actual_passing_yards    REAL,
    resolved_at             TEXT,
    PRIMARY KEY (game_id, player_id)
);
CREATE INDEX IF NOT EXISTS idx_nfl_pp_week ON nfl_player_projections(season, week);
"""

# Columns added after the first release of the tables (additive migration in ``connect``).
_PLAYER_MIGRATIONS: list[tuple[str, str]] = [
    ("proj_tds", "REAL"), ("p_anytime_td", "REAL"), ("p_two_plus_td", "REAL"), ("proj_passing_tds", "REAL"),
    ("actual_tds", "REAL"), ("actual_passing_tds", "REAL"),
]

_GAME_COLS = ["game_id", "season", "week", "gameday", "gametime", "kickoff_utc", "home_team", "away_team",
              "neutral", "roof", "proj_home", "proj_away", "proj_margin", "proj_total", "p_home_win",
              "home_qb_id", "away_qb_id", "home_qb_name", "away_qb_name", "market_home_margin", "market_total"]
_PLAYER_COLS = (["game_id", "player_id", "season", "week", "gameday", "kickoff_utc", "team", "opp",
                 "player_name", "position", "is_starting_qb", "proj_targets", "proj_carries"]
                + [f"{p}_{s}" for s in ("receptions", "receiving_yards", "rushing_yards")
                   for p in ("proj", "mean", "p10", "p90")]
                + ["proj_passing_yards", "p10_passing_yards", "p90_passing_yards"]
                + ["proj_tds", "p_anytime_td", "p_two_plus_td", "proj_passing_tds"])


def db_path() -> Path:
    """projections.db location: $EVMAX_PROJ_DB, else <repo>/data/projections.db."""
    env = os.environ.get("EVMAX_PROJ_DB")
    return Path(env) if env else DEFAULT_DB_PATH


def connect(path: Optional[Path] = None) -> sqlite3.Connection:
    """Open projections.db (created if missing) with both NFL tables."""
    p = Path(path) if path is not None else db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    have = {r[1] for r in conn.execute("PRAGMA table_info(nfl_player_projections)")}
    for col, typ in _PLAYER_MIGRATIONS:
        if col not in have:
            try:
                conn.execute(f"ALTER TABLE nfl_player_projections ADD COLUMN {col} {typ}")
            except sqlite3.OperationalError as e:  # another process migrated first
                if "duplicate column" not in str(e):
                    raise
    conn.commit()
    return conn


def kickoff_utc(gameday, gametime: Optional[str]) -> Optional[str]:
    """ISO UTC kickoff from the nflverse ET date + 'HH:MM' (None when unknown)."""
    if gametime is None or (isinstance(gametime, float) and np.isnan(gametime)) or not str(gametime).strip():
        return None
    day = pd.Timestamp(gameday).date()
    hh, mm = (int(x) for x in str(gametime).split(":")[:2])
    et = datetime(day.year, day.month, day.day, hh, mm, tzinfo=ET)
    return et.astimezone(ZoneInfo("UTC")).isoformat(timespec="minutes")


def _now_utc(now: Optional[datetime]) -> str:
    n = now or datetime.now(ZoneInfo("UTC"))
    return n.astimezone(ZoneInfo("UTC")).isoformat(timespec="minutes")


def _clean(v):
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return int(v)
    if isinstance(v, pd.Timestamp):
        return v.date().isoformat()
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v


def _open_ids(conn: sqlite3.Connection, table: str, ids: list[str], now_iso: str) -> set[str]:
    """Game ids whose rows may still be written: not resolved and not yet kicked off."""
    if not ids:
        return set()
    q = ",".join("?" * len(ids))
    frozen = {r[0] for r in conn.execute(
        f"SELECT DISTINCT game_id FROM {table} WHERE game_id IN ({q}) "
        f"AND (resolved_at IS NOT NULL OR (kickoff_utc IS NOT NULL AND kickoff_utc <= ?))", (*ids, now_iso))}
    return set(ids) - frozen


def log_games(conn: sqlite3.Connection, proj: pd.DataFrame, model_version: Optional[str] = None,
              now: Optional[datetime] = None) -> int:
    """Upsert ``live.project_week`` output. Returns rows written (frozen games are skipped)."""
    if proj.empty:
        return 0
    now_iso = _now_utc(now)
    df = proj.copy()
    df["kickoff_utc"] = [kickoff_utc(d, t) for d, t in zip(df["gameday"], df.get("gametime", [None] * len(df)))]
    df["market_home_margin"] = df.get("market_spread_line")
    df["market_total"] = df.get("market_total_line")
    for c in _GAME_COLS:
        if c not in df:
            df[c] = None
    df = df[[k is None or k > now_iso for k in df["kickoff_utc"]]]
    open_ids = _open_ids(conn, "nfl_game_projections", df["game_id"].tolist(), now_iso)
    df = df[df["game_id"].isin(open_ids)]
    cols = _GAME_COLS + ["model_version", "logged_at", "updated_at"]
    upd = ", ".join(f"{c}=excluded.{c}" for c in cols if c not in ("game_id", "logged_at"))
    sql = (f"INSERT INTO nfl_game_projections ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))}) "
           f"ON CONFLICT(game_id) DO UPDATE SET {upd}")
    rows = [tuple(_clean(v) for v in r) + (model_version, now_iso, now_iso)
            for r in df[_GAME_COLS].itertuples(index=False)]
    conn.executemany(sql, rows)
    conn.commit()
    return len(rows)


def log_players(conn: sqlite3.Connection, proj: pd.DataFrame, model_version: Optional[str] = None,
                now: Optional[datetime] = None) -> int:
    """Upsert ``live.project_week_players`` output for games not yet kicked off.

    For every such game, player rows absent from this run (ruled out since the
    last run) are deleted, so the stored set is the latest pre-kickoff roster.
    """
    if proj.empty:
        return 0
    now_iso = _now_utc(now)
    df = proj.rename(columns={"player_display_name": "player_name"}).copy()
    gametime = df["gametime"] if "gametime" in df else pd.Series([None] * len(df), index=df.index)
    df["kickoff_utc"] = [kickoff_utc(d, t) for d, t in zip(df["gameday"], gametime)]
    for c in _PLAYER_COLS:
        if c not in df:
            df[c] = None
    df = df[[k is None or k > now_iso for k in df["kickoff_utc"]]]
    open_ids = _open_ids(conn, "nfl_player_projections", sorted(set(df["game_id"])), now_iso)
    df = df[df["game_id"].isin(open_ids)]
    for gid, grp in df.groupby("game_id"):
        keep = grp["player_id"].tolist()
        conn.execute(f"DELETE FROM nfl_player_projections WHERE game_id = ? AND resolved_at IS NULL "
                     f"AND player_id NOT IN ({','.join('?' * len(keep))})", (gid, *keep))
    cols = _PLAYER_COLS + ["model_version", "logged_at", "updated_at"]
    upd = ", ".join(f"{c}=excluded.{c}" for c in cols if c not in ("game_id", "player_id", "logged_at"))
    sql = (f"INSERT INTO nfl_player_projections ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))}) "
           f"ON CONFLICT(game_id, player_id) DO UPDATE SET {upd}")
    rows = [tuple(_clean(v) for v in r) + (model_version, now_iso, now_iso)
            for r in df[_PLAYER_COLS].itertuples(index=False)]
    conn.executemany(sql, rows)
    conn.commit()
    return len(rows)


def resolve(conn: sqlite3.Connection, games: pd.DataFrame, player_games: pd.DataFrame,
            now: Optional[datetime] = None, ready_games: Optional[set[str]] = None) -> dict[str, int]:
    """Grade pending rows from nflverse results.

    ``games``: ``data.load_games()`` (scores + closing lines). ``player_games``:
    ``player_games.build_player_games(...)`` (official stats incl. zero rows).
    A player row is graded only after its game has player data (nflverse
    publishes weekly stats a few hours to a day after the game); ``ready_games``
    restricts grading to games whose stat AND snap files are both published.
    """
    now_iso = _now_utc(now)
    final = games[games["home_score"].notna()].set_index("game_id")
    n_games = 0
    for (gid,) in conn.execute("SELECT game_id FROM nfl_game_projections WHERE resolved_at IS NULL").fetchall():
        if gid not in final.index:
            continue
        g = final.loc[gid]
        conn.execute("UPDATE nfl_game_projections SET actual_home=?, actual_away=?, close_home_margin=?, "
                     "close_total=?, resolved_at=? WHERE game_id=?",
                     (_clean(g["home_score"]), _clean(g["away_score"]), _clean(g.get("spread_line")),
                      _clean(g.get("total_line")), now_iso, gid))
        n_games += 1

    pending = pd.read_sql_query("SELECT game_id, player_id FROM nfl_player_projections WHERE resolved_at IS NULL",
                                conn)
    n_players = 0
    if not pending.empty and not player_games.empty:
        pg = player_games[player_games["game_id"].isin(set(pending["game_id"]))]
        with_data = set(pg["game_id"]) if ready_games is None else set(pg["game_id"]) & ready_games
        stats = pg.drop_duplicates(["game_id", "player_id"]).set_index(["game_id", "player_id"])
        for r in pending.itertuples(index=False):
            if r.game_id not in with_data or r.game_id not in final.index:
                continue
            key = (r.game_id, r.player_id)
            if key in stats.index:
                s = stats.loc[key]
                tds = _clean(s.get("rushing_tds", 0) + s.get("receiving_tds", 0)) if "rushing_tds" in s else None
                vals = (1, *(_clean(s[st]) for st in PLAYER_STATS), tds, _clean(s.get("passing_tds")))
            else:
                vals = (0, None, None, None, None, None, None)
            conn.execute("UPDATE nfl_player_projections SET played=?, actual_receptions=?, "
                         "actual_receiving_yards=?, actual_rushing_yards=?, actual_passing_yards=?, "
                         "actual_tds=?, actual_passing_tds=?, "
                         "resolved_at=? WHERE game_id=? AND player_id=?", (*vals, now_iso, *key))
            n_players += 1
    conn.commit()
    return {"games": n_games, "players": n_players}


def accuracy(conn: sqlite3.Connection, season: Optional[int] = None,
             weeks: Optional[tuple[int, int]] = None) -> dict:
    """Tracked accuracy of graded projections (optionally one season / week range).

    Games: model margin/total MAE vs the closing line's on the same games, and
    the share of winners picked. Players (tracked populations: receiving stats
    when proj_targets >= 3, rushing when proj_carries >= 5, passing for the
    starting QB; players who did not play excluded): MAE, bias, and the share
    of results below p10 / at or below p90 (ideal 10% / 90%).
    """
    where, args = ["resolved_at IS NOT NULL"], []
    if season is not None:
        where.append("season = ?"); args.append(season)
    if weeks is not None:
        where.append("week BETWEEN ? AND ?"); args.extend(weeks)
    w = " AND ".join(where)
    g = pd.read_sql_query(f"SELECT * FROM nfl_game_projections WHERE {w}", conn, params=args)
    out: dict = {"games": {"n": int(len(g))}, "players": {}}
    if len(g):
        margin = g["actual_home"] - g["actual_away"]
        total = g["actual_home"] + g["actual_away"]
        has_close = g["close_home_margin"].notna() & g["close_total"].notna()
        decided = margin != 0
        out["games"].update({
            "margin_mae": float((g["proj_margin"] - margin).abs().mean()),
            "total_mae": float((g["proj_total"] - total).abs().mean()),
            "close_margin_mae": float((g.loc[has_close, "close_home_margin"] - margin[has_close]).abs().mean())
            if has_close.any() else None,
            "close_total_mae": float((g.loc[has_close, "close_total"] - total[has_close]).abs().mean())
            if has_close.any() else None,
            "winner_pct": float(((g["proj_margin"] > 0) == (margin > 0))[decided].mean()) if decided.any() else None,
        })
    p = pd.read_sql_query(f"SELECT * FROM nfl_player_projections WHERE {w} AND played = 1", conn, params=args)
    pops = {"receptions": p["proj_targets"] >= TRACK_MIN_TARGETS,
            "receiving_yards": p["proj_targets"] >= TRACK_MIN_TARGETS,
            "rushing_yards": p["proj_carries"] >= TRACK_MIN_CARRIES,
            "passing_yards": p["is_starting_qb"] == 1} if len(p) else {}
    for st, mask in pops.items():
        d = p[mask & p[f"actual_{st}"].notna()]
        if d.empty:
            continue
        err = d[f"proj_{st}"] - d[f"actual_{st}"]
        out["players"][st] = {
            "n": int(len(d)), "mae": float(err.abs().mean()), "bias": float(err.mean()),
            "below_p10": float((d[f"actual_{st}"] < d[f"p10_{st}"]).mean()),
            "at_or_below_p90": float((d[f"actual_{st}"] <= d[f"p90_{st}"]).mean()),
        }
    if len(p) and "p_anytime_td" in p:
        d = p[(pops.get("receiving_yards", False) | pops.get("rushing_yards", False))
              & p["p_anytime_td"].notna() & p["actual_tds"].notna()]
        if len(d):
            y = (d["actual_tds"] >= 1).astype(float)
            pr = d["p_anytime_td"].clip(1e-4, 1 - 1e-4)
            out["players"]["anytime_td"] = {
                "n": int(len(d)), "brier": float(((pr - y) ** 2).mean()),
                "log_loss": float(-(y * np.log(pr) + (1 - y) * np.log(1 - pr)).mean()),
                "mean_p": float(pr.mean()), "rate": float(y.mean()),
            }
        q = p[(p["is_starting_qb"] == 1) & p["proj_passing_tds"].notna() & p["actual_passing_tds"].notna()]
        if len(q):
            err = q["proj_passing_tds"] - q["actual_passing_tds"]
            out["players"]["passing_tds"] = {"n": int(len(q)), "mae": float(err.abs().mean()), "bias": float(err.mean())}
    return out


def latest_week(conn: sqlite3.Connection) -> Optional[tuple[int, int]]:
    """(season, week) of the most recent logged game week, or None."""
    r = conn.execute("SELECT season, week FROM nfl_game_projections ORDER BY season DESC, week DESC LIMIT 1").fetchone()
    return (int(r[0]), int(r[1])) if r else None


def week_rows(conn: sqlite3.Connection, season: int, week: int,
              team: Optional[str] = None) -> tuple[list[dict], list[dict]]:
    """Stored game and player rows for one week (players optionally one team), as dicts."""
    games = [dict(r) for r in conn.execute(
        "SELECT * FROM nfl_game_projections WHERE season=? AND week=? ORDER BY kickoff_utc, game_id",
        (season, week))]
    q = "SELECT * FROM nfl_player_projections WHERE season=? AND week=?"
    args: list = [season, week]
    if team:
        q += " AND team=?"; args.append(team.upper())
    players = [dict(r) for r in conn.execute(q + " ORDER BY kickoff_utc, game_id, team, proj_receiving_yards DESC",
                                             args)]
    return games, players


def favorite_line(home: str, away: str, home_margin: Optional[float]) -> str:
    """Favorite-perspective line from a HOME margin: 'DAL -3.1', 'TB -2.0', 'PK' or '—'."""
    if home_margin is None or (isinstance(home_margin, float) and np.isnan(home_margin)):
        return "—"
    if abs(home_margin) < 0.05:
        return "PK"
    fav = home if home_margin > 0 else away
    return f"{fav} -{abs(home_margin):.1f}"
