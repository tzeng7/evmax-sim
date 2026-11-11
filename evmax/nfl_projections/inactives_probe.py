"""Record-only probe: when do game-day inactives reach our two live sources?

Phase 0 of the pre-kickoff inactives refresh (see the Week 5 review). The
projection pipeline already treats a player as out when the nflverse weekly
roster lists him ``INA`` or the ESPN injury feed lists him Out/Doubtful/IR. What
nobody has measured is the LAG: the league posts the inactive list about 90
minutes before kickoff, and the Week 5 Sunday refresh (21 minutes after the 1pm
ET deadline) still projected a Questionable WR who was inactive. Before any
refresh task is built, this probe snapshots both sources around each kickoff
and answers "how many minutes before kickoff does each source show the
inactive?".

The probe writes only to its own two tables in ``data/projections.db``. It never
reads or writes ``nfl_game_projections`` / ``nfl_player_projections`` and it never
touches the cached nflverse roster file the projection pipeline reads: the
roster is fetched into memory.

Tables:

* ``nfl_inactives_probe_runs`` — one row per (snapshot, game, source, team) with
  coverage: did the fetch work, how many rows, which roster week, the HTTP
  ``Last-Modified`` of the nflverse file (the publish time), and status counts.
* ``nfl_inactives_probe`` — the players seen: roster ``INA`` rows and every
  non-Active ESPN injury entry for the two teams.
"""
from __future__ import annotations

import io
import json
import sqlite3
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import pandas as pd
import structlog

from evmax.nfl_projections import data, live, store
from evmax.nfl_projections.player_model import ROSTER_ACTIVE, ROSTER_INACTIVE

logger = structlog.get_logger(__name__)

SOURCE_ROSTER = "nflverse_roster"
SOURCE_ESPN = "espn_injury"

# Inactives are due 90 minutes before kickoff. Start 150 minutes out so the
# snapshots before the deadline show what was visible while a player was still
# only Questionable; keep going 30 minutes past kickoff to catch a late publish.
DEFAULT_LOOKAHEAD_MIN = 150
DEFAULT_LOOKBACK_MIN = 30

# ESPN statuses that mean "not playing" (same set the projection pipeline uses).
ESPN_OUT_LIKE = live.ESPN_OUT_STATUSES

_SCHEMA = """
CREATE TABLE IF NOT EXISTS nfl_inactives_probe_runs (
    snapshot_at         TEXT NOT NULL,
    game_id             TEXT NOT NULL,
    source              TEXT NOT NULL,
    team                TEXT NOT NULL,
    season              INTEGER NOT NULL,
    week                INTEGER NOT NULL,
    kickoff_utc         TEXT,
    minutes_to_kickoff  REAL,
    ok                  INTEGER NOT NULL,
    n_rows              INTEGER,
    roster_week         INTEGER,
    status_counts       TEXT,
    source_updated      TEXT,
    PRIMARY KEY (snapshot_at, game_id, source, team)
);
CREATE TABLE IF NOT EXISTS nfl_inactives_probe (
    snapshot_at         TEXT NOT NULL,
    game_id             TEXT NOT NULL,
    source              TEXT NOT NULL,
    team                TEXT NOT NULL,
    player_id           TEXT,
    player_name         TEXT NOT NULL,
    position            TEXT,
    status              TEXT NOT NULL,
    reported_at         TEXT,
    PRIMARY KEY (snapshot_at, game_id, source, team, player_name, status)
);
CREATE INDEX IF NOT EXISTS idx_nfl_inactives_probe_game ON nfl_inactives_probe(game_id, source);
"""


def connect(path=None) -> sqlite3.Connection:
    """projections.db with the probe tables added (the projection tables are left as they are)."""
    conn = store.connect(path)
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


# ── which games to probe ─────────────────────────────────────────────────────

def games_in_window(games: pd.DataFrame, now: datetime, lookahead_min: int = DEFAULT_LOOKAHEAD_MIN,
                    lookback_min: int = DEFAULT_LOOKBACK_MIN) -> list[dict]:
    """Games whose kickoff is within [now - lookback, now + lookahead], from the nflverse schedule.

    The schedule (not the stored projections) is the source, so a game the
    projection pipeline never stored — the Week 5 London game — is still probed.
    """
    now = now.astimezone(timezone.utc)
    lo, hi = now - timedelta(minutes=lookback_min), now + timedelta(minutes=lookahead_min)
    out: list[dict] = []
    for g in games.itertuples(index=False):
        ko = store.kickoff_utc(g.gameday, getattr(g, "gametime", None))
        if ko is None:
            continue
        kdt = datetime.fromisoformat(ko).astimezone(timezone.utc)
        if lo <= kdt <= hi:
            out.append({"game_id": g.game_id, "season": int(g.season), "week": int(g.week),
                        "kickoff_utc": ko, "kickoff": kdt, "home": g.home_team, "away": g.away_team})
    return out


# ── fetchers (injectable for tests) ──────────────────────────────────────────

def fetch_roster(season: int, timeout: int = 60) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """(weekly roster frame, HTTP Last-Modified) straight from nflverse, held in memory.

    Not written to the projection cache: ``data.ensure_rosters`` owns that file and
    the projection run may be reading it. (None, None) on any failure.
    """
    url = f"{data.NFLVERSE}/weekly_rosters/roster_weekly_{season}.parquet"
    req = urllib.request.Request(url, headers={"User-Agent": "evmax-nfl-inactives-probe",
                                               "Cache-Control": "no-cache"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            modified = r.headers.get("Last-Modified")
            body = r.read()
        df = pd.read_parquet(io.BytesIO(body))
    except (urllib.error.URLError, OSError, ValueError) as e:
        logger.warning("nfl_inactives_probe_roster_failed", error=str(e))
        return None, None
    df["team"] = df["team"].replace(data.TEAM_ALIASES)
    return df, modified


def fetch_espn() -> dict:
    """ESPN injury feed ({} on failure, as ``live.fetch_espn_injury_reports``)."""
    return live.fetch_espn_injury_reports()


# ── building rows ────────────────────────────────────────────────────────────

def roster_observations(roster: pd.DataFrame, week: int, team: str) -> tuple[dict, list[dict]]:
    """(coverage, INA players) for ``team`` from the weekly roster frame.

    ``roster_week`` is the team's latest published roster week at or before
    ``week``: when it is below ``week``, this week's roster is not out yet.
    INA players are listed only when the roster is the week's own (an earlier
    week's INA belonged to another game).
    """
    r = roster[(roster["team"] == team) & (roster["week"] <= week)]
    if r.empty:
        return {"n_rows": 0, "roster_week": None, "status_counts": {}}, []
    rw = int(r["week"].max())
    cur = r[r["week"] == rw]
    counts = {str(k): int(v) for k, v in cur["status"].value_counts().items()}
    ina: list[dict] = []
    if rw == week:
        for p in cur[cur["status"] == ROSTER_INACTIVE].itertuples(index=False):
            ina.append({"player_id": None if pd.isna(p.gsis_id) else str(p.gsis_id),
                        "player_name": str(p.full_name), "position": None if pd.isna(p.position) else str(p.position),
                        "status": ROSTER_INACTIVE, "reported_at": None})
    return {"n_rows": int(len(cur)), "roster_week": rw, "status_counts": counts}, ina


def espn_observations(reports: dict, team: str) -> tuple[dict, list[dict]]:
    """(coverage, listed players) for ``team`` from the ESPN feed."""
    players: list[dict] = []
    for name, report in (reports or {}).items():
        if live._espn_abbr(name) != team:
            continue
        for p in report.players:
            players.append({"player_id": None, "player_name": p.name, "position": p.position or None,
                            "status": p.status.upper().strip(), "reported_at": p.reported_at})
    counts: dict[str, int] = {}
    for p in players:
        counts[p["status"]] = counts.get(p["status"], 0) + 1
    return {"n_rows": len(players), "roster_week": None, "status_counts": counts}, players


# ── one snapshot ─────────────────────────────────────────────────────────────

def probe_once(conn: sqlite3.Connection, now: Optional[datetime] = None, games: Optional[pd.DataFrame] = None,
               roster_fetch: Callable = fetch_roster, espn_fetch: Callable = fetch_espn,
               lookahead_min: int = DEFAULT_LOOKAHEAD_MIN, lookback_min: int = DEFAULT_LOOKBACK_MIN) -> dict:
    """Snapshot both sources for every game near kickoff. Returns a small summary.

    Does nothing (no network) when no game is inside the window. A failed fetch
    is recorded as an ``ok = 0`` run row, never raised.
    """
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if games is None:
        data.ensure_games()
        games = data.load_games()
    wave = games_in_window(games, now, lookahead_min, lookback_min)
    if not wave:
        return {"games": 0, "run_rows": 0, "player_rows": 0}

    snap = now.isoformat(timespec="seconds")
    rosters: dict[int, tuple[Optional[pd.DataFrame], Optional[str]]] = {}
    for season in sorted({g["season"] for g in wave}):
        rosters[season] = roster_fetch(season)
    espn = espn_fetch()
    espn_ok = bool(espn)

    run_rows = player_rows = 0
    for g in wave:
        mtk = round((g["kickoff"] - now).total_seconds() / 60.0, 1)
        roster, modified = rosters[g["season"]]
        for team in (g["home"], g["away"]):
            for source, ok in ((SOURCE_ROSTER, roster is not None), (SOURCE_ESPN, espn_ok)):
                if not ok:
                    cov, players = {"n_rows": None, "roster_week": None, "status_counts": {}}, []
                elif source == SOURCE_ROSTER:
                    cov, players = roster_observations(roster, g["week"], team)
                else:
                    cov, players = espn_observations(espn, team)
                conn.execute(
                    "INSERT OR REPLACE INTO nfl_inactives_probe_runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (snap, g["game_id"], source, team, g["season"], g["week"], g["kickoff_utc"], mtk, int(ok),
                     cov["n_rows"], cov["roster_week"], json.dumps(cov["status_counts"], sort_keys=True),
                     modified if source == SOURCE_ROSTER else None))
                run_rows += 1
                for p in players:
                    conn.execute("INSERT OR REPLACE INTO nfl_inactives_probe VALUES (?,?,?,?,?,?,?,?,?)",
                                 (snap, g["game_id"], source, team, p["player_id"], p["player_name"],
                                  p["position"], p["status"], p["reported_at"]))
                    player_rows += 1
    conn.commit()
    return {"games": len(wave), "run_rows": run_rows, "player_rows": player_rows}


# ── readout ──────────────────────────────────────────────────────────────────

def lead_times(conn: sqlite3.Connection, season: Optional[int] = None, week: Optional[int] = None) -> pd.DataFrame:
    """Per (game, source, team, player): the first snapshot that showed him out, in minutes before kickoff.

    Out-like = roster ``INA`` or an ESPN status in ``ESPN_OUT_LIKE``. A positive
    ``lead_min`` means the source showed the player out BEFORE kickoff (usable
    by a refresh); negative means only after. ``first_listed_min`` is the first
    appearance in any status (Questionable counts), so the gap between the two
    columns is how long a player sat on the report before the source resolved him.
    """
    q = ("SELECT p.snapshot_at, p.game_id, p.source, p.team, p.player_name, p.status, r.minutes_to_kickoff, r.week "
         "FROM nfl_inactives_probe p JOIN nfl_inactives_probe_runs r "
         "ON r.snapshot_at = p.snapshot_at AND r.game_id = p.game_id AND r.source = p.source AND r.team = p.team")
    args: list = []
    if season is not None:
        q += " WHERE r.season = ?"
        args.append(season)
        if week is not None:
            q += " AND r.week = ?"
            args.append(week)
    rows = pd.read_sql_query(q + " ORDER BY p.snapshot_at", conn, params=args)
    if rows.empty:
        return pd.DataFrame(columns=["week", "game_id", "source", "team", "player_name", "first_listed_min",
                                     "first_out_min", "lead_min"])
    out_like = (((rows["source"] == SOURCE_ROSTER) & (rows["status"] == ROSTER_INACTIVE))
                | ((rows["source"] == SOURCE_ESPN) & rows["status"].isin(ESPN_OUT_LIKE)))
    key = ["week", "game_id", "source", "team", "player_name"]
    first_any = rows.groupby(key)["minutes_to_kickoff"].max().rename("first_listed_min")
    first_out = rows[out_like].groupby(key)["minutes_to_kickoff"].max().rename("first_out_min")
    res = pd.concat([first_any, first_out], axis=1).reset_index()
    res["lead_min"] = res["first_out_min"]
    return res.sort_values(["week", "game_id", "source", "team", "player_name"]).reset_index(drop=True)


def source_lag(conn: sqlite3.Connection, season: Optional[int] = None, week: Optional[int] = None) -> pd.DataFrame:
    """Per (game, source): the first snapshot where this week's roster existed / any INA or ESPN Out appeared."""
    q = "SELECT * FROM nfl_inactives_probe_runs"
    args: list = []
    if season is not None:
        q += " WHERE season = ?"
        args.append(season)
        if week is not None:
            q += " AND week = ?"
            args.append(week)
    runs = pd.read_sql_query(q, conn, params=args)
    if runs.empty:
        return pd.DataFrame()
    runs["roster_current"] = (runs["source"] == SOURCE_ROSTER) & (runs["roster_week"] == runs["week"])
    g = runs.groupby(["game_id", "source"])
    out = g.agg(snapshots=("snapshot_at", "nunique"), failed=("ok", lambda s: int((s == 0).sum())),
                last_modified=("source_updated", "max")).reset_index()
    cur = runs[runs["roster_current"]].groupby(["game_id", "source"])["minutes_to_kickoff"].max().rename("week_roster_first_min")
    return out.merge(cur.reset_index(), on=["game_id", "source"], how="left")
