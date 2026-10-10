"""Model picks against the Vegas line, plus an honest record.

A pick compares the game model's projection with a market line, the way
public model pages do ("model DAL -2.6, market DAL -8.5 -> TB +8.5"):

* **Spread (ATS).** Pick the home side when the projected home margin is above
  the line's home margin, else the away side. Edge = the gap in points.
* **Total (O/U).** Pick the over when the model's MEDIAN total is above the
  line. The game model projects the MEAN; NFL totals are right-skewed and
  books set totals near the median, so the mean sits ~0.9 pts high and
  picked the over 64% of the time (walk-forward 2011-25: 48% after the shift).
* **No cover probability.** The model's own P(cover) at the line is badly
  overconfident: picks it rated 62-73% to cover covered 49-51% (2011-25), so
  the edge in points is shown instead.
* **Check-news flags.** A big edge, a starting-QB change or a Week 14+ game is
  where the model is most often missing information (QB changes and late
  season are 33% / 38% of 5+ pt edges, which went 96-118 vs the close).

The record stores each game's FIRST published pick and the line it was made
against, and never re-prices it. It grades every pick twice — against that
line and against the closing line — and tracks whether the line moved toward
the pick. Backtest context (scripts/backtest_nfl_model_picks.py): 2011-25
spread picks went 50.2% vs the close (n 3,818) and 50.1% vs ESPN opening
lines (n 1,462), while the opening line moved toward the pick 59% of the time
(699 vs 492, +0.40 pts) — the model anticipates line moves, not results.

Line convention: ``line_margin`` is the expected HOME margin, nflverse's
``spread_line`` sign (+3 = home favored by 3); the home handicap is its
negative.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

# Walk-forward median of (actual total - projected total), REG 2016-24 = -0.93
# (2025 holdout -0.86); the shipped game model's mean projection runs high
# against a median-set line.
TOTAL_MEDIAN_SHIFT = -0.9
BIG_EDGE_PTS = 5.0          # 5+ pt edges: 44.9% vs the close 2016-25 (51.1% over 2011-25), a third QB changes
LATE_SEASON_WEEK = 14       # weeks 14-18 carry 35% of the model's gap to the close
QB_CHANGE_DELTA = 0.05      # |starter - team QB level| in EPA/dropback (the backtest's QB-change cut)


@dataclass(frozen=True)
class GamePick:
    spread_pick_home: Optional[bool]   # None = model equals the line (no pick)
    spread_pick: Optional[str]         # "TB +8.5"
    spread_edge: Optional[float]
    total_median: float
    total_pick: Optional[str]          # "Over 47.5" / "Under 47.5"
    total_pick_over: Optional[bool]
    total_edge: Optional[float]
    flags: tuple[str, ...]


def handicap_str(team: str, handicap: float) -> str:
    """'TB +8.5', 'DAL -3', 'NE PK'."""
    if abs(handicap) < 1e-9:
        return f"{team} PK"
    return f"{team} {handicap:+g}"


def make_pick(home: str, away: str, proj_margin: float, proj_total: float,
              line_margin: Optional[float], line_total: Optional[float], week: int,
              home_qb_delta: float = 0.0, away_qb_delta: float = 0.0) -> GamePick:
    """Spread and total picks for one game against ``line_margin`` / ``line_total``."""
    total_median = proj_total + TOTAL_MEDIAN_SHIFT
    pick_home = pick = edge = None
    if line_margin is not None and pd.notna(line_margin):
        edge = abs(proj_margin - line_margin)
        if proj_margin != line_margin:
            pick_home = bool(proj_margin > line_margin)
            pick = handicap_str(home, -line_margin) if pick_home else handicap_str(away, line_margin)
    over = tpick = tedge = None
    if line_total is not None and pd.notna(line_total):
        tedge = abs(total_median - line_total)
        if total_median != line_total:
            over = bool(total_median > line_total)
            tpick = f"{'Over' if over else 'Under'} {line_total:g}"
    flags = []
    if edge is not None and edge >= BIG_EDGE_PTS:
        flags.append("big edge")
    if max(abs(home_qb_delta or 0.0), abs(away_qb_delta or 0.0)) > QB_CHANGE_DELTA:
        flags.append("QB change")
    if week >= LATE_SEASON_WEEK:
        flags.append("late season")
    return GamePick(pick_home, pick, edge, total_median, tpick, over, tedge, tuple(flags))


def grade_spread(pick_home: Optional[bool], line_margin: Optional[float], margin: float) -> Optional[str]:
    """'W' / 'L' / 'P' for a spread pick graded at ``line_margin``; None if ungradable."""
    if pick_home is None or line_margin is None or pd.isna(line_margin) or pd.isna(margin):
        return None
    if margin == line_margin:
        return "P"
    return "W" if (margin > line_margin) == pick_home else "L"


def grade_total(pick_over: Optional[bool], line_total: Optional[float], total: float) -> Optional[str]:
    if pick_over is None or line_total is None or pd.isna(line_total) or pd.isna(total):
        return None
    if total == line_total:
        return "P"
    return "W" if (total > line_total) == pick_over else "L"


def line_move_toward(pick_up: Optional[bool], line_then: Optional[float], line_later: Optional[float]) -> Optional[float]:
    """Points the line moved TOWARD the pick between two lines (negative = away).

    ``pick_up`` is True when the pick wants the number to rise: the home side
    for a spread (home-margin line) or the over for a total.
    """
    if pick_up is None or line_then is None or line_later is None or pd.isna(line_then) or pd.isna(line_later):
        return None
    d = float(line_later) - float(line_then)
    return d if pick_up else -d


def add_picks(proj: pd.DataFrame) -> pd.DataFrame:
    """Append pick columns to a ``live.project_week`` frame (uses its market lines)."""
    picks = [make_pick(r.home_team, r.away_team, r.proj_margin, r.proj_total,
                       r.market_spread_line, r.market_total_line, int(r.week),
                       r.home_qb_delta, r.away_qb_delta) for r in proj.itertuples()]
    out = proj.copy()
    for field in GamePick.__dataclass_fields__:
        out[field] = [getattr(p, field) for p in picks]
    return out


# ── record (stored in data/projections.db) ───────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS nfl_picks (
    game_id             TEXT PRIMARY KEY,   -- one graded pick per game: the FIRST publish, never re-priced
    season              INTEGER NOT NULL,
    week                INTEGER NOT NULL,
    gameday             TEXT NOT NULL,
    home_team           TEXT NOT NULL,
    away_team           TEXT NOT NULL,
    published_at        TEXT NOT NULL,
    proj_home           REAL NOT NULL,
    proj_away           REAL NOT NULL,
    proj_margin         REAL NOT NULL,
    proj_total          REAL NOT NULL,
    total_median        REAL NOT NULL,
    line_margin         REAL,               -- line at publish, expected HOME margin (+ = home favored)
    line_total          REAL,
    line_source         TEXT,
    spread_pick         TEXT,
    spread_pick_home    INTEGER,
    spread_edge         REAL,
    total_pick          TEXT,
    total_pick_over     INTEGER,
    total_edge          REAL,
    flags               TEXT,
    home_score          REAL,
    away_score          REAL,
    close_margin        REAL,
    close_total         REAL,
    spread_result       TEXT,               -- W/L/P at the published line
    spread_result_close TEXT,               -- W/L/P at the closing line
    total_result        TEXT,
    total_result_close  TEXT,
    spread_move         REAL,               -- points the line moved toward the pick by the close
    total_move          REAL,
    graded_at           TEXT
);
"""


def connect(path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record_picks(conn: sqlite3.Connection, picks: pd.DataFrame, now: Optional[pd.Timestamp] = None,
                 line_source: str = "nflverse consensus") -> int:
    """Store each game's first pick made BEFORE kickoff with a spread line; returns rows inserted.

    An existing row is never re-priced. The one exception: a row stored before
    the total was posted gets its total pick once a total line exists.
    """
    now = now if now is not None else pd.Timestamp.now(tz="America/New_York")
    inserted = 0
    for r in picks.itertuples():
        if r.spread_pick is None or kickoff(r.gameday, r.gametime) <= now:
            continue
        cur = conn.execute(
            """INSERT OR IGNORE INTO nfl_picks
               (game_id, season, week, gameday, home_team, away_team, published_at,
                proj_home, proj_away, proj_margin, proj_total, total_median,
                line_margin, line_total, line_source, spread_pick, spread_pick_home, spread_edge,
                total_pick, total_pick_over, total_edge, flags)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (r.game_id, int(r.season), int(r.week), str(r.gameday), r.home_team, r.away_team, _now(),
             float(r.proj_home), float(r.proj_away), float(r.proj_margin), float(r.proj_total), float(r.total_median),
             _f(r.market_spread_line), _f(r.market_total_line), line_source, r.spread_pick,
             int(r.spread_pick_home), _f(r.spread_edge),
             r.total_pick, None if r.total_pick_over is None else int(r.total_pick_over), _f(r.total_edge),
             ", ".join(r.flags)))
        inserted += cur.rowcount
        if not cur.rowcount and r.total_pick is not None:
            conn.execute("""UPDATE nfl_picks SET line_total = ?, total_pick = ?, total_pick_over = ?, total_edge = ?
                            WHERE game_id = ? AND total_pick IS NULL""",
                         (_f(r.market_total_line), r.total_pick, int(r.total_pick_over), _f(r.total_edge), r.game_id))
    conn.commit()
    return inserted


def grade_picks(conn: sqlite3.Connection, games: pd.DataFrame) -> int:
    """Grade stored picks whose game has a final score; returns rows graded.

    nflverse's ``spread_line`` / ``total_line`` on a finished game are the
    closing consensus lines.
    """
    final = games[games["home_score"].notna()].set_index("game_id")
    n = 0
    for row in conn.execute("SELECT * FROM nfl_picks WHERE graded_at IS NULL").fetchall():
        if row["game_id"] not in final.index:
            continue
        g = final.loc[row["game_id"]]
        margin, total = g["home_score"] - g["away_score"], g["home_score"] + g["away_score"]
        cm, ct = _f(g.get("spread_line")), _f(g.get("total_line"))
        ph = None if row["spread_pick_home"] is None else bool(row["spread_pick_home"])
        po = None if row["total_pick_over"] is None else bool(row["total_pick_over"])
        conn.execute(
            """UPDATE nfl_picks SET home_score = ?, away_score = ?, close_margin = ?, close_total = ?,
               spread_result = ?, spread_result_close = ?, total_result = ?, total_result_close = ?,
               spread_move = ?, total_move = ?, graded_at = ? WHERE game_id = ?""",
            (float(g["home_score"]), float(g["away_score"]), cm, ct,
             grade_spread(ph, row["line_margin"], margin), grade_spread(ph, cm, margin),
             grade_total(po, row["line_total"], total), grade_total(po, ct, total),
             line_move_toward(ph, row["line_margin"], cm), line_move_toward(po, row["line_total"], ct),
             _now(), row["game_id"]))
        n += 1
    conn.commit()
    return n


def record_summary(rows: pd.DataFrame) -> dict[str, str]:
    """W-L-P strings for each result column of graded ``nfl_picks`` rows."""
    out = {}
    for col in ("spread_result", "spread_result_close", "total_result", "total_result_close"):
        s = rows[col].dropna()
        w, l, p = (s == "W").sum(), (s == "L").sum(), (s == "P").sum()
        out[col] = f"{w}-{l}" + (f"-{p}" if p else "") + (f"  {w / (w + l):.1%}" if w + l else "")
    return out


def kickoff(gameday, gametime) -> pd.Timestamp:
    """Kickoff as an America/New_York timestamp (nflverse ``gametime`` is ET; 13:00 when unknown)."""
    t = gametime if isinstance(gametime, str) and gametime else "13:00"
    return pd.Timestamp(f"{pd.Timestamp(gameday).date()} {t}", tz="America/New_York")


def _f(x) -> Optional[float]:
    return None if x is None or pd.isna(x) else float(x)
