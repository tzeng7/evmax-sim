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

import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

# Walk-forward median of (actual total - projected total), REG 2016-24 = -0.93
# (2025 holdout -0.86); the shipped game model's mean projection runs high
# against a median-set line.
TOTAL_MEDIAN_SHIFT = -0.9
# Calibrated P(win) of a pick vs its edge: logistic sigmoid(a + b * min(edge, CALIBRATION_MAX_EDGE)),
# fit on the 2011-25 walk-forward picks graded at the close, then shrunk per coefficient by the
# positive-part James-Stein factor max(0, 1 - 1/z^2) so that noise sizes nothing
# (scripts/backtest_nfl_model_picks.py --first-feature-season 2010 --seasons 2011-2025 --calibration).
# Raw fits: spread a -0.010 (z -0.2), b +0.0097 (z +0.5), n 3,818 -> shrunk to a flat 50%;
# total a -0.051 (z -1.0), b +0.024 (z +1.1), n 3,883 -> 50.8% at an 8-pt edge. So at -110
# (52.4% break-even) every stake is $0; only near-50c maker prices size anything, and only a little.
SPREAD_COVER_LOGIT = (0.0, 0.0)
TOTAL_WIN_LOGIT = (0.0, 0.0039)
CALIBRATION_MAX_EDGE = 8.0
KELLY_CAP = 0.05            # evmax's per-bet hard cap (evmax/ev/kelly.py, validated by scripts/backtest_sizing.py)
DEFAULT_KELLY = 0.5         # half Kelly, the evmax CLI default
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
    spread_p: Optional[float] = None   # calibrated P(the spread pick covers | no push)
    total_p: Optional[float] = None    # calibrated P(the O/U pick wins | no push)


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
    return GamePick(pick_home, pick, edge, total_median, tpick, over, tedge, tuple(flags),
                    spread_p=calibrated_prob(edge, "spread") if pick is not None else None,
                    total_p=calibrated_prob(tedge, "total") if tpick is not None else None)


# ── calibrated probability and stake sizing ──────────────────────────────────

def calibrated_prob(edge: Optional[float], kind: str) -> Optional[float]:
    """P(a pick with this edge wins | no push), from the backtest fit (``kind``: spread / total)."""
    if edge is None or pd.isna(edge):
        return None
    a, b = SPREAD_COVER_LOGIT if kind == "spread" else TOTAL_WIN_LOGIT
    x = min(float(edge), CALIBRATION_MAX_EDGE)
    return 1.0 / (1.0 + math.exp(-(a + b * x)))


def breakeven(american_odds: float) -> float:
    """The win probability a bet at ``american_odds`` needs to break even (-110 -> 0.524)."""
    o = float(american_odds)
    if -100 < o < 100:
        raise ValueError(f"American odds must be <= -100 or >= +100, got {american_odds}")
    return -o / (-o + 100.0) if o < 0 else 100.0 / (o + 100.0)


def american_from_breakeven(price: float) -> float:
    """American odds of a bet that costs ``price`` (fee included) to win 1."""
    return -100.0 * price / (1.0 - price) if price >= 0.5 else 100.0 * (1.0 - price) / price


def stake_fraction(p: Optional[float], american_odds: float, kelly: float = DEFAULT_KELLY,
                   cap: float = KELLY_CAP) -> float:
    """Fraction of bankroll to stake: ``kelly`` x full Kelly at these odds, 0 when the bet is
    not +EV, capped at ``cap``."""
    from evmax.ev.kelly import kelly_full

    if p is None:
        return 0.0
    decimal = 1.0 / breakeven(american_odds)
    return min(cap, max(0.0, kelly_full(p, decimal)) * kelly)


def price_presets() -> list[dict]:
    """Where a pick could be bet, as American odds: a -110 sportsbook and a ~50c exchange
    contract on Kalshi / Polymarket US with that venue's fee (evmax/fees.py) as taker or maker."""
    from evmax import fees

    out = [{"key": "book", "label": "Sportsbook -110", "odds": -110.0}]
    for key, label, fee in (("kalshi_taker", "Kalshi taker", fees.kalshi_fee_prob(0.5)),
                            ("kalshi_maker", "Kalshi maker", fees.kalshi_fee_prob(0.5, maker=True)),
                            ("polymarket_us_taker", "Polymarket US taker", fees.polymarket_us_fee_prob(0.5)),
                            ("polymarket_us_maker", "Polymarket US maker", fees.polymarket_us_fee_prob(0.5, maker=True))):
        odds = round(american_from_breakeven(0.5 + fee), 1)
        out.append({"key": key, "label": f"{label} at 50c ({odds:+.0f})", "odds": odds})
    return out


def sizing_spec() -> dict:
    """What the dashboard needs to size picks client-side."""
    return {
        "presets": price_presets(), "default_kelly": DEFAULT_KELLY, "cap": KELLY_CAP,
        "note": ("Stake = Kelly on the calibrated win probability (the 2011-25 backtest's win rate at that "
                 "edge, shrunk toward 50% where the evidence is weak), capped at 5% of bankroll. Spread picks "
                 "calibrate to 50% and totals to at most 50.8%, so at -110 (52.4% break-even) every stake is "
                 "$0; maker prices assume a fill at 50c."),
    }


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
        # object dtype keeps None as None: a plain list of strings and Nones would turn the
        # missing picks of a mixed slate into NaN, which is truthy and `is not None`.
        out[field] = pd.Series([getattr(p, field) for p in picks], index=out.index, dtype=object)
    return out


# ── record (``nfl_picks`` in projections.db, beside evmax.nfl_projections.store) ──

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


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def connect(path=None) -> sqlite3.Connection:
    """Open projections.db (``store.db_path()`` by default, which honors $EVMAX_PROJ_DB) with ``nfl_picks``."""
    from evmax.nfl_projections import store

    conn = store.connect(path)
    ensure_schema(conn)
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
    ensure_schema(conn)
    inserted = 0
    for r in picks.itertuples():
        if _missing(r.spread_pick) or kickoff(r.gameday, r.gametime) <= now:
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
             None if _missing(r.total_pick) else r.total_pick,
             None if _missing(r.total_pick_over) else int(r.total_pick_over), _f(r.total_edge),
             ", ".join(r.flags)))
        inserted += cur.rowcount
        if not cur.rowcount and not _missing(r.total_pick):
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
    ensure_schema(conn)
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


def _missing(x) -> bool:
    """None or NaN (a pick column read back through pandas may carry either)."""
    return x is None or (isinstance(x, float) and x != x)
