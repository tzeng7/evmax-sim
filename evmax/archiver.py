"""Historical data archiver — stores all fetched Pinnacle odds and Kalshi markets.

Captures raw data from every scan cycle so that we build up a free historical
dataset over time (avoiding TheOddsAPI's paid historical data tier).

Schema: data/archive.db
  scan_sessions           — one row per scan cycle
  archived_sharp_odds     — ALL Pinnacle odds fetched (not just matched)
  archived_kalshi_markets — ALL Kalshi market prices fetched (not just matched)
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evmax.models.market import PredictionMarket
    from evmax.models.odds import SharpOdds

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "archive.db"

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS scan_sessions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT    NOT NULL UNIQUE,
    started_at   TEXT    NOT NULL,
    sectors      TEXT    NOT NULL,
    source       TEXT    NOT NULL,
    kalshi_count INTEGER NOT NULL DEFAULT 0,
    sharp_count  INTEGER NOT NULL DEFAULT 0,
    duration_ms  INTEGER
);

CREATE TABLE IF NOT EXISTS archived_sharp_odds (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id           TEXT    NOT NULL,
    fetched_at           TEXT    NOT NULL,
    sector               TEXT    NOT NULL,
    event_id             TEXT    NOT NULL,
    book                 TEXT    NOT NULL,
    outcome_a_label      TEXT,
    outcome_b_label      TEXT,
    outcome_a_decimal    REAL    NOT NULL,
    outcome_b_decimal    REAL    NOT NULL,
    outcome_draw_decimal REAL,
    true_prob_a          REAL    NOT NULL,   -- moneyline-style: 0.0 for props
    true_prob_b          REAL    NOT NULL,   -- moneyline-style: 0.0 for props
    true_prob_draw       REAL,
    margin               REAL    NOT NULL,
    spread_line          REAL,
    event_date           TEXT,
    -- Player-prop specific. Populated when prop_player_name IS NOT NULL.
    -- The non-null prop columns are how we partition prop rows from
    -- moneyline / spread rows in queries.
    true_prob_over       REAL,
    true_prob_under      REAL,
    total_line           REAL,
    prop_player_name     TEXT,
    prop_stat_type       TEXT,
    UNIQUE(session_id, event_id, book)
);

CREATE INDEX IF NOT EXISTS idx_sharp_sector_date
    ON archived_sharp_odds(sector, event_date);
CREATE INDEX IF NOT EXISTS idx_sharp_fetched_at
    ON archived_sharp_odds(fetched_at);

CREATE TABLE IF NOT EXISTS archived_kalshi_markets (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id        TEXT    NOT NULL,
    fetched_at        TEXT    NOT NULL,
    sector            TEXT    NOT NULL,
    ticker            TEXT    NOT NULL,
    title             TEXT,
    market_type       TEXT    NOT NULL,
    yes_price         REAL    NOT NULL,
    no_price          REAL    NOT NULL,
    volume_usd        REAL,
    open_interest_usd REAL,
    team_home         TEXT,
    team_away         TEXT,
    yes_team          TEXT,
    line              REAL,
    event_date        TEXT,
    event_id          TEXT,
    UNIQUE(session_id, ticker)
);

CREATE INDEX IF NOT EXISTS idx_kalshi_sector_date
    ON archived_kalshi_markets(sector, event_date);
CREATE INDEX IF NOT EXISTS idx_kalshi_ticker
    ON archived_kalshi_markets(ticker);

CREATE TABLE IF NOT EXISTS archived_outcomes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker      TEXT    NOT NULL UNIQUE,
    result      INTEGER,          -- 1=YES won, 0=NO won, NULL=still open
    resolved_at TEXT,
    source      TEXT    NOT NULL DEFAULT 'kalshi_api'
);

CREATE INDEX IF NOT EXISTS idx_outcomes_ticker
    ON archived_outcomes(ticker);
"""


_MIGRATIONS = [
    # Add archived_outcomes table if upgrading from older archive.db
    """CREATE TABLE IF NOT EXISTS archived_outcomes (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker      TEXT    NOT NULL UNIQUE,
        result      INTEGER,
        resolved_at TEXT,
        source      TEXT    NOT NULL DEFAULT 'kalshi_api'
    )""",
    "CREATE INDEX IF NOT EXISTS idx_outcomes_ticker ON archived_outcomes(ticker)",
    # Prop columns added 2026-05-10 — pre-existing rows will be NULL on these.
    # Each ALTER is wrapped in try/except in _get_connection() so a second run
    # against a freshly-created table (which already has the columns from
    # SCHEMA) doesn't crash on "duplicate column".
    "ALTER TABLE archived_sharp_odds ADD COLUMN true_prob_over REAL",
    "ALTER TABLE archived_sharp_odds ADD COLUMN true_prob_under REAL",
    "ALTER TABLE archived_sharp_odds ADD COLUMN total_line REAL",
    "ALTER TABLE archived_sharp_odds ADD COLUMN prop_player_name TEXT",
    "ALTER TABLE archived_sharp_odds ADD COLUMN prop_stat_type TEXT",
]


def _get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    for migration in _MIGRATIONS:
        try:
            conn.execute(migration)
            conn.commit()
        except Exception:
            pass
    return conn


def _fmt(dt: object) -> str | None:
    """Convert datetime (or None) to ISO8601 string."""
    if dt is None:
        return None
    if isinstance(dt, datetime):
        return dt.isoformat()
    return str(dt)


class DataArchiver:
    """Archives all raw API data fetched during a scan cycle."""

    def open_session(self, session_id: str, sectors: list[str], source: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with _get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO scan_sessions (session_id, started_at, sectors, source)"
                " VALUES (?, ?, ?, ?)",
                (session_id, now, ",".join(sectors), source),
            )

    def archive_sharp_odds(self, session_id: str, sector: str, odds: list[SharpOdds]) -> int:
        if not odds:
            return 0
        rows = [
            (
                session_id,
                _fmt(o.fetched_at),
                sector,
                o.event_id,
                o.book.value if hasattr(o.book, "value") else str(o.book),
                o.outcome_a_label,
                o.outcome_b_label,
                o.outcome_a_decimal,
                o.outcome_b_decimal,
                o.outcome_draw_decimal,
                o.true_prob_a,
                o.true_prob_b,
                o.true_prob_draw,
                o.margin,
                o.spread_line,
                _fmt(o.event_date),
                # Prop fields — None for moneyline / spread / total rows
                o.true_prob_over,
                o.true_prob_under,
                o.total_line,
                o.prop_player_name,
                o.prop_stat_type,
            )
            for o in odds
        ]
        with _get_connection() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO archived_sharp_odds "
                "(session_id, fetched_at, sector, event_id, book, "
                " outcome_a_label, outcome_b_label, outcome_a_decimal, outcome_b_decimal, "
                " outcome_draw_decimal, true_prob_a, true_prob_b, true_prob_draw, "
                " margin, spread_line, event_date, "
                " true_prob_over, true_prob_under, total_line, "
                " prop_player_name, prop_stat_type) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
        return len(rows)

    def archive_kalshi_markets(self, session_id: str, sector: str, markets: list[PredictionMarket]) -> int:
        if not markets:
            return 0
        rows = [
            (
                session_id,
                _fmt(m.fetched_at),
                sector,
                m.ticker,
                m.title,
                m.market_type.value if hasattr(m.market_type, "value") else str(m.market_type),
                m.yes_price,
                m.no_price,
                m.volume_usd,
                m.open_interest_usd,
                m.team_home,
                m.team_away,
                m.yes_team,
                m.line,
                _fmt(m.event_date),
                m.event_id,
            )
            for m in markets
        ]
        with _get_connection() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO archived_kalshi_markets "
                "(session_id, fetched_at, sector, ticker, title, market_type, "
                " yes_price, no_price, volume_usd, open_interest_usd, "
                " team_home, team_away, yes_team, line, event_date, event_id) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
        return len(rows)

    def get_latest_sharp_odds(self, sector: str, event_date: str) -> list[dict]:
        """Return the most recent archived Pinnacle odds for a sector + event_date.

        Used by LiveScanner when Pinnacle closes pre-game lines once a game starts.
        Returns one row per event_id (latest fetched_at snapshot).
        """
        with _get_connection() as conn:
            rows = conn.execute(
                """
                SELECT a.event_id, a.book, a.outcome_a_label, a.outcome_b_label,
                       a.outcome_a_decimal, a.outcome_b_decimal, a.outcome_draw_decimal,
                       a.true_prob_a, a.true_prob_b, a.true_prob_draw,
                       a.margin, a.spread_line, a.event_date
                FROM archived_sharp_odds a
                INNER JOIN (
                    SELECT event_id, MAX(fetched_at) AS latest_at
                    FROM archived_sharp_odds
                    WHERE sector = ? AND event_date LIKE ?
                    GROUP BY event_id
                ) latest ON a.event_id = latest.event_id
                        AND a.fetched_at = latest.latest_at
                WHERE a.sector = ?
                """,
                (sector, f"{event_date}%", sector),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_unresolved_tickers(self, event_date: str) -> list[str]:
        """Return tickers archived on event_date that have no outcome stored yet."""
        with _get_connection() as conn:
            rows = conn.execute(
                """SELECT DISTINCT k.ticker
                   FROM archived_kalshi_markets k
                   LEFT JOIN archived_outcomes o ON k.ticker = o.ticker
                   WHERE k.event_date = ? AND o.ticker IS NULL""",
                (event_date,),
            ).fetchall()
        return [r["ticker"] for r in rows]

    def store_outcomes(self, outcomes: list[tuple[str, int | None]]) -> int:
        """Bulk-store resolved outcomes. outcomes = [(ticker, result), ...]
        result: 1=YES won, 0=NO won, None=still open (skipped).
        Returns count of newly stored rows.
        """
        settled = [(ticker, result) for ticker, result in outcomes if result is not None]
        if not settled:
            return 0
        now = datetime.now(timezone.utc).isoformat()
        rows = [(ticker, result, now) for ticker, result in settled]
        with _get_connection() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO archived_outcomes (ticker, result, resolved_at)"
                " VALUES (?, ?, ?)",
                rows,
            )
        return len(rows)

    def get_line_history(
        self,
        event_id: str,
        hours: int = 6,
    ) -> list[tuple[str, float]]:
        """Return archived Pinnacle prob snapshots for an event over the last N hours.

        Returns list of (fetched_at_iso, true_prob_a) sorted by time ascending.
        Used for CLV calculation and line velocity (steam) detection.
        """
        with _get_connection() as conn:
            rows = conn.execute(
                """SELECT fetched_at, true_prob_a
                   FROM archived_sharp_odds
                   WHERE event_id = ?
                     AND spread_line IS NULL
                     AND fetched_at >= datetime('now', ?)
                   ORDER BY fetched_at ASC""",
                (event_id, f"-{hours} hours"),
            ).fetchall()
        return [(r["fetched_at"], r["true_prob_a"]) for r in rows]

    def get_closing_line(self, event_id: str) -> float | None:
        """Return the last archived Pinnacle true_prob_a for an event (moneyline only).

        This is the 'closing line' — the last snapshot before the game started.
        Used for CLV calculation. DEPRECATED for ev_outcomes writes — prefer
        :meth:`get_closing_line_aligned` so CLV reflects the YES side actually
        bet on rather than always-side-a.
        """
        with _get_connection() as conn:
            row = conn.execute(
                """SELECT true_prob_a
                   FROM archived_sharp_odds
                   WHERE event_id = ?
                     AND spread_line IS NULL
                   ORDER BY fetched_at DESC
                   LIMIT 1""",
                (event_id,),
            ).fetchone()
        return row["true_prob_a"] if row else None

    def get_closing_line_aligned(self, event_id: str, yes_team: str | None) -> float | None:
        """Return the closing Pinnacle prob aligned to the bet's YES side.

        Pinnacle records (outcome_a, outcome_b) by its own convention; Kalshi
        bets can be on either side. Returning true_prob_a blindly inverts CLV
        for every bet where yes_team == outcome_b. This helper reads both
        labels + probs and picks the correct side via
        :func:`evmax.agents.cleanup.resolver.yes_aligned_close_prob`.

        Filters snapshots to `fetched_at < event_date` (i.e. strictly
        pre-tipoff) — Pinnacle keeps quoting odds during live games and
        immediately after settlement, and the post-tipoff snapshots collapse
        toward the eventual outcome (winners trending to 1.0, losers to 0.0).
        Without this filter, ``ORDER BY fetched_at DESC LIMIT 1`` picks the
        last archived sample which is almost always post-tipoff for any
        event resolved before the next archive run.
        """
        from evmax.agents.cleanup.resolver import yes_aligned_close_prob

        with _get_connection() as conn:
            row = conn.execute(
                """SELECT outcome_a_label, outcome_b_label,
                          true_prob_a, true_prob_b, true_prob_draw
                   FROM archived_sharp_odds
                   WHERE event_id = ?
                     AND spread_line IS NULL
                     AND event_date IS NOT NULL
                     AND fetched_at < event_date
                   ORDER BY fetched_at DESC
                   LIMIT 1""",
                (event_id,),
            ).fetchone()
        if not row:
            return None
        return yes_aligned_close_prob(
            yes_team=yes_team,
            outcome_a_label=row["outcome_a_label"],
            outcome_b_label=row["outcome_b_label"],
            true_prob_a=row["true_prob_a"],
            true_prob_b=row["true_prob_b"],
            true_prob_draw=row["true_prob_draw"],
        )

    def close_session(
        self,
        session_id: str,
        duration_ms: int,
        kalshi_count: int,
        sharp_count: int,
    ) -> None:
        with _get_connection() as conn:
            conn.execute(
                "UPDATE scan_sessions SET duration_ms=?, kalshi_count=?, sharp_count=?"
                " WHERE session_id=?",
                (duration_ms, kalshi_count, sharp_count, session_id),
            )
