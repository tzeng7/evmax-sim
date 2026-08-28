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

# When a spread/total bet's line has no exact archived snapshot, we fall back to
# the nearest pre-tipoff snapshot — but only within this tolerance (points).
# Pre-tipoff line drift of ±0.5–1.0pt is routine and shouldn't kill CLV
# measurement. But the archive only captures Pinnacle's PRIMARY (balanced) line
# per game, which always devigs to ~0.50; an unbounded fallback would silently
# match an alternate-spread bet (e.g. dog -5.5) to a several-point-off main line
# (e.g. -8.0 at 0.50) and record a fabricated ~0.50 "close". Bounding the
# fallback makes missing alternate-line data read as None (no data) instead.
_CLOSING_LINE_FALLBACK_TOLERANCE = 1.0

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
CREATE INDEX IF NOT EXISTS idx_sharp_event_id
    ON archived_sharp_odds(event_id);

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

-- Order-book depth time series for the listing→scan window (2026-07-01).
-- One row per (sweep, ticker) from `cleanup watch-listings`. Distinguishes an
-- MM placeholder quote (yes_ask set, *_depth ~0) from a fillable market — the
-- signal the depth-aware entry rule and lay-side CLV gate evaluate on.
CREATE TABLE IF NOT EXISTS archived_orderbook_depth (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id         TEXT    NOT NULL,
    fetched_at         TEXT    NOT NULL,
    sector             TEXT    NOT NULL,
    ticker             TEXT    NOT NULL,
    yes_ask            REAL,   -- 1 − best NO bid (price a YES taker pays)
    yes_ask_depth_usd  REAL,   -- $ resting at best NO bid (YES-taker fillable size)
    yes_bid            REAL,   -- best resting YES bid
    yes_bid_depth_usd  REAL,   -- $ at best YES bid
    yes_book_usd       REAL,   -- total $ on the YES bid ladder
    no_book_usd        REAL,   -- total $ on the NO bid ladder
    source             TEXT,   -- 'ws' | 'rest'
    UNIQUE(session_id, ticker)
);

CREATE INDEX IF NOT EXISTS idx_orderbook_depth_ticker
    ON archived_orderbook_depth(ticker);
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

    def archive_kalshi_snapshot(
        self,
        session_id: str,
        sector: str,
        snapshots: list[dict],
        fetched_at: object = None,
    ) -> int:
        """Lightweight near-tipoff Kalshi price capture for CLV close anchoring.

        The full ``archive_kalshi_markets`` path needs PredictionMarket objects;
        a near-tip price sweep (see ``cleanup watch-closes``) only has live asks,
        so this writes just the columns ``get_kalshi_close_price`` reads (ticker,
        yes_price, fetched_at) plus event context, defaulting the rest to NULL.

        Each sweep MUST pass a fresh ``session_id`` — the table's UNIQUE
        constraint is (session_id, ticker), so reusing a scan's session_id would
        be IGNORE'd and no post-entry snapshot would land. ``snapshots`` is a
        list of dicts with at least ``ticker`` and ``yes_price``; optional keys
        mirror the full-market columns.
        """
        if not snapshots:
            return 0
        fa = _fmt(fetched_at) or datetime.now(timezone.utc).isoformat()

        def _no_price(s: dict) -> object:
            # no_price is NOT NULL in the schema; derive from yes if absent so the
            # INSERT OR IGNORE doesn't silently drop the row.
            if s.get("no_price") is not None:
                return s["no_price"]
            yp = s.get("yes_price")
            return (1.0 - yp) if yp is not None else None

        rows = [
            (
                session_id,
                fa,
                sector,
                s["ticker"],
                s.get("title"),
                s.get("market_type") or "moneyline",  # NOT NULL in schema
                s.get("yes_price"),
                _no_price(s),
                s.get("volume_usd"),
                s.get("open_interest_usd"),
                s.get("team_home"),
                s.get("team_away"),
                s.get("yes_team"),
                s.get("line"),
                _fmt(s.get("event_date")),
                s.get("event_id"),
            )
            for s in snapshots
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

    def archive_orderbook_depth(
        self,
        session_id: str,
        sector: str,
        books: list[dict],
        fetched_at: object = None,
    ) -> int:
        """Persist one sweep's order-book depth metrics (see watch-listings).

        ``books`` is a list of dicts with at least ``ticker`` plus the
        ``book_depth_metrics`` keys (yes_ask, yes_ask_depth_usd, yes_bid,
        yes_bid_depth_usd, yes_book_usd, no_book_usd, source). Each sweep must
        pass a fresh ``session_id`` — UNIQUE(session_id, ticker) would IGNORE
        re-inserts within one sweep, which is the intended dedup.
        """
        if not books:
            return 0
        fa = _fmt(fetched_at) or datetime.now(timezone.utc).isoformat()
        rows = [
            (
                session_id,
                fa,
                sector,
                b["ticker"],
                b.get("yes_ask"),
                b.get("yes_ask_depth_usd"),
                b.get("yes_bid"),
                b.get("yes_bid_depth_usd"),
                b.get("yes_book_usd"),
                b.get("no_book_usd"),
                b.get("source"),
            )
            for b in books
        ]
        with _get_connection() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO archived_orderbook_depth "
                "(session_id, fetched_at, sector, ticker, yes_ask, yes_ask_depth_usd, "
                " yes_bid, yes_bid_depth_usd, yes_book_usd, no_book_usd, source) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
        return len(rows)

    def get_latest_sharp_odds(self, sector: str, event_date: str) -> list[dict]:
        """Return the most recent archived Pinnacle odds for a sector + event_date.

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

    def get_kalshi_close_price(
        self,
        ticker: str,
        event_id: str,
        minutes_before: int = 30,
        not_before: object = None,
    ) -> float | None:
        """Latest Kalshi yes_price snapshot at least ``minutes_before`` before tipoff.

        Kalshi never truly closes — binary YES/NO markets trade continuously
        until event resolution and the late-game / post-result prints are
        dominated by information shocks. The industry convention is to use a
        snapshot ~30 minutes before scheduled start as the "close" reference.

        Tipoff timestamp comes from archived_sharp_odds.event_date (Pinnacle's
        startTime, ISO with timezone). If no Pinnacle row exists for the
        event_id we return None — without a tipoff anchor we'd be guessing
        when the line was "fresh."

        ``not_before`` (a datetime or ISO string, e.g. a placed bet's
        ``placed_at``) anchors the close to be at or AFTER our entry, so CLV
        measures forward from the fill instead of against a price that preceded
        it. For a bet placed inside the ``minutes_before`` window, the upper
        bound relaxes from T-minutes_before to tipoff so a genuine post-entry,
        pre-tipoff snapshot can still be found. Returns None when no snapshot at
        or after ``not_before`` exists — a missing forward close is better than a
        fabricated backward one. When ``not_before`` is None the behaviour is
        unchanged (latest snapshot ≤ T-minutes_before).
        """
        sel = self._select_kalshi_close(ticker, event_id, minutes_before, not_before)
        return sel[0] if sel else None

    def _select_kalshi_close(
        self,
        ticker: str,
        event_id: str,
        minutes_before: int = 30,
        not_before: object = None,
    ) -> "tuple[float, datetime, datetime] | None":
        """Snapshot :meth:`get_kalshi_close_price` selects, with its timestamps.

        Returns ``(yes_price, snapshot_fetched_at, target_dt)`` where target_dt
        is the T-minutes_before close target (tipoff - minutes_before). The gap
        ``target_dt - snapshot_fetched_at`` is the close snapshot's STALENESS —
        how far the archived price we call "close" actually sits from the
        near-tip moment we wanted. Shared by :meth:`get_kalshi_close_price`
        (price) and :meth:`get_kalshi_close_staleness_h` (staleness) so both read
        the exact same snapshot. Returns None with no tipoff anchor / no snapshot.
        """
        from datetime import datetime, timedelta

        with _get_connection() as conn:
            tip_row = conn.execute(
                """SELECT event_date FROM archived_sharp_odds
                   WHERE event_id = ?
                     AND event_date IS NOT NULL
                   ORDER BY fetched_at DESC LIMIT 1""",
                (event_id,),
            ).fetchone()
            if not tip_row:
                return None
            try:
                tipoff = datetime.fromisoformat(tip_row["event_date"])
            except (ValueError, TypeError):
                return None

            upper_dt = tipoff - timedelta(minutes=minutes_before)
            target_dt = upper_dt  # T-minutes_before close target (staleness anchor)

            lower_dt: datetime | None = None
            if not_before is not None:
                lower_dt = not_before if isinstance(not_before, datetime) else None
                if lower_dt is None:
                    try:
                        lower_dt = datetime.fromisoformat(str(not_before))
                    except (ValueError, TypeError):
                        lower_dt = None
                if lower_dt is not None and lower_dt.tzinfo is None:
                    lower_dt = lower_dt.replace(tzinfo=timezone.utc)
                # Late placement: entry is already past the T-N proxy close, so
                # relax the upper bound to tipoff to find a post-entry snapshot.
                if lower_dt is not None and lower_dt > upper_dt:
                    upper_dt = tipoff

            def _latest(upper_iso: str, inclusive: bool):
                op = "<=" if inclusive else "<"
                sql = (
                    "SELECT yes_price, fetched_at FROM archived_kalshi_markets "
                    f"WHERE ticker = ? AND fetched_at {op} ?"
                )
                params: list[object] = [ticker, upper_iso]
                if lower_dt is not None:
                    sql += " AND fetched_at >= ?"
                    params.append(lower_dt.isoformat())
                sql += " ORDER BY fetched_at DESC LIMIT 1"
                return conn.execute(sql, params).fetchone()

            # Primary window: latest snapshot at/before the T-minutes_before
            # target. ~30 min pre-tip is the "close" proxy that avoids late
            # info-shock prints; a densely-snapshotted ticker (Kalshi ladders
            # listed days out) always has a snapshot here.
            row = _latest(upper_dt.isoformat(), inclusive=True)

            # Near-tip-only capture fallback. Some venues (Polymarket US) are
            # only snapshotted in the final ~30 min before tip, so EVERY
            # snapshot lands INSIDE the T-minutes_before window and the primary
            # query finds nothing — silently dropping the row's CLV. When no
            # pre-target snapshot exists, relax the upper bound to STRICTLY
            # before tipoff and take the latest pre-tip print: it is the truest
            # available close. Bounded ``< tipoff`` so in-game post-tip prints
            # (info-corrupted, trending to 0/1) are never selected. Skipped when
            # the upper bound was already relaxed to tipoff for a late fill.
            if row is None and upper_dt < tipoff:
                row = _latest(tipoff.isoformat(), inclusive=False)
        if not row:
            return None
        try:
            snap_dt = datetime.fromisoformat(row["fetched_at"])
        except (ValueError, TypeError):
            return None
        if snap_dt.tzinfo is None:
            snap_dt = snap_dt.replace(tzinfo=timezone.utc)
        return row["yes_price"], snap_dt, target_dt

    def get_kalshi_close_staleness_h(
        self,
        ticker: str,
        event_id: str,
        minutes_before: int = 30,
        not_before: object = None,
    ) -> float | None:
        """Hours the "close" snapshot sits before the T-minutes_before target.

        A large value means the archived price we scored as "close" is a stale
        mid-day read, not a genuine near-tip price — a watch-closes/launchd
        capture gap rather than a flat market. Used by the shadow CLV gate to
        optionally exclude rows whose close is untrustworthy (measurement
        quality, the sibling of the code-version ``is_contaminated`` filter).

        Computed against the SAME snapshot :meth:`get_kalshi_close_price` picks,
        so it exactly matches the CLV of unplaced (shadow) bets (``not_before``
        None). Returns None with no tipoff anchor / no snapshot (treat as
        unknown — exclude).
        """
        sel = self._select_kalshi_close(ticker, event_id, minutes_before, not_before)
        if sel is None:
            return None
        _price, snap_dt, target_dt = sel
        return (target_dt - snap_dt).total_seconds() / 3600.0

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

    def get_spread_closing_line_aligned(
        self,
        event_id: str,
        yes_team: str | None,
        line: float | None = None,
    ) -> float | None:
        """Return the latest pre-tipoff Pinnacle spread cover prob aligned to
        the bet's YES side.

        Mirrors :meth:`get_closing_line_aligned` but filters to
        ``spread_line IS NOT NULL`` so it picks up spread snapshots instead of
        moneylines. When ``line`` is provided we prefer the snapshot whose
        absolute spread matches (Pinnacle stores the favorite's line as a
        negative number; the underdog's bet line is the positive mirror).
        If no exact match is found we fall back to the nearest spread snapshot
        WITHIN ``_CLOSING_LINE_FALLBACK_TOLERANCE`` points — line drift of
        ±0.5–1.0pt happens routinely pre-tipoff and isn't a reason to drop CLV
        measurement, but a several-point-off main-line snapshot (which always
        devigs to ~0.50) must NOT masquerade as our alternate-line's close.
        When ``line`` is None we can't bound the fallback, so we keep the
        legacy best-effort latest-snapshot behavior.
        """
        from evmax.agents.cleanup.resolver import yes_aligned_close_prob

        with _get_connection() as conn:
            row = None
            if line is not None:
                row = conn.execute(
                    """SELECT outcome_a_label, outcome_b_label,
                              true_prob_a, true_prob_b, true_prob_draw
                       FROM archived_sharp_odds
                       WHERE event_id = ?
                         AND spread_line IS NOT NULL
                         AND ABS(ABS(spread_line) - ABS(?)) < 0.01
                         AND event_date IS NOT NULL
                         AND fetched_at < event_date
                       ORDER BY fetched_at DESC
                       LIMIT 1""",
                    (event_id, float(line)),
                ).fetchone()
                if row is None:
                    # Tolerance-bounded fallback: nearest line within ±tol pt.
                    row = conn.execute(
                        """SELECT outcome_a_label, outcome_b_label,
                                  true_prob_a, true_prob_b, true_prob_draw
                           FROM archived_sharp_odds
                           WHERE event_id = ?
                             AND spread_line IS NOT NULL
                             AND ABS(ABS(spread_line) - ABS(?)) <= ?
                             AND event_date IS NOT NULL
                             AND fetched_at < event_date
                           ORDER BY ABS(ABS(spread_line) - ABS(?)) ASC,
                                    fetched_at DESC
                           LIMIT 1""",
                        (
                            event_id,
                            float(line),
                            _CLOSING_LINE_FALLBACK_TOLERANCE,
                            float(line),
                        ),
                    ).fetchone()
            else:
                row = conn.execute(
                    """SELECT outcome_a_label, outcome_b_label,
                              true_prob_a, true_prob_b, true_prob_draw
                       FROM archived_sharp_odds
                       WHERE event_id = ?
                         AND spread_line IS NOT NULL
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

    def get_total_closing_line_aligned(
        self,
        event_id: str,
        yes_team: str | None,
        line: float | None = None,
    ) -> float | None:
        """Return the latest pre-tipoff Pinnacle total prob aligned to YES.

        For totals, ``yes_team`` is ``"over"`` or ``"under"`` (the scanner
        normalizes the YES side label). Matches the snapshot with the same
        total_line when possible, else falls back to the nearest pre-tipoff
        total snapshot WITHIN ``_CLOSING_LINE_FALLBACK_TOLERANCE`` points —
        an alternate total several points off the archived line must not be
        scored against an unrelated line's prob. When ``line`` is None we keep
        the legacy best-effort latest-snapshot behavior.
        """
        side = (yes_team or "").strip().lower()
        if side not in ("over", "under"):
            return None

        with _get_connection() as conn:
            row = None
            if line is not None:
                row = conn.execute(
                    """SELECT true_prob_over, true_prob_under
                       FROM archived_sharp_odds
                       WHERE event_id = ?
                         AND total_line IS NOT NULL
                         AND ABS(total_line - ?) < 0.01
                         AND event_date IS NOT NULL
                         AND fetched_at < event_date
                       ORDER BY fetched_at DESC
                       LIMIT 1""",
                    (event_id, float(line)),
                ).fetchone()
                if row is None:
                    # Tolerance-bounded fallback: nearest line within ±tol pt.
                    row = conn.execute(
                        """SELECT true_prob_over, true_prob_under
                           FROM archived_sharp_odds
                           WHERE event_id = ?
                             AND total_line IS NOT NULL
                             AND ABS(total_line - ?) <= ?
                             AND event_date IS NOT NULL
                             AND fetched_at < event_date
                           ORDER BY ABS(total_line - ?) ASC, fetched_at DESC
                           LIMIT 1""",
                        (
                            event_id,
                            float(line),
                            _CLOSING_LINE_FALLBACK_TOLERANCE,
                            float(line),
                        ),
                    ).fetchone()
            else:
                row = conn.execute(
                    """SELECT true_prob_over, true_prob_under
                       FROM archived_sharp_odds
                       WHERE event_id = ?
                         AND total_line IS NOT NULL
                         AND event_date IS NOT NULL
                         AND fetched_at < event_date
                       ORDER BY fetched_at DESC
                       LIMIT 1""",
                    (event_id,),
                ).fetchone()
        if not row:
            return None
        return row["true_prob_over"] if side == "over" else row["true_prob_under"]

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
