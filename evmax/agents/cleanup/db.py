"""SQLite database for prediction logging and outcome tracking.

Schema:
  ev_predictions    — one row per +EV gap found per scan per market
  ev_outcomes       — one row per resolved market (outcome = 1/0)
  prop_observations — one row per prop line seen per scan (all props, not just +EV)
                      used for training the player prop model
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[3] / "data" / "predictions.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS ev_predictions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at           TEXT    NOT NULL DEFAULT (datetime('now')),
    scan_date           TEXT    NOT NULL,           -- YYYY-MM-DD of the FIRST scan that flagged this market as +EV
    market_id           TEXT    NOT NULL,
    event_id            TEXT    NOT NULL,
    sector              TEXT    NOT NULL,
    yes_team            TEXT    NOT NULL,
    market_type         TEXT    NOT NULL,
    event_title         TEXT,
    event_date          TEXT,                       -- YYYY-MM-DD of the game
    kalshi_yes_price    REAL    NOT NULL,           -- Kalshi implied prob (e.g. 0.42) — frozen at first flag
    sharp_true_prob     REAL    NOT NULL,           -- devigged Pinnacle prob — frozen at first flag
    blended_true_prob   REAL    NOT NULL,           -- final blended prob — frozen at first flag
    ev_pct              REAL    NOT NULL,           -- EV fraction (e.g. 0.07 for 7%) — frozen at first flag
    kelly_fraction      REAL    NOT NULL,
    volume_usd          REAL,
    model_sources       TEXT,
    sharp_weight_used   REAL,                       -- sharp_weight at first-flag time
    bankroll_used       REAL,                       -- bankroll passed to scan (for stake calc)
    line                REAL,                       -- spread/total line (e.g. -8.5)
    voided              INTEGER NOT NULL DEFAULT 0, -- 1=voided (game cancelled/untraceable)
    placed              INTEGER NOT NULL DEFAULT 0, -- 1=user manually placed this bet
    placed_at           TEXT,                       -- ISO timestamp when bet was placed
    placed_price        REAL,                       -- Kalshi ask at time of placement
    placed_stake        REAL,                       -- dollar amount staked
    UNIQUE(market_id)
);

CREATE TABLE IF NOT EXISTS prop_observations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at       TEXT NOT NULL DEFAULT (datetime('now')),
    scan_date       TEXT NOT NULL,          -- YYYY-MM-DD when scan ran
    event_date      TEXT,                   -- YYYY-MM-DD of the game
    sector          TEXT NOT NULL,          -- "nba", "nfl"
    player_name     TEXT NOT NULL,
    stat_type       TEXT NOT NULL,          -- "points", "rebounds", "assists", "threes", "steals", "blocks", "pra"
    line            REAL NOT NULL,          -- threshold (e.g. 24.5)
    kalshi_price    REAL,                   -- Kalshi implied probability (0-1)
    sharp_prob      REAL,                   -- sharp/model estimated probability (0-1)
    ev_pct          REAL,                   -- EV at scan time (may be negative — we log all lines)
    l15_games       INTEGER,                -- sample size used for sharp_prob
    market_id       TEXT,
    event_id        TEXT,
    event_title     TEXT,                   -- "LeBron James — LAL vs GSW"
    actual_value    REAL,                   -- filled in after game (e.g. 28.0 for 28 pts)
    outcome         INTEGER,                -- 1=over hit, 0=under, NULL=unresolved
    resolved_at     TEXT,
    UNIQUE(market_id, scan_date)
);

CREATE TABLE IF NOT EXISTS ev_outcomes (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id           TEXT    NOT NULL,
    event_id            TEXT    NOT NULL,
    event_date          TEXT    NOT NULL,
    sector              TEXT    NOT NULL,
    yes_team            TEXT    NOT NULL,
    outcome             INTEGER,                    -- 1=YES won, 0=NO won, NULL=pending
    sharp_true_prob     REAL,
    blended_true_prob   REAL,
    pinnacle_close_prob REAL,                       -- Pinnacle true prob captured at close
    resolved_at         TEXT,
    result_source       TEXT,                       -- "espn", "bo3gg", "manual"
    UNIQUE(market_id)
);
"""


def _migrate_unique_market_id(conn: sqlite3.Connection) -> None:
    """Rebuild ev_predictions with UNIQUE(market_id) if the legacy composite
    UNIQUE(market_id, scan_date) is still in place.

    Freeze-on-first-insert semantics: each market gets exactly one row, whose
    snapshot fields are set when the pipeline first flags it as +EV. Re-scans
    on later days become no-ops via INSERT OR IGNORE. This makes `scan_date`
    the "first flagged" date, makes CLV `(entry − close)` meaningful, and
    eliminates the multi-scan Brier double-counting class of bugs.

    Idempotent: re-running is a no-op once the new constraint is present.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='ev_predictions'"
    ).fetchone()
    if not row or not row[0]:
        return
    ddl_normalized = " ".join(row[0].split())
    if "UNIQUE(market_id, scan_date)" not in ddl_normalized \
       and "UNIQUE (market_id, scan_date)" not in ddl_normalized:
        return  # already migrated

    # Safety net: dedupe any remaining rows before the constraint change.
    # Keep the row with the strongest claim per market_id:
    #   1. placed = 1 beats placed = 0
    #   2. then oldest scan_date (freeze-on-first-insert semantics)
    #   3. then lowest id (deterministic tiebreak)
    conn.execute(
        """DELETE FROM ev_predictions
           WHERE id NOT IN (
               SELECT id FROM ev_predictions p
               WHERE id = (
                   SELECT id FROM ev_predictions p2
                   WHERE p2.market_id = p.market_id
                   ORDER BY p2.placed DESC, p2.scan_date ASC, p2.id ASC
                   LIMIT 1
               )
           )"""
    )

    conn.executescript("""
        BEGIN;
        ALTER TABLE ev_predictions RENAME TO ev_predictions_old;
        CREATE TABLE ev_predictions (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            logged_at           TEXT    NOT NULL DEFAULT (datetime('now')),
            scan_date           TEXT    NOT NULL,
            market_id           TEXT    NOT NULL,
            event_id            TEXT    NOT NULL,
            sector              TEXT    NOT NULL,
            yes_team            TEXT    NOT NULL,
            market_type         TEXT    NOT NULL,
            event_title         TEXT,
            event_date          TEXT,
            kalshi_yes_price    REAL    NOT NULL,
            sharp_true_prob     REAL    NOT NULL,
            blended_true_prob   REAL    NOT NULL,
            ev_pct              REAL    NOT NULL,
            kelly_fraction      REAL    NOT NULL,
            volume_usd          REAL,
            model_sources       TEXT,
            sharp_weight_used   REAL,
            bankroll_used       REAL,
            line                REAL,
            voided              INTEGER NOT NULL DEFAULT 0,
            placed              INTEGER NOT NULL DEFAULT 0,
            placed_at           TEXT,
            placed_price        REAL,
            placed_stake        REAL,
            clv_pct             REAL,
            UNIQUE(market_id)
        );
        INSERT INTO ev_predictions (
            id, logged_at, scan_date, market_id, event_id, sector, yes_team,
            market_type, event_title, event_date, kalshi_yes_price,
            sharp_true_prob, blended_true_prob, ev_pct, kelly_fraction,
            volume_usd, model_sources, sharp_weight_used, bankroll_used, line,
            voided, placed, placed_at, placed_price, placed_stake, clv_pct
        )
        SELECT
            id, logged_at, scan_date, market_id, event_id, sector, yes_team,
            market_type, event_title, event_date, kalshi_yes_price,
            sharp_true_prob, blended_true_prob, ev_pct, kelly_fraction,
            volume_usd, model_sources, sharp_weight_used, bankroll_used, line,
            voided, placed, placed_at, placed_price, placed_stake, clv_pct
        FROM ev_predictions_old;
        DROP TABLE ev_predictions_old;
        COMMIT;
    """)


def get_connection() -> sqlite3.Connection:
    """Open (and initialize) the predictions database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    # Migrate: add voided column to existing databases that predate it
    for migration in [
        "ALTER TABLE ev_predictions ADD COLUMN voided INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE ev_outcomes ADD COLUMN pinnacle_close_prob REAL",
        "ALTER TABLE ev_predictions ADD COLUMN placed INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE ev_predictions ADD COLUMN placed_at TEXT",
        "ALTER TABLE ev_predictions ADD COLUMN placed_price REAL",
        "ALTER TABLE ev_predictions ADD COLUMN placed_stake REAL",
        "ALTER TABLE ev_predictions ADD COLUMN bankroll_used REAL",
        "ALTER TABLE ev_predictions ADD COLUMN clv_pct REAL",
    ]:
        try:
            conn.execute(migration)
            conn.commit()
        except Exception:
            pass  # column already exists
    # Table-rebuild migration runs AFTER column-add ALTERs so the new schema
    # and the old table have matching column sets for the INSERT ... SELECT.
    _migrate_unique_market_id(conn)
    return conn
