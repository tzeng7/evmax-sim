"""SQLite database for prediction logging and outcome tracking.

Schema:
  ev_predictions — one row per +EV gap found per scan per market
  ev_outcomes    — one row per resolved market (outcome = 1/0)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[3] / "data" / "predictions.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS ev_predictions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at           TEXT    NOT NULL DEFAULT (datetime('now')),
    scan_date           TEXT    NOT NULL,           -- YYYY-MM-DD when scan ran
    market_id           TEXT    NOT NULL,
    event_id            TEXT    NOT NULL,
    sector              TEXT    NOT NULL,
    yes_team            TEXT    NOT NULL,
    market_type         TEXT    NOT NULL,
    event_title         TEXT,
    event_date          TEXT,                       -- YYYY-MM-DD of the game
    kalshi_yes_price    REAL    NOT NULL,           -- Kalshi implied prob (e.g. 0.42)
    sharp_true_prob     REAL    NOT NULL,           -- devigged Pinnacle prob
    blended_true_prob   REAL    NOT NULL,           -- final blended prob
    ev_pct              REAL    NOT NULL,           -- EV fraction (e.g. 0.07 for 7%)
    kelly_fraction      REAL    NOT NULL,
    volume_usd          REAL,
    model_sources       TEXT,
    sharp_weight_used   REAL,                       -- sharp_weight at scan time
    line                REAL,                       -- spread/total line (e.g. -8.5)
    voided              INTEGER NOT NULL DEFAULT 0, -- 1=voided (game cancelled/untraceable)
    placed              INTEGER NOT NULL DEFAULT 0, -- 1=user manually placed this bet
    placed_at           TEXT,                       -- ISO timestamp when bet was placed
    placed_price        REAL,                       -- Kalshi ask at time of placement
    placed_stake        REAL,                       -- dollar amount staked
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
    ]:
        try:
            conn.execute(migration)
            conn.commit()
        except Exception:
            pass  # column already exists
    return conn
