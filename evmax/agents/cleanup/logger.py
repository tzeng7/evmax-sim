"""PredictionLogger — persists EVGap objects to SQLite after each scan."""

from __future__ import annotations

from datetime import date
from typing import Optional

import structlog

from evmax.agents.cleanup.db import get_connection
from evmax.agents.odds.ev_gap_agent import EVGap

logger = structlog.get_logger(__name__)


def get_logged_market_ids(scan_date: Optional[date] = None) -> set[str]:
    """Return market_ids already logged for a given scan_date (default today)."""
    sd = (scan_date or date.today()).isoformat()
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT market_id FROM ev_predictions WHERE scan_date = ?", (sd,)
        ).fetchall()
    return {r["market_id"] for r in rows}


def log_gaps(
    gaps: list[EVGap],
    scan_date: Optional[date] = None,
    sharp_weight_used: float = 0.85,
    bankroll_used: Optional[float] = None,
) -> int:
    """
    Persist EVGap objects to ev_predictions.

    Skips duplicates (same market_id + scan_date).
    Returns the number of newly inserted rows.
    """
    if not gaps:
        return 0

    sd = (scan_date or date.today()).isoformat()
    inserted = 0

    with get_connection() as conn:
        for g in gaps:
            event_date_str: Optional[str] = None
            if g.event_date is not None:
                ed = g.event_date.date() if hasattr(g.event_date, "date") else g.event_date
                event_date_str = ed.isoformat()

            try:
                conn.execute(
                    """INSERT OR IGNORE INTO ev_predictions
                    (scan_date, market_id, event_id, sector, yes_team, market_type,
                     event_title, event_date, kalshi_yes_price, sharp_true_prob,
                     blended_true_prob, ev_pct, kelly_fraction, volume_usd,
                     model_sources, sharp_weight_used, bankroll_used, line)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        sd,
                        g.market_id,
                        g.event_id,
                        g.sector,
                        g.yes_team,
                        g.market_type,
                        g.event_title,
                        event_date_str,
                        g.kalshi_yes_price,
                        g.sharp_true_prob,
                        g.blended_true_prob,
                        g.ev_pct,
                        g.kelly_fraction,
                        g.volume_usd,
                        g.model_sources,
                        sharp_weight_used,
                        bankroll_used,
                        g.line,
                    ),
                )
                if conn.execute("SELECT changes()").fetchone()[0]:
                    inserted += 1
            except Exception as e:
                logger.warning("prediction_log_error", market_id=g.market_id, error=str(e))

        conn.commit()

    logger.info("predictions_logged", inserted=inserted, total=len(gaps), date=sd)
    return inserted
