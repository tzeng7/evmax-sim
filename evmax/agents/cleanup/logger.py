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

    Freeze-on-first-insert: the table has UNIQUE(market_id), and this function
    uses INSERT OR IGNORE, so the row written on the first scan that flags a
    market as +EV is permanent. Re-scans on later days are no-ops — the stored
    sharp/blended/ev_pct snapshot stays locked to the first-flag values.
    This makes Brier, CLV, and calibration reflect the model's actual call at
    the moment it was made, not a refreshed scan closer to close.

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


def log_prop_observations(
    gaps: list[EVGap],
    scan_date: Optional[date] = None,
) -> int:
    """
    Persist ALL prop EVGap objects to prop_observations for model training.

    Logs every prop line seen — not just +EV ones — so we have the full
    distribution of lines and outcomes for calibration. Skips duplicates.
    Returns the number of newly inserted rows.
    """
    prop_gaps = [g for g in gaps if "::prop::" in g.event_id]
    if not prop_gaps:
        return 0

    sd = (scan_date or date.today()).isoformat()
    inserted = 0

    with get_connection() as conn:
        for g in prop_gaps:
            event_date_str: Optional[str] = None
            if g.event_date is not None:
                ed = g.event_date.date() if hasattr(g.event_date, "date") else g.event_date
                event_date_str = ed.isoformat()

            try:
                conn.execute(
                    """INSERT OR IGNORE INTO prop_observations
                    (scan_date, event_date, sector, player_name, stat_type, line,
                     kalshi_price, sharp_prob, ev_pct, l15_games,
                     market_id, event_id, event_title)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        sd,
                        event_date_str,
                        g.sector,
                        g.prop_player_name or g.yes_team,
                        g.prop_stat_type or "",
                        g.prop_threshold or g.line,
                        g.kalshi_yes_price,
                        g.sharp_true_prob,
                        g.ev_pct,
                        g.prop_l15_games or None,
                        g.market_id,
                        g.event_id,
                        g.event_title,
                    ),
                )
                if conn.execute("SELECT changes()").fetchone()[0]:
                    inserted += 1
            except Exception as e:
                logger.warning("prop_log_error", market_id=g.market_id, error=str(e))

        conn.commit()

    logger.info("props_logged", inserted=inserted, total=len(prop_gaps), date=sd)
    return inserted


def log_prop_from_sharp(
    pairs: list[tuple["SharpOdds", "PredictionMarket"]],
    scan_date: Optional[date] = None,
) -> int:
    """Log prop observations directly from SharpOdds + PredictionMarket pairs.

    Used when _evaluate_prop is disabled — props aren't turned into EVGaps but
    we still want calibration data in prop_observations.
    """
    if not pairs:
        return 0

    sd = (scan_date or date.today()).isoformat()
    inserted = 0

    with get_connection() as conn:
        for sharp, market in pairs:
            event_date_str: Optional[str] = None
            if sharp.event_date is not None:
                ed = sharp.event_date.date() if hasattr(sharp.event_date, "date") else sharp.event_date
                event_date_str = ed.isoformat()

            # EV vs Kalshi price (over side)
            kalshi_price = market.yes_price
            model_prob = sharp.true_prob_over or 0.0
            ev_pct = (model_prob / kalshi_price - 1.0) if kalshi_price > 0 else 0.0

            try:
                conn.execute(
                    """INSERT OR IGNORE INTO prop_observations
                    (scan_date, event_date, sector, player_name, stat_type, line,
                     kalshi_price, sharp_prob, ev_pct, l15_games,
                     market_id, event_id, event_title)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        sd,
                        event_date_str,
                        sharp.sector,
                        sharp.prop_player_name or "",
                        sharp.prop_stat_type or "",
                        sharp.total_line,
                        kalshi_price,
                        model_prob,
                        ev_pct,
                        sharp.prop_l15_games or None,
                        market.id,
                        sharp.event_id,
                        market.title,
                    ),
                )
                if conn.execute("SELECT changes()").fetchone()[0]:
                    inserted += 1
            except Exception as e:
                logger.warning("prop_log_error", market_id=market.id, error=str(e))

        conn.commit()

    logger.info("props_logged_from_sharp", inserted=inserted, total=len(pairs), date=sd)
    return inserted
