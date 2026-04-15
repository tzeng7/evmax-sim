"""PredictionLogger — persists EVGap objects to SQLite after each scan.

Honors the ARCH-11 category mode config:
  - `live`     → row inserted with mode='live'; `evmax cleanup show` etc.
                 see it and it counts toward bankroll exposure
  - `shadow`   → row inserted with mode='shadow'; visible only to the
                 `evmax cleanup shadow *` commands and the MODEL-9
                 validation harness. Does NOT count toward bankroll
  - `disabled` → row is dropped before insert. The scan still ran and
                 the gap is still visible in the in-memory CLI output
                 for this session, but nothing is persisted.

`mode_resolver` defaults to ``evmax.modes.get_mode`` which reads the
YAML base + env var + runtime overrides in the right precedence. Pass
a custom callable in tests or in code paths that want to force a mode.
"""

from __future__ import annotations

from datetime import date
from typing import Callable, Optional

import structlog

from evmax.agents.cleanup.db import get_connection
from evmax.agents.odds.ev_gap_agent import EVGap

logger = structlog.get_logger(__name__)


def _default_mode_resolver(category: str) -> str:
    """Lazy wrapper around evmax.modes.get_mode so tests can monkeypatch.

    We import inside the function because evmax.modes triggers category
    registry validation at import time, and we don't want `log_gaps` to
    crash in test environments that monkeypatch the registry YAML.
    """
    from evmax.modes import get_mode

    return get_mode(category)


def _gap_category_key(gap: EVGap) -> str:
    """Resolve the category registry key for a gap.

    Game markets use the sector directly (`nba`, `nfl`, `soccer`, …);
    player props use the `{sector}_props` variant that matches the
    Kalshi suffix convention in SECTOR_SERIES_MAP and data/categories.yaml.
    """
    if gap.event_id and "::prop::" in gap.event_id:
        return f"{gap.sector}_props"
    return gap.sector


def get_logged_market_ids(scan_date: Optional[date] = None) -> set[str]:
    """Return market_ids already logged for a given scan_date (default today).

    Mode-agnostic on purpose: the UNIQUE(market_id, scan_date) constraint
    allows at most one row per (market, scan) across all modes, so the
    dedupe check should see rows regardless of which mode they were
    logged under.
    """
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
    mode_resolver: Optional[Callable[[str], str]] = None,
    model_version: Optional[str] = None,
) -> int:
    """
    Persist EVGap objects to ev_predictions.

    Skips duplicates (same market_id + scan_date). Disabled categories
    are dropped before insert. Live and shadow rows are written to the
    same table with different values in the `mode` column.

    Returns the number of newly inserted rows (sum of live + shadow;
    excludes dropped-disabled and duplicate-ignored rows).
    """
    if not gaps:
        return 0

    resolver = mode_resolver or _default_mode_resolver
    sd = (scan_date or date.today()).isoformat()
    inserted = 0
    dropped_disabled = 0
    counts_by_mode: dict[str, int] = {"live": 0, "shadow": 0}

    with get_connection() as conn:
        for g in gaps:
            category = _gap_category_key(g)
            try:
                mode = resolver(category)
            except Exception as e:
                # Unknown category / broken registry → default to 'live'
                # to preserve pre-ARCH-11 behavior. Log loudly so the
                # drift surfaces in scan output.
                logger.warning(
                    "prediction_mode_resolver_failed",
                    category=category,
                    market_id=g.market_id,
                    error=str(e),
                )
                mode = "live"

            if mode == "disabled":
                dropped_disabled += 1
                continue

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
                     model_sources, sharp_weight_used, bankroll_used, line,
                     mode, captured_yes_price, model_version)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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
                        mode,
                        g.kalshi_yes_price,  # captured_yes_price == pre-game YES ask
                        model_version,
                    ),
                )
                if conn.execute("SELECT changes()").fetchone()[0]:
                    inserted += 1
                    counts_by_mode[mode] = counts_by_mode.get(mode, 0) + 1
            except Exception as e:
                logger.warning("prediction_log_error", market_id=g.market_id, error=str(e))

        conn.commit()

    logger.info(
        "predictions_logged",
        inserted=inserted,
        live=counts_by_mode.get("live", 0),
        shadow=counts_by_mode.get("shadow", 0),
        dropped_disabled=dropped_disabled,
        total=len(gaps),
        date=sd,
    )
    return inserted


def log_prop_observations(
    gaps: list[EVGap],
    scan_date: Optional[date] = None,
    mode_resolver: Optional[Callable[[str], str]] = None,
    model_version: Optional[str] = None,
) -> int:
    """
    Persist ALL prop EVGap objects to prop_observations for model training.

    Logs every prop line seen — not just +EV ones — so we have the full
    distribution of lines and outcomes for calibration. Skips duplicates.
    Honors the same category mode rules as log_gaps: disabled categories
    are dropped, live/shadow use the `mode` column on the table.

    Returns the number of newly inserted rows.
    """
    prop_gaps = [g for g in gaps if g.event_id and "::prop::" in g.event_id]
    if not prop_gaps:
        return 0

    resolver = mode_resolver or _default_mode_resolver
    sd = (scan_date or date.today()).isoformat()
    inserted = 0
    dropped_disabled = 0
    counts_by_mode: dict[str, int] = {"live": 0, "shadow": 0}

    with get_connection() as conn:
        for g in prop_gaps:
            category = _gap_category_key(g)
            try:
                mode = resolver(category)
            except Exception as e:
                logger.warning(
                    "prop_mode_resolver_failed",
                    category=category,
                    market_id=g.market_id,
                    error=str(e),
                )
                mode = "live"

            if mode == "disabled":
                dropped_disabled += 1
                continue

            event_date_str: Optional[str] = None
            if g.event_date is not None:
                ed = g.event_date.date() if hasattr(g.event_date, "date") else g.event_date
                event_date_str = ed.isoformat()

            try:
                conn.execute(
                    """INSERT OR IGNORE INTO prop_observations
                    (scan_date, event_date, sector, player_name, stat_type, line,
                     kalshi_price, sharp_prob, ev_pct, l15_games,
                     market_id, event_id, event_title,
                     mode, captured_yes_price, model_version)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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
                        mode,
                        g.kalshi_yes_price,
                        model_version,
                    ),
                )
                if conn.execute("SELECT changes()").fetchone()[0]:
                    inserted += 1
                    counts_by_mode[mode] = counts_by_mode.get(mode, 0) + 1
            except Exception as e:
                logger.warning("prop_log_error", market_id=g.market_id, error=str(e))

        conn.commit()

    logger.info(
        "props_logged",
        inserted=inserted,
        live=counts_by_mode.get("live", 0),
        shadow=counts_by_mode.get("shadow", 0),
        dropped_disabled=dropped_disabled,
        total=len(prop_gaps),
        date=sd,
    )
    return inserted
