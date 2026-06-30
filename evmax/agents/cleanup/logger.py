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
from evmax.models.market import is_prop_event
from evmax.provenance import code_version

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
    if is_prop_event(gap.event_id):
        return f"{gap.sector}_props"
    return gap.sector


def get_logged_market_ids(scan_date: Optional[date] = None) -> set[str]:
    """Return market_ids first-flagged on a given scan_date (default today).

    Mode-agnostic on purpose: under freeze-on-first-insert semantics the
    table holds exactly one row per market, so this returns the markets
    whose first +EV flag happened on `scan_date`, regardless of mode.
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

    Freeze-on-first-insert: the table has UNIQUE(market_id), and this function
    uses INSERT OR IGNORE, so the row written on the first scan that flags a
    market as +EV is permanent. Re-scans on later days are no-ops — the stored
    sharp/blended/ev_pct snapshot stays locked to the first-flag values.
    This makes Brier, CLV, and calibration reflect the model's actual call at
    the moment it was made, not a refreshed scan closer to close.

    Disabled categories are dropped before insert. Live and shadow rows are
    written to the same table with different values in the `mode` column.

    When `model_version` is not given it defaults to the git code provenance
    of the running checkout (`evmax.provenance.code_version()`, e.g.
    "5901aa1-dirty@feature-branch") so every row records which code state
    produced it. Pass an explicit value to override; NULL only when git is
    unavailable.

    Returns the number of newly inserted rows (sum of live + shadow;
    excludes dropped-disabled and duplicate-ignored rows).
    """
    if not gaps:
        return 0

    if model_version is None:
        model_version = code_version()
    resolver = mode_resolver or _default_mode_resolver
    sd = (scan_date or date.today()).isoformat()
    inserted = 0
    dropped_disabled = 0
    unvoided = 0
    upgraded = 0
    counts_by_mode: dict[str, int] = {"live": 0, "shadow": 0}

    with get_connection() as conn:
        for g in gaps:
            category = _gap_category_key(g)
            try:
                # The default resolver supports per-market-type downgrades
                # (shadow_market_types). Custom resolvers stay single-arg so
                # legacy callers / tests work unchanged.
                if mode_resolver is None:
                    from evmax.modes import get_mode as _get_mode
                    mode = _get_mode(category, g.market_type)
                else:
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

            # Partial-blend demotion: sectors with a REQUIRED_BLEND_MODELS
            # entry (e.g. tennis) only bet when the full model blend
            # contributed. Gaps flagged full_blend=False still get logged —
            # they're calibration data — but never as live bankroll rows.
            if mode == "live" and not getattr(g, "full_blend", True):
                mode = "shadow"
                logger.info(
                    "prediction_demoted_partial_blend",
                    market_id=g.market_id,
                    sector=g.sector,
                    model_sources=g.model_sources,
                )

            event_date_str: Optional[str] = None
            if g.event_date is not None:
                from evmax.clients.time_util import kalshi_game_day
                if hasattr(g.event_date, "tzinfo"):
                    event_date_str = kalshi_game_day(g.event_date, g.sector)
                else:
                    event_date_str = g.event_date.isoformat()

            try:
                conn.execute(
                    """INSERT OR IGNORE INTO ev_predictions
                    (scan_date, market_id, event_id, sector, yes_team, market_type,
                     event_title, event_date, kalshi_yes_price, sharp_true_prob,
                     blended_true_prob, ev_pct, kelly_fraction, volume_usd,
                     model_sources, sharp_weight_used, bankroll_used, line,
                     mode, captured_yes_price, model_version, minutes_to_tipoff)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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
                        getattr(g, "minutes_to_tipoff", None),
                    ),
                )
                if conn.execute("SELECT changes()").fetchone()[0]:
                    inserted += 1
                    counts_by_mode[mode] = counts_by_mode.get(mode, 0) + 1
                elif mode == "live":
                    # Row already existed (freeze-on-first-insert) but this
                    # fresh EVGap qualifies as a live bet. Two un-stick cases,
                    # both gated on the existing row being unplaced (placed
                    # rows were a deliberate user action — never touch them):
                    #
                    #   1. shadow → live upgrade. The market was frozen as
                    #      shadow on an earlier scan — e.g. a tennis
                    #      REQUIRED_BLEND_MODELS partial blend before
                    #      form/advanced had data, or a sector that has since
                    #      been promoted live. Now that it resolves to a live
                    #      bet we upgrade mode and REFRESH the frozen snapshot
                    #      to this call. The freeze rationale (lock the call,
                    #      no look-ahead refresh) only protects a *live* row; a
                    #      shadow row was never a bettable call, so this is the
                    #      first real one and gets locked instead.
                    #   2. voided → un-stick. Row was marked voided by cleanup
                    #      but Kalshi is quoting it again. Clear voided only;
                    #      the live snapshot stays frozen.
                    existing = conn.execute(
                        "SELECT mode, placed, voided FROM ev_predictions "
                        "WHERE market_id = ?",
                        (g.market_id,),
                    ).fetchone()
                    if existing is not None and existing["placed"] == 0:
                        if existing["mode"] == "shadow":
                            conn.execute(
                                """UPDATE ev_predictions SET
                                       mode = 'live', voided = 0,
                                       scan_date = ?, event_date = ?,
                                       kalshi_yes_price = ?, sharp_true_prob = ?,
                                       blended_true_prob = ?, ev_pct = ?,
                                       kelly_fraction = ?, volume_usd = ?,
                                       model_sources = ?, sharp_weight_used = ?,
                                       bankroll_used = ?, line = ?,
                                       captured_yes_price = ?, model_version = ?,
                                       minutes_to_tipoff = ?
                                   WHERE market_id = ? AND placed = 0""",
                                (
                                    sd,
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
                                    g.kalshi_yes_price,
                                    model_version,
                                    getattr(g, "minutes_to_tipoff", None),
                                    g.market_id,
                                ),
                            )
                            upgraded += 1
                            counts_by_mode["live"] = counts_by_mode.get("live", 0) + 1
                            logger.info(
                                "prediction_upgraded_shadow_to_live",
                                market_id=g.market_id,
                                sector=g.sector,
                                model_sources=g.model_sources,
                            )
                        elif existing["voided"] == 1:
                            conn.execute(
                                "UPDATE ev_predictions SET voided = 0 "
                                "WHERE market_id = ? AND voided = 1 "
                                "AND placed = 0 AND mode = 'live'",
                                (g.market_id,),
                            )
                            unvoided += 1
            except Exception as e:
                logger.warning("prediction_log_error", market_id=g.market_id, error=str(e))

        conn.commit()

    logger.info(
        "predictions_logged",
        inserted=inserted,
        live=counts_by_mode.get("live", 0),
        shadow=counts_by_mode.get("shadow", 0),
        dropped_disabled=dropped_disabled,
        unvoided=unvoided,
        upgraded=upgraded,
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

    `model_version` defaults to git code provenance like log_gaps; pass an
    explicit tag to override.

    Returns the number of newly inserted rows.
    """
    prop_gaps = [g for g in gaps if is_prop_event(g.event_id)]
    if not prop_gaps:
        return 0

    if model_version is None:
        model_version = code_version()
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
                from evmax.clients.time_util import kalshi_game_day
                if hasattr(g.event_date, "tzinfo"):
                    event_date_str = kalshi_game_day(g.event_date, g.sector)
                else:
                    event_date_str = g.event_date.isoformat()

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


def log_prop_from_sharp(
    pairs: list[tuple["SharpOdds", "PredictionMarket"]],
    scan_date: Optional[date] = None,
    model_version: Optional[str] = None,
    mode_resolver: Optional[Callable[[str], str]] = None,
) -> int:
    """Log prop observations directly from SharpOdds + PredictionMarket pairs.

    Companion to log_prop_observations for the path where props haven't been
    turned into EVGaps yet (still useful as a back-channel even after
    _evaluate_prop_pair was re-enabled, e.g. for the raw anchor-priced log).

    `model_version` is written to the eponymous column so calibration queries
    can partition rows by which probability source produced them (e.g.
    `pinnacle-anchor-v1` for the current model, NULL for the legacy L15).

    Honors `data/categories.yaml` mode per (sector + "_props") just like
    log_prop_observations — without this resolver, the `mode` column
    defaulted to 'live' on every insert and bypassed the shadow firewall.
    """
    if not pairs:
        return 0

    resolver = mode_resolver or _default_mode_resolver
    sd = (scan_date or date.today()).isoformat()
    inserted = 0
    counts_by_mode: dict[str, int] = {"live": 0, "shadow": 0}
    dropped_disabled = 0

    with get_connection() as conn:
        for sharp, market in pairs:
            event_date_str: Optional[str] = None
            if sharp.event_date is not None:
                from evmax.clients.time_util import kalshi_game_day
                if hasattr(sharp.event_date, "tzinfo"):
                    event_date_str = kalshi_game_day(sharp.event_date, sharp.sector)
                else:
                    event_date_str = sharp.event_date.isoformat()

            # Categorical key for mode lookup. log_prop_observations uses
            # _gap_category_key on EVGap; we don't have an EVGap here, but
            # the categorical key for props is always "{sector}_props".
            category = f"{sharp.sector}_props" if sharp.sector else ""
            try:
                mode = resolver(category)
            except Exception as e:
                logger.warning(
                    "prop_mode_resolver_failed",
                    category=category,
                    market_id=market.id,
                    error=str(e),
                )
                mode = "live"

            if mode == "disabled":
                dropped_disabled += 1
                continue

            # EV vs Kalshi price (over side)
            kalshi_price = market.yes_price
            model_prob = sharp.true_prob_over or 0.0
            ev_pct = (model_prob / kalshi_price - 1.0) if kalshi_price > 0 else 0.0

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
                        mode,
                        kalshi_price,
                        model_version,
                    ),
                )
                if conn.execute("SELECT changes()").fetchone()[0]:
                    inserted += 1
                    counts_by_mode[mode] = counts_by_mode.get(mode, 0) + 1
            except Exception as e:
                logger.warning("prop_log_error", market_id=market.id, error=str(e))

        conn.commit()

    logger.info(
        "props_logged_from_sharp",
        inserted=inserted,
        total=len(pairs),
        date=sd,
        model_version=model_version,
        live=counts_by_mode.get("live", 0),
        shadow=counts_by_mode.get("shadow", 0),
        dropped_disabled=dropped_disabled,
    )
    return inserted
