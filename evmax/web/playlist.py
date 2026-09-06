"""The dashboard's play-list view, factored out of the FastAPI app.

This module is the SINGLE definition of "what the dashboard shows after a
scan": how one EVGap becomes a display row (:func:`gap_to_dict`), which gaps
make the actionable list (:func:`dashboard_play_dicts` — full-blend plays,
best-execution collapsed, venue-cash capped) and the post-filters the scan
view applies before rendering (:func:`filter_scan_view` — date window, no
esports map handicaps, no player props, nothing already placed).

``evmax.web.app`` (the React dashboard's ``/api/scan``) and
``evmax.discord_bot`` (the Discord scan feed + ``/scan`` slash command) both
build their tables from these functions, so the two surfaces cannot drift:
a Discord post is, row for row and in the same order, what the dashboard's
Scan Results panel would show for the same cycle.

Kept free of FastAPI imports so the notifier can call it from a CLI or
scheduled scan process without loading the web app.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Iterable, Optional

from evmax.formatting import format_outcome_label_for_row


def gap_to_dict(g, bankroll: float) -> dict[str, Any]:
    """Shape a single EVGap into the canonical dict used by the dashboard scan
    view, the portfolio fan-out and the Discord feed. Carries both
    ``kalshi_yes_price`` / ``blended_true_prob`` (portfolio names) and
    ``kalshi_price`` / ``true_prob`` (display names) so downstream consumers can
    read either."""
    from evmax.agents.cleanup.logger import _gap_category_key
    from evmax.modes import get_mode

    label_row = {
        "market_type": g.market_type or "",
        "yes_team": g.yes_team or "",
        "line": g.line,
        "prop_player_name": g.prop_player_name,
        "prop_stat_type": g.prop_stat_type,
        "prop_threshold": g.prop_threshold,
    }
    try:
        gap_mode = get_mode(_gap_category_key(g), g.market_type)
    except Exception:
        gap_mode = "live"
    gap_venue = getattr(g, "venue", "kalshi") or "kalshi"
    # Mirror the venue shadow firewall so the dashboard's mode badge matches
    # what log_gaps will persist (see prediction_demoted_shadow_venue).
    if gap_mode == "live" and gap_venue == "polymarket_us":
        from evmax.settings import get_settings
        if not get_settings().polymarket_us_sector_live(getattr(g, "sector", None)):
            gap_mode = "shadow"
    # League shadow list — mirror prediction_demoted_shadow_league.
    if gap_mode == "live":
        from evmax.sectors.soccer_tiers import league_is_live
        if not league_is_live(getattr(g, "league", None)):
            gap_mode = "shadow"
    # Maker-only gaps clear the floor only as a resting limit order — not
    # crossable at the ask, so they are never a live taker pick. Mirror the
    # shadow demotion log_gaps applies, so the dashboard badge and the (disabled)
    # pick checkbox match how the row will actually persist.
    if gap_mode == "live" and getattr(g, "maker_only", False):
        gap_mode = "shadow"
    line_val = (
        None if g.line is None
        else float(g.line) if isinstance(g.line, (int, float))
        else str(g.line)
    )
    return {
        "event_title": g.event_title or "",
        "event_id": g.event_id or "",
        "yes_team": g.yes_team or "",
        "market_type": g.market_type or "",
        "display_label": format_outcome_label_for_row(label_row),
        "line": line_val,
        "sector": g.sector or "",
        "kalshi_price": round(g.kalshi_yes_price, 2),
        "kalshi_yes_price": round(g.kalshi_yes_price, 4),
        "true_prob": round(g.blended_true_prob, 3),
        "blended_true_prob": round(g.blended_true_prob, 4),
        "sharp_true_prob": round(getattr(g, "sharp_true_prob", 0) or 0, 4),
        "ev_pct_raw": round(g.ev_pct, 4),
        "ev_pct": round(g.ev_pct * 100, 2),
        # Maker execution: EV if opened as a resting limit order (maker fee),
        # whether it clears ONLY as a maker, and the max price to rest the buy at.
        "maker_ev_pct": (
            round(g.maker_ev_pct * 100, 2)
            if getattr(g, "maker_ev_pct", None) is not None else None
        ),
        "maker_only": bool(getattr(g, "maker_only", False)),
        "maker_limit_price": (
            round(g.maker_limit_price, 4)
            if getattr(g, "maker_limit_price", None) is not None else None
        ),
        # Actionable maker rest price (the bid to set), its EV if filled there,
        # and a Kelly-sized stake fraction at that fill. See suggested_maker_bid.
        "maker_bid_price": (
            round(g.maker_bid_price, 4)
            if getattr(g, "maker_bid_price", None) is not None else None
        ),
        "maker_bid_ev_pct": (
            round(g.maker_bid_ev_pct * 100, 2)
            if getattr(g, "maker_bid_ev_pct", None) is not None else None
        ),
        "maker_bid_kelly_fraction": (
            round(g.maker_bid_kelly_fraction, 4)
            if getattr(g, "maker_bid_kelly_fraction", None) is not None else None
        ),
        "kelly_pct": round(g.kelly_fraction * 100, 2),
        "kelly_fraction": round(g.kelly_fraction, 4),
        "stake": round(bankroll * g.kelly_fraction, 2),
        "model_sources": g.model_sources or "",
        "market_id": g.market_id or "",
        "event_date": str(g.event_date.astimezone().strftime("%Y-%m-%d") if g.event_date else ""),
        "volume": g.volume_usd or 0,
        "volume_usd": g.volume_usd or 0,
        "mode": gap_mode,
        "venue": gap_venue,
        # Best-execution alternative (GAP 3): when the same bet is also +EV on
        # the OTHER venue, the display collapses to this (better) row and carries
        # the alternative's price/EV so the user can still line-shop. None when
        # this bet is quoted on only one venue.
        "alt_venue": getattr(g, "alt_venue", None),
        "alt_venue_price": (
            round(g.alt_venue_price, 2)
            if getattr(g, "alt_venue_price", None) is not None else None
        ),
        "alt_venue_ev_pct": (
            round(g.alt_venue_ev_pct * 100, 2)
            if getattr(g, "alt_venue_ev_pct", None) is not None else None
        ),
        # Full venue option set for the display dropdown (best-execution winner
        # first). Each entry is the same dict shape as this row, so the frontend
        # can swap the row's price/EV/stake/market_id to the chosen venue. None
        # when the bet is quoted on a single venue. Nested legs carry
        # venue_options=None, so this serialization never recurses.
        "venue_options": (
            [gap_to_dict(leg, bankroll) for leg in g.venue_options]
            if getattr(g, "venue_options", None) else None
        ),
    }


def dashboard_play_dicts(
    cycle,
    bankroll: float,
    cash_by_venue: Optional[dict[str, float]] = None,
) -> list[dict[str, Any]]:
    """The dashboard's actionable play list for one coordinator cycle.

    ``cycle.plays()`` drops partial-blend (shadow, $0.00-stake) gaps; the
    dashboard deliberately does NOT apply the CLI's min_prob/tiered-EV floor
    (it shows all ≥2% gaps). The SAME bet quoted on both venues collapses to
    one best-execution row (GAP 3), then each venue's summed stakes are scaled
    to its deployable cash when that is known (GAP 2; no-op for a manual
    bankroll). Order is EV-descending — ``cycle.plays()`` sorts and the
    collapse keeps first-appearance order — and that order is what every
    surface renders. View-layer only: nothing here persists.
    """
    from evmax.ev.best_execution import apply_venue_cash_cap, collapse_best_execution

    collapsed = collapse_best_execution(list(cycle.plays(require_full_blend=True)))
    if cash_by_venue:
        collapsed = apply_venue_cash_cap(collapsed, bankroll, cash_by_venue)
    return [gap_to_dict(g, bankroll) for g in collapsed]


def default_scan_window() -> tuple[str, str]:
    """The scan view's default date window: today and tomorrow (ISO dates)."""
    today = date.today()
    return today.isoformat(), (today + timedelta(days=1)).isoformat()


def placed_market_ids() -> set[str]:
    """Market ids already placed via pick — excluded from the scan view."""
    from evmax.agents.cleanup.db import get_connection

    with get_connection() as conn:
        return {
            r[0] for r in conn.execute(
                "SELECT DISTINCT market_id FROM ev_predictions "
                "WHERE placed = 1 AND voided = 0 AND mode = 'live'"
            ).fetchall()
        }


def filter_scan_view(
    gaps: Iterable[dict[str, Any]],
    date_from: str = "",
    date_to: str = "",
    placed_mids: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    """The scan-view post-filters, exactly as ``/api/scan`` applies them.

    1. Date window on the row's ``event_date``: ``[date_from, date_to]`` when
       both are given, one-sided when only one is, else today + tomorrow.
    2. Drop esports map handicaps (LoL/CS2 set handicaps not on Kalshi).
    3. Hide player props — anchor pricing produces hundreds of prop gaps per
       cycle; they still flow into prop_observations via the cycle.
    4. Drop markets already placed (``placed_mids``; ``None`` = look them up).
    Input order is preserved.
    """
    rows = list(gaps)
    if date_from and date_to:
        rows = [g for g in rows if date_from <= g["event_date"] <= date_to]
    elif date_from:
        rows = [g for g in rows if g["event_date"] >= date_from]
    elif date_to:
        rows = [g for g in rows if g["event_date"] <= date_to]
    else:
        window = default_scan_window()
        rows = [g for g in rows if g["event_date"] in window]

    rows = [g for g in rows if g["market_type"] != "map_handicap"]
    rows = [g for g in rows if g["market_type"] != "player_prop"]

    if placed_mids is None:
        placed_mids = placed_market_ids()
    return [g for g in rows if g["market_id"] not in placed_mids]
