"""FastAPI dashboard for evmax — tables, profit chart, and action buttons."""
from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from evmax.formatting import format_outcome_label_for_row

ROOT = Path(__file__).resolve().parent.parent.parent
PREDICTIONS_DB = ROOT / "data" / "predictions.db"
SPA_DIST = ROOT / "frontend" / "dist"

app = FastAPI(title="evmax dashboard")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"], allow_methods=["*"], allow_headers=["*"])


# Outcome-label rendering moved to evmax.formatting (single source). Thin
# wrapper kept so existing call sites and tests importing it from here work.
_display_label_for_row = format_outcome_label_for_row


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _conn() -> sqlite3.Connection:
    """Open the predictions DB, running any pending migrations.

    Routes through `evmax.agents.cleanup.db.get_connection()` instead of
    calling `sqlite3.connect()` directly so that ALTER TABLE migrations
    (e.g. the ARCH-11 `mode` / `captured_yes_price` / `model_version`
    columns) run on every process that opens the DB — not just the CLI.
    """
    from evmax.agents.cleanup.db import get_connection

    return get_connection()


def _settled_bets() -> list[dict[str, Any]]:
    with _conn() as conn:
        rows = conn.execute(
            """
            SELECT p.id, p.scan_date, p.event_date, p.sector, p.event_title,
                   p.yes_team, p.market_type, p.kalshi_yes_price,
                   p.blended_true_prob, p.sharp_true_prob, p.ev_pct,
                   p.kelly_fraction, p.bankroll_used, p.model_sources,
                   p.placed, p.placed_stake, p.placed_price, p.line, p.venue,
                   p.maker_fill, o.outcome
            FROM ev_predictions p
            JOIN ev_outcomes o ON p.market_id = o.market_id
            WHERE p.voided = 0 AND o.outcome IS NOT NULL
              AND p.market_type != 'map_handicap'
              AND p.mode = 'live'
            ORDER BY p.event_date ASC, p.id ASC
            """
        ).fetchall()
    out = [dict(r) for r in rows]
    for b in out:
        b["display_label"] = _display_label_for_row(b)
    return out


def _placed_bets() -> list[dict[str, Any]]:
    """Return placed but unresolved bets — always shown regardless of newer scans."""
    with _conn() as conn:
        rows = conn.execute(
            """
            SELECT p.id, p.scan_date, p.event_date, p.sector, p.event_title,
                   p.yes_team, p.market_type, p.kalshi_yes_price,
                   p.blended_true_prob, p.ev_pct, p.kelly_fraction,
                   p.bankroll_used, p.volume_usd, p.model_sources,
                   p.placed, p.placed_stake, p.placed_price, p.market_id, p.line,
                   p.venue, p.maker_fill
            FROM ev_predictions p
            LEFT JOIN ev_outcomes o ON p.market_id = o.market_id
            WHERE p.placed = 1 AND p.voided = 0 AND (o.outcome IS NULL)
              AND p.market_type != 'map_handicap'
              AND p.mode = 'live'
            ORDER BY p.event_date DESC, p.ev_pct DESC
            """
        ).fetchall()
    out = [dict(r) for r in rows]
    for b in out:
        b["display_label"] = _display_label_for_row(b)
    return out


# Started-but-unresolved open positions linger in the panel this many days
# past their event_date before being treated as stale and dropped. Covers the
# game-start → next-resolve-run gap (resolve is daily) plus weekend lag, so a
# scanned-but-unplaced position stays visible through resolution instead of
# silently vanishing at game time. Bounded so permanently-unresolved rows
# (resolution failures, dead markets) can't accumulate into a backlog.
OPEN_STARTED_LOOKBACK_DAYS = 7


def _open_bets() -> list[dict[str, Any]]:
    """Return open (unresolved, unplaced) bets, deduplicated by market_id.

    Surfaces a forward window (games on/after today) PLUS started-but-unresolved
    positions within OPEN_STARTED_LOOKBACK_DAYS. Each row is tagged with a
    ``status``: ``upcoming`` (event today/future, or rare dateless) vs
    ``in_progress`` (game has started but no outcome yet). Previously the panel
    hard-cut at ``event_date >= today``, so a scanned-but-unplaced position
    dropped out at game time and only reappeared once resolved (in Recent
    Settled) — making it look like it had never been an open position even
    though the row was in the DB the whole time. The lookback keeps it visible
    through that gap while still bounding stale, never-resolving rows. The
    window is independent of the scan date range — Open Positions reads the
    persisted DB, not the in-memory scan.
    """
    today = date.today()
    today_str = today.isoformat()
    lookback_str = (today - timedelta(days=OPEN_STARTED_LOOKBACK_DAYS)).isoformat()
    with _conn() as conn:
        rows = conn.execute(
            """
            SELECT p.id, p.scan_date, p.event_date, p.sector, p.event_title,
                   p.yes_team, p.market_type, p.kalshi_yes_price,
                   p.blended_true_prob, p.ev_pct, p.kelly_fraction,
                   p.bankroll_used, p.volume_usd, p.model_sources,
                   p.placed, p.placed_stake, p.market_id, p.line, p.venue,
                   p.maker_fill
            FROM ev_predictions p
            INNER JOIN (
                SELECT market_id, MAX(id) AS max_id
                FROM ev_predictions
                WHERE voided = 0 AND mode = 'live'
                GROUP BY market_id
            ) latest ON p.id = latest.max_id
            LEFT JOIN ev_outcomes o ON p.market_id = o.market_id
            WHERE p.voided = 0 AND p.placed = 0 AND (o.outcome IS NULL)
              AND p.market_type != 'map_handicap'
              AND p.mode = 'live'
              AND (p.event_date IS NULL OR p.event_date = '' OR p.event_date >= ?)
            ORDER BY p.event_date DESC, p.ev_pct DESC
            LIMIT 200
            """,
            (lookback_str,),
        ).fetchall()
    out = [dict(r) for r in rows]
    for b in out:
        b["display_label"] = _display_label_for_row(b)
        ev_date = b.get("event_date")
        b["status"] = "in_progress" if (ev_date and ev_date < today_str) else "upcoming"
    return out


def _synth_kelly_fraction(yes_price: float | None, sharp_prob: float | None) -> float:
    """Half-Kelly fraction from (yes_price, sharp_prob), capped at 5%.

    Used for prop bets — prop_observations doesn't store a kelly_fraction since
    nba_props is shadow-mode and was never sized in production. We compute the
    same half-Kelly the live scanner would have applied so the simulation row
    on the dashboard reflects what would have happened if the category had
    been live.
    """
    if not yes_price or not sharp_prob:
        return 0.0
    if yes_price <= 0 or yes_price >= 1:
        return 0.0
    if sharp_prob <= yes_price:
        return 0.0
    b = 1.0 / yes_price - 1.0  # decimal odds - 1
    p = float(sharp_prob)
    f_full = (b * p - (1.0 - p)) / b
    return max(0.0, min(0.05, f_full * 0.5))


def _settled_prop_bets() -> list[dict[str, Any]]:
    """Resolved prop_observations rows, shaped like settled game bets.

    Sector is suffixed with '_props' so the breakdown table splits e.g.
    'nba' (game markets) and 'nba_props'. Half-Kelly stake is synthesized
    from (kalshi_price, sharp_prob) — see _synth_kelly_fraction.

    Filters:
      - outcome IS NOT NULL (resolved)
      - sharp_prob IS NOT NULL (priced — excludes legacy unmatched rows)
      - ev_pct >= 0.02 (matches the live scanner's actionable threshold; props
        are logged for ALL lines for calibration, but only +EV ones would
        have been bet on)
      - model_version = ANCHOR_MODEL_VERSION (only the new anchor-pricing
        model — legacy L15-era rows preserved as historical record but
        excluded from the forward-looking simulation view)
    """
    from evmax.portfolios import ANCHOR_MODEL_VERSION

    with _conn() as conn:
        rows = conn.execute(
            """
            SELECT id, scan_date, event_date, sector, player_name, stat_type,
                   line, kalshi_price, sharp_prob, ev_pct, outcome, mode,
                   model_version, event_title
            FROM prop_observations
            WHERE outcome IS NOT NULL
              AND sharp_prob IS NOT NULL
              AND ev_pct >= 0.02
              AND model_version = ?
            ORDER BY event_date ASC, id ASC
            """,
            (ANCHOR_MODEL_VERSION,),
        ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        kelly = _synth_kelly_fraction(d.get("kalshi_price"), d.get("sharp_prob"))
        if kelly <= 0:
            continue
        out.append({
            "id": d["id"],
            "scan_date": d["scan_date"],
            "event_date": d["event_date"] or d["scan_date"],
            "sector": f"{d['sector']}_props",
            "event_title": d.get("event_title") or f"{d.get('player_name')} {d.get('stat_type')}",
            "yes_team": d.get("player_name"),
            "market_type": "player_prop",
            "kalshi_yes_price": d["kalshi_price"],
            "blended_true_prob": d["sharp_prob"],
            "sharp_true_prob": d["sharp_prob"],
            "ev_pct": d["ev_pct"],
            "kelly_fraction": kelly,
            "bankroll_used": 250.0,  # standard simulation bankroll
            "model_sources": d.get("model_version") or "",
            "placed": 0,
            "placed_stake": None,
            "placed_price": None,
            "line": d.get("line"),
            "outcome": d["outcome"],
            "display_label": f"{d.get('player_name')} {d.get('line')}+ {d.get('stat_type')}",
        })
    return out


def _bet_pnl(bet: dict[str, Any]) -> float:
    from evmax.fees import bet_pnl as _fee_bet_pnl
    from evmax.settings import get_settings

    bankroll = bet.get("bankroll_used") or 250.0
    kelly = bet.get("kelly_fraction") or 0.0
    stake = bet.get("placed_stake") or (bankroll * kelly)
    price = bet.get("placed_price") or bet.get("kalshi_yes_price") or 0.5
    if price <= 0 or price >= 1:
        return 0.0
    # Net of the venue trading fee (flat entry cost, hits win and loss equally)
    # when fees_in_pricing is on — matches how the bet was gated/sized. A maker
    # fill (recorded via `evmax agents fill` or POST /api/fill) is charged the
    # maker fee, not taker.
    fee_venue = (bet.get("venue") or "kalshi") if get_settings().fees_in_pricing else None
    return _fee_bet_pnl(
        stake, price, bet.get("outcome") == 1,
        venue=fee_venue, maker=bool(bet.get("maker_fill")),
    )


def _profit_series(bets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_date: dict[str, float] = {}
    for b in bets:
        pnl = _bet_pnl(b)
        d = b["event_date"] or b["scan_date"]
        by_date[d] = by_date.get(d, 0.0) + pnl
    cum = 0.0
    out = []
    for dt in sorted(by_date.keys()):
        cum += by_date[dt]
        out.append({"date": dt, "daily": round(by_date[dt], 2), "cumulative": round(cum, 2)})
    return out


def _summary_stats(bets: list[dict[str, Any]]) -> dict[str, Any]:
    if not bets:
        return {"total_bets": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
                "total_pnl": 0.0, "roi_pct": 0.0, "avg_ev": 0.0, "total_staked": 0.0}
    wins = sum(1 for b in bets if b["outcome"] == 1)
    losses = sum(1 for b in bets if b["outcome"] == 0)
    total_pnl = sum(_bet_pnl(b) for b in bets)
    total_staked = sum(
        (b.get("placed_stake") or (b.get("bankroll_used") or 250.0) * (b.get("kelly_fraction") or 0.0))
        for b in bets
    )
    avg_ev = sum((b.get("ev_pct") or 0.0) for b in bets) / len(bets)
    return {
        "total_bets": len(bets), "wins": wins, "losses": losses,
        "win_rate": round(100.0 * wins / max(wins + losses, 1), 1),
        "total_pnl": round(total_pnl, 2), "total_staked": round(total_staked, 2),
        "roi_pct": round(100.0 * total_pnl / max(total_staked, 0.01), 2),
        "avg_ev": round(100.0 * avg_ev, 2),
    }


def _sector_breakdown(bets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_sector: dict[str, list[dict[str, Any]]] = {}
    for b in bets:
        by_sector.setdefault(b["sector"], []).append(b)
    rows = []
    for sector, bs in by_sector.items():
        wins = sum(1 for b in bs if b["outcome"] == 1)
        pnl = sum(_bet_pnl(b) for b in bs)
        staked = sum(
            (b.get("placed_stake") or (b.get("bankroll_used") or 250.0) * (b.get("kelly_fraction") or 0.0))
            for b in bs
        )
        rows.append({
            "sector": sector, "bets": len(bs), "wins": wins,
            "win_rate": round(100.0 * wins / len(bs), 1),
            "pnl": round(pnl, 2),
            "roi_pct": round(100.0 * pnl / max(staked, 0.01), 2),
        })
    rows.sort(key=lambda r: r["pnl"], reverse=True)
    return rows


# ---------------------------------------------------------------------------
# JSON APIs — consumed by the React SPA in frontend/
# ---------------------------------------------------------------------------

@app.get("/api/dashboard")
def api_dashboard() -> JSONResponse:
    """Return all dashboard data as JSON for the React SPA."""
    settled = _settled_bets()
    settled_props = _settled_prop_bets()
    open_bets = _open_bets()

    # Game-only series + summary stay unchanged so the existing chart and KPI
    # cards reflect actual placed bets only. The sector breakdown gets BOTH
    # so '{sector}_props' rows appear alongside '{sector}' (simulation view).
    summary_all = _summary_stats(settled)
    series_all = _profit_series(settled)
    sectors_all = _sector_breakdown(settled + settled_props)

    settled_placed = [b for b in settled if b.get("placed")]
    summary_placed = _summary_stats(settled_placed)
    series_placed = _profit_series(settled_placed)

    recent = list(reversed(settled))[:50]
    placed_bets = _placed_bets()

    return JSONResponse({
        "summary_all": summary_all,
        "summary_placed": summary_placed,
        "series_all": series_all,
        "series_placed": series_placed,
        "sectors": sectors_all,
        "placed_bets": placed_bets,
        "unplaced_bets": open_bets,
        "recent": recent,
    })


@app.get("/api/profit")
def api_profit() -> JSONResponse:
    return JSONResponse(_profit_series(_settled_bets()))


@app.get("/api/sectors")
def api_sectors(period: str = Query("all", alias="range")) -> JSONResponse:
    """Sector breakdown filtered by rolling date window.

    range: "1d", "1w", "1m", "1y", or "all" (default).
    """
    days_map = {"1d": 1, "1w": 7, "1m": 30, "1y": 365, "all": 0}
    days = days_map.get(period, 0)
    # Include synthesized prop bets so '{sector}_props' rows appear alongside
    # game sectors in the date-range filtered breakdown too.
    bets = _settled_bets() + _settled_prop_bets()
    if days > 0:
        cutoff = (date.today() - timedelta(days=days - 1)).isoformat()
        bets = [b for b in bets if (b.get("event_date") or b.get("scan_date") or "") >= cutoff]
    return JSONResponse(_sector_breakdown(bets))


@app.get("/api/promotion-board")
def api_promotion_board(
    days: int = Query(30, ge=1, le=365),
    sector: str = Query(None),
    staleness_h: float = Query(3.0, ge=0.0),
) -> JSONResponse:
    """Promotion scoreboard — per (sector, market_type, venue) health rows.

    Same data as `evmax cleanup shadow board`: sample counts, Brier
    blend-vs-sharp, CLV gate status, and blend divergence (the
    sharp-passthrough detector). See promotion_board.compute_promotion_board.
    """
    from evmax.agents.cleanup.promotion_board import compute_promotion_board

    rows = compute_promotion_board(
        days=days,
        staleness_h=staleness_h if staleness_h > 0 else None,
        sector=sector,
    )
    return JSONResponse({"days": days, "rows": rows})


_MLB_PROPS_CALIBRATION = (
    Path(__file__).resolve().parents[2] / "data" / "backtest" / "mlb_props" / "calibration_2025.json"
)


@app.get("/api/mlb-props")
def api_mlb_props() -> JSONResponse:
    """MLB player-prop status: walk-forward backtest calibration + live shadow tally.

    `calibration` is the per-stat Brier (model vs empirical vs base-rate
    baseline) + decile buckets from scripts/backtest_mlb_props.py — the offline
    signal check. `shadow` is the live baseball_props logging tally from
    prop_observations (everything stays shadow until edge-vs-sharp validates).
    """
    calibration: dict[str, Any] = {}
    if _MLB_PROPS_CALIBRATION.exists():
        try:
            calibration = json.loads(_MLB_PROPS_CALIBRATION.read_text())
        except (json.JSONDecodeError, OSError):
            calibration = {}

    shadow: dict[str, Any] = {"by_stat": [], "total": 0, "resolved": 0}
    try:
        with _conn() as conn:
            rows = conn.execute(
                """SELECT stat_type,
                          COUNT(*) AS n,
                          SUM(CASE WHEN outcome IS NOT NULL THEN 1 ELSE 0 END) AS resolved,
                          AVG(CASE WHEN outcome IS NOT NULL
                                   THEN (sharp_prob - outcome) * (sharp_prob - outcome) END) AS brier
                   FROM prop_observations
                   WHERE sector = 'baseball'
                   GROUP BY stat_type
                   ORDER BY n DESC"""
            ).fetchall()
        by_stat = [
            {
                "stat_type": r["stat_type"],
                "n": r["n"],
                "resolved": r["resolved"] or 0,
                "brier": round(r["brier"], 5) if r["brier"] is not None else None,
            }
            for r in rows
        ]
        shadow = {
            "by_stat": by_stat,
            "total": sum(s["n"] for s in by_stat),
            "resolved": sum(s["resolved"] for s in by_stat),
        }
    except sqlite3.Error:
        pass

    return JSONResponse({"calibration": calibration, "shadow": shadow})


@app.get("/api/summary")
def api_summary(days: int = 0, view: str = "all") -> JSONResponse:
    """Summary stats filtered by rolling date window and view.

    days=0 → all-time; otherwise keep bets with event_date >= today - days.
    view=placed → only placed bets.
    """
    bets = _settled_bets()
    if view == "placed":
        bets = [b for b in bets if b.get("placed")]
    if days > 0:
        cutoff = (date.today() - timedelta(days=days - 1)).isoformat()
        bets = [b for b in bets if (b.get("event_date") or b.get("scan_date") or "") >= cutoff]
    return JSONResponse(_summary_stats(bets))


def _gap_to_dict(g, bankroll: float) -> dict[str, Any]:
    """Shape a single EVGap into the canonical dict used by both the dashboard
    scan view and the portfolio fan-out. Carries both ``kalshi_yes_price`` /
    ``blended_true_prob`` (portfolio names) and ``kalshi_price`` / ``true_prob``
    (display names) so downstream consumers can read either."""
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
        "display_label": _display_label_for_row(label_row),
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
    }


def _portfolio_gap_category(gap: dict[str, Any]) -> str:
    """Portfolio-matching key for a gap dict. Props use a '_props' suffix even
    though gap['sector'] is the base sector (e.g. 'nba')."""
    if gap.get("market_type") == "player_prop":
        return f"{gap.get('sector') or ''}_props"
    return gap.get("sector") or ""


def _gap_in_scan_window(
    event_date: Any, sector: str, date_from: str, date_to: str
) -> bool:
    """True if a gap's Kalshi game-day falls within the requested scan window.

    Uses ``kalshi_game_day`` — the SAME convention ``log_gaps`` writes to the
    ``ev_predictions.event_date`` column — so the persisted set matches the
    range the user selected exactly (the displayed gap dict's ``event_date``
    uses a local-tz convention, which can differ by a day at TZ edges; the
    persistence boundary must follow what actually lands in the DB).

    Empty range → today+tomorrow, mirroring the scan display default. Dateless
    gaps (rare) are kept so the prior persist-all behavior for them is preserved.
    """
    if event_date is None:
        return True
    from evmax.clients.time_util import kalshi_game_day

    day = kalshi_game_day(event_date, sector)
    if date_from and date_to:
        return date_from <= day <= date_to
    if date_from:
        return day >= date_from
    if date_to:
        return day <= date_to
    today_str = date.today().isoformat()
    tomorrow_str = (date.today() + timedelta(days=1)).isoformat()
    return day in (today_str, tomorrow_str)


async def _run_unified_scan(
    sectors: list[str],
    bankroll: float,
    kelly: float,
    fan_out_portfolio_ids: list[str] | None,
    date_from: str = "",
    date_to: str = "",
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]]]:
    """Single shared scan path. Runs the coordinator once, logs game gaps to
    ``ev_predictions``, and (if requested) fans the same gaps out to active
    portfolios via ``log_portfolio_bet``.

    ``fan_out_portfolio_ids``:
      - ``None``  → skip portfolio fan-out entirely
      - ``[]``    → fan out to ALL active portfolios
      - ``[...]`` → fan out only to the listed portfolio ids

    Returns ``(cycle, gap_dicts, portfolio_results)``.
    """
    from evmax.agents.coordinator import AgentCoordinator

    coord = AgentCoordinator(sectors=sectors, bankroll=bankroll, kelly_fraction=kelly)
    cycle = await coord.run_cycle()

    # Log game-level gaps to ev_predictions so "Pick Selected" can resolve by
    # market_id. Props live in prop_observations via the underlying cycle.
    try:
        from evmax.agents.cleanup.logger import log_gaps as _log_gaps
        # Persist only gaps inside the requested date window. The coordinator
        # cycle surfaces every game Kalshi has listed (days out, incl. series
        # games), but a ranged scan must not dump out-of-range rows into
        # ev_predictions — they'd leak into the dashboard's Open Positions,
        # which reads the persisted DB. The date filter previously applied only
        # to the displayed gap list (see api_scan), so persistence ignored the
        # range entirely. Portfolio fan-out below is intentionally NOT date-
        # bounded (it wants 2-4-days-out series games — see the comment there).
        loggable = [
            g for g in cycle.loggable_gaps()
            if _gap_in_scan_window(g.event_date, g.sector, date_from, date_to)
        ]
        if loggable:
            _log_gaps(loggable, bankroll_used=bankroll)
    except Exception as _log_err:
        import structlog
        structlog.get_logger(__name__).warning("web_scan_log_failed", error=str(_log_err))

    # The dashboard scan view + portfolio fan-out show only actionable plays:
    # cycle.plays() drops partial-blend (shadow, $0.00-stake) gaps. They're
    # still logged above via loggable_gaps(). The dashboard deliberately does
    # NOT apply the CLI's min_prob/tiered-EV floor (it shows all ≥2% gaps).
    gap_dicts = [_gap_to_dict(g, bankroll) for g in cycle.plays(require_full_blend=True)]

    portfolio_results: list[dict[str, Any]] = []
    if fan_out_portfolio_ids is not None:
        from evmax.portfolios import (
            list_portfolios, log_portfolio_bet, is_excluded_from_portfolio,
            select_best_execution_plays,
        )
        portfolios = list_portfolios(active_only=True)
        if fan_out_portfolio_ids:
            wanted = set(fan_out_portfolio_ids)
            portfolios = [p for p in portfolios if p.id in wanted]

        today_str = date.today().isoformat()

        # Best-execution collapse for the LEDGER only: the same market quoted
        # on both Kalshi and Polymarket US is one portfolio bet, not two. The
        # displayed gap_dicts keep both venue legs.
        fan_out_gaps = select_best_execution_plays(gap_dicts)

        for portfolio in portfolios:
            logged = 0
            for gap in fan_out_gaps:
                if _portfolio_gap_category(gap) not in portfolio.sectors:
                    continue
                if gap.get("market_type") == "map_handicap":
                    continue
                if is_excluded_from_portfolio(gap.get("sector"), gap.get("market_type")):
                    continue
                # No event-date floor: the scanner only emits gaps for games
                # Kalshi has actually listed, so anything that reaches here is
                # a real bettable market — restricting to today+tomorrow would
                # silently drop Finals/series games 2-4 days out.
                gap_for_log = dict(gap)
                gap_for_log["scan_date"] = today_str
                if log_portfolio_bet(
                    portfolio_id=portfolio.id,
                    gap=gap_for_log,
                    bankroll=portfolio.current_bankroll,
                    kelly=portfolio.kelly_fraction,
                ):
                    logged += 1
            portfolio_results.append({
                "portfolio_id": portfolio.id,
                "portfolio_name": portfolio.name,
                "gaps_logged": logged,
            })

    return cycle, gap_dicts, portfolio_results


def _scan_sector_specs() -> list[Any]:
    """Game (non-prop) categories eligible for the dashboard sector pickers.

    Sourced from the category registry (``data/categories.yaml``) so the
    frontend dropdowns and the scan default stay in sync with the one source
    of truth — no hardcoded sector lists. Props are excluded because the scan
    UI hides player-prop gaps, and ``disabled`` categories are dropped because
    nothing persists for them.
    """
    from evmax.categories import all_categories

    return [c for c in all_categories() if not c.is_prop and c.mode != "disabled"]


def _default_scan_sectors() -> str:
    """Comma-joined default sector string for a scan with no explicit sectors."""
    return ",".join(c.key for c in _scan_sector_specs())


@app.get("/api/categories")
async def api_categories() -> JSONResponse:
    """Sector options for the dashboard, sourced from the category registry.

    Keeps the frontend's sector dropdowns driven by ``data/categories.yaml``
    instead of duplicated hardcoded arrays.
    """
    cats = [
        {"key": c.key, "display_name": c.display_name, "mode": c.mode}
        for c in _scan_sector_specs()
    ]
    return JSONResponse({"categories": cats})


@app.get("/api/balance")
async def api_balance(venue: str = Query("", alias="venue")) -> JSONResponse:
    """Live TOTAL account wealth per venue (cash + open positions), for the
    venue filter to seed the Kelly bankroll.

    ``{"balances": {"kalshi": 512.34, "polymarket_us": null}}`` — a null means
    no credentials / fetch failure, and the UI keeps the manual bankroll for
    that venue rather than sizing against a guess. Read-only; never mutates.
    """
    from evmax.clients.balances import SUPPORTED_VENUES, fetch_all_balances

    targets = [venue.lower()] if venue else list(SUPPORTED_VENUES)
    targets = [v for v in targets if v in SUPPORTED_VENUES]
    if not targets:
        return JSONResponse({"error": f"unknown venue {venue!r}"}, status_code=400)
    balances = await fetch_all_balances(targets)
    return JSONResponse({"balances": balances})


@app.post("/api/scan")
async def api_scan(request: Request) -> JSONResponse:
    """Run a full agent scan and return EV gaps.

    Also fans the scan out to all active portfolios by default so the placement
    table and the portfolio cards are guaranteed to come from the same cycle.
    Set ``fan_out_portfolios: false`` to opt out, or pass ``portfolio_ids`` to
    target a subset.
    """
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    sectors_str = body.get("sectors") or _default_scan_sectors()
    bankroll = float(body.get("bankroll", 500))
    kelly = float(body.get("kelly", 0.5))
    # Optionally source the bankroll from a venue's LIVE balance (the venue
    # filter's "size against my actual balance" flow). Falls back to the passed
    # bankroll if the balance is unavailable; the source is echoed back so the
    # UI can show whether real capital or the manual figure was used.
    bankroll_venue = (body.get("bankroll_venue") or "").strip().lower() or None
    bankroll_source = "manual"
    if bankroll_venue:
        from evmax.clients.balances import resolve_bankroll
        bankroll, bankroll_source = await resolve_bankroll(bankroll, bankroll_venue)
    date_from = body.get("date_from", "")
    date_to = body.get("date_to", "")
    fan_out = body.get("fan_out_portfolios", True)
    portfolio_ids = body.get("portfolio_ids", []) or []

    sectors = [s.strip() for s in sectors_str.split(",") if s.strip()]
    fan_out_arg: list[str] | None = list(portfolio_ids) if fan_out else None

    cycle, gap_dicts, portfolio_results = await _run_unified_scan(
        sectors=sectors,
        bankroll=bankroll,
        kelly=kelly,
        fan_out_portfolio_ids=fan_out_arg,
        date_from=date_from,
        date_to=date_to,
    )

    # Filter for the placement table view
    gaps = list(gap_dicts)
    if date_from and date_to:
        gaps = [g for g in gaps if date_from <= g["event_date"] <= date_to]
    elif date_from:
        gaps = [g for g in gaps if g["event_date"] >= date_from]
    elif date_to:
        gaps = [g for g in gaps if g["event_date"] <= date_to]
    else:
        today_str = date.today().isoformat()
        tomorrow_str = (date.today() + timedelta(days=1)).isoformat()
        gaps = [g for g in gaps if g["event_date"] in (today_str, tomorrow_str)]

    # Drop esports map handicaps (LoL/CS2 set handicaps not on Kalshi).
    gaps = [g for g in gaps if g["market_type"] != "map_handicap"]

    # Hide player props from the scan UI — anchor pricing produces hundreds
    # of prop gaps per cycle which clutter the game-market display. Props
    # still flow into prop_observations + portfolio_bets via the underlying
    # cycle, just not surfaced in the dashboard's scan list.
    gaps = [g for g in gaps if g["market_type"] != "player_prop"]

    # Exclude markets already placed
    with _conn() as conn:
        placed_mids = {r[0] for r in conn.execute(
            "SELECT DISTINCT market_id FROM ev_predictions "
            "WHERE placed = 1 AND voided = 0 AND mode = 'live'"
        ).fetchall()}
    gaps = [g for g in gaps if g["market_id"] not in placed_mids]

    return JSONResponse({
        "gaps": gaps,
        "markets_fetched": cycle.markets_fetched,
        "markets_matched": cycle.markets_matched,
        "sectors": sectors,
        "portfolio_results": portfolio_results,
        "bankroll": round(bankroll, 2),
        "bankroll_source": bankroll_source,
    })


@app.post("/api/pick")
async def api_pick(request: Request) -> JSONResponse:
    """Mark selected bets as placed in the database.

    Accepts either:
      { "bets": [{"market_id": "...", "fill_price": 0.45, "fill_stake": 12.50}, ...] }
    or legacy:
      { "market_ids": ["..."] }
    """
    body = await request.json()
    bets_list: list[dict] = body.get("bets", [])

    # Legacy fallback: bare market_ids without price/stake
    if not bets_list:
        market_ids = body.get("market_ids", [])
        bets_list = [{"market_id": mid} for mid in market_ids]

    if not bets_list:
        return JSONResponse({"error": "No bets provided"}, status_code=400)

    now_str = datetime.now(timezone.utc).isoformat()
    conn = _conn()
    placed = 0
    skipped: list[dict] = []
    for bet in bets_list:
        mid = bet.get("market_id")
        if not mid:
            skipped.append({"market_id": None, "reason": "missing market_id"})
            continue

        # Diagnose why a market_id might not be pickable. Each branch is a
        # distinct silent-skip this endpoint used to swallow.
        diag = conn.execute(
            "SELECT kalshi_yes_price, kelly_fraction, bankroll_used, voided, mode, placed "
            "FROM ev_predictions WHERE market_id = ? ORDER BY id DESC LIMIT 1",
            (mid,),
        ).fetchone()
        if not diag:
            skipped.append({"market_id": mid, "reason": "not in ev_predictions (try rescanning)"})
            continue
        if diag["placed"] == 1:
            skipped.append({"market_id": mid, "reason": "already placed"})
            continue
        if diag["voided"] == 1:
            skipped.append({"market_id": mid, "reason": "voided (run a fresh scan to un-stick)"})
            continue
        if diag["mode"] != "live":
            skipped.append({"market_id": mid, "reason": f"mode={diag['mode']} (only live is pickable)"})
            continue

        # Use user-provided fill price/stake, fall back to scan defaults
        fill_price = bet.get("fill_price") or diag["kalshi_yes_price"]
        default_stake = round((diag["bankroll_used"] or 500.0) * (diag["kelly_fraction"] or 0.0), 2)
        fill_stake = bet.get("fill_stake") or default_stake

        conn.execute(
            "UPDATE ev_predictions SET placed = 1, placed_at = ?, placed_price = ?, placed_stake = ? WHERE market_id = ?",
            (now_str, fill_price, fill_stake, mid),
        )
        placed += 1
    conn.commit()
    conn.close()
    return JSONResponse({"placed": placed, "skipped": skipped})


@app.post("/api/fill")
async def api_fill(request: Request) -> JSONResponse:
    """Record filled maker limit orders as real placed positions.

    The dashboard counterpart of ``evmax agents fill``. A maker-only play is not
    crossable at the ask, so it persists as ``mode='shadow'`` and ``/api/pick``
    correctly refuses it — picking it would record a taker fill at a price that
    was never +EV. This endpoint is the other half of that workflow: once your
    resting order actually fills, it promotes the row to a live position with
    ``maker_fill=1`` so the P&L charges the maker fee.

    Body::

        { "fills": [{"market_id": "...", "fill_price": 0.47, "fill_stake": 12.5}, ...] }

    ``fill_price`` accepts cents (47) or a fraction (0.47). Rows that cannot be
    filled are reported in ``skipped`` with a reason slug rather than failing the
    whole batch, so one stale selection never discards the rest of the fills.
    """
    from evmax.agents.cleanup.maker_fill import (
        MakerFillError, _as_dict, record_maker_fill,
    )

    body = await request.json()
    fills_list: list[dict] = body.get("fills", [])
    if not fills_list:
        return JSONResponse({"error": "No fills provided"}, status_code=400)

    conn = _conn()
    recorded: list[dict[str, Any]] = []
    skipped: list[dict] = []
    for item in fills_list:
        mid = item.get("market_id")
        try:
            result = record_maker_fill(
                conn, mid, item.get("fill_price"), item.get("fill_stake"),
            )
        except MakerFillError as err:
            skipped.append({"market_id": mid, "reason": str(err)})
            continue
        recorded.append(_as_dict(result))
    conn.commit()
    conn.close()
    return JSONResponse({"filled": len(recorded), "fills": recorded, "skipped": skipped})


@app.post("/api/update-placed")
async def api_update_placed(request: Request) -> JSONResponse:
    """Update fill price and stake on already-placed bets."""
    body = await request.json()
    edits: list[dict] = body.get("edits", [])
    if not edits:
        return JSONResponse({"error": "No edits provided"}, status_code=400)

    conn = _conn()
    updated = 0
    for edit in edits:
        mid = edit.get("market_id")
        price = edit.get("fill_price")
        stake = edit.get("fill_stake")
        if not mid:
            continue
        # Partial updates: only set columns the caller provided. If the user
        # only edits stake (odds left blank), price comes through as None and
        # we must not overwrite the existing placed_price with NULL.
        sets, params = [], []
        if price is not None:
            sets.append("placed_price = ?")
            params.append(price)
        if stake is not None:
            sets.append("placed_stake = ?")
            params.append(stake)
        if not sets:
            continue
        params.append(mid)
        cur = conn.execute(
            f"UPDATE ev_predictions SET {', '.join(sets)} WHERE market_id = ? AND placed = 1",
            params,
        )
        if cur.rowcount > 0:
            updated += 1
    conn.commit()
    conn.close()
    return JSONResponse({"updated": updated})


@app.post("/api/unplace")
async def api_unplace(request: Request) -> JSONResponse:
    """Remove placed status from bets (un-pick them)."""
    body = await request.json()
    market_ids: list[str] = body.get("market_ids", [])
    if not market_ids:
        return JSONResponse({"error": "No market_ids provided"}, status_code=400)

    conn = _conn()
    removed = 0
    for mid in market_ids:
        # Un-placing a maker fill must undo the whole promotion record_maker_fill
        # made, not just the placement:
        #   maker_fill -> 0, because _bet_pnl charges the MAKER fee off this flag.
        #     Leaving it set would price a later TAKER pick at the maker fee.
        #   mode -> 'shadow', because the row only became live as part of that
        #     promotion. A maker-only play is not crossable at the ask, so
        #     leaving it live would make it taker-pickable at a price that was
        #     never +EV.
        # Demoting to shadow is safe even for a row that was already live before
        # its fill: log_gaps upgrades an unplaced shadow row back to live on the
        # next scan that still qualifies it (prediction_upgraded_shadow_to_live).
        cur = conn.execute(
            "UPDATE ev_predictions SET placed = 0, placed_at = NULL, placed_price = NULL, "
            "placed_stake = NULL, mode = CASE WHEN maker_fill = 1 THEN 'shadow' ELSE mode END, "
            "maker_fill = 0 WHERE market_id = ? AND placed = 1",
            (mid,),
        )
        if cur.rowcount > 0:
            removed += 1
    conn.commit()
    conn.close()
    return JSONResponse({"removed": removed})


@app.post("/api/resolve")
async def api_resolve(request: Request) -> JSONResponse:
    """Resolve outcomes for a given date (defaults to yesterday).

    Symmetric with ``/api/scan``'s portfolio fan-out: after resolving the
    date's ev_outcomes, this also syncs portfolio bets against those freshly
    resolved outcomes (unless ``sync_portfolios=false``), so the dashboard
    "Resolve" button is one click for both dashboard *and* portfolio outcomes —
    no separate "Sync Outcomes" trip required.
    """
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    target = body.get("date")
    if target:
        d = date.fromisoformat(target)
    else:
        d = date.today() - timedelta(days=1)
    sync_portfolios = body.get("sync_portfolios", True)

    from evmax.agents.cleanup.resolver import resolve_outcomes_for_date
    result = await resolve_outcomes_for_date(d)

    portfolio_resolved = 0
    if sync_portfolios:
        # sync_portfolio_outcomes() calls asyncio.run() internally, so it
        # cannot run inside this already-running event loop — hand it to a
        # worker thread (same way Starlette runs the sync /sync-outcomes route).
        from starlette.concurrency import run_in_threadpool
        from evmax.portfolios import sync_portfolio_outcomes
        portfolio_resolved = await run_in_threadpool(sync_portfolio_outcomes)

    return JSONResponse({
        "date": d.isoformat(),
        "resolved": result["resolved"],
        "failed": result["failed"],
        "unmatched": result.get("unmatched", [])[:20],
        "portfolio_resolved": portfolio_resolved,
    })


@app.get("/api/portfolios")
def api_portfolios() -> JSONResponse:
    from evmax.portfolios import list_portfolios, get_portfolio_stats
    portfolios = list_portfolios(active_only=True)
    result = []
    for p in portfolios:
        stats = get_portfolio_stats(p.id)
        result.append({**p.to_dict(), **stats})
    return JSONResponse(result)


@app.post("/api/portfolios")
async def api_create_portfolio(request: Request) -> JSONResponse:
    from evmax.portfolios import create_portfolio
    body = await request.json()
    p = create_portfolio(
        portfolio_id=body["id"],
        name=body["name"],
        sectors=body["sectors"],
        bankroll=float(body["bankroll"]),
        kelly_fraction=float(body.get("kelly_fraction", 0.5)),
        scenario=body.get("scenario", "moderate"),
    )
    return JSONResponse(p.to_dict())


@app.post("/api/portfolios/create-defaults")
def api_create_defaults() -> JSONResponse:
    from evmax.portfolios import create_default_portfolios
    created = create_default_portfolios()
    return JSONResponse({"created": len(created), "portfolios": [p.to_dict() for p in created]})


@app.post("/api/portfolios/create-prop-defaults")
def api_create_prop_defaults() -> JSONResponse:
    """Create prop-only portfolios with simulated bets backfilled from
    prop_observations. Safe to call repeatedly — INSERT OR IGNORE on both
    portfolios and portfolio_bets means existing rows are preserved."""
    from evmax.portfolios import create_prop_portfolios
    created = create_prop_portfolios(backfill=True)
    return JSONResponse({"created": len(created), "portfolios": [p.to_dict() for p in created]})


@app.get("/api/portfolios/{portfolio_id}")
def api_portfolio_detail(portfolio_id: str) -> JSONResponse:
    from evmax.portfolios import get_portfolio, get_portfolio_stats, get_portfolio_bets
    p = get_portfolio(portfolio_id)
    if not p:
        return JSONResponse({"error": "Portfolio not found"}, status_code=404)
    stats = get_portfolio_stats(portfolio_id)
    bets = get_portfolio_bets(portfolio_id)
    return JSONResponse({**p.to_dict(), **stats, "bets": bets})


@app.delete("/api/portfolios/{portfolio_id}")
def api_delete_portfolio(portfolio_id: str) -> JSONResponse:
    from evmax.portfolios import delete_portfolio
    ok = delete_portfolio(portfolio_id)
    return JSONResponse({"deleted": ok})


@app.post("/api/portfolios/scan")
async def api_portfolio_scan(request: Request) -> JSONResponse:
    """Thin alias over ``/api/scan`` for the PortfolioGrid "Scan All" button.
    Runs the same unified scan path so portfolio-card results and the
    placement table always come from the same coordinator cycle.
    """
    from evmax.portfolios import list_portfolios

    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    portfolio_ids: list[str] = body.get("portfolio_ids", [])

    portfolios = list_portfolios(active_only=True)
    if portfolio_ids:
        portfolios = [p for p in portfolios if p.id in portfolio_ids]

    if not portfolios:
        return JSONResponse({"error": "No active portfolios"}, status_code=400)

    # Coordinator scans by base sector — strip the '_props' suffix from
    # portfolio sector keys when telling it what to fetch. The fan-out helper
    # re-derives the '{sector}_props' key when matching gaps to portfolios.
    base_sectors = sorted({
        (s[: -len("_props")] if s.endswith("_props") else s)
        for p in portfolios for s in p.sectors
    })

    cycle, _gap_dicts, portfolio_results = await _run_unified_scan(
        sectors=base_sectors,
        bankroll=500,
        kelly=0.5,
        fan_out_portfolio_ids=[p.id for p in portfolios],
    )

    return JSONResponse({
        "portfolios_scanned": len(portfolios),
        "markets_fetched": cycle.markets_fetched,
        "markets_matched": cycle.markets_matched,
        "results": portfolio_results,
    })


@app.post("/api/portfolios/sync-outcomes")
def api_sync_outcomes() -> JSONResponse:
    from evmax.portfolios import sync_portfolio_outcomes
    count = sync_portfolio_outcomes()
    return JSONResponse({"resolved": count})


@app.post("/api/metrics")
async def api_metrics(request: Request) -> JSONResponse:
    """Return calibration metrics."""
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    weeks = int(body.get("weeks", 4))

    conn = _conn()
    since = (date.today() - timedelta(weeks=weeks)).isoformat()
    rows = conn.execute(
        """
        SELECT p.sector, p.market_type, p.blended_true_prob, p.sharp_true_prob,
               p.ev_pct, p.kalshi_clv_pct, p.kelly_fraction, p.bankroll_used,
               p.placed_stake, p.placed_price, p.kalshi_yes_price, p.venue, o.outcome
        FROM ev_predictions p
        JOIN ev_outcomes o ON p.market_id = o.market_id
        WHERE p.voided = 0 AND o.outcome IS NOT NULL AND p.event_date >= ?
          AND p.mode = 'live'
        """,
        (since,),
    ).fetchall()
    conn.close()

    if not rows:
        return JSONResponse({"error": "No resolved bets in date range", "weeks": weeks})

    rows = [dict(r) for r in rows]
    n = len(rows)
    brier_model = sum((r["blended_true_prob"] - r["outcome"]) ** 2 for r in rows) / n
    brier_sharp = sum((r["sharp_true_prob"] - r["outcome"]) ** 2 for r in rows) / n
    wins = sum(1 for r in rows if r["outcome"] == 1)

    # Calibration buckets
    buckets: dict[str, dict] = {}
    for r in rows:
        p = r["blended_true_prob"]
        if p < 0.3:
            label = "<30%"
        elif p < 0.5:
            label = "30-50%"
        elif p < 0.7:
            label = "50-70%"
        else:
            label = ">70%"
        if label not in buckets:
            buckets[label] = {"n": 0, "pred_sum": 0.0, "actual_sum": 0}
        buckets[label]["n"] += 1
        buckets[label]["pred_sum"] += p
        buckets[label]["actual_sum"] += r["outcome"]

    calibration = []
    for label in ["<30%", "30-50%", "50-70%", ">70%"]:
        b = buckets.get(label)
        if b and b["n"] > 0:
            calibration.append({
                "bucket": label, "n": b["n"],
                "predicted": round(b["pred_sum"] / b["n"] * 100, 1),
                "actual": round(b["actual_sum"] / b["n"] * 100, 1),
            })

    # Per-sector performance — CLV, EV%, ROI. CLV (kalshi_clv_pct) is already
    # stored in percentage points; ev_pct is a fraction (0.07 → 7%). ROI is
    # realized P&L / staked, same sizing fallback as _sector_breakdown.
    by_sector: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_sector.setdefault(r["sector"] or "?", []).append(r)
    sectors = []
    for sector, bs in by_sector.items():
        wins_s = sum(1 for b in bs if b["outcome"] == 1)
        pnl = sum(_bet_pnl(b) for b in bs)
        staked = sum(
            (b.get("placed_stake") or (b.get("bankroll_used") or 250.0) * (b.get("kelly_fraction") or 0.0))
            for b in bs
        )
        clvs = [b["kalshi_clv_pct"] for b in bs if b["kalshi_clv_pct"] is not None]
        evs = [b["ev_pct"] for b in bs if b["ev_pct"] is not None]
        brier_m = sum((b["blended_true_prob"] - b["outcome"]) ** 2 for b in bs) / len(bs)
        brier_s = sum((b["sharp_true_prob"] - b["outcome"]) ** 2 for b in bs) / len(bs)
        sectors.append({
            "sector": sector,
            "bets": len(bs),
            "win_rate": round(100.0 * wins_s / len(bs), 1),
            "pnl": round(pnl, 2),
            "roi_pct": round(100.0 * pnl / max(staked, 0.01), 2),
            "avg_ev_pct": round(100.0 * sum(evs) / len(evs), 2) if evs else None,
            "avg_clv_pct": round(sum(clvs) / len(clvs), 2) if clvs else None,
            "clv_n": len(clvs),
            "brier_model": round(brier_m, 4),
            "brier_sharp": round(brier_s, 4),
        })
    sectors.sort(key=lambda s: s["pnl"], reverse=True)

    return JSONResponse({
        "weeks": weeks, "n": n, "wins": wins, "win_rate": round(100 * wins / n, 1),
        "brier_model": round(brier_model, 4), "brier_sharp": round(brier_sharp, 4),
        "calibration": calibration,
        "sectors": sectors,
    })


@app.post("/api/arb/scan")
async def api_arb_scan(request: Request) -> JSONResponse:
    """Cross-venue (Kalshi vs Polymarket US) arb scan — mirrors `evmax arb scan`.

    Read-only: fetches both venues' live books per sector and returns every
    complete-outcome basket whose fee-inclusive cost is at most ``max_cost``
    (default 1.02 keeps 2pp near-misses visible; 1.0 = true arbs only).
    Nothing is persisted and no orders are placed. Asks are REST snapshots —
    depth/price must be verified live before acting.
    """
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    sectors_str = body.get("sectors") or ""
    max_cost = float(body.get("max_cost", 1.02))
    include_in_play = bool(body.get("include_in_play", False))

    from evmax.arb import fetch_arb_markets, find_arbs
    from evmax.clients.polymarket_us import POLYMARKET_US_LEAGUE_MAP

    # Default matches the CLI: every sector both venues carry. Kalshi-only
    # sectors are still scannable (self-arb baskets only).
    sector_list = (
        [s.strip().lower() for s in sectors_str.split(",") if s.strip()]
        or list(POLYMARKET_US_LEAGUE_MAP)
    )
    pools = await fetch_arb_markets(sector_list)

    arbs = []
    sector_counts = []
    for sector, pool in pools.items():
        n_kalshi = sum(1 for m in pool if m.source.value == "kalshi")
        found = find_arbs(pool, max_net_cost=max_cost, include_in_play=include_in_play)
        arbs.extend(found)
        sector_counts.append({
            "sector": sector,
            "kalshi": n_kalshi,
            "polymarket_us": len(pool) - n_kalshi,
            "baskets": len(found),
        })

    arbs.sort(key=lambda a: a.net_cost)
    return JSONResponse({
        "arbs": [
            {
                "sector": a.sector,
                "event_title": a.event_title,
                "event_date": a.event_date.isoformat() if a.event_date else None,
                "market_desc": a.market_desc,
                "legs": [
                    {
                        "venue": leg.venue,
                        "market_id": leg.market_id,
                        "side": leg.side,
                        "outcome": leg.outcome,
                        "ask": round(leg.ask, 4),
                        "fee": round(leg.fee, 4),
                        "volume_usd": round(leg.volume_usd, 2),
                    }
                    for leg in a.legs
                ],
                "gross_cost": round(a.gross_cost, 4),
                "net_cost": round(a.net_cost, 4),
                "net_edge": round(a.net_edge, 4),
                "cross_venue": a.cross_venue,
                "in_play": a.in_play,
                "min_volume_usd": round(min(leg.volume_usd for leg in a.legs), 2),
            }
            for a in arbs
        ],
        "sectors": sector_counts,
        "max_cost": max_cost,
    })


# ---------------------------------------------------------------------------
# React SPA — must be last so /api/* routes match first
# ---------------------------------------------------------------------------

if (SPA_DIST / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=str(SPA_DIST / "assets")), name="spa-assets")

    @app.get("/", response_class=HTMLResponse)
    def spa_root() -> FileResponse:
        return FileResponse(SPA_DIST / "index.html")

    @app.get("/{path:path}", response_class=HTMLResponse)
    def spa_catchall(path: str) -> FileResponse:
        candidate = SPA_DIST / path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(SPA_DIST / "index.html")
else:
    @app.get("/", response_class=HTMLResponse)
    def spa_missing() -> HTMLResponse:
        return HTMLResponse(
            "<h1>frontend not built</h1>"
            "<p>Run <code>cd frontend &amp;&amp; npm install &amp;&amp; npm run build</code>, "
            "then restart <code>evmax dashboard serve</code>.</p>",
            status_code=503,
        )
