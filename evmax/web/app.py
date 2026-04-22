"""FastAPI dashboard for evmax — tables, profit chart, and action buttons."""
from __future__ import annotations

import asyncio
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

ROOT = Path(__file__).resolve().parent.parent.parent
PREDICTIONS_DB = ROOT / "data" / "predictions.db"
TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

app = FastAPI(title="evmax dashboard")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"], allow_methods=["*"], allow_headers=["*"])


def _prob_to_american(prob: float) -> str:
    """Convert probability (0-1) to American odds string."""
    if prob <= 0 or prob >= 1:
        return "+100"
    if prob < 0.5:
        return f"+{round((1 / prob - 1) * 100)}"
    return f"-{round((prob / (1 - prob)) * 100)}"


TEMPLATES.env.globals["probToAmerican"] = _prob_to_american


_STAT_LABELS = {
    "points": "PTS", "assists": "AST", "rebounds": "REB",
    "threes": "3PM", "pra": "PRA", "pts_reb": "P+R", "pts_ast": "P+A",
    "reb_ast": "R+A", "steals": "STL", "blocks": "BLK",
    "strikeouts": "K", "hits": "H", "total_bases": "TB",
}


def _display_label_for_row(row: dict[str, Any]) -> str:
    """Render a human-readable outcome label for one predictions.db row or scan gap.

    Mirrors EVGap.display_label but works off a plain dict so we can use it for
    both DB-backed bets and live scan gaps. Props show as "Mathurin 4+ AST"
    instead of the old "bennedict_mathurin player_prop 4".
    """
    mt = (row.get("market_type") or "").lower()
    yes_team = (row.get("yes_team") or "?").strip()
    line = row.get("line")

    if mt == "player_prop":
        player = (row.get("prop_player_name") or yes_team or "?").replace("_", " ")
        player = " ".join(p.capitalize() for p in player.split())
        stat_raw = (row.get("prop_stat_type") or "").lower()
        stat = _STAT_LABELS.get(stat_raw, stat_raw.replace("_", " ").upper() or "PROP")
        thr = row.get("prop_threshold") if row.get("prop_threshold") is not None else line
        if thr is not None:
            try:
                thr_str = f"{float(thr):g}+"
            except (TypeError, ValueError):
                thr_str = str(thr)
            return f"{player} {thr_str} {stat}"
        return f"{player} {stat}"

    team = yes_team.capitalize() if yes_team else "?"
    if mt == "moneyline":
        return f"{team} ML"
    if mt == "spread" and line is not None:
        try:
            line_str = f"{float(line):.1f}".rstrip("0").rstrip(".")
        except (TypeError, ValueError):
            line_str = str(line)
        return f"{team} {line_str}"
    if mt in ("over_under", "total") and line is not None:
        try:
            return f"O/U {float(line):.1f}"
        except (TypeError, ValueError):
            return f"O/U {line}"
    return team


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
                   p.placed, p.placed_stake, p.placed_price, p.line,
                   o.outcome
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
                   p.placed, p.placed_stake, p.placed_price, p.market_id, p.line
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


def _open_bets() -> list[dict[str, Any]]:
    """Return open (unresolved, unplaced) bets, deduplicated by market_id."""
    with _conn() as conn:
        rows = conn.execute(
            """
            SELECT p.id, p.scan_date, p.event_date, p.sector, p.event_title,
                   p.yes_team, p.market_type, p.kalshi_yes_price,
                   p.blended_true_prob, p.ev_pct, p.kelly_fraction,
                   p.bankroll_used, p.volume_usd, p.model_sources,
                   p.placed, p.placed_stake, p.market_id, p.line
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
            ORDER BY p.event_date DESC, p.ev_pct DESC
            LIMIT 200
            """
        ).fetchall()
    out = [dict(r) for r in rows]
    for b in out:
        b["display_label"] = _display_label_for_row(b)
    return out


def _bet_pnl(bet: dict[str, Any]) -> float:
    bankroll = bet.get("bankroll_used") or 250.0
    kelly = bet.get("kelly_fraction") or 0.0
    stake = bet.get("placed_stake") or (bankroll * kelly)
    price = bet.get("placed_price") or bet.get("kalshi_yes_price") or 0.5
    if price <= 0 or price >= 1:
        return 0.0
    if bet.get("outcome") == 1:
        return stake * (1.0 / price - 1.0)
    return -stake


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
# Pages
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> Any:
    settled = _settled_bets()
    open_bets = _open_bets()

    # All scanned bets
    summary_all = _summary_stats(settled)
    series_all = _profit_series(settled)
    sectors_all = _sector_breakdown(settled)

    # Placed-only subset
    settled_placed = [b for b in settled if b.get("placed")]
    summary_placed = _summary_stats(settled_placed)
    series_placed = _profit_series(settled_placed)

    recent = list(reversed(settled))[:50]
    placed_bets = _placed_bets()
    unplaced_bets = open_bets
    return TEMPLATES.TemplateResponse(
        request, "dashboard.html",
        {"summary_all": summary_all, "summary_placed": summary_placed,
         "sectors": sectors_all, "placed_bets": placed_bets,
         "unplaced_bets": unplaced_bets, "recent": recent,
         "series_all": series_all, "series_placed": series_placed},
    )


# ---------------------------------------------------------------------------
# JSON APIs (for dashboard AJAX)
# ---------------------------------------------------------------------------

@app.get("/api/dashboard")
def api_dashboard() -> JSONResponse:
    """Return all dashboard data as JSON for the React SPA."""
    settled = _settled_bets()
    open_bets = _open_bets()

    summary_all = _summary_stats(settled)
    series_all = _profit_series(settled)
    sectors_all = _sector_breakdown(settled)

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
    bets = _settled_bets()
    if days > 0:
        cutoff = (date.today() - timedelta(days=days - 1)).isoformat()
        bets = [b for b in bets if (b.get("event_date") or b.get("scan_date") or "") >= cutoff]
    return JSONResponse(_sector_breakdown(bets))


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


@app.post("/api/scan")
async def api_scan(request: Request) -> JSONResponse:
    """Run a full agent scan and return EV gaps as JSON."""
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    sectors_str = body.get("sectors", "nba,soccer,tennis")
    bankroll = float(body.get("bankroll", 500))
    kelly = float(body.get("kelly", 0.5))
    date_from = body.get("date_from", "")
    date_to = body.get("date_to", "")

    from evmax.agents.coordinator import AgentCoordinator
    sectors = [s.strip() for s in sectors_str.split(",") if s.strip()]
    coord = AgentCoordinator(
        sectors=sectors, bankroll=bankroll, kelly_fraction=kelly,
    )
    cycle = await coord.run_cycle()

    # Persist game-level gaps to ev_predictions so "Pick Selected" can find
    # them by market_id. Props are excluded (scan shows them but they aren't
    # logged as tradeable predictions).
    try:
        from evmax.agents.cleanup.logger import log_gaps as _log_gaps
        loggable = [g for g in cycle.top_gaps if "::prop::" not in (g.event_id or "")]
        if loggable:
            _log_gaps(loggable, bankroll_used=bankroll)
    except Exception as _log_err:
        import structlog
        structlog.get_logger(__name__).warning("web_scan_log_failed", error=str(_log_err))

    gaps = []
    for g in cycle.top_gaps:
        # Use the same helper as DB-backed rows so scan + history labels
        # match and props render as "Mathurin 4+ AST" instead of
        # "Assists O4.0" (EVGap.display_label omits the player name).
        label_row = {
            "market_type": g.market_type or "",
            "yes_team": g.yes_team or "",
            "line": g.line,
            "prop_player_name": g.prop_player_name,
            "prop_stat_type": g.prop_stat_type,
            "prop_threshold": g.prop_threshold,
        }
        gaps.append({
            "event_title": g.event_title or "",
            "yes_team": g.yes_team or "",
            "market_type": g.market_type or "",
            "display_label": _display_label_for_row(label_row),
            "line": g.line if g.line is None else float(g.line) if isinstance(g.line, (int, float)) else str(g.line),
            "sector": g.sector or "",
            "kalshi_price": round(g.kalshi_yes_price, 2),
            "true_prob": round(g.blended_true_prob, 3),
            "ev_pct": round(g.ev_pct * 100, 2),
            "kelly_pct": round(g.kelly_fraction * 100, 2),
            "stake": round(bankroll * g.kelly_fraction, 2),
            "model_sources": g.model_sources or "",
            "market_id": g.market_id or "",
            "event_date": str(g.event_date.astimezone().strftime("%Y-%m-%d") if g.event_date else ""),
            "volume": g.volume_usd or 0,
        })

    # Filter by requested date range (defaults to today + tomorrow)
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
        if not mid or price is None or stake is None:
            continue
        conn.execute(
            "UPDATE ev_predictions SET placed_price = ?, placed_stake = ? WHERE market_id = ? AND placed = 1",
            (price, stake, mid),
        )
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
        cur = conn.execute(
            "UPDATE ev_predictions SET placed = 0, placed_at = NULL, placed_price = NULL, placed_stake = NULL WHERE market_id = ? AND placed = 1",
            (mid,),
        )
        if cur.rowcount > 0:
            removed += 1
    conn.commit()
    conn.close()
    return JSONResponse({"removed": removed})


@app.post("/api/resolve")
async def api_resolve(request: Request) -> JSONResponse:
    """Resolve outcomes for a given date (defaults to yesterday)."""
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    target = body.get("date")
    if target:
        d = date.fromisoformat(target)
    else:
        d = date.today() - timedelta(days=1)

    from evmax.agents.cleanup.resolver import resolve_outcomes_for_date
    result = await resolve_outcomes_for_date(d)
    return JSONResponse({
        "date": d.isoformat(),
        "resolved": result["resolved"],
        "failed": result["failed"],
        "unmatched": result.get("unmatched", [])[:20],
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
    """Run a scan and fan out bets to all matching portfolios."""
    from evmax.portfolios import list_portfolios, log_portfolio_bet

    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    portfolio_ids: list[str] = body.get("portfolio_ids", [])
    date_from = body.get("date_from", "")
    date_to = body.get("date_to", "")

    portfolios = list_portfolios(active_only=True)
    if portfolio_ids:
        portfolios = [p for p in portfolios if p.id in portfolio_ids]

    if not portfolios:
        return JSONResponse({"error": "No active portfolios"}, status_code=400)

    all_sectors = sorted(set(s for p in portfolios for s in p.sectors))

    from evmax.agents.coordinator import AgentCoordinator
    coord = AgentCoordinator(sectors=all_sectors, bankroll=500, kelly_fraction=0.5)
    cycle = await coord.run_cycle()

    today_str = date.today().isoformat()
    tomorrow_str = (date.today() + timedelta(days=1)).isoformat()
    d_from = date_from or today_str
    d_to = date_to or tomorrow_str

    results = []
    for portfolio in portfolios:
        portfolio_gaps = []
        for g in cycle.top_gaps:
            if g.sector not in portfolio.sectors:
                continue
            if g.market_type == "map_handicap":
                continue

            event_date_str = str(g.event_date.astimezone().strftime("%Y-%m-%d") if g.event_date else "")
            if event_date_str < d_from or event_date_str > d_to:
                continue

            label_row = {
                "market_type": g.market_type or "",
                "yes_team": g.yes_team or "",
                "line": g.line,
                "prop_player_name": g.prop_player_name,
                "prop_stat_type": g.prop_stat_type,
                "prop_threshold": g.prop_threshold,
            }

            gap_dict = {
                "market_id": g.market_id or "",
                "event_id": g.event_id or "",
                "event_title": g.event_title or "",
                "yes_team": g.yes_team or "",
                "market_type": g.market_type or "",
                "display_label": _display_label_for_row(label_row),
                "line": g.line if g.line is None else float(g.line) if isinstance(g.line, (int, float)) else str(g.line),
                "sector": g.sector or "",
                "kalshi_yes_price": round(g.kalshi_yes_price, 4),
                "sharp_true_prob": round(getattr(g, 'sharp_true_prob', 0) or 0, 4),
                "blended_true_prob": round(g.blended_true_prob, 4),
                "ev_pct": round(g.ev_pct, 4),
                "kelly_fraction": round(g.kelly_fraction, 4),
                "event_date": event_date_str,
                "volume_usd": g.volume_usd or 0,
                "model_sources": g.model_sources or "",
                "scan_date": today_str,
            }

            log_portfolio_bet(
                portfolio_id=portfolio.id,
                gap=gap_dict,
                bankroll=portfolio.current_bankroll,
                kelly=portfolio.kelly_fraction,
            )
            portfolio_gaps.append(gap_dict)

        results.append({
            "portfolio_id": portfolio.id,
            "portfolio_name": portfolio.name,
            "gaps_logged": len(portfolio_gaps),
        })

    return JSONResponse({
        "portfolios_scanned": len(portfolios),
        "markets_fetched": cycle.markets_fetched,
        "markets_matched": cycle.markets_matched,
        "results": results,
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
        SELECT p.blended_true_prob, p.sharp_true_prob, o.outcome
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

    return JSONResponse({
        "weeks": weeks, "n": n, "wins": wins, "win_rate": round(100 * wins / n, 1),
        "brier_model": round(brier_model, 4), "brier_sharp": round(brier_sharp, 4),
        "calibration": calibration,
    })
