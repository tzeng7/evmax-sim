"""Read model for the dashboard "Resting orders" panel (record-only).

The maker workflow is: a maker-only play logs as ``mode='shadow'``, the user
rests a limit order on the exchange BY HAND, and records the fill with
``POST /api/fill`` once it executes. Nothing here talks to an exchange — the
panel only tracks what the user should have resting and warns when a game has
started with an order still unfilled (cancel it on the exchange: Kalshi trades
in-play, so a stale resting order is a free option for the other side).

Kickoff time is not stored as a column. It is reconstructed as
``logged_at + minutes_to_tipoff`` (both captured when the row was first logged;
rows are freeze-on-first-insert, so the pair is self-consistent).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from evmax.ev.maker_pilot import (
    MAKER_PILOT_KELLY_MULT,
    MAKER_PILOT_SECTORS,
    MAKER_PILOT_TOKEN,
    PILOT_CHECKPOINT_GAMES,
    PILOT_GATE_GAMES,
    checkpoint_verdict,
)

LOOKBACK_DAYS = 4

STATUS_RESTING = "resting"
STATUS_FILLED = "filled"
STATUS_PAST_KICKOFF = "past_kickoff"


def _parse_utc(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def kickoff_time(logged_at: Optional[str], minutes_to_tipoff: Optional[int]) -> Optional[datetime]:
    logged = _parse_utc(logged_at)
    if logged is None or minutes_to_tipoff is None:
        return None
    return logged + timedelta(minutes=int(minutes_to_tipoff))


def order_status(placed: int, kickoff: Optional[datetime], now: datetime) -> str:
    if placed:
        return STATUS_FILLED
    if kickoff is not None and now >= kickoff:
        return STATUS_PAST_KICKOFF
    return STATUS_RESTING


def list_resting_orders(
    conn: sqlite3.Connection, *, now: Optional[datetime] = None
) -> list[dict[str, Any]]:
    """Unresolved maker rows: unfilled shadow rows plus recorded maker fills."""
    now = now or datetime.now(timezone.utc)
    since = (now - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    rows = conn.execute(
        """SELECT p.market_id, p.sector, p.event_title, p.yes_team, p.market_type,
                  p.line, p.venue, p.mode, p.placed, p.placed_price, p.placed_stake,
                  p.kalshi_yes_price, p.captured_yes_price, p.maker_ev_pct,
                  p.kelly_fraction, p.model_sources, p.logged_at, p.minutes_to_tipoff,
                  p.placed_at
           FROM ev_predictions p
           LEFT JOIN ev_outcomes o ON o.market_id = p.market_id
           WHERE COALESCE(p.voided, 0) = 0
             AND o.outcome IS NULL
             AND p.scan_date >= ?
             AND (p.maker_ev_pct IS NOT NULL OR p.model_sources LIKE ?)
             AND ((p.mode = 'shadow' AND COALESCE(p.placed, 0) = 0)
                  OR (p.placed = 1 AND COALESCE(p.maker_fill, 0) = 1))
           ORDER BY p.logged_at""",
        (since, f"%{MAKER_PILOT_TOKEN}%"),
    ).fetchall()

    out: list[dict[str, Any]] = []
    for r in rows:
        kickoff = kickoff_time(r["logged_at"], r["minutes_to_tipoff"])
        logged = _parse_utc(r["logged_at"])
        placed = int(r["placed"] or 0)
        out.append({
            "market_id": r["market_id"],
            "sector": r["sector"],
            "event_title": r["event_title"],
            "yes_team": r["yes_team"],
            "market_type": r["market_type"],
            "line": r["line"],
            "venue": r["venue"] or "kalshi",
            "pilot": MAKER_PILOT_TOKEN in (r["model_sources"] or ""),
            "status": order_status(placed, kickoff, now),
            "kickoff_at": kickoff.isoformat() if kickoff else None,
            "logged_at": logged.isoformat() if logged else None,
            "age_minutes": int((now - logged).total_seconds() // 60) if logged else None,
            "ask_price": r["captured_yes_price"] or r["kalshi_yes_price"],
            "maker_ev_pct": r["maker_ev_pct"],
            "kelly_fraction": r["kelly_fraction"],
            "placed_price": r["placed_price"],
            "placed_stake": r["placed_stake"],
        })
    return out


def fill_rate(conn: sqlite3.Connection, sector: str) -> tuple[int, int, int]:
    """(pilot candidates on resolved games, of which filled, candidates still open)."""
    rows = conn.execute(
        """SELECT p.market_id, MAX(p.placed) AS placed,
                  MAX(o.outcome IS NOT NULL) AS resolved
           FROM ev_predictions p
           LEFT JOIN ev_outcomes o ON o.market_id = p.market_id
           WHERE p.sector = ? AND p.model_sources LIKE ?
             AND COALESCE(p.voided, 0) = 0
           GROUP BY p.market_id""",
        (sector, f"%{MAKER_PILOT_TOKEN}%"),
    ).fetchall()
    resolved = [r for r in rows if r["resolved"]]
    return len(resolved), sum(1 for r in resolved if r["placed"]), len(rows) - len(resolved)


def pilot_readout(
    conn: sqlite3.Connection, sector: str = "nfl", max_staleness_h: Optional[float] = 3.0
) -> dict[str, Any]:
    """Checkpoint readout shared by scripts/maker_pilot_readout.py and the dashboard."""
    # Lazy import: shadow.py pulls in the CLI stack.
    from evmax.cli.commands.shadow import _fetch_clv_rows

    n_res, n_filled, n_open = fill_rate(conn, sector)
    rows, excluded = _fetch_clv_rows(
        sector, market_type="spread", mode="live", venue="kalshi",
        sources_token=MAKER_PILOT_TOKEN, max_staleness_h=max_staleness_h,
    )
    verdict = checkpoint_verdict([dict(r) for r in rows])
    return {
        "sector": sector,
        "enabled": sector in MAKER_PILOT_SECTORS,
        "kelly_mult": MAKER_PILOT_KELLY_MULT,
        "checkpoint_games": PILOT_CHECKPOINT_GAMES,
        "gate_games": PILOT_GATE_GAMES,
        "candidates_resolved": n_res,
        "candidates_filled": n_filled,
        "candidates_open": n_open,
        "fill_rate": (n_filled / n_res) if n_res else None,
        "stale_close_excluded": excluded,
        **verdict,
    }
