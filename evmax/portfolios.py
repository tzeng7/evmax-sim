"""Portfolio management — multiple simulated betting portfolios.

Each portfolio targets specific sectors with a defined bankroll and Kelly
fraction.  Three scenarios (conservative / moderate / aggressive) can be
created per sector to compare risk profiles side by side.

Portfolios are stored in SQLite alongside predictions so everything lives
in a single database.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from evmax.agents.cleanup.db import DB_PATH

PORTFOLIO_SCHEMA = """
CREATE TABLE IF NOT EXISTS portfolios (
    id                  TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    sectors             TEXT NOT NULL,           -- JSON array: ["nba", "soccer"]
    initial_bankroll    REAL NOT NULL,
    current_bankroll    REAL NOT NULL,
    kelly_fraction      REAL NOT NULL DEFAULT 0.5,
    scenario            TEXT NOT NULL DEFAULT 'moderate',  -- conservative / moderate / aggressive
    active              INTEGER NOT NULL DEFAULT 1,
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS portfolio_bets (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_id        TEXT NOT NULL REFERENCES portfolios(id),
    market_id           TEXT NOT NULL,
    scan_date           TEXT NOT NULL,
    event_id            TEXT,
    sector              TEXT NOT NULL,
    yes_team            TEXT,
    market_type         TEXT,
    event_title         TEXT,
    event_date          TEXT,
    display_label       TEXT,
    kalshi_yes_price    REAL,
    sharp_true_prob     REAL,
    blended_true_prob   REAL,
    ev_pct              REAL,
    kelly_fraction      REAL,
    stake               REAL,
    bankroll_at_scan    REAL,
    line                REAL,
    volume_usd          REAL,
    model_sources       TEXT,
    placed              INTEGER NOT NULL DEFAULT 0,
    placed_at           TEXT,
    placed_price        REAL,
    placed_stake        REAL,
    outcome             INTEGER,                -- 1=won, 0=lost, NULL=pending
    pnl                 REAL,
    resolved_at         TEXT,
    UNIQUE(portfolio_id, market_id)
);
"""

SCENARIOS = {
    "conservative": {"bankroll": 100, "kelly": 0.25},
    "moderate":     {"bankroll": 250, "kelly": 0.50},
    "aggressive":   {"bankroll": 500, "kelly": 1.00},
}

SECTOR_GROUPS = {
    "nba":     {"sectors": ["nba", "nba_props"], "label": "NBA"},
    "soccer":  {"sectors": ["soccer"],           "label": "Soccer"},
    "tennis":  {"sectors": ["tennis"],           "label": "Tennis"},
    "ncaab":   {"sectors": ["ncaab"],            "label": "NCAAB"},
    "baseball":{"sectors": ["baseball"],         "label": "Baseball"},
    "esports": {"sectors": ["lol", "cs2"],       "label": "Esports"},
    "nhl":     {"sectors": ["nhl"],              "label": "NHL"},
}


@dataclass
class Portfolio:
    id: str
    name: str
    sectors: list[str]
    initial_bankroll: float
    current_bankroll: float
    kelly_fraction: float
    scenario: str
    active: bool = True
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "sectors": self.sectors,
            "initial_bankroll": self.initial_bankroll,
            "current_bankroll": round(self.current_bankroll, 2),
            "kelly_fraction": self.kelly_fraction,
            "scenario": self.scenario,
            "active": self.active,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.executescript(PORTFOLIO_SCHEMA)
    return conn


def _portfolio_from_row(row: sqlite3.Row) -> Portfolio:
    return Portfolio(
        id=row["id"],
        name=row["name"],
        sectors=json.loads(row["sectors"]),
        initial_bankroll=row["initial_bankroll"],
        current_bankroll=row["current_bankroll"],
        kelly_fraction=row["kelly_fraction"],
        scenario=row["scenario"],
        active=bool(row["active"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def list_portfolios(active_only: bool = True) -> list[Portfolio]:
    conn = _get_conn()
    q = "SELECT * FROM portfolios"
    if active_only:
        q += " WHERE active = 1"
    q += " ORDER BY name"
    rows = conn.execute(q).fetchall()
    conn.close()
    return [_portfolio_from_row(r) for r in rows]


def get_portfolio(portfolio_id: str) -> Optional[Portfolio]:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM portfolios WHERE id = ?", (portfolio_id,)).fetchone()
    conn.close()
    return _portfolio_from_row(row) if row else None


def create_portfolio(
    portfolio_id: str,
    name: str,
    sectors: list[str],
    bankroll: float,
    kelly_fraction: float,
    scenario: str = "moderate",
) -> Portfolio:
    now = datetime.now(timezone.utc).isoformat()
    conn = _get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO portfolios
           (id, name, sectors, initial_bankroll, current_bankroll, kelly_fraction, scenario, active, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)""",
        (portfolio_id, name, json.dumps(sectors), bankroll, bankroll, kelly_fraction, scenario, now, now),
    )
    conn.commit()
    conn.close()
    return Portfolio(
        id=portfolio_id, name=name, sectors=sectors,
        initial_bankroll=bankroll, current_bankroll=bankroll,
        kelly_fraction=kelly_fraction, scenario=scenario,
        active=True, created_at=now, updated_at=now,
    )


def delete_portfolio(portfolio_id: str) -> bool:
    conn = _get_conn()
    cur = conn.execute("UPDATE portfolios SET active = 0 WHERE id = ?", (portfolio_id,))
    conn.commit()
    conn.close()
    return cur.rowcount > 0


def create_default_portfolios() -> list[Portfolio]:
    """Create 3 scenarios × each sector group."""
    created = []
    for group_key, group in SECTOR_GROUPS.items():
        for scenario_name, params in SCENARIOS.items():
            pid = f"{group_key}_{scenario_name}"
            name = f"{group['label']} {scenario_name.capitalize()}"
            p = create_portfolio(
                portfolio_id=pid,
                name=name,
                sectors=group["sectors"],
                bankroll=params["bankroll"],
                kelly_fraction=params["kelly"],
                scenario=scenario_name,
            )
            created.append(p)
    return created


def log_portfolio_bet(
    portfolio_id: str,
    gap: dict[str, Any],
    bankroll: float,
    kelly: float,
) -> None:
    """Log a single EV gap into a portfolio's bet ledger."""
    now = datetime.now(timezone.utc).isoformat()
    stake = round(bankroll * kelly * (gap.get("kelly_fraction") or gap.get("kelly_pct", 0) / 100), 2)
    scan_date = gap.get("scan_date") or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    conn = _get_conn()
    conn.execute(
        """INSERT OR IGNORE INTO portfolio_bets
           (portfolio_id, market_id, scan_date, event_id, sector, yes_team,
            market_type, event_title, event_date, display_label,
            kalshi_yes_price, sharp_true_prob, blended_true_prob,
            ev_pct, kelly_fraction, stake, bankroll_at_scan,
            line, volume_usd, model_sources)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            portfolio_id,
            gap.get("market_id"),
            scan_date,
            gap.get("event_id"),
            gap.get("sector"),
            gap.get("yes_team"),
            gap.get("market_type"),
            gap.get("event_title"),
            gap.get("event_date"),
            gap.get("display_label"),
            gap.get("kalshi_yes_price") or gap.get("kalshi_price"),
            gap.get("sharp_true_prob") or gap.get("true_prob"),
            gap.get("blended_true_prob") or gap.get("true_prob"),
            gap.get("ev_pct"),
            gap.get("kelly_fraction") or (gap.get("kelly_pct", 0) / 100),
            stake,
            bankroll,
            gap.get("line"),
            gap.get("volume_usd") or gap.get("volume"),
            gap.get("model_sources"),
        ),
    )
    conn.commit()
    conn.close()


def get_portfolio_bets(
    portfolio_id: str,
    status: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Return bets for a portfolio. status: 'open', 'settled', or None (all)."""
    conn = _get_conn()
    q = "SELECT * FROM portfolio_bets WHERE portfolio_id = ?"
    params: list[Any] = [portfolio_id]
    if status == "open":
        q += " AND outcome IS NULL"
    elif status == "settled":
        q += " AND outcome IS NOT NULL"
    q += " ORDER BY event_date DESC, id DESC"
    rows = conn.execute(q, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_portfolio_stats(portfolio_id: str) -> dict[str, Any]:
    """Compute summary stats for a portfolio."""
    conn = _get_conn()
    portfolio = conn.execute("SELECT * FROM portfolios WHERE id = ?", (portfolio_id,)).fetchone()
    if not portfolio:
        conn.close()
        return {}

    bets = conn.execute(
        "SELECT * FROM portfolio_bets WHERE portfolio_id = ?", (portfolio_id,)
    ).fetchall()
    conn.close()

    bets = [dict(b) for b in bets]
    settled = [b for b in bets if b["outcome"] is not None]
    open_bets = [b for b in bets if b["outcome"] is None]

    wins = sum(1 for b in settled if b["outcome"] == 1)
    losses = sum(1 for b in settled if b["outcome"] == 0)
    total_pnl = sum(b.get("pnl") or 0 for b in settled)
    total_staked = sum(b.get("stake") or 0 for b in settled)
    avg_ev = (sum(b.get("ev_pct") or 0 for b in bets) / len(bets) * 100) if bets else 0

    return {
        "portfolio_id": portfolio_id,
        "name": portfolio["name"],
        "scenario": portfolio["scenario"],
        "sectors": json.loads(portfolio["sectors"]),
        "initial_bankroll": portfolio["initial_bankroll"],
        "current_bankroll": portfolio["current_bankroll"],
        "total_bets": len(bets),
        "open_bets": len(open_bets),
        "settled_bets": len(settled),
        "wins": wins,
        "losses": losses,
        "win_rate": round(100 * wins / max(wins + losses, 1), 1),
        "total_pnl": round(total_pnl, 2),
        "total_staked": round(total_staked, 2),
        "roi_pct": round(100 * total_pnl / max(total_staked, 0.01), 2) if total_staked else 0,
        "avg_ev": round(avg_ev, 2),
    }


def resolve_portfolio_bet(portfolio_id: str, market_id: str, outcome: int) -> None:
    """Resolve a portfolio bet and update the portfolio balance."""
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()

    bet = conn.execute(
        "SELECT * FROM portfolio_bets WHERE portfolio_id = ? AND market_id = ?",
        (portfolio_id, market_id),
    ).fetchone()
    if not bet:
        conn.close()
        return

    stake = bet["stake"] or 0
    price = bet["placed_price"] or bet["kalshi_yes_price"] or 0.5
    if price <= 0 or price >= 1:
        pnl = 0.0
    elif outcome == 1:
        pnl = stake * (1.0 / price - 1.0)
    else:
        pnl = -stake

    conn.execute(
        "UPDATE portfolio_bets SET outcome = ?, pnl = ?, resolved_at = ? WHERE portfolio_id = ? AND market_id = ?",
        (outcome, round(pnl, 2), now, portfolio_id, market_id),
    )
    conn.execute(
        "UPDATE portfolios SET current_bankroll = current_bankroll + ?, updated_at = ? WHERE id = ?",
        (round(pnl, 2), now, portfolio_id),
    )
    conn.commit()
    conn.close()


def sync_portfolio_outcomes() -> int:
    """Resolve portfolio bets from ev_outcomes. Returns count resolved.

    Two-stage:
      1. For any unresolved portfolio_bets with event_date <= today that
         don't yet have an ev_outcomes row, call the ESPN resolver first
         so ev_outcomes gets populated. This is what makes the web "Sync"
         button one-click: you don't have to hit "Resolve" first.
      2. JOIN portfolio_bets → ev_outcomes and copy outcome + compute pnl.
    """
    import asyncio
    from datetime import date as _date

    from evmax.agents.cleanup.resolver import resolve_outcomes_for_date

    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    today_str = _date.today().isoformat()

    pending_dates = [
        r["event_date"] for r in conn.execute(
            """SELECT DISTINCT pb.event_date
               FROM portfolio_bets pb
               LEFT JOIN ev_outcomes o ON pb.market_id = o.market_id
               WHERE pb.outcome IS NULL
                 AND o.market_id IS NULL
                 AND pb.event_date IS NOT NULL
                 AND pb.event_date <= ?
               ORDER BY pb.event_date""",
            (today_str,),
        ).fetchall()
    ]
    conn.close()

    for d_str in pending_dates:
        try:
            asyncio.run(resolve_outcomes_for_date(_date.fromisoformat(d_str)))
        except Exception:
            # Swallow resolver errors — the JOIN below will still copy any
            # outcomes that did land, and unresolvable dates just stay open.
            pass

    conn = _get_conn()

    unresolved = conn.execute(
        """SELECT pb.portfolio_id, pb.market_id, pb.stake,
                  pb.placed_price, pb.kalshi_yes_price,
                  o.outcome
           FROM portfolio_bets pb
           JOIN ev_outcomes o ON pb.market_id = o.market_id
           WHERE pb.outcome IS NULL AND o.outcome IS NOT NULL"""
    ).fetchall()

    count = 0
    for row in unresolved:
        stake = row["stake"] or 0
        price = row["placed_price"] or row["kalshi_yes_price"] or 0.5
        if price <= 0 or price >= 1:
            pnl = 0.0
        elif row["outcome"] == 1:
            pnl = stake * (1.0 / price - 1.0)
        else:
            pnl = -stake

        conn.execute(
            "UPDATE portfolio_bets SET outcome = ?, pnl = ?, resolved_at = ? WHERE portfolio_id = ? AND market_id = ?",
            (row["outcome"], round(pnl, 2), now, row["portfolio_id"], row["market_id"]),
        )
        conn.execute(
            "UPDATE portfolios SET current_bankroll = current_bankroll + ?, updated_at = ? WHERE id = ?",
            (round(pnl, 2), now, row["portfolio_id"]),
        )
        count += 1

    if count:
        conn.commit()
    conn.close()
    return count
