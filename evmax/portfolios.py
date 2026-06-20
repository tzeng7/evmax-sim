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
    "nba":      {"sectors": ["nba"],              "label": "NBA"},
    "wnba":     {"sectors": ["wnba"],             "label": "WNBA"},
    "soccer":   {"sectors": ["soccer"],           "label": "Soccer"},
    "worldcup": {"sectors": ["worldcup"],         "label": "World Cup"},
    "tennis":   {"sectors": ["tennis"],           "label": "Tennis"},
    "ncaab":    {"sectors": ["ncaab"],            "label": "NCAAB"},
    "baseball": {"sectors": ["baseball"],         "label": "Baseball"},
    "esports":  {"sectors": ["lol", "cs2"],       "label": "Esports"},
    "nhl":      {"sectors": ["nhl"],              "label": "NHL"},
}

# Player-prop portfolio groups — kept separate from SECTOR_GROUPS so
# game and prop performance can be evaluated independently. Bets get
# simulated from prop_observations (resolved + ev_pct >= 2%) since
# nba_props is in shadow mode and doesn't run through the live scan path.
PROP_SECTOR_GROUPS = {
    "nba_props":  {"sectors": ["nba_props"],  "label": "NBA Props"},
    "nfl_props":  {"sectors": ["nfl_props"],  "label": "NFL Props"},
}

# Single source of truth for which prop probability source the simulation
# trusts. Changes here also need to match validate_prop_pricing.py and the
# log_prop_from_sharp model_version tag in cli/commands/agents.py. Legacy
# rows (NULL or "pinnacle-v1") are preserved as historical record but
# excluded from forward-looking simulations because their sharp_prob came
# from the L15 model that bled −223u in shadow.
ANCHOR_MODEL_VERSION = "pinnacle-anchor-v1"


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
    conn = sqlite3.connect(str(DB_PATH), timeout=5.0)
    conn.row_factory = sqlite3.Row
    # WAL is set persistently by evmax/agents/cleanup/db.py::get_connection,
    # which shares this DB file; we just need busy_timeout to keep retries
    # working when the dashboard scan races a CLI write.
    conn.execute("PRAGMA busy_timeout=5000")
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
    """Create 3 scenarios × each sector group.

    Safe to call repeatedly: uses INSERT OR IGNORE so existing portfolios
    keep their current_bankroll intact. Only missing portfolios are inserted.
    """
    now = datetime.now(timezone.utc).isoformat()
    created: list[Portfolio] = []
    conn = _get_conn()
    for group_key, group in SECTOR_GROUPS.items():
        for scenario_name, params in SCENARIOS.items():
            pid = f"{group_key}_{scenario_name}"
            name = f"{group['label']} {scenario_name.capitalize()}"
            sectors_json = json.dumps(group["sectors"])
            bankroll = params["bankroll"]
            kelly = params["kelly"]
            conn.execute(
                """INSERT OR IGNORE INTO portfolios
                   (id, name, sectors, initial_bankroll, current_bankroll,
                    kelly_fraction, scenario, active, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)""",
                (pid, name, sectors_json, bankroll, bankroll, kelly,
                 scenario_name, now, now),
            )
            row = conn.execute(
                "SELECT * FROM portfolios WHERE id = ?", (pid,)
            ).fetchone()
            if row is not None:
                created.append(_portfolio_from_row(row))
    conn.commit()
    conn.close()
    return created


def create_prop_portfolios(backfill: bool = True) -> list[Portfolio]:
    """Create prop-only portfolios (Conservative/Moderate/Aggressive × each
    prop sector group), simulating performance from resolved prop_observations.

    Safe to call repeatedly:
      - INSERT OR IGNORE on the portfolios row → existing portfolios keep
        their current_bankroll (won't trigger the create_default trap).
      - INSERT OR IGNORE on portfolio_bets → re-running the backfill
        appends new resolved props but doesn't disturb existing rows.

    Pass backfill=False to create the empty portfolio rows without seeding
    the simulated bet ledger (e.g. when you want to wire a different feed).
    """
    now = datetime.now(timezone.utc).isoformat()
    created: list[Portfolio] = []
    conn = _get_conn()
    for group_key, group in PROP_SECTOR_GROUPS.items():
        for scenario_name, params in SCENARIOS.items():
            pid = f"{group_key}_{scenario_name}"
            name = f"{group['label']} {scenario_name.capitalize()}"
            sectors_json = json.dumps(group["sectors"])
            bankroll = params["bankroll"]
            kelly = params["kelly"]

            cur = conn.execute(
                """INSERT OR IGNORE INTO portfolios
                   (id, name, sectors, initial_bankroll, current_bankroll,
                    kelly_fraction, scenario, active, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)""",
                (pid, name, sectors_json, bankroll, bankroll, kelly,
                 scenario_name, now, now),
            )
            row = conn.execute(
                "SELECT * FROM portfolios WHERE id = ?", (pid,)
            ).fetchone()
            if row is not None:
                created.append(_portfolio_from_row(row))
    conn.commit()
    conn.close()

    if backfill:
        for p in created:
            backfill_portfolio_from_prop_observations(p.id)

    return created


def backfill_portfolio_from_prop_observations(portfolio_id: str) -> int:
    """Simulate Kelly-sized prop bets from resolved prop_observations.

    Walks every resolved row in prop_observations whose sector matches one
    of the portfolio's sectors with ev_pct >= 2% and synthesizes a portfolio
    bet sized at the portfolio's Kelly. Computes pnl from the row's outcome
    and updates portfolio.current_bankroll = initial + sum(pnl).

    Idempotent: portfolio_bets has a UNIQUE(portfolio_id, market_id) so
    re-runs don't double-log.
    """
    portfolio = get_portfolio(portfolio_id)
    if portfolio is None:
        return 0

    # Pull resolved props from the predictions DB. We import lazily to avoid
    # a circular dep — predictions.db lives next door but uses a different
    # connection object per its own schema migrations.
    pred_db = DB_PATH.parent / "predictions.db"
    if not pred_db.exists():
        return 0

    # Portfolio sectors are e.g. ["nba_props"], but prop_observations stores
    # the base sector ("nba") — the "_props" tag is the category key, not the
    # row's sector value. Strip the suffix when querying.
    base_sectors = [
        s[: -len("_props")] if s.endswith("_props") else s
        for s in portfolio.sectors
    ]
    placeholders = ",".join("?" * len(base_sectors))
    sql = f"""
        SELECT id, scan_date, event_date, sector, player_name, stat_type,
               line, kalshi_price, sharp_prob, ev_pct, outcome, market_id,
               event_id, event_title
        FROM prop_observations
        WHERE outcome IS NOT NULL
          AND sharp_prob IS NOT NULL
          AND ev_pct >= 0.02
          AND model_version = ?
          AND sector IN ({placeholders})
    """
    src = sqlite3.connect(str(pred_db))
    src.row_factory = sqlite3.Row
    rows = src.execute(sql, (ANCHOR_MODEL_VERSION, *base_sectors)).fetchall()
    src.close()

    bankroll = portfolio.initial_bankroll
    kelly_frac_setting = portfolio.kelly_fraction  # e.g. 0.25 / 0.50 / 1.00

    inserted = 0
    total_pnl = 0.0
    now = datetime.now(timezone.utc).isoformat()
    conn = _get_conn()
    for r in rows:
        d = dict(r)
        yes_price = d["kalshi_price"] or 0.0
        prob = d["sharp_prob"] or 0.0
        if yes_price <= 0 or yes_price >= 1 or prob <= yes_price:
            continue
        # Half-Kelly per market, then scaled by the portfolio's Kelly setting.
        # Cap at 5% of bankroll to mirror the live Kelly sizing.
        b = 1.0 / yes_price - 1.0
        f_full = (b * prob - (1.0 - prob)) / b
        market_kelly = max(0.0, min(0.05, f_full * 0.5))
        if market_kelly <= 0:
            continue
        stake = round(bankroll * kelly_frac_setting * market_kelly, 2)
        if stake <= 0:
            continue

        outcome = int(d["outcome"])
        pnl = round(stake * (1.0 / yes_price - 1.0), 2) if outcome == 1 else -stake
        total_pnl += pnl

        market_id = d["market_id"] or f"prop_obs::{d['id']}"
        display = f"{d.get('player_name')} {d.get('line')}+ {d.get('stat_type')}"

        cur = conn.execute(
            """INSERT OR IGNORE INTO portfolio_bets
               (portfolio_id, market_id, scan_date, event_id, sector, yes_team,
                market_type, event_title, event_date, display_label,
                kalshi_yes_price, sharp_true_prob, blended_true_prob,
                ev_pct, kelly_fraction, stake, bankroll_at_scan,
                line, volume_usd, model_sources,
                outcome, pnl, resolved_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                portfolio_id, market_id, d["scan_date"], d["event_id"],
                d["sector"], d.get("player_name"), "player_prop",
                d.get("event_title") or display,
                d["event_date"] or d["scan_date"], display,
                yes_price, prob, prob, d["ev_pct"], market_kelly,
                stake, bankroll, d.get("line"), None, "pinnacle-anchor-sim",
                outcome, pnl, now,
            ),
        )
        if cur.rowcount > 0:
            inserted += 1

    # Roll the simulated PnL into current_bankroll so the dashboard card
    # shows initial + cumulative simulated profit.
    conn.execute(
        "UPDATE portfolios SET current_bankroll = ?, updated_at = ? WHERE id = ?",
        (round(bankroll + total_pnl, 2), now, portfolio_id),
    )
    conn.commit()
    conn.close()
    return inserted


# (sector, market_type) pairs intentionally kept OUT of the portfolio
# simulation while still being scanned/logged (shadow) for data collection.
# Baseball totals are −CLV from a night-before-only workflow: the over-bias is
# selection on a noisy stale sharp line, not real edge, and the archive shows
# no exploitable open→close drift. See the baseball `notes:` in
# data/categories.yaml for the full diagnosis. Excluded 2026-06-02 per user.
_PORTFOLIO_EXCLUDED_MARKETS: set[tuple[str, str]] = {
    ("baseball", "total"),
}


def is_excluded_from_portfolio(
    sector: Optional[str], market_type: Optional[str]
) -> bool:
    """True when a (sector, market_type) is barred from portfolio simulation.

    Such gaps still scan and log to ev_predictions in shadow; they just don't
    enter any simulated portfolio's bet ledger or bankroll P&L.
    """
    return ((sector or "").lower(), (market_type or "").lower()) in _PORTFOLIO_EXCLUDED_MARKETS


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
            gap.get("ev_pct_raw", (gap.get("ev_pct") or 0) / 100),
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


def drop_voided_portfolio_bets() -> int:
    """Drop open portfolio bets whose underlying Kalshi market was voided.

    When a match is cancelled / walkover / withdrawal before a ball is
    played, Kalshi finalizes the binary market to a *scalar* fair-price
    refund. The cleanup resolver records this as ``ev_predictions.voided=1``
    and writes NO ``ev_outcomes`` row — there is no binary outcome to score.
    Those bets can never resolve through :func:`sync_portfolio_outcomes`'
    ``ev_outcomes`` JOIN, so without this they sit "open" forever in every
    portfolio's ledger (the symptom the user hit on Paul vs Mpetshi
    Perricard, 2026-06-08).

    A void is a refund: stake returned, no win/loss. An open portfolio bet
    has not yet touched ``current_bankroll`` (that only moves on resolution),
    so deleting the row is bankroll-neutral — and mirrors how voided rows are
    excluded from ``ev_predictions`` metrics (``WHERE voided = 0``).

    Returns the number of ``portfolio_bets`` rows dropped.
    """
    conn = _get_conn()
    # ev_predictions / ev_outcomes share this DB file (DB_PATH). Guard against
    # a fresh portfolio-only DB where the predictions schema isn't present yet.
    has_preds = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='ev_predictions'"
    ).fetchone()
    if not has_preds:
        conn.close()
        return 0

    ids = [
        r["id"]
        for r in conn.execute(
            """SELECT DISTINCT pb.id
               FROM portfolio_bets pb
               JOIN ev_predictions ep ON pb.market_id = ep.market_id
               LEFT JOIN ev_outcomes o ON pb.market_id = o.market_id
               WHERE pb.outcome IS NULL
                 AND ep.voided = 1
                 AND o.market_id IS NULL"""
        ).fetchall()
    ]
    if ids:
        conn.executemany(
            "DELETE FROM portfolio_bets WHERE id = ?", [(i,) for i in ids]
        )
        conn.commit()
    conn.close()
    return len(ids)


def sync_portfolio_outcomes() -> int:
    """Resolve portfolio bets from ev_outcomes. Returns count resolved/dropped.

    Three-stage:
      1. For any unresolved portfolio_bets with event_date <= today that
         don't yet have an ev_outcomes row, call the ESPN resolver first
         so ev_outcomes gets populated. This is what makes the web "Sync"
         button one-click: you don't have to hit "Resolve" first.
      2. JOIN portfolio_bets → ev_outcomes and copy outcome + compute pnl.
      3. Drop open bets whose Kalshi market voided (scalar refund) — they
         have no ev_outcomes row and would otherwise never leave the open
         pool. See :func:`drop_voided_portfolio_bets`.
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

    # Stage 3: drop voided markets that can never produce an ev_outcomes row.
    dropped = drop_voided_portfolio_bets()
    return count + dropped
