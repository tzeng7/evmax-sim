"""MaintenanceAgent — post-scan sanity checks + auto-remediation on ev_predictions.

Checks and fixes run after every scan:
  1. ev_below_threshold   — EV < 2% → DELETE row (filter bypass, corrupts P/L)
  2. kelly_above_cap      — Kelly > 5% → UPDATE to hard cap (clamp in-place)
  3. invalid_probability  — prob outside (0, 1) → DELETE row (EV untrustworthy)
  4. duplicate_event      — same event logged N times → keep highest EV, DELETE rest
  5. stale_resolved_market— row INSERTED AFTER its ev_outcomes resolution → DELETE
                            (original bet rows and placed bets are never touched)
  6. prob_divergence      — blended drifts > 50% from sharp → RESET to sharp_true_prob
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import structlog

from evmax.agents.cleanup.db import get_connection

logger = structlog.get_logger(__name__)

MIN_EV = 0.02           # 2% floor
MAX_KELLY = 0.05        # 5% hard cap
MAX_PROB_DRIFT = 0.50   # blended may not differ from sharp by > 50% relative


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MaintenanceIssue:
    level: str      # "error" | "warning"
    check: str      # short label for the failed check
    market_id: str
    event_id: str
    detail: str
    fix: str = ""   # description of the remediation applied


@dataclass
class MaintenanceReport:
    scan_date: str
    total_predictions: int
    issues: list[MaintenanceIssue] = field(default_factory=list)
    fixes_applied: int = 0

    @property
    def errors(self) -> list[MaintenanceIssue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[MaintenanceIssue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.issues

    def summary(self) -> str:
        n_err = len(self.errors)
        n_warn = len(self.warnings)
        if not self.issues:
            return f"[green]Maintenance OK[/green] — {self.total_predictions} predictions, no issues"
        parts = []
        if n_err:
            parts.append(f"[red]{n_err} error(s)[/red]")
        if n_warn:
            parts.append(f"[yellow]{n_warn} warning(s)[/yellow]")
        if self.fixes_applied:
            parts.append(f"[cyan]{self.fixes_applied} fix(es) applied[/cyan]")
        return f"Maintenance: {', '.join(parts)} across {self.total_predictions} predictions"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _delete(conn: sqlite3.Connection, market_id: str, scan_date: str) -> None:
    conn.execute(
        "DELETE FROM ev_predictions WHERE market_id = ? AND scan_date = ?",
        (market_id, scan_date),
    )


# ---------------------------------------------------------------------------
# Check + fix functions
# ---------------------------------------------------------------------------

def _check_rule_violations(
    rows: list, scan_date: str, conn: sqlite3.Connection, report: MaintenanceReport
) -> None:
    """EV threshold, Kelly cap, probability bounds."""
    for r in rows:
        market_id = r["market_id"]
        event_id = r["event_id"]

        # --- EV below threshold: delete the row entirely ---
        if r["ev_pct"] < MIN_EV:
            _delete(conn, market_id, scan_date)
            report.fixes_applied += 1
            report.issues.append(MaintenanceIssue(
                level="error",
                check="ev_below_threshold",
                market_id=market_id,
                event_id=event_id,
                detail=f"ev_pct={r['ev_pct']:.4f} < {MIN_EV:.2f}",
                fix="deleted row (sub-threshold bet should not be tracked)",
            ))
            continue  # row gone — skip further checks on it

        # --- Kelly above hard cap: clamp to MAX_KELLY ---
        if r["kelly_fraction"] > MAX_KELLY:
            conn.execute(
                "UPDATE ev_predictions SET kelly_fraction = ? WHERE market_id = ? AND scan_date = ?",
                (MAX_KELLY, market_id, scan_date),
            )
            report.fixes_applied += 1
            report.issues.append(MaintenanceIssue(
                level="error",
                check="kelly_above_cap",
                market_id=market_id,
                event_id=event_id,
                detail=f"kelly_fraction={r['kelly_fraction']:.4f} > {MAX_KELLY:.2f}",
                fix=f"clamped kelly_fraction to {MAX_KELLY:.2f}",
            ))

        # --- Invalid probability: delete the row ---
        bad_cols = [
            col for col in ("kalshi_yes_price", "sharp_true_prob", "blended_true_prob")
            if r[col] is None or not (0 < r[col] < 1)
        ]
        if bad_cols:
            _delete(conn, market_id, scan_date)
            report.fixes_applied += 1
            report.issues.append(MaintenanceIssue(
                level="error",
                check="invalid_probability",
                market_id=market_id,
                event_id=event_id,
                detail=f"invalid value(s) in: {', '.join(bad_cols)}",
                fix="deleted row (EV derived from invalid prob is untrustworthy)",
            ))


def _check_duplicate_events(
    rows: list, scan_date: str, conn: sqlite3.Connection, report: MaintenanceReport
) -> None:
    """Same (event_id, sector, yes_team) logged N times — keep highest EV, drop the rest."""
    # Group by deduplication key, preserving full row data
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r["event_id"], r["sector"], r["yes_team"])
        groups.setdefault(key, []).append(dict(r))

    for (event_id, sector, yes_team), group_rows in groups.items():
        if len(group_rows) <= 1:
            continue

        # Keep the row with the highest EV; on tie, keep the latest logged_at
        group_rows.sort(key=lambda x: (x["ev_pct"], x["logged_at"]), reverse=True)
        keeper = group_rows[0]
        to_delete = group_rows[1:]

        for dup in to_delete:
            _delete(conn, dup["market_id"], scan_date)
            report.fixes_applied += 1

        report.issues.append(MaintenanceIssue(
            level="warning",
            check="duplicate_event",
            market_id=keeper["market_id"],
            event_id=event_id,
            detail=(
                f"'{event_id}' ({sector}/{yes_team}) had {len(group_rows)} entries: "
                + ", ".join(r["market_id"] for r in group_rows)
            ),
            fix=(
                f"kept {keeper['market_id']} (ev={keeper['ev_pct']:.3f}), "
                f"deleted {len(to_delete)} duplicate(s)"
            ),
        ))


def _check_stale_markets(
    scan_date: str, conn: sqlite3.Connection, report: MaintenanceReport
) -> None:
    """Delete rows that were inserted AFTER their market resolved.

    Under the UNIQUE(market_id) freeze-on-first-insert schema (2026-06-29
    migration) a resolved market re-appearing in a scan is an INSERT OR IGNORE
    no-op, so a row whose scan_date is today AND whose outcome is resolved is
    almost always the ORIGINAL bet — the game simply resolved the same local
    day it was first scanned (WNBA/tennis/cs2/lol resolve evening-of). Deleting
    it orphans the ev_outcomes row and erases the bet from P&L (this happened:
    the 2026-07-04 Valkyries ML win, restored as id 12323).

    The only genuinely stale case left is a row RE-inserted after resolution
    (possible only if the original row was deleted first), so we require
    logged_at > resolved_at — both are UTC ``datetime('now')`` strings, so
    lexical comparison is chronological. Placed bets are never deleted, and
    legacy outcomes without a resolved_at stamp are left alone (can't prove
    the row post-dates resolution).
    """
    stale = conn.execute(
        """
        SELECT p.market_id, p.event_id
        FROM   ev_predictions p
        JOIN   ev_outcomes    o ON o.market_id = p.market_id
        WHERE  p.scan_date = ?
          AND  o.outcome   IS NOT NULL
          AND  o.resolved_at IS NOT NULL
          AND  p.logged_at > o.resolved_at
          AND  p.placed = 0
        """,
        (scan_date,),
    ).fetchall()

    for r in stale:
        _delete(conn, r["market_id"], scan_date)
        report.fixes_applied += 1
        report.issues.append(MaintenanceIssue(
            level="warning",
            check="stale_resolved_market",
            market_id=r["market_id"],
            event_id=r["event_id"],
            detail="row was inserted after its market resolved in ev_outcomes",
            fix="deleted row (post-resolution re-insert should not be re-bet)",
        ))


def _check_prob_divergence(
    rows: list, scan_date: str, conn: sqlite3.Connection, report: MaintenanceReport
) -> None:
    """blended_true_prob > 50% relative drift from sharp → reset to sharp and recalculate EV."""
    for r in rows:
        sharp = r["sharp_true_prob"]
        blended = r["blended_true_prob"]
        kalshi = r["kalshi_yes_price"]
        if not (sharp and sharp > 0):
            continue
        drift = abs(blended - sharp) / sharp
        if drift <= MAX_PROB_DRIFT:
            continue

        new_ev = (sharp / kalshi - 1.0) if kalshi and kalshi > 0 else 0.0
        if new_ev < 0:
            conn.execute(
                "DELETE FROM ev_predictions WHERE market_id = ? AND scan_date = ?",
                (r["market_id"], scan_date),
            )
            report.fixes_applied += 1
            report.issues.append(MaintenanceIssue(
                level="warning",
                check="prob_divergence",
                market_id=r["market_id"],
                event_id=r["event_id"],
                detail=f"blended={blended:.3f} vs sharp={sharp:.3f} ({drift:.0%} drift), negative EV after reset",
                fix="deleted row (negative EV after drift correction)",
            ))
        else:
            conn.execute(
                "UPDATE ev_predictions SET blended_true_prob = ?, ev_pct = ?, kelly_fraction = 0 "
                "WHERE market_id = ? AND scan_date = ?",
                (sharp, new_ev, r["market_id"], scan_date),
            )
            report.fixes_applied += 1
            report.issues.append(MaintenanceIssue(
                level="warning",
                check="prob_divergence",
                market_id=r["market_id"],
                event_id=r["event_id"],
                detail=f"blended={blended:.3f} vs sharp={sharp:.3f} ({drift:.0%} drift)",
                fix=f"reset blended_true_prob to {sharp:.3f}, ev_pct to {new_ev:.3f}",
            ))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_maintenance(scan_date: Optional[date] = None) -> MaintenanceReport:
    """Run all sanity checks + auto-remediation for a scan date.

    All fixes are committed in a single transaction. If any fix fails the
    entire transaction is rolled back to avoid partial state.
    """
    sd = (scan_date or date.today()).isoformat()

    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM ev_predictions WHERE scan_date = ?", (sd,)
        ).fetchall()

        report = MaintenanceReport(scan_date=sd, total_predictions=len(rows))

        if not rows:
            logger.info("maintenance_skipped", reason="no_predictions", date=sd)
            return report

        # Placed bets and rows whose market already resolved are historical
        # records — no check may delete or rewrite them. prob_divergence
        # deleted the resolved 2026-07-06 Sun ML winner (65% drift, negative
        # EV after reset) and orphaned its ev_outcomes row — the same failure
        # mode as the 2026-07-05 stale_resolved_market bug, one check over.
        # _check_stale_markets stays on the full table: post-resolution
        # re-inserts are exactly the resolved-market rows it must delete,
        # and it carries its own logged_at/placed guards.
        resolved = {
            r[0]
            for r in conn.execute(
                "SELECT market_id FROM ev_outcomes WHERE outcome IS NOT NULL"
            )
        }
        mutable_rows = [
            r for r in rows
            if not r["placed"] and r["market_id"] not in resolved
        ]

        # All check+fix functions share the same open connection/transaction
        _check_rule_violations(mutable_rows, sd, conn, report)
        _check_duplicate_events(mutable_rows, sd, conn, report)
        _check_stale_markets(sd, conn, report)
        _check_prob_divergence(mutable_rows, sd, conn, report)

        conn.commit()

    logger.info(
        "maintenance_complete",
        date=sd,
        total=len(rows),
        errors=len(report.errors),
        warnings=len(report.warnings),
        fixes=report.fixes_applied,
    )
    return report
