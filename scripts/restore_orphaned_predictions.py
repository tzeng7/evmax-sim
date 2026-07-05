"""One-time restoration of ev_predictions rows deleted by the stale_resolved_market bug.

Between the 2026-06-29 UNIQUE(market_id) migration and the 2026-07-05 fix,
`_check_stale_markets` deleted the ORIGINAL prediction row whenever a game
resolved on the same local day it was scanned and a later scan ran
run_maintenance. That orphaned the ev_outcomes row and erased the bet from
P&L.

This script finds resolved game-level ev_outcomes rows with no matching
ev_predictions row and reconstructs the deleted prediction:

  - scan_date          — local date of resolved_at (the bug could only delete
                         rows whose scan_date equaled the deletion day, which
                         was the resolution's local day)
  - kalshi_yes_price   — earliest archive.db Kalshi snapshot on that local
                         date (no_price for ':no' market_ids); falls back to
                         the last snapshot before resolution
  - sharp/blended prob — from ev_outcomes (already side-adjusted at resolve)
  - ev_pct             — recomputed as blended/price − 1
  - kelly_fraction     — 0.0 (scan-time inputs like spread_pct are gone;
                         restored rows must not retroactively claim stakes)
  - mode               — current data/categories.yaml semantics via
                         evmax.modes.get_mode (incl. shadow_market_types)
  - logged_at          — the snapshot's fetched_at, so the row provably
                         pre-dates resolved_at and the (fixed) stale check
                         can never delete it again
  - model_version      — 'restored-orphan-<today>' so these rows are
                         identifiable/excludable in shadow scoring

Inserts use INSERT OR IGNORE (idempotent; already-restored rows like the
manually restored Valkyries ML id 12323 are naturally skipped by the
orphan query). Pre-cutoff orphans (April–May legacy artifacts) are excluded
via --since.

Usage:
    python scripts/restore_orphaned_predictions.py --dry-run   # preview
    python scripts/restore_orphaned_predictions.py             # apply
    python scripts/restore_orphaned_predictions.py --since 2026-06-30
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import date, datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

PREDICTIONS_DB = REPO_ROOT / "data" / "predictions.db"
ARCHIVE_DB = REPO_ROOT / "data" / "archive.db"


def _utc_str_to_local_date(ts: str) -> str:
    """'2026-07-05 00:32:41' (UTC) → local ISO date, e.g. '2026-07-04'."""
    dt = datetime.fromisoformat(ts.replace("T", " ").split("+")[0])
    return dt.replace(tzinfo=timezone.utc).astimezone().date().isoformat()


def _fetched_at_to_utc_sql(ts: str) -> str:
    """Archive ISO fetched_at → 'YYYY-MM-DD HH:MM:SS' UTC (logged_at format)."""
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def find_orphans(pred_conn: sqlite3.Connection, since: str) -> list[sqlite3.Row]:
    return pred_conn.execute(
        """
        SELECT o.*
        FROM   ev_outcomes o
        LEFT JOIN ev_predictions p ON p.market_id = o.market_id
        WHERE  p.market_id IS NULL
          AND  o.event_id NOT LIKE '%::prop::%'
          AND  o.outcome IS NOT NULL
          AND  o.resolved_at >= ?
        ORDER BY o.resolved_at
        """,
        (since,),
    ).fetchall()


def find_entry_snapshot(
    arch_conn: sqlite3.Connection, ticker: str, scan_date: str, resolved_at: str
) -> sqlite3.Row | None:
    """Earliest snapshot whose LOCAL date == scan_date; else last pre-resolution one."""
    rows = arch_conn.execute(
        "SELECT * FROM archived_kalshi_markets WHERE ticker = ? ORDER BY fetched_at",
        (ticker,),
    ).fetchall()
    same_day = [r for r in rows if _utc_str_to_local_date(r["fetched_at"]) == scan_date]
    if same_day:
        return same_day[0]
    resolved_utc = resolved_at.replace("T", " ").split("+")[0]
    before = [r for r in rows if _fetched_at_to_utc_sql(r["fetched_at"]) <= resolved_utc]
    return before[-1] if before else None


def build_row(
    orphan: sqlite3.Row, snap: sqlite3.Row, mode: str, model_version: str
) -> dict | None:
    market_id: str = orphan["market_id"]
    is_no = market_id.endswith(":no")

    if is_no:
        price = snap["no_price"]
        if price is None or not (0 < price < 1):
            price = 1.0 - snap["yes_price"]
    else:
        price = snap["yes_price"]
    if price is None or not (0 < price < 1):
        return None

    blended = orphan["blended_true_prob"]
    sharp = orphan["sharp_true_prob"]
    if blended is None or sharp is None:
        return None

    line = snap["line"]
    if is_no and line is not None:
        line = -line

    return {
        "logged_at": _fetched_at_to_utc_sql(snap["fetched_at"]),
        "scan_date": _utc_str_to_local_date(orphan["resolved_at"]),
        "market_id": market_id,
        "event_id": orphan["event_id"],
        "sector": orphan["sector"],
        "yes_team": orphan["yes_team"],
        "market_type": snap["market_type"],
        "event_title": snap["title"],
        "event_date": orphan["event_date"] or snap["event_date"],
        "kalshi_yes_price": price,
        "sharp_true_prob": sharp,
        "blended_true_prob": blended,
        "ev_pct": blended / price - 1.0,
        "kelly_fraction": 0.0,
        "volume_usd": snap["volume_usd"],
        "line": line,
        "mode": mode,
        "captured_yes_price": price,
        "model_version": model_version,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--since", default="2026-06-30",
                    help="only restore outcomes resolved at/after this UTC date "
                         "(default 2026-06-30 — the first post-migration deletions)")
    ap.add_argument("--dry-run", action="store_true", help="preview, no writes")
    ap.add_argument("--db", type=Path, default=PREDICTIONS_DB)
    ap.add_argument("--archive-db", type=Path, default=ARCHIVE_DB)
    args = ap.parse_args()

    from evmax.modes import get_mode

    pred_conn = sqlite3.connect(args.db)
    pred_conn.row_factory = sqlite3.Row
    arch_conn = sqlite3.connect(f"file:{args.archive_db}?mode=ro", uri=True)
    arch_conn.row_factory = sqlite3.Row

    orphans = find_orphans(pred_conn, args.since)
    print(f"Found {len(orphans)} game-level orphaned outcomes resolved >= {args.since}\n")

    model_version = f"restored-orphan-{date.today().isoformat()}"
    restored, skipped = 0, 0
    for o in orphans:
        ticker = o["market_id"].removeprefix("kalshi:").removesuffix(":no")
        scan_date = _utc_str_to_local_date(o["resolved_at"])
        snap = find_entry_snapshot(arch_conn, ticker, scan_date, o["resolved_at"])
        if snap is None:
            print(f"  SKIP {o['market_id']}: no archive snapshot")
            skipped += 1
            continue

        market_type = snap["market_type"]
        try:
            mode = get_mode(o["sector"], market_type)
        except Exception as e:
            print(f"  SKIP {o['market_id']}: mode resolution failed ({e})")
            skipped += 1
            continue
        if mode == "disabled":
            print(f"  SKIP {o['market_id']}: {o['sector']}/{market_type} is disabled")
            skipped += 1
            continue

        row = build_row(o, snap, mode, model_version)
        if row is None:
            print(f"  SKIP {o['market_id']}: unusable snapshot price or missing probs")
            skipped += 1
            continue

        won = "W" if o["outcome"] == 1 else "L"
        line_str = f" {row['line']:+g}" if row["line"] else ""
        print(
            f"  {'DRY ' if args.dry_run else ''}RESTORE {row['market_id']}\n"
            f"      {row['event_title'] or row['event_id']} — {row['yes_team']} "
            f"{market_type}{line_str} [{won}]\n"
            f"      scan_date={row['scan_date']} price={row['kalshi_yes_price']:.2f} "
            f"blended={row['blended_true_prob']:.3f} ev={row['ev_pct']:+.1%} "
            f"mode={mode} logged_at={row['logged_at']}"
        )
        if not args.dry_run:
            cols = ", ".join(row)
            ph = ", ".join(f":{c}" for c in row)
            cur = pred_conn.execute(
                f"INSERT OR IGNORE INTO ev_predictions ({cols}) VALUES ({ph})", row
            )
            if cur.rowcount:
                restored += 1
            else:
                skipped += 1
        else:
            restored += 1

    if not args.dry_run:
        pred_conn.commit()
    print(f"\n{'Would restore' if args.dry_run else 'Restored'} {restored}, skipped {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
