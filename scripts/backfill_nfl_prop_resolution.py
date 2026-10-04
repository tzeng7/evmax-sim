"""One-off backfill: resolve pending NFL prop_observations beyond the daily window.

The daily resolve step only looks back PROP_RESOLVE_LOOKBACK_DAYS (3) days, so the
NFL rows logged before the NFL resolver existed fall outside it. This re-runs the
same resolver with a wide lookback. Idempotent: only rows with outcome IS NULL are
touched. Dry-run by default (rolls the transaction back and prints the counts).

    python scripts/backfill_nfl_prop_resolution.py [--days 45] [--apply] [--db PATH]
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import date

from evmax.agents.cleanup.resolver import _resolve_prop_observations


def _counts(conn: sqlite3.Connection) -> tuple[int, int]:
    return conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(outcome IS NOT NULL), 0) "
        "FROM prop_observations WHERE sector = 'nfl'"
    ).fetchone()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=45, help="lookback window in days")
    ap.add_argument("--apply", action="store_true", help="commit (default: dry run)")
    ap.add_argument("--db", default=None, help="predictions.db path (default: live DB)")
    args = ap.parse_args()

    if args.db:
        conn = sqlite3.connect(args.db)
        conn.row_factory = sqlite3.Row
    else:
        from evmax.agents.cleanup.db import get_connection

        conn = get_connection()

    total, before = _counts(conn)
    n = _resolve_prop_observations(conn, date.today(), lookback_days=args.days)
    _, after = _counts(conn)
    print(f"nfl prop rows: {total} total | resolved before {before} -> after {after} (+{n} this pass)")
    if args.apply:
        conn.commit()
        print("committed")
    else:
        conn.rollback()
        print("dry run — rolled back (pass --apply to commit)")


if __name__ == "__main__":
    main()
