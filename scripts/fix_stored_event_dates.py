"""One-shot correction script for event_date rows stored under the
pre-fix UTC convention.

For every ev_predictions / portfolio_bets row in a US sector whose
market_id encodes a Kalshi ticker date (`YYMONDD`), compare the
ticker-encoded date against the stored event_date. When they
disagree, rewrite event_date (and event_id's date segment when it
matches the old event_date exactly) to the ticker date.

Run manually:
    python scripts/fix_stored_event_dates.py --dry-run     # preview
    python scripts/fix_stored_event_dates.py --apply       # commit
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "predictions.db"

US_SECTORS = {"nba", "nfl", "mlb", "nhl", "ncaab", "ncaaw"}
_MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}
_TICKER_DATE_RE = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})")


def ticker_date(market_id: str) -> str | None:
    m = _TICKER_DATE_RE.search(market_id.upper())
    if not m:
        return None
    try:
        year = 2000 + int(m.group(1))
        month = _MONTH_MAP[m.group(2)]
        day = int(m.group(3))
        return f"{year:04d}-{month:02d}-{day:02d}"
    except (ValueError, KeyError):
        return None


def fix_table(conn: sqlite3.Connection, table: str, dry_run: bool) -> int:
    rows = conn.execute(
        f"SELECT id, market_id, event_id, sector, event_date FROM {table}"
    ).fetchall()
    changes: list[tuple[str, str, str, int]] = []
    for r in rows:
        if r["sector"] not in US_SECTORS:
            continue
        td = ticker_date(r["market_id"])
        if not td or td == r["event_date"]:
            continue
        old_eid = r["event_id"] or ""
        new_eid = old_eid
        if r["event_date"] and r["event_date"] in old_eid:
            new_eid = old_eid.replace(r["event_date"], td, 1)
        changes.append((td, new_eid, r["market_id"], r["id"]))
        print(
            f"  [{table}#{r['id']}] {r['market_id']}\n"
            f"    event_date: {r['event_date']} -> {td}\n"
            f"    event_id:   {old_eid} -> {new_eid}"
        )

    if changes and not dry_run:
        conn.executemany(
            f"UPDATE {table} SET event_date = ?, event_id = ? WHERE id = ?",
            [(td, eid, rid) for td, eid, _mid, rid in changes],
        )
        conn.commit()
    return len(changes)


def main() -> int:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = p.parse_args()

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    total = 0
    for table in ("ev_predictions", "portfolio_bets"):
        has = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
            (table,),
        ).fetchone()
        if not has:
            continue
        print(f"=== {table} ===")
        total += fix_table(conn, table, dry_run=args.dry_run)

    conn.close()
    mode = "dry-run" if args.dry_run else "applied"
    print(f"\n{mode}: {total} row(s) needed correction")
    return 0


if __name__ == "__main__":
    sys.exit(main())
