"""Relabel legacy derived player-prop rungs in archive.db.

Pinnacle posts ONE line per (player, stat). The scanner re-lines that anchor to
every Kalshi threshold with evmax/ev/prop_pricing.py and, before 2026-10-09,
archived each re-lined rung as ``book='pinnacle'`` — so a backtest reading
``archived_sharp_odds`` saw evmax's own model output as if it were a Pinnacle
alternate line (the tell: identical outcome decimals on every rung of a
player). New code files those rungs under ``'<book>_derived'`` with
``derived = 1`` and archives the real anchors as ``'pinnacle'`` / ``derived = 0``
(see evmax/archiver.py).

Before the change only the scanner wrote prop rows, and it wrote ONLY re-lined
rungs, so every prop row with ``derived IS NULL`` is a derived rung. This script
gives those rows the new label:

    book    → book || '_derived'
    derived → 1

``anchor_line`` / ``anchor_prob_over`` stay NULL on relabelled rows — that pair
(``derived = 1 AND anchor_line IS NULL``) is how ``--revert`` finds exactly the
rows this script touched. (The anchor is still recoverable offline: the stored
decimals devig to the anchor prob, and the pricing family at the time inverts
the rung prob to the anchor line — see scripts/fit_nfl_prop_dispersion.py.)

Idempotent and batched by row id so the write lock is held briefly — the
archive is written by launchd services (watch-closes / watch-listings) while
this runs. No backup is written (the archive is several GB); ``--revert`` undoes
the relabel. Run it after the 2026-10-09 change is deployed; rows a pre-change
scanner writes afterwards are picked up by a re-run.

    python scripts/relabel_derived_prop_rungs.py                  # dry-run counts
    python scripts/relabel_derived_prop_rungs.py --apply          # relabel
    python scripts/relabel_derived_prop_rungs.py --revert --apply # undo
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evmax.archiver import _MIGRATIONS, DERIVED_BOOK_SUFFIX  # noqa: E402

DEFAULT_DB = REPO_ROOT / "data" / "archive.db"

_LEGACY = (
    "prop_player_name IS NOT NULL AND derived IS NULL "
    f"AND book NOT LIKE '%{DERIVED_BOOK_SUFFIX}'"
)
_RELABELLED = (
    "prop_player_name IS NOT NULL AND derived = 1 AND anchor_line IS NULL "
    f"AND book LIKE '%{DERIVED_BOOK_SUFFIX}'"
)


def has_derived_columns(conn: sqlite3.Connection) -> bool:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(archived_sharp_odds)")}
    return {"derived", "anchor_line", "anchor_prob_over"} <= cols


def ensure_columns(conn: sqlite3.Connection) -> None:
    """Apply the archiver's archived_sharp_odds migrations (no-op when present)."""
    for stmt in _MIGRATIONS:
        if "ALTER TABLE archived_sharp_odds" not in stmt:
            continue
        try:
            conn.execute(stmt)
            conn.commit()
        except sqlite3.OperationalError:
            pass


def plan(conn: sqlite3.Connection, revert: bool = False) -> list[tuple[str, str, int]]:
    """(sector, book, rows) the run would change."""
    if not has_derived_columns(conn):
        if revert:
            return []
        where = "prop_player_name IS NOT NULL"  # pre-migration: every prop row is legacy
    else:
        where = _RELABELLED if revert else _LEGACY
    return [
        (r[0], r[1], r[2])
        for r in conn.execute(
            f"SELECT sector, book, COUNT(*) FROM archived_sharp_odds WHERE {where} "
            "GROUP BY sector, book ORDER BY sector, book"
        )
    ]


def apply(conn: sqlite3.Connection, revert: bool = False, batch: int = 50_000) -> int:
    """Relabel (or revert) in id-range batches, one short transaction each."""
    ensure_columns(conn)
    lo, hi = conn.execute("SELECT MIN(id), MAX(id) FROM archived_sharp_odds").fetchone()
    if lo is None:
        return 0
    if revert:
        sql = (
            "UPDATE OR IGNORE archived_sharp_odds "
            f"SET book = substr(book, 1, length(book) - {len(DERIVED_BOOK_SUFFIX)}), "
            f"derived = NULL WHERE id >= ? AND id < ? AND {_RELABELLED}"
        )
    else:
        sql = (
            "UPDATE OR IGNORE archived_sharp_odds "
            f"SET book = book || '{DERIVED_BOOK_SUFFIX}', derived = 1 "
            f"WHERE id >= ? AND id < ? AND {_LEGACY}"
        )
    changed = 0
    for start in range(lo, hi + 1, batch):
        cur = conn.execute(sql, (start, start + batch))
        conn.commit()
        changed += cur.rowcount
    return changed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", type=Path, default=DEFAULT_DB, help="archive.db path")
    ap.add_argument("--apply", action="store_true", help="write (default: dry-run)")
    ap.add_argument("--revert", action="store_true", help="undo a previous relabel")
    ap.add_argument("--batch", type=int, default=50_000, help="ids per transaction")
    args = ap.parse_args(argv)

    if not args.db.exists():
        print(f"no archive at {args.db}")
        return 1
    if args.apply:
        conn = sqlite3.connect(str(args.db), timeout=30.0)
        conn.execute("PRAGMA busy_timeout=30000")
    else:
        conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        rows = plan(conn, revert=args.revert)
        verb = "revert" if args.revert else "relabel"
        total = sum(n for _, _, n in rows)
        print(f"{verb} plan ({args.db}):")
        for sector, book, n in rows:
            target = (
                book[: -len(DERIVED_BOOK_SUFFIX)] if args.revert else book + DERIVED_BOOK_SUFFIX
            )
            print(f"  {sector:<10} {book:<20} → {target:<20} {n:>9,}")
        print(f"  total {total:,}")
        if not args.apply:
            print("dry-run: nothing written (pass --apply)")
            return 0
        changed = apply(conn, revert=args.revert, batch=args.batch)
        left = sum(n for _, _, n in plan(conn, revert=args.revert))
        print(f"{verb}ed {changed:,} rows; {left:,} still match")
        return 0 if left == 0 else 2
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
