"""Classify zero-CLV WNBA spread rows: genuine-flat market vs stale-capture gap.

36% of the clean Kalshi lay rows in the WNBA spread CLV gate show exact
kalshi_clv_pct = 0.0 (close price == entry price). Two very different causes:

  genuine-flat    : the archive HAS a snapshot near the T-30 close target and
                    the price genuinely didn't move between entry and close.
                    A real, correctly-measured zero.
  stale-capture   : the archive's last snapshot before the T-30 cutoff sits
                    many hours earlier (a watch-closes/launchd capture gap), so
                    "close" is a stale mid-day read that happens to equal entry.
                    A zero that measures the WRONG moment.

For each zero-CLV row this asks the archiver for the STALENESS of the snapshot
get_kalshi_close_price() would pick (target T-minutes_before minus the selected
snapshot's fetched_at, in hours) via the same helper the shadow CLV gate uses
(DataArchiver.get_kalshi_close_staleness_h). A large staleness = stale-capture.

Read-only. Points at the MAIN repo DBs by default (the worktree copies are
empty); override with --predictions-db / --archive-db.

Run:  .venv/bin/python scripts/diagnose_wnba_clv_zeros.py [--market-type spread]
      [--venue kalshi] [--stale-threshold-h 3] [--minutes-before 30]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Default to the canonical (non-worktree) checkout's live data.
MAIN_REPO = Path("/Users/ktzeng/Projects/evmax")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sector", default="wnba")
    ap.add_argument("--market-type", default="spread")
    ap.add_argument("--venue", default="kalshi",
                    help="Only classify this venue's rows (default kalshi).")
    ap.add_argument("--minutes-before", type=int, default=30)
    ap.add_argument("--stale-threshold-h", type=float, default=3.0,
                    help="Selected close snapshot older than this vs the T-N "
                         "target = stale-capture; else genuine-flat.")
    ap.add_argument("--predictions-db", default=str(MAIN_REPO / "data" / "predictions.db"))
    ap.add_argument("--archive-db", default=str(MAIN_REPO / "data" / "archive.db"))
    args = ap.parse_args()

    # Point the archiver at the requested archive.db, then reuse its close-
    # staleness helper (single source of truth for the snapshot selection).
    import evmax.archiver as archiver_mod
    archiver_mod.DB_PATH = Path(args.archive_db)
    from evmax.agents.cleanup.resolver import close_lookup_ticker
    archiver = archiver_mod.DataArchiver()

    pred = sqlite3.connect(args.predictions_db)
    pred.row_factory = sqlite3.Row
    rows = pred.execute(
        """SELECT p.market_id, p.event_id, p.kalshi_clv_pct, p.line
           FROM ev_predictions p
           INNER JOIN ev_outcomes o ON p.market_id = o.market_id
           WHERE p.sector = ? AND p.market_type = ? AND p.venue = ?
             AND o.outcome IS NOT NULL
             AND p.kalshi_clv_pct IS NOT NULL AND p.kalshi_clv_pct = 0.0""",
        (args.sector, args.market_type, args.venue),
    ).fetchall()

    print(f"zero-CLV {args.sector}/{args.market_type} venue={args.venue}: "
          f"n={len(rows)}")
    if not rows:
        return

    genuine = stale = no_anchor = 0
    detail: list[tuple[float, str]] = []
    for r in rows:
        ticker, _is_no = close_lookup_ticker(r["market_id"])
        staleness = (
            archiver.get_kalshi_close_staleness_h(
                ticker, r["event_id"], minutes_before=args.minutes_before
            )
            if ticker else None
        )
        if staleness is None:
            no_anchor += 1
            detail.append((float("inf"),
                           f"{r['market_id']:<48} NO close snapshot / tip anchor"))
            continue
        is_stale = staleness > args.stale_threshold_h
        stale += is_stale
        genuine += not is_stale
        tag = "STALE " if is_stale else "flat  "
        detail.append((staleness,
                       f"{r['market_id']:<48} staleness={staleness:5.1f}h  {tag}"))

    detail.sort(key=lambda t: -t[0])
    for _s, line in detail:
        print(" ", line)

    n_class = genuine + stale
    print(f"\nclassified n={n_class}  (no-anchor excluded: {no_anchor})")
    if n_class:
        print(f"  stale-capture : {stale:3d}  ({stale/n_class*100:.0f}%)  "
              f"(close snapshot > {args.stale_threshold_h:.0f}h before T-{args.minutes_before})")
        print(f"  genuine-flat  : {genuine:3d}  ({genuine/n_class*100:.0f}%)")


if __name__ == "__main__":
    main()
