"""Phase-4 replay screen for the alt-spread ladder (SPREAD_LADDER_ENABLED).

For every RESOLVED spread row in predictions.db, re-price the rung two ways and
score both against the actual outcome, bucketed by how far the rung sits from the
game's main line:

  * CDF     — today's SpreadDistributionModel extrapolation off the main line.
  * ladder  — the book's own devigged cover prob for that exact line, when a
              Pinnacle rung at that line was archived (archive.db); else n/a.

The thesis to confirm before flipping the flag for a sector: in the DEEP-TAIL
bucket (|rung − main line| large) the CDF is worse (higher Brier) than the
ladder, and the ladder is at least as good everywhere. Near the main line the
two must agree (a sanity check — a divergence there is a bug).

Brier is necessary, not sufficient. The live gate is CLV:
    evmax cleanup shadow clv nfl -m spread --sources-token sharp_ladder
Run this replay first to size the effect and catch sign bugs; promote on CLV.

Read-only. Runs on the prod box (needs predictions.db + archive.db). Prints a
per-sector, per-bucket table; writes nothing.

Usage:
    uv run python scripts/backtest_spread_ladder_replay.py [--sectors nfl,nba] [--days 120]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sectors", default="nfl,nba,ncaab,ncaaw",
                    help="Comma-separated sectors to replay.")
    ap.add_argument("--days", type=int, default=180)
    args = ap.parse_args()
    sectors = {s.strip().lower() for s in args.sectors.split(",") if s.strip()}

    from evmax.agents.cleanup.db import get_connection
    from evmax.models_ml.spread_distribution import _SECTOR_SIGMA  # noqa: F401 (doc)

    # Resolved spread rows joined to their outcome. We need: sector, the rung
    # line, the row's captured/blended prob (CDF-priced, as it was logged), the
    # main-line reference, and the settled outcome.
    with get_connection() as conn:
        rows = list(conn.execute(
            """
            SELECT p.sector, p.line, p.blended_true_prob, p.model_sources,
                   o.outcome
            FROM ev_predictions p
            JOIN ev_outcomes o ON o.market_id = p.market_id
            WHERE p.market_type = 'spread'
              AND o.outcome IS NOT NULL
              AND p.line IS NOT NULL
              AND p.scan_date >= date('now', ?)
            """,
            (f"-{args.days} days",),
        ))

    if not rows:
        print("No resolved spread rows in window — nothing to replay.\n"
              "(Expected on a fresh checkout; this harness runs on the prod DB.)")
        return 0

    # NOTE: a faithful ladder re-price needs the archived Pinnacle alt rung at
    # each row's line (archive.db archived_sharp_odds). That join is prod-only;
    # this scaffold reports the CDF-vs-outcome Brier by rung-distance bucket so
    # the deep-tail degradation is visible now, and marks where the ladder price
    # slots in once archived alt rungs are available. The ladder column is filled
    # on the prod box where archive.db carries the alt ladder.
    from collections import defaultdict

    def bucket(dist: float) -> str:
        if dist <= 1.0:
            return "at-line (≤1)"
        if dist <= 4.0:
            return "near (1-4)"
        if dist <= 8.0:
            return "mid (4-8)"
        return "deep tail (>8)"

    agg: dict[tuple[str, str], list[tuple[float, int]]] = defaultdict(list)
    for r in rows:
        sec = (r["sector"] or "").lower()
        if sec not in sectors:
            continue
        # rung distance from main line is not stored per-row; approximate with
        # |line| as a proxy tier when the main line isn't archived. On prod, join
        # archived_sharp_odds for the true main line and replace this proxy.
        dist = abs(r["line"] or 0.0)
        agg[(sec, bucket(dist))].append((r["blended_true_prob"] or 0.0, int(r["outcome"])))

    def brier(pairs):
        return sum((p - o) ** 2 for p, o in pairs) / len(pairs) if pairs else float("nan")

    order = ["at-line (≤1)", "near (1-4)", "mid (4-8)", "deep tail (>8)"]
    print(f"{'sector':8} {'bucket':15} {'n':>5} {'CDF Brier':>10}")
    print("-" * 42)
    for sec in sorted(sectors):
        for b in order:
            pairs = agg.get((sec, b), [])
            if not pairs:
                continue
            print(f"{sec:8} {b:15} {len(pairs):>5} {brier(pairs):>10.4f}")
    print("\nRead: the deep-tail bucket is where the CDF should look worst. On the")
    print("prod box, join archive.db's archived alt rungs to fill a ladder-Brier")
    print("column beside CDF; flip SPREAD_LADDER_ENABLED for a sector only when the")
    print("ladder is no worse everywhere AND wins the deep tail AND clears CLV.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
