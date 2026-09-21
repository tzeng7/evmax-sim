"""Phase-4 replay screen for the alt-spread ladder (SPREAD_LADDER_ENABLED).

For every RESOLVED spread row in predictions.db, re-price the rung two ways and
score both against the actual outcome, bucketed by how far the rung sits from the
game's main line (the TRUE distance, joined from the archived main line — not the
|line| proxy the old scaffold used):

  * CDF     — today's SpreadDistributionModel extrapolation off the main line
              (the logged blended_true_prob).
  * ladder  — the book's own devigged cover prob for that exact line, read from
              the archived Pinnacle alt-spread ladder (archived_sharp_odds rows
              with ::spread::<line> ids). n/a for a rung with no archived rung
              near its line.

The thesis to confirm before flipping the flag for a sector: in the DEEP-TAIL
bucket (|rung - main line| large) the CDF is worse (higher Brier) than the
ladder, and the ladder is at least as good everywhere. Near the main line the
two must agree (a sanity check — a divergence there is a bug).

The ladder column is populated only once the archive holds alt rungs. Capture
them WITHOUT changing live pricing via:

    evmax cleanup watch-listings -s nfl -m spread --capture-alt-spreads --once

Until a few declustered weeks of ladder capture accrue, the ladder column reads
"n/a (0 rungs)" and only the true-distance CDF calibration is available.

Brier is necessary, not sufficient. The live gate is CLV:
    evmax cleanup shadow clv nfl -m spread --sources-token sharp_ladder
Run this replay first to size the effect and catch sign bugs; promote on CLV.

Read-only. Runs on the prod box (needs predictions.db + archive.db). Prints a
per-sector, per-bucket table; writes nothing.

Usage:
    uv run python scripts/backtest_spread_ladder_replay.py [--sectors nfl,nba] [--days 400]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _base_game_id(event_id: str) -> str:
    """Reduce any spread event_id to the bare game key ``...::spread``.

    Main-line rows already carry the bare id; alt rungs carry
    ``...::spread::<line>`` — strip the trailing ``::<line>`` so main line, alt
    ladder and prediction rows all key on the same game.
    """
    marker = "::spread"
    idx = event_id.find(marker)
    if idx == -1:
        return event_id
    return event_id[: idx + len(marker)]


def _load_archive(sectors: set[str], archive_db: Path):
    """Return (main_lines, ladders) keyed by base game id.

    main_lines[game] = abs(main favorite line), latest pre-tip snapshot.
    ladders[game]    = {favorite_cover_point: true_prob_a}, latest per line.
    """
    main_lines: dict[str, float] = {}
    ladders: dict[str, dict[float, float]] = defaultdict(dict)
    if not archive_db.exists():
        return main_lines, ladders

    conn = sqlite3.connect(f"file:{archive_db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT sector, event_id, spread_line, true_prob_a, fetched_at
            FROM archived_sharp_odds
            WHERE spread_line IS NOT NULL
              AND event_id LIKE '%::spread%'
            ORDER BY fetched_at ASC
            """
        ).fetchall()
    except sqlite3.OperationalError:
        conn.close()
        return main_lines, ladders
    conn.close()

    # ORDER BY fetched_at ASC → later snapshots overwrite earlier, so each dict
    # ends holding the latest (closest-to-tip) archived price.
    for r in rows:
        if (r["sector"] or "").lower() not in sectors:
            continue
        game = _base_game_id(r["event_id"])
        line = r["spread_line"]
        is_alt = "::spread::" in r["event_id"]
        if is_alt:
            ladders[game][float(line)] = float(r["true_prob_a"])
        else:
            main_lines[game] = abs(float(line))
    return main_lines, ladders


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sectors", default="nfl,nba,ncaab,ncaaw",
                    help="Comma-separated sectors to replay.")
    ap.add_argument("--days", type=int, default=400)
    args = ap.parse_args()
    sectors = {s.strip().lower() for s in args.sectors.split(",") if s.strip()}

    from evmax.agents.cleanup.db import DB_PATH, get_connection
    from evmax.models_ml.spread_ladder_eval import (
        BUCKET_ORDER,
        brier,
        ladder_yes_prob,
        rung_distance_bucket,
    )
    from evmax.models_ml.spread_distribution import SPREAD_LADDER_LINE_TOLERANCE

    with get_connection() as conn:
        rows = list(conn.execute(
            """
            SELECT p.sector, p.event_id, p.line, p.blended_true_prob,
                   p.model_sources, o.outcome
            FROM ev_predictions p
            JOIN ev_outcomes o ON o.market_id = p.market_id
            WHERE p.market_type = 'spread'
              AND o.outcome IS NOT NULL
              AND p.line IS NOT NULL
              AND p.blended_true_prob IS NOT NULL
              AND p.scan_date >= date('now', ?)
            """,
            (f"-{args.days} days",),
        ))

    if not rows:
        print("No resolved spread rows in window — nothing to replay.\n"
              "(Expected on a fresh checkout; this harness runs on the prod DB.)")
        return 0

    archive_db = DB_PATH.parent / "archive.db"
    main_lines, ladders = _load_archive(sectors, archive_db)

    # (sector, bucket) -> {"cdf": [(p,o)], "ladder": [(p,o)]}
    agg: dict[tuple[str, str], dict[str, list[tuple[float, int]]]] = defaultdict(
        lambda: {"cdf": [], "ladder": []}
    )
    games_with_main = 0
    games_with_ladder = 0
    proxy_rows = 0

    for r in rows:
        sec = (r["sector"] or "").lower()
        if sec not in sectors:
            continue
        line = float(r["line"])
        outcome = int(r["outcome"])
        game = _base_game_id(r["event_id"])

        main_abs = main_lines.get(game)
        if main_abs is not None:
            dist = abs(abs(line) - main_abs)
        else:
            # No archived main line for this game — fall back to |line| as a
            # rough tier and mark the row so the coverage line stays honest.
            dist = abs(line)
            proxy_rows += 1
        bucket = rung_distance_bucket(dist)

        agg[(sec, bucket)]["cdf"].append((float(r["blended_true_prob"]), outcome))

        lprob = ladder_yes_prob(
            line, ladders.get(game, {}), SPREAD_LADDER_LINE_TOLERANCE
        )
        if lprob is not None:
            agg[(sec, bucket)]["ladder"].append((lprob, outcome))

    games_with_main = len(main_lines)
    games_with_ladder = len(ladders)

    print(f"{'sector':8} {'bucket':16} {'n':>5} {'CDF Brier':>10} "
          f"{'lad-n':>6} {'Lad Brier':>10}")
    print("-" * 60)
    for sec in sorted(sectors):
        printed_any = False
        for b in BUCKET_ORDER:
            cell = agg.get((sec, b))
            if not cell or not cell["cdf"]:
                continue
            printed_any = True
            cdf_b = brier(cell["cdf"])
            lad = cell["ladder"]
            if lad:
                lad_str = f"{brier(lad):>10.4f}"
            else:
                lad_str = f"{'n/a':>10}"
            print(f"{sec:8} {b:16} {len(cell['cdf']):>5} {cdf_b:>10.4f} "
                  f"{len(lad):>6} {lad_str}")
        if not printed_any:
            print(f"{sec:8} {'(no resolved spread rows in window)':40}")

    print()
    print(f"archive coverage: {games_with_main} games with a main line, "
          f"{games_with_ladder} games with an alt ladder.")
    if games_with_ladder == 0:
        print("Ladder column is n/a everywhere — no alt rungs archived yet. "
              "Capture them with:\n"
              "  evmax cleanup watch-listings -s nfl -m spread "
              "--capture-alt-spreads --once")
    if proxy_rows:
        print(f"note: {proxy_rows} rows had no archived main line; bucketed by "
              "|line| proxy (true rung-distance unavailable for those games).")
    print()
    print("Read: the deep-tail bucket is where the CDF should look worst. Flip")
    print("SPREAD_LADDER_ENABLED for a sector only when the ladder is no worse")
    print("everywhere AND wins the deep tail AND clears CLV.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
