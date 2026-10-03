"""CLV by entry timing (minutes-to-tipoff bucket), declustered by game.

Question: for a given sector / market type / venue, how does entry->close CLV
change with how long before tip the row was first logged?

Read-only. Works on a COPY of predictions.db (pass --db) because the CLI
migrates on connect. Uses `kalshi_clv_pct` (percentage points) and
`minutes_to_tipoff` (frozen at first insert), so a row's bucket is its ENTRY
timing. Rows with minutes_to_tipoff <= 0 (at/after tip) are dropped, cancelled
voids are dropped, `stale_reverted` voids are KEPT (dropping them biases CLV up).

Declustering: each (bucket, game) contributes ONE value (mean of its rungs),
so n is independent games (game_key), not alt-spread rungs. A game can appear in several
buckets (it may be logged at different times), but never twice in one bucket.

Run:
  .venv/bin/python scripts/clv_by_entry_timing.py --db /path/to/copy.db
  .venv/bin/python scripts/clv_by_entry_timing.py --sectors nfl,ncaaf --by-venue
"""
from __future__ import annotations

import argparse
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

REPO_ROOT = Path(__file__).resolve().parent.parent

# (label, lo_minutes inclusive, hi_minutes exclusive)
BUCKETS = [
    ("<1h", 1, 60),
    ("1-3h", 60, 180),
    ("3-6h", 180, 360),
    ("6-12h", 360, 720),
    ("12-24h", 720, 1440),
    ("24h+", 1440, 10**9),
]


def game_key(event_id: str) -> str:
    """`nfl::2026-09-13::lions_vs_saints::spread::-3.5` -> `nfl::2026-09-13::lions_vs_saints`.

    Alt-ladder rungs and market types all collapse onto one game, so n is games.
    """
    return "::".join(event_id.split("::")[:3])


def bucket_of(m: int) -> str | None:
    for label, lo, hi in BUCKETS:
        if lo <= m < hi:
            return label
    return None


def load(db: Path, sectors: list[str], since: str | None):
    sql = f"""
        SELECT sector, market_type, venue, event_id, minutes_to_tipoff,
               kalshi_clv_pct, line, yes_team
        FROM ev_predictions
        WHERE sector IN ({",".join("?" * len(sectors))})
          AND kalshi_clv_pct IS NOT NULL
          AND minutes_to_tipoff > 0
          AND (voided = 0 OR void_reason = 'stale_reverted')
    """
    params: list = list(sectors)
    if since:
        sql += " AND scan_date >= ?"
        params.append(since)
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def summarize(vals: list[float]) -> str:
    n = len(vals)
    if n == 0:
        return "n=0"
    m = mean(vals)
    pos = sum(v > 0 for v in vals) / n * 100
    if n > 1:
        se = stdev(vals) / math.sqrt(n)
        t = m / se if se > 0 else float("nan")
        return f"n={n:<3} mean={m:+.2f}pp  pos={pos:3.0f}%  t={t:+.1f}"
    return f"n={n:<3} mean={m:+.2f}pp  pos={pos:3.0f}%  t=  -"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "predictions.db")
    ap.add_argument("--sectors", default="nfl,ncaaf")
    ap.add_argument("--since", default=None)
    ap.add_argument("--by-venue", action="store_true",
                    help="split by venue (default pools kalshi + polymarket_us)")
    args = ap.parse_args()

    rows = load(args.db, args.sectors.split(","), args.since)
    if not rows:
        print("no rows")
        return

    # (sector, market_type, venue|*, bucket) -> event_id -> [clv]
    groups: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        b = bucket_of(r["minutes_to_tipoff"])
        if b is None:
            continue
        ven = r["venue"] if args.by_venue else "*"
        groups[(r["sector"], r["market_type"], ven, b)][game_key(r["event_id"])].append(r["kalshi_clv_pct"])

    lanes = sorted({k[:3] for k in groups})
    for lane in lanes:
        print(f"\n== {lane[0]} {lane[1]} venue={lane[2]} ==")
        all_games: dict[str, list[float]] = defaultdict(list)
        for label, _, _ in BUCKETS:
            games = groups.get((*lane, label), {})
            per_game = [mean(v) for v in games.values()]
            rungs = sum(len(v) for v in games.values())
            print(f"  {label:<7} {summarize(per_game)}  (rows={rungs})")
            for ev, v in games.items():
                all_games[ev].extend(v)
        print(f"  {'ALL':<7} {summarize([mean(v) for v in all_games.values()])}")


if __name__ == "__main__":
    main()
