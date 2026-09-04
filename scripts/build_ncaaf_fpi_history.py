#!/usr/bin/env python3
"""Build data/backtest/ncaaf_fpi/fpi_{season}.json — PRESEASON ESPN FPI per
ESPN team id, from Wayback Machine snapshots of espn.com/college-football/fpi.

These files are the backtest-time source for the ``ncaaf_efficiency_v2``
FPI-mixed prior (scripts/backtest_ncaaf_v2.py). The live seed uses the fitt
powerindex endpoint instead (evmax/clients/cfb_fpi.py::fetch_fpi).

Snapshot timestamps are chosen BEFORE each season's Week-0 kickoff so the
rating is a genuine preseason projection; the page's ``lastUpdated`` is
printed so a post-kickoff snapshot is visible. Idempotent — re-running
overwrites the same files. Aborts a season without writing when the snapshot
parses to fewer than --min-teams rows.

    python scripts/build_ncaaf_fpi_history.py                 # 2023-2025
    python scripts/build_ncaaf_fpi_history.py --seasons 2025
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evmax.clients import cfb_fpi as F  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "backtest" / "ncaaf_fpi"

# Closest-before-Week-0 Wayback timestamps (verified 2026-09-03: page lastUpdated
# 2023-08-03 / 2024-08-22 / 2025-08-15, all pre-kickoff).
SNAPSHOT_TS: dict[int, str] = {
    2023: "20230824",
    2024: "20240823",
    2025: "20250818",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", default=",".join(str(s) for s in SNAPSHOT_TS))
    ap.add_argument("--min-teams", type=int, default=120)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rc = 0
    for s in (int(x) for x in args.seasons.split(",")):
        ts = SNAPSHOT_TS.get(s)
        if not ts:
            print(f"{s}: no snapshot timestamp configured (add to SNAPSHOT_TS)", file=sys.stderr)
            rc = 1
            continue
        ratings, url = F.fetch_preseason_fpi_wayback(ts)
        if len(ratings) < args.min_teams:
            print(f"{s}: only {len(ratings)} teams parsed from {url} — NOT written", file=sys.stderr)
            rc = 1
            continue
        top = sorted(ratings.items(), key=lambda kv: -kv[1]["fpi"])[:3]
        print(f"{s}: {len(ratings)} teams from {url}; top {[(v['name'], v['fpi']) for _, v in top]}",
              file=sys.stderr)
        if args.dry_run:
            continue
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / f"fpi_{s}.json").write_text(json.dumps(
            {"season": s, "snapshot": url, "teams": ratings}, indent=1, sort_keys=True))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
