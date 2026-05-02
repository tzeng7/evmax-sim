"""Backfill `drew=True` on existing soccer Form records.

Pre-migration behavior recorded draws as `won=False` for BOTH teams, which
made drawing teams look identical to losing teams and pushed the soccer Form
Brier above the always-home baseline (see walk-forward diagnosis 2026-04-23).

This script reads data/models/form_state.json, finds pairs of records where
team A played team B on date D and both have won=False, marks both as
drew=True, and writes the state back. It is idempotent — re-running does
nothing because records with drew=True are skipped.

Non-soccer sectors are left untouched (they have no draws).

Usage:
    python scripts/migrate_form_soccer_draws.py            # apply
    python scripts/migrate_form_soccer_draws.py --dry-run  # preview counts
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

STATE_PATH = Path("data/models/form_state.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not STATE_PATH.exists():
        print(f"No state file at {STATE_PATH}")
        return

    with STATE_PATH.open() as f:
        state = json.load(f)

    soccer = state.get("soccer")
    if not soccer:
        print("No soccer section — nothing to migrate.")
        return

    # Build (date, {team_a, team_b}) → list of (team, record) pointers
    bucket: dict[tuple[str, frozenset], list[tuple[str, dict]]] = defaultdict(list)
    total_records = 0
    for team, records in soccer.items():
        for rec in records:
            total_records += 1
            opp = rec.get("opp", "")
            date = rec.get("date", "")
            if not opp or not date:
                continue
            key = (date, frozenset({team, opp}))
            bucket[key].append((team, rec))

    fixed = 0
    already_drew = 0
    orphan_loss = 0  # team has won=False but opponent record missing — ambiguous

    for (date, pair), entries in bucket.items():
        if len(entries) != 2:
            # One-sided record (e.g. team in state but opponent wasn't re-seeded).
            # If the single side is won=False, we can't tell draw from loss.
            if len(entries) == 1 and entries[0][1].get("won") is False and not entries[0][1].get("drew"):
                orphan_loss += 1
            continue
        r1 = entries[0][1]
        r2 = entries[1][1]
        # If both already marked drew, count and skip.
        if r1.get("drew") and r2.get("drew"):
            already_drew += 1
            continue
        # Draw iff both sides recorded won=False.
        if r1.get("won") is False and r2.get("won") is False:
            r1["drew"] = True
            r2["drew"] = True
            fixed += 1

    print(f"Total soccer records:   {total_records}")
    print(f"Matched opponent pairs: {sum(1 for e in bucket.values() if len(e) == 2)}")
    print(f"Already drew=True:      {already_drew}")
    print(f"Newly marked drew=True: {fixed}  (affects {fixed * 2} records)")
    print(f"Orphan-loss ambiguous:  {orphan_loss}  (single-sided 'loss' we can't verify)")

    if args.dry_run:
        print("\n--dry-run: no file written")
        return

    # Backup
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup = STATE_PATH.with_name(f"form_state.backup_{ts}.json")
    shutil.copy2(STATE_PATH, backup)
    print(f"\nBackup: {backup}")

    with STATE_PATH.open("w") as f:
        json.dump(state, f, indent=2)
    print(f"Wrote:  {STATE_PATH}")


if __name__ == "__main__":
    main()
