"""Fetch historical UFC fight results + fighter bios from ESPN's public API.

Writes two committed dataset files (repo precedent: data/backtest/soccer CSVs,
data/backtest/tennis xlsx):

  data/backtest/ufc/fights.csv    — one row per bout, chronological
  data/backtest/ufc/fighters.csv  — one row per distinct fighter (bio)

Source rationale lives in evmax/clients/ufc_espn.py — ufcstats.com is behind
a JS anti-bot interstitial, ESPN is the repo-standard public source.

All raw JSON is disk-cached under data/cache/ufc_espn/ so re-runs cost ~zero
network. Idempotent: re-running overwrites the CSVs from the (cached) source.

Usage:
    python scripts/fetch_ufc_history.py                 # 2010-01 → today
    python scripts/fetch_ufc_history.py --start 2012-01
    python scripts/fetch_ufc_history.py --no-methods    # skip per-fight method fetch
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evmax.clients.ufc_espn import UFCESPNClient, FightResult, fight_to_row

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "backtest" / "ufc"

FIGHT_FIELDS = [
    "event_id", "comp_id", "event_name", "date", "weight_class",
    "fighter_a_id", "fighter_a", "fighter_b_id", "fighter_b",
    "winner", "completed", "round", "clock", "method", "method_detail",
]
FIGHTER_FIELDS = ["athlete_id", "name", "dob", "height_in", "reach_in", "stance"]


def _months(start: str, end: str) -> list[str]:
    sy, sm = int(start[:4]), int(start[5:7])
    ey, em = int(end[:4]), int(end[5:7])
    out = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        out.append(f"{y}{m:02d}")
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


async def main(start: str, with_methods: bool) -> None:
    today = date.today()
    months = _months(start, f"{today.year}-{today.month:02d}")
    print(f"Fetching {len(months)} months of UFC cards from ESPN ({start} → today)")

    # Months near today are still accruing events — bypass their cache so a
    # weekly reseed picks up the newest cards. Older months are immutable.
    recent = {
        f"{today.year}{today.month:02d}",
        f"{today.year - (today.month == 1)}{(today.month - 2) % 12 + 1:02d}",
    }

    async with UFCESPNClient() as client:
        # 1. Scoreboards — one request per month, disk-cached.
        month_results = await asyncio.gather(
            *(client.fetch_month(m, refresh=(m in recent)) for m in months)
        )
        fights: list[FightResult] = [f for batch in month_results for f in batch]
        # Keep completed bouts only; sort chronologically.
        fights = [f for f in fights if f.completed and f.fighter_a and f.fighter_b]
        fights.sort(key=lambda f: (f.date, f.event_id, f.comp_id))
        # Dedupe (a month query can repeat a card at month boundaries).
        seen: set[str] = set()
        unique: list[FightResult] = []
        for f in fights:
            if f.comp_id not in seen:
                seen.add(f.comp_id)
                unique.append(f)
        fights = unique
        print(f"  completed bouts: {len(fights)}")

        # 2. Method of victory — one core-status request per bout (cached).
        if with_methods:
            sem = asyncio.Semaphore(8)

            async def _method(f: FightResult) -> None:
                async with sem:
                    f.method, f.method_detail = await client.fetch_method(
                        f.event_id, f.comp_id
                    )

            await asyncio.gather(*(_method(f) for f in fights))
            n_meth = sum(1 for f in fights if f.method)
            print(f"  methods resolved: {n_meth}/{len(fights)}")

        # 3. Fighter bios — one core-athlete request per distinct fighter.
        athlete_ids = sorted(
            {f.fighter_a_id for f in fights if f.fighter_a_id}
            | {f.fighter_b_id for f in fights if f.fighter_b_id}
        )
        print(f"  distinct fighters: {len(athlete_ids)}")
        sem2 = asyncio.Semaphore(8)

        async def _bio(aid: str):
            async with sem2:
                return await client.fetch_athlete(aid)

        bios = [b for b in await asyncio.gather(*(_bio(a) for a in athlete_ids)) if b]
        print(f"  bios fetched: {len(bios)}")

    if not fights:
        print("ABORT: no fights fetched — leaving existing CSVs untouched.")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fights_path = OUT_DIR / "fights.csv"
    with fights_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIGHT_FIELDS)
        writer.writeheader()
        for f in fights:
            writer.writerow({k: fight_to_row(f)[k] for k in FIGHT_FIELDS})
    print(f"  wrote {fights_path} ({len(fights)} rows)")

    fighters_path = OUT_DIR / "fighters.csv"
    with fighters_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIGHTER_FIELDS)
        writer.writeheader()
        for b in sorted(bios, key=lambda x: x.athlete_id):
            writer.writerow({
                "athlete_id": b.athlete_id, "name": b.name, "dob": b.dob,
                "height_in": b.height_in, "reach_in": b.reach_in, "stance": b.stance,
            })
    print(f"  wrote {fighters_path} ({len(bios)} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2010-01", help="first month YYYY-MM")
    ap.add_argument("--no-methods", action="store_true",
                    help="skip per-fight method-of-victory fetch (faster)")
    args = ap.parse_args()
    asyncio.run(main(args.start, with_methods=not args.no_methods))
