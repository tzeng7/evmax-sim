"""Reseed ONLY the tennis ranking-trend state from Sackmann weekly rankings.

Why this exists: `seed_tennis_models.py` reads `{tour}_rankings_current.csv`,
but Sackmann rolls that file each season — in June 2026 it only contains
2026 weeks, and our state was frozen at 2024-12-30 because the last full
seed ran when "current" ended there. This script merges the decade file
(`{tour}_rankings_20s.csv`, 2020→latest-rolled-year) with the current file
(this season) so the agent gets a continuous weekly history, then REPLACES
the agent's history store (dropping dead/retired entries instead of letting
them linger stale).

Run this whenever the ranking state goes stale (weekly cron-able alongside
the surface-Elo calibration run). Other tennis agents are untouched.

Usage:
    uv run python scripts/reseed_tennis_rankings.py            # since 2024-01-01
    uv run python scripts/reseed_tennis_rankings.py --since 2025-01-01
    uv run python scripts/reseed_tennis_rankings.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import io
import shutil
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Optional

import httpx

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from evmax.agents.models.tennis_ranking_trend_agent import TennisRankingTrendAgent  # noqa: E402

REPO_BASE = {
    "atp": "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master",
    "wta": "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master",
}
# Decade file first (older weeks), current file second (this season's weeks).
RANKING_FILES = ("{tour}_rankings_20s.csv", "{tour}_rankings_current.csv")
STATE_PATH = _REPO_ROOT / "data" / "models" / "tennis_ranking_trend_state.json"


def _fetch_csv(url: str, timeout: float = 120.0) -> Optional[list[dict]]:
    try:
        r = httpx.get(url, timeout=timeout)
    except httpx.HTTPError as e:
        print(f"  ! fetch failed: {url} ({e})", file=sys.stderr)
        return None
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return list(csv.DictReader(io.StringIO(r.text)))


def load_history(tour: str, since: date) -> dict[str, list[dict]]:
    """Merged weekly snapshots {player → [{date, rank}, ...]} from since-date on."""
    base = REPO_BASE[tour]
    players = _fetch_csv(f"{base}/{tour}_players.csv")
    if players is None:
        print(f"  ! {tour}: players file unavailable")
        return {}
    pid_to_name = {}
    for p in players:
        pid = p.get("player_id")
        name = f"{(p.get('name_first') or '').strip()} {(p.get('name_last') or '').strip()}".strip()
        if pid and name:
            pid_to_name[pid] = name.lower()

    history: dict[str, list[dict]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()  # (player, date) — current may overlap 20s
    for tmpl in RANKING_FILES:
        rows = _fetch_csv(f"{base}/{tmpl.format(tour=tour)}")
        if rows is None:
            print(f"  ! {tour}: {tmpl.format(tour=tour)} unavailable")
            continue
        n_kept = 0
        for row in rows:
            pid, raw_date, rank = row.get("player"), row.get("ranking_date"), row.get("rank")
            if not pid or not raw_date or not rank or len(raw_date) < 8:
                continue
            iso = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
            try:
                if date.fromisoformat(iso) < since:
                    continue
                rank_i = int(rank)
            except ValueError:
                continue
            name = pid_to_name.get(pid)
            if not name or (name, iso) in seen:
                continue
            seen.add((name, iso))
            history[name].append({"date": iso, "rank": rank_i})
            n_kept += 1
        print(f"  + {tour} {tmpl.format(tour=tour)}: kept {n_kept} rows")

    for snaps in history.values():
        snaps.sort(key=lambda s: s["date"])
    return dict(history)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=str, default="2024-01-01",
                    help="Keep snapshots from this date on (default 2024-01-01 — "
                         "covers the walk-forward train window).")
    ap.add_argument("--tours", type=str, default="atp,wta")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    since = date.fromisoformat(args.since)
    tours = [t.strip().lower() for t in args.tours.split(",") if t.strip()]

    combined: dict[str, list[dict]] = {}
    for tour in tours:
        print(f"=== {tour.upper()} ===")
        combined.update(load_history(tour, since))  # ATP/WTA player sets are disjoint

    if not combined:
        print("No ranking history fetched — abort, state untouched.")
        return 1

    all_dates = sorted({s["date"] for snaps in combined.values() for s in snaps})
    print(f"\nFetched {len(combined)} players, weeks {all_dates[0]} → {all_dates[-1]}")

    if args.dry_run:
        print("[dry-run] state untouched")
        return 0

    if STATE_PATH.exists():
        backup = STATE_PATH.with_suffix(".json.bak")
        shutil.copy2(STATE_PATH, backup)
        print(f"Backed up old state → {backup}")

    agent = TennisRankingTrendAgent()
    agent._state["history"] = {}  # full replace — drop stale/retired entries
    agent.seed_history(combined)
    print(f"Seeded {len(combined)} players → {STATE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
