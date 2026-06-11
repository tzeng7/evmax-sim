"""Append the current week's ATP/WTA top-150 from ESPN to the ranking-trend state.

The freshness layer of the two-source ranking pipeline:

  - Sackmann GitHub (scripts/reseed_tennis_rankings.py) — full depth (~2000
    ranks) + history, but volunteer-updated on a delay and its `_current.csv`
    rolls each season (the failure that froze our state for 18 months).
  - ESPN public API (this script) — keyless JSON, current week, top 150 for
    both tours, same reliability profile as the ESPN injury/scoreboard
    dependencies we already run on. Covers both players in ~80% of priced
    matches; the sub-150 tail still comes from Sackmann.

Snapshots are stamped with the most recent ranking-release Monday (UTC) and
appended via the agent's idempotent `append_snapshot`, so re-runs and overlap
with Sackmann rows are no-ops. If a weekly run is missed, the staleness guard
in the trend agent (6 weeks) bounds the damage to "model goes silent".

When scheduling, run AFTER reseed_tennis_rankings.py — the reseed fully
replaces the history store, so it would drop any ESPN-appended weeks that
Sackmann doesn't have yet.

Usage:
    uv run python scripts/refresh_tennis_rankings_espn.py
    uv run python scripts/refresh_tennis_rankings_espn.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import httpx

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from evmax.agents.models.tennis_common import resolve_player  # noqa: E402
from evmax.agents.models.tennis_ranking_trend_agent import TennisRankingTrendAgent  # noqa: E402

ESPN_RANKINGS_URL = "https://site.api.espn.com/apis/site/v2/sports/tennis/{tour}/rankings"
TOURS = ("atp", "wta")


def latest_release_monday(today: Optional[date] = None) -> date:
    """ATP/WTA rankings release on Mondays — stamp snapshots with the most
    recent Monday so they align with Sackmann's ranking_date convention."""
    today = today or date.today()
    return today - timedelta(days=today.weekday())


def fetch_espn_rankings(tour: str, timeout: float = 30.0) -> list[tuple[str, int]]:
    """Return [(player_name_lower, rank), ...] from ESPN's site API.

    Raises on HTTP/shape errors — the caller decides whether a partial run
    (one tour failing) should still persist the other.
    """
    r = httpx.get(ESPN_RANKINGS_URL.format(tour=tour), timeout=timeout)
    r.raise_for_status()
    blocks = r.json().get("rankings") or []
    if not blocks:
        raise ValueError(f"{tour}: ESPN payload has no rankings block")
    out: list[tuple[str, int]] = []
    for entry in blocks[0].get("ranks", []):
        name = ((entry.get("athlete") or {}).get("displayName") or "").strip().lower()
        rank = entry.get("current")
        if name and isinstance(rank, int) and rank > 0:
            out.append((name, rank))
    if len(out) < 50:
        # A top-150 feed suddenly returning a handful of rows means the
        # shape changed — better to fail loudly than seed garbage.
        raise ValueError(f"{tour}: only {len(out)} usable rows from ESPN (expected ~150)")
    return out


def resolve_store_key(name: str, store: dict) -> str:
    """Map an ESPN display name onto the existing Sackmann-seeded key so a
    player's history doesn't split across spelling variants. Falls back to
    the ESPN name itself (new key) only when nothing matches.

    Known divergences handled: hyphenated surnames ("felix auger-aliassime"
    vs Sackmann "felix auger aliassime") and Chinese surname-first order
    ("wang xinyu" vs "xinyu wang").
    """
    if name in store:
        return name
    dehyphen = name.replace("-", " ")
    if dehyphen in store:
        return dehyphen
    parts = name.split()
    if len(parts) == 2:
        reversed_name = f"{parts[1]} {parts[0]}"
        if reversed_name in store:
            return reversed_name
    resolved = resolve_player(name, store, weight_fn=lambda k: len(store.get(k, [])))
    # Fuzzy resolution matches on surname — guard with the first initial so a
    # new player can't inherit a same-surname sibling's history (Andreeva,
    # Pliskova, ...). On initial mismatch, start a fresh key instead.
    if resolved and resolved.split()[0][:1] == name.split()[0][:1]:
        return resolved
    return name


def refresh(agent: TennisRankingTrendAgent, snap_date: str,
            dry_run: bool = False) -> dict[str, int]:
    """Fetch both tours and append snapshots. Returns per-tour appended counts.

    Same idempotency contract as agent.append_snapshot (skip if the player
    already has this date), but batched into ONE save_state() at the end —
    per-row append_snapshot would rewrite the multi-MB state file ~300 times.
    """
    store = agent._state.setdefault("history", {})
    counts: dict[str, int] = {}
    for tour in TOURS:
        rows = fetch_espn_rankings(tour)
        appended = 0
        for name, rank in rows:
            key = resolve_store_key(name, store)
            snaps = store.setdefault(key, [])
            if any(s["date"] == snap_date for s in snaps):
                continue
            if not dry_run:
                snaps.append({"date": snap_date, "rank": int(rank)})
                snaps.sort(key=lambda s: s["date"])
            appended += 1
        counts[tour] = appended
        print(f"  {tour}: {len(rows)} rows fetched, "
              f"{'would append' if dry_run else 'appended'} {appended} @ {snap_date}")
    if not dry_run and any(counts.values()):
        agent.save_state()
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    snap_date = latest_release_monday().isoformat()
    print(f"ESPN tennis rankings refresh — snapshot date {snap_date}")
    agent = TennisRankingTrendAgent()
    try:
        refresh(agent, snap_date, dry_run=args.dry_run)
    except (httpx.HTTPError, ValueError) as e:
        print(f"  ! refresh failed: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
