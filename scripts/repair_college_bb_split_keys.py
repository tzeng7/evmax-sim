#!/usr/bin/env python3
"""Merge split team records in the ncaab / ncaaw Elo and Form stores.

Before the ESPN full-name aliases existed, a result fed under a school's short
name ("eastern kentucky") could not find the stored full-name record
("eastern kentucky colonels"), so the identity-only write path created a second
record holding that one game. Reads then preferred the 1-game record. See
``scripts/build_college_bb_fullname_aliases.py``.

With the aliases in place both keys share one canonical. This script, for each
canonical held by more than one stored key:

- keeps the key with the most Elo games as the real record;
- Form: moves the split key's records into the real key's list (same dedup,
  ordering and 2*WINDOW cap as ``FormModelAgent.update``), then drops the split
  key;
- Elo: drops the split key, then re-applies its game through
  ``EloModelAgent.update`` so the real record takes the result. The opponent
  already absorbed that game once, so its rating, counts and the H2H table are
  restored afterwards. When BOTH sides of the game were split, both real
  records take the result and nothing is restored.

The re-applied game lands after later games in Elo update order (the accepted
out-of-order precedent for state repairs). Default is a dry run.

Usage
-----
    python scripts/repair_college_bb_split_keys.py           # dry run
    python scripts/repair_college_bb_split_keys.py --apply
"""

from __future__ import annotations

import argparse
import copy
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evmax.agents.models._team_lookup import write_team_key  # noqa: E402
from evmax.agents.models.elo_agent import EloModelAgent  # noqa: E402
from evmax.agents.models.form_agent import WINDOW, FormModelAgent  # noqa: E402
from evmax.matching.normalizer import NameNormalizer  # noqa: E402

SECTORS = ("ncaab", "ncaaw")


def split_groups(sector: str, elo_state: dict) -> dict[str, list[str]]:
    """split key -> real key, for every canonical held by 2+ Elo keys."""
    norm = NameNormalizer(sector)
    counts = elo_state["game_counts"]
    groups: dict[str, list[str]] = defaultdict(list)
    for key in elo_state["ratings"]:
        groups[norm.normalize(key)].append(key)
    out: dict[str, str] = {}
    for keys in groups.values():
        if len(keys) < 2:
            continue
        real = max(keys, key=lambda k: counts.get(k, 0))
        for k in keys:
            if k != real:
                out[k] = real
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    elo = EloModelAgent()
    form = FormModelAgent()
    for sector in SECTORS:
        es = elo._sector_state(sector)
        fs = form._state.setdefault(sector, {})
        splits = split_groups(sector, es)
        print(f"{sector}: {len(splits)} split key(s)")
        games = []
        for split, real in sorted(splits.items()):
            recs = fs.get(split, [])
            print(f"  {split!r} (n={es['game_counts'].get(split, 0)}) -> {real!r}: {recs}")
            # Form: move the records.
            merged = list(fs.get(real, []))
            for r in recs:
                if not any((x["date"], x["opp"], x["home"]) == (r["date"], r["opp"], r["home"]) for x in merged):
                    merged.append(r)
            fs[real] = sorted(merged, key=lambda r: r["date"], reverse=True)[: WINDOW * 2]
            fs.pop(split, None)
            games.extend((split, real, r) for r in recs)

        # Elo: drop every split key first so writes resolve to the real keys.
        for split in splits:
            for table in ("ratings", "game_counts", "season_games"):
                es.get(table, {}).pop(split, None)

        done: set[tuple[str, str]] = set()
        for split, real, r in games:
            opp_raw = r["opp"]
            opp_real = splits.get(opp_raw) or write_team_key(sector, opp_raw, es["ratings"])
            pair = tuple(sorted((real, opp_real)))
            if (r["date"], *pair) in done:
                continue
            done.add((r["date"], *pair))
            both_split = opp_raw in splits
            snap = {t: copy.deepcopy(es.get(t, {}).get(opp_real)) for t in ("ratings", "game_counts", "season_games")}
            h2h = copy.deepcopy(es.get("h2h", {}))
            stamp = es.get("last_updated")
            margin = float(r["margin"])
            own, other = 100.0 + max(margin, 0.0), 100.0 + max(-margin, 0.0)
            if r["home"]:
                elo.update(real, opp_real, own, other, sector, event_date=r["date"])
            else:
                elo.update(opp_real, real, other, own, sector, event_date=r["date"])
            es["h2h"] = h2h
            es["last_updated"] = stamp
            if not both_split:
                for t, v in snap.items():
                    if v is None:
                        es.get(t, {}).pop(opp_real, None)
                    else:
                        es.setdefault(t, {})[opp_real] = v
            print(f"  elo: {r['date']} {real} vs {opp_real} margin {margin:+.0f}"
                  f" -> {round(es['ratings'][real])} (n={es['game_counts'][real]})"
                  + (" [both sides split]" if both_split else ""))

    if args.apply:
        elo.save_state()
        form.save_state()
        print("saved")
    else:
        print("dry run: nothing written")


if __name__ == "__main__":
    main()
