#!/usr/bin/env python3
"""Generate the ESPN full-name aliases for the ncaab / ncaaw alias maps.

Why
---
The college-basketball Elo and Form stores are keyed by ESPN's full team name
("idaho state bengals", from the D1 history seed), while every other surface
(Pinnacle, Kalshi, the efficiency state, logged predictions) uses the school
name alone ("idaho state"). Model READS bridged the two through the guarded
fuzzy fallback in ``_team_lookup``. Model WRITES are identity-only, so a result
fed under the short name created a second, fresh "idaho state" record next to
the real one, and every later read preferred that 1-game record.

With an alias "idaho state bengals" -> "idaho state", both spellings share one
canonical, so the identity tier (canonical equality) finds the stored record on
read and on write.

Source
------
The efficiency states (``data/models/{ncaab,ncaaw}_efficiency_state.json``)
store, per ESPN team id, the school key and the ESPN ``full_name``. Each pair
names one school by construction, so no name matching is involved.

Rules
-----
- Alias key = the full name after NameNormalizer's cleanup (lowercase,
  apostrophes dropped, other punctuation -> space), the form the alias map is
  looked up with.
- Target = the school key normalized with the curated alias map.
- Curated entries outside the generated block always win.
- An entry is skipped when key == target.

Usage
-----
    python scripts/build_college_bb_fullname_aliases.py            # write both
    python scripts/build_college_bb_fullname_aliases.py --dry-run  # print counts
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evmax.matching.normalizer import NameNormalizer  # noqa: E402

BEGIN = "  # --- BEGIN generated ESPN full names (scripts/build_college_bb_fullname_aliases.py) ---"
END = "  # --- END generated ESPN full names ---"


def clean(name: str) -> str:
    """NameNormalizer's pre-alias cleanup (no handler folds for college sectors)."""
    out = name.lower().strip()
    out = re.sub(r"['’‘]", "", out)
    out = re.sub(r"[^\w\s\-\.]", " ", out)
    return re.sub(r"\s+", " ", out).strip()


def strip_block(text: str) -> str:
    lines = text.splitlines()
    if BEGIN in lines:
        i, j = lines.index(BEGIN), lines.index(END)
        lines = lines[:i] + lines[j + 1:]
    return "\n".join(lines).rstrip("\n") + "\n"


def build(sector: str, alias_path: Path, state_path: Path) -> tuple[str, list[tuple[str, str]]]:
    base = strip_block(alias_path.read_text())
    # Normalize targets against the curated map only (block removed).
    alias_path.write_text(base)
    try:
        norm = NameNormalizer(sector)
        import yaml

        curated = (yaml.safe_load(base) or {}).get("aliases", {}) or {}
        teams = json.loads(state_path.read_text())["teams"]
        entries: dict[str, str] = {}
        for key, rec in sorted(teams.items()):
            full = rec.get("full_name")
            if not full:
                continue
            akey = clean(full)
            target = norm.normalize(key) or key
            if akey == target or akey in curated:
                continue
            prev = entries.get(akey)
            if prev is not None and prev != target:
                raise SystemExit(f"{sector}: {akey!r} maps to both {prev!r} and {target!r}")
            entries[akey] = target
    except BaseException:
        raise
    pairs = sorted(entries.items())
    block = [BEGIN] + [f'  "{k}": "{v}"' for k, v in pairs] + [END]
    return base + "\n".join(block) + "\n", pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    for sector in ("ncaab", "ncaaw"):
        alias_path = ROOT / "evmax" / "sectors" / "aliases" / f"{sector}.yaml"
        state_path = ROOT / "data" / "models" / f"{sector}_efficiency_state.json"
        original = alias_path.read_text()
        try:
            text, pairs = build(sector, alias_path, state_path)
        finally:
            alias_path.write_text(original)
        print(f"{sector}: {len(pairs)} generated aliases")
        if not args.dry_run:
            alias_path.write_text(text)


if __name__ == "__main__":
    main()
