#!/usr/bin/env python3
"""Build (or refresh) evmax/sectors/aliases/ncaaf.yaml from ESPN's college-football
scoreboard.

Why a generated map: FBS is ~134 teams plus a rotating set of FCS opponents that
appear on early-season FBS schedules. Enumerating them by hand is error-prone and
goes stale every year (realignment). This walks a full past season's ESPN
scoreboard (groups=80 = FBS games), collects every distinct team that played, and
emits a variant → canonical alias map.

Canonical form
--------------
The canonical name is the NameNormalizer-normalized ESPN ``location`` (school name
without mascot), e.g. "Ohio State", "Miami" (FL), "Miami (OH)". Every ESPN surface
form for the team (displayName, shortDisplayName, abbreviation, and the
mascot-suffixed name) maps to that canonical string so that Kalshi/Pinnacle/ESPN
all collapse to one key at match and resolution time.

What this does NOT do
---------------------
It does not know Kalshi's per-market ticker outcome codes (e.g. "OSU", "MIA") —
those are only knowable once Kalshi lists real KXNCAAFGAME markets. When the season
lists, run ``scripts/check_kalshi_series.py --probe`` / inspect a live scan and add
any unmatched short codes to the ``# --- Kalshi ticker codes ---`` block by hand
(the MLS/Torino precedent). The generated map plus the shadow-mode + MIN_NONSHARP
guard means an unmatched team demotes to shadow rather than mispricing a live bet.

Curated overrides (ambiguity guards + known Pinnacle spellings) live in
CURATED_ALIASES below and always win over the generated entries.

Usage
-----
    python scripts/build_ncaaf_aliases.py                 # walk 2024 season, write yaml
    python scripts/build_ncaaf_aliases.py --season 2023   # different season
    python scripts/build_ncaaf_aliases.py --dry-run       # print, don't write
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import httpx
import yaml

ESPN_SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard"
)
OUT_PATH = Path(__file__).resolve().parents[1] / "evmax/sectors/aliases/ncaaf.yaml"

# Season date windows (regular season through bowls / CFP). A CFB "season" label
# is the fall calendar year; games run late Aug → mid Jan of the next year.
SEASON_WINDOWS: dict[int, tuple[str, str]] = {
    2022: ("2022-08-25", "2023-01-10"),
    2023: ("2023-08-25", "2024-01-09"),
    2024: ("2024-08-24", "2025-01-21"),
    2025: ("2025-08-23", "2026-01-20"),
}

# Curated aliases that always win over generated ESPN forms. These reconcile the
# well-known cross-source naming disagreements and ambiguity traps. Keys are
# raw-lowercased source names (pre-normalization is fine — NameNormalizer applies
# the sector map after its own lowercasing/punctuation pass, and also tries the
# full cleaned name against the map before noise stripping). Values are the
# canonical school name.
#
# IMPORTANT: every canonical target (value) MUST equal the NameNormalizer-
# normalized ESPN ``location`` for that program, because outcome resolution reads
# ESPN. Verified against the 2024 FBS scoreboard — do not "tidy" a target to a
# prettier spelling (e.g. ESPN uses "Ole Miss", "App State", "UConn", "Miami
# (OH)"→"miami oh", "San José State" WITH the accent — normalizer does not fold
# diacritics, so the no-accent source form must alias to the accented canonical).
CURATED_ALIASES: dict[str, str] = {
    # --- Ambiguity guards: two schools share a short name ---
    # ESPN: Florida's program location = "Miami"; Ohio's = "Miami (OH)"
    # (normalizes to "miami oh"). Pinnacle/Kalshi spell them "Miami FL"/"Miami OH".
    "miami fl": "miami",
    "miami (fl)": "miami",
    "miami florida": "miami",
    "miami hurricanes": "miami",
    "miami oh": "miami oh",
    "miami (oh)": "miami oh",
    "miami ohio": "miami oh",
    "miami redhawks": "miami oh",
    # USC = Southern California (Trojans); ESPN location = "USC".
    "southern california": "usc",
    "southern cal": "usc",
    "so california": "usc",
    # South Carolina (Gamecocks); guard the abbreviations against USC.
    "so carolina": "south carolina",
    "s carolina": "south carolina",
    # --- Common Pinnacle / Kalshi spellings differing from ESPN location ---
    # Ole Miss: ESPN location = "Ole Miss"; Pinnacle often "Mississippi".
    # Mississippi State stays distinct (ESPN "Mississippi State").
    "mississippi": "ole miss",
    "ole miss rebels": "ole miss",
    "miss state": "mississippi state",
    "southern mississippi": "southern miss",
    "pitt": "pittsburgh",
    # UConn: ESPN location = "UConn" (NOT "Connecticut", which is FCS Central Conn).
    "connecticut": "uconn",
    "umass": "massachusetts",
    "central florida": "ucf",
    "southern methodist": "smu",
    "texas christian": "tcu",
    "brigham young": "byu",
    "louisiana state": "lsu",
    "texas san antonio": "utsa",
    "texas el paso": "utep",
    "fiu": "florida international",
    "fau": "florida atlantic",
    # Appalachian State: ESPN location = "App State".
    "appalachian state": "app state",
    "appalachian st": "app state",
    # San José State: ESPN keeps the accent → canonical "san josé state".
    "san jose state": "san josé state",
    "san jose st": "san josé state",
    "san jose state spartans": "san josé state",
    # Hawai'i: ESPN "Hawai'i" → apostrophe stripped → "hawaii".
    "hawaii": "hawaii",
    "north carolina state": "nc state",
    "n.c. state": "nc state",
    # Louisiana (Ragin' Cajuns); ULM = "UL Monroe" on ESPN.
    "louisiana lafayette": "louisiana",
    "ul lafayette": "louisiana",
    "louisiana monroe": "ul monroe",
    "la monroe": "ul monroe",
    "sam houston state": "sam houston",
    "jax state": "jacksonville state",
    "jacksonville st": "jacksonville state",
}

# Away/home game-name mascot words are NOT stripped by NameNormalizer (that only
# strips soccer/esports noise). So the mascot-suffixed ESPN displayName
# ("Alabama Crimson Tide") must be mapped explicitly to the location.


def _norm_key(s: str) -> str:
    """Lowercase + collapse whitespace, matching NameNormalizer's first pass
    closely enough for a stable alias key."""
    import re

    s = s.lower().strip()
    s = re.sub(r"['’‘]", "", s)
    s = re.sub(r"[^\w\s\-\.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def fetch_season_teams(season: int) -> dict[str, dict]:
    """Walk the season's scoreboard, return {espn_team_id: team_dict}."""
    start_s, end_s = SEASON_WINDOWS[season]
    start = dt.date.fromisoformat(start_s)
    end = dt.date.fromisoformat(end_s)
    teams: dict[str, dict] = {}
    day = start
    n_calls = 0
    with httpx.Client(timeout=30) as client:
        while day <= end:
            # CFB plays Tue–Sat plus bowl weekdays; walking every day is cheapest
            # to reason about and the calls are free/public.
            params = {"dates": day.strftime("%Y%m%d"), "groups": "80", "limit": 400}
            try:
                r = client.get(ESPN_SCOREBOARD, params=params)
                r.raise_for_status()
                events = r.json().get("events", [])
            except Exception as e:  # noqa: BLE001
                print(f"  WARN {day}: {e}", file=sys.stderr)
                events = []
            n_calls += 1
            for e in events:
                for c in e["competitions"][0]["competitors"]:
                    t = c["team"]
                    tid = t.get("id")
                    if tid and tid not in teams:
                        teams[tid] = t
            day += dt.timedelta(days=1)
    print(f"  {n_calls} scoreboard calls, {len(teams)} distinct teams", file=sys.stderr)
    return teams


def build_alias_map(teams: dict[str, dict]) -> dict[str, str]:
    """Emit variant → canonical alias entries from ESPN team dicts."""
    aliases: dict[str, str] = {}
    for t in teams.values():
        location = t.get("location") or t.get("shortDisplayName") or ""
        canonical = _norm_key(location)
        if not canonical:
            continue
        variants = {
            t.get("location"),
            t.get("displayName"),
            t.get("shortDisplayName"),
            t.get("name"),  # mascot alone — skip if it collides (handled below)
            t.get("abbreviation"),
        }
        for v in variants:
            if not v:
                continue
            key = _norm_key(v)
            if not key or key == canonical:
                continue
            # Never let a bare mascot ("bulldogs") become an alias — many schools
            # share mascots. Only map multi-token or school-identifying forms.
            if key == _norm_key(t.get("name") or ""):
                continue
            aliases.setdefault(key, canonical)
    # Curated overrides always win.
    for k, v in CURATED_ALIASES.items():
        aliases[_norm_key(k)] = _norm_key(v)
    return dict(sorted(aliases.items()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, default=2024, choices=sorted(SEASON_WINDOWS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"Walking {args.season} FBS scoreboard…", file=sys.stderr)
    teams = fetch_season_teams(args.season)
    aliases = build_alias_map(teams)

    doc = {
        "aliases": aliases,
    }
    header = (
        "# NCAA Football (FBS) team-name alias map.\n"
        "# GENERATED by scripts/build_ncaaf_aliases.py — do not hand-edit generated\n"
        "# entries; add durable overrides to CURATED_ALIASES in that script and\n"
        f"# regenerate. Source: ESPN {args.season} FBS scoreboard (groups=80).\n"
        "#\n"
        "# Canonical = NameNormalizer-normalized ESPN school location. Kalshi\n"
        "# per-market ticker outcome codes are NOT here yet (only knowable once\n"
        "# KXNCAAFGAME lists live markets) — add them under a '# Kalshi codes'\n"
        "# block during the first live scan of the season (MLS/Torino precedent).\n"
    )
    text = header + yaml.safe_dump(doc, sort_keys=True, allow_unicode=True, width=100)
    if args.dry_run:
        print(text)
        print(f"\n({len(aliases)} alias entries; dry-run, not written)", file=sys.stderr)
        return 0
    OUT_PATH.write_text(text)
    print(f"Wrote {OUT_PATH} ({len(aliases)} alias entries)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
