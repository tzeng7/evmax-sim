"""Seed the national-team Elo state from ESPN international results.

The `worldcup` Elo namespace (data/models/elo_state.json['worldcup']) is a
SEPARATE pool from club soccer — national teams never appear in the club
ratings and club strength is meaningless for them. This walk-forward seeds it
from real international matches (World Cup + qualifiers + friendlies + the
continental cups) so the 48 WC participants enter the 2026 tournament with
data-driven priors instead of a flat 1500.

Method: gather completed games across the ESPN international competitions,
sort chronologically, and replay them through EloModelAgent.update(sector=
"worldcup"). Home advantage is 0 for this sector (neutral venues) and K=40
(national teams play ~10 games/yr, so each result is informative). The agent's
recency-weighted K, margin-of-victory, and strength-of-schedule multipliers all
apply, and `last_updated` is stamped to the most recent match date so the
predict-time staleness guard treats the seed as fresh through the tournament.

Usage:
    python scripts/seed_national_team_elo.py                 # default 2023-06 → today
    python scripts/seed_national_team_elo.py --since 2022-01-01
    python scripts/seed_national_team_elo.py --dry-run       # print, don't write
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import unicodedata
from datetime import date, timedelta
from pathlib import Path

import httpx

# Import evmax from THIS checkout (worktree), not a globally-installed editable
# copy — otherwise `python scripts/seed_national_team_elo.py` would run stale
# code and write to the wrong data/models. Mirrors scripts/seed_nfl_efficiency.py.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from evmax.agents.models.elo_agent import EloModelAgent  # noqa: E402
from evmax.matching.normalizer import NameNormalizer  # noqa: E402

SECTOR = "worldcup"
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"

# International competitions ESPN exposes under the soccer sport path. Qualifiers
# + friendlies + Nations League give continuous coverage; the continental cups
# (Euro/Copa/AFCON/Asian Cup) add high-signal knockout results. Slugs that
# return nothing for a given window are simply skipped.
COMPETITIONS = [
    "fifa.world",
    "fifa.friendly",
    "fifa.worldq.uefa",
    "fifa.worldq.conmebol",
    "fifa.worldq.concacaf",
    "fifa.worldq.afc",
    "fifa.worldq.caf",
    "fifa.worldq.ofc",
    "uefa.nations",
    "uefa.euro",
    "conmebol.america",
    "concacaf.gold",
    "concacaf.nations",
    "caf.nations",
    "afc.asian",
]

# The 48 nations in the 2026 World Cup (canonical form) — used only for the
# post-seed sanity report, not to filter the seed (rating every international
# team that shows up makes opponent-strength adjustments meaningful).
WC_2026 = {
    "argentina", "australia", "austria", "belgium", "bosnia and herzegovina",
    "brazil", "canada", "ivory coast", "colombia", "cape verde", "croatia",
    "curacao", "czechia", "algeria", "ecuador", "egypt", "england", "spain",
    "france", "germany", "ghana", "haiti", "iran", "iraq", "jordan", "japan",
    "south korea", "saudi arabia", "morocco", "mexico", "netherlands", "norway",
    "new zealand", "panama", "paraguay", "portugal", "qatar", "south africa",
    "scotland", "senegal", "switzerland", "sweden", "tunisia", "turkiye",
    "uruguay", "united states", "uzbekistan", "dr congo",
}


def _norm(name: str, nn: NameNormalizer) -> str:
    """Normalize an ESPN team name to the worldcup canonical.

    NFKD-strips diacritics first (ESPN ships "Türkiye"/"Curaçao"); NameNormalizer
    keeps unicode letters, so without this "türkiye" would miss the alias map.
    Mirrors the predict-time path (NameNormalizer + worldcup aliases) so seed
    keys equal the keys the live blend will look up.
    """
    stripped = "".join(
        c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c)
    )
    return nn.normalize(stripped)


async def _fetch_range(client: httpx.AsyncClient, slug: str, start: date, end: date) -> list[dict]:
    """Fetch completed games for one competition over a date range."""
    params = {
        "dates": f"{start.isoformat().replace('-', '')}-{end.isoformat().replace('-', '')}",
        "limit": 500,
    }
    try:
        r = await client.get(f"{ESPN_BASE}/{slug}/scoreboard", params=params)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"    WARN {slug} {start}..{end}: {e}")
        return []

    games: list[dict] = []
    for ev in data.get("events", []):
        comps = ev.get("competitions", [])
        if not comps:
            continue
        comp = comps[0]
        if not comp.get("status", {}).get("type", {}).get("completed"):
            continue
        cs = comp.get("competitors", [])
        home = next((c for c in cs if c.get("homeAway") == "home"), None)
        away = next((c for c in cs if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        try:
            hs = int(home.get("score", 0))
            as_ = int(away.get("score", 0))
        except (ValueError, TypeError):
            continue
        gd = (ev.get("date") or "")[:10]
        if not gd:
            continue
        games.append({
            "date": gd,
            "home": home.get("team", {}).get("displayName", ""),
            "away": away.get("team", {}).get("displayName", ""),
            "hs": hs, "as": as_,
        })
    return games


async def gather_games(since: date, until: date) -> list[dict]:
    """Fetch all completed international games across competitions, deduped + sorted."""
    seen: set[tuple] = set()
    all_games: list[dict] = []
    # Chunk into 180-day windows so each ESPN response stays bounded.
    async with httpx.AsyncClient(timeout=30) as client:
        for slug in COMPETITIONS:
            slug_total = 0
            cur = since
            while cur < until:
                window_end = min(cur + timedelta(days=180), until)
                games = await _fetch_range(client, slug, cur, window_end)
                for g in games:
                    key = (g["date"], g["home"], g["away"], g["hs"], g["as"])
                    if key in seen:
                        continue
                    seen.add(key)
                    all_games.append(g)
                    slug_total += 1
                cur = window_end + timedelta(days=1)
            print(f"  {slug:24} {slug_total} games")
    all_games.sort(key=lambda g: g["date"])
    return all_games


def seed(games: list[dict], dry_run: bool) -> EloModelAgent:
    nn = NameNormalizer(SECTOR)
    agent = EloModelAgent()
    # Idempotent re-seed: drop any prior worldcup namespace so re-running the
    # walk-forward doesn't double-count games (inflating game_counts and drifting
    # ratings). Other sectors' state is untouched.
    agent._state.pop(SECTOR, None)
    applied = 0
    for g in games:
        home = _norm(g["home"], nn)
        away = _norm(g["away"], nn)
        if not home or not away or home == away:
            continue
        agent.update(home, away, g["hs"], g["as"], SECTOR, event_date=g["date"])
        applied += 1
    print(f"\nApplied {applied} results to worldcup Elo.")
    n = len(agent.all_ratings(SECTOR))
    if not dry_run:
        agent.save_state()
        print(f"Saved {n} national-team ratings to elo_state.json['{SECTOR}'].")
    else:
        print(f"[dry-run] would save {n} ratings (not written).")
    return agent


def report(agent: EloModelAgent) -> None:
    # Read the agent's live ratings at print time (not a captured snapshot) so
    # the report always reflects exactly what was seeded/saved.
    ratings = agent.all_ratings(SECTOR)
    print("\n=== 2026 World Cup participants — seeded Elo (desc) ===")
    wc = sorted(
        ((t, r) for t, r in ratings.items() if t in WC_2026),
        key=lambda x: -x[1],
    )
    for t, r in wc:
        print(f"  {r:7.0f}  {t}")
    missing = WC_2026 - {t for t, _ in wc}
    print(f"\nWC teams rated: {len(wc)}/48", f"(default-1500/missing: {sorted(missing)})" if missing else "")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2023-06-01", help="YYYY-MM-DD start (default 2023-06-01)")
    ap.add_argument("--until", default=date.today().isoformat(), help="YYYY-MM-DD end (default today)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    since = date.fromisoformat(args.since)
    until = date.fromisoformat(args.until)
    print(f"Seeding national-team Elo from ESPN international results {since} → {until}")
    games = await gather_games(since, until)
    print(f"\nTotal completed international games: {len(games)}")
    agent = seed(games, args.dry_run)
    report(agent)


if __name__ == "__main__":
    asyncio.run(main())
