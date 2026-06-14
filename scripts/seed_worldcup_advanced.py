"""Seed the World Cup *advanced-metric* models from ESPN international results.

The `worldcup` blend mirrors club soccer (poisson 0.40 · xg 0.25 · elo 0.15 ·
form 0.10) but every model reads its OWN national-team namespace — never the
club `soccer` pool. This script seeds the three soccer-like models that aren't
the (separately seeded) Elo:

  - Poisson attack/defense  → poisson_state.json['worldcup']
  - xG rolling shot window  → soccer_xg_state.json['worldcup']
  - Form W/D/L records      → form_state.json['worldcup']

It walks the SAME ESPN international competitions as scripts/seed_national_team_elo.py
(World Cup + qualifiers + friendlies + Nations League + continental cups),
chronologically, replaying each completed match through every model's update
path. Goals drive Poisson + Form; ESPN's per-competitor shotsOnTarget/totalShots
drive xG (matches without shot stats — some smaller-nation friendlies — are
simply skipped for xG but still feed Poisson/Form).

Each model's `worldcup` namespace is reset first so re-running is idempotent
(no double-counted games). Other sectors' state is left untouched.

Usage:
    python scripts/seed_worldcup_advanced.py                  # 2023-06 → today
    python scripts/seed_worldcup_advanced.py --since 2022-01-01
    python scripts/seed_worldcup_advanced.py --dry-run        # print, don't write
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
# copy — otherwise this would run stale code / write to the wrong data/models.
# Mirrors scripts/seed_national_team_elo.py.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from evmax.agents.models.form_agent import FormModelAgent  # noqa: E402
from evmax.agents.models.poisson_agent import PoissonModelAgent  # noqa: E402
from evmax.agents.models.soccer_xg_agent import SoccerXgAgent  # noqa: E402
from evmax.matching.normalizer import NameNormalizer  # noqa: E402

SECTOR = "worldcup"
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"

# Same competition set as the Elo seed — qualifiers + friendlies + Nations
# League give continuous coverage; the continental cups add knockout signal.
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


def _norm(name: str, nn: NameNormalizer) -> str:
    """Normalize an ESPN team name to the worldcup canonical (NFKD-strip first)."""
    stripped = "".join(
        c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c)
    )
    return nn.normalize(stripped)


def _stat(competitor: dict, name: str) -> int | None:
    """Read an integer box-score stat from a competitor, or None if absent."""
    for s in competitor.get("statistics", []):
        if s.get("name") == name:
            try:
                return int(float(s.get("displayValue", 0)))
            except (ValueError, TypeError):
                return None
    return None


async def _fetch_range(client: httpx.AsyncClient, slug: str, start: date, end: date) -> list[dict]:
    """Fetch completed games (with shot stats when present) for one competition."""
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
            "home_sot": _stat(home, "shotsOnTarget"),
            "away_sot": _stat(away, "shotsOnTarget"),
            "home_shots": _stat(home, "totalShots"),
            "away_shots": _stat(away, "totalShots"),
        })
    return games


async def gather_games(since: date, until: date) -> list[dict]:
    """Fetch all completed international games across competitions, deduped + sorted."""
    seen: set[tuple] = set()
    all_games: list[dict] = []
    async with httpx.AsyncClient(
        timeout=30,
        headers={"User-Agent": "evmax-seeder/1.0"},
        follow_redirects=True,
    ) as client:
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


def seed(games: list[dict], dry_run: bool) -> tuple[PoissonModelAgent, SoccerXgAgent, FormModelAgent]:
    nn = NameNormalizer(SECTOR)
    poisson = PoissonModelAgent()
    xg = SoccerXgAgent()
    form = FormModelAgent()

    # Idempotent re-seed: drop the prior worldcup namespace from each model so a
    # re-run doesn't double-count games. Other sectors are untouched.
    poisson._state.pop(SECTOR, None)
    xg._state.pop(SECTOR, None)
    form._state.pop(SECTOR, None)

    applied = xg_applied = 0
    for g in games:
        home = _norm(g["home"], nn)
        away = _norm(g["away"], nn)
        if not home or not away or home == away:
            continue

        poisson.update(home, away, g["hs"], g["as"], SECTOR, event_date=g["date"])
        form.update(home, away, g["hs"], g["as"], SECTOR, event_date=g["date"])
        applied += 1

        shots = (g["home_sot"], g["away_sot"], g["home_shots"], g["away_shots"])
        if all(v is not None for v in shots) and (g["home_shots"] or g["away_shots"]):
            xg.record_match(
                team=home, goals_for=g["hs"], goals_against=g["as"],
                shots_on_target=g["home_sot"], total_shots=g["home_shots"],
                opponent_sot=g["away_sot"], opponent_shots=g["away_shots"],
                match_date=g["date"], is_home=True, sector=SECTOR,
            )
            xg.record_match(
                team=away, goals_for=g["as"], goals_against=g["hs"],
                shots_on_target=g["away_sot"], total_shots=g["away_shots"],
                opponent_sot=g["home_sot"], opponent_shots=g["home_shots"],
                match_date=g["date"], is_home=False, sector=SECTOR,
            )
            xg_applied += 1

    n_poisson = len(poisson._state.get(SECTOR, {}).get("teams", {}))
    n_xg = len(xg._teams_for(SECTOR))
    n_form = len(form._state.get(SECTOR, {}))
    print(
        f"\nApplied {applied} results "
        f"({xg_applied} with shot data) to worldcup advanced models."
    )
    if not dry_run:
        poisson.save_state()
        xg.save_state()
        form.save_state()
        print(
            f"Saved: poisson {n_poisson} teams · xg {n_xg} teams · form {n_form} teams."
        )
    else:
        print(
            f"[dry-run] would save: poisson {n_poisson} · xg {n_xg} · form {n_form} (not written)."
        )
    return poisson, xg, form


def report(poisson: PoissonModelAgent, xg: SoccerXgAgent) -> None:
    teams = poisson._state.get(SECTOR, {}).get("teams", {})
    ranked = sorted(
        (
            (t, v["attack"] / max(v["defense"], 0.1), v.get("games", 0))
            for t, v in teams.items()
            if v.get("games", 0) >= 5
        ),
        key=lambda x: -x[1],
    )
    print("\n=== Poisson attack/defense ratio (desc, ≥5 games) ===")
    for t, ratio, gp in ranked[:15]:
        print(f"  {ratio:5.2f}  {t}  (n={gp})")

    xg_teams = xg._teams_for(SECTOR)
    xg_ranked = []
    for team, data in xg_teams.items():
        ms = data.get("matches", [])[:10]
        if len(ms) < 4:
            continue
        xg_avg = sum(m["xg"] for m in ms) / len(ms)
        goals_avg = sum(m["goals_for"] for m in ms) / len(ms)
        xg_ranked.append((team, xg_avg, goals_avg, len(ms)))
    xg_ranked.sort(key=lambda x: -x[1])
    print("\n=== Top by xG/game (last 10) ===")
    for team, xg_avg, goals, n in xg_ranked[:15]:
        ratio = goals / xg_avg if xg_avg > 0 else 0
        flag = " ↓REGRESS" if ratio > 1.3 else " ↑UNLUCKY" if ratio < 0.7 else ""
        print(f"  {team:22s}  xG={xg_avg:.2f}  goals={goals:.2f}  ratio={ratio:.2f}{flag}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2023-06-01", help="YYYY-MM-DD start (default 2023-06-01)")
    ap.add_argument("--until", default=date.today().isoformat(), help="YYYY-MM-DD end (default today)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    since = date.fromisoformat(args.since)
    until = date.fromisoformat(args.until)
    print(f"Seeding worldcup advanced models from ESPN international results {since} → {until}")
    games = await gather_games(since, until)
    print(f"\nTotal completed international games: {len(games)}")
    poisson, xg, _form = seed(games, args.dry_run)
    report(poisson, xg)


if __name__ == "__main__":
    asyncio.run(main())
