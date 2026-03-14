"""Seed Elo and Form models for esports sectors (LoL, CS2, Valorant).

Data sources:
  LoL:      Leaguepedia MediaWiki API (free, no auth) — win/loss only
  CS2:      bo3.gg API (free, no auth) — series results
  Valorant: Leaguepedia Valorant wiki (free, no auth) — win/loss only

Poisson is NOT seeded for esports — it's tuned for scoring sports.
Elo + Form only.

Usage:
    python scripts/seed_esports.py                      # all esports
    python scripts/seed_esports.py --sectors lol,cs2    # specific
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evmax.agents.models.elo_agent import EloModelAgent
from evmax.agents.models.form_agent import FormModelAgent
from evmax.matching.normalizer import NameNormalizer


# ---------------------------------------------------------------------------
# League of Legends — Leaguepedia
# ---------------------------------------------------------------------------

# Major leagues to include (Leaguepedia tournament name substrings)
LOL_MAJOR_TOURNAMENTS = [
    "LCK", "LEC", "LCS", "LPL", "CBLOL", "LLA", "VCS",
    "First Stand", "MSI", "Worlds",
]


async def fetch_lol_games(client: httpx.AsyncClient, since: str = "2025-01-01") -> list[dict]:
    """Fetch LoL match results from Leaguepedia (win/loss only, no scores)."""
    url = "https://lol.fandom.com/api.php"
    games = []
    offset = 0
    batch = 500

    while True:
        try:
            r = await client.get(url, params={
                "action": "cargoquery",
                "tables": "ScoreboardGames",
                "fields": "Team1,Team2,WinTeam,DateTime_UTC,Tournament",
                "where": f"DateTime_UTC > '{since}' AND WinTeam IS NOT NULL AND WinTeam != ''",
                "order_by": "DateTime_UTC ASC",
                "limit": batch,
                "offset": offset,
                "format": "json",
            })
            r.raise_for_status()
            rows = r.json().get("cargoquery", [])
        except Exception as e:
            print(f"  WARN Leaguepedia fetch error: {e}")
            break

        if not rows:
            break

        for row in rows:
            t = row.get("title", row)
            team1 = (t.get("Team1") or "").strip()
            team2 = (t.get("Team2") or "").strip()
            winner = (t.get("WinTeam") or "").strip()
            dt = (t.get("DateTime UTC") or t.get("DateTime_UTC") or "")[:10]
            tournament = (t.get("Tournament") or "")

            if not team1 or not team2 or not winner or not dt:
                continue

            # Only include major tournaments to avoid noise from amateur leagues
            if not any(kw.lower() in tournament.lower() for kw in LOL_MAJOR_TOURNAMENTS):
                continue

            if winner == team1:
                score_home, score_away = 1.0, 0.0
            elif winner == team2:
                score_home, score_away = 0.0, 1.0
            else:
                continue

            games.append({
                "date": dt,
                "home": team1,
                "away": team2,
                "score_home": score_home,
                "score_away": score_away,
            })

        if len(rows) < batch:
            break
        offset += batch

    return games


# ---------------------------------------------------------------------------
# bo3.gg — shared helpers for CS2 and Valorant
# ---------------------------------------------------------------------------

BO3_GG_BASE = "https://api.bo3.gg/api/v1"
# discipline_id: 1=CS2, 2=Valorant, 3=LoL
BO3_DISCIPLINE = {"cs2": 1, "valorant": 2}


async def fetch_bo3_teams(client: httpx.AsyncClient, discipline_id: int) -> dict[int, str]:
    """Fetch team id → name mappings from bo3.gg for a specific discipline."""
    teams: dict[int, str] = {}
    offset = 0
    limit = 100
    while True:
        try:
            r = await client.get(
                f"{BO3_GG_BASE}/teams",
                params={
                    "filter[teams.discipline_id][eq]": discipline_id,
                    "page[limit]": limit,
                    "page[offset]": offset,
                },
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  WARN bo3.gg teams fetch error: {e}")
            break

        results = data.get("results", [])
        if not results:
            break
        for t in results:
            teams[t["id"]] = t.get("name", "")
        if len(results) < limit:
            break
        offset += limit

    return teams


async def fetch_bo3_games(
    client: httpx.AsyncClient,
    discipline_id: int,
    sector: str,
    since: str = "2025-01-01",
    teams: dict[int, str] | None = None,
) -> list[dict]:
    """Fetch completed series from bo3.gg for a given discipline."""
    if teams is None:
        print(f"  Fetching bo3.gg team list...")
        teams = await fetch_bo3_teams(client, discipline_id)
        print(f"  Got {len(teams)} teams")

    limit = 100
    # Get total match count for this discipline
    try:
        r = await client.get(
            f"{BO3_GG_BASE}/matches",
            params={"filter[matches.discipline_id][eq]": discipline_id, "page[limit]": 1, "page[offset]": 0},
        )
        total = r.json().get("total", {}).get("count", 0)
    except Exception:
        total = 5000

    print(f"  Total bo3.gg {sector} matches: {total}")

    # Work backwards from last page to find matches since cutoff
    last_offset = max(0, ((total // limit) * limit) - limit)
    games: list[dict] = []
    found_cutoff = False

    for offset in range(last_offset, max(-1, last_offset - 50 * limit), -limit):
        try:
            r = await client.get(
                f"{BO3_GG_BASE}/matches",
                params={
                    "filter[matches.discipline_id][eq]": discipline_id,
                    "page[limit]": limit,
                    "page[offset]": max(0, offset),
                },
            )
            r.raise_for_status()
            batch = r.json().get("results", [])
        except Exception as e:
            print(f"  WARN bo3.gg match fetch error offset {offset}: {e}")
            continue

        if not batch:
            break

        for m in batch:
            team1_id = m.get("team1_id")
            team2_id = m.get("team2_id")
            t1_score = m.get("team1_score", 0)
            t2_score = m.get("team2_score", 0)
            start_date = (m.get("start_date") or "")[:10]

            if not start_date or start_date < since:
                found_cutoff = True
                continue

            team1_name = teams.get(team1_id, "")
            team2_name = teams.get(team2_id, "")
            if not team1_name or not team2_name:
                continue
            if t1_score == 0 and t2_score == 0:
                continue

            games.append({
                "date": start_date,
                "home": team1_name,
                "away": team2_name,
                "score_home": float(t1_score),
                "score_away": float(t2_score),
            })

        if found_cutoff:
            break

    return games


async def fetch_cs2_games(client: httpx.AsyncClient, since: str = "2025-01-01") -> list[dict]:
    """Fetch CS2 series results from bo3.gg."""
    return await fetch_bo3_games(client, discipline_id=1, sector="cs2", since=since)


async def fetch_valorant_games(client: httpx.AsyncClient, since: str = "2025-01-01") -> list[dict]:
    """Fetch Valorant series results from bo3.gg."""
    return await fetch_bo3_games(client, discipline_id=2, sector="valorant", since=since)


# ---------------------------------------------------------------------------
# Seeding helpers
# ---------------------------------------------------------------------------

def seed_elo_form(games: list[dict], sector: str) -> int:
    """Feed games chronologically into Elo + Form agents."""
    elo = EloModelAgent()
    form = FormModelAgent()
    norm = NameNormalizer(sector)

    sorted_games = sorted(games, key=lambda g: g["date"])
    count = 0
    for g in sorted_games:
        home = norm.normalize(g["home"])
        away = norm.normalize(g["away"])
        if not home or not away:
            continue
        elo.update(home, away, g["score_home"], g["score_away"], sector, g["date"])
        form.update(home, away, g["score_home"], g["score_away"], sector, g["date"])
        count += 1

    elo.save_state()
    form.save_state()

    ratings = elo.all_ratings(sector)
    if ratings:
        top = sorted(ratings.items(), key=lambda x: x[1], reverse=True)[:8]
        print(f"  Top Elo: " + " | ".join(f"{t}: {r:.0f}" for t, r in top))

    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(sectors: list[str], since: str = "2025-01-01") -> None:
    print(f"Esports model seeder — sectors: {', '.join(sectors)}")
    print(f"Date: {date.today()}")

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(30.0),
        headers={"User-Agent": "evmax-seeder/1.0"},
        follow_redirects=True,
    ) as client:

        if "lol" in sectors:
            print(f"\n{'='*60}")
            print(f"  Seeding LOL from Leaguepedia")
            print(f"{'='*60}")
            games = await fetch_lol_games(client, since)
            print(f"  Raw games: {len(games)}")
            n = seed_elo_form(games, "lol")
            print(f"  Elo/Form: {n} games processed")

        if "cs2" in sectors:
            print(f"\n{'='*60}")
            print(f"  Seeding CS2 from bo3.gg")
            print(f"{'='*60}")
            games = await fetch_cs2_games(client, since)
            print(f"  Raw games: {len(games)}")
            n = seed_elo_form(games, "cs2")
            print(f"  Elo/Form: {n} games processed")

        if "valorant" in sectors:
            print(f"\n{'='*60}")
            print(f"  Seeding Valorant from bo3.gg")
            print(f"{'='*60}")
            games = await fetch_valorant_games(client, since)
            print(f"  Raw games: {len(games)}")
            n = seed_elo_form(games, "valorant")
            print(f"  Elo/Form: {n} games processed")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed esports models")
    parser.add_argument(
        "--sectors", "-s",
        default="lol,cs2,valorant",
        help="Comma-separated: lol, cs2, valorant (default: all)",
    )
    parser.add_argument(
        "--since",
        default="2025-01-01",
        help="Fetch results since this date (YYYY-MM-DD)",
    )
    args = parser.parse_args()
    sector_list = [s.strip().lower() for s in args.sectors.split(",") if s.strip()]
    asyncio.run(main(sector_list, since=args.since))
