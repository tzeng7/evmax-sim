"""Seed Elo, Form, and Poisson models from ESPN public API.

Covers: NBA, NFL, NCAAB, MLB Baseball, Soccer (EPL, La Liga, Bundesliga, Serie A, Ligue 1, UCL).
No API key required — ESPN's public scoreboard endpoint.

Usage:
    python scripts/seed_espn.py                         # all sectors
    python scripts/seed_espn.py --sectors nba,nfl       # specific sectors
    python scripts/seed_espn.py --sectors baseball      # baseball only
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import httpx

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evmax.agents.models.elo_agent import EloModelAgent
from evmax.agents.models.form_agent import FormModelAgent
from evmax.agents.models.poisson_agent import PoissonModelAgent
from evmax.matching.normalizer import NameNormalizer

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"


def _months(start: str, end: str) -> list[str]:
    """Generate YYYYMM strings from start to end inclusive."""
    sy, sm = int(start[:4]), int(start[5:7])
    ey, em = int(end[:4]), int(end[5:7])
    months = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        months.append(f"{y}{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


# Each entry: sector, ESPN sport, ESPN league, season months to fetch
SECTOR_CONFIGS: dict[str, dict] = {
    "nba": {
        "sport": "basketball",
        "league": "nba",
        "months": _months("2025-10", "2026-03"),
    },
    "nfl": {
        "sport": "football",
        "league": "nfl",
        "months": _months("2025-09", "2026-02"),
    },
    "ncaab": {
        "sport": "basketball",
        "league": "mens-college-basketball",
        "months": _months("2025-11", "2026-03"),
    },
    "baseball": {
        "sport": "baseball",
        "league": "mlb",
        "months": _months("2025-03", "2025-10"),  # 2025 MLB season
    },
    "ufc": {
        "sport": "mma",
        "league": "ufc",
        "months": _months("2025-01", "2026-03"),
    },
    "f1": {
        "sport": "racing",
        "league": "f1",
        "months": _months("2025-03", "2026-03"),  # 2025 F1 season
    },
    "soccer": {
        "leagues": {
            "epl":        ("soccer", "eng.1"),
            "laliga":     ("soccer", "esp.1"),
            "bundesliga": ("soccer", "ger.1"),
            "serie_a":    ("soccer", "ita.1"),
            "ligue1":     ("soccer", "fra.1"),
            "ucl":        ("soccer", "UEFA.CHAMPIONS"),
        },
        "months": _months("2025-08", "2026-03"),
    },
}


async def fetch_espn_games(
    client: httpx.AsyncClient,
    sport: str,
    league: str,
    month: str,
) -> list[dict]:
    """Fetch all completed games for a sport/league in a given YYYYMM month."""
    url = f"{ESPN_BASE}/{sport}/{league}/scoreboard"
    games = []
    page = 1
    while True:
        try:
            r = await client.get(url, params={"dates": month, "limit": 100, "page": page})
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"    WARN: ESPN fetch failed ({sport}/{league} {month} p{page}): {e}")
            break

        events = data.get("events", [])
        if not events:
            break

        for event in events:
            competition = event.get("competitions", [{}])[0]
            status = competition.get("status", {}).get("type", {})
            if not status.get("completed", False):
                continue

            competitors = competition.get("competitors", [])
            if len(competitors) < 2:
                continue

            home = next((c for c in competitors if c.get("homeAway") == "home"), None)
            away = next((c for c in competitors if c.get("homeAway") == "away"), None)
            if not home or not away:
                continue

            try:
                score_home = float(home.get("score", 0))
                score_away = float(away.get("score", 0))
            except (ValueError, TypeError):
                continue

            date_str = event.get("date", "")[:10]  # YYYY-MM-DD

            games.append({
                "date": date_str,
                "home": home["team"].get("displayName", ""),
                "away": away["team"].get("displayName", ""),
                "score_home": score_home,
                "score_away": score_away,
            })

        # ESPN doesn't paginate scoreboard the same way — one page per month is typical
        break

    return games


def normalize_games(games: list[dict], sector: str) -> list[dict]:
    """Normalize team names using the sector normalizer."""
    norm = NameNormalizer(sector)
    normalized = []
    for g in games:
        hn = norm.normalize(g["home"])
        an = norm.normalize(g["away"])
        if not hn or not an:
            continue
        normalized.append({**g, "home": hn, "away": an})
    return normalized


def seed_elo_form(games: list[dict], sector: str, elo_agent: EloModelAgent, form_agent: FormModelAgent) -> int:
    """Feed games chronologically into Elo + Form agents. Returns game count."""
    sorted_games = sorted(games, key=lambda g: g["date"])
    for g in sorted_games:
        elo_agent.update(g["home"], g["away"], g["score_home"], g["score_away"], sector, g["date"])
        form_agent.update(g["home"], g["away"], g["score_home"], g["score_away"], sector, g["date"])
    return len(sorted_games)


def seed_poisson(games: list[dict], sector: str, poisson_agent: PoissonModelAgent) -> int:
    """Compute and seed Poisson attack/defense stats from game results."""
    if not games:
        return 0

    home_scored: dict[str, list] = defaultdict(list)
    away_scored: dict[str, list] = defaultdict(list)
    home_conceded: dict[str, list] = defaultdict(list)
    away_conceded: dict[str, list] = defaultdict(list)

    for g in games:
        home_scored[g["home"]].append(g["score_home"])
        away_scored[g["away"]].append(g["score_away"])
        home_conceded[g["home"]].append(g["score_away"])
        away_conceded[g["away"]].append(g["score_home"])

    all_home = [g["score_home"] for g in games]
    all_away = [g["score_away"] for g in games]
    league_avg_home = sum(all_home) / len(all_home) if all_home else 1.0
    league_avg_away = sum(all_away) / len(all_away) if all_away else 1.0
    league_avg = (league_avg_home + league_avg_away) / 2

    teams = {}
    all_team_names = set(home_scored) | set(away_scored)
    for team in all_team_names:
        hs = home_scored.get(team, [])
        as_ = away_scored.get(team, [])
        hc = home_conceded.get(team, [])
        ac = away_conceded.get(team, [])
        total_games = len(hs) + len(as_)
        if total_games < 3:
            continue
        total_scored = sum(hs) + sum(as_)
        total_conceded = sum(hc) + sum(ac)
        avg_scored = total_scored / total_games
        avg_conceded = total_conceded / total_games
        teams[team] = {
            "attack": round(avg_scored / league_avg, 4),
            "defense": round(avg_conceded / league_avg, 4),
            "games": total_games,
        }

    poisson_agent.seed_team_stats(
        sector=sector,
        team_stats=teams,
        league_avg={"home": round(league_avg_home, 3), "away": round(league_avg_away, 3)},
    )
    return len(teams)


async def seed_standard_sector(sector: str, cfg: dict, client: httpx.AsyncClient) -> None:
    """Seed a single non-soccer sector (NBA, NFL, NCAAB)."""
    sport = cfg["sport"]
    league = cfg["league"]
    months = cfg["months"]

    print(f"\n{'='*60}")
    print(f"  Seeding {sector.upper()} from ESPN ({len(months)} months)")
    print(f"{'='*60}")

    all_games: list[dict] = []
    for month in months:
        games = await fetch_espn_games(client, sport, league, month)
        all_games.extend(games)
        print(f"  {month}: {len(games)} completed games")

    print(f"  Total raw games: {len(all_games)}")
    normalized = normalize_games(all_games, sector)
    print(f"  Normalized: {len(normalized)} games")

    if not normalized:
        print(f"  WARN: No normalized games for {sector}")
        return

    elo = EloModelAgent()
    form = FormModelAgent()
    poisson = PoissonModelAgent()

    n_elo = seed_elo_form(normalized, sector, elo, form)
    n_poisson = seed_poisson(normalized, sector, poisson)

    elo.save_state()
    form.save_state()
    poisson.save_state()

    print(f"  Elo/Form: {n_elo} games processed")
    print(f"  Poisson: {n_poisson} teams seeded")

    # Print top Elo ratings
    ratings = elo.all_ratings(sector)
    if ratings:
        top = sorted(ratings.items(), key=lambda x: x[1], reverse=True)[:8]
        print(f"  Top Elo: " + " | ".join(f"{t}: {r:.0f}" for t, r in top))


async def seed_soccer(cfg: dict, client: httpx.AsyncClient) -> None:
    """Seed soccer across all configured leagues, merged into one sector."""
    months = cfg["months"]
    leagues = cfg["leagues"]
    sector = "soccer"

    print(f"\n{'='*60}")
    print(f"  Seeding SOCCER from ESPN ({len(leagues)} leagues, {len(months)} months)")
    print(f"{'='*60}")

    all_games: list[dict] = []
    for league_name, (sport, espn_league) in leagues.items():
        league_games: list[dict] = []
        for month in months:
            games = await fetch_espn_games(client, sport, espn_league, month)
            league_games.extend(games)
        print(f"  {league_name}: {len(league_games)} completed games")
        all_games.extend(league_games)

    print(f"  Total raw games: {len(all_games)}")
    normalized = normalize_games(all_games, sector)
    print(f"  Normalized: {len(normalized)} games")

    if not normalized:
        print(f"  WARN: No normalized games for soccer")
        return

    elo = EloModelAgent()
    form = FormModelAgent()
    poisson = PoissonModelAgent()

    n_elo = seed_elo_form(normalized, sector, elo, form)
    n_poisson = seed_poisson(normalized, sector, poisson)

    elo.save_state()
    form.save_state()
    poisson.save_state()

    print(f"  Elo/Form: {n_elo} games processed")
    print(f"  Poisson: {n_poisson} teams seeded")

    ratings = elo.all_ratings(sector)
    if ratings:
        top = sorted(ratings.items(), key=lambda x: x[1], reverse=True)[:8]
        print(f"  Top Elo: " + " | ".join(f"{t}: {r:.0f}" for t, r in top))


async def seed_ufc(cfg: dict, client: httpx.AsyncClient) -> None:
    """
    Seed UFC fighter Elo + Form from fight results.

    ESPN MMA uses an event-based API — each event contains individual bouts.
    We fetch the events list, then each event's competitors (winner/loser).
    """
    sector = "ufc"
    months = cfg["months"]

    print(f"\n{'='*60}")
    print(f"  Seeding UFC from ESPN ({len(months)} months)")
    print(f"{'='*60}")

    elo = EloModelAgent()
    form = FormModelAgent()
    norm = NameNormalizer(sector)
    total_bouts = 0

    for month in months:
        # ESPN MMA scoreboard with calendar=true returns events by month
        url = f"{ESPN_BASE}/mma/ufc/scoreboard"
        try:
            r = await client.get(url, params={"dates": month, "limit": 100, "calendar": "true"})
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"    WARN: ESPN UFC fetch failed ({month}): {e}")
            continue

        events = data.get("events", [])
        bouts_this_month = 0

        for event in events:
            comps = event.get("competitions", [])
            date_str = event.get("date", "")[:10]

            for comp in comps:
                if not comp.get("status", {}).get("type", {}).get("completed"):
                    continue
                competitors = comp.get("competitors", [])
                if len(competitors) < 2:
                    continue

                winner = next((c for c in competitors if c.get("winner")), None)
                loser  = next((c for c in competitors if not c.get("winner")), None)
                if not winner or not loser:
                    continue

                w_name = norm.normalize(
                    winner.get("athlete", {}).get("displayName") or winner.get("displayName", "")
                )
                l_name = norm.normalize(
                    loser.get("athlete", {}).get("displayName") or loser.get("displayName", "")
                )
                if not w_name or not l_name:
                    continue

                elo.update(w_name, l_name, 1.0, 0.0, sector, date_str)
                form.update(w_name, l_name, 1.0, 0.0, sector, date_str)
                total_bouts += 1
                bouts_this_month += 1

        if bouts_this_month:
            print(f"  {month}: {bouts_this_month} bouts")

    elo.save_state()
    form.save_state()

    ratings = elo.all_ratings(sector)
    if ratings:
        top = sorted(ratings.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"  Top Elo: " + " | ".join(f"{t}: {r:.0f}" for t, r in top))
    print(f"  Total bouts seeded: {total_bouts}")


async def seed_f1(cfg: dict, client: httpx.AsyncClient) -> None:
    """
    Seed F1 driver Elo + Form from race results.

    F1 races return finishing positions for each driver. We convert each race
    into pairwise head-to-head results: driver who finished higher 'beats'
    adjacent finisher. This builds meaningful Elo ratings from race positions.
    """
    sport, league = cfg["sport"], cfg["league"]
    months = cfg["months"]
    sector = "f1"

    print(f"\n{'='*60}")
    print(f"  Seeding F1 from ESPN ({len(months)} months)")
    print(f"{'='*60}")

    elo = EloModelAgent()
    form = FormModelAgent()
    norm = NameNormalizer(sector)
    total_pairs = 0

    for month in months:
        url = f"{ESPN_BASE}/{sport}/{league}/scoreboard"
        try:
            r = await client.get(url, params={"dates": month, "limit": 100})
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"    WARN: ESPN F1 fetch failed ({month}): {e}")
            continue

        events = data.get("events", [])
        races_this_month = 0

        for event in events:
            comp = (event.get("competitions") or [{}])[0]
            if not comp.get("status", {}).get("type", {}).get("completed"):
                continue

            date_str = event.get("date", "")[:10]
            competitors = comp.get("competitors", [])

            # Build sorted finishing order
            finishers: list[tuple[int, str]] = []
            for c in competitors:
                pos_str = c.get("order") or c.get("rank") or c.get("score")
                try:
                    pos = int(pos_str)
                except (TypeError, ValueError):
                    continue
                name_raw = c.get("athlete", {}).get("displayName") or c.get("displayName", "")
                name = norm.normalize(name_raw)
                if name:
                    finishers.append((pos, name))

            finishers.sort(key=lambda x: x[0])

            # Pairwise: each driver beats all drivers who finished behind them
            # To keep updates tractable, only use adjacent pairs (1v2, 2v3, 3v4, ...)
            for i in range(len(finishers) - 1):
                pos_a, driver_a = finishers[i]
                pos_b, driver_b = finishers[i + 1]
                elo.update(driver_a, driver_b, 1.0, 0.0, sector, date_str)
                form.update(driver_a, driver_b, 1.0, 0.0, sector, date_str)
                total_pairs += 1

            races_this_month += 1

        if races_this_month:
            print(f"  {month}: {races_this_month} race(s), {total_pairs} total pairwise updates so far")

    elo.save_state()
    form.save_state()

    ratings = elo.all_ratings(sector)
    if ratings:
        top = sorted(ratings.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"  Top Elo: " + " | ".join(f"{t}: {r:.0f}" for t, r in top))
    print(f"  Total pairwise updates: {total_pairs}")


async def main(sectors: list[str], since: str | None = None) -> None:
    print(f"ESPN model seeder — sectors: {', '.join(sectors)}")
    print(f"Date: {date.today()}")

    # Override hardcoded month ranges if --since is provided
    if since:
        today = date.today()
        since_months = _months(since[:7], f"{today.year}-{today.month:02d}")
        for cfg in SECTOR_CONFIGS.values():
            if "months" in cfg:
                cfg["months"] = since_months
            if "leagues" in cfg:
                # soccer sub-config inherits same months
                cfg["months"] = since_months

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(20.0),
        headers={"User-Agent": "evmax-seeder/1.0"},
        follow_redirects=True,
    ) as client:
        for sector in sectors:
            if sector not in SECTOR_CONFIGS:
                print(f"WARN: Unknown sector {sector!r}, skipping")
                continue
            cfg = SECTOR_CONFIGS[sector]
            if sector == "soccer":
                await seed_soccer(cfg, client)
            elif sector == "f1":
                await seed_f1(cfg, client)
            elif sector == "ufc":
                await seed_ufc(cfg, client)
            else:
                await seed_standard_sector(sector, cfg, client)

    print("\nDone. Run 'evmax agents scan' to see model contributions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed models from ESPN data")
    parser.add_argument(
        "--sectors", "-s",
        default="nba,nfl,ncaab,baseball,ufc,f1,soccer",
        help="Comma-separated sectors to seed (default: nba,nfl,ncaab,baseball,ufc,f1,soccer)",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="Seed from this month onward (YYYY-MM-DD or YYYY-MM). Overrides default date ranges.",
    )
    args = parser.parse_args()
    sector_list = [s.strip().lower() for s in args.sectors.split(",") if s.strip()]
    asyncio.run(main(sector_list, since=args.since))
