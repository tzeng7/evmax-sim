"""Seed the SoccerXgAgent with shot data from ESPN box scores.

Fetches ESPN scoreboard data for all configured soccer leagues going back
to `--since` (default 2026-01-01), extracts shotsOnTarget / totalShots per
team, and feeds them into SoccerXgAgent.record_match().

Usage:
    python scripts/seed_soccer_xg.py
    python scripts/seed_soccer_xg.py --since 2025-09-01
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date, timedelta
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evmax.agents.models.soccer_xg_agent import SoccerXgAgent

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"

SOCCER_LEAGUES = [
    ("soccer", "eng.1"),     # EPL
    ("soccer", "esp.1"),     # La Liga
    ("soccer", "ger.1"),     # Bundesliga
    ("soccer", "ita.1"),     # Serie A
    ("soccer", "fra.1"),     # Ligue 1
    ("soccer", "uefa.champions"),  # UCL
    ("soccer", "usa.1"),     # MLS
    ("soccer", "uefa.europa"),     # UEL
]


def _stat(competitor: dict, name: str) -> int:
    for s in competitor.get("statistics", []):
        if s.get("name") == name:
            try:
                return int(float(s.get("displayValue", 0)))
            except (ValueError, TypeError):
                pass
    return 0


async def fetch_league_matches(
    client: httpx.AsyncClient,
    sport: str,
    league: str,
    since: str,
) -> list[dict]:
    """Fetch completed matches with shot stats from ESPN."""
    matches = []
    current = date.fromisoformat(since)
    today = date.today()

    while current <= today:
        month_str = current.strftime("%Y%m")
        url = f"{ESPN_BASE}/{sport}/{league}/scoreboard"
        try:
            r = await client.get(url, params={"dates": month_str, "limit": 200})
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  WARN: {league} {month_str}: {e}")
            current = (current.replace(day=1) + timedelta(days=32)).replace(day=1)
            continue

        for event in data.get("events", []):
            comp = event.get("competitions", [{}])[0]
            status = comp.get("status", {}).get("type", {})
            if not status.get("completed"):
                continue

            competitors = comp.get("competitors", [])
            home = next((c for c in competitors if c.get("homeAway") == "home"), None)
            away = next((c for c in competitors if c.get("homeAway") == "away"), None)
            if not home or not away:
                continue

            event_date = event.get("date", "")[:10]
            if event_date < since:
                continue

            home_sot = _stat(home, "shotsOnTarget")
            away_sot = _stat(away, "shotsOnTarget")
            home_shots = _stat(home, "totalShots")
            away_shots = _stat(away, "totalShots")

            if home_shots == 0 and away_shots == 0:
                continue

            try:
                home_score = int(home.get("score", 0))
                away_score = int(away.get("score", 0))
            except (ValueError, TypeError):
                continue

            matches.append({
                "date": event_date,
                "home": home.get("team", {}).get("displayName", ""),
                "away": away.get("team", {}).get("displayName", ""),
                "home_score": home_score,
                "away_score": away_score,
                "home_sot": home_sot,
                "away_sot": away_sot,
                "home_shots": home_shots,
                "away_shots": away_shots,
            })

        current = (current.replace(day=1) + timedelta(days=32)).replace(day=1)
        await asyncio.sleep(0.3)

    return matches


async def main(since: str) -> None:
    agent = SoccerXgAgent()
    agent._state = {"teams": {}}

    print(f"Soccer xG seeder — since {since}")
    total = 0

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(30.0),
        headers={"User-Agent": "evmax-seeder/1.0"},
        follow_redirects=True,
    ) as client:
        for sport, league in SOCCER_LEAGUES:
            print(f"\n  {league}...")
            matches = await fetch_league_matches(client, sport, league, since)
            matches.sort(key=lambda m: m["date"])
            print(f"    {len(matches)} matches with shot data")

            for m in matches:
                agent.record_match(
                    team=m["home"], goals_for=m["home_score"], goals_against=m["away_score"],
                    shots_on_target=m["home_sot"], total_shots=m["home_shots"],
                    opponent_sot=m["away_sot"], opponent_shots=m["away_shots"],
                    match_date=m["date"], is_home=True,
                )
                agent.record_match(
                    team=m["away"], goals_for=m["away_score"], goals_against=m["home_score"],
                    shots_on_target=m["away_sot"], total_shots=m["away_shots"],
                    opponent_sot=m["home_sot"], opponent_shots=m["home_shots"],
                    match_date=m["date"], is_home=False,
                )
                total += 1

    agent.save_state()
    teams = agent._state.get("teams", {})
    print(f"\nSeeded {total} matches across {len(teams)} teams")

    # Show top teams by xG/game
    ranked = []
    for team, data in teams.items():
        matches = data.get("matches", [])[:10]
        if len(matches) < 4:
            continue
        xg_avg = sum(m["xg"] for m in matches) / len(matches)
        goals_avg = sum(m["goals_for"] for m in matches) / len(matches)
        ranked.append((team, xg_avg, goals_avg, len(matches)))
    ranked.sort(key=lambda x: x[1], reverse=True)
    print("\nTop by xG/game (last 10):")
    for team, xg, goals, n in ranked[:12]:
        ratio = goals / xg if xg > 0 else 0
        flag = " ↓REGRESS" if ratio > 1.3 else " ↑UNLUCKY" if ratio < 0.7 else ""
        print(f"  {team:25s}  xG={xg:.2f}  goals={goals:.2f}  ratio={ratio:.2f}{flag}")
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed soccer xG model")
    parser.add_argument("--since", default="2026-01-01", help="Fetch since (YYYY-MM-DD)")
    args = parser.parse_args()
    asyncio.run(main(args.since))
