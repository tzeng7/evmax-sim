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

# Major LoL tournaments. Matched as whole-word tokens (see `_is_major_lol_tournament`)
# so that substrings like "LCK Challengers League" do NOT count as LCK — academy/
# Challengers circuits had been polluting Elo with tier-2 teams ranked above T1.
LOL_MAJOR_TOKENS = {
    "lck", "lec", "lcs", "lpl", "cblol", "lla", "vcs",
    "msi", "worlds",
}
# Phrase-level majors (multi-word names that are uniquely top-tier)
LOL_MAJOR_PHRASES = [
    "first stand",
    "world championship",
    "mid-season invitational",
]
# Hard exclusions — tier-2 / academy / amateur circuits and regional splits that
# reuse major-league names.
LOL_EXCLUDE_KEYWORDS = [
    "challengers", "academy", "amateur", "scholastic", "collegiate",
    "development", "promotion", "proving grounds", "second division",
    "national", "superliga", "ultraliga", "nlc", "pg nationals",
    "esports balkan", "hitpoint", "elite series", "arabian league",
    "liga portuguesa", "greek legends",
]


# Whole-word tokens that mark a tier-2 circuit even when sitting next to a
# major-league abbreviation — "LCK CL", "LEC CL", etc.
LOL_EXCLUDE_TOKENS = {"cl"}  # "LCK CL" / "LEC CL" / etc.


def _is_major_lol_tournament(name: str) -> bool:
    """True iff `name` is a recognized tier-1 LoL event."""
    if not name:
        return False
    lowered = name.lower()
    if any(bad in lowered for bad in LOL_EXCLUDE_KEYWORDS):
        return False
    tokens = set(lowered.replace("/", " ").replace("-", " ").split())
    if tokens & LOL_EXCLUDE_TOKENS:
        return False
    if any(phrase in lowered for phrase in LOL_MAJOR_PHRASES):
        return True
    # whole-word token match — "LCK" matches "LCK Spring 2025" but not "LCK CL"
    return bool(tokens & LOL_MAJOR_TOKENS)


async def fetch_lol_games(client: httpx.AsyncClient, since: str = "2025-01-01") -> list[dict]:
    """Fetch LoL match results from Leaguepedia (win/loss only, no scores).

    Leaguepedia's MediaWiki API silently returns `{"error": {"code": "ratelimited"}}`
    with an empty cargoquery when anon rate limits trip. We detect that explicitly
    and back off, because the original seed was getting 0 rows without warning.
    """
    url = "https://lol.fandom.com/api.php"
    games: list[dict] = []
    offset = 0
    batch = 500
    backoff = 2.0

    # Server-side filter via LIKE on Tournament cuts the result set from
    # ~100k rows (all leagues) down to ~5k rows (majors only), so we finish
    # well before rate limits bite.
    major_likes = " OR ".join(
        f"Tournament LIKE '%{kw}%'" for kw in ("LCK","LEC","LCS","LPL","CBLOL","LLA","VCS","First Stand","MSI","World Championship")
    )
    where = (
        f"DateTime_UTC > '{since}' AND WinTeam IS NOT NULL AND WinTeam != '' "
        f"AND ({major_likes})"
    )

    while True:
        try:
            r = await client.get(url, params={
                "action": "cargoquery",
                "tables": "ScoreboardGames",
                "fields": "Team1,Team2,WinTeam,DateTime_UTC,Tournament",
                "where": where,
                "order_by": "DateTime_UTC ASC",
                "limit": batch,
                "offset": offset,
                "format": "json",
            })
            r.raise_for_status()
            payload = r.json()
        except Exception as e:
            print(f"  WARN Leaguepedia fetch error: {e}")
            break

        err = payload.get("error") if isinstance(payload, dict) else None
        if err:
            code = err.get("code", "")
            if code == "ratelimited":
                if backoff > 300:
                    print(f"  WARN Leaguepedia rate-limited (gave up after {backoff:.0f}s backoff)")
                    break
                print(f"  Leaguepedia rate-limited at offset {offset}, sleeping {backoff:.0f}s")
                await asyncio.sleep(backoff)
                backoff *= 2
                continue
            print(f"  WARN Leaguepedia error {code}: {err.get('info','')}")
            break

        # Reset backoff on success
        backoff = 2.0
        rows = payload.get("cargoquery", [])
        if not rows:
            break
        # Throttle to be a polite anon client
        await asyncio.sleep(0.5)

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
            if not _is_major_lol_tournament(tournament):
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
    tiers: tuple[str, ...] = ("s", "a"),
) -> list[dict]:
    """Fetch completed series from bo3.gg for a given discipline.

    Uses server-side `tier in {s,a}` filter to exclude tier-2/amateur matches
    (`b`/`c`), and sorts by `-start_date` to iterate newest → oldest, stopping
    when we hit the `since` cutoff. Previously used a blind offset-walk that
    pulled 5k amateur matches and left Vitality/Spirit/Sentinels off the
    Elo leaderboard entirely.
    """
    if teams is None:
        print(f"  Fetching bo3.gg team list...")
        teams = await fetch_bo3_teams(client, discipline_id)
        print(f"  Got {len(teams)} teams")

    limit = 100
    games: list[dict] = []
    offset = 0
    tier_filter = ",".join(tiers)

    while True:
        try:
            r = await client.get(
                f"{BO3_GG_BASE}/matches",
                params={
                    "filter[matches.discipline_id][eq]": discipline_id,
                    "filter[matches.tier][in]": tier_filter,
                    "sort": "-start_date",
                    "page[limit]": limit,
                    "page[offset]": offset,
                },
            )
            r.raise_for_status()
            payload = r.json()
        except Exception as e:
            print(f"  WARN bo3.gg match fetch error offset {offset}: {e}")
            break

        batch = payload.get("results", [])
        if not batch:
            break

        hit_cutoff = False
        for m in batch:
            start_date = (m.get("start_date") or "")[:10]
            if not start_date:
                continue
            if start_date < since:
                hit_cutoff = True
                continue

            team1_id = m.get("team1_id")
            team2_id = m.get("team2_id")
            if not team1_id or not team2_id:
                continue

            t1_score = m.get("team1_score", 0) or 0
            t2_score = m.get("team2_score", 0) or 0
            if t1_score == 0 and t2_score == 0:
                continue  # unplayed / walkover
            if m.get("winner_team_id") is None:
                continue

            team1_name = teams.get(team1_id, "")
            team2_name = teams.get(team2_id, "")
            if not team1_name or not team2_name:
                continue

            games.append({
                "date": start_date,
                "home": team1_name,
                "away": team2_name,
                "score_home": float(t1_score),
                "score_away": float(t2_score),
            })

        if hit_cutoff:
            break
        if len(batch) < limit:
            break
        offset += limit

    return games


async def fetch_cs2_games(client: httpx.AsyncClient, since: str = "2025-01-01") -> list[dict]:
    """Fetch CS2 series results from bo3.gg.

    s + a tiers — s covers Tier-1 circuits (BLAST, IEM, ESL Pro League),
    a covers national leagues. Empirically yields Vitality/Spirit/NAVI at top.
    """
    return await fetch_bo3_games(client, discipline_id=1, sector="cs2", since=since, tiers=("s", "a"))


async def fetch_valorant_games(client: httpx.AsyncClient, since: str = "2025-01-01") -> list[dict]:
    """Fetch Valorant series results from bo3.gg.

    s-tier only — a-tier is VCT Challengers/Ascension which fragments the
    rating pool with regional teams that never face VCT-International opponents
    but still inflate Elo by beating other Challengers teams.
    """
    return await fetch_bo3_games(client, discipline_id=2, sector="valorant", since=since, tiers=("s",))


# ---------------------------------------------------------------------------
# Seeding helpers
# ---------------------------------------------------------------------------

def seed_elo_form(games: list[dict], sector: str, reset: bool = True) -> int:
    """Feed games chronologically into Elo + Form agents.

    If `reset=True` (default), drops any existing state for `sector` before
    replaying. This prevents stale tier-2 teams from a prior seed pass
    lingering in the leaderboard when the new filter excludes them.
    """
    elo = EloModelAgent()
    form = FormModelAgent()
    norm = NameNormalizer(sector)

    if reset:
        if sector in elo._state:
            elo._state[sector] = {"ratings": {}, "game_counts": {}, "h2h": {}}
        if sector in form._state:
            form._state[sector] = {}

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
