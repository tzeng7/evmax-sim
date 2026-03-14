"""ResultsResolver — fetches actual game outcomes and writes to ev_outcomes.

Data sources:
  NBA / NFL / NCAAB  — ESPN scoreboard API (public, no auth)
  Soccer             — ESPN soccer league scoreboards
  CS2 / Valorant     — bo3.gg matches API
  LoL                — bo3.gg discipline_id=3

Matching strategy:
  ESPN: normalize team display names and compare against yes_team + event_title.
  bo3.gg: match via team name substring in event_id slug.
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import date, timedelta
from typing import Optional

import httpx
import structlog

from evmax.agents.cleanup.db import get_connection

logger = structlog.get_logger(__name__)

# ESPN sport → (sport_path, league_path)
ESPN_SPORT_MAP: dict[str, tuple[str, str]] = {
    "nba":   ("basketball", "nba"),
    "ncaab": ("basketball", "mens-college-basketball"),
    "nfl":   ("football", "nfl"),
}

# Soccer league slugs for ESPN
ESPN_SOCCER_LEAGUES = [
    "eng.1",       # EPL
    "esp.1",       # La Liga
    "ger.1",       # Bundesliga
    "ita.1",       # Serie A
    "fra.1",       # Ligue 1
    "uefa.champions",  # UCL
]

# bo3.gg discipline IDs
BO3_BASE = "https://api.bo3.gg/api/v1"
BO3_DISCIPLINE: dict[str, int] = {"cs2": 1, "valorant": 2, "lol": 3}


# ---------------------------------------------------------------------------
# ESPN helpers
# ---------------------------------------------------------------------------

async def _fetch_espn_scores(
    client: httpx.AsyncClient, sport: str, league: str, espn_date: str
) -> list[dict]:
    """Fetch completed game scores from ESPN (date as YYYYMMDD)."""
    url = f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard"
    try:
        r = await client.get(url, params={"dates": espn_date, "limit": 200})
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.warning("espn_fetch_failed", sport=sport, league=league, error=str(e))
        return []

    results = []
    for event in data.get("events", []):
        comps = event.get("competitions", [])
        if not comps:
            continue
        comp = comps[0]
        status = comp.get("status", {}).get("type", {})
        if not status.get("completed"):
            continue

        competitors = comp.get("competitors", [])
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue

        try:
            home_score = int(home.get("score", 0))
            away_score = int(away.get("score", 0))
        except (ValueError, TypeError):
            continue

        results.append({
            "home_name": home.get("team", {}).get("displayName", ""),
            "away_name": away.get("team", {}).get("displayName", ""),
            "home_score": home_score,
            "away_score": away_score,
            "home_won": home_score > away_score,
        })

    return results


# ---------------------------------------------------------------------------
# bo3.gg helpers
# ---------------------------------------------------------------------------

async def _fetch_bo3_scores(
    client: httpx.AsyncClient, sector: str, target_date: date
) -> list[dict]:
    """Fetch completed series from bo3.gg for a given sector and date."""
    discipline_id = BO3_DISCIPLINE.get(sector, 1)
    date_str = target_date.isoformat()
    try:
        r = await client.get(
            f"{BO3_BASE}/matches",
            params={
                "filter[matches.discipline_id][eq]": discipline_id,
                "filter[matches.start_date][gte]": date_str,
                "filter[matches.start_date][lte]": date_str,
                "page[limit]": 200,
                "page[offset]": 0,
            },
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.warning("bo3_scores_failed", sector=sector, error=str(e))
        return []

    results = []
    for m in data.get("results", []):
        t1_score = m.get("team1_score", 0) or 0
        t2_score = m.get("team2_score", 0) or 0
        if t1_score == 0 and t2_score == 0:
            continue
        results.append({
            "team1_id":  m.get("team1_id"),
            "team2_id":  m.get("team2_id"),
            "team1_score": t1_score,
            "team2_score": t2_score,
            "team1_won": t1_score > t2_score,
        })

    return results


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------

def _norm(name: str) -> str:
    return name.lower().replace(" ", "_").replace(".", "").replace("-", "_")


def _match_espn(pred: dict, scores: list[dict]) -> Optional[int]:
    """Return 1 (YES won) / 0 (YES lost) / None (no match)."""
    yes_team = _norm(pred["yes_team"])
    title = pred.get("event_title", "")

    title_teams: list[str] = []
    if " vs " in title:
        title_teams = [_norm(p.strip()) for p in title.split(" vs ")]

    for score in scores:
        home_n = _norm(score["home_name"])
        away_n = _norm(score["away_name"])

        # Quick gate: one of the title teams must appear in the score
        if title_teams:
            if not any(t in home_n or home_n in t or t in away_n or away_n in t
                       for t in title_teams):
                continue

        yes_is_home = yes_team in home_n or home_n in yes_team
        yes_is_away = yes_team in away_n or away_n in yes_team

        if yes_is_home:
            return 1 if score["home_won"] else 0
        if yes_is_away:
            return 1 if not score["home_won"] else 0

    return None


def _match_bo3(pred: dict, scores: list[dict]) -> Optional[int]:
    """Return 1 / 0 / None — matches via event_id slug."""
    # event_id: "cs2::2026-03-14::navi_vs_faze"
    parts = pred.get("event_id", "").split("::")
    if len(parts) < 3 or "_vs_" not in parts[2]:
        return None

    slug_a, slug_b = parts[2].split("_vs_", 1)
    yes_team = _norm(pred["yes_team"])

    for score in scores:
        yes_is_a = yes_team in slug_a or slug_a in yes_team
        yes_is_b = yes_team in slug_b or slug_b in yes_team

        if yes_is_a:
            return 1 if score["team1_won"] else 0
        if yes_is_b:
            return 1 if not score["team1_won"] else 0

    return None


def _write_outcome(conn: sqlite3.Connection, pred: dict, outcome: int, source: str) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO ev_outcomes
           (market_id, event_id, event_date, sector, yes_team,
            outcome, sharp_true_prob, blended_true_prob, resolved_at, result_source)
           VALUES (?,?,?,?,?,?,?,?,datetime('now'),?)""",
        (
            pred["market_id"],
            pred["event_id"],
            pred.get("event_date", ""),
            pred["sector"],
            pred["yes_team"],
            outcome,
            pred.get("sharp_true_prob"),
            pred.get("blended_true_prob"),
            source,
        ),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def resolve_outcomes_for_date(target_date: Optional[date] = None) -> dict:
    """
    Fetch and store outcomes for all pending predictions on target_date.

    Returns {"resolved": int, "failed": int}.
    """
    target_date = target_date or (date.today() - timedelta(days=1))
    date_str = target_date.isoformat()
    espn_date = target_date.strftime("%Y%m%d")

    conn = get_connection()
    pending = conn.execute(
        """SELECT p.market_id, p.event_id, p.sector, p.yes_team,
                  p.event_title, p.event_date, p.sharp_true_prob, p.blended_true_prob
           FROM ev_predictions p
           LEFT JOIN ev_outcomes o ON p.market_id = o.market_id
           WHERE (p.event_date = ? OR p.scan_date = ?)
             AND o.market_id IS NULL""",
        (date_str, date_str),
    ).fetchall()

    if not pending:
        logger.info("no_pending_outcomes", date=date_str)
        conn.close()
        return {"resolved": 0, "failed": 0}

    logger.info("resolving_outcomes", count=len(pending), date=date_str)

    by_sector: dict[str, list[dict]] = {}
    for row in pending:
        d = dict(row)
        by_sector.setdefault(d["sector"], []).append(d)

    resolved = 0
    failed = 0

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(20.0),
        headers={"User-Agent": "evmax-cleanup/1.0"},
        follow_redirects=True,
    ) as client:

        for sector, rows in by_sector.items():
            if sector in ESPN_SPORT_MAP:
                sport, league = ESPN_SPORT_MAP[sector]
                scores = await _fetch_espn_scores(client, sport, league, espn_date)
                for pred in rows:
                    outcome = _match_espn(pred, scores)
                    if outcome is not None:
                        _write_outcome(conn, pred, outcome, "espn")
                        resolved += 1
                    else:
                        failed += 1

            elif sector == "soccer":
                scores: list[dict] = []
                for espn_league in ESPN_SOCCER_LEAGUES:
                    scores.extend(
                        await _fetch_espn_scores(client, "soccer", espn_league, espn_date)
                    )
                for pred in rows:
                    outcome = _match_espn(pred, scores)
                    if outcome is not None:
                        _write_outcome(conn, pred, outcome, "espn")
                        resolved += 1
                    else:
                        failed += 1

            elif sector in BO3_DISCIPLINE:
                scores = await _fetch_bo3_scores(client, sector, target_date)
                for pred in rows:
                    outcome = _match_bo3(pred, scores)
                    if outcome is not None:
                        _write_outcome(conn, pred, outcome, "bo3gg")
                        resolved += 1
                    else:
                        failed += 1

            else:
                logger.warning("no_resolver_for_sector", sector=sector)
                failed += len(rows)

    conn.commit()
    conn.close()

    logger.info("resolve_done", date=date_str, resolved=resolved, failed=failed)
    return {"resolved": resolved, "failed": failed}
