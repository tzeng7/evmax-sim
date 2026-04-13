"""ResultsResolver — fetches actual game outcomes and writes to ev_outcomes.

Data sources:
  NBA / NFL / NCAAB  — ESPN scoreboard API (public, no auth)
  Soccer             — ESPN soccer league scoreboards
  CS2 / Valorant     — bo3.gg matches API
  LoL                — bo3.gg discipline_id=3

Matching strategy:
  Primary team names are extracted from the event_id slug
  (e.g. "soccer::2026-03-18::atletico_madrid_vs_real_madrid" → ["atletico madrid", "real madrid"]).
  Names are fuzzy-matched against ESPN displayName / bo3.gg team names using
  rapidfuzz token_sort_ratio at a lenient threshold (72) — lower than the 88
  used at scan time because we have far fewer candidates post-hoc.
  Falls back to yes_team direct matching when the slug is unavailable.
"""

from __future__ import annotations

import asyncio
import sqlite3
import unicodedata
from datetime import date, timedelta
from typing import Optional

import httpx
import structlog
from rapidfuzz import fuzz

from evmax.agents.cleanup.db import get_connection

logger = structlog.get_logger(__name__)

# Lenient threshold for post-hoc resolution (fewer candidates, controlled domain)
FUZZY_THRESHOLD = 72

# ESPN sport → (sport_path, league_path, extra_params)
ESPN_SPORT_MAP: dict[str, tuple[str, str, dict]] = {
    "nba":      ("basketball", "nba", {}),
    # groups=50 includes NIT + NCAA tournament first-four games that are
    # excluded from the default (groups=1) scoreboard.
    "ncaab":    ("basketball", "mens-college-basketball", {"groups": "50"}),
    "ncaaw":    ("basketball", "womens-college-basketball", {"groups": "50"}),
    "nfl":      ("football", "nfl", {}),
    "baseball": ("baseball", "mlb", {}),
    "nhl":      ("hockey", "nhl", {}),
    "ufc":      ("mma", "ufc", {}),
    "f1":       ("racing", "f1", {}),
}

# Soccer league slugs for ESPN
ESPN_SOCCER_LEAGUES = [
    "eng.1",          # EPL
    "esp.1",          # La Liga
    "ger.1",          # Bundesliga
    "ita.1",          # Serie A
    "fra.1",          # Ligue 1
    "uefa.champions", # UCL
    "uefa.europa",    # UEL
    "usa.1",          # MLS
]

# bo3.gg discipline IDs
BO3_BASE = "https://api.bo3.gg/api/v1"
BO3_DISCIPLINE: dict[str, int] = {"cs2": 1, "valorant": 2, "lol": 3}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _slug_teams(event_id: str) -> tuple[str, str]:
    """Extract (team_a, team_b) from event_id slug as space-separated strings.

    e.g. "soccer::2026-03-18::atletico_madrid_vs_real_madrid"
         → ("atletico madrid", "real madrid")
    """
    parts = event_id.split("::")
    if len(parts) >= 3 and "_vs_" in parts[2]:
        a, b = parts[2].split("_vs_", 1)
        return a.replace("_", " "), b.replace("_", " ")
    return "", ""


# Short canonical forms OR ESPN display names that don't fuzzy-match our stored slugs.
# Keys are the normalised form (post-punctuation/unicode strip); values are
# the expanded form that will score ≥ FUZZY_THRESHOLD in comparison.
_ACRONYM_EXPAND: dict[str, str] = {
    # PSG / Bundesliga short forms
    "psg":                      "paris saint germain",
    "mgladbach":                "borussia monchengladbach",
    "m gladbach":               "borussia monchengladbach",
    "monchengladbach":          "borussia monchengladbach",
    # ESPN display names that differ significantly from our stored slugs
    # (ESPN uses English/formal names; we store short slugs from Kalshi tickers)
    "fc cologne":               "koln",         # ESPN: "FC Cologne" → slug: "koln"
    "cologne":                  "koln",
    "stade rennais":            "rennes",       # ESPN: "Stade Rennais" → slug: "rennes"
    "olympique lyonnais":       "lyon",         # ESPN: "Olympique Lyonnais" → slug: "lyon"
    "olympique de marseille":   "marseille",
    "losc lille":               "lille",
    "rc lens":                  "lens",
    "stade brestois":           "brest",
    "ogc nice":                 "nice",
    "toulouse fc":              "toulouse",
    "fc nantes":                "nantes",
    "le havre ac":              "le havre",
    "aj auxerre":               "auxerre",
    "montpellier hsc":          "montpellier",
    "as monaco":                "monaco",
    # EPL short-form slugs whose ESPN full names score just below threshold
    "manchester city":          "man city",
    "manchester united":        "man united",
    "bayer leverkusen":         "leverkusen",
    "rb leipzig":               "leipzig",
    "borussia dortmund":        "dortmund",
    "sc freiburg":              "freiburg",
    "eintracht frankfurt":      "frankfurt",
    "vfb stuttgart":            "stuttgart",
    "fc augsburg":              "augsburg",
    "sv werder bremen":         "werder bremen",
    "tsg hoffenheim":           "hoffenheim",
    "1 fc union berlin":        "union berlin",
    "fc st pauli":              "st pauli",
    "vfl wolfsburg":            "wolfsburg",
    "vfl bochum":               "bochum",
}


def _to_fuzz(name: str) -> str:
    """Normalise a team name for fuzzy comparison: lowercase, spaces only.

    1. Unicode-strip accents/umlauts (köln→koln, atlético→atletico, ö→o).
    2. Strip punctuation (., &, -, ') that appears in team names like
       "Texas A&M" or "Saint Mary's".
    3. Expand known short acronyms (psg, mgladbach) to full club names so
       they fuzzy-match ESPN's long-form display names above threshold.
    """
    # Strip accents / umlauts (NFKD decomposition → keep only ASCII)
    ascii_name = (
        unicodedata.normalize("NFKD", name)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    result = (
        ascii_name.lower()
        .replace("_", " ")
        .replace(".", "")
        .replace("-", " ")
        .replace("&", " ")
        .replace("'", "")
        .strip()
    )
    return _ACRONYM_EXPAND.get(result, result)


def _fuzzy_team_match(query: str, candidate: str) -> float:
    """Return rapidfuzz score between two team name strings.

    Uses token_set_ratio (not token_sort_ratio) so that short slugs like
    "nebraska" or "byu" match ESPN's full "Nebraska Cornhuskers" / "BYU Cougars"
    names — the extra mascot tokens are ignored when one string is a subset.
    """
    return fuzz.token_set_ratio(_to_fuzz(query), _to_fuzz(candidate))


# ---------------------------------------------------------------------------
# ESPN helpers
# ---------------------------------------------------------------------------

async def _fetch_espn_scores(
    client: httpx.AsyncClient, sport: str, league: str, espn_date: str,
    extra_params: Optional[dict] = None,
) -> list[dict]:
    """Fetch completed game scores from ESPN (date as YYYYMMDD)."""
    url = f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard"
    params: dict = {"dates": espn_date, "limit": 200}
    if extra_params:
        params.update(extra_params)
    try:
        r = await client.get(url, params=params)
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
        if not comp.get("status", {}).get("type", {}).get("completed"):
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

        # Extract actual game date from ESPN for cross-day series dedup
        game_date_str = comp.get("date", "") or event.get("date", "")
        game_date = game_date_str[:10] if game_date_str else espn_date[:4] + "-" + espn_date[4:6] + "-" + espn_date[6:8]

        results.append({
            "home_name": home.get("team", {}).get("displayName", ""),
            "away_name": away.get("team", {}).get("displayName", ""),
            "home_score": home_score,
            "away_score": away_score,
            "home_won": home_score > away_score,
            "game_date": game_date,
        })

    return results


# ---------------------------------------------------------------------------
# bo3.gg helpers
# ---------------------------------------------------------------------------

async def _fetch_bo3_scores(
    client: httpx.AsyncClient, sector: str, target_date: date
) -> list[dict]:
    """Fetch completed series from bo3.gg for a given sector and date.

    Extracts team names/slugs from nested team objects so downstream
    matching can fuzzy-compare against event_id slugs.
    """
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

        # bo3.gg may nest team data as {"team1": {"name": ..., "slug": ...}}
        # or keep it flat with team1_id only — handle both
        t1_obj = m.get("team1") or {}
        t2_obj = m.get("team2") or {}
        team1_name = (
            t1_obj.get("name") or t1_obj.get("slug") or str(m.get("team1_id", ""))
        )
        team2_name = (
            t2_obj.get("name") or t2_obj.get("slug") or str(m.get("team2_id", ""))
        )

        results.append({
            "team1_name": team1_name,
            "team2_name": team2_name,
            "team1_score": t1_score,
            "team2_score": t2_score,
            "team1_won": t1_score > t2_score,
        })

    return results


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------

def _match_espn(pred: dict, scores: list[dict]) -> Optional[int]:
    """Return 1 (YES won) / 0 (YES lost) / None (no match).

    Algorithm:
      1. Extract team_a / team_b from event_id slug (primary source).
      2. For each ESPN score, require BOTH teams to fuzzy-match home/away
         (this is the event-identity gate — prevents cross-match collisions).
      3. Determine which side (home/away) the YES team is on, return outcome.
      4. Fall back to yes_team direct fuzzy match when slug is missing.
    """
    slug_a, slug_b = _slug_teams(pred["event_id"])
    yes_raw = _to_fuzz(pred["yes_team"])

    # Determine yes alignment relative to slug (team_a vs team_b)
    if slug_a and slug_b:
        yes_a_score = _fuzzy_team_match(yes_raw, slug_a)
        yes_b_score = _fuzzy_team_match(yes_raw, slug_b)
        if yes_a_score >= FUZZY_THRESHOLD and yes_a_score >= yes_b_score:
            yes_is_team_a: Optional[bool] = True
        elif yes_b_score >= FUZZY_THRESHOLD:
            yes_is_team_a = False
        elif yes_a_score > yes_b_score:
            # yes_team is an abbreviation (e.g. "hp", "cbu") — both scores below
            # threshold but slug_a is relatively better. Trust the relative signal
            # once the event-identity gate passes.
            yes_is_team_a = True
        elif yes_b_score > yes_a_score:
            yes_is_team_a = False
        else:
            yes_is_team_a = None  # fall through to direct yes_team match (both score 0)
    else:
        slug_a = slug_b = ""
        yes_is_team_a = None

    # Prediction date for cross-day series guard
    pred_date = pred.get("event_date", "")

    for score in scores:
        home_n = score["home_name"]
        away_n = score["away_name"]

        # Cross-day series guard: if both the prediction and the ESPN result
        # carry a date, skip results from a different calendar day. ESPN
        # game_date is ISO format from competition.date (UTC); pred_date is
        # the stored event_date. Allow same-day or ±1 day for UTC offset, but
        # skip results clearly from a different game in the same series.
        score_date = score.get("game_date", "")
        if pred_date and score_date and len(pred_date) >= 10 and len(score_date) >= 10:
            try:
                pd = date.fromisoformat(pred_date[:10])
                sd = date.fromisoformat(score_date[:10])
                if abs((pd - sd).days) > 1:
                    continue  # different day — wrong game in multi-day series
            except ValueError:
                pass

        if slug_a and slug_b:
            # Gate: both slugs must match this score (ensures correct event)
            a_home = _fuzzy_team_match(slug_a, home_n) >= FUZZY_THRESHOLD
            a_away = _fuzzy_team_match(slug_a, away_n) >= FUZZY_THRESHOLD
            b_home = _fuzzy_team_match(slug_b, home_n) >= FUZZY_THRESHOLD
            b_away = _fuzzy_team_match(slug_b, away_n) >= FUZZY_THRESHOLD

            if not ((a_home and b_away) or (a_away and b_home)):
                continue  # not our event

            # Soccer draw/tie market — resolve immediately after event gate,
            # before yes_team side resolution (tie has no "home" side).
            if yes_raw in ("tie", "draw", "x"):
                is_draw = score["home_score"] == score["away_score"]
                return 1 if is_draw else 0

            # Resolve yes team side
            if yes_is_team_a is True:
                yes_is_home = a_home
            elif yes_is_team_a is False:
                yes_is_home = b_home
            else:
                # slug alignment failed — fall back to direct yes_team match
                yes_is_home = _fuzzy_team_match(yes_raw, home_n) >= FUZZY_THRESHOLD
                if not yes_is_home and _fuzzy_team_match(yes_raw, away_n) < FUZZY_THRESHOLD:
                    continue  # can't determine side
        else:
            # No slug — direct yes_team match only
            yes_is_home = _fuzzy_team_match(yes_raw, home_n) >= FUZZY_THRESHOLD
            if not yes_is_home:
                if _fuzzy_team_match(yes_raw, away_n) < FUZZY_THRESHOLD:
                    continue  # this score doesn't involve our team

        if yes_is_home:
            return 1 if score["home_won"] else 0
        else:
            # Explicit away win only — draws are not wins for the away side
            away_won = score["away_score"] > score["home_score"]
            return 1 if away_won else 0

    logger.debug(
        "espn_no_match",
        event_id=pred["event_id"],
        yes_team=pred["yes_team"],
        slug_a=slug_a,
        slug_b=slug_b,
        scores_checked=len(scores),
    )
    return None


def _match_bo3(pred: dict, scores: list[dict]) -> Optional[int]:
    """Return 1 / 0 / None.

    Fixes the original bug where team1_won was read from an arbitrary score
    without first verifying the score belongs to this specific match.

    Algorithm:
      1. Extract slug_a / slug_b from event_id.
      2. Find the score where slug_a fuzzy-matches one team AND slug_b matches
         the other (event-identity gate).
      3. Determine which team the YES side is, return outcome.
    """
    slug_a, slug_b = _slug_teams(pred["event_id"])
    if not slug_a:
        return None

    yes_raw = _to_fuzz(pred["yes_team"])
    yes_a_score = _fuzzy_team_match(yes_raw, slug_a)
    yes_b_score = _fuzzy_team_match(yes_raw, slug_b)

    if yes_a_score < FUZZY_THRESHOLD and yes_b_score < FUZZY_THRESHOLD:
        logger.debug(
            "bo3_yes_team_no_slug_match",
            event_id=pred["event_id"],
            yes_team=pred["yes_team"],
        )
        return None

    yes_is_team_a = yes_a_score >= yes_b_score

    for score in scores:
        t1n = score["team1_name"]
        t2n = score["team2_name"]

        # Gate: both event teams must match this score's teams
        a_t1 = _fuzzy_team_match(slug_a, t1n) >= FUZZY_THRESHOLD
        a_t2 = _fuzzy_team_match(slug_a, t2n) >= FUZZY_THRESHOLD
        b_t1 = _fuzzy_team_match(slug_b, t1n) >= FUZZY_THRESHOLD
        b_t2 = _fuzzy_team_match(slug_b, t2n) >= FUZZY_THRESHOLD

        if a_t1 and b_t2:
            # slug_a = team1, slug_b = team2
            return 1 if (yes_is_team_a == score["team1_won"]) else 0
        if a_t2 and b_t1:
            # slug_a = team2, slug_b = team1
            return 1 if (yes_is_team_a != score["team1_won"]) else 0

    logger.debug(
        "bo3_no_match",
        event_id=pred["event_id"],
        slug_a=slug_a,
        slug_b=slug_b,
        scores_checked=len(scores),
    )
    return None


async def _resolve_via_kalshi(preds: list[dict]) -> dict[str, Optional[int]]:
    """Resolve predictions by fetching their Kalshi market settlement prices.

    Returns market_id → 1 (YES) / 0 (NO) / None (still open or error).
    Settled markets have result="yes" or result="no" in the Kalshi API.
    """
    from evmax.clients.kalshi import KalshiClient

    out: dict[str, Optional[int]] = {}
    async with KalshiClient() as client:
        async def _fetch_one(pred: dict) -> tuple[str, Optional[int]]:
            ticker = pred["market_id"].removeprefix("kalshi:")
            price = await client.get_market_price(ticker)
            if price is None:
                return pred["market_id"], None
            if price >= 0.99:
                return pred["market_id"], 1
            if price <= 0.01:
                return pred["market_id"], 0
            return pred["market_id"], None  # still open

        results = await asyncio.gather(*(_fetch_one(p) for p in preds), return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                continue
            mid, outcome = r
            out[mid] = outcome
    return out


def _resolve_prop_outcome(
    player_name: str,
    stat_type: str,
    threshold: float,
    target_date: date,
) -> Optional[int]:
    """Look up a player's actual stat on target_date and compare to threshold.

    Returns 1 (over hit), 0 (under / no game), or None (player/stat not found).
    Uses a ±1-day window around target_date to absorb stored-date off-by-ones.
    """
    try:
        from evmax.clients.nba_stats import _find_player_id, STAT_COL, _SEASON
        import pandas as pd
        from nba_api.stats.endpoints import playergamelogs

        player_id = _find_player_id(player_name)
        if player_id is None:
            logger.debug("prop_player_not_found", player=player_name)
            return None

        df = playergamelogs.PlayerGameLogs(
            player_id_nullable=player_id,
            season_nullable=_SEASON,
            last_n_games_nullable=0,
        ).get_data_frames()[0]

        if df is None or df.empty:
            return None

        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.date
        lo = target_date - timedelta(days=1)
        hi = target_date + timedelta(days=1)
        game_row = df[(df["GAME_DATE"] >= lo) & (df["GAME_DATE"] <= hi)]

        if game_row.empty:
            logger.debug("prop_no_game_on_date", player=player_name, date=str(target_date))
            return None

        row = game_row.iloc[0]
        col = STAT_COL.get(stat_type)
        if col is None:
            return None

        if col == "__pra__":
            stat_val = float(row["PTS"]) + float(row["REB"]) + float(row["AST"])
        elif col == "__bs__":
            stat_val = float(row.get("BLK", 0)) + float(row.get("STL", 0))
        elif col in df.columns:
            stat_val = float(row[col])
        else:
            return None

        return 1 if stat_val >= threshold else 0

    except Exception as exc:
        logger.warning("prop_resolve_error", player=player_name, stat=stat_type, error=str(exc))
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

async def fetch_live_scores_for_sector(sector: str) -> list[dict]:
    """Fetch in-progress game scores from ESPN for a sector (no date filter).

    Returns list of dicts:
      {home_name, away_name, score_home, score_away, period, clock_secs, is_soccer}

    clock_secs semantics:
      - soccer: seconds ELAPSED in current half (clock counts up)
      - other:  seconds REMAINING in current period (clock counts down)

    Empty list if sector has no ESPN mapping or request fails.
    """
    if sector == "soccer":
        leagues = ESPN_SOCCER_LEAGUES
    elif sector in ESPN_SPORT_MAP:
        leagues = [(ESPN_SPORT_MAP[sector][0], ESPN_SPORT_MAP[sector][1])]
    else:
        return []

    results: list[dict] = []
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(10.0),
        headers={"User-Agent": "evmax-live/1.0"},
        follow_redirects=True,
    ) as client:
        if sector == "soccer":
            for espn_league in leagues:
                url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{espn_league}/scoreboard"
                try:
                    r = await client.get(url, params={"limit": 50})
                    r.raise_for_status()
                    results.extend(_parse_live_events(r.json(), is_soccer=True))
                except Exception as e:
                    logger.debug("live_score_fetch_failed", sector=sector, league=espn_league, error=str(e))
        else:
            sport, league, extra_params = ESPN_SPORT_MAP[sector]
            url = f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard"
            params: dict = {"limit": 50}
            if extra_params:
                params.update(extra_params)
            try:
                r = await client.get(url, params=params)
                r.raise_for_status()
                results.extend(_parse_live_events(r.json(), is_soccer=False))
            except Exception as e:
                logger.debug("live_score_fetch_failed", sector=sector, error=str(e))

    return results


def _parse_live_events(data: dict, is_soccer: bool) -> list[dict]:
    """Parse ESPN scoreboard JSON for in-progress events."""
    out = []
    for event in data.get("events", []):
        comps = event.get("competitions", [])
        if not comps:
            continue
        comp = comps[0]
        status = comp.get("status", {})
        if status.get("type", {}).get("state") != "in":
            continue

        competitors = comp.get("competitors", [])
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue

        try:
            score_home = int(home.get("score", 0))
            score_away = int(away.get("score", 0))
        except (ValueError, TypeError):
            continue

        period = status.get("period", 1)
        # status.clock is raw seconds (float or string)
        raw_clock = status.get("clock", 0)
        try:
            clock_secs = float(raw_clock)
        except (ValueError, TypeError):
            clock_secs = 0.0

        out.append({
            "home_name": home.get("team", {}).get("displayName", ""),
            "away_name": away.get("team", {}).get("displayName", ""),
            "score_home": score_home,
            "score_away": score_away,
            "period": period,
            "clock_secs": clock_secs,
            "is_soccer": is_soccer,
        })
    return out


async def fetch_completed_scores(sector: str, target_date: date) -> list[dict]:
    """Fetch completed game scores for a sector and date from ESPN.

    Returns same structure as _fetch_espn_scores: list of dicts with
    home_name, away_name, home_score, away_score, home_won.
    Returns empty list for sectors without ESPN coverage.
    """
    espn_date = target_date.isoformat().replace("-", "")
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(20.0),
        headers={"User-Agent": "evmax-update/1.0"},
        follow_redirects=True,
    ) as client:
        if sector == "soccer":
            results: list[dict] = []
            for espn_league in ESPN_SOCCER_LEAGUES:
                results.extend(
                    await _fetch_espn_scores(client, "soccer", espn_league, espn_date)
                )
            return results
        elif sector in ESPN_SPORT_MAP:
            sport, league, extra_params = ESPN_SPORT_MAP[sector]
            return await _fetch_espn_scores(client, sport, league, espn_date, extra_params)
    return []


async def resolve_outcomes_for_date(target_date: Optional[date] = None) -> dict:
    """Fetch and store outcomes for all pending predictions on target_date.

    Returns {"resolved": int, "failed": int, "unmatched": list[event_id]}.
    """
    target_date = target_date or (date.today() - timedelta(days=1))
    date_str = target_date.isoformat()

    conn = get_connection()
    today_str = date.today().isoformat()
    pending = conn.execute(
        """SELECT p.market_id, p.event_id, p.sector, p.yes_team,
                  p.event_title, p.event_date, p.sharp_true_prob, p.blended_true_prob
           FROM ev_predictions p
           LEFT JOIN ev_outcomes o ON p.market_id = o.market_id
           WHERE (p.event_date = ? OR p.scan_date = ?)
             AND o.market_id IS NULL
             AND COALESCE(p.event_date, p.scan_date) <= ?""",
        (date_str, date_str, today_str),
    ).fetchall()

    if not pending:
        logger.info("no_pending_outcomes", date=date_str)
        conn.close()
        return {"resolved": 0, "failed": 0, "unmatched": []}

    # Deduplicate by market_id — the same market may be logged on multiple
    # scan_dates; resolving duplicates produces duplicate unmatched IDs.
    seen_market_ids: set[str] = set()
    deduped: list[dict] = []
    for row in pending:
        d = dict(row)
        if d["market_id"] not in seen_market_ids:
            seen_market_ids.add(d["market_id"])
            deduped.append(d)
    pending = deduped

    logger.info("resolving_outcomes", count=len(pending), date=date_str)

    # Split prop predictions (event_id part[2] == "prop") from game predictions.
    prop_preds = [p for p in pending if p["event_id"].split("::")[2:3] == ["prop"]]
    game_preds = [p for p in pending if p not in prop_preds]

    resolved = 0
    failed = 0
    unmatched: list[str] = []

    # Resolve NBA player props synchronously (nba_api is blocking).
    for pred in prop_preds:
        parts = pred["event_id"].split("::")
        if len(parts) < 6:
            failed += 1
            unmatched.append(pred["event_id"])
            continue
        _player, _stat, _thr_str = parts[3], parts[4], parts[5]
        try:
            _threshold = float(_thr_str)
        except ValueError:
            failed += 1
            unmatched.append(pred["event_id"])
            continue
        _event_date = date.fromisoformat(parts[1])
        outcome = _resolve_prop_outcome(_player, _stat, _threshold, _event_date)
        if outcome is not None:
            _write_outcome(conn, pred, outcome, "nba_api")
            resolved += 1
        else:
            failed += 1
            unmatched.append(pred["event_id"])

    # Group by sector → event_date → rows for game markets.
    # Using event_date (not target_date) ensures we query ESPN for the correct
    # date even when scan_date differs (e.g. a game scanned on 3/19 that plays 3/20).
    by_sector_date: dict[str, dict[str, list[dict]]] = {}
    for row in game_preds:
        d = dict(row)
        event_date = d.get("event_date") or date_str
        by_sector_date.setdefault(d["sector"], {}).setdefault(event_date, []).append(d)

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(20.0),
        headers={"User-Agent": "evmax-cleanup/1.0"},
        follow_redirects=True,
    ) as client:

        for sector, date_groups in by_sector_date.items():
            if sector in ESPN_SPORT_MAP:
                sport, league, extra_params = ESPN_SPORT_MAP[sector]
                for group_date, rows in date_groups.items():
                    group_date_obj = date.fromisoformat(group_date)
                    # 3-day window: event_date-1, event_date, event_date+1
                    # Guards against off-by-one stored dates in either direction.
                    espn_dates = [
                        (group_date_obj - timedelta(days=1)).isoformat().replace("-", ""),
                        group_date.replace("-", ""),
                        (group_date_obj + timedelta(days=1)).isoformat().replace("-", ""),
                    ]
                    combined_scores: list[dict] = []
                    for d in espn_dates:
                        combined_scores.extend(
                            await _fetch_espn_scores(client, sport, league, d, extra_params)
                        )
                    for pred in rows:
                        outcome = _match_espn(pred, combined_scores)
                        if outcome is not None:
                            _write_outcome(conn, pred, outcome, "espn")
                            resolved += 1
                        else:
                            failed += 1
                            unmatched.append(pred["event_id"])

            elif sector == "soccer":
                for group_date, rows in date_groups.items():
                    group_date_obj = date.fromisoformat(group_date)
                    espn_dates = [
                        (group_date_obj - timedelta(days=1)).isoformat().replace("-", ""),
                        group_date.replace("-", ""),
                        (group_date_obj + timedelta(days=1)).isoformat().replace("-", ""),
                    ]
                    all_scores: list[dict] = []
                    for espn_league in ESPN_SOCCER_LEAGUES:
                        for d in espn_dates:
                            all_scores.extend(
                                await _fetch_espn_scores(client, "soccer", espn_league, d)
                            )
                    for pred in rows:
                        outcome = _match_espn(pred, all_scores)
                        if outcome is not None:
                            _write_outcome(conn, pred, outcome, "espn")
                            resolved += 1
                        else:
                            failed += 1
                            unmatched.append(pred["event_id"])

            elif sector in BO3_DISCIPLINE:
                # Kalshi settlement is ground truth — opaque esports ticker codes
                # (e.g. "th", "shg", "dcg") don't reliably fuzzy-match bo3.gg team
                # names. Fall back to bo3.gg only for markets Kalshi hasn't settled.
                all_rows = [r for rows in date_groups.values() for r in rows]
                kalshi_results = await _resolve_via_kalshi(all_rows)

                bo3_pending: dict[str, list[dict]] = {}
                for pred in all_rows:
                    outcome = kalshi_results.get(pred["market_id"])
                    if outcome is not None:
                        _write_outcome(conn, pred, outcome, "kalshi_settlement")
                        resolved += 1
                    else:
                        bo3_pending.setdefault(
                            pred.get("event_date") or target_date.isoformat(), []
                        ).append(pred)

                for group_date, rows in bo3_pending.items():
                    group_date_obj = (
                        date.fromisoformat(group_date) if group_date else target_date
                    )
                    scores = await _fetch_bo3_scores(client, sector, group_date_obj)
                    for pred in rows:
                        outcome = _match_bo3(pred, scores)
                        if outcome is not None:
                            _write_outcome(conn, pred, outcome, "bo3gg")
                            resolved += 1
                        else:
                            failed += 1
                            unmatched.append(pred["event_id"])

            elif sector == "tennis":
                # ESPN tennis doesn't return completed match results in a usable format.
                # Use Kalshi market settlement as ground truth: result="yes"→1.0, "no"→0.0.
                all_rows = [r for rows in date_groups.values() for r in rows]
                kalshi_results = await _resolve_via_kalshi(all_rows)
                for pred in all_rows:
                    outcome = kalshi_results.get(pred["market_id"])
                    if outcome is not None:
                        _write_outcome(conn, pred, outcome, "kalshi_settlement")
                        resolved += 1
                    else:
                        failed += 1
                        unmatched.append(pred["event_id"])

            else:
                logger.warning("no_resolver_for_sector", sector=sector)
                failed += sum(len(rows) for rows in date_groups.values())
                unmatched.extend(
                    r["event_id"] for rows in date_groups.values() for r in rows
                )

    conn.commit()
    conn.close()

    if unmatched:
        logger.warning(
            "unmatched_events",
            count=len(unmatched),
            date=date_str,
            events=unmatched[:20],  # cap log size
        )

    logger.info("resolve_done", date=date_str, resolved=resolved, failed=failed)
    return {"resolved": resolved, "failed": failed, "unmatched": unmatched}


def backfill_clv(since: Optional[date] = None, until: Optional[date] = None) -> dict:
    """Backfill CLV (Closing Line Value) for resolved bets from the archive.

    For each resolved bet missing pinnacle_close_prob, finds the last archived
    Pinnacle closing line from archived_sharp_odds and populates both:
      - ev_outcomes.pinnacle_close_prob (used by `cleanup show`)
      - ev_predictions.clv_pct (entry_prob - close_prob in pp)

    Positive CLV (entry_prob > close_prob) = we got a better price (real edge).

    Returns: {"updated": N, "skipped": M, "avg_clv": float}
    """
    from evmax.archiver import DataArchiver

    archiver = DataArchiver()
    conn = get_connection()

    since_str = (since or date(2020, 1, 1)).isoformat()
    until_str = (until or date.today()).isoformat()

    # Get resolved bets missing close prob
    rows = conn.execute(
        """SELECT p.id, p.market_id, p.event_id, p.sharp_true_prob, p.market_type
           FROM ev_predictions p
           INNER JOIN ev_outcomes o ON p.market_id = o.market_id
           WHERE o.outcome IS NOT NULL
             AND o.pinnacle_close_prob IS NULL
             AND p.scan_date >= ? AND p.scan_date <= ?""",
        (since_str, until_str),
    ).fetchall()

    updated = skipped = 0
    clv_values: list[float] = []

    for row in rows:
        close_prob = archiver.get_closing_line(row["event_id"])
        if close_prob is None:
            skipped += 1
            continue

        entry_prob = row["sharp_true_prob"]
        if entry_prob <= 0:
            skipped += 1
            continue

        # CLV in pp: positive means our entry price was better than close
        clv_pp = (entry_prob - close_prob) * 100

        # Update ev_outcomes.pinnacle_close_prob (used by `cleanup show`)
        conn.execute(
            "UPDATE ev_outcomes SET pinnacle_close_prob = ? WHERE market_id = ?",
            (close_prob, row["market_id"]),
        )
        # Update ev_predictions.clv_pct
        conn.execute(
            "UPDATE ev_predictions SET clv_pct = ? WHERE id = ?",
            (round(clv_pp, 2), row["id"]),
        )
        updated += 1
        clv_values.append(clv_pp)

    conn.commit()
    conn.close()

    avg_clv = sum(clv_values) / len(clv_values) if clv_values else 0.0
    logger.info(
        "clv_backfill_done",
        updated=updated,
        skipped=skipped,
        avg_clv=round(avg_clv, 2),
    )
    return {"updated": updated, "skipped": skipped, "avg_clv": round(avg_clv, 2)}
