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
    "wnba":     ("basketball", "wnba", {}),
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

        # Extract box-score statistics (soccer: shotsOnTarget, totalShots)
        def _stat(competitor: dict, name: str) -> Optional[int]:
            for s in competitor.get("statistics", []):
                if s.get("name") == name:
                    try:
                        return int(float(s.get("displayValue", 0)))
                    except (ValueError, TypeError):
                        pass
            return None

        results.append({
            "home_name": home.get("team", {}).get("displayName", ""),
            "away_name": away.get("team", {}).get("displayName", ""),
            "home_score": home_score,
            "away_score": away_score,
            "home_won": home_score > away_score,
            "game_date": game_date,
            "home_sot": _stat(home, "shotsOnTarget"),
            "away_sot": _stat(away, "shotsOnTarget"),
            "home_shots": _stat(home, "totalShots"),
            "away_shots": _stat(away, "totalShots"),
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

        market_type = (pred.get("market_type") or "moneyline").lower()
        hs = score.get("home_score")
        as_ = score.get("away_score")

        if market_type == "spread":
            # Kalshi spread tickers ask "does TEAM win by over N.5". We store
            # the ticker's line as a NEGATIVE float for YES bets (e.g. -7.5
            # = "wins by over 7.5"). The synthesized NO-side bet (the +spread
            # cover) is stored with a POSITIVE line (e.g. +7.5 = "doesn't
            # lose by 8 or more").
            #   line < 0  →  YES covers iff margin >  threshold (wins by N+)
            #   line > 0  →  YES covers iff margin > -threshold (loses by < N or wins)
            if hs is None or as_ is None or pred.get("line") is None:
                return None
            line_val = float(pred["line"])
            threshold = abs(line_val)
            yes_score = hs if yes_is_home else as_
            opp_score = as_ if yes_is_home else hs
            margin = yes_score - opp_score
            cover_threshold = -threshold if line_val > 0 else threshold
            return 1 if margin > cover_threshold else 0

        if market_type in ("total", "over_under"):
            # Kalshi totals ask "will combined score be > line" (yes=over)
            # or "< line" (yes=under). yes_team is set to "over"/"under"
            # at the kalshi parse step.
            if hs is None or as_ is None or pred.get("line") is None:
                return None
            threshold = float(pred["line"])
            total = hs + as_
            side = (pred.get("yes_team") or "").lower()
            if side == "over":
                return 1 if total > threshold else 0
            if side == "under":
                return 1 if total < threshold else 0
            return None

        # Moneyline (default)
        if yes_is_home:
            return 1 if score["home_won"] else 0
        else:
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
    result = _resolve_prop_outcome_with_value(player_name, stat_type, threshold, target_date)
    if result is None:
        return None
    return result[0]


# ESPN box-score → stat_type label map. Mirrors LABEL_MAP in
# _resolve_prop_observations; kept local so the two prop-resolution code paths
# stay independently editable. "labels" come from the per-team stat_block's
# `labels` list (display-facing) — ESPN doesn't publish a `keys` array for NBA
# the way it does for NFL, so labels are the stable join key here.
_ESPN_NBA_LABEL: dict[str, str] = {
    "points": "PTS",
    "rebounds": "REB",
    "assists": "AST",
    "steals": "STL",
    "blocks": "BLK",
    "threes": "3PT",
    "turnovers": "TO",
}


def _norm_prop_player(s: str) -> str:
    """Normalize player name: strip accents, lowercase, collapse to underscores."""
    nfkd = unicodedata.normalize("NFKD", s)
    ascii_only = "".join(c for c in nfkd if not unicodedata.combining(c))
    return (
        ascii_only.lower().strip()
        .replace(" ", "_").replace(".", "")
        .replace("’", "'").replace("‘", "'")
    )


def _fetch_espn_nba_player_stats(target_date: date) -> dict[str, dict[str, float]]:
    """Pull NBA box scores for `target_date` and return normalized_player→stats.

    One scoreboard call + N summary calls (N = games on the slate). Cached
    nowhere — caller is responsible for caching per resolution run.
    """
    espn_date = target_date.isoformat().replace("-", "")
    player_stats: dict[str, dict[str, float]] = {}
    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            r = client.get(
                "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
                params={"dates": espn_date},
            )
            r.raise_for_status()
            events = r.json().get("events", [])

            for event in events:
                eid = event.get("id")
                if not eid:
                    continue
                comp = event.get("competitions", [{}])[0]
                if not comp.get("status", {}).get("type", {}).get("completed", False):
                    continue
                try:
                    box_resp = client.get(
                        "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary",
                        params={"event": eid},
                    )
                    box_resp.raise_for_status()
                    box = box_resp.json()
                except Exception as exc:
                    logger.debug("prop_pred_espn_box_fail", event=eid, error=str(exc))
                    continue

                for team_box in box.get("boxscore", {}).get("players", []):
                    for stat_block in team_box.get("statistics", []):
                        labels = stat_block.get("labels", [])
                        for athlete in stat_block.get("athletes", []):
                            name = athlete.get("athlete", {}).get("displayName", "")
                            vals = athlete.get("stats", [])
                            if not name or not vals:
                                continue
                            stat_dict: dict[str, float] = {}
                            for lbl, val in zip(labels, vals):
                                try:
                                    if "-" in str(val) and lbl in ("FG", "3PT", "FT"):
                                        stat_dict[lbl] = float(str(val).split("-")[0])
                                    else:
                                        stat_dict[lbl] = float(val)
                                except (ValueError, TypeError):
                                    pass
                            player_stats[_norm_prop_player(name)] = stat_dict
    except Exception as exc:
        logger.warning("prop_pred_espn_scoreboard_fail", date=str(target_date), error=str(exc))
    return player_stats


def _resolve_prop_from_espn(
    espn_stats: dict[str, dict[str, float]],
    player_name: str,
    stat_type: str,
    threshold: float,
) -> Optional[tuple[int, float]]:
    """Resolve a single prop from a prefetched ESPN lookup, or return None."""
    norm = _norm_prop_player(player_name)
    stats = espn_stats.get(norm)
    if stats is None:
        last = norm.rsplit("_", 1)[-1] if "_" in norm else norm
        for pname, pstats in espn_stats.items():
            if pname.endswith(last):
                stats = pstats
                break
    if stats is None:
        return None

    if stat_type == "points_rebounds_assists":
        if not all(k in stats for k in ("PTS", "REB", "AST")):
            return None
        actual = stats["PTS"] + stats["REB"] + stats["AST"]
    elif stat_type == "blocks_steals":
        if not all(k in stats for k in ("BLK", "STL")):
            return None
        actual = stats["BLK"] + stats["STL"]
    else:
        label = _ESPN_NBA_LABEL.get(stat_type)
        if label is None:
            return None
        actual = stats.get(label)
        if actual is None:
            return None

    outcome = 1 if actual >= threshold else 0
    return outcome, float(actual)


def _resolve_prop_outcome_with_value(
    player_name: str,
    stat_type: str,
    threshold: float,
    target_date: date,
) -> Optional[tuple[int, float]]:
    """nba_api fallback path. ESPN should resolve the vast majority of props.

    Uses the LRU-cached `_fetch_gamelog_sync` so repeat lookups on the same
    player (e.g. Donovan Mitchell's points + rebounds + assists + threes) share
    a single network roundtrip instead of firing four full-season pulls
    against stats.nba.com.
    """
    try:
        from evmax.clients.nba_stats import (
            _find_player_id, _fetch_gamelog_sync, STAT_COL,
        )
        import pandas as pd

        player_id = _find_player_id(player_name)
        if player_id is None:
            logger.debug("prop_player_not_found", player=player_name)
            return None

        df = _fetch_gamelog_sync(player_id)
        if df is None or df.empty:
            return None

        df = df.copy()
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

        outcome = 1 if stat_val >= threshold else 0
        return outcome, stat_val

    except Exception as exc:
        logger.warning("prop_resolve_error", player=player_name, stat=stat_type, error=str(exc))
        return None


def yes_aligned_close_prob(
    yes_team: Optional[str],
    outcome_a_label: Optional[str],
    outcome_b_label: Optional[str],
    true_prob_a: float,
    true_prob_b: Optional[float] = None,
    true_prob_draw: Optional[float] = None,
) -> Optional[float]:
    """Return the YES-aligned Pinnacle close prob given a bet's yes_team.

    Pinnacle's snapshot stores probabilities for (outcome_a, outcome_b)
    where outcome_a is whichever team Pinnacle picked as side A. Kalshi
    bets have a `yes_team` that may be either side. Storing true_prob_a
    blindly inverts the CLV calculation for half the bets in any 2-team
    market — see the WNBA Tempo/Mystics case where both ev_outcomes rows
    ended up with the same pinnacle_close_prob = 0.491 even though one
    side closed at 49.1% and the other at 50.9%.

    Returns None when no alignment can be determined (caller should skip
    the bet rather than pollute CLV with the wrong side).
    """
    if not yes_team:
        return None
    yt = yes_team.lower().strip()

    # Draw / tie / over / under shortcuts
    if yt in ("draw", "tie", "x") and true_prob_draw is not None:
        return float(true_prob_draw)
    if yt == "over" and true_prob_b is None and true_prob_a > 0:
        # Some sports use a/b for over/under — caller should ideally pass
        # over/under probs; fall back to a as "over".
        return float(true_prob_a)
    if yt == "under" and true_prob_b is not None:
        return float(true_prob_b)

    a = (outcome_a_label or "").lower().strip()
    b = (outcome_b_label or "").lower().strip()
    # Substring match either direction handles minor normalization drift
    # ("indiana fever" vs "fever", etc.).
    if a and (yt == a or yt in a or a in yt):
        return float(true_prob_a)
    if b and (yt == b or yt in b or b in yt) and true_prob_b is not None:
        return float(true_prob_b)
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
    # Commit immediately so the WAL write lock is released between rows.
    # Resolution spans tens of seconds across NBA + soccer + tennis + esports
    # while making HTTP calls between writes; holding one transaction across
    # all of that starves concurrent writers (dashboard `sync_portfolio_outcomes`,
    # CLI cleanup) and produces "database is locked" once their 5s busy_timeout
    # expires. WAL commits are cheap.
    conn.commit()


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


def _resolve_prop_observations(conn: sqlite3.Connection, target_date: date) -> int:
    """Resolve pending rows in prop_observations using ESPN box scores.

    Fetches box scores for each game date that has pending props, extracts
    player stats, and fills outcome + actual_value.  One HTTP request per game
    (not per player) so this is fast and doesn't hit nba_api rate limits.
    """
    today_str = date.today().isoformat()
    rows = conn.execute(
        """SELECT id, scan_date, player_name, stat_type, line, event_id
           FROM prop_observations
           WHERE outcome IS NULL AND scan_date <= ?""",
        (today_str,),
    ).fetchall()

    if not rows:
        return 0

    # Group by game date
    by_date: dict[str, list[dict]] = {}
    for row in rows:
        r = dict(row)
        parts = (r.get("event_id") or "").split("::")
        game_date = parts[1] if len(parts) >= 2 else r["scan_date"]
        try:
            gd = date.fromisoformat(game_date)
        except ValueError:
            continue
        if gd > date.today():
            continue
        by_date.setdefault(game_date, []).append(r)

    if not by_date:
        return 0

    import unicodedata

    def _norm_player(s: str) -> str:
        """Normalize player name: strip accents, lowercase, underscores."""
        nfkd = unicodedata.normalize("NFKD", s)
        ascii_only = "".join(c for c in nfkd if not unicodedata.combining(c))
        return (
            ascii_only.lower().strip()
            .replace(" ", "_").replace(".", "")
            .replace("\u2019", "'").replace("\u2018", "'")  # curly → straight apostrophe
            .replace("'", "'")
        )

    # ESPN stat label → our stat_type mapping
    LABEL_MAP = {
        "points": "PTS", "rebounds": "REB", "assists": "AST",
        "steals": "STL", "blocks": "BLK", "threes": "3PT",
        "turnovers": "TO",
    }

    resolved = 0
    client = httpx.Client(timeout=15.0, follow_redirects=True)

    try:
        for game_date, prop_rows in by_date.items():
            espn_date = game_date.replace("-", "")

            # Fetch scoreboard to get game IDs
            try:
                r = client.get(
                    f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
                    params={"dates": espn_date},
                )
                r.raise_for_status()
                events = r.json().get("events", [])
            except Exception as e:
                logger.warning("prop_obs_scoreboard_fail", date=game_date, error=str(e))
                continue

            # Build player → stats lookup from all box scores on this date
            player_stats: dict[str, dict[str, float]] = {}  # normalized_name → {PTS, REB, ...}

            for event in events:
                eid = event.get("id")
                if not eid:
                    continue
                comp = event.get("competitions", [{}])[0]
                if not comp.get("status", {}).get("type", {}).get("completed", False):
                    continue

                try:
                    box_resp = client.get(
                        f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary",
                        params={"event": eid},
                    )
                    box_resp.raise_for_status()
                    box = box_resp.json()
                except Exception as e:
                    logger.debug("prop_obs_box_fail", event=eid, error=str(e))
                    continue

                for team_box in box.get("boxscore", {}).get("players", []):
                    for stat_block in team_box.get("statistics", []):
                        labels = stat_block.get("labels", [])
                        for athlete in stat_block.get("athletes", []):
                            name = athlete.get("athlete", {}).get("displayName", "")
                            vals = athlete.get("stats", [])
                            if not name or not vals:
                                continue
                            stat_dict: dict[str, float] = {}
                            for lbl, val in zip(labels, vals):
                                try:
                                    # Handle "7-22" FG format — extract made count
                                    if "-" in str(val) and lbl in ("FG", "3PT", "FT"):
                                        stat_dict[lbl] = float(str(val).split("-")[0])
                                    else:
                                        stat_dict[lbl] = float(val)
                                except (ValueError, TypeError):
                                    pass
                            norm_name = _norm_player(name)
                            player_stats[norm_name] = stat_dict

            # Now resolve each prop row
            for prop in prop_rows:
                player = prop["player_name"]
                stat_type = prop["stat_type"]
                threshold = prop["line"]
                if not player or not stat_type or threshold is None:
                    continue

                norm_p = _norm_player(player)
                stats = player_stats.get(norm_p)
                if stats is None:
                    # Try partial match (last name)
                    last = norm_p.rsplit("_", 1)[-1] if "_" in norm_p else norm_p
                    for pname, pstats in player_stats.items():
                        if pname.endswith(last):
                            stats = pstats
                            break

                if stats is None:
                    continue

                # Map our stat_type to ESPN label
                espn_label = LABEL_MAP.get(stat_type)
                if espn_label is None:
                    # Compound stats
                    if stat_type in ("pts+reb+ast", "pra"):
                        actual = stats.get("PTS", 0) + stats.get("REB", 0) + stats.get("AST", 0)
                    elif stat_type in ("blks+stls", "stocks"):
                        actual = stats.get("BLK", 0) + stats.get("STL", 0)
                    else:
                        continue
                else:
                    actual = stats.get(espn_label)
                    if actual is None:
                        continue

                outcome = 1 if actual >= threshold else 0
                conn.execute(
                    "UPDATE prop_observations SET outcome = ?, actual_value = ? WHERE id = ?",
                    (outcome, actual, prop["id"]),
                )
                resolved += 1

    finally:
        client.close()

    logger.info("prop_observations_resolved", count=resolved, total_pending=len(rows))
    return resolved


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
                  p.event_title, p.event_date, p.sharp_true_prob, p.blended_true_prob,
                  p.market_type, p.line
           FROM ev_predictions p
           LEFT JOIN ev_outcomes o ON p.market_id = o.market_id
           WHERE (p.event_date = ? OR p.scan_date = ?)
             AND o.market_id IS NULL
             AND COALESCE(p.event_date, p.scan_date) <= ?""",
        (date_str, date_str, today_str),
    ).fetchall()

    # Also pick up portfolio-only markets (scanned through the portfolio
    # path, which writes to portfolio_bets instead of ev_predictions).
    # The portfolios module creates this table lazily, so in fresh/test
    # environments it may not exist yet — treat that as "no portfolio bets".
    has_portfolio_bets = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='portfolio_bets'"
    ).fetchone()
    if has_portfolio_bets:
        portfolio_pending = conn.execute(
            """SELECT pb.market_id, pb.event_id, pb.sector, pb.yes_team,
                      pb.event_title, pb.event_date, pb.sharp_true_prob, pb.blended_true_prob,
                      pb.market_type, pb.line
               FROM portfolio_bets pb
               LEFT JOIN ev_outcomes o ON pb.market_id = o.market_id
               WHERE (pb.event_date = ? OR pb.scan_date = ?)
                 AND o.market_id IS NULL
                 AND COALESCE(pb.event_date, pb.scan_date) <= ?""",
            (date_str, date_str, today_str),
        ).fetchall()
        seen = {r["market_id"] for r in pending}
        pending = list(pending) + [r for r in portfolio_pending if r["market_id"] not in seen]

    if not pending:
        logger.info("no_pending_outcomes", date=date_str)
        # Still resolve prop_observations even if no ev_predictions pending
        prop_obs_resolved = _resolve_prop_observations(conn, target_date)
        conn.commit()
        conn.close()
        return {"resolved": prop_obs_resolved, "failed": 0, "unmatched": []}

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

    # Resolve NBA player props. ESPN box scores cover the slate in one
    # scoreboard + N summary calls per date — far cheaper than the previous
    # per-(player, stat) stats.nba.com hammering that was timing out at 30s
    # under the load of a full prop slate. nba_api stays as a fallback for
    # players/stats ESPN doesn't surface, with `_fetch_gamelog_sync` LRU-
    # cached so repeat stats on the same player share one roundtrip.
    #
    # Threshold authority: prefer pred["line"] (the Kalshi threshold the bet
    # was placed at) over event_id::parts[5]. Pre-2026-05-11 the synthetic
    # SharpOdds event_id encoded the Pinnacle anchor line (e.g. 4.5) while
    # the bet itself was at a different Kalshi threshold (e.g. 10), so
    # parsing event_id resolved a 10+ bet against a 4.5 cutoff and marked
    # losers as WON. event_id stays the fallback for player/stat/date
    # extraction since those don't drift the same way.
    espn_stats_cache: dict[date, dict[str, dict[str, float]]] = {}

    for pred in prop_preds:
        parts = pred["event_id"].split("::")
        if len(parts) < 6:
            failed += 1
            unmatched.append(pred["event_id"])
            continue
        _player, _stat = parts[3], parts[4]
        # Prefer the explicit line column; fall back to event_id parsing for
        # pre-line-column rows.
        _threshold = pred.get("line") if pred.get("line") is not None else None
        if _threshold is None:
            try:
                _threshold = float(parts[5])
            except ValueError:
                failed += 1
                unmatched.append(pred["event_id"])
                continue
        else:
            _threshold = float(_threshold)
        _event_date = date.fromisoformat(parts[1])

        outcome = None
        source = "nba_api"

        if pred.get("sector") == "nba" or pred.get("sector") is None:
            if _event_date not in espn_stats_cache:
                espn_stats_cache[_event_date] = _fetch_espn_nba_player_stats(_event_date)
            espn_outcome = _resolve_prop_from_espn(
                espn_stats_cache[_event_date], _player, _stat, _threshold,
            )
            if espn_outcome is not None:
                outcome = espn_outcome[0]
                source = "espn_boxscore"

        if outcome is None:
            outcome = _resolve_prop_outcome(_player, _stat, _threshold, _event_date)

        if outcome is not None:
            _write_outcome(conn, pred, outcome, source)
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

    # ------------------------------------------------------------------
    # Resolve prop_observations table (separate from ev_predictions props)
    # ------------------------------------------------------------------
    prop_obs_resolved = _resolve_prop_observations(conn, target_date)
    resolved += prop_obs_resolved

    conn.commit()
    conn.close()

    if unmatched:
        logger.warning(
            "unmatched_events",
            count=len(unmatched),
            date=date_str,
            events=unmatched[:20],  # cap log size
        )

    logger.info("resolve_done", date=date_str, resolved=resolved, failed=failed,
                prop_obs_resolved=prop_obs_resolved)
    return {"resolved": resolved, "failed": failed, "unmatched": unmatched}


def backfill_clv(since: Optional[date] = None, until: Optional[date] = None) -> dict:
    """Backfill two distinct CLV measurements for resolved bets.

    Both pull from data/archive.db (Pinnacle line snapshots + Kalshi market
    snapshots) for the bet's event:

      ev_predictions.pinnacle_drift_pct
          = pinnacle_close_prob - kalshi_entry_price (in pp)
          What it measures: how much Kalshi's quote at entry lagged Pinnacle's
          eventual pre-tipoff prob. Positive by construction for our system
          because we only pick bets where pinnacle_scan > kalshi_entry — so
          this number mostly captures Kalshi's softness-vs-Pinnacle, not
          model edge. Kept as a diagnostic; do not interpret as CLV.

      ev_predictions.kalshi_clv_pct
          = kalshi_close_yes_price - kalshi_entry_price (in pp)
          What it measures: did Kalshi's OWN market move toward our YES side
          between entry and T-30-minutes pre-tipoff? Pinnacle isn't in this
          formula. This is the conventional sharp-bettor CLV adapted to the
          fact that Kalshi never truly closes — T-30 is the industry proxy
          for "last price before event-info corrupts the line."

    Also populates ev_outcomes.pinnacle_close_prob (yes-aligned).

    Returns: {"updated": N, "skipped": M, "avg_pinn_drift": x, "avg_kalshi_clv": y}
    """
    from evmax.archiver import DataArchiver

    archiver = DataArchiver()
    conn = get_connection()

    since_str = (since or date(2020, 1, 1)).isoformat()
    until_str = (until or date.today()).isoformat()

    # Pull all bets that need backfill — either missing pinn drift OR missing
    # kalshi CLV. Resolved-only since both metrics are scored post-game.
    rows = conn.execute(
        """SELECT p.id, p.market_id, p.event_id, p.kalshi_yes_price,
                  p.yes_team, p.event_date, p.scan_date,
                  p.pinnacle_drift_pct, p.kalshi_clv_pct,
                  p.market_type, p.line
           FROM ev_predictions p
           INNER JOIN ev_outcomes o ON p.market_id = o.market_id
           WHERE o.outcome IS NOT NULL
             AND (p.pinnacle_drift_pct IS NULL OR p.kalshi_clv_pct IS NULL)
             AND p.scan_date >= ? AND p.scan_date <= ?""",
        (since_str, until_str),
    ).fetchall()

    updated = skipped = 0
    pinn_drift_values: list[float] = []
    kalshi_clv_values: list[float] = []

    for row in rows:
        entry_price = row["kalshi_yes_price"]
        if entry_price is None or entry_price <= 0 or entry_price >= 1:
            skipped += 1
            continue

        market_id = row["market_id"]
        ticker = market_id.removeprefix("kalshi:") if market_id else None

        # ---- 1. Pinnacle drift (legacy CLV) — pre-tipoff only ----
        # Dispatch by market_type so spread bets get spread snapshots instead
        # of an unrelated moneyline close. Totals align by over/under side
        # (yes_team is "over"/"under") rather than team label.
        mt = (row["market_type"] or "").lower()
        if mt == "spread":
            pinn_close = archiver.get_spread_closing_line_aligned(
                row["event_id"], row["yes_team"], row["line"]
            )
        elif mt == "total":
            pinn_close = archiver.get_total_closing_line_aligned(
                row["event_id"], row["yes_team"], row["line"]
            )
        elif mt in ("moneyline", "ml", ""):
            pinn_close = archiver.get_closing_line_aligned(
                row["event_id"], row["yes_team"]
            )
        else:
            pinn_close = None
        pinn_drift_pp: Optional[float] = None
        if pinn_close is not None:
            pinn_drift_pp = (pinn_close - entry_price) * 100
            conn.execute(
                "UPDATE ev_outcomes SET pinnacle_close_prob = ? WHERE market_id = ?",
                (pinn_close, market_id),
            )
            conn.execute(
                "UPDATE ev_predictions SET pinnacle_drift_pct = ? WHERE id = ?",
                (round(pinn_drift_pp, 2), row["id"]),
            )
            pinn_drift_values.append(pinn_drift_pp)

        # ---- 2. Kalshi-CLV — T-30 pre-tipoff snapshot ----
        # Use Pinnacle's archived tipoff timestamp as the anchor for "30 min
        # pre-tipoff." If that's missing fall back to event_date midnight UTC
        # (still better than the no-filter LIMIT 1 we had before).
        kalshi_clv_pp: Optional[float] = None
        if ticker:
            kalshi_close = archiver.get_kalshi_close_price(ticker, row["event_id"])
            if kalshi_close is not None:
                kalshi_clv_pp = (kalshi_close - entry_price) * 100
                conn.execute(
                    "UPDATE ev_predictions SET kalshi_clv_pct = ? WHERE id = ?",
                    (round(kalshi_clv_pp, 2), row["id"]),
                )
                kalshi_clv_values.append(kalshi_clv_pp)

        if pinn_drift_pp is None and kalshi_clv_pp is None:
            skipped += 1
        else:
            updated += 1

    conn.commit()
    conn.close()

    avg_pd = sum(pinn_drift_values) / len(pinn_drift_values) if pinn_drift_values else 0.0
    avg_kc = sum(kalshi_clv_values) / len(kalshi_clv_values) if kalshi_clv_values else 0.0
    logger.info(
        "clv_backfill_done",
        updated=updated,
        skipped=skipped,
        avg_pinn_drift=round(avg_pd, 2),
        avg_kalshi_clv=round(avg_kc, 2),
        n_pinn=len(pinn_drift_values),
        n_kalshi=len(kalshi_clv_values),
    )
    return {
        "updated": updated,
        "skipped": skipped,
        "avg_pinn_drift": round(avg_pd, 2),
        "avg_kalshi_clv": round(avg_kc, 2),
        "n_pinn": len(pinn_drift_values),
        "n_kalshi": len(kalshi_clv_values),
    }
