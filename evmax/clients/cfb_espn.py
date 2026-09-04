"""ESPN college-football play-by-play client (seed / backtest time only).

Why ESPN and not cfbfastR: the cfbfastR-data play-by-play mirror is only current
through 2022 (verified 2026-08-01), so it cannot feed live-season EPA. ESPN's
public college-football ``summary`` endpoint carries full drive/play data (down,
distance, yards-to-endzone, running score, turnover flags) for every FBS game,
current within minutes of final — and needs no API key. This client walks a
season's scoreboard for the game list, pulls each game's plays, and emits the
per-play rows the EPA math core (evmax/agents/models/_cfb_efficiency.py) consumes.

NOT used in the live scan path — only by scripts/seed_ncaaf_efficiency.py and
scripts/backtest_ncaaf_efficiency.py. Raw summaries are cached to disk by game_id
so the weekly reseed only fetches the week's new games and the backtest pays the
full-season fetch once.
"""

from __future__ import annotations

import datetime as dt
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Optional

import httpx
import structlog

logger = structlog.get_logger(__name__)

SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard"
)
SUMMARY = (
    "https://site.api.espn.com/apis/site/v2/sports/football/college-football/summary"
)
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "cfb_espn"

# CFB season windows (label = fall year; runs late Aug → mid Jan CFP).
SEASON_WINDOWS: dict[int, tuple[str, str]] = {
    2021: ("2021-08-28", "2022-01-11"),
    2022: ("2022-08-25", "2023-01-10"),
    2023: ("2023-08-25", "2024-01-09"),
    2024: ("2024-08-24", "2025-01-21"),
    2025: ("2025-08-23", "2026-01-20"),
    2026: ("2026-08-22", "2027-01-19"),
    2027: ("2027-08-28", "2028-01-18"),
}


def _half(period: Optional[int]) -> int:
    """ESPN period number → half (1 or 2). OT (period ≥ 5) folds into half 2 so
    the next-score scan doesn't leak across the regulation/OT boundary oddly;
    OT plays are dropped from EPA anyway (untimed, different EP regime)."""
    if not period:
        return 1
    return 1 if period <= 2 else 2


def fetch_scoreboard_day(client: httpx.Client, day: dt.date) -> list[dict]:
    """FBS games (groups=80) on one date. Returns lightweight game rows."""
    params = {"dates": day.strftime("%Y%m%d"), "groups": "80", "limit": 400}
    try:
        r = client.get(SCOREBOARD, params=params, timeout=30)
        r.raise_for_status()
        events = r.json().get("events", [])
    except Exception as e:  # noqa: BLE001
        logger.warning("cfb_scoreboard_fail", day=str(day), error=str(e))
        return []
    games = []
    for e in events:
        # ESPN occasionally lists a placeholder event (TBD/postponed) with no
        # competitions block — skip it rather than abort the whole day walk
        # (2026-09-03: the Week-2 scoreboard carried one and killed the reseed).
        comps = e.get("competitions") or []
        if not comps or not e.get("id"):
            continue
        comp = comps[0]
        status = comp.get("status", {}).get("type", {})
        home = away = None
        for c in comp["competitors"]:
            side = c.get("homeAway")
            rec = {
                "id": c["team"]["id"],
                "abbr": c["team"].get("abbreviation"),
                "location": c["team"].get("location"),
                "score": _safe_int(c.get("score")),
            }
            if side == "home":
                home = rec
            elif side == "away":
                away = rec
        if not home or not away:
            continue
        games.append(
            {
                "game_id": e["id"],
                "date": e.get("date", "")[:10],
                "neutral": bool(comp.get("neutralSite", False)),
                "completed": bool(status.get("completed", False)),
                "home": home,
                "away": away,
            }
        )
    return games


def _safe_int(v) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def fetch_season_games(season: int, only_completed: bool = True) -> list[dict]:
    """All FBS games in a season, walking the scoreboard day by day."""
    start_s, end_s = SEASON_WINDOWS[season]
    start = dt.date.fromisoformat(start_s)
    end = dt.date.fromisoformat(end_s)
    out: list[dict] = []
    seen: set[str] = set()
    with httpx.Client() as client:
        day = start
        while day <= end:
            for g in fetch_scoreboard_day(client, day):
                if g["game_id"] in seen:
                    continue
                if only_completed and not g["completed"]:
                    continue
                seen.add(g["game_id"])
                g["season"] = season
                out.append(g)
            day += dt.timedelta(days=1)
    logger.info("cfb_season_games", season=season, games=len(out))
    return out


def _cache_path(game_id: str) -> Path:
    return CACHE_DIR / f"{game_id}.json"


def fetch_game_summary(game_id: str, client: Optional[httpx.Client] = None,
                       use_cache: bool = True) -> Optional[dict]:
    """Raw ESPN summary for a game (disk-cached by game_id)."""
    cp = _cache_path(game_id)
    if use_cache and cp.exists():
        try:
            return json.loads(cp.read_text())
        except json.JSONDecodeError:
            pass
    own = client is None
    client = client or httpx.Client()
    try:
        r = client.get(SUMMARY, params={"event": game_id}, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:  # noqa: BLE001
        logger.warning("cfb_summary_fail", game_id=game_id, error=str(e))
        return None
    finally:
        if own:
            client.close()
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(data))
    return data


def parse_game_plays(summary: dict, game_meta: dict) -> list[dict]:
    """Extract per-play EPA-input rows from an ESPN summary.

    Emits scrimmage plays with: game_id, half, off_team, def_team, down,
    distance, yards_to_goal, end_down, end_distance, end_yards_to_goal,
    end_team, yards_gained, score_points, score_off, score_team, and the
    running score margin BEFORE the play (for garbage-time filtering). OT plays
    (period ≥ 5) are dropped.
    """
    drives = (summary.get("drives") or {}).get("previous") or []
    if not drives:
        return []
    home_id = game_meta["home"]["id"]
    away_id = game_meta["away"]["id"]
    rows: list[dict] = []
    prev_total = 0
    prev_home = prev_away = 0
    for drive in drives:
        off_team = (drive.get("team") or {}).get("id")
        if not off_team:
            continue
        def_team = away_id if off_team == home_id else home_id
        for p in drive.get("plays", []):
            period = (p.get("period") or {}).get("number")
            if period and period >= 5:  # overtime — different EP regime
                continue
            st = p.get("start") or {}
            en = p.get("end") or {}
            home_score = _safe_int(p.get("homeScore")) or 0
            away_score = _safe_int(p.get("awayScore")) or 0
            new_total = home_score + away_score
            score_points = 0
            score_team = None
            score_off = False
            if new_total > prev_total:
                score_points = new_total - prev_total
                if home_score > prev_home:
                    score_team = home_id
                elif away_score > prev_away:
                    score_team = away_id
                score_off = score_team == off_team
            # margin BEFORE this play, from the offense's perspective
            off_pre = prev_home if off_team == home_id else prev_away
            def_pre = prev_away if off_team == home_id else prev_home
            rows.append(
                {
                    "game_id": game_meta["game_id"],
                    "half": _half(period),
                    "period": period,
                    "off_team": off_team,
                    "def_team": def_team,
                    "down": _safe_int(st.get("down")),
                    "distance": _safe_int(st.get("distance")),
                    "yards_to_goal": _safe_int(st.get("yardsToEndzone")),
                    "end_down": _safe_int(en.get("down")),
                    "end_distance": _safe_int(en.get("distance")),
                    "end_yards_to_goal": _safe_int(en.get("yardsToEndzone")),
                    "end_team": (en.get("team") or {}).get("id"),
                    "yards_gained": _safe_int(p.get("statYardage")) or 0,
                    "type": (p.get("type") or {}).get("text", ""),
                    "score_points": score_points,
                    "score_team": score_team,
                    "score_off": score_off,
                    "off_margin_pre": off_pre - def_pre,
                }
            )
            prev_total = new_total
            prev_home, prev_away = home_score, away_score
    return rows


def fetch_season_plays(
    season: int,
    game_ids: Optional[Iterable[str]] = None,
    max_workers: int = 8,
    use_cache: bool = True,
    games: Optional[list[dict]] = None,
) -> tuple[list[dict], list[dict]]:
    """Fetch + parse plays for a full season (or a subset of game_ids).

    Returns (play_rows, games) where games is the scoreboard metadata list
    (used for FBS-id set, final scores, dates). Summaries are fetched
    concurrently and cached to disk. Pass ``games`` (e.g. a full-schedule
    list from ``fetch_season_games(season, only_completed=False)``) to skip
    the scoreboard walk; only its COMPLETED rows are fetched/parsed.
    """
    if games is None:
        games = fetch_season_games(season)
    else:
        games = [g for g in games if g.get("completed")]
    if game_ids is not None:
        wanted = set(game_ids)
        games = [g for g in games if g["game_id"] in wanted]
    by_id = {g["game_id"]: g for g in games}

    def _one(gid: str) -> list[dict]:
        with httpx.Client() as c:
            s = fetch_game_summary(gid, client=c, use_cache=use_cache)
        if not s:
            return []
        return parse_game_plays(s, by_id[gid])

    all_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for rows in ex.map(_one, [g["game_id"] for g in games]):
            all_rows.extend(rows)
    logger.info("cfb_season_plays", season=season, games=len(games), plays=len(all_rows))
    return all_rows, games
