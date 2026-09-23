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

from evmax.agents.models._cfb_efficiency import LEGAL_PLAY_POINTS

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


# A per-team score above this is a feed typo (e.g. "14" entered as 1414), not
# a football score; the play's score is treated as missing.
_MAX_PLAUSIBLE_SCORE = 200


def _scoring_delta(
    prev_home: int, prev_away: int, home_score: int, away_score: int
) -> tuple[int, Optional[str]]:
    """Points scored ON this play and which side ('home'/'away') scored them.

    ESPN reports a running scoreboard, so a play's points are the scoreboard
    delta. The delta is not safe to trust blindly: a missing score coerced to
    0, a skipped play, or a typo (1414 for 14) turns it into a multi-score or
    cumulative value (1400 observed on game 401866418; 720 illegal deltas over
    353 cached games, 2021-2026). Only a delta where EXACTLY ONE team gains a
    legal single-play amount counts; anything else is (0, None).
    """
    dh = home_score - prev_home
    da = away_score - prev_away
    if dh > 0 and da == 0 and dh in LEGAL_PLAY_POINTS:
        return dh, "home"
    if da > 0 and dh == 0 and da in LEGAL_PLAY_POINTS:
        return da, "away"
    return 0, None


def _read_scores(p: dict, prev_home: int, prev_away: int) -> tuple[int, int]:
    """Scoreboard after play ``p``, never below the running baseline.

    Carries the previous value forward when the feed omits a score, reports an
    implausible one, or LOWERS a team's score. ESPN routinely shows the
    PRE-score scoreboard on the kickoff row after a touchdown ((10,21) →
    kickoff (10,14) → next snap (10,21)); adopting the dip as the baseline
    re-credited the touchdown to that ordinary snap — 609–669 phantom credits a
    season (88–178 on scrimmage plays), all of LEGAL size, so the legal-amount
    check could not see them. A genuine downward correction (a TD overturned
    after the scoreboard moved) now costs at most one missed credit instead.
    """
    home = _safe_int(p.get("homeScore"))
    away = _safe_int(p.get("awayScore"))
    if home is None or not 0 <= home <= _MAX_PLAUSIBLE_SCORE:
        home = prev_home
    if away is None or not 0 <= away <= _MAX_PLAUSIBLE_SCORE:
        away = prev_away
    return max(home, prev_home), max(away, prev_away)


def _possession_team(p: dict, drive_team: str, home_id: str, away_id: str) -> str:
    """The offense on play ``p``: the play's own ``start.team`` when it names
    one of the two sides, else the drive's team.

    ESPN's drive-level ``team`` is the less reliable field: game 401856784
    labels every Baylor drive after the first quarter as Prairie View A&M's,
    which flipped the sign of ~70 plays' EPA (Baylor def_epa_adj −0.95 on a
    neutral success rate). ~2,100 scrimmage plays over ~800 cached games
    disagree, and a sample of them names the ``start.team`` side's players in
    the play text. Kickoffs keep the drive team: their ``start.team`` is the
    kicking side, not the possession the drive describes.
    """
    if "kickoff" in ((p.get("type") or {}).get("text") or "").lower():
        return drive_team
    start_team = ((p.get("start") or {}).get("team") or {}).get("id")
    if start_team in (home_id, away_id):
        return start_team
    return drive_team


def parse_game_plays(summary: dict, game_meta: dict) -> list[dict]:
    """Extract per-play EPA-input rows from an ESPN summary.

    Emits scrimmage plays with: game_id, half, off_team, def_team, down,
    distance, yards_to_goal, end_down, end_distance, end_yards_to_goal,
    end_team, yards_gained, score_points, score_off, score_team, and the
    running score margin BEFORE the play (for garbage-time filtering). OT plays
    (period ≥ 5) are dropped.

    ``off_team`` is the play's own possession team (see ``_possession_team``).

    ``score_points`` is always 0 or a legal single-play amount credited to one
    team (see ``_scoring_delta``).
    """
    drives = (summary.get("drives") or {}).get("previous") or []
    if not drives:
        return []
    home_id = game_meta["home"]["id"]
    away_id = game_meta["away"]["id"]
    rows: list[dict] = []
    prev_home = prev_away = 0
    for drive in drives:
        drive_team = (drive.get("team") or {}).get("id")
        if not drive_team:
            continue
        for p in drive.get("plays", []):
            period = (p.get("period") or {}).get("number")
            if period and period >= 5:  # overtime — different EP regime
                continue
            off_team = _possession_team(p, drive_team, home_id, away_id)
            def_team = away_id if off_team == home_id else home_id
            st = p.get("start") or {}
            en = p.get("end") or {}
            home_score, away_score = _read_scores(p, prev_home, prev_away)
            score_points, side = _scoring_delta(prev_home, prev_away, home_score, away_score)
            score_team = None
            score_off = False
            if side is not None:
                score_team = home_id if side == "home" else away_id
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
