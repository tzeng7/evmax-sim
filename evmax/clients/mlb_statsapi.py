"""Official MLB Stats API client (statsapi.mlb.com).

Free, no auth, structured JSON. Used as the primary source for live probable
starters — it replaces the ESPN scoreboard scrape, which gave ERA-only data
and keyed teams by ``shortDisplayName`` (so multi-word nicknames like
"Red Sox"/"Blue Jays" silently failed name matching and dropped the pitcher
model from the blend; see PitcherModelAgent).

Why this is the better source:
  - Teams carry a **stable integer ID**, so probable-starter resolution no
    longer depends on fuzzy nickname matching.
  - Probable pitchers come with a stable player ID + canonical fullName.
  - The same schedule endpoint also hydrates lineups / venue / day-night,
    which the bullpen + lineup signals will consume next.

The agent gets the pitcher's ERA/FIP from its own seeded ``pitchers`` DB
(scripts/seed_pitcher_fip.py) keyed by accent-folded fullName, so the live
feed only needs to answer "who is starting" — which this does cleanly.
"""

from __future__ import annotations

import time
import unicodedata
from datetime import date as _date
from typing import Any, Optional

import structlog

from evmax.clients.base import BaseAPIClient

logger = structlog.get_logger(__name__)

MLB_STATS_BASE = "https://statsapi.mlb.com/api/v1"

# Stable MLB team id → canonical nickname. These nicknames match the values
# used by scripts/seed_pitcher_fip.py::TEAM_MAP and ESPN's
# shortDisplayName.lower(), so downstream lookups (team_starters, the suffix
# matcher) stay consistent across data sources.
MLB_TEAM_NICKNAME: dict[int, str] = {
    108: "angels",
    109: "diamondbacks",
    110: "orioles",
    111: "red sox",
    112: "cubs",
    113: "reds",
    114: "guardians",
    115: "rockies",
    116: "tigers",
    117: "astros",
    118: "royals",
    119: "dodgers",
    120: "nationals",
    121: "mets",
    133: "athletics",
    134: "pirates",
    135: "padres",
    136: "mariners",
    137: "giants",
    138: "cardinals",
    139: "rays",
    140: "rangers",
    141: "blue jays",
    142: "twins",
    143: "phillies",
    144: "braves",
    145: "white sox",
    146: "marlins",
    147: "yankees",
    158: "brewers",
}


def ascii_fold(name: str) -> str:
    """Lowercase + strip accents, matching the seed's pitcher-name keys.

    'Cristopher Sánchez' → 'cristopher sanchez'. The seeded ``pitchers`` DB
    folds names this way (scripts/seed_pitcher_fip.py::_ascii_fold), so the
    live feed must fold identically or the ERA/FIP lookup misses.
    """
    nfkd = unicodedata.normalize("NFKD", name or "")
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


class MLBStatsClient(BaseAPIClient):
    """Async client for the public MLB Stats API."""

    def __init__(self, timeout: float = 10.0) -> None:
        super().__init__(base_url=MLB_STATS_BASE, concurrency=5, timeout=timeout)

    async def schedule(
        self, date: Optional[str] = None, hydrate: str = "probablePitcher"
    ) -> dict[str, Any]:
        """Raw schedule payload for a date (YYYY-MM-DD, defaults to today)."""
        params = {
            "sportId": 1,
            "date": date or _date.today().isoformat(),
            "hydrate": hydrate,
        }
        return await self._get("/schedule", params=params)

    async def probable_starters(self, date: Optional[str] = None) -> dict[str, dict]:
        """Return ``{nickname: {"name", "era", "ip", "mlb_id", "mlb_team_id"}}``.

        ``name`` is the accent-folded fullName (matches the seeded pitcher DB).
        ``era`` defaults to 0.0 / ``ip`` to 0 — the agent fills real values
        from its seed; the live feed's job is identity, not stats.
        """
        payload = await self.schedule(date=date)
        return _parse_probables(payload)

    async def boxscore(self, game_pk: int) -> dict[str, Any]:
        """Raw boxscore payload for one game."""
        return await self._get(f"/game/{game_pk}/boxscore")


def _parse_probables(payload: dict[str, Any]) -> dict[str, dict]:
    """Pure parser — extracted so it's unit-testable against a fixture."""
    result: dict[str, dict] = {}
    for date_block in payload.get("dates", []):
        for game in date_block.get("games", []):
            teams = game.get("teams", {})
            for side in ("home", "away"):
                t = teams.get(side, {})
                team_id = t.get("team", {}).get("id")
                nickname = MLB_TEAM_NICKNAME.get(team_id)
                if not nickname:
                    continue
                pp = t.get("probablePitcher") or {}
                full = pp.get("fullName")
                if not full:
                    continue  # TBD starter — no probable announced yet
                result[nickname] = {
                    "name": ascii_fold(full),
                    "era": 0.0,  # filled from seeded DB by the agent
                    "ip": 0,
                    "mlb_id": pp.get("id"),
                    "mlb_team_id": team_id,
                }
    return result


def parse_innings_pitched(raw: Any) -> float:
    """Baseball IP notation → float innings: "1.2" = 1 + 2/3, not 1.2."""
    try:
        text = str(raw)
        whole, _, frac = text.partition(".")
        outs = int(frac) if frac else 0
        return int(whole or 0) + min(outs, 2) / 3.0
    except (TypeError, ValueError):
        return 0.0


def parse_reliever_appearances(
    boxscore: dict[str, Any], game_date: str
) -> dict[str, tuple[str, float, int]]:
    """Extract each NON-starter pitcher's (date, ip, pitch_count) from one
    boxscore payload. Keyed by accent-folded fullName (the seeded pitcher DB
    key). Pure parser — unit-testable against a fixture.
    """
    out: dict[str, tuple[str, float, int]] = {}
    for side in ("home", "away"):
        team = (boxscore.get("teams") or {}).get(side) or {}
        players = team.get("players") or {}
        for pid in team.get("pitchers") or []:
            p = players.get(f"ID{pid}") or {}
            stats = (p.get("stats") or {}).get("pitching") or {}
            if not stats or stats.get("gamesStarted", 0):
                continue  # starter (or no line) — the fatigue log is pen-only
            full = (p.get("person") or {}).get("fullName")
            if not full:
                continue
            ip = parse_innings_pitched(stats.get("inningsPitched", "0"))
            pc = stats.get("numberOfPitches") or stats.get("pitchesThrown") or 0
            try:
                pc = int(pc)
            except (TypeError, ValueError):
                pc = 0
            out[ascii_fold(full)] = (game_date, ip, pc)
    return out


# ---------------------------------------------------------------------------
# Cached module-level accessors (drop-ins for the agent)
# ---------------------------------------------------------------------------

_CACHE_TTL = 30 * 60  # 30 minutes — same cadence as the old ESPN cache
_probable_cache: dict[str, dict] = {}
_probable_cache_ts: float = 0.0
_probable_cache_date: Optional[str] = None

# Reliever fatigue log: refreshed every 3h — fatigue signal changes daily,
# not intraday, but games finish through the evening so a fixed daily pull
# would miss the previous night's appearances on morning scans.
_APPEARANCES_TTL = 3 * 3600
_appearances_cache: dict[str, list] = {}
_appearances_cache_ts: float = 0.0
_appearances_cache_key: Optional[str] = None


async def fetch_probable_starters(date: Optional[str] = None) -> dict[str, dict]:
    """Cached probable starters from the MLB Stats API.

    Returns an empty dict on any error so callers can fall back to a secondary
    source (the agent falls back to the ESPN scoreboard).
    """
    global _probable_cache, _probable_cache_ts, _probable_cache_date
    key = date or _date.today().isoformat()
    if (
        _probable_cache
        and _probable_cache_date == key
        and (time.time() - _probable_cache_ts) < _CACHE_TTL
    ):
        return _probable_cache
    try:
        async with MLBStatsClient() as client:
            result = await client.probable_starters(date=date)
        _probable_cache = result
        _probable_cache_ts = time.time()
        _probable_cache_date = key
        logger.info("mlb_probable_starters_fetched", teams=len(result), date=key)
        return result
    except Exception as e:  # noqa: BLE001 — degrade to fallback source
        logger.warning("mlb_probable_starters_failed", error=str(e))
        return {}


async def fetch_reliever_appearances(
    days_back: int = 4, today: Optional[str] = None
) -> dict[str, list]:
    """Reliever appearance log for the trailing ``days_back`` days.

    Returns ``{name_key: [[iso_date, ip, pitch_count], ...]}`` (chronological),
    covering every non-starter pitching line in final games — the live feed
    for the bullpen fatigue heuristic (evmax/models_ml/bullpen.py:
    is_reliever_available). ~15 schedule days + ~15 boxscores per day at
    concurrency 5 completes in seconds.

    3h-TTL module cache; returns {} on total failure (the agent degrades to
    the seeded appearance log, then to starter-only rates).
    """
    global _appearances_cache, _appearances_cache_ts, _appearances_cache_key
    from datetime import timedelta

    anchor = _date.fromisoformat(today) if today else _date.today()
    key = f"{anchor.isoformat()}:{days_back}"
    if (
        _appearances_cache
        and _appearances_cache_key == key
        and (time.time() - _appearances_cache_ts) < _APPEARANCES_TTL
    ):
        return _appearances_cache

    appearances: dict[str, list] = {}
    try:
        async with MLBStatsClient() as client:
            for offset in range(days_back, 0, -1):
                day = (anchor - timedelta(days=offset)).isoformat()
                try:
                    sched = await client.schedule(date=day, hydrate="")
                except Exception as e:  # noqa: BLE001 — skip the day, keep going
                    logger.warning("mlb_appearances_day_failed", date=day, error=str(e))
                    continue
                game_pks = [
                    g.get("gamePk")
                    for block in sched.get("dates", [])
                    for g in block.get("games", [])
                    if (g.get("status") or {}).get("abstractGameState") == "Final"
                    and g.get("gamePk")
                ]
                import asyncio as _asyncio

                boxes = await _asyncio.gather(
                    *(client.boxscore(pk) for pk in game_pks),
                    return_exceptions=True,
                )
                for box in boxes:
                    if isinstance(box, Exception):
                        continue
                    for name_key, app in parse_reliever_appearances(box, day).items():
                        appearances.setdefault(name_key, []).append(list(app))
    except Exception as e:  # noqa: BLE001 — degrade entirely
        logger.warning("mlb_appearances_failed", error=str(e))
        return {}

    for entries in appearances.values():
        entries.sort(key=lambda a: a[0])

    _appearances_cache = appearances
    _appearances_cache_ts = time.time()
    _appearances_cache_key = key
    logger.info(
        "mlb_reliever_appearances_fetched",
        relievers=len(appearances),
        days=days_back,
    )
    return appearances
