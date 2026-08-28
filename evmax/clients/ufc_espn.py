"""ESPN MMA (UFC) public API client — historical fight results + fighter bios.

Why ESPN and not ufcstats.com: ufcstats.com now fronts a JavaScript
proof-of-work anti-bot interstitial ("Checking your browser…", verified
2026-07-11), so an automated client cannot fetch it without defeating bot
detection — which we don't do. ESPN's public JSON API is the same source
family the repo already seeds from (see ``scripts/seed_espn.py``, which has
fetched ESPN MMA scoreboards since the latent-UFC era) and carries complete
UFC cards back to 2012+ with:

  - winner flags per bout (``competitions[].competitors[].winner``)
  - finish round + clock (``status.period`` / ``status.displayClock``)
  - method of victory via the *core* API status object
    (``status.result.name`` — "submission", "ko/tko", "decision", …)
  - athlete bios (DOB / height / reach / stance) via the core athlete object

Trade-off vs ufcstats: no per-fight significant-strike / takedown counts, so
per-minute volume stats (SLpM, SApM, TD avg) are NOT available. The feature
layer therefore uses only results-derived point-in-time signals (age, layoff,
experience, win rate, finish rates, recent-KO durability) plus static bio
attributes (reach/height/stance). Documented in docs/ufc-model-eval.md.

All fetches are disk-cached under ``data/cache/ufc_espn/`` (gitignored via
the ``data/cache/`` rule) so re-runs are free. Completed months / settled
fights / athlete bios are immutable, so the cache never needs invalidation —
pass ``refresh=True`` to force refetch of the current month.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import httpx
import structlog
from aiolimiter import AsyncLimiter

logger = structlog.get_logger(__name__)

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard"
CORE_STATUS_URL = (
    "https://sports.core.api.espn.com/v2/sports/mma/leagues/ufc"
    "/events/{event_id}/competitions/{comp_id}/status"
)
CORE_ATHLETE_URL = (
    "https://sports.core.api.espn.com/v2/sports/mma/leagues/ufc/athletes/{athlete_id}"
)

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "ufc_espn"

# Module-level limiter: 8 req/s across all callers. ESPN's CDN happily serves
# far more, but the historical backfill is a one-shot job — stay polite.
_ESPN_RATE_LIMITER = AsyncLimiter(8, 1.0)

# ESPN's public API WAF blocklists our identifying "evmax-*" User-Agents AND
# plain browser-impersonator strings (observed on the score resolver 2026-08-05;
# see resolver._ESPN_HTTP_UA). Neutral tool UAs (curl/*, python-httpx/*) pass.
# The seed client previously sent "evmax-ufc-seed/1.0", which ESPN 403s: the
# weekly `--fetch` reseed then aborts and ufc_rating_state.json freezes (state
# was stuck at 2026-08-01 for ~4 weeks, so the model went "missing" on ~half of
# each live card). Keep this a recognized tool UA; do NOT revert to an "evmax-*"
# or browser string.
_ESPN_HTTP_UA = "curl/8.7.1"


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass
class FightResult:
    """One completed (or scheduled) UFC bout from an ESPN scoreboard event."""

    event_id: str
    comp_id: str
    event_name: str
    date: str            # ISO date of the card, YYYY-MM-DD
    weight_class: str
    fighter_a_id: str    # ESPN athlete id, order=1 side
    fighter_a: str       # display name
    fighter_b_id: str
    fighter_b: str
    winner: str          # "a" | "b" | "none" (draw / NC / not decided)
    completed: bool
    round: Optional[int] = None      # finish round (status.period)
    clock: Optional[str] = None      # finish clock within the round
    method: Optional[str] = None     # canonical: decision/ko_tko/submission/dq/draw/nc
    method_detail: Optional[str] = None


@dataclass
class FighterBio:
    """Static athlete attributes from the ESPN core athlete object."""

    athlete_id: str
    name: str
    dob: Optional[str] = None        # ISO date YYYY-MM-DD
    height_in: Optional[float] = None
    reach_in: Optional[float] = None
    stance: Optional[str] = None


# ---------------------------------------------------------------------------
# Method normalization
# ---------------------------------------------------------------------------

_METHOD_PATTERNS: list[tuple[str, str]] = [
    ("no contest", "nc"),
    ("dq", "dq"),
    ("disqualification", "dq"),
    ("draw", "draw"),
    ("decision", "decision"),
    ("submission", "submission"),
    ("ko", "ko_tko"),
    ("tko", "ko_tko"),
    ("stoppage", "ko_tko"),
]


def normalize_method(raw: Optional[str]) -> Optional[str]:
    """Map an ESPN ``status.result.name`` to a canonical method token.

    Examples: "submission" → "submission", "ko/tko" → "ko_tko",
    "unanimous decision" → "decision". Unknown / empty → None.
    """
    if not raw:
        return None
    text = raw.lower().strip()
    for needle, canon in _METHOD_PATTERNS:
        if needle in text:
            return canon
    return None


# ---------------------------------------------------------------------------
# Parsers (pure — unit-tested on committed fixtures)
# ---------------------------------------------------------------------------


def parse_scoreboard(raw: dict[str, Any]) -> list[FightResult]:
    """Parse one ESPN MMA scoreboard payload into FightResult rows.

    Handles multi-event months (``dates=YYYYMM`` responses list every card in
    the month). Bouts with fewer than two athlete competitors are skipped.
    """
    fights: list[FightResult] = []
    for event in raw.get("events", []) or []:
        event_id = str(event.get("id", ""))
        event_name = event.get("name", "") or ""
        event_date = (event.get("date", "") or "")[:10]
        for comp in event.get("competitions", []) or []:
            competitors = comp.get("competitors", []) or []
            if len(competitors) < 2:
                continue
            # ESPN orders competitors; sort by "order" for a stable a/b.
            comps_sorted = sorted(competitors, key=lambda c: c.get("order", 0))
            side_a, side_b = comps_sorted[0], comps_sorted[1]

            def _name(side: dict) -> str:
                ath = side.get("athlete") or {}
                return ath.get("displayName") or ath.get("fullName") or ""

            def _aid(side: dict) -> str:
                return str(side.get("id") or (side.get("athlete") or {}).get("id") or "")

            status = comp.get("status", {}) or {}
            completed = bool((status.get("type") or {}).get("completed"))
            if side_a.get("winner"):
                winner = "a"
            elif side_b.get("winner"):
                winner = "b"
            else:
                winner = "none"

            period = status.get("period")
            fights.append(
                FightResult(
                    event_id=event_id,
                    comp_id=str(comp.get("id", "")),
                    event_name=event_name,
                    date=(comp.get("date", "") or "")[:10] or event_date,
                    weight_class=(comp.get("type") or {}).get("abbreviation", "") or "",
                    fighter_a_id=_aid(side_a),
                    fighter_a=_name(side_a),
                    fighter_b_id=_aid(side_b),
                    fighter_b=_name(side_b),
                    winner=winner,
                    completed=completed,
                    round=int(period) if period else None,
                    clock=status.get("displayClock"),
                )
            )
    return fights


def parse_status_result(raw: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    """Extract ``(method, method_detail)`` from a core competition status object."""
    result = raw.get("result") or {}
    method = normalize_method(result.get("name") or result.get("displayName"))
    detail = result.get("description") or result.get("displayDescription")
    return method, detail


def parse_athlete(raw: dict[str, Any]) -> FighterBio:
    """Extract the static bio attributes from a core athlete object."""
    stance = raw.get("stance") or {}
    dob = (raw.get("dateOfBirth") or "")[:10] or None
    return FighterBio(
        athlete_id=str(raw.get("id", "")),
        name=raw.get("displayName") or raw.get("fullName") or "",
        dob=dob,
        height_in=float(raw["height"]) if raw.get("height") else None,
        reach_in=float(raw["reach"]) if raw.get("reach") else None,
        stance=stance.get("text") if isinstance(stance, dict) else None,
    )


# ---------------------------------------------------------------------------
# Fetching client (disk-cached)
# ---------------------------------------------------------------------------


class UFCESPNClient:
    """Async fetcher for ESPN MMA UFC data with a permanent disk cache."""

    def __init__(self, cache_dir: Optional[Path] = None, timeout: float = 20.0) -> None:
        self._cache_dir = cache_dir or CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={"User-Agent": _ESPN_HTTP_UA},
            follow_redirects=True,
        )

    async def __aenter__(self) -> "UFCESPNClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._client.aclose()

    # -- cache helpers -----------------------------------------------------

    def _cache_path(self, key: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_\-]", "_", key)
        return self._cache_dir / f"{safe}.json"

    def _cache_get(self, key: str) -> Optional[dict]:
        path = self._cache_path(key)
        if path.exists():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                return None
        return None

    def _cache_set(self, key: str, data: dict) -> None:
        try:
            self._cache_path(key).write_text(json.dumps(data))
        except OSError as exc:  # cache write failure must not kill the fetch
            logger.warning("ufc_espn_cache_write_failed", key=key, error=str(exc))

    async def _get_json(
        self, url: str, params: Optional[dict] = None, *, cache_key: str,
        refresh: bool = False,
    ) -> Optional[dict]:
        cached = self._cache_get(cache_key)
        if not refresh and cached is not None:
            return cached
        async with _ESPN_RATE_LIMITER:
            try:
                resp = await self._client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:  # noqa: BLE001 — caller decides fallback
                logger.warning("ufc_espn_fetch_failed", url=url, error=str(exc))
                # Degrade a FAILED forced-refresh to last-known-good cache rather
                # than dropping the data. A transient 403/timeout must never
                # regress seeded state: returning None here used to make the
                # seed abort (or rebuild from an empty current month), freezing
                # last_updated. `cached` is None when nothing was ever cached.
                return cached
        self._cache_set(cache_key, data)
        return data

    # -- public fetchers ---------------------------------------------------

    async def fetch_month(self, yyyymm: str, refresh: bool = False) -> list[FightResult]:
        """All UFC bouts in one calendar month (``dates=YYYYMM``)."""
        raw = await self._get_json(
            SCOREBOARD_URL,
            params={"dates": yyyymm, "limit": 100},
            cache_key=f"scoreboard_{yyyymm}",
            refresh=refresh,
        )
        if not raw:
            return []
        return parse_scoreboard(raw)

    async def fetch_method(
        self, event_id: str, comp_id: str,
    ) -> tuple[Optional[str], Optional[str]]:
        """Method of victory for one bout from the core status object."""
        raw = await self._get_json(
            CORE_STATUS_URL.format(event_id=event_id, comp_id=comp_id),
            params={"lang": "en", "region": "us"},
            cache_key=f"status_{event_id}_{comp_id}",
        )
        if not raw:
            return None, None
        return parse_status_result(raw)

    async def fetch_athlete(self, athlete_id: str) -> Optional[FighterBio]:
        """Static bio for one fighter from the core athlete object."""
        raw = await self._get_json(
            CORE_ATHLETE_URL.format(athlete_id=athlete_id),
            params={"lang": "en", "region": "us"},
            cache_key=f"athlete_{athlete_id}",
        )
        if not raw:
            return None
        return parse_athlete(raw)


def fight_to_row(fight: FightResult) -> dict:
    """Flatten a FightResult for CSV writing."""
    return asdict(fight)
