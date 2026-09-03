"""Pinnacle odds via the guest Arcadia API — covers all sectors.

No credentials required. Replaces TheOddsAPI for Pinnacle sharp lines.

Sport IDs:
  4  = Basketball (NBA id=487, NCAA Men's id=493, WNCAA id=583)
  12 = E Sports   (CS2, LoL, Valorant)
  15 = Football   (NFL id=889 — re-cut from 258 for the 2026 season)
  19 = Hockey     (NHL id=1456)
  29 = Soccer     (EPL id=1980, La Liga id=2196, Bundesliga id=1842,
                   Serie A id=2436, Ligue 1 id=2036, UCL id=2627 (was 2186 through 2025-26),
                   UEL id=2630, MLS id=2663)

Odds are American format; we convert via american_to_decimal() + devig_two_way().
Soccer three-way (draw) odds use devig_three_way().
"""

from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
import structlog

from evmax.clients.base import BaseAPIClient
from evmax.clients.time_util import kalshi_game_day
from evmax.disk_cache import cache_get, cache_set, cache_get_offline
from evmax.ev.devig import (
    american_to_decimal,
    derive_advance_prob,
    devig_three_way,
    devig_two_way,
)
from evmax.models.odds import SharpBook, SharpOdds
from evmax.settings import get_settings

logger = structlog.get_logger(__name__)

# Pinnacle prefixes doubleheader-nightcap participant names with "G1 "/"G2 "
# (e.g. "G2 New York Yankees"). Baseball-only — see `_normalize`.
_DOUBLEHEADER_PREFIX_RE = re.compile(r"^g[12]\s+", re.IGNORECASE)

GUEST_API_BASE = "https://guest.api.arcadia.pinnacle.com/0.1"

# sport_id → list of league_ids we care about
SECTOR_SPORT_LEAGUES: dict[str, tuple[int, list[int]]] = {
    "nba":      (4,  [487]),
    "ncaab":    (4,  [493]),
    "ncaaw":    (4,  [583]),    # WNCAA
    "wnba":     (4,  [578]),    # WNBA
    # NFL league id was 258 through the 2025 season; Pinnacle re-cut it to 889
    # for 2026 (verified 2026-09-02 via GET /sports/15/leagues — the only
    # football leagues listed are 876 CFL / 880 NCAA / 889 NFL). A stale id
    # returns ZERO matchups, i.e. no sharp anchor and no NFL EV all season.
    # Re-verify every league id at each season boundary (see docs/SEASON_START.md).
    "nfl":      (15, [889]),
    "ncaaf":    (15, [880]),    # NCAA football (FBS) — sport 15, league 880 ("NCAA")
    "nhl":      (19, [1456]),   # NHL
    "baseball": (3,  [246]),    # MLB
    "ufc":      (22, []),       # Mixed Martial Arts — matched by league name
    "f1":       (44, []),       # Formula 1 — matched by league name
    # UCL re-cut 2186 → 2627 for 2026-27 (scripts/check_pinnacle_leagues.py,
    # 2026-09-02: 2186 served 0 matchups, 2627 "UEFA - Champions League" 18).
    "soccer":   (29, [1980, 2196, 1842, 2436, 2036, 2627, 2630, 2663]),
    #                EPL  LaLiga Bundes SerieA Ligue1 UCL   UEL   MLS
    "worldcup": (29, [2686]),   # FIFA - World Cup (national teams, 3-way w/ draw)
    "cs2":      (12, []),       # Esports — matched by league name
    "lol":      (12, []),
    "valorant": (12, []),
    "tennis":   (33, []),       # ATP/WTA — matched by league name (ATP Miami, WTA Miami, etc.)
}

# Sectors that use league-name matching instead of league IDs
NAME_MATCHED_LEAGUE_MAP: list[tuple[str, str]] = [
    ("CS2", "cs2"),
    ("Counter-Strike", "cs2"),
    ("League of Legends", "lol"),
    ("Valorant", "valorant"),
    ("UFC", "ufc"),
    ("Bellator", "ufc"),      # Bellator MMA also captured under ufc sector
    ("Formula 1", "f1"),
    ("Formula One", "f1"),
    ("ATP", "tennis"),
    ("WTA", "tennis"),
]

# All sectors this client can handle
ALL_SECTORS = set(SECTOR_SPORT_LEAGUES.keys())
# Sectors resolved by league name (not league ID list)
NAME_MATCHED_SECTORS = {"cs2", "lol", "valorant", "ufc", "f1", "tennis"}
ESPORTS_SECTORS = {"cs2", "lol", "valorant"}  # kept for backward compat
# Scoring team sports that publish a Pinnacle game-total market. Anything
# outside this set will have its `type == "total"` Pinnacle markets ignored,
# which makes downstream KXxxxTOTAL Kalshi alternates unmatchable. Add a
# sector here when its totals model goes live (e.g. wnba_possession_sim).
TOTALS_SECTORS = {"nba", "wnba", "nfl", "ncaab", "ncaaw", "ncaaf", "soccer", "baseball", "nhl"}

# Sectors that price ONLY the main total line, not Pinnacle's alternate ladder.
# Mirrors how spreads already take only the non-alternate line (see _parse).
# Kalshi lists ~10 alternate O/U strikes per baseball game; matching each to its
# exact Pinnacle alt counterpart floods the scan with deep-tail lines whose
# normal-CDF cover prob is unreliable (skewed/fat-tailed run distribution) and
# whose alt ladders are low-liquidity. Emitting only the main total collapses
# each game to its standard line; the totals model's distance cap then rejects
# any Kalshi strike that sits too far from it. Baseball totals are also disabled
# for persistence, so this is purely display-noise reduction there — but the
# same skew argument means alt totals were never bettable signal anyway.
TOTALS_MAIN_LINE_ONLY_SECTORS = {"baseball"}

# Soccer leagues that have draws (all of them); MLS uses draws too
SOCCER_DRAW_LEAGUES = {1980, 2196, 1842, 2436, 2036, 2627, 2630, 2663}

# --- World Cup knockout "to advance" anchors -------------------------------
# Pinnacle has NO live per-match "to advance" market. Its per-team "To Reach
# Quarter/Semi/The Final" Yes/No specials look like one, but they cut off at
# the team's PREVIOUS kickoff and never reopen between rounds (verified
# 2026-07-05: "Mexico To Reach Quarter Final" still served pre-Round-of-32
# prices — Yes +242 ≈ 29% — on R16 matchday, when the live advance prob was
# ~40%+). So the advance anchor for Kalshi's KXWCADVANCE markets is DERIVED
# from the live regulation 3-way instead, via derive_advance_prob():
#   P(A advances) = p_a + p_draw * p_a / (p_a + p_b)
# Kalshi's liquid advance consensus tracks this derivation closely (FRA-MAR
# QF: derived 0.75 vs market 0.76), and the regulation 3-way is Pinnacle's
# sharpest, most liquid market for the game — a better anchor than any
# thin special would have been.


def _name_matched_sector(league_name: str) -> Optional[str]:
    """Return sector for a league resolved by name matching (esports, UFC, F1)."""
    for keyword, sector in NAME_MATCHED_LEAGUE_MAP:
        if keyword.lower() in league_name.lower():
            return sector
    return None


# Pinnacle stat label → canonical evmax stat_type. Keep keys lowercase.
_PROP_STAT_MAP: dict[str, str] = {
    "points": "points", "pts": "points",
    "rebounds": "rebounds", "reb": "rebounds", "rebs": "rebounds",
    "assists": "assists", "ast": "assists", "asts": "assists",
    "threes": "threes", "threes made": "threes",
    "3-pointers": "threes", "3 pointers made": "threes",
    "steals": "steals", "stl": "steals",
    "blocks": "blocks", "blk": "blocks",
    "pts+reb+ast": "points_rebounds_assists",
    "points + rebounds + assists": "points_rebounds_assists",
    "pts & rebs & asts": "points_rebounds_assists",
    "pra": "points_rebounds_assists",
    "passing yards": "passing_yards", "passing yds": "passing_yards",
    "rushing yards": "rushing_yards", "rushing yds": "rushing_yards",
    "receiving yards": "receiving_yards", "receiving yds": "receiving_yards",
    # NFL per-game specials observed live 2026-09-02 ("Christian McCaffrey
    # Total Receptions", "Brock Purdy Total Touchdown Passes") — both are
    # Kalshi series we trade (KXNFLREC / KXNFLPASSTDS).
    "receptions": "receptions", "rec": "receptions",
    "touchdown passes": "passing_tds", "passing touchdowns": "passing_tds",
    "passing tds": "passing_tds", "td passes": "passing_tds",
    # --- MLB (added 2026-06-27). Pinnacle posts these as legacy-paren specials
    # with a "(must start)" qualifier suffix, e.g. "Rafael Devers (Home Runs)
    # (must start)" / "Michael Lorenzen (Total Strikeouts)(must start)". The
    # qualifier is stripped before parsing (see parse_prop_description). Only
    # the four stats Pinnacle actually posts for MLB are mapped; the pitcher's
    # Hits-Allowed / Earned-Runs specials have no Kalshi series we trade, so
    # they're intentionally left unmapped (parse → None → dropped). ---
    "total strikeouts": "strikeouts", "strikeouts": "strikeouts",
    "total bases": "total_bases",
    "home runs": "home_runs",
    "pitching outs": "pitching_outs", "outs recorded": "pitching_outs",
}

# Trailing qualifier annotations Pinnacle appends to MLB prop descriptions
# (e.g. "...(must start)"). Stripped before stat parsing so the real
# "(Stat)" group is the only parenthetical left.
_PROP_DESC_QUALIFIER_RE = re.compile(
    r"\s*\((?:must start|must play|must pitch|must be active)\)\s*$",
    re.IGNORECASE,
)
# Legacy / MLB format: "Luka Doncic (Points)" / "Devers (Home Runs)"
_PROP_DESC_PAREN_RE = re.compile(r"^(.+?)\s*\((.+?)\)\s*$")
# New NBA/NFL format (May 2026 onward): "Jalen Brunson Total Assists"
_PROP_DESC_TOTAL_RE = re.compile(r"^(.+?)\s+Total\s+(.+)$", re.IGNORECASE)


def parse_prop_description(description: str) -> Optional[tuple[str, str]]:
    """Parse a Pinnacle player-prop ``special.description`` into ``(player, stat_type)``.

    Handles three shapes:
      - legacy / MLB paren: ``"Luka Doncic (Points)"`` / ``"Devers (Home Runs)
        (must start)"`` (the ``(must start)`` qualifier is stripped first),
      - new NBA/NFL "Total" form: ``"Jalen Brunson Total Assists"``.

    The paren form is tried first: an MLB label like ``"Total Strikeouts"`` sits
    *inside* the parens, so the bare ``_PROP_DESC_TOTAL_RE`` would mis-split it
    on the word "Total". Returns ``None`` when the description can't be parsed
    or the stat label is not in :data:`_PROP_STAT_MAP`.
    """
    if not description:
        return None
    text = _PROP_DESC_QUALIFIER_RE.sub("", description.strip()).strip()
    for pattern in (_PROP_DESC_PAREN_RE, _PROP_DESC_TOTAL_RE):
        m = pattern.match(text)
        if not m:
            continue
        player = m.group(1).strip()
        stat_raw = m.group(2).strip().lower()
        stat_type = _PROP_STAT_MAP.get(stat_raw)
        if stat_type is None:
            return None
        return player, stat_type
    return None


def classify_pinnacle_error(exc: BaseException) -> tuple[Optional[int], str]:
    """Classify a Pinnacle fetch failure into ``(status_code, reason)``.

    Pinnacle is the sole sharp anchor; when it fails the scan must fail CLEAR
    (produce no plays — never price a bet against a stale line) but the operator
    needs to know WHY. Distinguishes the two failure modes that actually happen:
    a maintenance window (503) and the US geo-block (403 ``BAD_LOCATION``,
    observed intermittently), plus rate-limiting and network errors.

    Reasons: ``maintenance`` | ``geo_block`` | ``forbidden`` | ``rate_limited``
    | ``http_error`` | ``timeout`` | ``network`` | ``error``.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code == 503:
            return code, "maintenance"
        if code == 403:
            body = ""
            try:
                body = (exc.response.text or "").upper()
            except Exception:  # noqa: BLE001 — body may be unreadable
                body = ""
            if "BAD_LOCATION" in body or "LOCATION" in body:
                return code, "geo_block"
            return code, "forbidden"
        if code == 429:
            return code, "rate_limited"
        return code, "http_error"
    if isinstance(exc, httpx.TimeoutException):
        return None, "timeout"
    if isinstance(exc, httpx.TransportError):
        return None, "network"
    return None, "error"


class PinnacleGuestClient(BaseAPIClient):
    """
    Fetches sharp Pinnacle odds from the public guest Arcadia API.
    Covers NBA, NCAAB, NFL, Baseball, Soccer, UFC, F1, CS2, LoL, Valorant.
    No API key or account required.
    """

    def __init__(self) -> None:
        super().__init__(
            base_url=GUEST_API_BASE,
            concurrency=8,
            timeout=20.0,
            headers={
                "Accept": "application/json",
                "X-Api-Key": "CmX2KcMrXuFmNg6YFbmTxE0y9CIrOi0R",
            },
        )
        # Set to {"sector", "status", "reason"} when the most recent top-level
        # fetch failed (matchups list); reset to None on a successful fetch. Lets
        # a caller / the heartbeat probe know Pinnacle is down and WHY.
        self.last_error: Optional[dict] = None

    async def probe(self, sector: str = "nba") -> dict:
        """Lightweight reachability check (one matchups fetch), for the heartbeat.

        Returns ``{"ok": bool, "status": int|None, "reason": str}``. Measures
        whether the request SUCCEEDS, not whether markets are listed (an empty
        off-season list is still ``ok``). Never raises — a probe failure is data.
        """
        sector = sector.lower()
        if sector not in SECTOR_SPORT_LEAGUES:
            sector = "nba"
        sport_id, _ = SECTOR_SPORT_LEAGUES[sector]
        try:
            data = await self._logged_get(
                f"/sports/{sport_id}/matchups",
                params={"withSpecials": "false"},
                sector=sector,
                purpose="probe",
            )
        except Exception as e:  # noqa: BLE001 — probe reports, never raises
            status, reason = classify_pinnacle_error(e)
            self.last_error = {"sector": sector, "status": status, "reason": reason}
            return {"ok": False, "status": status, "reason": reason}
        self.last_error = None
        return {"ok": isinstance(data, list), "status": 200, "reason": "ok"}

    async def _logged_get(
        self,
        path: str,
        params: Optional[dict[str, Any]] = None,
        *,
        sector: str,
        purpose: str,
    ) -> Any:
        """Wrap BaseAPIClient._get with an INFO-level call log.

        Emits one `pinnacle_api_call` event per HTTP request with sector,
        purpose (`list_matchups`, `fetch_matchup_markets`, etc.), full URL,
        query params, wall-clock start time, duration, and item count. This
        is the source of truth for "what Pinnacle calls did scan make".
        """
        t0 = time.perf_counter()
        started_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        status = "ok"
        err: Optional[str] = None
        items = 0
        try:
            data = await self._get(path, params=params)
            if isinstance(data, list):
                items = len(data)
            elif isinstance(data, dict):
                items = len(data)
            return data
        except Exception as e:
            status = "error"
            err = str(e) or type(e).__name__
            raise
        finally:
            duration_ms = round((time.perf_counter() - t0) * 1000, 1)
            logger.info(
                "pinnacle_api_call",
                sector=sector,
                purpose=purpose,
                url=f"{self.base_url}{path}",
                params=params,
                started_at=started_at,
                duration_ms=duration_ms,
                status=status,
                items=items,
                **({"error": err} if err else {}),
            )

    async def get_prop_odds(self, sector: str) -> list[SharpOdds]:
        """Fetch devigged Pinnacle player prop lines from the guest Arcadia API.

        Pinnacle posts player props as 'special' matchups with
        special.category == 'Player Props' and description like 'Luka Doncic (Points)'.
        Each prop has one over/under market at a specific line.

        Supports: nba, nfl, ncaab, baseball.
        """
        sector = sector.lower()
        _PROP_SECTORS = {"nba", "nfl", "ncaab", "baseball"}
        if sector not in _PROP_SECTORS or sector not in SECTOR_SPORT_LEAGUES:
            return []

        sport_id, league_ids = SECTOR_SPORT_LEAGUES[sector]
        prop_t0 = time.perf_counter()

        try:
            all_matchups = await self._logged_get(
                f"/sports/{sport_id}/matchups",
                params={"withSpecials": "true"},
                sector=f"{sector}_props",
                purpose="list_prop_matchups",
            )
        except Exception as e:
            status, reason = classify_pinnacle_error(e)
            self.last_error = {"sector": f"{sector}_props", "status": status, "reason": reason}
            logger.warning(
                "pinnacle_guest_props_matchups_failed",
                sector=sector, status=status, reason=reason, error=str(e),
            )
            return []

        self.last_error = None

        if not isinstance(all_matchups, list):
            return []

        # Filter to player prop specials for the right leagues
        prop_matchups = [
            m for m in all_matchups
            if m.get("type") == "special"
            and m.get("league", {}).get("id") in league_ids
            and (m.get("special") or {}).get("category", "").lower() == "player props"
        ]

        if not prop_matchups:
            logger.info("pinnacle_guest_no_props", sector=sector)
            return []

        # Price the specials off ONE bulk sport-level straight-markets fetch.
        # The per-matchup ``/matchups/{id}/markets/related/straight`` endpoint
        # is geo-blocked from the US for SPECIAL matchups specifically (403
        # BAD_LOCATION — verified 2026-09-02: the parent game's matchup id
        # serves fine, every prop special's id 403s), which is why props were
        # the sector the intermittent geo-block "hit hardest". The bulk
        # ``/sports/{sport_id}/markets/straight`` endpoint (NO ``withSpecials``
        # param — passing it trips the same block) still carries every
        # special's over/under market keyed by ``matchupId``. Matchups the
        # bulk index lacks fall back to the per-matchup fetch, and a bulk
        # failure degrades to the old per-matchup path for everything.
        by_matchup = await self._fetch_bulk_straight_index(sport_id, sector)
        bulk_hits = [m for m in prop_matchups if m.get("id") in by_matchup]
        bulk_misses = [m for m in prop_matchups if m.get("id") not in by_matchup]

        props: list[SharpOdds] = []
        for m in bulk_hits:
            parsed = self._parse_prop_matchup(m, by_matchup[m.get("id")], sector)
            if parsed is not None:
                props.append(parsed)

        if bulk_misses:
            results = await asyncio.gather(
                *(self._fetch_prop_matchup(m, sector) for m in bulk_misses),
                return_exceptions=True,
            )
            for r in results:
                if isinstance(r, SharpOdds):
                    props.append(r)
                elif isinstance(r, list):
                    props.extend(r)

        logger.info(
            "pinnacle_props_fetched",
            sector=sector,
            count=len(props),
            bulk_priced=len(bulk_hits),
            per_matchup_fallback=len(bulk_misses),
        )
        logger.info(
            "pinnacle_scan_done",
            sector=f"{sector}_props",
            sport_id=sport_id,
            league_ids=league_ids or None,
            parent_matchups=len(prop_matchups),
            http_calls=1 + (1 if by_matchup else 0) + len(bulk_misses),
            total_markets=len(props),
            duration_ms=round((time.perf_counter() - prop_t0) * 1000, 1),
        )
        return props

    async def _fetch_bulk_straight_index(
        self, sport_id: int, sector: str
    ) -> dict[int, list[dict]]:
        """Fetch ``/sports/{sport_id}/markets/straight`` once and index it by matchupId.

        Returns an empty dict on any failure (callers fall back to per-matchup
        fetches) so a bulk outage can never make props WORSE than before.
        Deliberately sends no query params: ``withSpecials=true`` on this
        endpoint triggers the same US geo-block as the per-special fetch,
        while the bare call already includes the specials' markets.
        """
        try:
            data = await self._logged_get(
                f"/sports/{sport_id}/markets/straight",
                sector=f"{sector}_props",
                purpose="list_prop_markets_bulk",
            )
        except Exception as e:  # noqa: BLE001 — degrade to per-matchup path
            status, reason = classify_pinnacle_error(e)
            logger.warning(
                "pinnacle_props_bulk_markets_failed",
                sector=sector, status=status, reason=reason, error=str(e),
            )
            return {}
        if not isinstance(data, list):
            return {}
        by_matchup: dict[int, list[dict]] = {}
        for mk in data:
            mid = mk.get("matchupId")
            if mid is None:
                continue
            by_matchup.setdefault(mid, []).append(mk)
        return by_matchup

    async def _fetch_prop_matchup(self, matchup: dict, sector: str) -> Optional[SharpOdds]:
        """Fetch and parse a single player prop special matchup (per-matchup path).

        Kept as the fallback for specials the bulk straight-markets index
        lacks; from the US this endpoint 403s for special matchup ids, so the
        bulk path in ``get_prop_odds`` is the one that normally prices props.
        """
        matchup_id = matchup.get("id")
        if parse_prop_description((matchup.get("special") or {}).get("description", "")) is None:
            logger.debug(
                "pinnacle_prop_unparsed",
                description=(matchup.get("special") or {}).get("description", ""),
            )
            return None

        try:
            markets_data = await self._logged_get(
                f"/matchups/{matchup_id}/markets/related/straight",
                sector=f"{sector}_props",
                purpose="fetch_prop_markets",
            )
        except Exception as e:
            logger.debug("pinnacle_prop_markets_failed", matchup_id=matchup_id, error=str(e))
            return None

        if not isinstance(markets_data, list):
            return None

        # /matchups/{id}/markets/related/straight also returns markets for the
        # parent game (totals like 218.0, spreads, etc.). Filter to markets
        # that belong to this prop matchup itself, otherwise the first
        # type=='total' we'd pick is the parent game total — see the Randle
        # assists case where matching against a 218.0 line wiped 50% of props.
        own_markets = [mk for mk in markets_data if mk.get("matchupId") == matchup_id]
        return self._parse_prop_matchup(matchup, own_markets, sector)

    def _parse_prop_matchup(
        self, matchup: dict, own_markets: list[dict], sector: str
    ) -> Optional[SharpOdds]:
        """Parse one prop special + ITS OWN straight markets into a SharpOdds.

        Pure (no network): ``own_markets`` must already be filtered to this
        matchup's id, whether it came from the bulk sport-level index or the
        per-matchup endpoint.
        """
        special = matchup.get("special") or {}
        description = special.get("description", "")

        parsed = parse_prop_description(description)
        if parsed is None:
            logger.debug("pinnacle_prop_unparsed", description=description)
            return None
        player_raw, stat_type = parsed

        # Normalize player name
        from evmax.players import normalize_player_name
        player_norm = normalize_player_name(player_raw, sector)

        start_time = matchup.get("startTime", "")
        event_date: Optional[datetime] = None
        if start_time:
            try:
                event_date = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            except ValueError:
                pass

        ou_market = next(
            (mk for mk in own_markets if mk.get("type") == "total"),
            None,
        )
        if not ou_market:
            return None

        prices = ou_market.get("prices", [])
        # Prop market prices have designation=None; positional: index 0 = over, index 1 = under
        over_entry  = next((p for p in prices if p.get("designation") == "over"),  None)
        under_entry = next((p for p in prices if p.get("designation") == "under"), None)
        if not over_entry or not under_entry:
            # Fall back to positional indexing for prop markets
            if len(prices) >= 2:
                over_entry, under_entry = prices[0], prices[1]
            else:
                return None

        raw_line = over_entry.get("points") or over_entry.get("handicap")
        if raw_line is None:
            return None
        total_line = float(raw_line)

        try:
            over_dec  = american_to_decimal(int(over_entry["price"]))
            under_dec = american_to_decimal(int(under_entry["price"]))
            prob_over, prob_under, margin = devig_two_way(over_dec, under_dec)
        except Exception as e:
            logger.debug("pinnacle_prop_devig_failed", error=str(e))
            return None

        date_str = kalshi_game_day(event_date, sector)
        event_id = f"{sector}::{date_str}::prop::{player_norm}::{stat_type}::{total_line}"

        return SharpOdds(
            event_id=event_id,
            book=SharpBook.pinnacle,
            sector=sector,
            outcome_a_label="over",
            outcome_b_label="under",
            outcome_a_decimal=over_dec,
            outcome_b_decimal=under_dec,
            true_prob_a=0.0,
            true_prob_b=0.0,
            true_prob_over=prob_over,
            true_prob_under=prob_under,
            total_line=total_line,
            margin=margin,
            event_date=event_date,
            prop_player_name=player_norm,
            prop_stat_type=stat_type,
        )

    async def get_odds(self, sector: str) -> list[SharpOdds]:
        """Fetch devigged Pinnacle moneyline odds for a sector."""
        sector = sector.lower()
        if sector not in ALL_SECTORS:
            return []

        cfg = get_settings()
        cache_key = f"pinnacle_{sector}"

        # Offline mode: always use cache (stale is fine)
        if cfg.offline_mode:
            raw = cache_get_offline(cache_key)
            return [SharpOdds.model_validate(r) for r in raw]

        # Dev cache: skip API if fresh cached data exists
        if cfg.cache_ttl_secs > 0:
            raw = cache_get(cache_key, cfg.cache_ttl_secs)
            if raw is not None:
                return [SharpOdds.model_validate(r) for r in raw]

        sport_id, league_ids = SECTOR_SPORT_LEAGUES[sector]
        scan_t0 = time.perf_counter()

        try:
            all_matchups = await self._logged_get(
                f"/sports/{sport_id}/matchups",
                params={"withSpecials": "false"},
                sector=sector,
                purpose="list_matchups",
            )
        except Exception as e:
            status, reason = classify_pinnacle_error(e)
            self.last_error = {"sector": sector, "status": status, "reason": reason}
            # Fail CLEAR: no sharp anchor → no plays for this sector. Distinct
            # reason so a maintenance window / geo-block isn't an indistinct blip.
            logger.warning(
                "pinnacle_guest_matchups_failed",
                sector=sector, status=status, reason=reason, error=str(e),
            )
            return []

        # Fetch succeeded — clear any prior failure marker for this client.
        self.last_error = None

        if not isinstance(all_matchups, list):
            return []

        # Filter to parent matchups for this sector
        if sector in NAME_MATCHED_SECTORS:
            matchups = [
                m for m in all_matchups
                if m.get("parentId") is None
                and m.get("type") == "matchup"
                and _name_matched_sector(m.get("league", {}).get("name", "")) == sector
            ]
        else:
            matchups = [
                m for m in all_matchups
                if m.get("parentId") is None
                and m.get("type") == "matchup"
                and m.get("league", {}).get("id") in league_ids
            ]

        if not matchups:
            logger.info("pinnacle_guest_no_matchups", sector=sector)
            return []

        results = await asyncio.gather(
            *(self._fetch_matchup_odds(m, sector) for m in matchups),
            return_exceptions=True,
        )

        odds: list[SharpOdds] = []
        for r in results:
            if isinstance(r, SharpOdds):
                odds.append(r)
            elif isinstance(r, list):
                odds.extend(r)
            elif isinstance(r, Exception):
                logger.warning("pinnacle_guest_odds_error", error=str(r))

        # Per-market-type breakdown for the scan summary
        market_counts: dict[str, int] = {}
        for o in odds:
            mt = getattr(o, "market_type", None)
            key = str(mt.value) if mt is not None and hasattr(mt, "value") else (str(mt) if mt else "moneyline")
            market_counts[key] = market_counts.get(key, 0) + 1

        logger.info("sharp_fetched", sector=sector, count=len(odds),
                    avg_margin=round(sum(o.margin for o in odds) / len(odds), 4) if odds else 0.0)
        logger.info(
            "pinnacle_scan_done",
            sector=sector,
            sport_id=sport_id,
            league_ids=league_ids or None,
            parent_matchups=len(matchups),
            http_calls=1 + len(matchups),  # 1 list + N per-matchup market fetches
            market_counts=market_counts,
            total_markets=len(odds),
            duration_ms=round((time.perf_counter() - scan_t0) * 1000, 1),
        )

        # Write to dev cache if enabled
        if cfg.cache_ttl_secs > 0 and odds:
            cache_set(cache_key, [o.model_dump(mode="json") for o in odds])

        return odds

    async def _fetch_matchup_odds(self, matchup: dict, sector: str) -> Optional[SharpOdds] | list[SharpOdds]:
        """Fetch moneyline (and spread for non-soccer) for one matchup."""
        matchup_id = matchup.get("id")
        if not matchup_id:
            return None

        participants = matchup.get("participants", [])
        home = next((p["name"] for p in participants if p.get("alignment") == "home"), None)
        away = next((p["name"] for p in participants if p.get("alignment") == "away"), None)
        if not home or not away:
            return None

        start_time = matchup.get("startTime", "")
        event_date: Optional[datetime] = None
        if start_time:
            try:
                event_date = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            except ValueError:
                pass

        date_str = kalshi_game_day(event_date, sector)
        home_norm = self._normalize(home, sector)
        away_norm = self._normalize(away, sector)
        base_event_id = f"{sector}::{date_str}::{home_norm}_vs_{away_norm}"

        try:
            markets_data = await self._logged_get(
                f"/matchups/{matchup_id}/markets/related/straight",
                sector=sector,
                purpose="fetch_matchup_markets",
            )
        except Exception as e:
            logger.debug("pinnacle_guest_markets_failed", matchup_id=matchup_id, error=str(e))
            return None

        if not isinstance(markets_data, list):
            return None

        results: list[SharpOdds] = []

        # --- Moneyline ---
        ml_market = next(
            (m for m in markets_data
             if m.get("matchupId") == matchup_id
             and m.get("type") == "moneyline"
             and m.get("period") == 0
             and not m.get("isAlternate", False)
             and m.get("status") == "open"),
            None,
        )
        if ml_market:
            ml_odds = self._parse_moneyline(ml_market, base_event_id, sector, home, away, event_date)
            if ml_odds:
                results.append(ml_odds)
                # World Cup: derive the knockout "to advance" anchor from the
                # regulation 3-way (see the advance-anchor note up top). Group
                # games get one too, but no KXWCADVANCE market exists for them
                # and the ::advance pool filter keeps the record unmatchable.
                if sector == "worldcup":
                    advance_odds = self.derive_advance_odds(ml_odds)
                    if advance_odds:
                        results.append(advance_odds)

        # --- Spread (only for team sports with meaningful point spreads) ---
        # Soccer + World Cup use Asian-handicap goal lines we don't model and
        # whose Kalshi spread series aren't wired, so skip them like club soccer.
        if sector not in NAME_MATCHED_SECTORS and sector not in ("soccer", "worldcup"):
            spread_market = next(
                (m for m in markets_data
                 if m.get("matchupId") == matchup_id
                 and m.get("type") == "spread"
                 and m.get("period") == 0
                 and not m.get("isAlternate", False)
                 and m.get("status") == "open"),
                None,
            )
            if spread_market:
                spread_odds = self._parse_spread(spread_market, base_event_id, sector, home, away, event_date)
                if spread_odds:
                    results.append(spread_odds)

        # --- Totals (scoring team sports only — excludes esports, UFC, F1, tennis) ---
        # Pinnacle exposes the main total plus a long ladder of alternate
        # totals (isAlternate=True) at ±0.5/±1 intervals around the main line.
        # Kalshi's totals series carries ~10 alternate lines per game, only
        # one or two of which sit within 2pts of the Pinnacle main line —
        # without alternates, the matching engine's nearest-line fallback
        # drops the rest. Fetch ALL period-0 totals so each Kalshi line can
        # match its exact Pinnacle counterpart and devig from that book's
        # own juice rather than via extrapolation off a distant main line.
        if sector in TOTALS_SECTORS:
            main_line_only = sector in TOTALS_MAIN_LINE_ONLY_SECTORS
            totals_markets = [
                m for m in markets_data
                if m.get("matchupId") == matchup_id
                and m.get("type") == "total"
                and m.get("period") == 0
                and m.get("status") == "open"
                and not (main_line_only and m.get("isAlternate", False))
            ]
            for totals_market in totals_markets:
                totals_odds = self._parse_totals(totals_market, base_event_id, sector, event_date)
                if totals_odds:
                    results.append(totals_odds)

        return results if results else None

    @staticmethod
    def derive_advance_odds(ml_odds: SharpOdds) -> Optional[SharpOdds]:
        """Derive the knockout "to advance" sharp record from a regulation 3-way.

        See the World Cup advance-anchor note above SECTOR_SPORT_LEAGUES for
        why this is derived rather than read off a Pinnacle special. Emits a
        SharpOdds keyed `{event_id}::advance` that only ::advance-typed Kalshi
        markets can match (MatchingEngine filters the candidate pool by that
        suffix).
        """
        prob_a = derive_advance_prob(
            ml_odds.true_prob_a, ml_odds.true_prob_b, ml_odds.true_prob_draw,
        )
        if prob_a is None or not (0.0 < prob_a < 1.0):
            return None
        prob_b = 1.0 - prob_a
        return SharpOdds(
            event_id=f"{ml_odds.event_id}::advance",
            book=ml_odds.book,
            sector=ml_odds.sector,
            outcome_a_label=ml_odds.outcome_a_label,
            outcome_b_label=ml_odds.outcome_b_label,
            # Fair (devigged) decimals — this record is derived, so there are
            # no raw per-outcome book prices behind it.
            outcome_a_decimal=1.0 / prob_a,
            outcome_b_decimal=1.0 / prob_b,
            true_prob_a=prob_a,
            true_prob_b=prob_b,
            margin=ml_odds.margin,
            event_date=ml_odds.event_date,
        )

    def _parse_moneyline(
        self, market: dict, base_event_id: str, sector: str,
        home: str, away: str, event_date: Optional[datetime],
    ) -> Optional[SharpOdds]:
        prices = market.get("prices", [])

        home_price = next((p["price"] for p in prices if p.get("designation") == "home"), None)
        away_price = next((p["price"] for p in prices if p.get("designation") == "away"), None)
        draw_price = next((p["price"] for p in prices if p.get("designation") == "draw"), None)

        if home_price is None or away_price is None:
            return None

        try:
            home_dec = american_to_decimal(int(home_price))
            away_dec = american_to_decimal(int(away_price))

            if draw_price is not None:
                draw_dec = american_to_decimal(int(draw_price))
                prob_a, prob_b, prob_draw, margin = devig_three_way(home_dec, away_dec, draw_dec)
            else:
                draw_dec = None
                prob_draw = None
                prob_a, prob_b, margin = devig_two_way(home_dec, away_dec)
        except Exception as e:
            logger.debug("pinnacle_guest_devig_failed", error=str(e))
            return None

        return SharpOdds(
            event_id=base_event_id,
            book=SharpBook.pinnacle,
            sector=sector,
            outcome_a_label=home,
            outcome_b_label=away,
            outcome_a_decimal=home_dec,
            outcome_b_decimal=away_dec,
            outcome_draw_decimal=draw_dec,
            true_prob_a=prob_a,
            true_prob_b=prob_b,
            true_prob_draw=prob_draw,
            margin=margin,
            event_date=event_date,
        )

    def _parse_spread(
        self, market: dict, base_event_id: str, sector: str,
        home: str, away: str, event_date: Optional[datetime],
    ) -> Optional[SharpOdds]:
        prices = market.get("prices", [])
        if len(prices) < 2:
            return None

        home_entry = next((p for p in prices if p.get("designation") == "home"), None)
        away_entry = next((p for p in prices if p.get("designation") == "away"), None)
        if not home_entry or not away_entry:
            return None

        # Pinnacle guest API uses "points" for the handicap value (not "handicap")
        home_hdp = home_entry.get("points") if "points" in home_entry else home_entry.get("handicap", 0.0)
        away_hdp = away_entry.get("points") if "points" in away_entry else away_entry.get("handicap", 0.0)

        # Covering team = negative handicap (favorite)
        if home_hdp < 0:
            covering_team, other_team = home, away
            cover_price, other_price = home_entry["price"], away_entry["price"]
            cover_point = float(home_hdp)
        elif away_hdp < 0:
            covering_team, other_team = away, home
            cover_price, other_price = away_entry["price"], home_entry["price"]
            cover_point = float(away_hdp)
        else:
            return None

        try:
            cover_dec = american_to_decimal(int(cover_price))
            other_dec = american_to_decimal(int(other_price))
            prob_cover, prob_other, margin = devig_two_way(cover_dec, other_dec)
        except Exception as e:
            logger.debug("pinnacle_guest_spread_devig_failed", error=str(e))
            return None

        return SharpOdds(
            event_id=f"{base_event_id}::spread",
            book=SharpBook.pinnacle,
            sector=sector,
            outcome_a_label=covering_team,
            outcome_b_label=other_team,
            outcome_a_decimal=cover_dec,
            outcome_b_decimal=other_dec,
            true_prob_a=prob_cover,
            true_prob_b=prob_other,
            spread_line=cover_point,
            margin=margin,
            event_date=event_date,
        )

    def _parse_totals(
        self, market: dict, base_event_id: str, sector: str,
        event_date: Optional[datetime],
    ) -> Optional[SharpOdds]:
        prices = market.get("prices", [])
        over_entry  = next((p for p in prices if p.get("designation") == "over"),  None)
        under_entry = next((p for p in prices if p.get("designation") == "under"), None)
        if not over_entry or not under_entry:
            return None

        # "points" field holds the total line (e.g. 220.5)
        raw_line = over_entry.get("points") or over_entry.get("handicap")
        if raw_line is None:
            return None
        total_line = float(raw_line)

        try:
            over_dec  = american_to_decimal(int(over_entry["price"]))
            under_dec = american_to_decimal(int(under_entry["price"]))
            prob_over, prob_under, margin = devig_two_way(over_dec, under_dec)
        except Exception as e:
            logger.debug("pinnacle_guest_totals_devig_failed", error=str(e))
            return None

        return SharpOdds(
            event_id=f"{base_event_id}::total::{total_line}",
            book=SharpBook.pinnacle,
            sector=sector,
            outcome_a_label="over",
            outcome_b_label="under",
            outcome_a_decimal=over_dec,
            outcome_b_decimal=under_dec,
            true_prob_a=0.0,
            true_prob_b=0.0,
            true_prob_over=prob_over,
            true_prob_under=prob_under,
            total_line=total_line,
            margin=margin,
            event_date=event_date,
        )

    @staticmethod
    def _normalize(name: str, sector: str) -> str:
        if sector == "baseball":
            # Doubleheader nightcaps carry a "G2 " participant-name prefix
            # from Pinnacle (e.g. "G2 New York Yankees") with no matching
            # alias entry, so the pair silently never matches (produces
            # "g2_new_york_yankees" instead of "yankees"). Baseball-only:
            # "G2" is a real esports org name (G2 Esports, LoL/CS2) that
            # must never be stripped there.
            name = _DOUBLEHEADER_PREFIX_RE.sub("", name)
        from evmax.matching.normalizer import NameNormalizer
        normalized = NameNormalizer(sector).normalize(name)
        return normalized.replace(" ", "_").replace(".", "")
