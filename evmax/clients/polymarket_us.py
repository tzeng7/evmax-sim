"""Polymarket US gateway API client.

Polymarket US (QCX LLC, the CFTC-regulated exchange from the QCEX
acquisition) is a completely separate API from international Polymarket —
no CLOB, no Gamma, no blockchain identifiers. Market slugs and integer ids
are the only keys.

Docs:  https://docs.polymarket.us/  (index: /llms.txt)
Base:  https://gateway.polymarket.us  (v1 + v2)

Public market data requires NO authentication. The Ed25519 API key
(keyId + secretKey from polymarket.us/developer) is only needed for
trading, portfolio, and WebSocket subscriptions — signing is implemented
here for the read paths that will eventually need it, but every endpoint
this client calls today works unauthenticated.

Sports hierarchy: Sport → League → Event (game) → markets[] → marketSides[].
`GET /v2/leagues/{slug}/events` is the scanner workhorse: one call per
league returns every upcoming game with its full embedded markets array
(moneyline + alternate spread/total ladders) including live quotes.

Payload conventions (verified live 2026-07-08):
  - side.quote.value       — the ASK to buy that side, 0–1 decimal string
  - side.long              — True for the market's canonical (long) side
  - side.team.ordering     — "home"/"away" (authoritative; trust long/Yes
                             sides only — the No side of soccer outcome
                             markets carries a bogus ordering)
  - market.line            — spread/total line; for spreads it is the LONG
                             side's handicap (+3.5 = long team covers +3.5)
  - market.sportsMarketType — period qualifier lives here
                             ("baseball_team_full_game_spread" vs
                             "..._first_five_spread"); only full-game
                             markets are kept
  - soccer "drawable_outcome" markets are one market PER OUTCOME with
    Yes/No sides (team=None on the draw market) — same shape as Kalshi's
    per-outcome YES markets, so they map 1:1
"""

from __future__ import annotations

import asyncio
import base64
import time
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
import structlog
from aiolimiter import AsyncLimiter

from evmax.clients.base import BaseAPIClient
from evmax.disk_cache import cache_get, cache_set, cache_get_offline
from evmax.models.market import MarketSource, MarketType, PredictionMarket
from evmax.settings import get_settings

logger = structlog.get_logger(__name__)

# Module-level token bucket shared across all client instances. Documented
# limit is 20 req/s per IP (public) / per key (authed); stay under it.
_POLYMARKET_US_RATE_LIMITER = AsyncLimiter(15, 1.0)

# Map evmax sector names → Polymarket US league slugs. Only sectors with a
# live Polymarket US product are listed; everything else (worldcup, lol,
# cs2, ncaaw, props) stays Kalshi-only. Every key here MUST also exist in
# data/categories.yaml (tested in tests/test_polymarket_us_client.py).
POLYMARKET_US_LEAGUE_MAP: dict[str, list[str]] = {
    "nba": ["nba"],
    "wnba": ["wnba"],
    "nfl": ["nfl"],
    "ncaab": ["cbb"],
    "baseball": ["mlb"],
    "nhl": ["nhl"],
    "soccer": ["epl", "ucl", "mls"],
    "tennis": ["atp", "wta"],
}

# Sports whose event titles list the HOME side first ("Qairat vs Sutjeska";
# tennis lists the "home" player first per its side orderings). US team
# sports list the away side first — this is the fallback when no market
# side carries a usable team.ordering. Slugs verified against the /v2/sports
# catalog's per-league `ordering` field (all soccer + esports = "home").
_HOME_FIRST_LEAGUES = {
    "epl", "ucl", "mls", "atp", "wta",
    "fwc", "uefa", "lal", "bun", "sea",  # World Cup + extra soccer leagues
    "lol", "cs2",                         # esports
}

# sportsMarketType values we accept. Anything with a period qualifier
# (first_five, first_half, sets, quarters…) would otherwise fuzzy-match the
# same game's FULL-GAME Pinnacle line and produce phantom edges.
_FULL_GAME_MARKERS = ("full_game", "full_time", "match_winner")


def _is_full_game(sports_market_type: Optional[str]) -> bool:
    smt = (sports_market_type or "").lower()
    if not smt:
        # No qualifier at all — trust marketType alone.
        return True
    return any(marker in smt for marker in _FULL_GAME_MARKERS)


def _canon_team(sector: str, side: dict[str, Any]) -> Optional[str]:
    """Canonicalize a marketSide's team through the sector alias map.

    Produces the SAME canonical names Kalshi rows carry (Kalshi ticker codes
    go through NameNormalizer too), so outcome labels, matching keys, and
    resolution are venue-agnostic — 'Sparks ML' on both venues, never
    'Los angeles ML' on one. The team's nickname (`alias`) is appended when
    it adds information: 'Los Angeles' alone is ambiguous in the NBA, but
    'Los Angeles Lakers' resolves.
    """
    team = side.get("team") or {}
    name = (team.get("name") or side.get("description") or "").strip()
    if not name:
        return None
    from evmax.matching.normalizer import NameNormalizer
    normalizer = NameNormalizer(sector)
    plain = normalizer.normalize(name)
    alias = (team.get("alias") or "").strip()
    if alias and alias.lower() not in name.lower():
        # Try the name+nickname combination too ("Los Angeles" alone is
        # ambiguous in the NBA; "Los Angeles Lakers" resolves). Keep the
        # SHORTER result: canonical names are short nicknames, so a long
        # output means the combination missed the alias map — some teams'
        # `alias` field is marketing text, not a nickname (e.g. Qairat's
        # "Халық командасы"), and appending it just produces garbage.
        combined = normalizer.normalize(f"{name} {alias}")
        if combined and (not plain or len(combined) < len(plain)):
            return combined
    return plain or None


def _parse_quote(side: dict[str, Any]) -> Optional[float]:
    """Extract the 0–1 ask price from a marketSide's quote."""
    try:
        value = ((side.get("quote") or {}).get("value"))
        if value is None:
            return None
        price = float(value)
    except (TypeError, ValueError):
        return None
    if price <= 0.0 or price >= 1.0:
        return None
    return price


def _parse_start(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


class PolymarketUSClient(BaseAPIClient):
    """Read-only market-data client for the Polymarket US gateway API.

    Mirrors KalshiClient.get_markets(sector) so the coordinator can fan out
    to both venues with the same call shape. Every PredictionMarket it
    returns has source=MarketSource.polymarket_us and id
    "polymarket_us:{market_slug}[:{side}]".
    """

    def __init__(self) -> None:
        settings = get_settings()
        super().__init__(
            base_url=settings.polymarket_us_base_url,
            concurrency=10,
            timeout=20.0,
            max_retries=4,
            retry_max_wait=30.0,
        )
        self._key_id = settings.polymarket_api_key
        self._secret_key = settings.polymarket_us_secret_key

    # ------------------------------------------------------------------
    # Auth (trading / WebSocket only — market data is public)
    # ------------------------------------------------------------------

    def _sign_request(self, method: str, path: str) -> dict[str, str]:
        """Ed25519 signature headers per docs.polymarket.us/authentication.

        Returns {} when no key material is configured — all market-data
        endpoints work unauthenticated, so this is never fatal for scans.
        """
        if not self._key_id or not self._secret_key:
            return {}
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PrivateKey,
            )

            raw = base64.b64decode(self._secret_key)
            # Docs hand out the 32-byte seed (sometimes seed+pubkey, 64B).
            private_key = Ed25519PrivateKey.from_private_bytes(raw[:32])
            timestamp_ms = str(int(time.time() * 1000))
            message = f"{timestamp_ms}{method}{path}"
            signature = private_key.sign(message.encode("utf-8"))
            return {
                "X-PM-Access-Key": self._key_id,
                "X-PM-Timestamp": timestamp_ms,
                "X-PM-Signature": base64.b64encode(signature).decode(),
            }
        except Exception as e:
            logger.warning("polymarket_us_sign_failed", error=str(e))
            return {}

    # ------------------------------------------------------------------
    # Market fetch
    # ------------------------------------------------------------------

    async def get_markets(
        self,
        sector: str,
        status: str = "open",
        limit: int = 200,
        leagues: Optional[list[str]] = None,
    ) -> list[PredictionMarket]:
        """Fetch active full-game markets for a sector across its leagues.

        Signature mirrors KalshiClient.get_markets. `status` is accepted for
        interface parity; the league-events endpoint only returns active
        (non-closed) events, so it is not forwarded.

        ``leagues`` overrides the sector's POLYMARKET_US_LEAGUE_MAP entry —
        used by callers with broader coverage needs than the betting pipeline
        (e.g. the arb scanner's ARB_LEAGUE_MAP, which adds worldcup/lol/cs2
        without flipping them on for scans). Overridden fetches get their own
        cache namespace so they can't poison the betting pipeline's cache.
        """
        cfg = get_settings()
        cache_key = f"polymarket_us_{sector}_{status}"
        if leagues is not None:
            cache_key += "_" + "-".join(sorted(leagues))

        if cfg.offline_mode:
            raw = cache_get_offline(cache_key)
            return [PredictionMarket.model_validate(r) for r in raw]

        if cfg.cache_ttl_secs > 0:
            raw = cache_get(cache_key, cfg.cache_ttl_secs)
            if raw is not None:
                return [PredictionMarket.model_validate(r) for r in raw]

        if leagues is None:
            leagues = POLYMARKET_US_LEAGUE_MAP.get(sector.lower(), [])

        async def _fetch_league(league: str) -> list[PredictionMarket]:
            try:
                await _POLYMARKET_US_RATE_LIMITER.acquire()
                data = await self._get(
                    f"/v2/leagues/{league}/events",
                    params={"limit": limit},
                )
                parsed: list[PredictionMarket] = []
                for event in data.get("events", []) or []:
                    parsed.extend(self._parse_event(event, sector, league))
                return parsed
            except Exception as e:
                err = str(e) or repr(e) or type(e).__name__
                logger.warning(
                    "polymarket_us_fetch_failed", league=league, error=err
                )
                return []

        results = await asyncio.gather(*(_fetch_league(lg) for lg in leagues))
        all_markets: list[PredictionMarket] = [m for batch in results for m in batch]

        if cfg.cache_ttl_secs > 0 and all_markets:
            cache_set(cache_key, [m.model_dump(mode="json") for m in all_markets])

        return all_markets

    # ------------------------------------------------------------------
    # Settlement (outcome resolution)
    # ------------------------------------------------------------------

    async def get_market_settlement(self, market_slug: str) -> Optional[float]:
        """Settlement price for a market slug, or None if not yet settled.

        ``GET /v1/markets/{slug}/settlement`` 404s until the market settles.
        The price is quoted for the market's LONG side: 1 → long side won,
        0 → short side won. A fractional value is a non-binary settlement
        (tie 50-50, cancellation at last-traded prices, pre-match walkover
        per the market rules) — the caller's cue to void rather than score.
        """
        try:
            await _POLYMARKET_US_RATE_LIMITER.acquire()
            data = await self._get(f"/v1/markets/{market_slug}/settlement")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise
        settlement = data.get("settlement")
        return float(settlement) if settlement is not None else None

    async def get_market_sides(self, market_slug: str) -> list[dict[str, Any]]:
        """Raw ``marketSides`` for a market slug (works after close/settle).

        Used at resolve time to map a per-side market id
        (``polymarket_us:{slug}:{side_key}``) back to the long/short
        instrument the settlement price is quoted for.
        """
        await _POLYMARKET_US_RATE_LIMITER.acquire()
        data = await self._get(f"/v1/market/slug/{market_slug}")
        return (data.get("market") or {}).get("marketSides", []) or []

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_event(
        self, event: dict[str, Any], sector: str, league: str
    ) -> list[PredictionMarket]:
        """Parse one league event (game) into PredictionMarket objects.

        Emission rules (mirrors Kalshi's one-YES-market-per-outcome model):
          - moneyline          → one market per tradable side (YES = that team)
          - drawable_outcome   → one market per outcome market (YES = team/draw)
          - spreads            → one market, YES = long side at market.line;
                                 the short side is reachable via the existing
                                 NO-side spread derivation in EVGapAgent
          - totals             → one market, YES = Over (Kalshi convention);
                                 Under via the NO-side total derivation
        """
        markets_out: list[PredictionMarket] = []
        event_date = _parse_start(event.get("startTime") or event.get("startDate"))
        event_title = event.get("title") or event.get("description") or ""
        raw_markets = event.get("markets", []) or []
        team_home, team_away = self._extract_event_teams(
            raw_markets, event_title, league, sector
        )

        for m in raw_markets:
            try:
                if not m.get("active", True) or m.get("closed", False):
                    continue
                if not _is_full_game(m.get("sportsMarketType")):
                    continue
                market_type_raw = (m.get("marketType") or "").lower()
                if market_type_raw == "moneyline":
                    markets_out.extend(
                        self._parse_moneyline(
                            m, sector, event_date, event_title, team_home, team_away
                        )
                    )
                elif market_type_raw == "drawable_outcome":
                    parsed = self._parse_drawable_outcome(
                        m, sector, event_date, event_title, team_home, team_away
                    )
                    if parsed:
                        markets_out.append(parsed)
                elif market_type_raw == "spreads":
                    parsed = self._parse_spread(
                        m, sector, event_date, event_title, team_home, team_away
                    )
                    if parsed:
                        markets_out.append(parsed)
                elif market_type_raw == "totals":
                    parsed = self._parse_total(
                        m, sector, event_date, event_title, team_home, team_away
                    )
                    if parsed:
                        markets_out.append(parsed)
                # anything else (futures, specials) is skipped silently
            except Exception as e:
                logger.warning(
                    "polymarket_us_parse_failed",
                    slug=m.get("slug"),
                    error=str(e),
                )
        return markets_out

    def _extract_event_teams(
        self,
        raw_markets: list[dict[str, Any]],
        event_title: str,
        league: str,
        sector: str,
    ) -> tuple[Optional[str], Optional[str]]:
        """Resolve (team_home, team_away) for an event, canonicalized.

        Primary source: team.ordering — but ONLY from a market whose two
        sides are two DISTINCT teams carrying one "home" and one "away"
        (i.e. a real moneyline/spread pairing). Markets where both sides
        share one team (soccer drawable_outcome / per-team outcome markets)
        carry bogus orderings on either side and are never trusted.
        Fallback: split the event title on " vs. " using the league's
        listing convention (soccer and tennis list home first, US team
        sports away first).

        Names are canonicalized through the sector alias map so they carry
        the same values a Kalshi row would (see _canon_team).
        """
        home: Optional[str] = None
        away: Optional[str] = None
        for m in raw_markets:
            sides = m.get("marketSides", []) or []
            by_ordering: dict[str, str] = {}
            for side in sides:
                team = side.get("team") or {}
                name = team.get("name")
                ordering = (team.get("ordering") or "").lower()
                if name and ordering in ("home", "away"):
                    canon = _canon_team(sector, side)
                    if canon:
                        by_ordering.setdefault(ordering, canon)
            h, a = by_ordering.get("home"), by_ordering.get("away")
            if h and a and h != a:
                return h, a

        # Title fallback: "A vs. B" / "A vs B"
        from evmax.matching.normalizer import NameNormalizer
        normalizer = NameNormalizer(sector)
        title = event_title.replace(" vs. ", " vs ")
        if " vs " in title:
            a, _, b = title.partition(" vs ")
            a = normalizer.normalize(a.strip()) or None
            b = normalizer.normalize(b.strip()) or None
            if a and b:
                if league in _HOME_FIRST_LEAGUES:
                    return home or a, away or b
                return home or b, away or a
        return home, away

    def _base_kwargs(
        self,
        m: dict[str, Any],
        sector: str,
        event_date: Optional[datetime],
        event_title: str,
        team_home: Optional[str],
        team_away: Optional[str],
    ) -> dict[str, Any]:
        return {
            "source": MarketSource.polymarket_us,
            "sector": sector,
            "title": m.get("title") or m.get("question") or event_title,
            "ticker": m.get("slug", ""),
            "volume_usd": float(m.get("volume") or 0),
            "open_interest_usd": 0.0,
            "team_home": team_home,
            "team_away": team_away,
            "event_date": event_date,
        }

    def _parse_moneyline(
        self,
        m: dict[str, Any],
        sector: str,
        event_date: Optional[datetime],
        event_title: str,
        team_home: Optional[str],
        team_away: Optional[str],
    ) -> list[PredictionMarket]:
        """2-way moneyline → one PredictionMarket per tradable side.

        Mirrors Kalshi's separate per-team YES markets: buying side X at its
        quote is YES on X; the opposite side's quote is the NO ask.
        """
        sides = m.get("marketSides", []) or []
        if len(sides) != 2:
            return []
        out: list[PredictionMarket] = []
        slug = m.get("slug", "")
        for i, side in enumerate(sides):
            if not side.get("tradable", True):
                continue
            yes_price = _parse_quote(side)
            other = sides[1 - i]
            no_price = _parse_quote(other)
            if yes_price is None or no_price is None:
                continue  # need both sides quoted to trust the book
            team = side.get("team") or {}
            yes_team = _canon_team(sector, side)
            if not yes_team:
                continue
            side_key = team.get("abbreviation") or ("a" if side.get("long") else "b")
            out.append(
                PredictionMarket(
                    id=f"polymarket_us:{slug}:{side_key}",
                    market_type=MarketType.moneyline,
                    yes_price=yes_price,
                    no_price=no_price,
                    yes_team=yes_team,
                    line=None,
                    **self._base_kwargs(
                        m, sector, event_date, event_title, team_home, team_away
                    ),
                )
            )
        return out

    def _parse_drawable_outcome(
        self,
        m: dict[str, Any],
        sector: str,
        event_date: Optional[datetime],
        event_title: str,
        team_home: Optional[str],
        team_away: Optional[str],
    ) -> Optional[PredictionMarket]:
        """Soccer 3-way: one market per outcome with Yes/No sides.

        The draw market has team=None on its sides → yes_team="draw", which
        the EV pipeline already routes to true_prob_draw (Kalshi TIE path).
        """
        sides = m.get("marketSides", []) or []
        yes_side = next((s for s in sides if s.get("long")), None)
        no_side = next((s for s in sides if not s.get("long")), None)
        if yes_side is None or no_side is None:
            return None
        yes_price = _parse_quote(yes_side)
        no_price = _parse_quote(no_side)
        if yes_price is None or no_price is None:
            return None
        # Draw markets have team=None on their sides (descriptions are just
        # Yes/No) — only canonicalize when a real team is attached.
        has_team = bool((yes_side.get("team") or {}).get("name"))
        yes_team = (_canon_team(sector, yes_side) if has_team else None) or "draw"
        return PredictionMarket(
            id=f"polymarket_us:{m.get('slug', '')}",
            market_type=MarketType.moneyline,
            yes_price=yes_price,
            no_price=no_price,
            yes_team=yes_team,
            line=None,
            **self._base_kwargs(
                m, sector, event_date, event_title, team_home, team_away
            ),
        )

    def _parse_spread(
        self,
        m: dict[str, Any],
        sector: str,
        event_date: Optional[datetime],
        event_title: str,
        team_home: Optional[str],
        team_away: Optional[str],
    ) -> Optional[PredictionMarket]:
        """Spread market → YES = the long side at market.line.

        market.line is the LONG side's handicap (verified live: line=+3.5 →
        long side desc "+3.50"; line=-1.5 → long desc "-1.50"), which matches
        the evmax convention that PredictionMarket.line is the yes_team's
        handicap. The short side stays reachable as the NO-side derivation.
        """
        sides = m.get("marketSides", []) or []
        long_side = next((s for s in sides if s.get("long")), None)
        short_side = next((s for s in sides if not s.get("long")), None)
        if long_side is None or short_side is None:
            return None
        yes_price = _parse_quote(long_side)
        no_price = _parse_quote(short_side)
        line = m.get("line")
        if yes_price is None or no_price is None or line is None:
            return None
        yes_team = _canon_team(sector, long_side)
        if not yes_team:
            return None
        return PredictionMarket(
            id=f"polymarket_us:{m.get('slug', '')}",
            market_type=MarketType.spread,
            yes_price=yes_price,
            no_price=no_price,
            yes_team=yes_team,
            line=float(line),
            **self._base_kwargs(
                m, sector, event_date, event_title, team_home, team_away
            ),
        )

    def _parse_total(
        self,
        m: dict[str, Any],
        sector: str,
        event_date: Optional[datetime],
        event_title: str,
        team_home: Optional[str],
        team_away: Optional[str],
    ) -> Optional[PredictionMarket]:
        """Totals market → YES = Over (Kalshi convention), Under via NO side."""
        sides = m.get("marketSides", []) or []
        over_side = next(
            (s for s in sides if (s.get("description") or "").lower() == "over"),
            None,
        )
        under_side = next(
            (s for s in sides if (s.get("description") or "").lower() == "under"),
            None,
        )
        if over_side is None or under_side is None:
            return None
        yes_price = _parse_quote(over_side)
        no_price = _parse_quote(under_side)
        line = m.get("line")
        if yes_price is None or no_price is None or line is None:
            return None
        return PredictionMarket(
            id=f"polymarket_us:{m.get('slug', '')}",
            market_type=MarketType.total,
            yes_price=yes_price,
            no_price=no_price,
            yes_team="over",
            line=float(line),
            **self._base_kwargs(
                m, sector, event_date, event_title, team_home, team_away
            ),
        )
