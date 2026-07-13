"""Kalshi golf venue adapter.

Kalshi golf lives under the ``KXPGATOUR`` series family: one YES market per
golfer, ticker ``KXPGATOUR-{EVENT}-{CODE}``, with the golfer name carried
cleanly in ``yes_sub_title`` (no title parsing needed) and the market title
``"Will {Golfer} win the {Tournament}?"``. Prices are in CENTS (0-100).

Confirmed live shape (2026-07): the golf outright field lists Mon/Tue of
tournament week; deep-longshot contracts have no resting ask. Unlike PolyUS,
Kalshi golf PRICE fields require the authenticated trade API — the public
elections endpoint returns null bid/ask (verified), so the live fetch reuses
evmax's RSA-authenticated ``KalshiClient``. The parser is pure over raw market
dicts and is what the tests exercise (against a synthetic fixture), decoupled
from whether Kalshi credentials are configured.

Series → market-type mapping is kept extensible; today only the winner series
(``KXPGATOUR``) is wired, matching the PolyUS coverage.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from evmax.golf.models import FieldPrices, GolfMarketType, GolferMarket
from evmax.golf.names import canonical

_log = logging.getLogger(__name__)

_VENUE = "kalshi"

# Kalshi golf series prefix → market family. The per-golfer non-outright markets
# (make-cut / top-N) are live on Kalshi for majors and map 1:1 onto the
# simulator's outputs. KXPGACUTLINE is a cut-LINE-VALUE market (what score the cut
# lands at), NOT a per-golfer make-cut, so it is deliberately not mapped.
# Order matters: longer prefixes first so KXPGATOP10 isn't shadowed by KXPGATOP1.
SERIES_MARKET_TYPE = {
    "KXPGATOUR": GolfMarketType.win,
    "KXPGAMAKECUT": GolfMarketType.make_cut,
    "KXPGATOP10": GolfMarketType.top_10,
    "KXPGATOP20": GolfMarketType.top_20,
    "KXPGATOP5": GolfMarketType.top_5,
}

_TITLE_RE = re.compile(r"win the (.+?)\??$", re.IGNORECASE)


def _cents_to_prob(v) -> Optional[float]:
    if v is None:
        return None
    try:
        p = float(v) / 100.0
    except (TypeError, ValueError):
        return None
    return p if 0.0 < p < 1.0 else None


def _market_type_for(ticker: str, series_ticker: Optional[str]) -> Optional[GolfMarketType]:
    key = (series_ticker or ticker or "").upper()
    for prefix, mt in SERIES_MARKET_TYPE.items():
        if key.startswith(prefix):
            return mt
    return None


def _tournament_from_title(title: str, fallback: str) -> str:
    m = _TITLE_RE.search(title or "")
    return m.group(1).strip() if m else fallback


def parse_kalshi_markets(
    markets: list[dict], series_ticker: Optional[str] = None
) -> list[FieldPrices]:
    """Parse raw Kalshi golf market dicts into ``FieldPrices`` per (event, type).

    Groups by event ticker (the middle ``KXPGATOUR-{EVENT}-`` segment) so each
    tournament's field is one ``FieldPrices``.
    """
    by_event: dict[tuple, list[GolferMarket]] = {}
    titles: dict[tuple, str] = {}
    for m in markets:
        ticker = m.get("ticker") or ""
        mt = _market_type_for(ticker, series_ticker)
        if mt is None:
            continue
        name = m.get("yes_sub_title") or m.get("subtitle")
        if not name:
            continue
        parts = ticker.split("-")
        event = parts[1] if len(parts) >= 2 else ticker
        key = (event, mt)

        yes_ask = _cents_to_prob(m.get("yes_ask"))
        yes_bid = _cents_to_prob(m.get("yes_bid"))
        no_ask = _cents_to_prob(m.get("no_ask"))
        implied_no = no_ask if no_ask is not None else (
            1.0 - yes_bid if yes_bid is not None else None
        )
        gm = GolferMarket(
            venue=_VENUE,
            tournament=_tournament_from_title(m.get("title", ""), event),
            market_type=mt,
            golfer=canonical(name),
            implied_raw=yes_ask,
            ticker=ticker,
            yes_ask=yes_ask,
            yes_bid=yes_bid,
            implied_no_raw=implied_no,
            volume=m.get("volume"),
        )
        by_event.setdefault(key, []).append(gm)
        titles.setdefault(key, gm.tournament)

    fields: list[FieldPrices] = []
    for (event, mt), gms in by_event.items():
        fields.append(
            FieldPrices(venue=_VENUE, tournament=titles[(event, mt)], market_type=mt, markets=gms)
        )
    return fields


async def _fetch_markets_raw(series_ticker: str, event_ticker: Optional[str]) -> list[dict]:
    """Authenticated pull of raw golf market dicts via evmax's KalshiClient."""
    from evmax.clients.kalshi import KalshiClient  # lazy: avoids auth import in tests

    client = KalshiClient()
    params = f"?series_ticker={series_ticker}&limit=1000&status=open"
    if event_ticker:
        params += f"&event_ticker={event_ticker}"
    data = await client._get(f"/markets{params}")  # noqa: SLF001 (low-level reuse)
    return data.get("markets", []) if isinstance(data, dict) else []


def fetch_golf(
    series_ticker: str = "KXPGATOUR", event_ticker: Optional[str] = None
) -> list[FieldPrices]:
    """Live-fetch a Kalshi golf field (requires Kalshi RSA creds in settings)."""
    import asyncio

    raw = asyncio.run(_fetch_markets_raw(series_ticker, event_ticker))
    return parse_kalshi_markets(raw, series_ticker=series_ticker)
