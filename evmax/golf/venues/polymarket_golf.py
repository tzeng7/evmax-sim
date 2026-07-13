"""Polymarket US golf venue adapter.

PolyUS carries golf as ``sport=golf``, ``type=futures``: one EVENT per
(tournament, market family) — "The Open Championship Winner", "... End of Round
1 Leader" — each holding one YES/NO market per golfer. Confirmed live shape
(2026-07, ``pga-cham-2026-07-19-w``):

    market.title / titleShort  -> golfer name
    market.question            -> the market family ("... Winner")
    market.outcomes            -> ["Yes","No"]
    market.outcomePrices       -> ["<yes_bid>","<yes_ask>"]  (0-1 strings)
    market.bestBidQuote.value  -> yes_bid   market.bestAskQuote.value -> yes_ask

So YES ask = what you pay to back the golfer; NO ask ≈ 1 − yes_bid (the CLOB
complement). Illiquid longshots quote absurdly wide asks (0.47 for a no-hoper)
— those are placeholder quotes and get filtered on bid/ask spread downstream.

Parsers are pure; ``fetch_golf`` is the live wrapper (stdlib urllib; public,
unauthenticated).
"""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import Optional

from evmax.golf.models import FieldPrices, GolfMarketType, GolferMarket
from evmax.golf.names import canonical

_log = logging.getLogger(__name__)

_EVENTS_URL = "https://gateway.polymarket.us/v2/sports/golf/events?type=futures&limit={limit}"
_UA = "Mozilla/5.0 (evmax-golf research)"
_VENUE = "polymarket_us"


def _market_type_from_title(title: str) -> Optional[GolfMarketType]:
    t = (title or "").lower()
    if "round 1 leader" in t or "r1 leader" in t:
        return GolfMarketType.r1_leader
    if "round 2 leader" in t or "r2 leader" in t:
        return GolfMarketType.r2_leader
    if "round 3 leader" in t or "r3 leader" in t:
        return GolfMarketType.r3_leader
    if "make the cut" in t or "make cut" in t:
        return GolfMarketType.make_cut
    if "top 5" in t or "top-5" in t:
        return GolfMarketType.top_5
    if "top 10" in t or "top-10" in t:
        return GolfMarketType.top_10
    if "top 20" in t or "top-20" in t:
        return GolfMarketType.top_20
    if "winner" in t or "to win" in t:
        return GolfMarketType.win
    return None


def _num(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _yes_bid_ask(market: dict) -> tuple[Optional[float], Optional[float]]:
    """Return (yes_bid, yes_ask) from a PolyUS golf market."""
    bid = _num((market.get("bestBidQuote") or {}).get("value"))
    ask = _num((market.get("bestAskQuote") or {}).get("value"))
    if bid is None or ask is None:
        op = market.get("outcomePrices")
        if isinstance(op, str):
            try:
                op = json.loads(op)
            except json.JSONDecodeError:
                op = None
        if isinstance(op, (list, tuple)) and len(op) == 2:
            bid = bid if bid is not None else _num(op[0])
            ask = ask if ask is not None else _num(op[1])
    return bid, ask


def parse_golf_events(payload: dict) -> list[FieldPrices]:
    """Parse a PolyUS golf futures payload into one ``FieldPrices`` per event."""
    fields: list[FieldPrices] = []
    for ev in payload.get("events", []):
        title = ev.get("title") or ""
        mt = _market_type_from_title(title)
        if mt is None:
            continue
        tournament = title
        markets: list[GolferMarket] = []
        for m in ev.get("markets", []):
            name = m.get("titleShort") or m.get("title")
            if not name:
                continue
            bid, ask = _yes_bid_ask(m)
            implied_raw = ask if (ask is not None and 0.0 < ask < 1.0) else None
            implied_no = (1.0 - bid) if (bid is not None and 0.0 < bid < 1.0) else None
            markets.append(
                GolferMarket(
                    venue=_VENUE,
                    tournament=tournament,
                    market_type=mt,
                    golfer=canonical(name),
                    implied_raw=implied_raw,
                    ticker=m.get("slug") or str(m.get("id")),
                    yes_ask=ask,
                    yes_bid=bid,
                    implied_no_raw=implied_no,
                    volume=_num(m.get("volume")),
                )
            )
        fields.append(
            FieldPrices(venue=_VENUE, tournament=tournament, market_type=mt, markets=markets)
        )
    return fields


def fetch_golf(limit: int = 40, timeout: float = 25.0) -> list[FieldPrices]:
    """Live-fetch all current PolyUS golf futures fields."""
    url = _EVENTS_URL.format(limit=limit)
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        payload = json.load(resp)
    return parse_golf_events(payload)
