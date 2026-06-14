"""Kalshi REST + WebSocket API v2 client.

Authenticates via RSA key pair (PKCS8 private key).
Fetches active markets for each sector using series tickers.

REST:  https://trading-api.readme.io/reference/getmarkets
WS:    wss://api.elections.kalshi.com/trade-api/ws/v2
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import httpx
import structlog
from aiolimiter import AsyncLimiter

from evmax.clients.base import BaseAPIClient
from evmax.disk_cache import cache_get, cache_set, cache_get_offline
from evmax.models.market import MarketSource, MarketType, PredictionMarket
from evmax.settings import get_settings

logger = structlog.get_logger(__name__)

# Module-level token bucket: 8 requests/second across all KalshiClient instances.
# Kalshi's documented limit is 10/s; we stay slightly under to absorb bursts.
# Previous value (3/s) was overly conservative and caused 13s+ scan times.
_KALSHI_RATE_LIMITER = AsyncLimiter(10, 1.0)

# YYMONDD date format used in Kalshi game tickers (e.g. "26FEB24" → 2026-02-24)
_MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}
# Ticker date pattern: 2-digit year + 3-letter month + 2-digit day
_TICKER_DATE_RE = re.compile(r"(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2})", re.IGNORECASE)

# Map sector names → Kalshi series tickers for per-game/match markets
# Verified against live Kalshi series API (2026-02-23)
SECTOR_SERIES_MAP: dict[str, list[str]] = {
    "nfl": ["KXNFLGAME", "KXNFLTOTAL"],
    "nba": ["KXNBAGAME", "KXNBASPREAD", "KXNBATOTAL"],
    "ncaab": ["KXNCAABGAME", "KXNCAAMBGAME", "KXNCAAMBSPREAD", "KXNCAAMBTOTAL"],
    "ncaaw": ["KXNCAAWBGAME", "KXNCAAWBSPREAD", "KXNCAAWBTOTAL"],
    "baseball": ["KXMLBGAME", "KXMLBSPREAD", "KXMLBTOTAL"],
    "nhl": ["KXNHLGAME", "KXNHLSPREAD", "KXNHLTOTAL"],
    "wnba": ["KXWNBAGAME", "KXWNBASPREAD", "KXWNBATOTAL"],
    "nba_props": ["KXNBAPTS", "KXNBAREB", "KXNBAAST", "KXNBA3PT", "KXNBASTL", "KXNBABLK", "KXNBAPRA"],
    # NFL prop series verified live on Kalshi 2026-04 during PR #6 investigation.
    # The shorter KXNFLPAS / KXNFLRSH / KXNFLREC / KXNFLTD names that shipped
    # originally are not real tickers and returned zero markets on every scan.
    "nfl_props": [
        "KXNFLPASSYDS",   # passing yards
        "KXNFLRSHYDS",    # rushing yards
        "KXNFLRECYDS",    # receiving yards
        "KXNFLANYTD",     # anytime touchdown
        "KXNFLPASSTDS",   # passing TDs (1.5+, 2.5+)
        "KXNFLREC",       # receptions
    ],
    "soccer": [
        "KXEPLGAME",        # English Premier League
        "KXUCLGAME",        # UEFA Champions League
        "KXMLSGAME",        # Major League Soccer
        "KXLALIGAGAME",     # La Liga
        "KXBUNDESLIGAGAME", # Bundesliga
        "KXSERIEAGAME",     # Serie A
        "KXLIGUE1GAME",     # Ligue 1
        "KXUELGAME",        # UEFA Europa League
    ],
    "lol": ["KXLOLGAME"],
    "cs2": ["KXCS2GAME", "KXCS2GAMES"],
    "tennis": ["KXATPMATCH", "KXWTAMATCH"],
}


# Prop series prefix → canonical stat type
_PROP_SERIES_TO_STAT: dict[str, str] = {
    "KXNBAPTS": "points",
    "KXNBAREB": "rebounds",
    "KXNBAAST": "assists",
    "KXNBA3PT": "threes",
    "KXNBASTL": "steals",
    "KXNBABLK": "blocks",
    "KXNBAPRA": "points_rebounds_assists",
    "KXNFLPASSYDS": "passing_yards",
    "KXNFLRSHYDS": "rushing_yards",
    "KXNFLRECYDS": "receiving_yards",
    "KXNFLANYTD": "anytime_td",
    "KXNFLPASSTDS": "passing_tds",
    "KXNFLREC": "receptions",
}

# Prop market title patterns
# e.g. "Will LeBron James score 24+ points?" → player=LeBron James, threshold=24.0, stat=points
# e.g. "LeBron James - Points O/U 24.5" → player=LeBron James, threshold=24.5, stat=points
_PROP_TITLE_RE = re.compile(
    r"(?:Will\s+)?(.+?)\s+(?:score\s+|record\s+|make\s+|have\s+)?(\d+(?:\.\d+)?)\+?\s*"
    r"(?:or\s+more\s+)?(?:O/U\s+|over/under\s+)?",
    re.IGNORECASE,
)
_PROP_OU_RE = re.compile(
    r"^(.+?)\s*[-–]\s*(.+?)\s+[Oo]/[Uu]\s+(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


class KalshiWSClient:
    """Short-lived Kalshi WebSocket client for real-time orderbook snapshots.

    Opens one connection, authenticates, subscribes to orderbook_delta for all
    requested tickers, collects one full snapshot per ticker, then closes.

    Usage:
        async with KalshiWSClient(sign_fn, key_id, ws_url, timeout) as ws:
            prices = await ws.fetch_asks(["KXNBAGAME-...", ...])
            # prices: dict[ticker, yes_ask_float | None]
    """

    def __init__(
        self,
        sign_fn: Callable[[str, str], dict[str, str]],
        key_id: str,
        ws_url: str,
        snapshot_timeout: float = 5.0,
    ) -> None:
        self._sign_fn = sign_fn
        self._key_id = key_id
        self._ws_url = ws_url
        self._timeout = snapshot_timeout
        self._ws: Any = None  # websockets connection

    async def __aenter__(self) -> "KalshiWSClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def fetch_asks(self, tickers: list[str]) -> dict[str, Optional[float]]:
        """Subscribe to orderbook_delta, collect one snapshot per ticker.

        Returns dict mapping ticker → YES ask (0.0–1.0), or None if unavailable.
        Tickers that don't receive a snapshot within timeout fall through to REST.

        Note: this is a YES-only projection of `fetch_quotes`. If you need both
        yes_ask and no_ask (e.g. to preserve bid/ask spread), call
        `fetch_quotes` directly so the no-side ladder isn't discarded.
        """
        quotes = await self.fetch_quotes(tickers)
        return {t: q[0] for t, q in quotes.items()}

    async def fetch_quotes(
        self, tickers: list[str],
    ) -> dict[str, tuple[Optional[float], Optional[float]]]:
        """Subscribe to orderbook_delta, collect (yes_ask, no_ask) per ticker.

        Returns dict[ticker, (yes_ask, no_ask)]. Either side may be None if
        that side of the ladder is empty in the snapshot. Tickers that don't
        receive a snapshot within timeout fall through to REST.

        Both prices come from the SAME orderbook snapshot, so the resulting
        bid/ask spread (yes_ask + no_ask − 1.0 = vig) reflects the live book.
        """
        try:
            import websockets.asyncio.client as ws_client
        except ImportError:
            logger.debug("kalshi_ws_import_failed", hint="pip install 'websockets>=12.0'")
            return {t: (None, None) for t in tickers}

        results: dict[str, tuple[Optional[float], Optional[float]]] = {}
        remaining = set(tickers)

        try:
            # Authenticate via HTTP headers on the WebSocket upgrade request.
            # Kalshi requires KALSHI-ACCESS-KEY, KALSHI-ACCESS-SIGNATURE, and
            # KALSHI-ACCESS-TIMESTAMP as headers. The signature signs:
            #   {timestamp_ms}GET/trade-api/ws/v2
            auth_headers = self._sign_fn("GET", "/trade-api/ws/v2")
            extra_headers = {}
            if auth_headers:
                extra_headers = {
                    "KALSHI-ACCESS-KEY": auth_headers.get("KALSHI-ACCESS-KEY", ""),
                    "KALSHI-ACCESS-SIGNATURE": auth_headers.get("KALSHI-ACCESS-SIGNATURE", ""),
                    "KALSHI-ACCESS-TIMESTAMP": auth_headers.get("KALSHI-ACCESS-TIMESTAMP", ""),
                }

            async with ws_client.connect(
                self._ws_url,
                open_timeout=5,
                additional_headers=extra_headers if extra_headers else None,
            ) as ws:
                self._ws = ws

                # Auth is handled via HTTP headers on the upgrade request above.
                # No in-band login command needed for Kalshi WS v2.

                # Subscribe to orderbook for all tickers at once
                await ws.send(json.dumps({
                    "id": 1,
                    "cmd": "subscribe",
                    "params": {
                        "channels": ["orderbook_delta"],
                        "market_tickers": list(tickers),
                    },
                }))

                # Collect snapshots until all received or timeout
                deadline = asyncio.get_event_loop().time() + self._timeout
                while remaining:
                    wait = deadline - asyncio.get_event_loop().time()
                    if wait <= 0:
                        break
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=wait)
                        msg = json.loads(raw)
                        ticker, yes_ask, no_ask = self._parse_message(msg)
                        if ticker and ticker in remaining and (yes_ask is not None or no_ask is not None):
                            results[ticker] = (yes_ask, no_ask)
                            remaining.discard(ticker)
                    except asyncio.TimeoutError:
                        break
                    except Exception as e:
                        logger.debug("kalshi_ws_recv_error", error=str(e))
                        break

        except Exception as e:
            logger.debug("kalshi_ws_connect_error", error=str(e))

        # Mark remaining tickers as None (will fall back to REST)
        for t in remaining:
            results.setdefault(t, None)
        return results

    async def _login(self, ws: Any) -> None:
        """Send Kalshi WS login message and wait for acknowledgement."""
        headers = self._sign_fn("GET", "/trade-api/ws/v2")
        timestamp = headers.get("KALSHI-ACCESS-TIMESTAMP", str(int(time.time() * 1000)))
        signature = headers.get("KALSHI-ACCESS-SIGNATURE", "")

        await ws.send(json.dumps({
            "id": 0,
            "cmd": "login",
            "params": {
                "key_id": self._key_id,
                "signature": signature,
                "timestamp": timestamp,
            },
        }))

        # Read messages until we see login ack (or a non-login message, which
        # means auth was skipped — some Kalshi envs allow unauthed orderbook reads)
        for _ in range(5):
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
                msg = json.loads(raw)
                msg_type = msg.get("type", "")
                if msg_type in ("login_acknowledged", "authenticated"):
                    return
                if msg_type == "error":
                    logger.debug("kalshi_ws_login_error", msg=msg)
                    raise RuntimeError(f"WS login failed: {msg}")
                # Any other message (e.g. subscription ack) means auth was accepted
                return
            except asyncio.TimeoutError:
                break

    def _parse_message(
        self, msg: dict[str, Any],
    ) -> tuple[Optional[str], Optional[float], Optional[float]]:
        """Extract (ticker, yes_ask, no_ask) from a WS orderbook snapshot.

        WS v2 snapshot format:
          {"type": "orderbook_snapshot", "msg": {
            "market_ticker": "KXNBA...",
            "yes_dollars_fp": [["0.01", "1000.00"], ..., ["0.33", "5.00"]],
            "no_dollars_fp":  [["0.01", "2000.00"], ..., ["0.45", "5.00"]],
          }}

        Both ladders are BID ladders (buy orders), sorted ascending by price.
          - YES ask = 1 − best_no_bid  (highest entry in no_dollars_fp)
          - NO ask  = 1 − best_yes_bid (highest entry in yes_dollars_fp)

        Pulling both from the SAME snapshot preserves the bid/ask spread
        (yes_ask + no_ask − 1.0 = vig). Returning only yes_ask and forcing
        no_ask = 1 − yes_ask destroys this signal — the caller can no
        longer distinguish a real liquid book from an empty/stale one.
        """
        none = (None, None, None)
        msg_type = msg.get("type", "")
        if msg_type != "orderbook_snapshot":
            return none

        body = msg.get("msg", msg)
        ticker = body.get("market_ticker") or msg.get("market_ticker")
        if not ticker:
            return none

        def _best_top(entries) -> Optional[float]:
            """Return the highest price (best bid) in a BID ladder, or None."""
            if not entries:
                return None
            try:
                return float(entries[-1][0])
            except (ValueError, TypeError, IndexError):
                return None

        best_no_bid = _best_top(body.get("no_dollars_fp", []))
        best_yes_bid = _best_top(body.get("yes_dollars_fp", []))

        yes_ask = (1.0 - best_no_bid) if best_no_bid is not None else None
        no_ask = (1.0 - best_yes_bid) if best_yes_bid is not None else None

        # Sanity-clamp to (0, 1) — anything outside is a malformed snapshot.
        if yes_ask is not None and not (0 < yes_ask < 1):
            yes_ask = None
        if no_ask is not None and not (0 < no_ask < 1):
            no_ask = None

        return ticker, yes_ask, no_ask


class KalshiClient(BaseAPIClient):
    """Client for Kalshi's trading API v2."""

    def __init__(self) -> None:
        settings = get_settings()
        super().__init__(
            base_url=settings.kalshi_base_url,
            concurrency=50,     # _KALSHI_RATE_LIMITER is the real throttle; inner semaphore must not block
            timeout=20.0,
            max_retries=4,      # up to 3 retries for 429s
            retry_max_wait=30.0,  # wait up to 30s between retries (Kalshi quota window)
        )
        self._key_id = settings.kalshi_api_key_id
        self._private_key_path = settings.kalshi_private_key_path

    def _load_private_key(self):
        """Load RSA private key for request signing."""
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric import padding
            from cryptography.hazmat.primitives import hashes

            key_path = Path(self._private_key_path)
            if not key_path.exists():
                return None
            with open(key_path, "rb") as f:
                return serialization.load_pem_private_key(f.read(), password=None)
        except Exception as e:
            logger.warning("kalshi_key_load_failed", error=str(e))
            return None

    def _sign_request(self, method: str, path: str) -> dict[str, str]:
        """Generate Kalshi RSA signature headers."""
        private_key = self._load_private_key()
        if not private_key or not self._key_id:
            return {}

        try:
            from cryptography.hazmat.primitives.asymmetric import padding
            from cryptography.hazmat.primitives import hashes

            timestamp_ms = str(int(time.time() * 1000))
            message = f"{timestamp_ms}{method}{path}"
            signature = private_key.sign(
                message.encode("utf-8"),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH,
                ),
                hashes.SHA256(),
            )
            return {
                "KALSHI-ACCESS-KEY": self._key_id,
                "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
                "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
            }
        except Exception as e:
            logger.warning("kalshi_sign_failed", error=str(e))
            return {}

    async def get_markets(
        self,
        sector: str,
        status: str = "open",
        limit: int = 200,
    ) -> list[PredictionMarket]:
        """Fetch active markets for a given sector."""
        cfg = get_settings()
        cache_key = f"kalshi_{sector}_{status}"

        # Offline mode: always use cache (stale is fine)
        if cfg.offline_mode:
            raw = cache_get_offline(cache_key)
            return [PredictionMarket.model_validate(r) for r in raw]

        # Dev cache: skip API if fresh cached data exists
        if cfg.cache_ttl_secs > 0:
            raw = cache_get(cache_key, cfg.cache_ttl_secs)
            if raw is not None:
                return [PredictionMarket.model_validate(r) for r in raw]

        series_prefixes = SECTOR_SERIES_MAP.get(sector.lower(), [])
        is_prop_sector = sector.lower().endswith("_props")

        # Tennis only: also fetch parent events to get product_metadata.competition,
        # which is the primary signal for TennisModelAgent surface detection.
        # Non-tennis sectors skip this — no behavior change, no extra latency.
        fetch_events = sector.lower() == "tennis"

        async def _fetch_prefix(prefix: str) -> list[PredictionMarket]:
            try:
                # Each _get must independently acquire a rate-limiter token;
                # do not share one acquire between markets and events calls.
                await _KALSHI_RATE_LIMITER.acquire()
                markets_task = self._get(
                    "/markets",
                    params={"status": status, "series_ticker": prefix, "limit": limit},
                )
                if fetch_events:
                    await _KALSHI_RATE_LIMITER.acquire()
                    events_task = self._get(
                        "/events",
                        params={"status": status, "series_ticker": prefix, "limit": limit},
                    )
                    data, events_data = await asyncio.gather(markets_task, events_task)
                else:
                    data = await markets_task
                    events_data = None

                # Build event_ticker → competition dict for tennis joins.
                competitions: Optional[dict[str, str]] = None
                if events_data is not None:
                    competitions = {}
                    for ev in events_data.get("events", []) or []:
                        et = ev.get("event_ticker")
                        if not et:
                            continue
                        pm = ev.get("product_metadata") or {}
                        comp = pm.get("competition")
                        if comp:
                            competitions[et] = comp

                parsed = []
                for m in data.get("markets", []):
                    if is_prop_sector:
                        stat_type = _PROP_SERIES_TO_STAT.get(prefix.upper(), "unknown")
                        p = self._parse_prop_market(m, sector, stat_type)
                    else:
                        p = self._parse_market(m, sector, competitions=competitions)
                    if p:
                        parsed.append(p)
                return parsed
            except Exception as e:
                err = str(e) or repr(e) or type(e).__name__
                logger.warning("kalshi_fetch_failed", prefix=prefix, error=err)
                return []

        results = await asyncio.gather(*(_fetch_prefix(p) for p in series_prefixes))
        all_markets: list[PredictionMarket] = [m for batch in results for m in batch]

        # WebSocket price refresh: only for small batches (e.g. verify).
        # For large scans (>50 markets), REST prices are fresh enough —
        # WS subscribe + snapshot for 200+ tickers takes 8s+ and dominates scan time.
        if cfg.kalshi_ws_enabled and all_markets and len(all_markets) <= 50:
            tickers = [m.ticker for m in all_markets if m.ticker]
            if tickers:
                try:
                    async with self._ws_client() as ws:
                        ws_quotes = await ws.fetch_quotes(tickers)
                    refreshed = 0
                    for m in all_markets:
                        quote = ws_quotes.get(m.ticker)
                        if quote is None:
                            continue
                        ws_yes_ask, ws_no_ask = quote
                        # Update yes_price ONLY if we got a real yes_ask.
                        # Same for no_price. Don't synthesize one from the
                        # other — that destroys the bid/ask spread, which
                        # downstream uses (via spread_pct) as the empty-
                        # orderbook signal.
                        if ws_yes_ask is not None and 0 < ws_yes_ask < 1:
                            m.yes_price = ws_yes_ask
                            refreshed += 1
                        if ws_no_ask is not None and 0 < ws_no_ask < 1:
                            m.no_price = ws_no_ask
                    if refreshed:
                        logger.debug("kalshi_ws_refresh", refreshed=refreshed, total=len(tickers))
                except Exception as e:
                    logger.debug("kalshi_ws_refresh_failed", error=str(e))

        # Write to dev cache if enabled
        if cfg.cache_ttl_secs > 0 and all_markets:
            cache_set(cache_key, [m.model_dump(mode="json") for m in all_markets])

        return all_markets

    def _ws_client(self) -> "KalshiWSClient":
        """Create a KalshiWSClient bound to this client's auth credentials."""
        settings = get_settings()
        return KalshiWSClient(
            sign_fn=self._sign_request,
            key_id=self._key_id,
            ws_url=settings.kalshi_ws_url,
            snapshot_timeout=settings.kalshi_ws_snapshot_timeout,
        )

    async def get_market_asks_batch(
        self,
        tickers: list[str],
    ) -> dict[str, Optional[float]]:
        """Fetch YES ask prices for multiple tickers in one WebSocket session.

        Opens a single WS connection, subscribes to all tickers at once, and
        collects orderbook snapshots. Any ticker that doesn't receive a snapshot
        within the timeout falls back to individual REST orderbook calls.

        Args:
            tickers: Ticker strings (with or without "kalshi:" prefix).
        Returns:
            Dict mapping each input ticker to its YES ask (0.0–1.0) or None.
        """
        settings = get_settings()
        # Normalise tickers (strip source prefix) while keeping original keys
        api_tickers = {t: (t.split(":", 1)[-1] if ":" in t else t) for t in tickers}

        results: dict[str, Optional[float]] = {}

        if settings.kalshi_ws_enabled:
            async with self._ws_client() as ws:
                ws_results = await ws.fetch_asks(list(api_tickers.values()))

            # Map api_ticker results back to original ticker keys
            for orig, api in api_tickers.items():
                if ws_results.get(api) is not None:
                    results[orig] = ws_results[api]

        # REST fallback for any ticker not resolved by WS
        missing = [t for t in tickers if t not in results]
        if missing:
            rest_asks = await asyncio.gather(
                *(self.get_market_ask(t) for t in missing),
                return_exceptions=True,
            )
            for t, ask in zip(missing, rest_asks):
                results[t] = ask if not isinstance(ask, Exception) else None

        return results

    async def get_market_ask(self, ticker: str) -> Optional[float]:
        """Fetch the current YES ask price for a single market.

        Fast path: WebSocket real-time orderbook snapshot.
        Fallback: REST orderbook endpoint → market snapshot.

        Returns YES ask (0.0–1.0), or None if settled / unavailable.
        """
        api_ticker = ticker.split(":", 1)[-1] if ":" in ticker else ticker
        settings = get_settings()

        # Fast path: WebSocket (sub-second real-time price)
        if settings.kalshi_ws_enabled:
            try:
                async with self._ws_client() as ws:
                    ws_results = await ws.fetch_asks([api_ticker])
                    ws_price = ws_results.get(api_ticker)
                    if ws_price is not None:
                        return ws_price
            except Exception as e:
                logger.debug("kalshi_ws_ask_failed", ticker=api_ticker, error=str(e))

        # Fallback: check settlement status + REST orderbook
        try:
            data = await self._get(f"/markets/{api_ticker}")
            market = data.get("market", {})
            if market.get("result") in ("yes", "no"):
                return None  # settled
            if market.get("yes_ask_dollars") is not None:
                snapshot_ask = float(market.get("yes_ask_dollars") or 0)
            else:
                snapshot_ask = (market.get("yes_ask") or 0) / 100.0
        except Exception as e:
            logger.warning("kalshi_get_market_ask_failed", ticker=ticker, error=str(e))
            return None

        try:
            ob_data = await self._get(f"/markets/{api_ticker}/orderbook")
            ob = ob_data.get("orderbook_fp", ob_data.get("orderbook", {}))
            no_entries = ob.get("no_dollars", ob.get("no", []))
            if no_entries:
                last = no_entries[-1]
                best_no_bid = float(last[0]) if isinstance(last, (list, tuple)) else float(last.get("price", 0))
                yes_ask = 1.0 - best_no_bid
                if 0 < yes_ask < 1:
                    return yes_ask
        except Exception as e:
            logger.debug("kalshi_orderbook_failed", ticker=api_ticker, error=str(e))

        return snapshot_ask if snapshot_ask > 0 else None

    async def get_market_price(self, ticker: str) -> Optional[float]:
        """
        Fetch the current YES price for a single market by ticker.

        Unlike get_markets(), this returns 0.0 and 1.0 for settled markets so
        the resolver can detect finalized outcomes.

        Returns:
            YES price 0.0–1.0, or None on error.
        """
        api_ticker = ticker.split(":", 1)[-1] if ":" in ticker else ticker
        try:
            data = await self._get(f"/markets/{api_ticker}")
            market = data.get("market", {})
            result = market.get("result", "")
            # Kalshi settled markets: result="yes" or result="no"
            if result == "yes":
                return 1.0
            if result == "no":
                return 0.0
            # Still open — use mid-price
            yes_bid = market.get("yes_bid", 0) / 100.0
            yes_ask = market.get("yes_ask", 0) / 100.0
            if yes_ask > 0:
                return (yes_bid + yes_ask) / 2.0
            return yes_bid if yes_bid > 0 else None
        except Exception as e:
            logger.warning("kalshi_get_market_failed", ticker=ticker, error=str(e))
            return None

    async def get_market_settlement(self, ticker: str) -> Optional[str]:
        """Classify a market's settlement for outcome resolution.

        Like :meth:`get_market_price` but returns a discrete verdict instead of
        a price, and recognizes Kalshi *scalar* settlements — when a match is
        cancelled / walkover / withdrawal before a ball is played, Kalshi
        finalizes the binary market to a fair refund price (``result="scalar"``)
        rather than resolving Yes/No. Those have no binary outcome and must be
        voided, not scored.

        Returns:
            "yes"  — settled YES (or open but ≥0.99, effectively decided)
            "no"   — settled NO  (or open but ≤0.01 with a real price)
            "void" — finalized without a Yes/No result (scalar fair-price refund)
            None   — still open with no decisive price, or fetch error
        """
        api_ticker = ticker.split(":", 1)[-1] if ":" in ticker else ticker
        try:
            data = await self._get(f"/markets/{api_ticker}")
            market = data.get("market", {})
            result = market.get("result", "")
            if result == "yes":
                return "yes"
            if result == "no":
                return "no"
            # Finalized but not Yes/No → scalar fair-price refund (cancelled
            # match / walkover / withdrawal before play). No binary outcome.
            if result == "scalar" or market.get("status") == "finalized":
                return "void"
            # Still open — use mid-price to catch effectively-decided markets.
            # Prefer the new _dollars decimals; fall back to legacy cents.
            if market.get("yes_bid_dollars") is not None:
                yes_bid = float(market.get("yes_bid_dollars") or 0)
                yes_ask = float(market.get("yes_ask_dollars") or 0)
            else:
                yes_bid = (market.get("yes_bid") or 0) / 100.0
                yes_ask = (market.get("yes_ask") or 0) / 100.0
            mid = (yes_bid + yes_ask) / 2.0 if yes_ask > 0 else yes_bid
            if mid >= 0.99:
                return "yes"
            if 0 < mid <= 0.01:
                return "no"
            return None
        except Exception as e:
            logger.warning("kalshi_get_settlement_failed", ticker=ticker, error=str(e))
            return None

    def _parse_market(
        self,
        raw: dict[str, Any],
        sector: str,
        competitions: Optional[dict[str, str]] = None,
    ) -> Optional[PredictionMarket]:
        """Parse a raw Kalshi market dict into a PredictionMarket.

        The API uses '_dollars' suffix fields (values already in 0.0–1.0 decimal
        form) with '_fp' suffix for volume/open_interest counts.  Falls back to
        legacy integer cent fields ('yes_bid', 'yes_ask') if present.

        If ``competitions`` is provided (tennis only), the market's
        ``event_ticker`` is looked up in the dict to populate
        ``PredictionMarket.competition`` (e.g. "ATP Munich").
        """
        try:
            ticker = raw.get("ticker", "")

            # New API: _dollars fields are already 0.0–1.0 decimals
            if raw.get("yes_bid_dollars") is not None:
                yes_bid = float(raw.get("yes_bid_dollars") or 0)
                yes_ask = float(raw.get("yes_ask_dollars") or 0)
                no_bid = float(raw.get("no_bid_dollars") or 0)
                no_ask = float(raw.get("no_ask_dollars") or 0)
                volume = float(raw.get("volume_fp") or raw.get("volume_24h_fp") or 0)
                open_interest = float(raw.get("open_interest_fp") or 0)
            else:
                # Legacy API: integer cents → divide by 100
                yes_bid = (raw.get("yes_bid") or 0) / 100.0
                yes_ask = (raw.get("yes_ask") or 0) / 100.0
                no_bid = (raw.get("no_bid") or 0) / 100.0
                no_ask = (raw.get("no_ask") or 0) / 100.0
                volume = float(raw.get("volume") or 0)
                open_interest = float(raw.get("open_interest") or 0)

            # Use ask price (what you pay to buy YES/NO immediately at market).
            # Mid-price would understate cost, inflating apparent EV.
            yes_price = yes_ask if yes_ask > 0 else yes_bid
            no_price = no_ask if no_ask > 0 else no_bid

            # Fallback: use last_price_dollars if mid not available
            if yes_price <= 0 and raw.get("last_price_dollars"):
                yes_price = float(raw["last_price_dollars"])
                no_price = 1.0 - yes_price

            if yes_price <= 0 or yes_price >= 1.0:
                return None

            title = raw.get("title", "")

            # --- Game date: parse from ticker (YYMONDD), not close_time ---
            # close_time is the market resolution deadline (~2wks after game)
            event_date = self._parse_ticker_date(ticker)

            # Detect spread series by ticker prefix and extract line
            is_spread = any(
                ticker.upper().startswith(s)
                for s in [
                    "KXNBASPREAD", "KXNFLSPREAD", "KXNCAAWBSPREAD",
                    "KXNCAAMBSPREAD", "KXMLBSPREAD", "KXNHLSPREAD",
                    "KXWNBASPREAD",
                ]
            )
            # Detect total series similarly. Outcome is the line itself
            # (KXWNBATOTAL-26MAY08CONNNY-165 → line=165), not a team code.
            is_total = any(
                ticker.upper().startswith(s)
                for s in [
                    "KXNBATOTAL", "KXNFLTOTAL", "KXNCAAWBTOTAL",
                    "KXNCAAMBTOTAL", "KXMLBTOTAL", "KXNHLTOTAL",
                    "KXWNBATOTAL",
                ]
            )

            # --- Teams: for tennis use title (ticker codes are 3-letter abbreviations);
            # for other sectors parse 3-letter codes from ticker, fall back to title ---
            if sector == "tennis":
                team_home, team_away = self._extract_teams_from_title(title)
                yes_team = self._extract_tennis_yes_player(title, sector)
            else:
                team_home, team_away = self._extract_teams_from_ticker(ticker, sector)
                if not team_home:
                    # Totals titles carry noise like "Game 4:" prefix or
                    # ": Total Points/Goals/Runs" suffix that confuses the
                    # plain " vs "/" at " splitter — strip first.
                    parse_title = self._strip_totals_title_noise(title) if is_total else title
                    team_home, team_away = self._extract_teams_from_title(parse_title)
                    # Kalshi titles follow "AWAY vs HOME" convention (sports
                    # standard — visitors listed first), but the title splitter
                    # returns the " vs " sides as (team_a, team_b) which it
                    # assigns to (home, away). For totals, where matching keys
                    # are home_vs_away, this swap fixes WNBA's title-only path
                    # (e.g. Kalshi "Washington vs Seattle" → away=Mystics,
                    # home=Storm, matching Pinnacle's storm_vs_mystics key).
                    if is_total and team_home and team_away:
                        team_home, team_away = team_away, team_home
                # Kalshi totals tickers encode YES = OVER (the per-line market
                # threshold lives in `floor_strike`, not in the outcome code),
                # so there's no team-side YES for totals — set "over" so the
                # EV gap agent's yes_team_norm == "under" check resolves to False.
                yes_team = "over" if is_total else self._extract_yes_team(ticker, sector)
            # Prefer Kalshi's authoritative `floor_strike` field over parsing
            # the integer suffix out of the ticker. Kalshi uses different
            # ticker conventions per series — NBA encodes ticker_int = floor(line)
            # so SAS7 → floor_strike 7.5 → line -7.5 (matches the old +0.5
            # heuristic), but WNBA encodes ticker_int = ceil(line) so GS8 →
            # floor_strike 7.5 → line -7.5 (the +0.5 heuristic produced -8.5,
            # one full point too long). Same divergence on totals (NBA 233 →
            # 233.5, WNBA 192 → 191.5). The `floor_strike` field is set by
            # Kalshi server-side and matches the market title's "wins by over
            # X.5" / "scores over X.5" wording exactly, so we treat it as
            # ground truth and only fall back to ticker parsing for
            # WebSocket update messages that don't carry market metadata.
            floor_strike = raw.get("floor_strike")
            if is_spread:
                market_type = MarketType.spread
                if floor_strike is not None:
                    spread_line = -float(floor_strike)
                else:
                    spread_line = self._extract_spread_line(ticker)
            elif is_total:
                market_type = MarketType.total
                if floor_strike is not None:
                    spread_line = float(floor_strike)
                else:
                    spread_line = self._extract_total_line(ticker)
            else:
                market_type = self._infer_market_type(title, sector)
                spread_line = None

            competition: Optional[str] = None
            if competitions:
                event_ticker = raw.get("event_ticker")
                if event_ticker:
                    competition = competitions.get(event_ticker)

            return PredictionMarket(
                id=f"kalshi:{ticker}",
                source=MarketSource.kalshi,
                sector=sector,
                market_type=market_type,
                title=title,
                ticker=ticker,
                yes_price=max(0.01, min(0.99, yes_price)),
                no_price=max(0.01, min(0.99, no_price)),
                volume_usd=float(volume),
                open_interest_usd=float(open_interest),
                team_home=team_home,
                team_away=team_away,
                line=spread_line,
                event_date=event_date,
                yes_team=yes_team,
                competition=competition,
            )
        except Exception as e:
            logger.warning("kalshi_parse_failed", error=str(e), raw=raw)
            return None

    def _parse_ticker_date(self, ticker: str) -> Optional[datetime]:
        """
        Extract game date from a Kalshi game ticker.

        Format: {SERIES}-{YYMONDD}{TEAMS}-{OUTCOME}
        Example: KXNBAGAME-26FEB24ORLLAL-ORL  →  2026-02-24
        """
        m = _TICKER_DATE_RE.search(ticker.upper())
        if not m:
            return None
        try:
            year = 2000 + int(m.group(1))
            month = _MONTH_MAP[m.group(2).upper()]
            day = int(m.group(3))
            # Anchor at noon UTC (≈ morning across all US time zones) so any
            # downstream .astimezone() can't slide the date backward into the
            # previous calendar day.
            return datetime(year, month, day, 12, 0, tzinfo=timezone.utc)
        except (ValueError, KeyError):
            return None

    def _extract_teams_from_ticker(
        self, ticker: str, sector: Optional[str] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Extract away/home team codes from the ticker's team-pair segment.

        Format: {SERIES}-{YYMONDD}{AWAY}{HOME}-{OUTCOME[digits]}
        Example: KXNBAGAME-26FEB24ORLLAL-ORL  →  away=ORL, home=LAL
        Example: KXWNBAGAME-26MAY08GSSEA-SEA  →  away=GS,  home=SEA  (mixed lengths)
        Example: KXWNBAGAME-26MAY08CONNNY-NY →  away=CONN,home=NY   (mixed lengths)

        Strategy: anchor on the OUTCOME suffix (always one of the two teams).
        Whichever end of the pair the outcome matches identifies that team's
        side; the remainder is the other team. Falls back to the legacy
        3+3 split when outcome anchoring is ambiguous.

        Totals tickers have a numeric outcome (the line), so outcome anchoring
        is impossible. For sectors with fixed-width 3-char codes (NBA/NFL/MLB/
        NHL/NCAAM) we fall through to the deterministic 3+3 split. For WNBA
        (variable-width codes like CONN) we bail to the title fallback because
        a blind 3+3 would mis-split CONNNY as CON/NNY.
        """
        date_match = _TICKER_DATE_RE.search(ticker.upper())
        if not date_match:
            return None, None

        after_date = ticker.upper()[date_match.end():]
        # MLB tickers embed a 4-digit first-pitch time (HHMM) between the
        # date and the team pair — e.g. KXMLBGAME-26MAY121907TBTOR-TOR.
        # No other series ships this prefix today; strip it so team_pair
        # is just the two-team segment.
        after_date = re.sub(r"^\d{4}(?=[A-Z])", "", after_date)
        if "-" not in after_date:
            return None, None
        team_pair, outcome = after_date.rsplit("-", 1)
        # Strip trailing digits (spread tickers append the line, e.g. "OKC7").
        outcome_code = re.sub(r"\d+$", "", outcome)

        # Total markets encode the line as a pure-number outcome (e.g. "159").
        # WNBA's variable-width codes (CONN, etc.) would mis-split under the
        # 3+3 fallback, so bail to title parsing. Other sectors use fixed
        # 3-char codes and can safely 3+3.
        if not outcome_code:
            if sector and sector.lower() == "wnba":
                return None, None
            # Fall through to the 3+3 fallback below.

        if outcome_code and len(team_pair) > len(outcome_code):
            ends_with = team_pair.endswith(outcome_code)
            starts_with = team_pair.startswith(outcome_code)
            # Prefer endswith (outcome = HOME); fall back to startswith (outcome = AWAY).
            # If both match (e.g. palindromic codes — extremely unlikely), the 3+3
            # legacy split below would tie-break, but we never observe this in real data.
            if ends_with and not starts_with:
                return outcome_code.lower(), team_pair[:-len(outcome_code)].lower()
            if starts_with and not ends_with:
                return team_pair[len(outcome_code):].lower(), outcome_code.lower()

        # Legacy fallback: deterministic 3+3 split for sports with fixed-width codes.
        if len(team_pair) == 6:
            return team_pair[3:].lower(), team_pair[:3].lower()  # (home, away)

        # LoL/esports use variable-length team names — fall back to title
        return None, None

    def _strip_totals_title_noise(self, title: str) -> str:
        """Remove totals-specific noise from a market title before team-splitting.

        Kalshi titles for totals series carry extra context that breaks the
        plain " vs "/" at " splitter in `_extract_teams_from_title`:
          - `Game 4: Oklahoma City at San Antonio: Total Points` (NBA playoffs)
          - `Colorado vs Vegas: Total Goals`                      (NHL)
          - `Tampa Bay vs New York Y Total Runs?`                 (MLB)
        Stripping the `Game N:` prefix and the trailing `: Total <Stat>` /
        ` Total <Stat>` suffix leaves a clean `TeamA vs/at TeamB` for the
        existing parser to split.
        """
        if not title:
            return title
        cleaned = re.sub(r"^Game\s+\d+\s*:\s*", "", title, flags=re.IGNORECASE)
        cleaned = re.sub(
            r"\s*:?\s*Total\s+(?:Points|Goals|Runs|Yards)\s*\??\s*$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        return cleaned.strip()

    def _extract_teams_from_title(self, title: str) -> tuple[Optional[str], Optional[str]]:
        """Fall-back: extract teams from market title text."""
        # Tennis/prediction-market style: "Will X win the A vs B : Round Of 128 match?"
        win_the = re.search(r"win(?:s)?\s+the\s+(.+?)\s*(?:\s*:\s|\s+match[\?\.]?$)", title, re.IGNORECASE)
        if win_the:
            match_part = win_the.group(1).strip()
            for sep in [" vs ", " vs. "]:
                if sep.lower() in match_part.lower():
                    idx = match_part.lower().find(sep.lower())
                    return match_part[:idx].strip(), match_part[idx + len(sep):].strip()

        # Strip trailing noise like " Winner?", " Moneyline?", etc.
        clean = re.sub(r"\s+(winner|moneyline|ml|spread|total)\??$", "", title, flags=re.IGNORECASE).strip()

        separators = [" vs ", " vs. ", " @ ", " at "]
        for sep in separators:
            if sep.lower() in clean.lower():
                idx = clean.lower().find(sep.lower())
                team_a = clean[:idx].strip().rstrip("?")
                team_b = clean[idx + len(sep):].strip().rstrip("?")
                if "@" in sep or sep.strip().lower() == "at":
                    return team_b.strip(), team_a.strip()  # home, away
                return team_a.strip(), team_b.strip()
        return None, None

    def _extract_tennis_yes_player(self, title: str, sector: str) -> Optional[str]:
        """Extract and normalize the YES player from a tennis market title.

        Title format: "Will [Full Name] win the [A] vs [B] : Round..."
        Returns normalized last name of the YES player.
        """
        m = re.match(r"will\s+(.+?)\s+win\s+the\s+", title, re.IGNORECASE)
        if not m:
            return None
        full_name = m.group(1).strip()
        from evmax.matching.normalizer import NameNormalizer
        return NameNormalizer(sector).normalize(full_name)

    def _extract_yes_team(self, ticker: str, sector: str) -> Optional[str]:
        """
        Extract and normalize the team that the YES side represents.

        For game tickers like KXNBAGAME-26FEB23SASDET-DET:
          - The outcome code is the last segment after '-' → 'DET'
          - Normalize 'det' via sector alias → 'pistons'

        For spread tickers like KXNBASPREAD-26MAR09DENOKC-OKC7:
          - The outcome code is 'OKC7' → strip trailing digits → 'OKC'
          - Normalize 'okc' → 'thunder'
        """
        parts = ticker.rsplit("-", 1)
        if len(parts) < 2 or not parts[-1]:
            return None
        outcome_code = re.sub(r"\d+$", "", parts[-1]).lower()
        from evmax.matching.normalizer import NameNormalizer
        return NameNormalizer(sector).normalize(outcome_code)

    def _extract_spread_line(self, ticker: str) -> Optional[float]:
        """
        Extract the spread line from a spread ticker outcome segment.

        KXNBASPREAD-26MAR09DENOKC-OKC7 → outcome='OKC7' → digits=7 → line=-7.5
        The YES side covers -N.5 (wins by more than N.5 points).
        Returns the covering team's line as a negative float (e.g., -7.5).
        """
        parts = ticker.rsplit("-", 1)
        if len(parts) < 2:
            return None
        digits = re.search(r"(\d+)$", parts[-1])
        if not digits:
            return None
        line_int = int(digits.group(1))
        # Sanity check: spread lines should be 0–50 pts for any real sport
        if line_int == 0 or line_int > 50:
            logger.debug("kalshi_spread_line_out_of_bounds", ticker=ticker, line_int=line_int)
            return None
        return -(line_int + 0.5)

    def _extract_total_line(self, ticker: str) -> Optional[float]:
        """Extract the total (over/under) line from a totals ticker outcome.

        KXWNBATOTAL-26MAY08CONNNY-165 → outcome="165" → line=165.5
        Total outcomes encode the over/under threshold; YES = "OVER N.5",
        which is canonically N+0.5 pts. Returns the over-line as a positive float.
        """
        parts = ticker.rsplit("-", 1)
        if len(parts) < 2:
            return None
        digits = re.search(r"^(\d+)$", parts[-1])
        if not digits:
            return None
        line_int = int(digits.group(1))
        # Totals are bigger than spreads — sanity check: 50–400 pts
        if line_int < 50 or line_int > 400:
            logger.debug("kalshi_total_line_out_of_bounds", ticker=ticker, line_int=line_int)
            return None
        return line_int + 0.5

    def _infer_market_type(self, title: str, sector: str = "") -> MarketType:
        """Infer market type from title."""
        title_lower = title.lower()
        if any(kw in title_lower for kw in ["over", "under", "total"]):
            return MarketType.total
        if any(kw in title_lower for kw in ["spread", "+", "-"]) and "vs" not in title_lower:
            return MarketType.spread
        # "round" in tennis titles means tournament round (e.g. "Round Of 16"),
        # not set/game handicap. Only flag map_handicap for esports (LoL/CS2).
        if sector.lower() not in ("tennis",):
            if any(kw in title_lower for kw in ["map", "round"]):
                return MarketType.map_handicap
        if any(kw in title_lower for kw in ["series", "championship", "advance"]):
            return MarketType.series_winner
        return MarketType.moneyline

    def _parse_prop_market(
        self,
        raw: dict[str, Any],
        sector: str,
        stat_type: str,
    ) -> Optional[PredictionMarket]:
        """Parse a raw Kalshi prop market into a PredictionMarket with player_prop type.

        Extracts player name and threshold from the market title.
        YES = player records OVER the threshold.
        """
        try:
            ticker = raw.get("ticker", "")

            if raw.get("yes_bid_dollars") is not None:
                yes_bid = float(raw.get("yes_bid_dollars") or 0)
                yes_ask = float(raw.get("yes_ask_dollars") or 0)
                no_bid = float(raw.get("no_bid_dollars") or 0)
                no_ask = float(raw.get("no_ask_dollars") or 0)
                volume = float(raw.get("volume_fp") or raw.get("volume_24h_fp") or 0)
                open_interest = float(raw.get("open_interest_fp") or 0)
            else:
                yes_bid = (raw.get("yes_bid") or 0) / 100.0
                yes_ask = (raw.get("yes_ask") or 0) / 100.0
                no_bid = (raw.get("no_bid") or 0) / 100.0
                no_ask = (raw.get("no_ask") or 0) / 100.0
                volume = float(raw.get("volume") or 0)
                open_interest = float(raw.get("open_interest") or 0)

            yes_price = yes_ask if yes_ask > 0 else yes_bid
            no_price = no_ask if no_ask > 0 else no_bid

            if yes_price <= 0 or yes_price >= 1.0:
                return None

            title = raw.get("title", "")
            event_date = self._parse_ticker_date(ticker)

            # Extract player name and threshold from title
            player_name, threshold = self._parse_prop_title(title, stat_type)
            if player_name is None:
                return None

            # Normalize player name
            base_sector = sector.replace("_props", "")
            from evmax.players import normalize_player_name
            player_name_norm = normalize_player_name(player_name, base_sector)

            return PredictionMarket(
                id=f"kalshi:{ticker}",
                source=MarketSource.kalshi,
                sector=base_sector,
                market_type=MarketType.player_prop,
                title=title,
                ticker=ticker,
                yes_price=max(0.01, min(0.99, yes_price)),
                no_price=max(0.01, min(0.99, no_price)),
                volume_usd=float(volume),
                open_interest_usd=float(open_interest),
                event_date=event_date,
                player_name=player_name_norm,
                stat_type=stat_type,
                threshold=threshold,
            )
        except Exception as e:
            logger.warning("kalshi_prop_parse_failed", error=str(e), raw=raw)
            return None

    def _parse_prop_title(
        self,
        title: str,
        stat_type: str,
    ) -> tuple[Optional[str], Optional[float]]:
        """Extract (player_name, threshold) from a prop market title.

        Handles Kalshi title formats:
          - "Derik Queen: 20+ points"         ← current live format (2026)
          - "LeBron James - Points O/U 24.5"
          - "Will LeBron James score 25+ points?"
          - "LeBron James Points Over 24.5?"
        Returns (None, None) if parsing fails.
        """
        # Primary format: "First Last: 20+ stat"  (live Kalshi format as of 2026)
        colon_m = re.match(r"^(.+?):\s*(\d+(?:\.\d+)?)\+?\s+\w", title, re.IGNORECASE)
        if colon_m:
            player_name = colon_m.group(1).strip()
            threshold = float(colon_m.group(2))
            return player_name, threshold

        # Format: "Player - Stat O/U threshold"
        m = _PROP_OU_RE.match(title)
        if m:
            player_name = m.group(1).strip().rstrip(":").strip()
            threshold = float(m.group(3))
            return player_name, threshold

        # Format: "Will Player score/record N+ stat?"
        will_m = re.match(r"will\s+(.+?)\s+(?:score|record|make|have|get|grab|dish|hit)\s+(\d+(?:\.\d+)?)", title, re.IGNORECASE)
        if will_m:
            player_name = will_m.group(1).strip().rstrip(":").strip()
            threshold = float(will_m.group(2))
            return player_name, threshold

        # Format: "LeBron James Over 24.5 Points" or "LeBron James 25+ Points"
        over_m = re.match(r"^(.+?)\s+(?:over\s+)?(\d+(?:\.\d+)?)\+?\s+\w+", title, re.IGNORECASE)
        if over_m:
            player_name = over_m.group(1).strip().rstrip(":").strip()
            if len(player_name.split()) >= 2:
                threshold = float(over_m.group(2))
                return player_name, threshold

        return None, None
