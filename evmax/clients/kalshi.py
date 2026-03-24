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

from evmax.clients.base import BaseAPIClient
from evmax.models.market import MarketSource, MarketType, PredictionMarket
from evmax.settings import get_settings

logger = structlog.get_logger(__name__)

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
    "nfl": ["KXNFLGAME"],
    "nba": ["KXNBAGAME", "KXNBASPREAD"],
    "ncaab": ["KXNCAABGAME"],
    "ncaaw": ["KXNCAAWBGAME", "KXNCAAWBSPREAD", "KXNCAAWBTOTAL"],
    "nba_props": ["KXNBAPTS", "KXNBAREB", "KXNBAAST", "KXNBA3PT", "KXNBASTP", "KXNBABLK", "KXNBAPRA"],
    "nfl_props": ["KXNFLPAS", "KXNFLRSH", "KXNFLREC", "KXNFLTD"],
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
    "KXNBASTP": "steals",
    "KXNBABLK": "blocks",
    "KXNBAPRA": "points_rebounds_assists",
    "KXNFLPAS": "passing_yards",
    "KXNFLRSH": "rushing_yards",
    "KXNFLREC": "receiving_yards",
    "KXNFLTD": "touchdowns",
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
        """
        try:
            import websockets.asyncio.client as ws_client
        except ImportError:
            logger.debug("kalshi_ws_import_failed", hint="pip install 'websockets>=12.0'")
            return {t: None for t in tickers}

        results: dict[str, Optional[float]] = {}
        remaining = set(tickers)

        try:
            async with ws_client.connect(self._ws_url, open_timeout=5) as ws:
                self._ws = ws

                # Authenticate
                await self._login(ws)

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
                        ticker, price = self._parse_message(msg)
                        if ticker and ticker in remaining and price is not None:
                            results[ticker] = price
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

    def _parse_message(self, msg: dict[str, Any]) -> tuple[Optional[str], Optional[float]]:
        """Extract (ticker, yes_ask) from a WS message.

        Handles:
          - type=orderbook_snapshot  → full snapshot, parse no_dollars last entry
          - type=orderbook_delta     → incremental; ignored (wait for snapshot)
          - type=subscribed / other  → ignored
        """
        msg_type = msg.get("type", "")
        if msg_type != "orderbook_snapshot":
            return None, None

        ticker = msg.get("market_ticker") or msg.get("msg", {}).get("market_ticker")
        if not ticker:
            return None, None

        # Message body mirrors REST orderbook_fp format
        body = msg.get("msg", msg)
        ob = body.get("orderbook_fp", body.get("orderbook", {}))
        no_entries = ob.get("no_dollars", ob.get("no", []))
        if not no_entries:
            return ticker, None

        last = no_entries[-1]
        try:
            best_no_bid = float(last[0]) if isinstance(last, (list, tuple)) else float(last.get("price", 0))
            yes_ask = 1.0 - best_no_bid
            if 0 < yes_ask < 1:
                return ticker, yes_ask
        except (ValueError, TypeError, IndexError):
            pass
        return ticker, None


class KalshiClient(BaseAPIClient):
    """Client for Kalshi's trading API v2."""

    def __init__(self) -> None:
        settings = get_settings()
        super().__init__(
            base_url=settings.kalshi_base_url,
            concurrency=50,   # _KALSHI_FETCH_SEM is the real throttle; inner semaphore must not block
            timeout=20.0,
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
                padding.PKCS1v15(),
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
        series_prefixes = SECTOR_SERIES_MAP.get(sector.lower(), [])
        is_prop_sector = sector.lower().endswith("_props")

        async def _fetch_prefix(prefix: str, stagger_s: float) -> list[PredictionMarket]:
            if stagger_s:
                await asyncio.sleep(stagger_s)
            try:
                data = await self._get(
                    "/markets",
                    params={"status": status, "series_ticker": prefix, "limit": limit},
                )
                parsed = []
                for m in data.get("markets", []):
                    if is_prop_sector:
                        stat_type = _PROP_SERIES_TO_STAT.get(prefix.upper(), "unknown")
                        p = self._parse_prop_market(m, sector, stat_type)
                    else:
                        p = self._parse_market(m, sector)
                    if p:
                        parsed.append(p)
                return parsed
            except Exception as e:
                err = str(e) or repr(e) or type(e).__name__
                logger.warning("kalshi_fetch_failed", prefix=prefix, error=err)
                return []

        # Stagger launches 150ms apart — all run in parallel but avoid a simultaneous
        # burst that triggers Kalshi 429s (24 series × 0.15s = 3.6s spread, ~5s total).
        results = await asyncio.gather(*(
            _fetch_prefix(p, i * 0.15) for i, p in enumerate(series_prefixes)
        ))
        all_markets: list[PredictionMarket] = [m for batch in results for m in batch]
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

    def _parse_market(self, raw: dict[str, Any], sector: str) -> Optional[PredictionMarket]:
        """Parse a raw Kalshi market dict into a PredictionMarket.

        The API uses '_dollars' suffix fields (values already in 0.0–1.0 decimal
        form) with '_fp' suffix for volume/open_interest counts.  Falls back to
        legacy integer cent fields ('yes_bid', 'yes_ask') if present.
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

            # --- Teams: for tennis use title (ticker codes are 3-letter abbreviations);
            # for other sectors parse 3-letter codes from ticker, fall back to title ---
            if sector == "tennis":
                team_home, team_away = self._extract_teams_from_title(title)
                yes_team = self._extract_tennis_yes_player(title, sector)
            else:
                team_home, team_away = self._extract_teams_from_ticker(ticker)
                if not team_home:
                    team_home, team_away = self._extract_teams_from_title(title)
                yes_team = self._extract_yes_team(ticker, sector)

            # Detect spread series by ticker prefix and extract line
            is_spread = any(
                ticker.upper().startswith(s)
                for s in ["KXNBASPREAD", "KXNFLSPREAD"]
            )
            market_type = MarketType.spread if is_spread else self._infer_market_type(title)
            spread_line = self._extract_spread_line(ticker) if is_spread else None

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
            return datetime(year, month, day, tzinfo=timezone.utc)
        except (ValueError, KeyError):
            return None

    def _extract_teams_from_ticker(self, ticker: str) -> tuple[Optional[str], Optional[str]]:
        """
        Extract away/home team codes from the ticker's team-pair segment.

        Format: {SERIES}-{YYMONDD}{AWAY3}{HOME3}-{OUTCOME3}  (for 3-char code sports)
        Example: KXNBAGAME-26FEB24ORLLAL-ORL  →  away=ORL, home=LAL

        Returns lowercase codes for alias resolution. For non-3-char sports,
        returns (None, None) and falls back to title parsing.
        """
        date_match = _TICKER_DATE_RE.search(ticker.upper())
        if not date_match:
            return None, None

        # Everything after the date match up to the last '-'
        after_date = ticker.upper()[date_match.end():]
        # Strip the outcome suffix (after last '-')
        if "-" in after_date:
            team_pair = after_date.rsplit("-", 1)[0]
        else:
            team_pair = after_date

        # Standard game tickers use two 3-letter codes (NBA, NFL, soccer, cs2, etc.)
        if len(team_pair) == 6:
            away_code = team_pair[:3].lower()
            home_code = team_pair[3:].lower()
            return home_code, away_code  # (home, away)

        # LoL/esports may use variable-length team names — fall back to title
        return None, None

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

    def _infer_market_type(self, title: str) -> MarketType:
        """Infer market type from title."""
        title_lower = title.lower()
        if any(kw in title_lower for kw in ["over", "under", "total"]):
            return MarketType.total
        if any(kw in title_lower for kw in ["spread", "+", "-"]) and "vs" not in title_lower:
            return MarketType.spread
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
