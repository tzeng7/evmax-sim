"""Kalshi REST API v2 client.

Authenticates via RSA key pair (PKCS8 private key).
Fetches active markets for each sector using series tickers.

Docs: https://trading-api.readme.io/reference/getmarkets
"""

from __future__ import annotations

import base64
import hashlib
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

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
    "ncaab": ["KXNCAAMBGAME", "KXNCAABGAME"],
    "soccer": [
        "KXEPLGAME",       # English Premier League
        "KXUCLGAME",       # UEFA Champions League
        "KXMLSGAME",       # Major League Soccer
        "KXLALIGAGAME",    # La Liga
        "KXBUNDESLIGAGAME",# Bundesliga
        "KXSERIEAGAME",    # Serie A
        "KXLIGUE1GAME",    # Ligue 1
        "KXUELGAME",       # UEFA Europa League
        "KXUEFAGAME",      # UEFA Soccer (catch-all)
    ],
    "lol": ["KXLOLGAME", "KXLOLGAMES"],
    "cs2": ["KXCS2GAME", "KXCS2GAMES", "KXCSGOGAME"],
}


class KalshiClient(BaseAPIClient):
    """Client for Kalshi's trading API v2."""

    def __init__(self) -> None:
        settings = get_settings()
        super().__init__(
            base_url=settings.kalshi_base_url,
            concurrency=3,
            timeout=15.0,
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
        all_markets: list[PredictionMarket] = []

        for prefix in series_prefixes:
            try:
                data = await self._get(
                    "/markets",
                    params={
                        "status": status,
                        "series_ticker": prefix,
                        "limit": limit,
                    },
                )
                markets = data.get("markets", [])
                for m in markets:
                    parsed = self._parse_market(m, sector)
                    if parsed:
                        all_markets.append(parsed)
            except Exception as e:
                logger.warning("kalshi_fetch_failed", prefix=prefix, error=str(e))

        return all_markets

    async def get_market_price(self, ticker: str) -> Optional[float]:
        """
        Fetch the current YES price for a single market by ticker.

        Unlike get_markets(), this returns 0.0 and 1.0 for settled markets so
        the resolver can detect finalized outcomes.

        Returns:
            YES price 0.0–1.0, or None on error.
        """
        try:
            data = await self._get(f"/markets/{ticker}")
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

            yes_price = (yes_bid + yes_ask) / 2.0 if yes_ask > 0 else yes_bid
            no_price = (no_bid + no_ask) / 2.0 if no_ask > 0 else no_bid

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

            # --- Teams: parse 3-letter codes from ticker, fall back to title ---
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
