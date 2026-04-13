"""Tests for evmax.clients.kalshi.

The KalshiClient constructor needs RSA settings, so we exercise instance methods
through `__new__` to avoid touching the network or filesystem.
"""

from __future__ import annotations

from datetime import timezone, timedelta

from evmax.clients.kalshi import KalshiClient


def _client() -> KalshiClient:
    return KalshiClient.__new__(KalshiClient)


class TestParseTickerDate:
    """_parse_ticker_date — game date extraction from a Kalshi ticker."""

    def test_nba_ticker(self) -> None:
        d = _client()._parse_ticker_date("KXNBAGAME-26FEB24ORLLAL-ORL")
        assert d is not None
        assert (d.year, d.month, d.day) == (2026, 2, 24)

    def test_nfl_ticker(self) -> None:
        d = _client()._parse_ticker_date("KXNFLGAME-26JAN05BUFKC-KC")
        assert d is not None
        assert (d.year, d.month, d.day) == (2026, 1, 5)

    def test_unknown_format_returns_none(self) -> None:
        assert _client()._parse_ticker_date("NOT-A-VALID-TICKER") is None

    def test_invalid_month_returns_none(self) -> None:
        # "ZZZ" is not in _MONTH_MAP — ValueError swallowed → None
        assert _client()._parse_ticker_date("KXNBAGAME-26ZZZ24ORLLAL-ORL") is None

    def test_anchored_at_noon_utc(self) -> None:
        """Regression: parsed datetime must survive .astimezone() into negative-offset
        US time zones without rolling the calendar date back to the previous day.

        Original bug: anchored at 00:00 UTC, so a ticker for "26APR14" became
        "2026-04-13 20:00 ET" after .astimezone() and the dashboard showed the
        wrong game date.
        """
        d = _client()._parse_ticker_date("KXNBAGAME-26APR14CHAMIA-MIA")
        assert d is not None
        assert d.hour == 12
        assert d.tzinfo is timezone.utc

        # Eastern (UTC-4) — date must still be April 14
        et = d.astimezone(timezone(timedelta(hours=-4)))
        assert et.strftime("%Y-%m-%d") == "2026-04-14"

        # Pacific (UTC-7) — likewise
        pt = d.astimezone(timezone(timedelta(hours=-7)))
        assert pt.strftime("%Y-%m-%d") == "2026-04-14"

        # And going the other way (UTC+10) doesn't roll forward either
        au = d.astimezone(timezone(timedelta(hours=10)))
        assert au.strftime("%Y-%m-%d") == "2026-04-14"
