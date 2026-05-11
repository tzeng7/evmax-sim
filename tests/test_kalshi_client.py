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


class TestExtractTeamsFromTicker:
    """_extract_teams_from_ticker — team-pair parsing across series formats.

    Return order is (home, away) — see the caller at line 703 of kalshi.py
    which unpacks as `team_home, team_away`.
    """

    def test_nba_ticker(self) -> None:
        # KXNBAGAME-26FEB24ORLLAL-ORL → home=lal, away=orl
        home, away = _client()._extract_teams_from_ticker("KXNBAGAME-26FEB24ORLLAL-ORL")
        assert (home, away) == ("lal", "orl")

    def test_wnba_mixed_length(self) -> None:
        # CONN/NY mixed-length codes — outcome=NY anchors home
        home, away = _client()._extract_teams_from_ticker("KXWNBAGAME-26MAY08CONNNY-NY")
        assert (home, away) == ("ny", "conn")

    def test_mlb_home_side_strips_time_prefix(self) -> None:
        # KXMLBGAME-26MAY121907TBTOR-TOR: HHMM=1907 between date and TBTOR.
        # Regression: previously returned away="1907tb"; expected away="tb".
        home, away = _client()._extract_teams_from_ticker("KXMLBGAME-26MAY121907TBTOR-TOR")
        assert (home, away) == ("tor", "tb")

    def test_mlb_away_side_strips_time_prefix(self) -> None:
        home, away = _client()._extract_teams_from_ticker("KXMLBGAME-26MAY121907TBTOR-TB")
        assert (home, away) == ("tor", "tb")

    def test_mlb_spread_ticker(self) -> None:
        # Spread outcome has trailing digits (the line) — parser strips them
        # before anchoring on the team code.
        home, away = _client()._extract_teams_from_ticker("KXMLBSPREAD-26MAY111835NYYBAL-NYY4")
        assert (home, away) == ("bal", "nyy")

    def test_mlb_total_ticker_no_team_outcome(self) -> None:
        # Totals encode the line as the outcome (e.g. "-9"); team-pair parser
        # cannot anchor and returns (None, None) so the title fallback kicks in.
        home, away = _client()._extract_teams_from_ticker("KXMLBTOTAL-26MAY111835NYYBAL-9")
        assert (home, away) == (None, None)
