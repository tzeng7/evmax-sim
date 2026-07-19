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
        # Totals encode the line as the outcome (e.g. "-9"); outcome anchoring
        # is impossible. For fixed-width sectors (MLB/NBA/NFL/NHL/NCAAM) we
        # fall through to the deterministic 3+3 split. WNBA bails to the
        # title fallback because its codes are variable-length (CONN, etc.).
        home, away = _client()._extract_teams_from_ticker(
            "KXMLBTOTAL-26MAY111835NYYBAL-9", sector="baseball",
        )
        assert (home, away) == ("bal", "nyy")
        home, away = _client()._extract_teams_from_ticker(
            "KXWNBATOTAL-26MAY08CONNNY-165", sector="wnba",
        )
        assert (home, away) == (None, None)


class TestSeriesTeamCodeMaps:
    """Series-scoped team-code resolution for multi-league sectors.

    Regression for the 2026-07-09 finding: soccer's alias namespace is shared
    across leagues, so the flat alias "tor" → "torino" (Serie A) silently
    mis-normalized Kalshi's MLS Toronto FC markets to the Italian club. MLS
    codes must resolve through the KXMLSGAME-scoped map; other series keep
    the generic alias path.
    """

    def _mls_raw(self, ticker: str, title: str) -> dict:
        return {
            "ticker": ticker,
            "title": title,
            "yes_bid_dollars": "0.40",
            "yes_ask_dollars": "0.45",
            "no_bid_dollars": "0.50",
            "no_ask_dollars": "0.60",
            "volume_fp": 10,
            "open_interest_fp": 5,
        }

    def test_mls_toronto_not_torino(self) -> None:
        m = _client()._parse_market(
            self._mls_raw("KXMLSGAME-26JUL16MTLTOR-TOR", "Montreal vs Toronto Winner?"),
            "soccer",
        )
        assert m is not None
        assert m.yes_team == "toronto"
        assert m.team_home == "toronto"
        assert m.team_away == "montreal"

    def test_serie_a_tor_still_torino(self) -> None:
        m = _client()._parse_market(
            self._mls_raw("KXSERIEAGAME-26AUG23TORJUV-TOR", "Torino vs Juventus Winner?"),
            "soccer",
        )
        assert m is not None
        assert m.yes_team == "torino"

    def test_mls_codes_resolve_to_canonicals(self) -> None:
        m = _client()._parse_market(
            self._mls_raw("KXMLSGAME-26JUL16STLSKC-SKC", "Saint Louis vs Kansas City Winner?"),
            "soccer",
        )
        assert m is not None
        assert m.yes_team == "sporting kc"
        assert m.team_home == "sporting kc"
        assert m.team_away == "st louis"

    def test_mls_tie_outcome_unmapped(self) -> None:
        m = _client()._parse_market(
            self._mls_raw("KXMLSGAME-26JUL16SEAPOR-TIE", "Seattle vs Portland Winner?"),
            "soccer",
        )
        assert m is not None
        assert m.yes_team == "tie"
        assert m.team_home == "portland"
        assert m.team_away == "seattle"


class TestGetMarketCandlesticks:
    """get_market_candlesticks — historical OHLC bid/ask backfill endpoint."""

    @staticmethod
    def _stub(client: KalshiClient, response: dict) -> list:
        """Replace _get with a recorder returning `response`; return the call log."""
        calls: list = []

        async def fake_get(path, params=None):
            calls.append((path, params))
            return response

        client._get = fake_get  # type: ignore[method-assign]
        return calls

    async def test_path_and_params_construction(self) -> None:
        c = _client()
        calls = self._stub(c, {"candlesticks": []})
        await c.get_market_candlesticks(
            "KXWNBASPREAD", "KXWNBASPREAD-26JUL02SEAPHX-SEA7",
            start_ts=1000, end_ts=2000, period_interval=60,
        )
        path, params = calls[0]
        assert path == "/series/KXWNBASPREAD/markets/KXWNBASPREAD-26JUL02SEAPHX-SEA7/candlesticks"
        assert params == {"start_ts": 1000, "end_ts": 2000, "period_interval": 60}

    async def test_include_latest_before_start_param(self) -> None:
        c = _client()
        calls = self._stub(c, {"candlesticks": []})
        await c.get_market_candlesticks(
            "KXWNBATOTAL", "KXWNBATOTAL-26JUL02SEAPHX-T160",
            start_ts=1, end_ts=2, include_latest_before_start=True,
        )
        _, params = calls[0]
        assert params["include_latest_before_start"] == "true"

    async def test_strips_venue_prefix_and_no_suffix(self) -> None:
        c = _client()
        calls = self._stub(c, {"candlesticks": []})
        await c.get_market_candlesticks(
            "KXWNBASPREAD", "kalshi:KXWNBASPREAD-26JUL02SEAPHX-SEA7:no",
            start_ts=1, end_ts=2,
        )
        path, _ = calls[0]
        assert path.endswith("/markets/KXWNBASPREAD-26JUL02SEAPHX-SEA7/candlesticks")

    async def test_parses_dollars_string_fields(self) -> None:
        c = _client()
        self._stub(c, {"candlesticks": [{
            "end_period_ts": 1720000000,
            "yes_bid": {"close_dollars": "0.42"},
            "yes_ask": {"close_dollars": "0.48"},
            "price": {"close_dollars": "0.45"},
            "volume_fp": "12.0",
            "open_interest_fp": "300.0",
        }]})
        out = await c.get_market_candlesticks("KXWNBASPREAD", "T", 1, 2)
        assert out == [{
            "end_ts": 1720000000,
            "yes_bid_close": 0.42,
            "yes_ask_close": 0.48,
            "price_close": 0.45,
            "volume": 12.0,
            "open_interest": 300.0,
        }]

    async def test_legacy_cents_fallback(self) -> None:
        c = _client()
        self._stub(c, {"candlesticks": [{
            "end_period_ts": 1720000000,
            "yes_bid": {"close": 42},
            "yes_ask": {"close": 48},
            "price": {"close": 45},
            "volume": 7,
        }]})
        out = await c.get_market_candlesticks("KXWNBASPREAD", "T", 1, 2)
        assert out[0]["yes_bid_close"] == 0.42
        assert out[0]["yes_ask_close"] == 0.48
        assert out[0]["price_close"] == 0.45
        assert out[0]["volume"] == 7.0

    async def test_null_price_struct_is_none(self) -> None:
        # Markets with no trades in a period return a null price struct.
        c = _client()
        self._stub(c, {"candlesticks": [{
            "end_period_ts": 1720000000,
            "yes_bid": {"close_dollars": "0.40"},
            "yes_ask": {"close_dollars": "0.50"},
            "price": None,
        }]})
        out = await c.get_market_candlesticks("KXWNBASPREAD", "T", 1, 2)
        assert out[0]["price_close"] is None
        assert out[0]["yes_ask_close"] == 0.50

    async def test_sorted_by_end_ts_and_skips_missing_ts(self) -> None:
        c = _client()
        self._stub(c, {"candlesticks": [
            {"end_period_ts": 200, "yes_ask": {"close_dollars": "0.5"}},
            {"yes_ask": {"close_dollars": "0.9"}},          # no ts — dropped
            {"end_period_ts": 100, "yes_ask": {"close_dollars": "0.4"}},
        ]})
        out = await c.get_market_candlesticks("KXWNBASPREAD", "T", 1, 2)
        assert [o["end_ts"] for o in out] == [100, 200]

    async def test_error_returns_empty_list(self) -> None:
        c = _client()

        async def boom(path, params=None):
            raise RuntimeError("api down")

        c._get = boom  # type: ignore[method-assign]
        out = await c.get_market_candlesticks("KXWNBASPREAD", "T", 1, 2)
        assert out == []
