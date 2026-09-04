"""Tests for evmax.clients.kalshi.

The KalshiClient constructor needs RSA settings, so we exercise instance methods
through `__new__` to avoid touching the network or filesystem.
"""

from __future__ import annotations

from datetime import timezone, timedelta

from evmax.clients.kalshi import SECTOR_SERIES_MAP, KalshiClient


def _client() -> KalshiClient:
    return KalshiClient.__new__(KalshiClient)


class TestSectorSeriesMap:
    """SECTOR_SERIES_MAP — which Kalshi series each sector fetches."""

    def test_nfl_includes_spread_series(self) -> None:
        # NFL declares moneyline + spread + total in categories.yaml and the
        # ticker parser recognizes KXNFLSPREAD, but the fetch map originally
        # omitted it (verified 2026-02-23, offseason, when spreads weren't
        # listed). KXNFLSPREAD is live on Kalshi as of 2026-08-01 — wire it so
        # spread markets actually get scanned.
        assert SECTOR_SERIES_MAP["nfl"] == ["KXNFLGAME", "KXNFLSPREAD", "KXNFLTOTAL"]

    def test_nfl_series_cover_declared_market_types(self) -> None:
        """The fetch map must carry a series for every non-prop market type the
        NFL category declares — otherwise a declared type is silently unscanned."""
        from evmax.categories import get_category

        series = SECTOR_SERIES_MAP["nfl"]
        declared = {mt.value for mt in get_category("nfl").market_types}
        # market_type -> the series-ticker infix Kalshi uses for it.
        infix = {"moneyline": "GAME", "spread": "SPREAD", "total": "TOTAL"}
        for mt in declared:
            assert any(infix[mt] in s for s in series), f"no series for NFL {mt}"


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


class TestUclSeriesTeamCodeMap:
    """KXUCLGAME ticker codes resolve through the series-scoped map.

    Regression for 2026-09-04: 13 of 18 Champions League matchday-1 games
    never matched Pinnacle because their Kalshi codes (SLA/SBH/COM/SHA/PSV/
    FEN/SPO/SLO/VIK/FEY/RBB/FCP/ASK/AEK) were absent from soccer.yaml, and
    Real Madrid vs Inter failed on the Pinnacle side ("Internazionale" did
    not normalize to "inter"). Every code observed on the live tickers must
    map to the canonical Pinnacle's full name normalizes to.
    """

    # (ticker, expected yes_team, expected home, expected away)
    LIVE_TICKERS = [
        ("KXUCLGAME-26SEP10SLARCL-SLA", "slavia prague", "lens", "slavia prague"),
        ("KXUCLGAME-26SEP10MUNSBH-SBH", "sabah", "sabah", "man united"),
        ("KXUCLGAME-26SEP10COMRBL-COM", "como", "leipzig", "como"),
        ("KXUCLGAME-26SEP10PSVSHA-SHA", "shakhtar donetsk", "shakhtar donetsk", "psv"),
        ("KXUCLGAME-26SEP10FENROM-FEN", "fenerbahce", "roma", "fenerbahce"),
        ("KXUCLGAME-26SEP09SPOGAL-SPO", "sporting cp", "galatasaray", "sporting cp"),
        ("KXUCLGAME-26SEP09PSGSLO-SLO", "slovan bratislava", "slovan bratislava", "psg"),
        ("KXUCLGAME-26SEP09VFBVIK-VIK", "viking", "viking", "stuttgart"),
        ("KXUCLGAME-26SEP09BARFEY-FEY", "feyenoord", "feyenoord", "barcelona"),
        ("KXUCLGAME-26SEP08RMAINT-INT", "inter", "inter", "real madrid"),
        ("KXUCLGAME-26SEP08LILRBB-RBB", "betis", "betis", "lille"),
        ("KXUCLGAME-26SEP08FCPMCI-FCP", "porto", "man city", "porto"),
        ("KXUCLGAME-26SEP08AEKASK-ASK", "lask linz", "lask linz", "aek athens"),
        ("KXUCLGAME-26SEP10BMUBOG-BOG", "bodo glimt", "bodo glimt", "bayern"),
    ]

    def _raw(self, ticker: str) -> dict:
        return {
            "ticker": ticker,
            "title": "x wins",
            "yes_bid_dollars": "0.40",
            "yes_ask_dollars": "0.45",
            "no_bid_dollars": "0.50",
            "no_ask_dollars": "0.60",
            "volume_fp": 10,
            "open_interest_fp": 5,
        }

    def test_every_live_code_resolves(self) -> None:
        from evmax.clients.kalshi import _SERIES_TEAM_CODE_MAPS
        ucl = _SERIES_TEAM_CODE_MAPS["KXUCLGAME"]
        for ticker, yes, home, away in self.LIVE_TICKERS:
            m = _client()._parse_market(self._raw(ticker), "soccer")
            assert m is not None, ticker
            assert (m.yes_team, m.team_home, m.team_away) == (yes, home, away), ticker
        assert ucl["int"] == "inter" and ucl["sha"] == "shakhtar donetsk"

    def test_ucl_canonicals_match_pinnacle_names(self) -> None:
        """Pinnacle/ESPN full names must land on the same canonical as the code."""
        from evmax.clients.kalshi import _SERIES_TEAM_CODE_MAPS
        from evmax.matching.normalizer import NameNormalizer
        n = NameNormalizer("soccer")
        ucl = _SERIES_TEAM_CODE_MAPS["KXUCLGAME"]
        pairs = {
            "int": ["Internazionale", "Inter", "FC Internazionale"],
            "fey": ["Feyenoord", "Feyenoord Rotterdam"],
            "psv": ["PSV", "PSV Eindhoven"],
            "vfb": ["Stuttgart", "VfB Stuttgart"],
            "spo": ["Sporting CP", "Sporting Lisbon"],
            "rbb": ["Real Betis"],
            "fcp": ["Porto", "FC Porto"],
            "sbh": ["Sabah FK"],
            "ask": ["LASK Linz", "LASK"],
            "aek": ["AEK Athens"],
            "sha": ["Shakhtar Donetsk"],
            "sla": ["Slavia Prague"],
            "slo": ["Slovan Bratislava"],
            "vik": ["Viking", "Viking FK"],
            "com": ["Como"],
            "fen": ["Fenerbahce"],
            "bog": ["Bodo Glimt", "Bodo/Glimt"],
            "rcl": ["Lens", "RC Lens"],
        }
        for code, names in pairs.items():
            for name in names:
                assert n.normalize(name) == ucl[code], (code, name, n.normalize(name))

    def test_mls_map_unaffected(self) -> None:
        from evmax.clients.kalshi import _series_team_code_map
        assert _series_team_code_map("KXMLSGAME-26JUL16MTLTOR-TOR")["tor"] == "toronto"
        assert "tor" not in _series_team_code_map("KXUCLGAME-26SEP08RMAINT-INT")


class TestParseMarketBids:
    """_parse_market carries the REST book's best bids onto PredictionMarket so
    the maker-bid recommendation has a real resting price to improve on."""

    def _raw(self, **over) -> dict:
        base = {
            "ticker": "KXNBAGAME-26MAR15LALBOS-LAL",
            "title": "Boston vs Los Angeles Winner?",
            "yes_bid_dollars": "0.40",
            "yes_ask_dollars": "0.45",
            "no_bid_dollars": "0.53",
            "no_ask_dollars": "0.58",
            "volume_fp": 10,
            "open_interest_fp": 5,
        }
        base.update(over)
        return base

    def test_bids_populated_from_book(self) -> None:
        m = _client()._parse_market(self._raw(), "nba")
        assert m is not None
        # asks drive the prices, bids are carried separately
        assert m.yes_price == 0.45 and m.no_price == 0.58
        assert m.yes_bid == 0.40 and m.no_bid == 0.53

    def test_missing_bid_side_is_none(self) -> None:
        # Empty bid side (0) must surface as None — no fabricated rest price.
        m = _client()._parse_market(self._raw(yes_bid_dollars="0"), "nba")
        assert m is not None
        assert m.yes_bid is None
        assert m.no_bid == 0.53


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


class TestGetAllPages:
    """_get_all_pages follows Kalshi's cursor across every page.

    Regression for the 2026-08-30 finding: Kalshi's /markets endpoint is
    cursor-paginated and caps each page (~200) regardless of the `limit` sent,
    so a single GET silently truncated any high-volume series. NCAAF lists 500+
    open KXNCAAFGAME markets and the first page is dominated by early-sorting
    FCS tickers — the FBS games the scanner actually prices (e.g.
    KXNCAAFGAME-26AUG29SJSUUSC) sat on page 2 and never reached matching.
    """

    @staticmethod
    def _stub_pages(client: KalshiClient, pages: list) -> list:
        """Serve `pages` (list of (markets, cursor)) in order; record call params."""
        calls: list = []
        seq = iter(pages)

        async def fake_get(path, params=None):
            calls.append((path, params))
            markets, cursor = next(seq)
            return {"markets": markets, "cursor": cursor}

        client._get = fake_get  # type: ignore[method-assign]
        return calls

    async def test_follows_cursor_across_pages(self) -> None:
        c = _client()
        calls = self._stub_pages(c, [
            ([{"ticker": "A"}, {"ticker": "B"}], "CUR2"),
            ([{"ticker": "C"}], ""),  # empty cursor => stop
        ])
        out = await c._get_all_pages(
            "/markets", {"series_ticker": "X", "limit": 200}, "markets"
        )
        assert [m["ticker"] for m in out] == ["A", "B", "C"]
        assert calls[1][1]["cursor"] == "CUR2"  # page 2 carried page 1's cursor
        assert len(calls) == 2

    async def test_single_page_when_no_cursor(self) -> None:
        c = _client()
        calls = self._stub_pages(c, [([{"ticker": "A"}], None)])
        out = await c._get_all_pages(
            "/markets", {"series_ticker": "X", "limit": 200}, "markets"
        )
        assert [m["ticker"] for m in out] == ["A"]
        assert len(calls) == 1
        assert "cursor" not in calls[0][1]

    async def test_stops_on_empty_batch_even_with_cursor(self) -> None:
        # Kalshi can hand back a stale cursor with no markets; must not spin.
        c = _client()
        calls = self._stub_pages(c, [([], "GHOST")])
        out = await c._get_all_pages(
            "/markets", {"series_ticker": "X", "limit": 200}, "markets"
        )
        assert out == []
        assert len(calls) == 1

    async def test_respects_page_cap(self) -> None:
        from evmax.clients.kalshi import _MAX_KALSHI_PAGES

        c = _client()
        calls: list = []

        async def never_ending(path, params=None):
            calls.append(params)
            return {"markets": [{"ticker": f"T{len(calls)}"}], "cursor": "always"}

        c._get = never_ending  # type: ignore[method-assign]
        out = await c._get_all_pages(
            "/markets", {"series_ticker": "X", "limit": 200}, "markets"
        )
        assert len(calls) == _MAX_KALSHI_PAGES
        assert len(out) == _MAX_KALSHI_PAGES


class TestGetMarketsPagination:
    """get_markets surfaces markets from EVERY page, not just page 1.

    End-to-end regression for the SJSU/USC miss: the 8 Aug-29 FBS games sat on
    page 2 of KXNCAAFGAME and never reached the scanner because get_markets
    fetched only the first page.
    """

    @staticmethod
    def _raw(ticker: str, title: str) -> dict:
        return {
            "ticker": ticker, "title": title,
            "yes_bid_dollars": "0.40", "yes_ask_dollars": "0.45",
            "no_bid_dollars": "0.50", "no_ask_dollars": "0.60",
            "volume_fp": 10, "open_interest_fp": 5,
        }

    async def test_page_two_market_reaches_caller(self, monkeypatch) -> None:
        from types import SimpleNamespace
        import evmax.clients.kalshi as kalshi_mod

        cfg = SimpleNamespace(
            offline_mode=False, cache_ttl_secs=0, kalshi_ws_enabled=False
        )
        monkeypatch.setattr(kalshi_mod, "get_settings", lambda: cfg)

        c = _client()
        calls: list = []

        async def fake_get(path, params=None):
            calls.append(params)
            st = (params or {}).get("series_ticker")
            cur = (params or {}).get("cursor")
            if path == "/markets" and st == "KXNCAAFGAME":
                if cur is None:
                    return {"markets": [self._raw(
                        "KXNCAAFGAME-26AUG29ALSUFAMU-FAMU", "Florida A&M wins")],
                        "cursor": "PAGE2"}
                if cur == "PAGE2":
                    return {"markets": [self._raw(
                        "KXNCAAFGAME-26AUG29SJSUUSC-USC", "USC wins")],
                        "cursor": ""}
            return {"markets": [], "cursor": ""}

        c._get = fake_get  # type: ignore[method-assign]
        markets = await c.get_markets("ncaaf")
        tickers = [m.ticker for m in markets]

        # The page-2 FBS market (and the page-1 market) both surfaced.
        assert "KXNCAAFGAME-26AUG29SJSUUSC-USC" in tickers
        assert "KXNCAAFGAME-26AUG29ALSUFAMU-FAMU" in tickers

        # Pagination actually happened: KXNCAAFGAME was queried twice, the
        # second call carrying page 1's cursor.
        game_calls = [p for p in calls if (p or {}).get("series_ticker") == "KXNCAAFGAME"]
        assert len(game_calls) == 2
        assert game_calls[1]["cursor"] == "PAGE2"
