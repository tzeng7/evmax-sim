"""Tests for PolymarketUSClient — payload parsing and sector fetch.

Fixture payloads are trimmed copies of live gateway.polymarket.us responses
captured 2026-07-08 (WNBA moneyline, MLB spread/total ladders incl.
first-five variants, UCL drawable_outcome 3-way, ATP match winner).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from evmax.clients.polymarket_us import (
    POLYMARKET_US_LEAGUE_MAP,
    PolymarketUSClient,
    _is_full_game,
)
from evmax.models.market import MarketSource, MarketType


# ---------------------------------------------------------------------------
# Fixture payload builders
# ---------------------------------------------------------------------------

def _side(desc, quote, long, team_name=None, ordering=None, abbr=None, tradable=True):
    side = {
        "description": desc,
        "long": long,
        "quote": {"value": quote, "currency": "USD"} if quote is not None else None,
        "tradable": tradable,
    }
    if team_name:
        side["team"] = {
            "name": team_name,
            "ordering": ordering,
            "abbreviation": abbr,
        }
    return side


def _moneyline_market(slug="aec-wnba-gsv-tor-2026-07-08"):
    return {
        "slug": slug,
        "marketType": "moneyline",
        "sportsMarketType": "basketball_team_full_game_winner",
        "active": True,
        "closed": False,
        "volume": 12211,
        "title": "Golden State vs Toronto",
        "marketSides": [
            _side("Valkyries", "0.7300", True, "Golden State", "away", "gsv"),
            _side("Toronto", "0.28", False, "Toronto", "home", "tor"),
        ],
    }


def _spread_market(line=3.5, smt="baseball_team_full_game_spread",
                   slug="asc-mlb-tor-sf-2026-07-08-pos-3pt5"):
    return {
        "slug": slug,
        "marketType": "spreads",
        "sportsMarketType": smt,
        "active": True,
        "closed": False,
        "volume": 500,
        "line": line,
        "title": "Toronto Blue Jays vs San Francisco Giants",
        "marketSides": [
            _side("+3.50", "0.8700", True, "Toronto Blue Jays", "away", "tor"),
            _side("-3.50", "0.135", False, "San Francisco Giants", "home", "sf"),
        ],
    }


def _total_market(line=8.5, smt="baseball_team_full_game_total",
                  slug="tsc-mlb-tor-sf-2026-07-08-8pt5"):
    return {
        "slug": slug,
        "marketType": "totals",
        "sportsMarketType": smt,
        "active": True,
        "closed": False,
        "volume": 300,
        "line": line,
        "title": "Toronto Blue Jays vs San Francisco Giants",
        "marketSides": [
            _side("Over", "0.3400", True),
            _side("Under", "0.67", False),
        ],
    }


def _drawable_markets():
    """UCL 3-way: one Yes/No market per outcome, team=None on the draw."""
    return [
        {
            "slug": "dwo-ucl-kai-sut-2026-07-08-home",
            "marketType": "drawable_outcome",
            "sportsMarketType": "soccer_team_full_time_winner",
            "active": True, "closed": False, "volume": 100,
            "marketSides": [
                _side("Yes", "0.8000", True, "Qairat FK", "home"),
                # the No side carries a bogus ordering — must be ignored
                _side("No", "0.22", False, "Qairat FK", "away"),
            ],
        },
        {
            "slug": "dwo-ucl-kai-sut-2026-07-08-draw",
            "marketType": "drawable_outcome",
            "sportsMarketType": "soccer_team_full_time_winner",
            "active": True, "closed": False, "volume": 50,
            "marketSides": [
                _side("Yes", "0.1500", True),
                _side("No", "0.86", False),
            ],
        },
        {
            "slug": "dwo-ucl-kai-sut-2026-07-08-away",
            "marketType": "drawable_outcome",
            "sportsMarketType": "soccer_team_full_time_winner",
            "active": True, "closed": False, "volume": 40,
            "marketSides": [
                _side("Yes", "0.0800", True, "FK Sutjeska Nikšić", "home"),
                _side("No", "0.93", False, "FK Sutjeska Nikšić", "away"),
            ],
        },
    ]


def _event(markets, slug="wnba-gsv-tor-2026-07-08",
           title="Golden State vs. Toronto",
           start="2026-07-08T23:00:00Z"):
    return {
        "slug": slug,
        "ticker": slug,
        "title": title,
        "startDate": start,
        "startTime": start,
        "markets": markets,
    }


def _client() -> PolymarketUSClient:
    return PolymarketUSClient()


# ---------------------------------------------------------------------------
# _is_full_game filter
# ---------------------------------------------------------------------------

class TestFullGameFilter:

    def test_full_game_variants_accepted(self):
        assert _is_full_game("baseball_team_full_game_spread")
        assert _is_full_game("soccer_team_full_time_winner")
        assert _is_full_game("tennis_match_winner")
        assert _is_full_game(None)
        assert _is_full_game("")

    def test_period_markets_rejected(self):
        assert not _is_full_game("baseball_team_first_five_spread")
        assert not _is_full_game("baseball_team_first_five_total")
        assert not _is_full_game("basketball_team_first_half_winner")


# ---------------------------------------------------------------------------
# Event parsing
# ---------------------------------------------------------------------------

class TestParseMoneyline:

    def test_emits_one_market_per_side(self):
        markets = _client()._parse_event(_event([_moneyline_market()]), "wnba", "wnba")
        assert len(markets) == 2
        # yes_team is canonicalized through the sector alias map — the same
        # names a Kalshi WNBA row carries (Kalshi ticker code GSV → valkyries).
        by_team = {m.yes_team: m for m in markets}
        assert set(by_team) == {"valkyries", "tempo"}

        gsv = by_team["valkyries"]
        assert gsv.source == MarketSource.polymarket_us
        assert gsv.market_type == MarketType.moneyline
        assert gsv.yes_price == pytest.approx(0.73)
        assert gsv.no_price == pytest.approx(0.28)
        assert gsv.id == "polymarket_us:aec-wnba-gsv-tor-2026-07-08:gsv"
        assert gsv.ticker == "aec-wnba-gsv-tor-2026-07-08"

        tor = by_team["tempo"]
        assert tor.yes_price == pytest.approx(0.28)
        assert tor.no_price == pytest.approx(0.73)

    def test_home_away_from_side_ordering(self):
        markets = _client()._parse_event(_event([_moneyline_market()]), "wnba", "wnba")
        for m in markets:
            assert m.team_home == "tempo"
            assert m.team_away == "valkyries"

    def test_yes_team_matches_kalshi_convention(self):
        """The same team must canonicalize identically from a Kalshi ticker
        code and a Polymarket US team object (venue-agnostic naming)."""
        from evmax.matching.normalizer import NameNormalizer
        kalshi_name = NameNormalizer("wnba").normalize("gsv")  # Kalshi ticker code
        markets = _client()._parse_event(_event([_moneyline_market()]), "wnba", "wnba")
        pm_names = {m.yes_team for m in markets}
        assert kalshi_name in pm_names

    def test_event_date_parsed_utc(self):
        markets = _client()._parse_event(_event([_moneyline_market()]), "wnba", "wnba")
        assert markets[0].event_date is not None
        assert markets[0].event_date.isoformat() == "2026-07-08T23:00:00+00:00"

    def test_untradable_side_skipped(self):
        raw = _moneyline_market()
        raw["marketSides"][1]["tradable"] = False
        markets = _client()._parse_event(_event([raw]), "wnba", "wnba")
        assert len(markets) == 1
        assert markets[0].yes_team == "valkyries"

    def test_missing_quote_drops_both_sides(self):
        raw = _moneyline_market()
        raw["marketSides"][1]["quote"] = None
        markets = _client()._parse_event(_event([raw]), "wnba", "wnba")
        # one-sided book — neither side has a trustable (yes, no) pair
        assert markets == []

    def test_closed_market_skipped(self):
        raw = _moneyline_market()
        raw["closed"] = True
        assert _client()._parse_event(_event([raw]), "wnba", "wnba") == []


class TestParseSpread:

    def test_yes_is_long_side_with_line(self):
        markets = _client()._parse_event(
            _event([_spread_market()], slug="mlb-tor-sf-2026-07-08"),
            "baseball", "mlb",
        )
        assert len(markets) == 1
        m = markets[0]
        assert m.market_type == MarketType.spread
        assert m.yes_team == "blue jays"  # canonical, same as Kalshi rows
        assert m.line == pytest.approx(3.5)   # long side's handicap
        assert m.yes_price == pytest.approx(0.87)
        assert m.no_price == pytest.approx(0.135)

    def test_negative_line_passthrough(self):
        raw = _spread_market(line=-1.5, slug="asc-mlb-tor-sf-2026-07-08-neg-1pt5")
        raw["marketSides"][0]["description"] = "-1.50"
        raw["marketSides"][1]["description"] = "+1.50"
        markets = _client()._parse_event(_event([raw]), "baseball", "mlb")
        assert markets[0].line == pytest.approx(-1.5)

    def test_first_five_spread_filtered_out(self):
        raw = _spread_market(
            line=2.5, smt="baseball_team_first_five_spread",
            slug="asc-mlb-tor-sf-2026-07-08-f5-pos-2pt5",
        )
        assert _client()._parse_event(_event([raw]), "baseball", "mlb") == []


class TestParseTotal:

    def test_yes_is_over(self):
        markets = _client()._parse_event(_event([_total_market()]), "baseball", "mlb")
        assert len(markets) == 1
        m = markets[0]
        assert m.market_type == MarketType.total
        assert m.yes_team == "over"
        assert m.line == pytest.approx(8.5)
        assert m.yes_price == pytest.approx(0.34)
        assert m.no_price == pytest.approx(0.67)

    def test_first_five_total_filtered_out(self):
        raw = _total_market(line=3.5, smt="baseball_team_first_five_total")
        assert _client()._parse_event(_event([raw]), "baseball", "mlb") == []


class TestParseDrawableOutcome:

    def test_three_way_emits_three_markets(self):
        markets = _client()._parse_event(
            _event(
                _drawable_markets(),
                slug="ucl-kai-sut-2026-07-08",
                title="Qairat FK vs. FK Sutjeska Niksic",
            ),
            "soccer", "ucl",
        )
        assert len(markets) == 3
        by_team = {m.yes_team: m for m in markets}
        # canonicalized: "FK"-style noise words stripped by the soccer handler
        assert set(by_team) == {"qairat", "draw", "sutjeska nikšić"}
        assert by_team["qairat"].yes_price == pytest.approx(0.80)
        assert by_team["qairat"].no_price == pytest.approx(0.22)
        assert by_team["draw"].yes_price == pytest.approx(0.15)
        assert all(m.market_type == MarketType.moneyline for m in markets)

    def test_home_away_ignores_bogus_no_side_ordering(self):
        """Both teams claim ordering=home on their Yes side in this fixture's
        second team market — the extractor must only trust long sides and
        fall back to title order (soccer = home listed first)."""
        markets = _client()._parse_event(
            _event(
                _drawable_markets(),
                slug="ucl-kai-sut-2026-07-08",
                title="Qairat FK vs. FK Sutjeska Niksic",
            ),
            "soccer", "ucl",
        )
        m = markets[0]
        # Qairat's Yes side says home; Sutjeska's Yes side ALSO says home
        # (live API quirk) — orderings from same-team markets are never
        # trusted, so the title fallback resolves (soccer lists home first),
        # canonicalized through the alias map.
        assert m.team_home == "qairat"


# ---------------------------------------------------------------------------
# get_markets (mocked HTTP)
# ---------------------------------------------------------------------------

class TestGetMarkets:

    @pytest.mark.asyncio
    async def test_fetches_all_leagues_for_sector(self):
        client = _client()
        payload = {"events": [_event([_moneyline_market()])]}
        with patch.object(client, "_get", new=AsyncMock(return_value=payload)) as mget, \
             patch("evmax.clients.polymarket_us.cache_get", return_value=None), \
             patch("evmax.clients.polymarket_us.cache_set"):
            markets = await client.get_markets("soccer")
        from evmax.clients.polymarket_us import POLYMARKET_US_LEAGUE_MAP
        slugs = POLYMARKET_US_LEAGUE_MAP["soccer"]
        assert mget.await_count == len(slugs)
        called_paths = {c.args[0] for c in mget.await_args_list}
        assert called_paths == {f"/v2/leagues/{sl}/events" for sl in slugs}
        # epl/ucl/mls plus the 2026-09-05 league-shadowed additions
        assert {"epl", "ucl", "mls", "lmx", "bra", "eflch"} <= set(slugs)
        assert len(markets) == 2 * len(slugs)  # 2 markets per event per league

    @pytest.mark.asyncio
    async def test_unknown_sector_returns_empty(self):
        client = _client()
        with patch.object(client, "_get", new=AsyncMock()) as mget, \
             patch("evmax.clients.polymarket_us.cache_get", return_value=None):
            markets = await client.get_markets("lol")
        assert markets == []
        mget.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_league_error_does_not_kill_other_leagues(self):
        client = _client()
        payload = {"events": [_event([_moneyline_market()])]}

        async def _get_side_effect(path, params=None):
            if "epl" in path:
                raise RuntimeError("boom")
            return payload

        with patch.object(client, "_get", new=AsyncMock(side_effect=_get_side_effect)), \
             patch("evmax.clients.polymarket_us.cache_get", return_value=None), \
             patch("evmax.clients.polymarket_us.cache_set"):
            markets = await client.get_markets("soccer")
        # Every soccer league but epl succeeds; each event yields 2 moneyline
        # markets. Derive the count from the map so wiring a new league can't
        # silently break this (it grew from 3 to 6 leagues on 2026-09-05).
        from evmax.clients.polymarket_us import POLYMARKET_US_LEAGUE_MAP
        n_ok = len(POLYMARKET_US_LEAGUE_MAP["soccer"]) - 1  # epl raised
        assert len(markets) == 2 * n_ok

    @pytest.mark.asyncio
    async def test_fresh_bypasses_cache_read(self):
        # The pick/verify live-quote refresh passes fresh=True so it never
        # serves a stale scan-time cache — it must skip cache_get and hit the
        # network, even though cache_ttl_secs is > 0 by default.
        client = _client()
        payload = {"events": [_event([_moneyline_market()])]}
        with patch.object(client, "_get", new=AsyncMock(return_value=payload)) as mget, \
             patch("evmax.clients.polymarket_us.cache_get") as mcache_get, \
             patch("evmax.clients.polymarket_us.cache_set"):
            markets = await client.get_markets("wnba", fresh=True)
        mcache_get.assert_not_called()   # fresh → cache read bypassed
        assert mget.await_count == 1     # wnba → 1 league, fetched live
        assert len(markets) == 2         # both moneyline sides parsed


# ---------------------------------------------------------------------------
# Settlement lookups (outcome resolution)
# ---------------------------------------------------------------------------

class TestGetMarketSettlement:

    @pytest.mark.asyncio
    async def test_returns_settlement_price(self):
        client = _client()
        payload = {"slug": "aec-nfl-car-gb-2025-11-02", "settlement": 1}
        with patch.object(client, "_get", new=AsyncMock(return_value=payload)) as mget:
            s = await client.get_market_settlement("aec-nfl-car-gb-2025-11-02")
        assert s == 1.0
        mget.assert_awaited_once_with(
            "/v1/markets/aec-nfl-car-gb-2025-11-02/settlement"
        )

    @pytest.mark.asyncio
    async def test_unsettled_404_returns_none(self):
        import httpx

        client = _client()
        resp = httpx.Response(404, request=httpx.Request("GET", "http://x"))
        err = httpx.HTTPStatusError("404", request=resp.request, response=resp)
        with patch.object(client, "_get", new=AsyncMock(side_effect=err)):
            assert await client.get_market_settlement("aec-x") is None

    @pytest.mark.asyncio
    async def test_non_404_error_propagates(self):
        import httpx

        client = _client()
        resp = httpx.Response(500, request=httpx.Request("GET", "http://x"))
        err = httpx.HTTPStatusError("500", request=resp.request, response=resp)
        with patch.object(client, "_get", new=AsyncMock(side_effect=err)):
            with pytest.raises(httpx.HTTPStatusError):
                await client.get_market_settlement("aec-x")

    @pytest.mark.asyncio
    async def test_get_market_sides_unwraps_market(self):
        client = _client()
        payload = {"market": {"marketSides": [{"long": True}, {"long": False}]}}
        with patch.object(client, "_get", new=AsyncMock(return_value=payload)) as mget:
            sides = await client.get_market_sides("aec-x")
        assert sides == [{"long": True}, {"long": False}]
        mget.assert_awaited_once_with("/v1/market/slug/aec-x")


# ---------------------------------------------------------------------------
# Registry consistency
# ---------------------------------------------------------------------------

class TestLeagueMapRegistry:

    def test_every_league_map_key_is_a_known_category(self):
        from evmax.categories import all_categories
        known = {c.key for c in all_categories()}
        unknown = set(POLYMARKET_US_LEAGUE_MAP) - known
        assert not unknown, f"league map keys missing from categories.yaml: {unknown}"

    def test_no_prop_sectors_in_league_map(self):
        assert not any(k.endswith("_props") for k in POLYMARKET_US_LEAGUE_MAP)


class TestLeaguesOverride:
    """get_markets(leagues=...) — the arb scanner's coverage-extension hook."""

    @pytest.mark.asyncio
    async def test_override_fetches_given_leagues(self):
        client = _client()
        payload = {"events": [_event([_moneyline_market()])]}
        with patch.object(client, "_get", new=AsyncMock(return_value=payload)) as mget, \
             patch("evmax.clients.polymarket_us.cache_get", return_value=None), \
             patch("evmax.clients.polymarket_us.cache_set"):
            markets = await client.get_markets("worldcup", leagues=["fwc"])
        # worldcup has NO betting-map entry — only the override makes it fetch
        assert {c.args[0] for c in mget.await_args_list} == {"/v2/leagues/fwc/events"}
        assert len(markets) == 2

    @pytest.mark.asyncio
    async def test_override_uses_distinct_cache_namespace(self):
        """An overridden fetch must not read or poison the betting
        pipeline's cache entry for the same sector."""
        client = _client()
        payload = {"events": [_event([_moneyline_market()])]}
        seen_keys: list[str] = []

        def fake_cache_get(key, ttl):
            seen_keys.append(key)
            return None

        with patch.object(client, "_get", new=AsyncMock(return_value=payload)), \
             patch("evmax.clients.polymarket_us.cache_get", side_effect=fake_cache_get), \
             patch("evmax.clients.polymarket_us.cache_set") as mset:
            await client.get_markets("soccer")
            await client.get_markets("soccer", leagues=["lal", "uefa"])
        assert seen_keys[0] == "polymarket_us_soccer_open"
        assert seen_keys[1] == "polymarket_us_soccer_open_lal-uefa"
        written = [c.args[0] for c in mset.call_args_list]
        assert written == seen_keys
