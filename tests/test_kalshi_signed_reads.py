"""Kalshi read path: signed requests, limiter-aware retries, page size,
disabled-series skipping and partial-fetch reporting (2026-11-20).

Background: scan reads went out UNSIGNED and got Kalshi's anonymous rate
bucket (~8 req/s). 59% of a cold scan's Kalshi requests were 429s, the base
client's retries bypassed the limiter, and series that exhausted their retries
were dropped with no trace (the live NCAAB moneyline series among them).
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

import evmax.clients.kalshi as kalshi_mod
from evmax.agents.base import AgentRequest
from evmax.clients.kalshi import KalshiClient, _ratelimit_reset_seconds
from evmax.models.market import MarketType


def _bare_client() -> KalshiClient:
    c = KalshiClient.__new__(KalshiClient)
    c.base_url = "https://api.elections.kalshi.com/trade-api/v2"
    c._key_id = "key-id"
    c._private_key_path = "/nonexistent"
    c.last_failed_series = []
    c.last_failed_events = []
    return c


class _Limiter:
    def __init__(self) -> None:
        self.tokens = 0

    async def acquire(self) -> None:
        self.tokens += 1


@pytest.fixture
def limiter(monkeypatch) -> _Limiter:
    lim = _Limiter()
    monkeypatch.setattr(kalshi_mod, "_KALSHI_RATE_LIMITER", lim)
    return lim


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    waits: list[float] = []

    async def fake_sleep(s: float) -> None:
        waits.append(s)

    monkeypatch.setattr(kalshi_mod.asyncio, "sleep", fake_sleep)
    return waits


def _with_transport(c: KalshiClient, handler) -> None:
    c._client = httpx.AsyncClient(
        base_url=c.base_url, transport=httpx.MockTransport(handler)
    )


class TestSignedGet:
    async def test_signs_with_full_path_and_returns_json(self, limiter, monkeypatch) -> None:
        c = _bare_client()
        signed_paths: list[str] = []

        def fake_sign(method: str, path: str) -> dict:
            signed_paths.append(path)
            return {"KALSHI-ACCESS-KEY": "key-id"}

        monkeypatch.setattr(c, "_sign_request", fake_sign)
        seen: list[httpx.Request] = []

        def handler(req: httpx.Request) -> httpx.Response:
            seen.append(req)
            return httpx.Response(200, json={"markets": [], "cursor": ""})

        _with_transport(c, handler)
        out = await c._get("/markets", params={"series_ticker": "KXNBAGAME"})
        assert out == {"markets": [], "cursor": ""}
        # The signature covers the path (with the /trade-api/v2 prefix), not the query.
        assert signed_paths == ["/trade-api/v2/markets"]
        assert seen[0].headers["KALSHI-ACCESS-KEY"] == "key-id"
        assert limiter.tokens == 1

    async def test_unsigned_when_no_key_material(self, limiter, monkeypatch) -> None:
        c = _bare_client()
        monkeypatch.setattr(c, "_sign_request", lambda m, p: {})
        seen: list[httpx.Request] = []

        def handler(req: httpx.Request) -> httpx.Response:
            seen.append(req)
            return httpx.Response(200, json={"ok": 1})

        _with_transport(c, handler)
        assert await c._get("/markets") == {"ok": 1}
        assert "KALSHI-ACCESS-KEY" not in seen[0].headers

    async def test_429_waits_for_reset_and_every_attempt_takes_a_token(
        self, limiter, sleeps, monkeypatch,
    ) -> None:
        c = _bare_client()
        monkeypatch.setattr(c, "_sign_request", lambda m, p: {"KALSHI-ACCESS-KEY": "k"})
        responses = iter([
            httpx.Response(429, headers={"x-ratelimit-reset": "2.8"}),
            httpx.Response(429, headers={"x-ratelimit-reset": "2.8"}),
            httpx.Response(200, json={"markets": [1]}),
        ])
        _with_transport(c, lambda req: next(responses))
        assert await c._get("/markets") == {"markets": [1]}
        assert sleeps == [2.8, 2.8]
        # The old tenacity retries never touched the limiter (retry storm).
        assert limiter.tokens == 3

    async def test_429_exhausts_attempts_then_raises(self, limiter, sleeps, monkeypatch) -> None:
        c = _bare_client()
        monkeypatch.setattr(c, "_sign_request", lambda m, p: {})
        _with_transport(c, lambda req: httpx.Response(429))
        with pytest.raises(httpx.HTTPStatusError):
            await c._get("/markets")
        assert limiter.tokens == kalshi_mod._KALSHI_MAX_ATTEMPTS
        # No sleep after the final attempt.
        assert len(sleeps) == kalshi_mod._KALSHI_MAX_ATTEMPTS - 1

    async def test_non_retryable_error_raises_immediately(self, limiter, sleeps, monkeypatch) -> None:
        c = _bare_client()
        monkeypatch.setattr(c, "_sign_request", lambda m, p: {})
        _with_transport(c, lambda req: httpx.Response(400))
        with pytest.raises(httpx.HTTPStatusError):
            await c._get("/events", params={"limit": 201})
        assert limiter.tokens == 1 and sleeps == []

    async def test_5xx_backs_off_then_succeeds(self, limiter, sleeps, monkeypatch) -> None:
        c = _bare_client()
        monkeypatch.setattr(c, "_sign_request", lambda m, p: {})
        responses = iter([httpx.Response(503), httpx.Response(200, json={"x": 1})])
        _with_transport(c, lambda req: next(responses))
        assert await c._get("/markets") == {"x": 1}
        assert sleeps == [1.0]


class TestRateLimitResetSeconds:
    def test_reads_kalshi_header(self) -> None:
        assert _ratelimit_reset_seconds({"x-ratelimit-reset": "2.755"}) == 2.755

    def test_falls_back_to_retry_after_then_default(self) -> None:
        assert _ratelimit_reset_seconds({"retry-after": "3"}) == 3.0
        assert _ratelimit_reset_seconds({}) == 1.0
        assert _ratelimit_reset_seconds({"x-ratelimit-reset": "junk"}) == 1.0

    def test_clamped(self) -> None:
        assert _ratelimit_reset_seconds({"x-ratelimit-reset": "0"}) == 0.25
        assert _ratelimit_reset_seconds({"x-ratelimit-reset": "600"}) == kalshi_mod._KALSHI_429_MAX_WAIT_S


class TestPrivateKeyCache:
    def test_parsed_once_per_file_and_still_signs(self, tmp_path, monkeypatch) -> None:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        path = tmp_path / "kalshi.pem"
        path.write_bytes(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ))
        monkeypatch.setattr(kalshi_mod, "_PRIVATE_KEY_CACHE", {})
        calls = {"n": 0}
        real = serialization.load_pem_private_key

        def counting(data, password=None):
            calls["n"] += 1
            return real(data, password=password)

        monkeypatch.setattr(serialization, "load_pem_private_key", counting)
        c = _bare_client()
        c._private_key_path = str(path)
        h1 = c._sign_request("GET", "/trade-api/v2/markets")
        h2 = c._sign_request("GET", "/trade-api/v2/markets")
        assert calls["n"] == 1
        assert h1["KALSHI-ACCESS-KEY"] == "key-id" and h2["KALSHI-ACCESS-SIGNATURE"]

    def test_missing_file_is_not_cached(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(kalshi_mod, "_PRIVATE_KEY_CACHE", {})
        c = _bare_client()
        c._private_key_path = str(tmp_path / "absent.pem")
        assert c._load_private_key() is None
        assert kalshi_mod._PRIVATE_KEY_CACHE == {}

    def test_no_key_id_skips_key_load(self, monkeypatch) -> None:
        c = _bare_client()
        c._key_id = ""
        monkeypatch.setattr(c, "_load_private_key", lambda: pytest.fail("loaded key"))
        assert c._sign_request("GET", "/x") == {}


def _raw(ticker: str, title: str) -> dict:
    return {
        "ticker": ticker, "title": title,
        "yes_bid_dollars": "0.40", "yes_ask_dollars": "0.45",
        "no_bid_dollars": "0.50", "no_ask_dollars": "0.60",
        "volume_fp": 10, "open_interest_fp": 5,
    }


@pytest.fixture
def no_cache(monkeypatch):
    cfg = SimpleNamespace(offline_mode=False, cache_ttl_secs=0, kalshi_ws_enabled=False)
    monkeypatch.setattr(kalshi_mod, "get_settings", lambda: cfg)
    return cfg


class TestGetMarketsSeriesSelection:
    async def test_default_markets_page_is_1000(self, no_cache) -> None:
        c = _bare_client()
        calls: list = []

        async def fake_get(path, params=None):
            calls.append((path, dict(params or {})))
            return {"markets": [], "cursor": ""}

        c._get = fake_get  # type: ignore[method-assign]
        await c.get_markets("nfl")
        assert {p["limit"] for path, p in calls if path == "/markets"} == {1000}

    async def test_skipped_types_are_never_requested(self, no_cache) -> None:
        c = _bare_client()
        series: list[str] = []

        async def fake_get(path, params=None):
            series.append(params["series_ticker"])
            if params["series_ticker"] == "KXNCAAFGAME":
                return {"markets": [_raw("KXNCAAFGAME-26NOV21OSUMICH-OSU", "Ohio State wins")],
                        "cursor": ""}
            return {"markets": [], "cursor": ""}

        c._get = fake_get  # type: ignore[method-assign]
        out = await c.get_markets(
            "ncaaf", skip_market_types=frozenset({MarketType.spread, MarketType.total}),
        )
        assert series == ["KXNCAAFGAME"]
        assert [m.ticker for m in out] == ["KXNCAAFGAME-26NOV21OSUMICH-OSU"]

    async def test_unsuffixed_series_always_fetched(self, no_cache) -> None:
        # KXCS2GAMES has no known suffix (GAMES != GAME): never skipped.
        c = _bare_client()
        series: list[str] = []

        async def fake_get(path, params=None):
            if path == "/markets":
                series.append(params["series_ticker"])
            return {"markets": [], "events": [], "cursor": ""}

        c._get = fake_get  # type: ignore[method-assign]
        await c.get_markets("cs2", skip_market_types=frozenset({MarketType.moneyline}))
        assert "KXCS2GAMES" in series and "KXCS2GAME" not in series

    async def test_skip_uses_its_own_cache_key(self, no_cache, monkeypatch) -> None:
        # A skip-filtered scan must not serve a later unfiltered caller
        # (watch-listings archives the ladders) from the dev cache.
        no_cache.cache_ttl_secs = 120
        keys: list[str] = []
        monkeypatch.setattr(kalshi_mod, "cache_get", lambda k, ttl: keys.append(k))
        monkeypatch.setattr(kalshi_mod, "cache_set", lambda k, v: None)
        c = _bare_client()

        async def fake_get(path, params=None):
            return {"markets": [], "cursor": ""}

        c._get = fake_get  # type: ignore[method-assign]
        await c.get_markets("nfl")
        await c.get_markets("nfl", skip_market_types=frozenset({MarketType.total}))
        assert keys == ["kalshi_nfl_open", "kalshi_nfl_open_skip-total"]


class TestPartialFetchReporting:
    async def test_failed_series_recorded_and_rest_returned(self, no_cache) -> None:
        c = _bare_client()

        async def fake_get(path, params=None):
            if params["series_ticker"] == "KXNFLSPREAD":
                raise httpx.HTTPStatusError(
                    "429", request=httpx.Request("GET", "http://x"),
                    response=httpx.Response(429),
                )
            if params["series_ticker"] == "KXNFLGAME":
                return {"markets": [_raw("KXNFLGAME-26NOV22KCBUF-KC", "Kansas City wins")],
                        "cursor": ""}
            return {"markets": [], "cursor": ""}

        c._get = fake_get  # type: ignore[method-assign]
        out = await c.get_markets("nfl")
        assert len(out) == 1
        assert c.last_failed_series == ["KXNFLSPREAD"]
        assert c.last_failed_events == []

    async def test_failed_events_join_recorded(self, no_cache) -> None:
        c = _bare_client()

        async def fake_get(path, params=None):
            if path == "/events":
                raise httpx.HTTPStatusError(
                    "429", request=httpx.Request("GET", "http://x"),
                    response=httpx.Response(429),
                )
            return {"markets": [], "cursor": ""}

        c._get = fake_get  # type: ignore[method-assign]
        await c.get_markets("ufc")
        assert c.last_failed_events == ["KXUFCFIGHT"]
        assert c.last_failed_series == []

    async def test_failures_reset_each_call(self, no_cache) -> None:
        c = _bare_client()
        c.last_failed_series = ["STALE"]

        async def fake_get(path, params=None):
            return {"markets": [], "cursor": ""}

        c._get = fake_get  # type: ignore[method-assign]
        await c.get_markets("nhl")
        assert c.last_failed_series == []


class TestDisabledMarketTypes:
    def test_ncaaf_spread_and_total_are_disabled(self) -> None:
        from evmax.agents.odds.kalshi_agent import disabled_market_types
        from evmax.categories import get_category

        expected = {
            MarketType(mt) for mt in get_category("ncaaf").disabled_market_types
        }
        assert expected  # the registry still disables NCAAF ladders
        assert disabled_market_types("ncaaf") == frozenset(expected)

    def test_runtime_override_reenables_every_type(self) -> None:
        from evmax.agents.odds.kalshi_agent import disabled_market_types
        from evmax.modes import clear_runtime_overrides, set_runtime_overrides

        set_runtime_overrides({"ncaaf": "shadow"})
        try:
            assert disabled_market_types("ncaaf") == frozenset()
        finally:
            clear_runtime_overrides()

    def test_whole_disabled_sector_fetches_everything(self) -> None:
        # A disabled sector only runs when forced with --sectors (debugging).
        from evmax.agents.odds.kalshi_agent import disabled_market_types
        from evmax.modes import clear_runtime_overrides, set_runtime_overrides

        set_runtime_overrides({"nhl": "disabled"})
        try:
            assert disabled_market_types("nhl") == frozenset()
        finally:
            clear_runtime_overrides()

    def test_unknown_sector_disables_nothing(self) -> None:
        from evmax.agents.odds.kalshi_agent import disabled_market_types

        assert disabled_market_types("valorant_not_a_category") == frozenset()


class _FakeClient:
    def __init__(self, failed: list[str], failed_events: list[str]) -> None:
        self.kwargs: dict = {}
        self.last_failed_series = failed
        self.last_failed_events = failed_events

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return None

    async def get_markets(self, sector, **kwargs):
        self.kwargs = kwargs
        return []


class TestKalshiOddsAgent:
    async def test_partial_status_and_skip_passed_through(self, monkeypatch) -> None:
        import evmax.agents.odds.kalshi_agent as agent_mod

        fake = _FakeClient(["KXNCAAFGAME"], ["KXNCAAFGAME"])
        monkeypatch.setattr(agent_mod, "KalshiClient", lambda: fake)
        monkeypatch.setattr(
            agent_mod, "disabled_market_types",
            lambda s: frozenset({MarketType.spread}),
        )
        resp = await agent_mod.KalshiOddsAgent().run(AgentRequest(sector="ncaaf"))
        assert fake.kwargs["skip_market_types"] == frozenset({MarketType.spread})
        assert resp.status == "partial"
        assert resp.error == "kalshi series failed markets:KXNCAAFGAME events:KXNCAAFGAME"

    async def test_clean_fetch_is_ok_and_skip_none_when_empty(self, monkeypatch) -> None:
        import evmax.agents.odds.kalshi_agent as agent_mod

        fake = _FakeClient([], [])
        monkeypatch.setattr(agent_mod, "KalshiClient", lambda: fake)
        monkeypatch.setattr(agent_mod, "disabled_market_types", lambda s: frozenset())
        resp = await agent_mod.KalshiOddsAgent().run(AgentRequest(sector="nba"))
        assert fake.kwargs["skip_market_types"] is None
        assert resp.status == "ok" and resp.error is None
