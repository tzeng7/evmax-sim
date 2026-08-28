"""S5 — Pinnacle resilience: failure classification, last_error recording,
and the reachability probe. Pinnacle is the sole sharp anchor, so a scan must
fail CLEAR (no plays) on an outage while making the reason knowable. No network.
"""
from __future__ import annotations

import asyncio

import httpx

from evmax.clients.esports_pinnacle import PinnacleGuestClient, classify_pinnacle_error


def _http_error(code: int, body: str = "") -> httpx.HTTPStatusError:
    req = httpx.Request("GET", "http://pinnacle.test/x")
    resp = httpx.Response(code, request=req, content=body.encode("utf-8"))
    return httpx.HTTPStatusError("err", request=req, response=resp)


class TestClassify:
    def test_maintenance_503(self):
        assert classify_pinnacle_error(_http_error(503)) == (503, "maintenance")

    def test_geo_block_403_bad_location(self):
        assert classify_pinnacle_error(_http_error(403, "BAD_LOCATION")) == (403, "geo_block")

    def test_forbidden_403_plain(self):
        assert classify_pinnacle_error(_http_error(403, "denied")) == (403, "forbidden")

    def test_rate_limited_429(self):
        assert classify_pinnacle_error(_http_error(429)) == (429, "rate_limited")

    def test_other_http_status(self):
        assert classify_pinnacle_error(_http_error(500)) == (500, "http_error")

    def test_timeout(self):
        assert classify_pinnacle_error(httpx.ConnectTimeout("t")) == (None, "timeout")

    def test_network(self):
        assert classify_pinnacle_error(httpx.ConnectError("c")) == (None, "network")

    def test_generic(self):
        assert classify_pinnacle_error(ValueError("x")) == (None, "error")


class TestGetOddsFailClear:
    def test_failure_records_last_error_and_returns_empty(self, monkeypatch):
        async def _run():
            async with PinnacleGuestClient() as client:
                async def _boom(*a, **k):
                    raise _http_error(503)
                monkeypatch.setattr(client, "_logged_get", _boom)
                odds = await client.get_odds("nba")
                return odds, client.last_error

        odds, err = asyncio.run(_run())
        assert odds == []                       # fail clear — no plays
        assert err["reason"] == "maintenance"   # but the reason is knowable
        assert err["sector"] == "nba"

    def test_geo_block_recorded(self, monkeypatch):
        async def _run():
            async with PinnacleGuestClient() as client:
                async def _boom(*a, **k):
                    raise _http_error(403, "BAD_LOCATION")
                monkeypatch.setattr(client, "_logged_get", _boom)
                await client.get_odds("nba")
                return client.last_error

        err = asyncio.run(_run())
        assert err["reason"] == "geo_block"

    def test_success_resets_last_error(self, monkeypatch):
        async def _run():
            async with PinnacleGuestClient() as client:
                client.last_error = {"stale": True}

                async def _empty(*a, **k):
                    return []  # a successful fetch that lists no matchups
                monkeypatch.setattr(client, "_logged_get", _empty)
                await client.get_odds("nba")
                return client.last_error

        assert asyncio.run(_run()) is None


class TestProbe:
    def test_probe_reachable(self, monkeypatch):
        async def _run():
            async with PinnacleGuestClient() as client:
                async def _ok(*a, **k):
                    return []
                monkeypatch.setattr(client, "_logged_get", _ok)
                return await client.probe()

        assert asyncio.run(_run()) == {"ok": True, "status": 200, "reason": "ok"}

    def test_probe_down_geo_block(self, monkeypatch):
        async def _run():
            async with PinnacleGuestClient() as client:
                async def _boom(*a, **k):
                    raise _http_error(403, "BAD_LOCATION")
                monkeypatch.setattr(client, "_logged_get", _boom)
                return await client.probe()

        r = asyncio.run(_run())
        assert r["ok"] is False
        assert r["reason"] == "geo_block"

    def test_probe_never_raises(self, monkeypatch):
        async def _run():
            async with PinnacleGuestClient() as client:
                async def _boom(*a, **k):
                    raise RuntimeError("unexpected")
                monkeypatch.setattr(client, "_logged_get", _boom)
                return await client.probe()

        r = asyncio.run(_run())
        assert r["ok"] is False
        assert r["reason"] == "error"
