"""Live account-balance fetch — client methods + venue dispatcher.

The balance calls are FAIL-SOFT: no credentials / a bad response / a network
error must yield None (a manual-bankroll fallback), never a fabricated number.
These tests pin the parse (cents→USD for Kalshi, buyingPower for Poly), the
currency selection, and the None paths.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from evmax.clients.balances import (
    fetch_all_balances,
    fetch_balance,
    resolve_bankroll,
)
from evmax.clients.kalshi import KalshiClient
from evmax.clients.polymarket_us import PolymarketUSClient


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _kalshi_with_client(payload):
    c = KalshiClient()
    c._client = type("H", (), {"get": AsyncMock(return_value=_FakeResp(payload))})()
    return c


def _poly_with_client(payload):
    c = PolymarketUSClient()
    c._client = type("H", (), {"get": AsyncMock(return_value=_FakeResp(payload))})()
    return c


# ---------------------------------------------------------------------------
# Kalshi get_balance
# ---------------------------------------------------------------------------

class TestKalshiBalance:
    @pytest.mark.asyncio
    async def test_total_wealth_cash_plus_positions(self):
        # total = (balance 51234c + portfolio_value 8766c) / 100 = 600.00
        c = _kalshi_with_client({"balance": 51234, "portfolio_value": 8766})
        with patch.object(c, "_sign_request", return_value={"KALSHI-ACCESS-KEY": "k"}):
            assert await c.get_balance() == pytest.approx(600.0)

    @pytest.mark.asyncio
    async def test_no_positions_is_just_cash(self):
        c = _kalshi_with_client({"balance": 51234})
        with patch.object(c, "_sign_request", return_value={"KALSHI-ACCESS-KEY": "k"}):
            assert await c.get_balance() == pytest.approx(512.34)

    @pytest.mark.asyncio
    async def test_no_credentials_returns_none(self):
        c = _kalshi_with_client({"balance": 51234})
        with patch.object(c, "_sign_request", return_value={}):
            assert await c.get_balance() is None

    @pytest.mark.asyncio
    async def test_missing_balance_field_returns_none(self):
        c = _kalshi_with_client({"portfolio_value": 100})
        with patch.object(c, "_sign_request", return_value={"KALSHI-ACCESS-KEY": "k"}):
            assert await c.get_balance() is None

    @pytest.mark.asyncio
    async def test_http_error_returns_none(self):
        c = KalshiClient()
        c._client = type("H", (), {"get": AsyncMock(side_effect=RuntimeError("boom"))})()
        with patch.object(c, "_sign_request", return_value={"KALSHI-ACCESS-KEY": "k"}):
            assert await c.get_balance() is None

    @pytest.mark.asyncio
    async def test_signed_path_includes_base_prefix(self):
        """The signed path must carry the /trade-api/v2 base prefix, else the
        RSA signature won't match the request URL Kalshi reconstructs."""
        c = _kalshi_with_client({"balance": 100})
        with patch.object(c, "_sign_request", return_value={"KALSHI-ACCESS-KEY": "k"}) as sign:
            await c.get_balance()
        method, path = sign.call_args.args
        assert method == "GET"
        assert path == "/trade-api/v2/portfolio/balance"


# ---------------------------------------------------------------------------
# Polymarket US get_balance
# ---------------------------------------------------------------------------

class TestPolyBalance:
    @pytest.mark.asyncio
    async def test_total_wealth_cash_plus_asset_available(self):
        # total = currentBalance 300 + assetAvailable 250 = 550 (NOT buyingPower,
        # NOT assetNotional face value)
        payload = {"balances": [
            {"currency": "USD", "currentBalance": 300.0, "assetAvailable": 250.0,
             "assetNotional": 800.0, "buyingPower": 120.0},
        ]}
        c = _poly_with_client(payload)
        with patch.object(c, "_sign_request", return_value={"X-PM-Access-Key": "k"}):
            assert await c.get_balance() == pytest.approx(550.0)

    @pytest.mark.asyncio
    async def test_no_positions_is_just_cash(self):
        payload = {"balances": [{"currency": "USD", "currentBalance": 300.0}]}
        c = _poly_with_client(payload)
        with patch.object(c, "_sign_request", return_value={"X-PM-Access-Key": "k"}):
            assert await c.get_balance() == pytest.approx(300.0)

    @pytest.mark.asyncio
    async def test_currency_selection(self):
        payload = {"balances": [
            {"currency": "EUR", "currentBalance": 999.0},
            {"currency": "USD", "currentBalance": 111.0, "assetAvailable": 0.0},
        ]}
        c = _poly_with_client(payload)
        with patch.object(c, "_sign_request", return_value={"X-PM-Access-Key": "k"}):
            assert await c.get_balance() == pytest.approx(111.0)

    @pytest.mark.asyncio
    async def test_missing_currency_returns_none(self):
        payload = {"balances": [{"currency": "EUR", "currentBalance": 999.0}]}
        c = _poly_with_client(payload)
        with patch.object(c, "_sign_request", return_value={"X-PM-Access-Key": "k"}):
            assert await c.get_balance() is None

    @pytest.mark.asyncio
    async def test_no_credentials_returns_none(self):
        c = _poly_with_client({"balances": []})
        with patch.object(c, "_sign_request", return_value={}):
            assert await c.get_balance() is None


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

class TestDispatcher:
    @pytest.mark.asyncio
    async def test_unknown_venue_returns_none(self):
        assert await fetch_balance("betfair") is None

    @pytest.mark.asyncio
    async def test_fetch_all_maps_each_venue(self):
        async def fake(venue):
            return {"kalshi": 500.0, "polymarket_us": 250.0}[venue]

        with patch("evmax.clients.balances.fetch_balance", side_effect=fake):
            out = await fetch_all_balances(["kalshi", "polymarket_us"])
        assert out == {"kalshi": 500.0, "polymarket_us": 250.0}

    @pytest.mark.asyncio
    async def test_fetch_all_defaults_to_supported_venues(self):
        with patch("evmax.clients.balances.fetch_balance", AsyncMock(return_value=None)):
            out = await fetch_all_balances()
        assert set(out.keys()) == {"kalshi", "polymarket_us"}


class TestResolveBankroll:
    @pytest.mark.asyncio
    async def test_no_venue_uses_manual(self):
        bk, src = await resolve_bankroll(500.0, None)
        assert (bk, src) == (500.0, "manual")

    @pytest.mark.asyncio
    async def test_live_balance_replaces_manual(self):
        with patch("evmax.clients.balances.fetch_balance", AsyncMock(return_value=712.5)):
            bk, src = await resolve_bankroll(500.0, "kalshi")
        assert bk == pytest.approx(712.5)
        assert src == "live:kalshi"

    @pytest.mark.asyncio
    async def test_unavailable_balance_falls_back_to_manual(self):
        """Real money must never be sized against a fabricated figure — an
        unavailable balance falls back to the passed bankroll, flagged."""
        with patch("evmax.clients.balances.fetch_balance", AsyncMock(return_value=None)):
            bk, src = await resolve_bankroll(500.0, "polymarket_us")
        assert bk == pytest.approx(500.0)
        assert src == "manual_fallback"
