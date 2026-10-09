"""KalshiClient.get_market_price reads the *_dollars price fields.

Regression: the open-market branch read the legacy integer-cent ``yes_bid`` /
``yes_ask`` fields, which Kalshi no longer returns, so every open market priced
as None (and sub-cent escalator ticks could never be represented).
"""

from unittest.mock import AsyncMock

import pytest

from evmax.clients.kalshi import KalshiClient


async def _price(market: dict):
    c = KalshiClient()
    c._get = AsyncMock(return_value={"market": market})
    return await c.get_market_price("KXNFLRECYDS-26OCT11X-WR-60")


@pytest.mark.asyncio
async def test_open_market_uses_dollar_fields():
    assert await _price({"result": "", "yes_bid_dollars": "0.2100", "yes_ask_dollars": "0.2600"}) \
        == pytest.approx(0.235)


@pytest.mark.asyncio
async def test_sub_cent_ticks_survive():
    assert await _price({"result": "", "yes_bid_dollars": "0.0362", "yes_ask_dollars": "0.0363"}) \
        == pytest.approx(0.03625)


@pytest.mark.asyncio
async def test_legacy_cent_fields_still_parse():
    assert await _price({"result": "", "yes_bid": 40, "yes_ask": 44}) == pytest.approx(0.42)


@pytest.mark.asyncio
async def test_settled_results_and_bid_only():
    assert await _price({"result": "yes"}) == 1.0
    assert await _price({"result": "no"}) == 0.0
    assert await _price({"result": "", "yes_bid_dollars": "0.1500", "yes_ask_dollars": "0.0000"}) \
        == pytest.approx(0.15)
    assert await _price({"result": "", "yes_bid_dollars": "0.0000", "yes_ask_dollars": "0.0000"}) is None
