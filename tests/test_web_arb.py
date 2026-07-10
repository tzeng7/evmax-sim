"""Dashboard arb endpoints — /api/arb/sectors and /api/arb/scan.

The Arbs tab runs the read-only cross-venue scanner (evmax.arb) on demand.
Nothing is persisted, so these tests only need the market fetch mocked —
no database seeding.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from evmax.arb import ARB_LEAGUE_MAP
from evmax.models.market import MarketSource, MarketType, PredictionMarket
from evmax.web.app import app

client = TestClient(app)

KDATE = datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc)
PDATE = datetime(2026, 7, 11, 0, 5, tzinfo=timezone.utc)


def _mk(mid, source, yes_team, yes_price, no_price, event_date):
    return PredictionMarket(
        id=mid,
        source=source,
        sector="baseball",
        market_type=MarketType.moneyline,
        title="Rangers vs Astros Winner?",
        ticker=mid.split(":", 1)[-1],
        yes_price=yes_price,
        no_price=no_price,
        volume_usd=100.0,
        team_home="astros",
        team_away="rangers",
        event_date=event_date,
        yes_team=yes_team,
    )


class TestArbSectors:
    def test_returns_arb_league_map_keys(self):
        resp = client.get("/api/arb/sectors")
        assert resp.status_code == 200
        assert resp.json()["sectors"] == list(ARB_LEAGUE_MAP.keys())


class TestArbScan:
    def _pool(self):
        return {
            "baseball": [
                _mk("kalshi:T1", MarketSource.kalshi, "astros", 0.40, 0.62, KDATE),
                _mk("polymarket_us:t1", MarketSource.polymarket_us, "astros",
                    0.55, 0.50, PDATE),
            ]
        }

    def test_scan_returns_serialized_baskets(self):
        with patch(
            "evmax.arb.fetch_arb_markets", new=AsyncMock(return_value=self._pool())
        ):
            resp = client.post("/api/arb/scan", json={"sectors": "baseball"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["sector_stats"] == [
            {"sector": "baseball", "kalshi_markets": 1, "polyus_markets": 1, "baskets": 1}
        ]
        assert len(data["baskets"]) == 1
        basket = data["baskets"][0]
        assert basket["sector"] == "baseball"
        assert basket["gross_cost"] == 0.90
        assert basket["net_edge"] > 0
        assert basket["cross_venue"] is True
        assert basket["in_play"] is False
        assert {leg["venue"] for leg in basket["legs"]} == {"kalshi", "polymarket_us"}
        assert data["scanned_at"]

    def test_max_cost_filters(self):
        with patch(
            "evmax.arb.fetch_arb_markets", new=AsyncMock(return_value=self._pool())
        ):
            resp = client.post("/api/arb/scan", json={"sectors": "baseball", "max_cost": 0.5})
        assert resp.json()["baskets"] == []

    def test_defaults_to_all_arb_sectors(self):
        mock = AsyncMock(return_value={})
        with patch("evmax.arb.fetch_arb_markets", new=mock):
            resp = client.post("/api/arb/scan", json={})
        assert resp.status_code == 200
        mock.assert_awaited_once_with(list(ARB_LEAGUE_MAP.keys()))
