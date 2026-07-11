"""Dashboard arb endpoint — /api/arb/scan mirrors `evmax arb scan` (read-only)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

import evmax.arb as arb_module
from evmax.models.market import MarketSource, MarketType, PredictionMarket
from evmax.web.app import app

# Far-future dates so in_play (start <= now) can never trip as wall-clock
# time advances. Kalshi noon-UTC anchor + PolyUS true start, same ET day.
KDATE = datetime(2030, 7, 10, 12, 0, tzinfo=timezone.utc)
PDATE = datetime(2030, 7, 11, 0, 5, tzinfo=timezone.utc)


def mk(mid, source, yes_team, yes_price, no_price, event_date=KDATE, volume=100.0):
    return PredictionMarket(
        id=mid,
        source=source,
        sector="baseball",
        market_type=MarketType.moneyline,
        title="rangers vs astros",
        ticker=mid.split(":", 1)[-1],
        yes_price=yes_price,
        no_price=no_price,
        volume_usd=volume,
        team_home="astros",
        team_away="rangers",
        event_date=event_date,
        yes_team=yes_team,
    )


def _patch_pool(monkeypatch, pool):
    async def fake_fetch(sectors):
        return {"baseball": pool}

    monkeypatch.setattr(arb_module, "fetch_arb_markets", fake_fetch)


def test_arb_scan_returns_cross_venue_basket(monkeypatch):
    pool = [
        mk("kalshi:T1", MarketSource.kalshi, "astros", 0.40, 0.62),
        mk("polymarket_us:t1", MarketSource.polymarket_us, "astros", 0.55, 0.50,
           event_date=PDATE, volume=250.0),
    ]
    _patch_pool(monkeypatch, pool)

    with TestClient(app) as client:
        resp = client.post("/api/arb/scan", json={"sectors": "baseball", "max_cost": 1.0})

    assert resp.status_code == 200
    data = resp.json()
    assert data["max_cost"] == 1.0
    assert data["sectors"] == [
        {"sector": "baseball", "kalshi": 1, "polymarket_us": 1, "baskets": 1}
    ]

    (arb,) = data["arbs"]
    assert arb["sector"] == "baseball"
    assert arb["event_title"]  # Event column identifier
    assert arb["market_desc"] == "moneyline"  # Outcome column identifier
    assert arb["cross_venue"] is True
    assert arb["in_play"] is False
    # cheapest cover: astros K-yes @0.40 + rangers P-no @0.50 = 0.90 gross
    assert arb["gross_cost"] == pytest.approx(0.90)
    assert arb["net_cost"] > arb["gross_cost"]  # taker fees added
    assert arb["net_edge"] == pytest.approx(1.0 - arb["net_cost"], abs=1e-4)
    assert arb["min_volume_usd"] == 100.0
    sides = {(leg["venue"], leg["side"]) for leg in arb["legs"]}
    assert sides == {("kalshi", "yes"), ("polymarket_us", "no")}
    for leg in arb["legs"]:
        assert leg["fee"] > 0


def test_arb_scan_efficient_books_no_baskets(monkeypatch):
    pool = [
        mk("kalshi:T1", MarketSource.kalshi, "astros", 0.51, 0.51),
        mk("polymarket_us:t1", MarketSource.polymarket_us, "astros", 0.51, 0.51,
           event_date=PDATE),
    ]
    _patch_pool(monkeypatch, pool)

    with TestClient(app) as client:
        resp = client.post("/api/arb/scan", json={"sectors": "baseball", "max_cost": 1.0})

    assert resp.status_code == 200
    data = resp.json()
    assert data["arbs"] == []
    assert data["sectors"][0]["baskets"] == 0


def test_arb_scan_defaults_to_both_venue_sectors(monkeypatch):
    """No sectors in the body → scan every sector both venues carry (CLI parity)."""
    from evmax.clients.polymarket_us import POLYMARKET_US_LEAGUE_MAP

    seen: list[str] = []

    async def fake_fetch(sectors):
        seen.extend(sectors)
        return {s: [] for s in sectors}

    monkeypatch.setattr(arb_module, "fetch_arb_markets", fake_fetch)

    with TestClient(app) as client:
        resp = client.post("/api/arb/scan", json={})

    assert resp.status_code == 200
    assert seen == list(POLYMARKET_US_LEAGUE_MAP)
    assert resp.json()["max_cost"] == 1.02  # near-miss ceiling default
