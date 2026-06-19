"""The dashboard /api/resolve route fans out to portfolio outcome syncing,
mirroring how /api/scan fans out to portfolio logging — so one "Resolve"
click covers both dashboard and portfolio outcomes. See evmax.web.app.api_resolve.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

import evmax.agents.cleanup.resolver as resolver_mod
import evmax.portfolios as portfolios_mod
from evmax.web.app import app


def _patch(monkeypatch, *, resolved=3, portfolio_resolved=2):
    """Stub the resolver + portfolio sync so the route runs without network."""
    calls: dict[str, int] = {"resolve_dates": 0, "sync": 0}

    async def fake_resolve(d):
        calls["resolve_dates"] += 1
        return {"resolved": resolved, "failed": 0, "unmatched": []}

    def fake_sync():
        calls["sync"] += 1
        return portfolio_resolved

    # The route imports these names lazily from their source modules, so
    # patching at the source is what the import inside the handler will see.
    monkeypatch.setattr(resolver_mod, "resolve_outcomes_for_date", fake_resolve)
    monkeypatch.setattr(portfolios_mod, "sync_portfolio_outcomes", fake_sync)
    return calls


def test_resolve_syncs_portfolios_by_default(monkeypatch):
    calls = _patch(monkeypatch, resolved=3, portfolio_resolved=2)
    with TestClient(app) as client:
        resp = client.post("/api/resolve", json={"date": "2026-06-17"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["resolved"] == 3
    assert body["portfolio_resolved"] == 2
    assert calls["resolve_dates"] == 1
    assert calls["sync"] == 1  # portfolio sync ran once


def test_resolve_can_skip_portfolio_sync(monkeypatch):
    calls = _patch(monkeypatch, resolved=1, portfolio_resolved=9)
    with TestClient(app) as client:
        resp = client.post(
            "/api/resolve",
            json={"date": "2026-06-17", "sync_portfolios": False},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["resolved"] == 1
    assert body["portfolio_resolved"] == 0  # not synced
    assert calls["sync"] == 0  # portfolio sync skipped


def test_resolve_defaults_date_to_yesterday(monkeypatch):
    """No date in body → resolves yesterday and still reports a date."""
    _patch(monkeypatch)
    with TestClient(app) as client:
        resp = client.post("/api/resolve", json={})
    assert resp.status_code == 200
    assert resp.json()["date"]  # an ISO date string is echoed back
