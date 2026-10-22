"""Reason attribution for the read-only /api/modes report (evmax.mode_report)."""

from __future__ import annotations

from datetime import date

import pytest
from fastapi.testclient import TestClient

from evmax import mode_report as mr
from evmax.modes import clear_runtime_overrides, reset_overrides_cache, set_runtime_overrides
from evmax.categories import get_category
from evmax.settings import get_settings
from evmax.web.app import app

IN_SEASON = date(2026, 10, 21)
OFF_SEASON = date(2026, 7, 1)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("EVMAX_CATEGORY_MODES", raising=False)
    reset_overrides_cache()
    clear_runtime_overrides()
    yield
    reset_overrides_cache()
    clear_runtime_overrides()


def _row(cat, mt, venue="kalshi", today=IN_SEASON):
    return mr.explain_mode(get_category(cat), mt, venue, today)


def test_yaml_base_reason():
    r = _row("nhl", "moneyline")
    assert (r["mode"], r["reason"]) == ("shadow", mr.YAML_BASE)


def test_disabled_market_type():
    r = _row("baseball", "total")
    assert (r["mode"], r["reason"]) == ("disabled", mr.MARKET_TYPE_DISABLED)


def test_market_type_shadow():
    r = _row("nfl", "spread")
    assert (r["mode"], r["reason"]) == ("shadow", mr.MARKET_TYPE_SHADOW)


def test_venue_market_type_shadow_is_venue_specific():
    spec = get_category("nfl")
    venues = spec.shadow_venue_market_types
    assert "total" in venues.get("kalshi", ())
    r = _row("nfl", "total", "kalshi")
    assert (r["mode"], r["reason"]) == ("shadow", mr.VENUE_SHADOW)


def test_off_season_downgrade_wins_over_market_type():
    r = _row("nfl", "moneyline", today=OFF_SEASON)
    assert (r["mode"], r["reason"]) == ("shadow", mr.OFF_SEASON)
    # off-season is applied first, so a market-type-shadow type reports it too
    assert _row("nfl", "spread", today=OFF_SEASON)["reason"] == mr.OFF_SEASON


def test_runtime_override_reason():
    set_runtime_overrides({"nhl": "live"})
    r = _row("nhl", "moneyline")
    assert (r["mode"], r["reason"]) == ("live", mr.RUNTIME_OVERRIDE)


def test_env_override_reason(monkeypatch):
    monkeypatch.setenv("EVMAX_CATEGORY_MODES", '{"nhl": "disabled"}')
    reset_overrides_cache()
    r = _row("nhl", "moneyline")
    assert (r["mode"], r["reason"]) == ("disabled", mr.ENV_OVERRIDE)


def test_venue_firewall_demotes_live_polyus(monkeypatch):
    cfg = get_settings()
    monkeypatch.setattr(cfg, "polymarket_us_live", False)
    monkeypatch.setattr(cfg, "polymarket_us_live_sectors", "")
    r = _row("nba", "total", "polymarket_us")
    # nba moneyline is shadow-typed; total is live on kalshi, firewalled on polyus
    k = _row("nba", "total", "kalshi")
    assert k["mode"] == "live"
    assert (r["mode"], r["reason"]) == ("shadow", mr.VENUE_FIREWALL)


def test_firewall_cleared_by_allowlist(monkeypatch):
    cfg = get_settings()
    monkeypatch.setattr(cfg, "polymarket_us_live", False)
    monkeypatch.setattr(cfg, "polymarket_us_live_sectors", "nba")
    assert _row("nba", "total", "polymarket_us")["mode"] == "live"


def test_report_shape_and_route():
    rep = mr.mode_report(IN_SEASON)
    keys = {c.key for c in __import__("evmax.categories", fromlist=["x"]).all_categories()}
    assert {r["category"] for r in rep["rows"]} == keys
    assert "ligamx" in rep["shadow_leagues"]
    assert not any(r["venue"] == "polymarket_us" and r["category"].endswith("_props") for r in rep["rows"])
    res = TestClient(app).get("/api/modes")
    assert res.status_code == 200
    assert res.json()["rows"]
