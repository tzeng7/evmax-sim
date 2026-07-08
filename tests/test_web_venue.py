"""Dashboard venue plumbing — _gap_to_dict carries venue + firewall mode."""

from __future__ import annotations

from evmax.agents.odds.ev_gap_agent import EVGap
from evmax.settings import get_settings
from evmax.web.app import _gap_to_dict


def _gap(venue: str = "kalshi") -> EVGap:
    return EVGap(
        market_id=f"{venue}:VEN-TEST",
        event_id="nba::2026-07-08::lakers_vs_warriors",
        sector="nba",
        yes_team="lakers",
        market_type="moneyline",
        kalshi_yes_price=0.45,
        sharp_true_prob=0.55,
        blended_true_prob=0.55,
        ev_pct=0.07,
        kelly_full=0.10,
        kelly_fraction=0.02,
        match_confidence=0.95,
        volume_usd=1000.0,
        spread_pct=0.02,
        event_title="Lakers vs Warriors",
        venue=venue,
    )


def test_gap_to_dict_carries_kalshi_venue():
    d = _gap_to_dict(_gap("kalshi"), bankroll=500.0)
    assert d["venue"] == "kalshi"


def test_gap_to_dict_polymarket_gap_shows_shadow_before_promotion(monkeypatch):
    monkeypatch.setattr(get_settings(), "polymarket_us_live", False)
    d = _gap_to_dict(_gap("polymarket_us"), bankroll=500.0)
    assert d["venue"] == "polymarket_us"
    # The mode badge must match what log_gaps will persist (venue firewall).
    assert d["mode"] == "shadow"


def test_gap_to_dict_polymarket_gap_live_after_promotion(monkeypatch):
    monkeypatch.setattr(get_settings(), "polymarket_us_live", True)
    d = _gap_to_dict(_gap("polymarket_us"), bankroll=500.0)
    assert d["mode"] == "live"
