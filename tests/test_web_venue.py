"""Dashboard venue plumbing — _gap_to_dict carries venue + firewall mode."""

from __future__ import annotations

import dataclasses

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


def test_gap_to_dict_single_venue_has_no_options():
    """A single-venue gap serializes venue_options as None (no dropdown)."""
    d = _gap_to_dict(_gap("kalshi"), bankroll=500.0)
    assert d["venue_options"] is None


def test_gap_to_dict_serializes_venue_options(monkeypatch):
    """A dual-venue winner serializes each leg as a full row dict (with its own
    venue / market_id) and never recurses into nested options."""
    monkeypatch.setattr(get_settings(), "polymarket_us_live", True)
    winner = _gap("polymarket_us")
    alt = dataclasses.replace(
        _gap("kalshi"), market_id="kalshi:VEN-TEST", ev_pct=0.04, kelly_fraction=0.0,
    )
    winner = dataclasses.replace(winner, venue_options=[winner, alt])
    d = _gap_to_dict(winner, bankroll=500.0)
    opts = d["venue_options"]
    assert opts is not None and len(opts) == 2
    assert {o["venue"] for o in opts} == {"polymarket_us", "kalshi"}
    assert {o["market_id"] for o in opts} == {"polymarket_us:VEN-TEST", "kalshi:VEN-TEST"}
    # Each serialized option is a leaf — no nested dropdown payload.
    assert all(o["venue_options"] is None for o in opts)
