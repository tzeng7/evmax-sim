"""Per-sector Polymarket US venue firewall.

The venue firewall demotes Polymarket US gaps to shadow UNLESS the sector is
cleared. Clearing is per-sector (``polymarket_us_live_sectors``) mirroring the
Kalshi per-category mode registry, with ``polymarket_us_live`` as a master
switch that clears every sector at once. These tests pin the resolution logic
(settings) and its two effects: the dashboard mode badge (web) matches what
persistence will write.
"""

from __future__ import annotations

from evmax.agents.odds.ev_gap_agent import EVGap
from evmax.settings import Settings, get_settings
from evmax.web.app import _gap_to_dict


# ---------------------------------------------------------------------------
# settings.polymarket_us_sector_live — the single source of truth
# ---------------------------------------------------------------------------

def test_allowlist_parsing_normalizes_case_and_whitespace():
    s = Settings(polymarket_us_live=False, polymarket_us_live_sectors=" WNBA , Tennis ")
    assert s.polymarket_us_live_sector_set() == {"wnba", "tennis"}


def test_empty_allowlist_is_empty_set():
    s = Settings(polymarket_us_live=False, polymarket_us_live_sectors="")
    assert s.polymarket_us_live_sector_set() == set()


def test_sector_live_true_for_allowlisted_sector():
    s = Settings(polymarket_us_live=False, polymarket_us_live_sectors="wnba")
    assert s.polymarket_us_sector_live("wnba") is True


def test_sector_live_false_for_non_allowlisted_sector():
    s = Settings(polymarket_us_live=False, polymarket_us_live_sectors="wnba")
    assert s.polymarket_us_sector_live("tennis") is False


def test_sector_live_case_insensitive_on_query():
    s = Settings(polymarket_us_live=False, polymarket_us_live_sectors="wnba")
    assert s.polymarket_us_sector_live("WNBA") is True


def test_master_switch_clears_every_sector():
    s = Settings(polymarket_us_live=True, polymarket_us_live_sectors="")
    assert s.polymarket_us_sector_live("nba") is True
    assert s.polymarket_us_sector_live("tennis") is True


def test_none_sector_is_never_live_without_master():
    s = Settings(polymarket_us_live=False, polymarket_us_live_sectors="wnba")
    assert s.polymarket_us_sector_live(None) is False


def test_shipped_default_clears_wnba_only():
    """The committed baseline: wnba is Poly-live, other sectors are not."""
    s = Settings()  # reads defaults (no .env in test env)
    assert s.polymarket_us_sector_live("wnba") is True
    assert s.polymarket_us_sector_live("tennis") is False
    assert s.polymarket_us_sector_live("baseball") is False


# ---------------------------------------------------------------------------
# Web mode badge mirrors the per-sector firewall (regression on _gap_to_dict)
# ---------------------------------------------------------------------------

def _gap(sector: str, venue: str = "polymarket_us") -> EVGap:
    return EVGap(
        market_id=f"{venue}:VEN-{sector}",
        event_id=f"{sector}::2026-07-08::a_vs_b",
        sector=sector,
        yes_team="a",
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
        event_title="A vs B",
        venue=venue,
    )


def test_web_badge_wnba_poly_is_live_under_allowlist(monkeypatch):
    monkeypatch.setattr(get_settings(), "polymarket_us_live", False)
    monkeypatch.setattr(get_settings(), "polymarket_us_live_sectors", "wnba")
    d = _gap_to_dict(_gap("wnba"), bankroll=500.0)
    assert d["venue"] == "polymarket_us"
    assert d["mode"] == "live"


def test_web_badge_tennis_poly_stays_shadow_under_wnba_allowlist(monkeypatch):
    monkeypatch.setattr(get_settings(), "polymarket_us_live", False)
    monkeypatch.setattr(get_settings(), "polymarket_us_live_sectors", "wnba")
    d = _gap_to_dict(_gap("tennis"), bankroll=500.0)
    assert d["mode"] == "shadow"


# ---------------------------------------------------------------------------
# _venue_is_live — per-scan venue selection ∩ firewall (GAP: venue scoping)
# ---------------------------------------------------------------------------

from evmax.agents.coordinator import _venue_is_live  # noqa: E402


class _FakeSettings:
    def __init__(self, live_sectors):
        self._live = {s.lower() for s in live_sectors}

    def polymarket_us_sector_live(self, sector):
        return (sector or "").lower() in self._live


def test_venue_is_live_kalshi_always_when_no_selection():
    assert _venue_is_live("kalshi", "nba", None, _FakeSettings([])) is True


def test_venue_is_live_poly_gated_by_firewall():
    s = _FakeSettings(["wnba"])
    assert _venue_is_live("polymarket_us", "wnba", None, s) is True
    assert _venue_is_live("polymarket_us", "nba", None, s) is False


def test_selection_restricts_unselected_venue_out():
    """Selecting only PolyUS makes a Kalshi gap non-live (the user opted out of
    Kalshi for this scan)."""
    s = _FakeSettings(["wnba"])
    assert _venue_is_live("kalshi", "nba", {"polymarket_us"}, s) is False


def test_selection_cannot_override_firewall():
    """Picking a venue never forces an un-validated PolyUS sector live — the
    firewall still gates within the selection."""
    s = _FakeSettings([])  # no PolyUS sector cleared
    assert _venue_is_live("polymarket_us", "wnba", {"polymarket_us"}, s) is False


def test_both_selected_allows_each_subject_to_firewall():
    s = _FakeSettings(["wnba"])
    both = {"kalshi", "polymarket_us"}
    assert _venue_is_live("kalshi", "nba", both, s) is True
    assert _venue_is_live("polymarket_us", "wnba", both, s) is True
    assert _venue_is_live("polymarket_us", "nba", both, s) is False
