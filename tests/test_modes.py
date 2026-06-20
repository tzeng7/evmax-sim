"""Tests for evmax.modes — get_mode, runtime overrides, env-var overrides."""

from __future__ import annotations

import pytest

from evmax import modes
from evmax.modes import (
    clear_runtime_overrides,
    effective_modes,
    get_mode,
    is_disabled,
    is_live,
    is_shadow,
    reset_overrides_cache,
    set_runtime_overrides,
)


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """Reset all overrides before and after each test to avoid cross-talk."""
    monkeypatch.delenv("EVMAX_CATEGORY_MODES", raising=False)
    reset_overrides_cache()
    clear_runtime_overrides()
    yield
    reset_overrides_cache()
    clear_runtime_overrides()


# -------------------------------------------------------------------------
# Base mode (from YAML)
# -------------------------------------------------------------------------


def test_base_mode_returns_yaml_value():
    # The shipped YAML has nba=live, nfl_props=shadow
    assert get_mode("nba") == "live"
    assert get_mode("nfl_props") == "shadow"


def test_is_live_shadow_disabled_booleans():
    assert is_live("nba")
    assert not is_shadow("nba")
    assert not is_disabled("nba")

    assert is_shadow("nfl_props")
    assert not is_live("nfl_props")


# -------------------------------------------------------------------------
# Runtime overrides
# -------------------------------------------------------------------------


def test_runtime_override_wins_over_yaml():
    set_runtime_overrides({"nba": "shadow"})
    assert get_mode("nba") == "shadow"


def test_runtime_override_for_multiple_categories():
    set_runtime_overrides({"nba": "shadow", "nfl": "disabled"})
    assert get_mode("nba") == "shadow"
    assert get_mode("nfl") == "disabled"
    # Unaffected categories fall through
    assert get_mode("ncaab") == "live"


def test_runtime_override_clear_reverts_to_yaml():
    set_runtime_overrides({"nba": "disabled"})
    assert get_mode("nba") == "disabled"
    clear_runtime_overrides()
    assert get_mode("nba") == "live"


def test_runtime_override_rejects_illegal_mode():
    with pytest.raises(ValueError, match="illegal mode"):
        set_runtime_overrides({"nba": "maybe"})


# -------------------------------------------------------------------------
# Env-var overrides
# -------------------------------------------------------------------------


def test_env_var_override_wins_over_yaml(monkeypatch):
    monkeypatch.setenv("EVMAX_CATEGORY_MODES", '{"nba": "shadow"}')
    reset_overrides_cache()
    assert get_mode("nba") == "shadow"


def test_env_var_override_multiple(monkeypatch):
    monkeypatch.setenv(
        "EVMAX_CATEGORY_MODES",
        '{"nba": "shadow", "nfl_props": "live"}',
    )
    reset_overrides_cache()
    assert get_mode("nba") == "shadow"
    assert get_mode("nfl_props") == "live"


def test_env_var_rejects_invalid_json(monkeypatch):
    monkeypatch.setenv("EVMAX_CATEGORY_MODES", "not-json")
    reset_overrides_cache()
    with pytest.raises(ValueError, match="not valid JSON"):
        get_mode("nba")


def test_env_var_rejects_illegal_mode(monkeypatch):
    monkeypatch.setenv("EVMAX_CATEGORY_MODES", '{"nba": "halfway"}')
    reset_overrides_cache()
    with pytest.raises(ValueError, match="illegal mode"):
        get_mode("nba")


# -------------------------------------------------------------------------
# Precedence: runtime > env > yaml
# -------------------------------------------------------------------------


def test_runtime_beats_env(monkeypatch):
    monkeypatch.setenv("EVMAX_CATEGORY_MODES", '{"nba": "shadow"}')
    reset_overrides_cache()
    set_runtime_overrides({"nba": "disabled"})
    assert get_mode("nba") == "disabled"


def test_env_beats_yaml_when_no_runtime(monkeypatch):
    monkeypatch.setenv("EVMAX_CATEGORY_MODES", '{"nba": "disabled"}')
    reset_overrides_cache()
    assert get_mode("nba") == "disabled"


# -------------------------------------------------------------------------
# effective_modes() snapshot
# -------------------------------------------------------------------------


def test_effective_modes_includes_every_category():
    from evmax.categories import all_categories

    snap = effective_modes()
    assert set(snap.keys()) == {c.key for c in all_categories()}


def test_effective_modes_reflects_overrides():
    set_runtime_overrides({"nba": "disabled"})
    snap = effective_modes()
    assert snap["nba"] == "disabled"
    # And the unaffected ones stay at YAML base
    assert snap["ncaab"] == "live"


# -------------------------------------------------------------------------
# Per-market-type shadow downgrade (shadow_market_types)
# -------------------------------------------------------------------------


def test_shadow_market_types_downgrades_listed_type(monkeypatch):
    """When a category lists a market type in shadow_market_types, get_mode
    returns 'shadow' for that type even though the sector is live."""
    from evmax.categories import CategorySpec, MarketType

    fake = CategorySpec(
        key="testcat",
        display_name="Test",
        market_types=(MarketType.moneyline, MarketType.spread, MarketType.total),
        models=("sharp",),
        mode="live",
        resolver="none",
        status="wip",
        shadow_market_types=("total",),
    )
    monkeypatch.setattr("evmax.modes.get_category", lambda k: fake)

    assert get_mode("testcat", "moneyline") == "live"
    assert get_mode("testcat", "spread") == "live"
    assert get_mode("testcat", "total") == "shadow"
    # No market_type → sector default
    assert get_mode("testcat") == "live"


def test_shadow_market_types_inert_when_sector_already_shadow(monkeypatch):
    """If sector is shadow already, the downgrade is a no-op (still shadow)."""
    from evmax.categories import CategorySpec, MarketType

    fake = CategorySpec(
        key="testcat2",
        display_name="Test",
        market_types=(MarketType.moneyline, MarketType.total),
        models=("sharp",),
        mode="shadow",
        resolver="none",
        status="wip",
        shadow_market_types=("total",),
    )
    monkeypatch.setattr("evmax.modes.get_category", lambda k: fake)

    assert get_mode("testcat2", "moneyline") == "shadow"
    assert get_mode("testcat2", "total") == "shadow"


def test_runtime_override_beats_shadow_market_types(monkeypatch):
    """Runtime override is highest precedence — it applies uniformly across
    all market types and ignores shadow_market_types."""
    from evmax.categories import CategorySpec, MarketType

    fake = CategorySpec(
        key="testcat3",
        display_name="Test",
        market_types=(MarketType.moneyline, MarketType.total),
        models=("sharp",),
        mode="live",
        resolver="none",
        status="wip",
        shadow_market_types=("total",),
    )
    monkeypatch.setattr("evmax.modes.get_category", lambda k: fake)

    set_runtime_overrides({"testcat3": "disabled"})
    assert get_mode("testcat3", "moneyline") == "disabled"
    assert get_mode("testcat3", "total") == "disabled"  # override wins, not shadow


def test_wnba_shipped_yaml_has_total_and_spread_in_shadow():
    """The shipped categories.yaml keeps WNBA total AND spread in shadow while
    only ML is live. Spread was demoted 2026-06-19: it does not clear the CLV
    gate (model trails Kalshi-at-close on Brier by construction; kalshi_clv_pct
    is the +EV yardstick and the live sample was break-even). Regression guard
    so a future YAML edit doesn't silently re-promote either without an explicit
    decision via `evmax cleanup shadow clv`."""
    assert get_mode("wnba", "moneyline") == "live"
    assert get_mode("wnba", "spread") == "shadow"
    assert get_mode("wnba", "total") == "shadow"


# -------------------------------------------------------------------------
# Per-market-type hard disable (disabled_market_types)
# -------------------------------------------------------------------------


def test_disabled_market_types_drops_listed_type(monkeypatch):
    """A market type in disabled_market_types resolves to 'disabled' even
    when the sector is live; other types are untouched."""
    from evmax.categories import CategorySpec, MarketType

    fake = CategorySpec(
        key="testcat4",
        display_name="Test",
        market_types=(MarketType.moneyline, MarketType.spread, MarketType.total),
        models=("sharp",),
        mode="live",
        resolver="none",
        status="wip",
        disabled_market_types=("total",),
    )
    monkeypatch.setattr("evmax.modes.get_category", lambda k: fake)

    assert get_mode("testcat4", "moneyline") == "live"
    assert get_mode("testcat4", "spread") == "live"
    assert get_mode("testcat4", "total") == "disabled"
    # No market_type → sector default
    assert get_mode("testcat4") == "live"


def test_disabled_market_types_applies_under_shadow_base(monkeypatch):
    """Unlike shadow_market_types (inert when base mode is shadow), the
    disable applies regardless of base mode — a shadow sector still drops
    the disabled market type before persistence."""
    from evmax.categories import CategorySpec, MarketType

    fake = CategorySpec(
        key="testcat5",
        display_name="Test",
        market_types=(MarketType.moneyline, MarketType.total),
        models=("sharp",),
        mode="shadow",
        resolver="none",
        status="wip",
        disabled_market_types=("total",),
    )
    monkeypatch.setattr("evmax.modes.get_category", lambda k: fake)

    assert get_mode("testcat5", "moneyline") == "shadow"
    assert get_mode("testcat5", "total") == "disabled"


def test_baseball_shipped_yaml_disables_totals():
    """Regression guard for the 2026-06-10 decision: baseball totals are
    disabled (n=102 NO-side unders went -15.2% ROI, 8.7pp above Pinnacle
    close — stale-line selection, not model bias). ML/spread stay at the
    sector base mode."""
    assert get_mode("baseball", "total") == "disabled"
    assert get_mode("baseball", "moneyline") != "disabled"
    assert get_mode("baseball", "spread") != "disabled"
