"""Tests for evmax.modes — get_mode, runtime overrides, env-var overrides."""

from __future__ import annotations

from datetime import date

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
    # The shipped YAML has nba=live, nhl=shadow, nfl_props=disabled
    assert get_mode("nba") == "live"
    assert get_mode("nhl") == "shadow"
    assert get_mode("nfl_props") == "disabled"


def test_is_live_shadow_disabled_booleans():
    assert is_live("nba")
    assert not is_shadow("nba")
    assert not is_disabled("nba")

    assert is_shadow("nhl")
    assert not is_live("nhl")

    # Player props were disabled 2026-08-08.
    assert is_disabled("nfl_props")
    assert not is_live("nfl_props")
    assert not is_shadow("nfl_props")


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
    # Unaffected categories fall through. Pin an in-season date so the
    # season-window off-season downgrade (ncaab window 11-01..04-10) doesn't
    # turn this fall-through into shadow — this test is about override scoping,
    # not seasonality.
    assert get_mode("ncaab", today=date(2026, 1, 15)) == "live"


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
    # In-season date so the windowed ncaab fall-through stays at its YAML base.
    snap = effective_modes(today=date(2026, 1, 15))
    assert snap["nba"] == "disabled"
    # And the unaffected ones stay at YAML base
    assert snap["ncaab"] == "live"


# -------------------------------------------------------------------------
# Season-window off-season downgrade (live -> shadow outside the window)
# -------------------------------------------------------------------------


def test_windowed_live_sector_shadows_out_of_season():
    """A live category with a season_window logs as shadow OUTSIDE its window
    (pre-season / off-season) and self-restores to live once the window opens.
    This keeps a forced ``--sectors nfl`` scan from staking real bankroll on
    pre-season games where the models run on a frozen prior-season seed."""
    # NFL window is 09-04 .. 02-15.
    assert get_mode("nfl", today=date(2026, 8, 20)) == "shadow"   # pre-season
    assert get_mode("nfl", today=date(2026, 7, 1)) == "shadow"    # deep off-season
    assert get_mode("nfl", today=date(2026, 9, 14)) == "live"     # in season (Week 1)
    assert get_mode("nfl", today=date(2027, 1, 20)) == "live"     # in season (playoffs)


def test_off_season_downgrade_applies_per_market_type():
    """The off-season downgrade composes with the market-type refinements:
    a disabled market type stays disabled; other types ride the shadow base."""
    # NFL out of season: moneyline -> shadow (downgraded), and no NFL market
    # type is disabled, so spread/total also shadow.
    assert get_mode("nfl", "moneyline", today=date(2026, 8, 20)) == "shadow"
    assert get_mode("nfl", "spread", today=date(2026, 8, 20)) == "shadow"


def test_off_season_downgrade_yields_to_overrides():
    """Runtime + env overrides outrank the off-season downgrade — a deliberate
    force is honored even outside the window."""
    set_runtime_overrides({"nfl": "live"})
    assert get_mode("nfl", today=date(2026, 8, 20)) == "live"


def test_no_window_live_sector_never_downgrades():
    """A live category WITHOUT a season_window (e.g. tennis, wnba) is always in
    season, so the downgrade never fires regardless of date."""
    assert get_mode("tennis", today=date(2026, 8, 20)) == "live"
    assert get_mode("wnba", today=date(2026, 1, 1)) == "live"


def test_effective_modes_off_season_snapshot():
    """effective_modes() is date-aware: windowed live sectors show shadow out
    of season."""
    # Mid-August: NFL/NCAAB/NCAAW all out of season.
    snap = effective_modes(today=date(2026, 8, 20))
    assert snap["nfl"] == "shadow"
    assert snap["ncaab"] == "shadow"
    assert snap["ncaaw"] == "shadow"
    # No-window live sectors unaffected.
    assert snap["tennis"] == "live"


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


def test_wnba_shipped_yaml_disables_scanner_spread_and_total():
    """The shipped categories.yaml keeps WNBA ML live but DISABLES spread and
    total for the scanner (2026-07-19): those market_ids belong to the
    watch-listings anchored-entry trigger, which logs shadow rows via an
    explicit mode_resolver bypass. Under UNIQUE(market_id) freeze-on-first-
    insert, a scanner row would pre-empt the anchored entry at the wrong
    (scan-time) price — so the scanner must drop them entirely, not shadow
    them. Regression guard so a future YAML edit doesn't silently hand the
    market_id space back to the scanner without an explicit decision."""
    assert get_mode("wnba", "moneyline") == "live"
    assert get_mode("wnba", "spread") == "disabled"
    assert get_mode("wnba", "total") == "disabled"


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
