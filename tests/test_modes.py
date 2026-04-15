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
