"""Tests for evmax.categories — registry parsing, validator, public API."""

from __future__ import annotations

from pathlib import Path

import pytest

from evmax.categories import (
    KNOWN_MODELS,
    KNOWN_RESOLVERS,
    CategorySpec,
    all_categories,
    categories_in_mode,
    get_category,
    reload_registry,
    validate_registry,
)
from evmax.models.market import MarketType


# -------------------------------------------------------------------------
# Registry content — verifies the as-shipped YAML
# -------------------------------------------------------------------------


def test_every_sector_series_map_key_has_registry_entry():
    """The validator runs at import, so if this file imports at all, it
    passed. But assert explicitly here too for future readers."""
    from evmax.clients.kalshi import SECTOR_SERIES_MAP

    reg_keys = {c.key for c in all_categories()}
    missing = set(SECTOR_SERIES_MAP.keys()) - reg_keys
    assert not missing, f"missing from registry: {missing}"


def test_every_registry_entry_has_sector_series_map_key():
    from evmax.clients.kalshi import SECTOR_SERIES_MAP

    reg_keys = {c.key for c in all_categories()}
    orphans = reg_keys - set(SECTOR_SERIES_MAP.keys())
    assert not orphans, f"orphan categories: {orphans}"


def test_all_game_categories_have_no_prop_stat_types():
    for spec in all_categories():
        if not spec.is_prop:
            assert not spec.prop_stat_types, (
                f"{spec.key} is a game category but has prop_stat_types"
            )


def test_all_prop_categories_have_prop_stat_types():
    props = [c for c in all_categories() if c.is_prop]
    assert props, "expected at least one prop category registered"
    for spec in props:
        assert spec.prop_stat_types, (
            f"{spec.key} is a prop category but prop_stat_types is empty"
        )


def test_every_referenced_model_is_known():
    for spec in all_categories():
        unknown = set(spec.models) - KNOWN_MODELS
        assert not unknown, f"{spec.key}: unknown models {unknown}"


def test_every_resolver_is_known():
    for spec in all_categories():
        assert spec.resolver in KNOWN_RESOLVERS, (
            f"{spec.key}: unknown resolver {spec.resolver!r}"
        )


# -------------------------------------------------------------------------
# Public API — get_category, categories_in_mode, all_categories
# -------------------------------------------------------------------------


def test_get_category_returns_typed_spec():
    spec = get_category("nba")
    assert isinstance(spec, CategorySpec)
    assert spec.key == "nba"
    assert MarketType.moneyline in spec.market_types


def test_get_category_raises_on_unknown():
    with pytest.raises(KeyError):
        get_category("nope")


def test_categories_in_mode_live_contains_nba():
    live = categories_in_mode("live")
    assert "nba" in live
    assert "nfl" in live


def test_nfl_props_is_shadow_by_default():
    """Stage 4 ships with nfl_props in shadow until MODEL-9 validates."""
    shadow = categories_in_mode("shadow")
    assert "nfl_props" in shadow
    assert get_category("nfl_props").mode == "shadow"


def test_categories_in_mode_rejects_bad_mode():
    with pytest.raises(ValueError):
        categories_in_mode("invalid")  # type: ignore[arg-type]


def test_nba_props_is_a_prop_category():
    spec = get_category("nba_props")
    assert spec.is_prop
    assert spec.market_types == (MarketType.player_prop,)
    assert "points" in spec.prop_stat_types


def test_nba_is_not_a_prop_category():
    assert not get_category("nba").is_prop


# -------------------------------------------------------------------------
# Validator — negative tests using a temporary broken YAML
#
# Each negative test writes a FULL-coverage YAML (every SECTOR_SERIES_MAP
# key present) with exactly one field corrupted. This way the
# missing-from-SECTOR_SERIES_MAP check doesn't fire first and mask the
# specific failure mode we're testing.
# -------------------------------------------------------------------------


def _base_entry(
    *,
    mode: str = "live",
    resolver: str = "espn_scoreboard",
    models: list[str] | None = None,
    market_types: list[str] | None = None,
    prop_stat_types: list[str] | None = None,
    extra: str = "",
) -> str:
    models = models or ["elo", "form", "poisson", "sharp"]
    market_types = market_types or ["moneyline", "spread", "total"]
    prop = ""
    if prop_stat_types is not None:
        prop = f"\n  prop_stat_types: [{', '.join(prop_stat_types)}]"
    return (
        f"  display_name: \"stub\"\n"
        f"  market_types: [{', '.join(market_types)}]{prop}\n"
        f"  models: [{', '.join(models)}]\n"
        f"  mode: {mode}\n"
        f"  resolver: {resolver}\n"
        f"  status: shipped\n"
        f"{extra}"
    )


def _full_yaml(override_key: str, override_body: str) -> str:
    """Build a YAML document that covers every SECTOR_SERIES_MAP key, with
    `override_key` replaced by `override_body`. Every other entry is a
    known-good stub using defaults.
    """
    from evmax.clients.kalshi import SECTOR_SERIES_MAP

    parts: list[str] = []
    for key in sorted(SECTOR_SERIES_MAP.keys()):
        if key == override_key:
            parts.append(f"{key}:\n{override_body}")
            continue
        if key.endswith("_props"):
            body = _base_entry(
                market_types=["player_prop"],
                prop_stat_types=["points"],
                models=["nba_props_cache"],
                resolver="espn_boxscore",
            )
        else:
            body = _base_entry()
        parts.append(f"{key}:\n{body}")
    return "\n".join(parts)


def _write_yaml(path: Path, body: str) -> Path:
    path.write_text(body)
    return path


def test_validator_catches_missing_fields(tmp_path):
    # Just one field — no market_types / models / etc.
    bad = tmp_path / "broken.yaml"
    _write_yaml(bad, _full_yaml("nba", "  display_name: 'NBA'\n"))
    with pytest.raises(ValueError, match="missing required fields"):
        reload_registry(bad)
    reload_registry()


def test_validator_catches_unknown_model(tmp_path):
    bad = tmp_path / "broken.yaml"
    _write_yaml(
        bad,
        _full_yaml("nba", _base_entry(models=["magic_ml"])),
    )
    reload_registry(bad)
    with pytest.raises(ValueError, match="unknown models"):
        validate_registry()
    reload_registry()


def test_validator_catches_unknown_resolver(tmp_path):
    bad = tmp_path / "broken.yaml"
    _write_yaml(
        bad,
        _full_yaml("nba", _base_entry(resolver="magic_tv")),
    )
    reload_registry(bad)
    with pytest.raises(ValueError, match="unknown resolver"):
        validate_registry()
    reload_registry()


def test_validator_catches_illegal_mode(tmp_path):
    bad = tmp_path / "broken.yaml"
    _write_yaml(
        bad,
        _full_yaml("nba", _base_entry(mode="halfway")),
    )
    with pytest.raises(ValueError, match="illegal mode"):
        reload_registry(bad)
    reload_registry()


def test_validator_catches_prop_without_stat_types(tmp_path):
    # nba_props with market_types=[player_prop] but no prop_stat_types
    bad = tmp_path / "broken.yaml"
    _write_yaml(
        bad,
        _full_yaml(
            "nba_props",
            _base_entry(
                market_types=["player_prop"],
                models=["nba_props_cache"],
                resolver="espn_boxscore",
                # intentionally omit prop_stat_types
            ),
        ),
    )
    reload_registry(bad)
    with pytest.raises(ValueError, match="empty prop_stat_types"):
        validate_registry()
    reload_registry()


def test_validator_catches_game_with_stat_types(tmp_path):
    bad = tmp_path / "broken.yaml"
    _write_yaml(
        bad,
        _full_yaml("nba", _base_entry(prop_stat_types=["points"])),
    )
    reload_registry(bad)
    with pytest.raises(ValueError, match="game category but has"):
        validate_registry()
    reload_registry()


def test_validator_catches_unknown_market_type(tmp_path):
    bad = tmp_path / "broken.yaml"
    _write_yaml(
        bad,
        _full_yaml("nba", _base_entry(market_types=["tiebreaker"])),
    )
    with pytest.raises(ValueError, match="unknown MarketType"):
        reload_registry(bad)
    reload_registry()


# -------------------------------------------------------------------------
# Season windows — is_in_season + SeasonWindow.contains
# -------------------------------------------------------------------------


def test_season_window_contains_simple_range():
    """Non-wrapping window (MLB Mar 20 → Nov 5)."""
    from datetime import date

    from evmax.categories import SeasonWindow

    mlb = SeasonWindow(start_month=3, start_day=20, end_month=11, end_day=5)
    assert mlb.contains(date(2026, 5, 23)) is True
    assert mlb.contains(date(2026, 3, 20)) is True   # start inclusive
    assert mlb.contains(date(2026, 11, 5)) is True   # end inclusive
    assert mlb.contains(date(2026, 3, 19)) is False  # one day before
    assert mlb.contains(date(2026, 11, 6)) is False  # one day after
    assert mlb.contains(date(2026, 1, 15)) is False  # deep winter


def test_season_window_wraps_year_end():
    """Wrapping window (NFL Sep 1 → Feb 15)."""
    from datetime import date

    from evmax.categories import SeasonWindow

    nfl = SeasonWindow(start_month=9, start_day=1, end_month=2, end_day=15)
    assert nfl.contains(date(2026, 9, 1)) is True    # start inclusive
    assert nfl.contains(date(2026, 12, 25)) is True  # late season
    assert nfl.contains(date(2026, 1, 10)) is True   # playoffs
    assert nfl.contains(date(2026, 2, 15)) is True   # end inclusive
    assert nfl.contains(date(2026, 5, 23)) is False  # offseason
    assert nfl.contains(date(2026, 8, 31)) is False  # day before preseason gate
    assert nfl.contains(date(2026, 2, 16)) is False  # day after Super Bowl


def test_is_in_season_year_round_categories_always_true():
    """Tennis, LoL, CS2, soccer have no season_window — always in season."""
    from datetime import date

    from evmax.categories import get_category, is_in_season

    for key in ("tennis", "lol", "cs2", "soccer"):
        spec = get_category(key)
        assert spec.season_window is None, (
            f"{key} should be year-round (no season_window) but has one"
        )
        # Both summer and dead of winter — always in season.
        assert is_in_season(key, date(2026, 1, 5)) is True
        assert is_in_season(key, date(2026, 7, 15)) is True


def test_seasonal_categories_have_windows():
    """The four sectors the user wants gated MUST have a season_window."""
    from evmax.categories import get_category

    for key in ("nfl", "nfl_props", "ncaab", "ncaaw"):
        assert get_category(key).season_window is not None, (
            f"{key} must have season_window to be gated out of season"
        )


def test_nfl_out_of_season_in_may():
    """Today is 2026-05-23 — NFL/NCAAB/NCAAW must be out of season."""
    from datetime import date

    from evmax.categories import is_in_season

    may_23 = date(2026, 5, 23)
    assert is_in_season("nfl", may_23) is False
    assert is_in_season("nfl_props", may_23) is False
    assert is_in_season("ncaab", may_23) is False
    assert is_in_season("ncaaw", may_23) is False
    # And the in-season ones for sanity
    assert is_in_season("baseball", may_23) is True
    assert is_in_season("wnba", may_23) is True


def test_is_in_season_unknown_category_raises():
    from evmax.categories import is_in_season

    with pytest.raises(KeyError):
        is_in_season("not_a_sector")


def test_in_season_keys_filters_correctly():
    from datetime import date

    from evmax.categories import in_season_keys

    may_23 = date(2026, 5, 23)
    kept = in_season_keys(["nfl", "baseball", "ncaab", "tennis"], today=may_23)
    assert kept == ["baseball", "tennis"]  # order preserved, off-season dropped


def test_season_window_str_format():
    from evmax.categories import SeasonWindow

    w = SeasonWindow(start_month=9, start_day=1, end_month=2, end_day=15)
    assert str(w) == "09-01 → 02-15"


# -------------------------------------------------------------------------
# Season-window YAML parser — negative tests
# -------------------------------------------------------------------------


def test_validator_catches_bad_season_window_mmdd(tmp_path):
    bad = tmp_path / "broken.yaml"
    _write_yaml(
        bad,
        _full_yaml(
            "nba",
            _base_entry(extra='  season_window: {start: "13-01", end: "02-15"}\n'),
        ),
    )
    with pytest.raises(ValueError, match="season_window"):
        reload_registry(bad)
    reload_registry()


def test_validator_catches_season_window_missing_end(tmp_path):
    bad = tmp_path / "broken.yaml"
    _write_yaml(
        bad,
        _full_yaml(
            "nba",
            _base_entry(extra='  season_window: {start: "09-01"}\n'),
        ),
    )
    with pytest.raises(ValueError, match="season_window missing fields"):
        reload_registry(bad)
    reload_registry()


def test_validator_accepts_omitted_season_window(tmp_path):
    """Categories without a season_window key parse to None (year-round)."""
    bad = tmp_path / "no_window.yaml"
    _write_yaml(bad, _full_yaml("nba", _base_entry()))
    reload_registry(bad)
    from evmax.categories import get_category as _gc
    assert _gc("nba").season_window is None
    reload_registry()


def test_validator_catches_disabled_market_type_not_in_market_types(tmp_path):
    bad = tmp_path / "broken.yaml"
    _write_yaml(
        bad,
        _full_yaml("nba", _base_entry(
            market_types=["moneyline", "spread"],
            extra="  disabled_market_types: [total]\n",
        )),
    )
    with pytest.raises(ValueError, match="disabled_market_types entry"):
        reload_registry(bad)
    reload_registry()


def test_validator_catches_disabled_shadow_overlap(tmp_path):
    bad = tmp_path / "broken.yaml"
    _write_yaml(
        bad,
        _full_yaml("nba", _base_entry(
            extra="  shadow_market_types: [total]\n  disabled_market_types: [total]\n",
        )),
    )
    with pytest.raises(ValueError, match="both"):
        reload_registry(bad)
    reload_registry()


def test_baseball_spec_has_totals_disabled():
    """Shipped YAML: baseball totals moved from shadow_market_types to
    disabled_market_types on 2026-06-10."""
    spec = get_category("baseball")
    assert "total" in spec.disabled_market_types
    assert "total" not in spec.shadow_market_types
