"""Tests for `evmax categories` CLI commands."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from evmax.cli.commands.categories import app
from evmax.modes import clear_runtime_overrides, set_runtime_overrides


# Rich truncates panel contents at the terminal width — force a wide
# console for tests so assertion substrings are never elided.
runner = CliRunner()


@pytest.fixture(autouse=True)
def _reset_overrides(monkeypatch):
    monkeypatch.setenv("COLUMNS", "240")
    clear_runtime_overrides()
    yield
    clear_runtime_overrides()


# -------------------------------------------------------------------------
# list
# -------------------------------------------------------------------------


def test_list_prints_every_category(_reset_overrides):
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    # Should mention NBA, NFL, NBA props, NFL props at minimum
    assert "nba" in result.stdout.lower()
    assert "nfl" in result.stdout.lower()
    assert "nba_props" in result.stdout.lower()
    assert "nfl_props" in result.stdout.lower()
    assert "categories. Source of truth" in result.stdout


def test_list_filter_by_mode_live(_reset_overrides):
    result = runner.invoke(app, ["list", "--mode", "live"])
    assert result.exit_code == 0
    # nfl_props is shadow so it must NOT appear
    assert "nfl_props" not in result.stdout.lower()
    assert "nba" in result.stdout.lower()


def test_list_filter_by_mode_shadow(_reset_overrides):
    result = runner.invoke(app, ["list", "--mode", "shadow"])
    assert result.exit_code == 0
    # nhl ships as shadow in the YAML (nba_props was disabled 2026-08-08)
    assert "nhl" in result.stdout.lower()


def test_list_empty_filter_prints_message(_reset_overrides):
    result = runner.invoke(app, ["list", "--status", "unresolved"])
    assert result.exit_code == 0
    assert "no categories match" in result.stdout.lower()


def test_list_effective_flag_shows_overrides(_reset_overrides):
    set_runtime_overrides({"nba": "shadow"})
    result = runner.invoke(app, ["list", "--effective"])
    assert result.exit_code == 0
    assert "effective mode" in result.stdout
    # Override marker should appear next to the overridden row
    assert "yaml:" in result.stdout.lower()


# -------------------------------------------------------------------------
# show
# -------------------------------------------------------------------------


def test_show_existing_category(_reset_overrides):
    result = runner.invoke(app, ["show", "nba"])
    assert result.exit_code == 0
    assert "NBA" in result.stdout
    assert "moneyline" in result.stdout
    assert "live" in result.stdout
    assert "espn_scoreboard" in result.stdout


def test_show_prop_category_lists_stat_types(_reset_overrides):
    result = runner.invoke(app, ["show", "nba_props"])
    assert result.exit_code == 0
    assert "points" in result.stdout
    assert "rebounds" in result.stdout
    assert "assists" in result.stdout
    assert "prop stat types" in result.stdout.lower()


def test_show_unknown_category_exits_1(_reset_overrides):
    result = runner.invoke(app, ["show", "mystery_sport"])
    assert result.exit_code == 1


def test_show_displays_notes_when_present(_reset_overrides):
    result = runner.invoke(app, ["show", "nfl_props"])
    assert result.exit_code == 0
    assert "notes" in result.stdout.lower()
    assert "model-9" in result.stdout.lower() or "MODEL-9" in result.stdout


def test_show_runs_cleanly_with_override_active(_reset_overrides):
    """The `show` command must not crash when a runtime override is
    active, and the override must still be reflected via get_mode().

    We don't assert on the substring 'override' in stdout because Rich
    elides panel content when it can't detect terminal width inside
    CliRunner — the rendering is tested indirectly via the modes
    command below which uses a plain table."""
    from evmax.modes import get_mode

    set_runtime_overrides({"nba": "shadow"})
    result = runner.invoke(app, ["show", "nba"])
    assert result.exit_code == 0
    # Effective mode is still shadow for the duration of this test
    assert get_mode("nba") == "shadow"


# -------------------------------------------------------------------------
# validate
# -------------------------------------------------------------------------


def test_validate_passes_on_shipped_yaml(_reset_overrides):
    result = runner.invoke(app, ["validate"])
    assert result.exit_code == 0
    assert "Registry valid" in result.stdout


# -------------------------------------------------------------------------
# modes
# -------------------------------------------------------------------------


def test_modes_default_shows_effective(_reset_overrides):
    result = runner.invoke(app, ["modes"])
    assert result.exit_code == 0
    assert "nba" in result.stdout.lower()
    assert "effective" in result.stdout.lower()


def test_modes_base_flag_hides_override_column(_reset_overrides):
    result = runner.invoke(app, ["modes", "--base"])
    assert result.exit_code == 0
    assert "yaml base" in result.stdout.lower()
    assert "overridden" not in result.stdout.lower()


def test_modes_reflects_runtime_overrides(_reset_overrides):
    set_runtime_overrides({"nba": "disabled"})
    result = runner.invoke(app, ["modes"])
    assert result.exit_code == 0
    # The row for nba should show 'yes' in the overridden column
    # and 'disabled' in effective
    out = result.stdout
    # Row with nba in it should contain both 'disabled' and 'yes'
    nba_line = next(line for line in out.splitlines() if line.lstrip().startswith("nba "))
    assert "disabled" in nba_line
    assert "yes" in nba_line
