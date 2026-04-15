"""Tests for ARCH-11 CLI mode-override flags on `evmax agents scan`.

Covers:
  - --shadow / --live / --disabled each install overrides correctly
  - Comma-separated values parse into multiple category overrides
  - Conflicting category (same key on two flags) exits with code 1
  - Overrides visible via evmax.modes.get_mode after CLI invocation
  - Empty / omitted flags produce no overrides

The scan loop itself hits the network and is not exercised here — we
just verify that the mode-override hook runs before any
scan/persistence step so the rest of the pipeline sees the right mode.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from evmax.cli.commands.agents import app
from evmax.modes import clear_runtime_overrides, get_mode


runner = CliRunner()


@pytest.fixture(autouse=True)
def _reset_overrides():
    clear_runtime_overrides()
    yield
    clear_runtime_overrides()


def _invoke(*args):
    """Invoke `evmax agents scan` with a tiny sector list so the rest of
    the pipeline runs (with network calls) but doesn't take forever."""
    return runner.invoke(app, ["scan", "--sectors", "nba", *args])


def test_shadow_flag_installs_override(_reset_overrides):
    result = _invoke("--shadow", "nba_props")
    assert result.exit_code == 0
    assert get_mode("nba_props") == "shadow"
    assert "Mode overrides for this run" in result.stdout
    assert "nba_props=shadow" in result.stdout


def test_disabled_flag_installs_override(_reset_overrides):
    result = _invoke("--disabled", "nhl")
    assert result.exit_code == 0
    assert get_mode("nhl") == "disabled"


def test_live_flag_can_force_shadow_to_live(_reset_overrides):
    """The YAML ships nfl_props as shadow. --live should flip it back."""
    # Baseline (with the autouse reset): mode reflects YAML
    assert get_mode("nfl_props") == "shadow"
    result = _invoke("--live", "nfl_props")
    assert result.exit_code == 0
    assert get_mode("nfl_props") == "live"


def test_multiple_categories_in_one_flag(_reset_overrides):
    result = _invoke("--shadow", "nba_props,nhl")
    assert result.exit_code == 0
    assert get_mode("nba_props") == "shadow"
    assert get_mode("nhl") == "shadow"


def test_multiple_flags_compose(_reset_overrides):
    result = _invoke("--shadow", "nba_props", "--disabled", "nhl", "--live", "nfl_props")
    assert result.exit_code == 0
    assert get_mode("nba_props") == "shadow"
    assert get_mode("nhl") == "disabled"
    assert get_mode("nfl_props") == "live"


def test_conflicting_category_rejects_with_exit_1(_reset_overrides):
    """Same category on --live and --shadow is a user error, not silently resolved."""
    result = _invoke("--shadow", "nba_props", "--live", "nba_props")
    assert result.exit_code == 1
    assert "more than one" in result.stdout


def test_no_flags_produces_no_override_message(_reset_overrides):
    result = _invoke()
    assert result.exit_code == 0
    assert "Mode overrides for this run" not in result.stdout
    # And the YAML defaults apply
    assert get_mode("nba") == "live"
    assert get_mode("nfl_props") == "shadow"
