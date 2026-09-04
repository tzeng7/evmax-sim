"""Tests for ARCH-11 CLI mode-override flags on `evmax agents scan`.

Covers:
  - --shadow / --live / --disabled each install overrides correctly
  - Comma-separated values parse into multiple category overrides
  - Conflicting category (same key on two flags) exits with code 1
  - Overrides visible via evmax.modes.get_mode after CLI invocation
  - Empty / omitted flags produce no overrides

The scan cycle itself is stubbed out (no network, no model-state or
predictions.db writes) — we just verify that the mode-override hook
runs before any scan/persistence step so the rest of the pipeline sees
the right mode.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from evmax.agents.coordinator import AgentCoordinator, CycleResult
from evmax.cli.commands.agents import app
from evmax.modes import clear_runtime_overrides, get_mode


runner = CliRunner()


@pytest.fixture(autouse=True)
def _reset_overrides():
    clear_runtime_overrides()
    yield
    clear_runtime_overrides()


@pytest.fixture(autouse=True)
def _stub_scan_pipeline():
    """Replace the real scan cycle with an empty result.

    Without this every test here ran a full live cycle: Kalshi +
    Pinnacle fetches, and — whenever the seeded NBA state was older
    than the last completed game — live nba_api refreshes that rewrote
    data/models/{efficiency,matchup,shot_quality}_state.json on disk.
    The post-cycle maintenance/CLV steps are stubbed too so the tests
    never open predictions.db.
    """
    maint = MagicMock()
    maint.summary.return_value = "maintenance stubbed"
    maint.errors = []
    maint.warnings = []
    with (
        patch.object(AgentCoordinator, "run_cycle", new=AsyncMock(return_value=CycleResult())),
        patch("evmax.agents.cleanup.maintenance.run_maintenance", return_value=maint),
        patch("evmax.agents.cleanup.resolver.backfill_clv", return_value={"updated": 0, "avg_kalshi_clv": 0.0}),
    ):
        yield


def _invoke(*args):
    """Invoke `evmax agents scan` with a tiny sector list; the cycle
    itself is stubbed by _stub_scan_pipeline."""
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


def test_live_flag_can_force_disabled_to_live(_reset_overrides):
    """The YAML ships nba_props as disabled. --live should flip it to live."""
    # Baseline (with the autouse reset): mode reflects YAML
    assert get_mode("nba_props") == "disabled"
    result = _invoke("--live", "nba_props")
    assert result.exit_code == 0
    assert get_mode("nba_props") == "live"


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
    assert get_mode("nba_props") == "disabled"
    assert get_mode("nfl_props") == "shadow"   # re-opened for the 2026 season


def test_clv_backfill_summary_prints_without_error(_reset_overrides):
    """Regression: `scan`'s post-cycle CLV summary must read the SAME keys
    `backfill_clv()` actually returns (avg_kalshi_clv/avg_pinn_drift/n_pinn/
    n_kalshi), not a stale/renamed key — a mismatch throws inside the broad
    except Exception and gets silently swallowed as a console warning."""
    fake_result = {
        "updated": 3,
        "skipped": 1,
        "avg_pinn_drift": 1.5,
        "avg_kalshi_clv": -0.75,
        "n_pinn": 2,
        "n_kalshi": 3,
    }
    with patch("evmax.agents.cleanup.resolver.backfill_clv", return_value=fake_result):
        result = _invoke()
    assert result.exit_code == 0
    assert "Warning: CLV backfill failed" not in result.stdout
    assert "CLV backfilled: 3 bet(s)" in result.stdout
    assert "-0.8pp" in result.stdout or "-0.75pp" in result.stdout
