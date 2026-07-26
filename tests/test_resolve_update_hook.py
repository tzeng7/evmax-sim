"""Regression + wiring tests for the resolve-time model-update hook.

Two layers:
  1. Library regression — running update_models_for_date against a real (but
     state-isolated) AgentCoordinator advances both the elo `last_updated` stamp
     and the form most-recent-record date for the fed game. This is the actual
     fix: it proves states stop freezing once the hook runs.
  2. CLI wiring — `evmax cleanup resolve` calls the hook by default, respects
     --no-update-models, and a hook failure does NOT break resolution.
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import date
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from evmax.agents.cleanup import model_updater
from evmax.cli.commands import cleanup as cleanup_cmd

runner = CliRunner()


def _empty_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE ev_predictions (
            id INTEGER PRIMARY KEY, event_id TEXT, sector TEXT, event_date TEXT
        )"""
    )
    return conn


def _isolated_coordinator(tmp_path):
    """A real coordinator whose elo/form/poisson state paths point at tmp_path.

    Reading from production data/models/*.json would make the date assertions
    non-deterministic; redirecting _state_path and reloading gives empty state.
    """
    from evmax.agents.coordinator import AgentCoordinator

    coord = AgentCoordinator(
        sectors=["baseball"], enable_models=True, respect_season_window=False
    )
    for agent in (coord.elo_agent, coord.form_agent, coord.poisson_agent):
        agent._state_path = tmp_path / f"{agent.name}_state.json"
        agent._state = {}
    return coord


def test_hook_advances_elo_and_form_state_dates(tmp_path):
    coord = _isolated_coordinator(tmp_path)
    scores = [{
        "home_name": "Yankees", "away_name": "Red Sox",
        "home_score": 5, "away_score": 3,
    }]

    with patch.object(model_updater, "fetch_completed_scores", return_value=scores), \
         patch.object(model_updater, "get_connection", return_value=_empty_conn()):
        result = asyncio.run(
            model_updater.update_models_for_date(
                ["baseball"], date(2026, 6, 9), coordinator=coord
            )
        )

    assert result.updated == 1

    # Elo: per-sector last_updated stamp advanced to the event date.
    assert coord.elo_agent._state["baseball"]["last_updated"] == "2026-06-09"

    # Form: the most-recent record for each team is the event date.
    form_state = coord.form_agent._state["baseball"]
    all_dates = [rec["date"] for recs in form_state.values() for rec in recs]
    assert all_dates, "form state did not record the game"
    assert max(all_dates) == "2026-06-09"


def _runner_returning(*returns):
    """asyncio.run side_effect that closes each coroutine and yields canned values.

    Closing the coroutine avoids 'coroutine was never awaited' RuntimeWarnings
    while still letting us assert how many times asyncio.run was invoked.
    """
    it = iter(returns)

    def _run(coro):
        if hasattr(coro, "close"):
            coro.close()
        return next(it)

    return _run


def test_resolve_runs_update_hook_by_default():
    resolve_ret = {"resolved": 0, "failed": 0, "unmatched": []}
    # `skipped` mirrors UpdateResult's applied-games-ledger field; the hook
    # renders it, so the stub must carry it.
    hook_ret = type("R", (), {"updated": 2, "skipped": 0})()
    run = _runner_returning(resolve_ret, hook_ret)

    with patch.object(cleanup_cmd.asyncio, "run", side_effect=run) as mock_run, \
         patch("evmax.agents.cleanup.resolver.backfill_clv", return_value={"updated": 0, "skipped": 0}):
        res = runner.invoke(cleanup_cmd.app, ["resolve", "--date", "2026-06-09"])

    assert res.exit_code == 0
    # Two asyncio.run calls = resolution + the update hook.
    assert mock_run.call_count == 2
    assert "Model state refreshed" in res.stdout


def test_resolve_skips_hook_when_disabled():
    run = _runner_returning({"resolved": 0, "failed": 0, "unmatched": []})

    with patch.object(cleanup_cmd.asyncio, "run", side_effect=run) as mock_run, \
         patch("evmax.agents.cleanup.resolver.backfill_clv", return_value={"updated": 0, "skipped": 0}):
        res = runner.invoke(
            cleanup_cmd.app, ["resolve", "--date", "2026-06-09", "--no-update-models"]
        )

    assert res.exit_code == 0
    # Only resolution ran — no second asyncio.run for the hook.
    assert mock_run.call_count == 1
    assert "Model state refreshed" not in res.stdout


def test_resolve_survives_hook_failure():
    # A model-update failure must NOT break resolution (cron depends on it).
    run = _runner_returning({"resolved": 1, "failed": 0, "unmatched": []})

    with patch.object(cleanup_cmd.asyncio, "run", side_effect=run), \
         patch("evmax.agents.cleanup.resolver.backfill_clv", return_value={"updated": 0, "skipped": 0}), \
         patch.object(model_updater, "update_models_for_date", side_effect=RuntimeError("boom")):
        res = runner.invoke(cleanup_cmd.app, ["resolve", "--date", "2026-06-09"])

    assert res.exit_code == 0
    assert "Warning: model update failed" in res.stdout
