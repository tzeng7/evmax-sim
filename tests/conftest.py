"""Pytest configuration and shared fixtures."""

import pytest
from datetime import datetime, timezone


@pytest.fixture
def sample_datetime():
    return datetime(2026, 2, 22, 18, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _isolate_model_state(tmp_path, monkeypatch):
    """Never let a test rewrite production model state under data/models/.

    Reads still come from the repo's seeded state files, but any
    save_state() whose target is the real data/models/ directory is
    diverted to a per-test tmp dir. Tests that point _state_path at
    their own tmp_path (the normal pattern) are unaffected. This is the
    same discipline scripts/backtest_tennis_walkforward.py applies by
    no-op'ing save_state — without it, a stale efficiency/matchup/
    shot_quality state file plus one unstubbed NBA predict_pair triggers
    a live nba_api fetch that rewrites the seeded state on disk.

    Also redirects the nba_api circuit-breaker file so a test that trips
    the breaker can't leave data/models/.nba_api_breaker.json behind and
    poison real scans for the next 10 minutes.
    """
    from evmax.agents.models import _nba_freshness as freshness
    from evmax.agents.models import base as model_base

    real_dir = model_base.STATE_DIR.resolve()
    orig_save = model_base.ModelAgent.save_state

    def diverted_save(self):
        try:
            if self._state_path.resolve().parent == real_dir:
                self._state_path = tmp_path / self._state_path.name
        except OSError:
            pass
        return orig_save(self)

    monkeypatch.setattr(model_base.ModelAgent, "save_state", diverted_save)
    monkeypatch.setattr(freshness, "_BREAKER_PATH", tmp_path / ".nba_api_breaker.json")
