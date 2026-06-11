"""Guards for the _isolate_model_state autouse fixture in conftest.py.

A pytest run must never rewrite the seeded model state under
data/models/ (efficiency/matchup/shot_quality were silently refreshed
with live nba_api data whenever the seed was older than the last
completed NBA game). These tests fail if the conftest diversion is
removed or broken.
"""

from __future__ import annotations

from evmax.agents.models import _nba_freshness as freshness
from evmax.agents.models import base as model_base
from evmax.agents.models.efficiency_agent import EfficiencyModelAgent


def test_save_state_is_diverted_away_from_production_dir(tmp_path):
    real_path = model_base.STATE_DIR / "efficiency_state.json"
    before = real_path.read_bytes()

    agent = EfficiencyModelAgent()
    agent._state["__isolation_sentinel__"] = "must-not-reach-disk"
    agent.save_state()

    assert real_path.read_bytes() == before, (
        "save_state() wrote to the production data/models/ directory — "
        "the _isolate_model_state conftest fixture is not diverting writes"
    )
    diverted = tmp_path / "efficiency_state.json"
    assert diverted.exists()
    assert "__isolation_sentinel__" in diverted.read_text()


def test_explicit_tmp_state_path_is_untouched_by_diversion(tmp_path):
    """The normal test pattern — pointing _state_path at tmp_path — must
    keep working exactly as before."""
    agent = EfficiencyModelAgent()
    agent._state_path = tmp_path / "my_state.json"
    agent._state = {"k": "v"}
    agent.save_state()
    assert (tmp_path / "my_state.json").read_text() == '{\n  "k": "v"\n}'


def test_nba_api_breaker_file_is_redirected():
    real_breaker = model_base.STATE_DIR / ".nba_api_breaker.json"
    assert not str(freshness._BREAKER_PATH).startswith(str(model_base.STATE_DIR))

    freshness.mark_nba_api_failure()
    try:
        assert not real_breaker.exists(), (
            "mark_nba_api_failure() wrote the breaker file into the real "
            "data/models/ directory"
        )
        assert freshness._BREAKER_PATH.exists()
    finally:
        freshness._reset_cache_for_tests()
