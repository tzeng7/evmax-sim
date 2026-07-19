"""Tests for the tennis ranking-prior supplement (WS3, 2026-07-19).

merge_ranking_priors widens the surface model's 0.48-confidence ranking
fallback to players the weekly TA Elo leaderboard churned off; the seed
script's latest_rank_snapshots applies the recency filter that keeps
retirees out of the store.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import pytest

from evmax.agents.models.tennis_model_agent import TennisModelAgent, ranking_to_elo

_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.setattr("evmax.agents.models.base.STATE_DIR", tmp_path)
    a = TennisModelAgent()
    a._state_path = tmp_path / "tennis_surface_state.json"
    a._state = {}
    return a


def _load_seed_script():
    spec = importlib.util.spec_from_file_location(
        "seed_tennis_models", _ROOT / "scripts" / "seed_tennis_models.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["seed_tennis_models"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestMergeRankingPriors:
    """Insert-only supplement: leaderboard entries always win."""

    def test_inserts_only_absent_players(self, agent):
        agent.seed_rankings({"iga swiatek": 2, "aryna sabalenka": 1}, tour="wta")
        added = agent.merge_ranking_priors(
            {"iga swiatek": 5, "jelena ostapenko": 21}, tour="wta"
        )
        assert added == 1
        store = agent._state["wta_rankings"]
        # Leaderboard entry NOT overridden by the supplement's rank.
        assert store["iga swiatek"] == 2
        assert store["jelena ostapenko"] == 21

    def test_invalid_ranks_skipped(self, agent):
        added = agent.merge_ranking_priors(
            {"a": 0, "b": -3, "c": "not-a-rank", "": 5, "d": 12}, tour="atp"
        )
        assert added == 1
        assert agent._state["atp_rankings"] == {"d": 12}

    def test_merged_player_fires_ranking_prior_fallback(self, agent):
        # Player absent from every rating store but merged into the priors:
        # _get_rating must return the ranking-derived Elo, not DEFAULT_ELO.
        agent.merge_ranking_priors({"jelena ostapenko": 21}, tour="wta")
        assert agent.get_rating("jelena ostapenko", "hard") == pytest.approx(
            ranking_to_elo(21)
        )


class TestLatestRankSnapshots:
    """The seed-script recency filter feeding merge_ranking_priors."""

    def test_recent_player_included_stale_excluded(self):
        mod = _load_seed_script()
        history = {
            "active player": [
                {"date": "2026-05-01", "rank": 40},
                {"date": "2026-07-01", "rank": 35},
            ],
            "retired player": [
                {"date": "2025-09-01", "rank": 18},
                {"date": "2025-12-01", "rank": 25},
            ],
        }
        out = mod.latest_rank_snapshots(history, today=date(2026, 7, 19))
        # Active: latest snapshot (35), within 90d. Retired: 230d stale → out.
        assert out == {"active player": 35}

    def test_malformed_snapshots_skipped(self):
        mod = _load_seed_script()
        history = {
            "bad date": [{"date": "not-a-date", "rank": 10}],
            "no rank": [{"date": "2026-07-10"}],
            "empty": [],
            "ok": [{"date": "2026-07-10", "rank": 7}],
        }
        out = mod.latest_rank_snapshots(history, today=date(2026, 7, 19))
        assert out == {"ok": 7}
