"""Tests for the ESPN weekly rankings refresher (scripts/refresh_tennis_rankings_espn.py).

Network is never hit — fetch_espn_rankings is monkeypatched. Covers:
  - Monday snapshot-date stamping
  - append idempotency (re-run = no-op) with a single batched save
  - store-key resolution: exact, hyphen variant, reversed Chinese name order,
    fuzzy with first-initial guard (no sibling contamination)
  - payload sanity guard (suspiciously short feed fails loudly)
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import refresh_tennis_rankings_espn as mod
from evmax.agents.models.tennis_ranking_trend_agent import TennisRankingTrendAgent


def _agent_with(history: dict) -> tuple[TennisRankingTrendAgent, list]:
    saves: list = []
    agent = TennisRankingTrendAgent.__new__(TennisRankingTrendAgent)
    agent._state = {"history": history}
    agent.save_state = lambda: saves.append(1)
    return agent, saves


class TestLatestReleaseMonday:
    def test_midweek_maps_to_monday(self):
        assert mod.latest_release_monday(date(2026, 6, 10)) == date(2026, 6, 8)

    def test_monday_maps_to_itself(self):
        assert mod.latest_release_monday(date(2026, 6, 8)) == date(2026, 6, 8)

    def test_sunday_maps_to_previous_monday(self):
        assert mod.latest_release_monday(date(2026, 6, 14)) == date(2026, 6, 8)


class TestResolveStoreKey:
    def test_exact_match(self):
        store = {"jannik sinner": []}
        assert mod.resolve_store_key("jannik sinner", store) == "jannik sinner"

    def test_hyphen_variant_maps_to_existing(self):
        store = {"felix auger aliassime": []}
        assert mod.resolve_store_key("felix auger-aliassime", store) == "felix auger aliassime"

    def test_reversed_two_token_name_maps(self):
        store = {"xinyu wang": []}
        assert mod.resolve_store_key("wang xinyu", store) == "xinyu wang"

    def test_fuzzy_with_matching_initial(self):
        store = {"daniel merida aguilar": [{"date": "2026-01-05", "rank": 100}]}
        assert mod.resolve_store_key("daniel merida", store) == "daniel merida aguilar"

    def test_sibling_with_different_initial_gets_new_key(self):
        # "erika andreeva" must NOT inherit "mirra andreeva"'s history.
        store = {"mirra andreeva": [{"date": "2026-01-05", "rank": 6}]}
        assert mod.resolve_store_key("erika andreeva", store) == "erika andreeva"

    def test_unknown_player_gets_new_key(self):
        assert mod.resolve_store_key("totally newplayer", {}) == "totally newplayer"


class TestRefresh:
    @pytest.fixture
    def fake_fetch(self, monkeypatch):
        payloads = {
            "atp": [("jannik sinner", 1), ("new face", 99)],
            "wta": [("aryna sabalenka", 1)],
        }
        monkeypatch.setattr(mod, "fetch_espn_rankings", lambda tour, **kw: payloads[tour])
        return payloads

    def test_appends_and_saves_once(self, fake_fetch):
        agent, saves = _agent_with({"jannik sinner": [{"date": "2026-06-01", "rank": 1}]})
        counts = mod.refresh(agent, "2026-06-08")
        assert counts == {"atp": 2, "wta": 1}
        assert len(saves) == 1
        snaps = agent._state["history"]["jannik sinner"]
        assert [s["date"] for s in snaps] == ["2026-06-01", "2026-06-08"]
        assert agent._state["history"]["new face"] == [{"date": "2026-06-08", "rank": 99}]

    def test_second_run_is_noop(self, fake_fetch):
        agent, saves = _agent_with({})
        mod.refresh(agent, "2026-06-08")
        counts = mod.refresh(agent, "2026-06-08")
        assert counts == {"atp": 0, "wta": 0}
        assert len(saves) == 1  # second run appended nothing → no extra save

    def test_dry_run_does_not_mutate_or_save(self, fake_fetch):
        agent, saves = _agent_with({})
        counts = mod.refresh(agent, "2026-06-08", dry_run=True)
        assert counts == {"atp": 2, "wta": 1}
        assert saves == []
        assert all(not snaps for snaps in agent._state["history"].values())


class TestFetchSanityGuard:
    def test_short_payload_raises(self, monkeypatch):
        class _Resp:
            status_code = 200
            def raise_for_status(self): pass
            def json(self):
                return {"rankings": [{"ranks": [
                    {"athlete": {"displayName": "Lone Player"}, "current": 1}
                ]}]}
        monkeypatch.setattr(mod.httpx, "get", lambda *a, **kw: _Resp())
        with pytest.raises(ValueError, match="only 1 usable rows"):
            mod.fetch_espn_rankings("atp")
