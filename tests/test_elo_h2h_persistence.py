"""Regression tests for EloModelAgent head-to-head persistence.

These exist because `data/models/elo_state.json` was discovered in the wild
with the `h2h` key missing for soccer / nfl / baseball / lol — the side-effect
of seed runs that predated h2h tracking and never got re-run after. The tests
pin down two invariants:

1. An update() call records an h2h entry in the in-memory state dict.
2. save_state() → load_state() round-trips h2h intact.
"""

from __future__ import annotations

import json
from pathlib import Path

from evmax.agents.models.elo_agent import EloModelAgent


def _fresh_agent(tmp_path: Path) -> EloModelAgent:
    """An EloModelAgent with state pinned to a temp file and no preloaded data."""
    agent = EloModelAgent()
    agent._state = {}
    agent._state_path = tmp_path / "elo_state.json"
    return agent


class TestEloH2HPersistence:
    def test_update_records_h2h(self, tmp_path: Path) -> None:
        agent = _fresh_agent(tmp_path)
        agent.update("Team A", "Team B", 1.0, 0.0, sector="nba", event_date="2026-01-01")

        h2h = agent._state["nba"]["h2h"]
        assert len(h2h) == 1
        # Canonical key is alphabetical (team a::team b)
        key = next(iter(h2h))
        assert "::" in key
        rec = h2h[key]
        assert rec["games"] == 1
        assert rec["a_wins"] + rec["b_wins"] == 1

    def test_draws_do_not_record_h2h(self, tmp_path: Path) -> None:
        """Draws skip the h2h recorder — only decisive results count."""
        agent = _fresh_agent(tmp_path)
        agent.update("Arsenal", "Chelsea", 1.0, 1.0, sector="soccer", event_date="2026-01-01")
        assert agent._state["soccer"]["h2h"] == {}

    def test_save_and_load_roundtrip_preserves_h2h(self, tmp_path: Path) -> None:
        """save_state → load_state must not lose the h2h dict. This was the
        shape of the bug observed on disk: ratings/game_counts present, h2h
        silently missing."""
        a1 = _fresh_agent(tmp_path)
        a1.update("Alpha", "Beta", 1.0, 0.0, sector="lol", event_date="2026-01-01")
        a1.update("Alpha", "Beta", 1.0, 0.0, sector="lol", event_date="2026-01-02")
        a1.update("Beta", "Alpha", 1.0, 0.0, sector="lol", event_date="2026-01-03")
        a1.save_state()

        raw = json.loads((tmp_path / "elo_state.json").read_text())
        assert "h2h" in raw["lol"], "h2h key missing from persisted state"
        assert raw["lol"]["h2h"], "h2h dict was empty after updates"

        a2 = EloModelAgent()
        a2._state_path = tmp_path / "elo_state.json"
        a2.load_state()
        h2h = a2._state["lol"]["h2h"]
        # 3 games between the same pair → one key, 3 games total
        assert len(h2h) == 1
        rec = next(iter(h2h.values()))
        assert rec["games"] == 3
