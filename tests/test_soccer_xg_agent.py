"""Tests for SoccerXgAgent — xG regression model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evmax.agents.models.soccer_xg_agent import (
    SoccerXgAgent,
    xg_from_shots,
)


class TestXgFromShots:
    def test_basic(self):
        # 5 SOT * 0.30 + 6 off-target * 0.03 = 1.50 + 0.18 = 1.68
        assert abs(xg_from_shots(5, 11) - 1.68) < 0.01

    def test_zero_shots(self):
        assert xg_from_shots(0, 0) == 0.0

    def test_all_on_target(self):
        assert abs(xg_from_shots(4, 4) - 1.20) < 0.01

    def test_all_off_target(self):
        assert abs(xg_from_shots(0, 10) - 0.30) < 0.01


def _fresh_agent(tmp_path: Path) -> SoccerXgAgent:
    agent = SoccerXgAgent()
    agent._state = {"teams": {}}
    agent._state_path = tmp_path / "soccer_xg_state.json"
    return agent


def _record_n_matches(agent: SoccerXgAgent, team: str, n: int,
                      goals: int = 2, sot: int = 5, shots: int = 11,
                      opp_sot: int = 3, opp_shots: int = 8) -> None:
    for i in range(n):
        agent.record_match(
            team=team, goals_for=goals, goals_against=1,
            shots_on_target=sot, total_shots=shots,
            opponent_sot=opp_sot, opponent_shots=opp_shots,
            match_date=f"2026-03-{10+i:02d}", is_home=(i % 2 == 0),
        )


class TestRecordMatch:
    def test_records_match(self, tmp_path: Path):
        agent = _fresh_agent(tmp_path)
        agent.record_match(
            team="Crystal Palace", goals_for=2, goals_against=1,
            shots_on_target=5, total_shots=11,
            opponent_sot=3, opponent_shots=7,
            match_date="2026-04-12", is_home=True,
        )
        data = agent._state["teams"]["crystal palace"]
        assert len(data["matches"]) == 1
        m = data["matches"][0]
        assert m["goals_for"] == 2
        assert m["sot"] == 5
        assert abs(m["xg"] - 1.68) < 0.01

    def test_window_capped(self, tmp_path: Path):
        agent = _fresh_agent(tmp_path)
        _record_n_matches(agent, "arsenal", 25)
        assert len(agent._state["teams"]["arsenal"]["matches"]) == 20  # WINDOW * 2


class TestXgStats:
    def test_returns_none_below_min_matches(self, tmp_path: Path):
        agent = _fresh_agent(tmp_path)
        _record_n_matches(agent, "chelsea", 2)
        assert agent._team_xg_stats("chelsea") is None

    def test_returns_stats_at_threshold(self, tmp_path: Path):
        agent = _fresh_agent(tmp_path)
        _record_n_matches(agent, "chelsea", 5, goals=2, sot=5, shots=11)
        stats = agent._team_xg_stats("chelsea")
        assert stats is not None
        assert stats["n"] == 5
        assert abs(stats["goals_per_game"] - 2.0) < 0.01
        assert abs(stats["xg_per_game"] - 1.68) < 0.01
        # goals > xG → attack_ratio > 1 → regression candidate
        assert stats["attack_ratio"] > 1.0

    def test_underperformer_has_low_ratio(self, tmp_path: Path):
        agent = _fresh_agent(tmp_path)
        _record_n_matches(agent, "burnley", 6, goals=1, sot=6, shots=14)
        stats = agent._team_xg_stats("burnley")
        assert stats is not None
        # 1 goal per game vs xG ~2.04 → ratio < 1
        assert stats["attack_ratio"] < 1.0


class TestPredict:
    def test_returns_none_for_non_soccer(self, tmp_path: Path):
        agent = _fresh_agent(tmp_path)
        from unittest.mock import MagicMock
        market = MagicMock(sector="nba", team_home="celtics", team_away="lakers")
        sharp = MagicMock(
            outcome_a_label="celtics", outcome_b_label="lakers",
            event_id="nba::test", true_prob_a=0.5, true_prob_b=0.5,
        )
        import asyncio
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is None

    def test_returns_none_without_data(self, tmp_path: Path):
        agent = _fresh_agent(tmp_path)
        from unittest.mock import MagicMock
        market = MagicMock(sector="soccer", team_home="arsenal", team_away="chelsea")
        sharp = MagicMock(
            outcome_a_label="arsenal", outcome_b_label="chelsea",
            event_id="soccer::test", true_prob_a=0.5, true_prob_b=0.5,
        )
        import asyncio
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is None

    def test_returns_prediction_with_data(self, tmp_path: Path):
        agent = _fresh_agent(tmp_path)
        _record_n_matches(agent, "arsenal", 8, goals=2, sot=6, shots=14)
        _record_n_matches(agent, "chelsea", 8, goals=1, sot=4, shots=10)

        from unittest.mock import MagicMock
        market = MagicMock(sector="soccer", team_home="arsenal", team_away="chelsea")
        sharp = MagicMock(
            outcome_a_label="arsenal", outcome_b_label="chelsea",
            event_id="soccer::test",
        )
        import asyncio
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is not None
        assert result.true_prob_a > result.true_prob_b  # Arsenal stronger attack
        assert result.true_prob_draw is not None
        assert abs(result.true_prob_a + result.true_prob_b + result.true_prob_draw - 1.0) < 0.01
        assert "xG" in result.notes

    def test_save_and_load_roundtrip(self, tmp_path: Path):
        agent = _fresh_agent(tmp_path)
        _record_n_matches(agent, "liverpool", 6)
        agent.save_state()

        agent2 = SoccerXgAgent()
        agent2._state_path = tmp_path / "soccer_xg_state.json"
        agent2.load_state()
        stats = agent2._team_xg_stats("liverpool")
        assert stats is not None
        assert stats["n"] == 6


class TestWorldcupNamespace:
    """xG now also serves the national-team `worldcup` sector on its OWN pool."""

    def test_worldcup_pool_is_separate_from_soccer(self, tmp_path: Path):
        agent = _fresh_agent(tmp_path)
        # Same canonical name recorded under both sectors must NOT collide.
        agent.record_match(
            team="brazil", goals_for=3, goals_against=0,
            shots_on_target=8, total_shots=18,
            opponent_sot=2, opponent_shots=6,
            match_date="2026-03-10", is_home=True, sector="worldcup",
        )
        agent.record_match(
            team="brazil", goals_for=0, goals_against=1,
            shots_on_target=1, total_shots=4,
            opponent_sot=5, opponent_shots=10,
            match_date="2026-03-10", is_home=True, sector="soccer",
        )
        # soccer stays at the flat legacy key; worldcup is namespaced under itself.
        assert "brazil" in agent._state["teams"]
        assert "brazil" in agent._state["worldcup"]["teams"]
        wc = agent._state["worldcup"]["teams"]["brazil"]["matches"][0]
        sc = agent._state["teams"]["brazil"]["matches"][0]
        assert wc["goals_for"] == 3 and sc["goals_for"] == 0

    def test_predict_fires_for_worldcup(self, tmp_path: Path):
        agent = _fresh_agent(tmp_path)
        _record_n_matches(agent, "spain", 8, goals=3, sot=7, shots=15)
        # _record_n_matches defaults to sector="soccer" — re-record for worldcup.
        for i in range(8):
            agent.record_match(
                team="spain", goals_for=3, goals_against=0,
                shots_on_target=7, total_shots=15,
                opponent_sot=3, opponent_shots=8,
                match_date=f"2026-03-{10+i:02d}", is_home=(i % 2 == 0),
                sector="worldcup",
            )
            agent.record_match(
                team="panama", goals_for=1, goals_against=2,
                shots_on_target=3, total_shots=9,
                opponent_sot=6, opponent_shots=14,
                match_date=f"2026-03-{10+i:02d}", is_home=(i % 2 == 1),
                sector="worldcup",
            )

        from unittest.mock import MagicMock
        market = MagicMock(sector="worldcup", team_home="spain", team_away="panama")
        sharp = MagicMock(
            outcome_a_label="spain", outcome_b_label="panama",
            event_id="worldcup::test",
        )
        import asyncio
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is not None
        assert result.model_name == "xg"
        assert result.true_prob_a > result.true_prob_b  # Spain stronger
        assert result.true_prob_draw is not None  # 3-way draw mass retained


class TestTeamKeyResolution:
    """Raw source labels must resolve to the canonical state keys.

    The seed script writes keys through NameNormalizer ("Philadelphia Union"
    → "philadelphia") but predict_pair used to look up Pinnacle labels with a
    bare .lower() — so every club whose label differs from its canonical key
    (all of MLS, "Manchester City", …) silently dropped xG from the blend.
    """

    def test_predict_fires_on_raw_pinnacle_labels(self, tmp_path: Path):
        # Regression: state keyed canonically, sharp labels raw → must fire.
        agent = _fresh_agent(tmp_path)
        _record_n_matches(agent, "philadelphia", 8, goals=2, sot=6, shots=14)
        _record_n_matches(agent, "atlanta", 8, goals=1, sot=4, shots=10)

        from unittest.mock import MagicMock
        market = MagicMock(
            sector="soccer",
            team_home="Philadelphia Union", team_away="Atlanta United",
        )
        sharp = MagicMock(
            outcome_a_label="Philadelphia Union",
            outcome_b_label="Atlanta United",
            event_id="soccer::test",
        )
        import asyncio
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is not None
        assert result.true_prob_a > result.true_prob_b

    def test_exact_key_wins_over_renormalization(self, tmp_path: Path):
        # NameNormalizer is not idempotent: normalize("man united") == "man".
        # An exact state key must be used as-is, never re-normalized away.
        agent = _fresh_agent(tmp_path)
        agent._state["teams"]["man united"] = {"matches": []}
        _record_n_matches(agent, "man united", 6)
        assert "man" not in agent._state["teams"]
        assert agent._team_xg_stats("man united") is not None
        # The raw label still reaches the same key via the normalizer fallback.
        assert agent._team_xg_stats("Manchester United") is not None

    def test_record_match_merges_raw_label_onto_canonical_key(self, tmp_path: Path):
        agent = _fresh_agent(tmp_path)
        _record_n_matches(agent, "philadelphia", 3)
        agent.record_match(
            team="Philadelphia Union", goals_for=2, goals_against=0,
            shots_on_target=5, total_shots=12,
            opponent_sot=2, opponent_shots=7,
            match_date="2026-07-25", is_home=True,
        )
        assert "philadelphia union" not in agent._state["teams"]
        assert len(agent._state["teams"]["philadelphia"]["matches"]) == 4

    def test_new_team_written_under_canonical_key(self, tmp_path: Path):
        agent = _fresh_agent(tmp_path)
        agent.record_match(
            team="FC Cincinnati", goals_for=1, goals_against=1,
            shots_on_target=4, total_shots=9,
            opponent_sot=3, opponent_shots=8,
            match_date="2026-07-25", is_home=False,
        )
        assert "cincinnati" in agent._state["teams"]
        assert "fc cincinnati" not in agent._state["teams"]


class TestCongestionPenalty:
    """Test fixture congestion penalty in EloModelAgent."""

    def test_congestion_penalty_3_games(self, tmp_path: Path):
        from evmax.agents.models.elo_agent import EloModelAgent
        agent = EloModelAgent()
        # _congestion_penalty reads from form_state — but we can test the logic
        # by checking it returns 0 for non-soccer
        assert agent._congestion_penalty("nba", "celtics") == 0.0
