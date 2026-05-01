"""Tests for NflQbEloModelAgent — Elo with QB-rating layer for NFL."""

from __future__ import annotations

import asyncio

import pytest

from evmax.agents.models.nfl_qb_elo_agent import (
    NflQbEloModelAgent,
    apply_qb_elo_update,
    DEFAULT_ELO,
    HOME_ADVANTAGE_ELO,
    K_FACTOR,
    QB_UPDATE_SHARE,
    QB_DELTA_CLAMP,
)
from evmax.models.market import PredictionMarket, MarketSource
from evmax.models.odds import SharpOdds, SharpBook


def _pair(home: str, away: str, sector: str = "nfl") -> tuple[PredictionMarket, SharpOdds]:
    market = PredictionMarket(
        id="t", market_id="t", event_id=f"t_{home}_{away}",
        sector=sector, team_home=home, team_away=away,
        source=MarketSource.kalshi, yes_price=0.5, no_price=0.5,
    )
    sharp = SharpOdds(
        event_id=f"t_{home}_{away}", book=SharpBook.pinnacle, sector=sector,
        outcome_a_label=home, outcome_b_label=away,
        outcome_a_decimal=2.0, outcome_b_decimal=2.0,
        true_prob_a=0.5, true_prob_b=0.5,
    )
    return market, sharp


def _agent_with(state: dict) -> NflQbEloModelAgent:
    agent = NflQbEloModelAgent()
    agent._state = {"nfl": state}
    return agent


# ── Sector gating ──────────────────────────────────────────────────────────

class TestSectorGating:
    def test_non_nfl_returns_none(self):
        agent = _agent_with({
            "team_base": {"kansas city chiefs": 1550.0, "buffalo bills": 1540.0},
        })
        market, sharp = _pair("kansas city chiefs", "buffalo bills", sector="nba")
        assert asyncio.run(agent.predict_pair(market, sharp)) is None

    def test_empty_state_returns_none(self):
        agent = NflQbEloModelAgent()
        agent._state = {}
        market, sharp = _pair("kansas city chiefs", "buffalo bills")
        assert asyncio.run(agent.predict_pair(market, sharp)) is None


# ── Team / QB resolution ───────────────────────────────────────────────────

class TestResolution:
    def test_full_name_lookup(self):
        agent = _agent_with({
            "team_base": {"kansas city chiefs": 1550.0, "buffalo bills": 1540.0},
            "qb_deltas": {"P.Mahomes": 100.0, "J.Allen": 80.0},
            "current_starters": {"kansas city chiefs": "P.Mahomes", "buffalo bills": "J.Allen"},
            "qb_games": {"P.Mahomes": 100, "J.Allen": 90},
        })
        market, sharp = _pair("kansas city chiefs", "buffalo bills")
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is not None
        # Both elite QBs, both teams above-average → P(home) reasonable
        assert 0.40 < result.true_prob_a < 0.85

    def test_nickname_resolves(self):
        agent = _agent_with({
            "team_base": {"kansas city chiefs": 1550.0, "buffalo bills": 1540.0},
            "qb_deltas": {"P.Mahomes": 100.0, "J.Allen": 80.0},
            "current_starters": {"kansas city chiefs": "P.Mahomes", "buffalo bills": "J.Allen"},
            "qb_games": {"P.Mahomes": 100, "J.Allen": 90},
        })
        market, sharp = _pair("chiefs", "bills")  # Nickname-only
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is not None

    def test_qb_delta_lookup_case_insensitive(self):
        agent = _agent_with({
            "team_base": {"kansas city chiefs": 1500.0},
            "qb_deltas": {"P.Mahomes": 100.0},
        })
        # Stored as "P.Mahomes" — agent should find it even on mixed case
        delta = agent._qb_delta(agent._state["nfl"]["qb_deltas"], "p.mahomes")
        assert delta == 100.0


# ── QB-swap behavior (the whole point of this agent) ──────────────────────

class TestQbSwap:
    def test_starter_change_shifts_probability(self):
        """Swapping the starter to a much weaker QB should drop P(home) materially."""
        state = {
            "team_base": {"miami dolphins": 1520.0, "buffalo bills": 1540.0},
            "qb_deltas": {"T.Tagovailoa": 60.0, "Backup.QB": -80.0, "J.Allen": 90.0},
            "current_starters": {"miami dolphins": "T.Tagovailoa", "buffalo bills": "J.Allen"},
            "qb_games": {"T.Tagovailoa": 70, "Backup.QB": 5, "J.Allen": 90},
        }
        agent = _agent_with(state)
        market, sharp = _pair("miami dolphins", "buffalo bills")
        with_starter = asyncio.run(agent.predict_pair(market, sharp))

        # Now flip Miami's starter to the backup
        state["current_starters"]["miami dolphins"] = "Backup.QB"
        agent_swap = _agent_with(state)
        with_backup = asyncio.run(agent_swap.predict_pair(market, sharp))

        assert with_starter is not None and with_backup is not None
        # Tua → backup represents a 60 - (-80) = 140 Elo point swing for Miami
        # That should clearly show in P(home) — at least ~12pp shift.
        assert with_starter.true_prob_a - with_backup.true_prob_a > 0.10

    def test_unknown_starter_falls_back_to_zero_delta(self):
        """If a starter is in current_starters but missing from qb_deltas, treat
        as average (delta=0) rather than crashing or returning None."""
        state = {
            "team_base": {"miami dolphins": 1520.0, "buffalo bills": 1540.0},
            "qb_deltas": {"J.Allen": 90.0},  # Tua not in this map
            "current_starters": {"miami dolphins": "T.Tagovailoa", "buffalo bills": "J.Allen"},
            "qb_games": {"J.Allen": 90},
        }
        agent = _agent_with(state)
        market, sharp = _pair("miami dolphins", "buffalo bills")
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is not None
        # Tua treated as 0 delta → home_eff = 1520 + 0 + 48 = 1568,
        # away_eff = 1540 + 90 = 1630 → P(home) ≈ 0.41
        assert 0.35 < result.true_prob_a < 0.50

    def test_no_starter_assigned_lower_confidence(self):
        """Missing starter info should knock confidence down."""
        state = {
            "team_base": {"chicago bears": 1490.0, "detroit lions": 1560.0},
            "qb_deltas": {},
            "current_starters": {},  # Neither team has a known starter
            "qb_games": {},
        }
        agent = _agent_with(state)
        market, sharp = _pair("chicago bears", "detroit lions")
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is not None
        # Confidence tier when starter info missing
        assert result.confidence == 0.45


# ── Probability semantics ──────────────────────────────────────────────────

class TestProbabilitySemantics:
    def test_even_teams_no_qbs_get_home_edge(self):
        state = {
            "team_base": {"team a": 1500.0, "team b": 1500.0},
            "qb_deltas": {},
            "current_starters": {},
            "qb_games": {},
        }
        agent = _agent_with(state)
        market, sharp = _pair("team a", "team b")
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is not None
        # 48 Elo home advantage → P(home) ≈ 0.567
        expected = 1.0 / (1.0 + 10 ** (-HOME_ADVANTAGE_ELO / 400))
        assert result.true_prob_a == pytest.approx(expected, abs=0.001)

    def test_probabilities_clamped(self):
        """Even with extreme rating differentials the output stays in [0.02, 0.98]."""
        state = {
            "team_base": {"strong": 2000.0, "weak": 1000.0},
            "qb_deltas": {"Elite.QB": 200.0, "Bad.QB": -200.0},
            "current_starters": {"strong": "Elite.QB", "weak": "Bad.QB"},
            "qb_games": {"Elite.QB": 100, "Bad.QB": 100},
        }
        agent = _agent_with(state)
        market, sharp = _pair("strong", "weak")
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is not None
        assert 0.02 <= result.true_prob_a <= 0.98


# ── apply_qb_elo_update — the walk-forward Elo math ───────────────────────

class TestApplyUpdate:
    def test_winner_qb_gains_loser_qb_loses(self):
        state = {"team_base": {}, "qb_deltas": {}, "qb_games": {}}
        apply_qb_elo_update(
            state,
            home_team="kansas city chiefs", away_team="denver broncos",
            home_starter="P.Mahomes", away_starter="B.Nix",
            home_won=True,
        )
        # Both QBs should have moved relative to 0
        assert state["qb_deltas"]["P.Mahomes"] > 0
        assert state["qb_deltas"]["B.Nix"] < 0
        assert state["qb_games"]["P.Mahomes"] == 1
        assert state["qb_games"]["B.Nix"] == 1

    def test_qb_delta_clamped_after_many_wins(self):
        """A QB on a long winning streak should be clamped to QB_DELTA_CLAMP."""
        state = {"team_base": {}, "qb_deltas": {}, "qb_games": {}}
        for _ in range(200):  # 200 consecutive wins is more than enough to hit cap
            apply_qb_elo_update(
                state,
                home_team="winners", away_team="losers",
                home_starter="GOAT.QB", away_starter="Sad.QB",
                home_won=True,
            )
        assert state["qb_deltas"]["GOAT.QB"] <= QB_DELTA_CLAMP
        assert state["qb_deltas"]["Sad.QB"] >= -QB_DELTA_CLAMP

    def test_team_base_share_plus_qb_share_equals_one(self):
        """Total Elo delta should split correctly: QB_UPDATE_SHARE + (1-...) = 1."""
        state = {"team_base": {}, "qb_deltas": {}, "qb_games": {}}
        apply_qb_elo_update(
            state,
            home_team="A", away_team="B",
            home_starter="qb_a", away_starter="qb_b",
            home_won=True,
        )
        # Home team gains team_share of delta; home QB gains qb_share of delta.
        # The sum (excluding home advantage residual) should reconstruct full delta.
        team_gain = state["team_base"]["A"] - DEFAULT_ELO
        qb_gain = state["qb_deltas"]["qb_a"]
        total = team_gain + qb_gain
        # Equal & opposite for away
        team_loss = DEFAULT_ELO - state["team_base"]["B"]
        qb_loss = -state["qb_deltas"]["qb_b"]
        assert team_gain == pytest.approx(team_loss, abs=0.001)
        assert qb_gain == pytest.approx(qb_loss, abs=0.001)
        # Per-side total should equal full Elo delta (K-factor × prediction error)
        # Just sanity check it's positive and bounded by K_FACTOR
        assert 0 < total < K_FACTOR

    def test_missing_starter_only_updates_team_base(self):
        """If we don't know who the starter was, the QB rating dict shouldn't gain
        a None or empty-string key — only team_base should move."""
        state = {"team_base": {}, "qb_deltas": {}, "qb_games": {}}
        apply_qb_elo_update(
            state,
            home_team="A", away_team="B",
            home_starter=None, away_starter=None,
            home_won=True,
        )
        assert state["team_base"]["A"] > DEFAULT_ELO
        assert state["team_base"]["B"] < DEFAULT_ELO
        assert "" not in state["qb_deltas"]
        assert None not in state["qb_deltas"]
        assert state["qb_games"] == {}


# ── update() is no-op (state is seed-driven) ──────────────────────────────

class TestUpdateNoop:
    def test_update_does_not_mutate(self):
        agent = _agent_with({
            "team_base": {"kansas city chiefs": 1550.0, "buffalo bills": 1540.0},
            "qb_deltas": {"P.Mahomes": 100.0},
            "current_starters": {"kansas city chiefs": "P.Mahomes"},
            "qb_games": {"P.Mahomes": 100},
        })
        before = dict(agent._state["nfl"])
        agent.update("kansas city chiefs", "buffalo bills", 27, 24, "nfl", "2025-09-08")
        after = dict(agent._state["nfl"])
        assert before == after
