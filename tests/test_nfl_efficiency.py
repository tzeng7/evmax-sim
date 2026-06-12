"""Tests for NflEfficiencyModelAgent — opponent-adjusted EPA → win probability."""

from __future__ import annotations

import asyncio

import pytest

from evmax.agents.models.nfl_efficiency_agent import (
    NflEfficiencyModelAgent,
    NFL_ABBREV_TO_NAME,
    NFL_NICKNAME_TO_NAME,
    HOME_EDGE_PTS,
    PLAYS_PER_TEAM_GAME,
    SCORE_STDEV,
    MIN_GAMES,
    _normal_cdf,
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


def _team_stats(off: float, defn: float, gp: int = 17) -> dict:
    """Build a minimal team stats dict for testing predict_pair logic."""
    return {
        "abbrev": "XXX",
        "off_epa_adj": off,
        "def_epa_adj": defn,
        "off_success_rate": 0.45,
        "def_success_rate": 0.45,
        "off_explosive_rate": 0.10,
        "def_explosive_rate": 0.10,
        "rz_td_rate_off": 0.55,
        "rz_td_rate_def": 0.55,
        "gp": gp,
    }


def _agent_with(teams: dict[str, dict]) -> NflEfficiencyModelAgent:
    agent = NflEfficiencyModelAgent()
    agent._state = {"nfl": {"teams": teams, "fetched_at": "2026-05-01"}}
    return agent


# ── Lookup tables ──────────────────────────────────────────────────────────

class TestLookupTables:
    def test_all_32_teams_in_abbrev_map(self):
        assert len(NFL_ABBREV_TO_NAME) == 32

    def test_nickname_map_resolves_chiefs(self):
        assert NFL_NICKNAME_TO_NAME["chiefs"] == "kansas city chiefs"

    def test_nickname_map_resolves_49ers(self):
        # The 49ers full name is "san francisco 49ers"; rsplit(" ", 1)[-1] = "49ers"
        assert NFL_NICKNAME_TO_NAME["49ers"] == "san francisco 49ers"


# ── Sector gating ──────────────────────────────────────────────────────────

class TestSectorGating:
    def test_non_nfl_returns_none(self):
        agent = _agent_with({"kansas city chiefs": _team_stats(0.1, -0.05)})
        market, sharp = _pair("kansas city chiefs", "buffalo bills", sector="nba")
        assert asyncio.run(agent.predict_pair(market, sharp)) is None

    def test_empty_state_returns_none(self):
        agent = NflEfficiencyModelAgent()
        agent._state = {}
        market, sharp = _pair("kansas city chiefs", "buffalo bills")
        assert asyncio.run(agent.predict_pair(market, sharp)) is None


# ── Team resolution ────────────────────────────────────────────────────────

class TestTeamResolution:
    def test_full_name(self):
        agent = _agent_with({
            "kansas city chiefs": _team_stats(0.10, -0.05),
            "buffalo bills":      _team_stats(-0.05, 0.10),
        })
        market, sharp = _pair("kansas city chiefs", "buffalo bills")
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is not None

    def test_nickname_resolves(self):
        agent = _agent_with({
            "kansas city chiefs": _team_stats(0.10, -0.05),
            "buffalo bills":      _team_stats(-0.05, 0.10),
        })
        # Pinnacle/Kalshi sometimes use just nicknames
        market, sharp = _pair("chiefs", "bills")
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is not None

    def test_unknown_team_returns_none(self):
        agent = _agent_with({"kansas city chiefs": _team_stats(0.10, -0.05)})
        market, sharp = _pair("kansas city chiefs", "fake team")
        assert asyncio.run(agent.predict_pair(market, sharp)) is None


# ── Confidence gating on game count ───────────────────────────────────────

class TestConfidenceGate:
    def test_min_games_gate(self):
        # Both teams below MIN_GAMES → return None
        agent = _agent_with({
            "kansas city chiefs": _team_stats(0.10, -0.05, gp=MIN_GAMES - 1),
            "buffalo bills":      _team_stats(-0.05, 0.10, gp=MIN_GAMES - 1),
        })
        market, sharp = _pair("kansas city chiefs", "buffalo bills")
        assert asyncio.run(agent.predict_pair(market, sharp)) is None

    def test_one_team_below_min_games_blocks(self):
        agent = _agent_with({
            "kansas city chiefs": _team_stats(0.10, -0.05, gp=17),
            "buffalo bills":      _team_stats(-0.05, 0.10, gp=MIN_GAMES - 1),
        })
        market, sharp = _pair("kansas city chiefs", "buffalo bills")
        assert asyncio.run(agent.predict_pair(market, sharp)) is None

    def test_low_confidence_under_9_games(self):
        agent = _agent_with({
            "kansas city chiefs": _team_stats(0.10, -0.05, gp=7),
            "buffalo bills":      _team_stats(-0.05, 0.10, gp=7),
        })
        market, sharp = _pair("kansas city chiefs", "buffalo bills")
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is not None
        assert result.confidence == 0.55

    def test_high_confidence_at_full_season(self):
        agent = _agent_with({
            "kansas city chiefs": _team_stats(0.10, -0.05, gp=17),
            "buffalo bills":      _team_stats(-0.05, 0.10, gp=17),
        })
        market, sharp = _pair("kansas city chiefs", "buffalo bills")
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is not None
        assert result.confidence == 0.85


# ── Probability semantics ──────────────────────────────────────────────────

class TestProbabilitySemantics:
    def test_strong_home_beats_weak_away(self):
        """Top-tier net EPA at home should produce P(home) > 0.75."""
        agent = _agent_with({
            "good team": _team_stats(0.20, -0.10, gp=17),
            "bad team":  _team_stats(-0.20, 0.10, gp=17),
        })
        market, sharp = _pair("good team", "bad team")
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is not None
        assert result.true_prob_a > 0.75
        assert result.true_prob_b < 0.25
        assert result.true_prob_a + result.true_prob_b == pytest.approx(1.0)

    def test_even_teams_at_home_get_small_edge(self):
        """Identical teams should resolve to ~Φ(HOME_EDGE / σ) ≈ 0.56."""
        agent = _agent_with({
            "team a": _team_stats(0.0, 0.0, gp=17),
            "team b": _team_stats(0.0, 0.0, gp=17),
        })
        market, sharp = _pair("team a", "team b")
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is not None
        expected = _normal_cdf(HOME_EDGE_PTS / SCORE_STDEV)
        assert result.true_prob_a == pytest.approx(expected, abs=0.001)

    def test_probabilities_clamped_at_extremes(self):
        """No matter how lopsided, output must stay within [0.02, 0.98]."""
        agent = _agent_with({
            "monster": _team_stats(0.50, -0.50, gp=17),
            "potato":  _team_stats(-0.50, 0.50, gp=17),
        })
        market, sharp = _pair("monster", "potato")
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is not None
        assert 0.02 <= result.true_prob_a <= 0.98
        assert 0.02 <= result.true_prob_b <= 0.98

    def test_margin_calculation_matches_formula(self):
        """Sanity-check the projected margin against the explicit formula."""
        off_a, def_a = 0.10, -0.05
        off_b, def_b = -0.05, 0.10
        agent = _agent_with({
            "team a": _team_stats(off_a, def_a, gp=17),
            "team b": _team_stats(off_b, def_b, gp=17),
        })
        market, sharp = _pair("team a", "team b")
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is not None

        net_a = off_a - def_a
        net_b = off_b - def_b
        expected_margin = (net_a - net_b) * PLAYS_PER_TEAM_GAME + HOME_EDGE_PTS
        expected_prob = _normal_cdf(expected_margin / SCORE_STDEV)
        assert result.true_prob_a == pytest.approx(expected_prob, abs=0.001)


# ── Staleness guard (Gap-3) ────────────────────────────────────────────────

class TestNflEfficiencyStalenessGuard:
    """Prior-season EPA seed must not leak into in-season predictions once a
    new NFL season has started — mirrors the WNBA efficiency staleness guard.
    """

    def test_stale_seed_returns_none_during_season(self):
        from datetime import date
        import evmax.agents.models.nfl_efficiency_agent as mod
        agent = mod.NflEfficiencyModelAgent()
        agent._state = {"nfl": {
            "teams": {
                "kansas city chiefs": _team_stats(0.10, -0.05),
                "buffalo bills": _team_stats(0.05, -0.02),
            },
            "seasons_used": [2023, 2024],  # behind active 2025
        }}
        orig = mod.nfl_state_is_stale_for_today
        mod.nfl_state_is_stale_for_today = lambda s, today=date(2025, 11, 1): orig(s, today)
        try:
            market, sharp = _pair("kansas city chiefs", "buffalo bills")
            assert asyncio.run(agent.predict_pair(market, sharp)) is None
        finally:
            mod.nfl_state_is_stale_for_today = orig

    def test_fresh_seed_still_predicts(self):
        from datetime import date
        import evmax.agents.models.nfl_efficiency_agent as mod
        agent = mod.NflEfficiencyModelAgent()
        agent._state = {"nfl": {
            "teams": {
                "kansas city chiefs": _team_stats(0.10, -0.05),
                "buffalo bills": _team_stats(0.05, -0.02),
            },
            "seasons_used": [2024, 2025],
        }}
        orig = mod.nfl_state_is_stale_for_today
        mod.nfl_state_is_stale_for_today = lambda s, today=date(2025, 11, 1): orig(s, today)
        try:
            market, sharp = _pair("kansas city chiefs", "buffalo bills")
            assert asyncio.run(agent.predict_pair(market, sharp)) is not None
        finally:
            mod.nfl_state_is_stale_for_today = orig


# ── Update is no-op ────────────────────────────────────────────────────────

class TestUpdateIsNoOp:
    def test_update_does_not_raise_or_mutate(self):
        agent = _agent_with({"kansas city chiefs": _team_stats(0.10, -0.05)})
        before = dict(agent._state["nfl"]["teams"]["kansas city chiefs"])
        agent.update("kansas city chiefs", "buffalo bills", 27, 21, "nfl", "2025-09-08")
        after = dict(agent._state["nfl"]["teams"]["kansas city chiefs"])
        assert before == after
