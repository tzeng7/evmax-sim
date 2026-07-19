"""Tests for NhlXgModelAgent — MoneyPuck 5v5 xG → win probability."""

from __future__ import annotations

import asyncio

import pytest

from evmax.agents.models.nhl_xg_agent import (
    NhlXgModelAgent,
    NHL_ABBREV_TO_NAME,
    NHL_NICKNAME_TO_NAME,
    NHL_MONEYPUCK_ABBREV_ALIASES,
    HOME_EDGE_GOALS,
    GOAL_STDEV,
    MIN_5V5_PER_GAME,
    MIN_GAMES,
    LOW_CONF_GAMES,
    HIGH_CONF_GAMES,
    _normal_cdf,
)
from evmax.models.market import PredictionMarket, MarketSource
from evmax.models.odds import SharpOdds, SharpBook


def _pair(home: str, away: str, sector: str = "nhl") -> tuple[PredictionMarket, SharpOdds]:
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


def _team_stats(xgf: float, xga: float, gp: int = 60, abbrev: str = "XXX") -> dict:
    """Build a minimal team stats dict for testing predict_pair logic."""
    return {
        "abbrev": abbrev,
        "xgf_per_60": xgf,
        "xga_per_60": xga,
        "gf_per_60": xgf,    # secondary signal — tests just mirror xG
        "ga_per_60": xga,
        "gp": gp,
    }


def _agent_with(teams: dict[str, dict]) -> NhlXgModelAgent:
    agent = NhlXgModelAgent()
    agent._state = {
        "nhl": {
            "teams": teams,
            "league_avg_xg_per_60": 2.50,
            "fetched_at": "2026-05-02",
        }
    }
    return agent


# ── Lookup tables ──────────────────────────────────────────────────────────

class TestLookupTables:
    def test_all_32_teams_in_abbrev_map(self):
        assert len(NHL_ABBREV_TO_NAME) == 32

    def test_nickname_map_resolves_bruins(self):
        assert NHL_NICKNAME_TO_NAME["bruins"] == "boston bruins"

    def test_multiword_nickname_resolves_maple_leafs(self):
        assert NHL_NICKNAME_TO_NAME["maple leafs"] == "toronto maple leafs"

    def test_moneypuck_dotted_abbrev_aliases_to_standard(self):
        # MoneyPuck uses "L.A" / "T.B" — agent must accept those at lookup
        assert NHL_MONEYPUCK_ABBREV_ALIASES["L.A"] == "LAK"
        assert NHL_MONEYPUCK_ABBREV_ALIASES["T.B"] == "TBL"


# ── Sector gating ──────────────────────────────────────────────────────────

class TestSectorGating:
    def test_non_nhl_returns_none(self):
        agent = _agent_with({"boston bruins": _team_stats(2.7, 2.3)})
        market, sharp = _pair("boston bruins", "toronto maple leafs", sector="nba")
        assert asyncio.run(agent.predict_pair(market, sharp)) is None

    def test_empty_state_returns_none(self):
        agent = NhlXgModelAgent()
        agent._state = {}
        market, sharp = _pair("boston bruins", "toronto maple leafs")
        assert asyncio.run(agent.predict_pair(market, sharp)) is None


# ── Team resolution ────────────────────────────────────────────────────────

class TestTeamResolution:
    def test_full_name(self):
        agent = _agent_with({
            "boston bruins": _team_stats(2.7, 2.3),
            "toronto maple leafs": _team_stats(2.6, 2.5),
        })
        market, sharp = _pair("boston bruins", "toronto maple leafs")
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is not None

    def test_single_word_nickname_resolves(self):
        agent = _agent_with({
            "boston bruins": _team_stats(2.7, 2.3),
            "vegas golden knights": _team_stats(2.6, 2.5),
        })
        # Pinnacle/Kalshi sometimes use just "bruins"
        market, sharp = _pair("bruins", "knights")
        # "knights" resolves via last-word lookup → "vegas golden knights"
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is not None

    def test_multiword_nickname_resolves(self):
        agent = _agent_with({
            "toronto maple leafs": _team_stats(2.6, 2.5),
            "boston bruins": _team_stats(2.7, 2.3),
        })
        market, sharp = _pair("maple leafs", "bruins")
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is not None

    def test_abbreviation_resolves(self):
        agent = _agent_with({
            "boston bruins": _team_stats(2.7, 2.3),
            "toronto maple leafs": _team_stats(2.6, 2.5),
        })
        market, sharp = _pair("BOS", "TOR")
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is not None

    def test_moneypuck_dotted_abbrev_resolves(self):
        agent = _agent_with({
            "los angeles kings": _team_stats(2.6, 2.4),
            "tampa bay lightning": _team_stats(2.8, 2.2),
        })
        # MoneyPuck-style dotted variants must resolve via the alias map
        market, sharp = _pair("L.A", "T.B")
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is not None

    def test_unknown_team_returns_none(self):
        agent = _agent_with({"boston bruins": _team_stats(2.7, 2.3)})
        market, sharp = _pair("boston bruins", "nonexistent franchise")
        assert asyncio.run(agent.predict_pair(market, sharp)) is None


# ── Confidence gating on game count ────────────────────────────────────────

class TestConfidenceGate:
    def test_below_min_games_returns_none(self):
        agent = _agent_with({
            "boston bruins": _team_stats(2.7, 2.3, gp=MIN_GAMES - 1),
            "toronto maple leafs": _team_stats(2.6, 2.5, gp=MIN_GAMES - 1),
        })
        market, sharp = _pair("boston bruins", "toronto maple leafs")
        assert asyncio.run(agent.predict_pair(market, sharp)) is None

    def test_one_team_below_min_blocks(self):
        agent = _agent_with({
            "boston bruins": _team_stats(2.7, 2.3, gp=60),
            "toronto maple leafs": _team_stats(2.6, 2.5, gp=MIN_GAMES - 1),
        })
        market, sharp = _pair("boston bruins", "toronto maple leafs")
        assert asyncio.run(agent.predict_pair(market, sharp)) is None

    def test_low_confidence_below_low_conf_games(self):
        agent = _agent_with({
            "boston bruins": _team_stats(2.7, 2.3, gp=LOW_CONF_GAMES - 5),
            "toronto maple leafs": _team_stats(2.6, 2.5, gp=LOW_CONF_GAMES - 5),
        })
        market, sharp = _pair("boston bruins", "toronto maple leafs")
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is not None
        assert result.confidence == 0.55

    def test_high_confidence_at_full_season(self):
        agent = _agent_with({
            "boston bruins": _team_stats(2.7, 2.3, gp=HIGH_CONF_GAMES),
            "toronto maple leafs": _team_stats(2.6, 2.5, gp=HIGH_CONF_GAMES),
        })
        market, sharp = _pair("boston bruins", "toronto maple leafs")
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is not None
        assert result.confidence == 0.85


# ── Probability semantics ──────────────────────────────────────────────────

class TestProbabilitySemantics:
    def test_stronger_xg_team_at_home_wins(self):
        """Big xG edge at home should produce P(home) > 0.65."""
        agent = _agent_with({
            "good team": _team_stats(3.1, 2.0, gp=60),     # net +1.1/60
            "bad team":  _team_stats(2.1, 3.0, gp=60),     # net -0.9/60
        })
        market, sharp = _pair("good team", "bad team")
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is not None
        assert result.true_prob_a > 0.65
        assert result.true_prob_b < 0.35
        assert result.true_prob_a + result.true_prob_b == pytest.approx(1.0)

    def test_even_teams_at_home_get_small_edge(self):
        """Identical xG → home P = Φ(HOME_EDGE_GOALS / GOAL_STDEV) ≈ 0.535."""
        agent = _agent_with({
            "team a": _team_stats(2.5, 2.5, gp=60),
            "team b": _team_stats(2.5, 2.5, gp=60),
        })
        market, sharp = _pair("team a", "team b")
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is not None
        expected = _normal_cdf(HOME_EDGE_GOALS / GOAL_STDEV)
        assert result.true_prob_a == pytest.approx(expected, abs=0.001)

    def test_extreme_lopsided_clamped_at_98(self):
        """Probabilities must clamp into [0.02, 0.98]."""
        agent = _agent_with({
            "monster": _team_stats(5.0, 1.0, gp=60),
            "potato":  _team_stats(1.0, 5.0, gp=60),
        })
        market, sharp = _pair("monster", "potato")
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is not None
        assert 0.02 <= result.true_prob_a <= 0.98
        assert 0.02 <= result.true_prob_b <= 0.98

    def test_margin_calculation_matches_formula(self):
        """Sanity-check the projected margin against the explicit formula."""
        xgf_a, xga_a = 2.80, 2.30
        xgf_b, xga_b = 2.40, 2.60
        agent = _agent_with({
            "team a": _team_stats(xgf_a, xga_a, gp=60),
            "team b": _team_stats(xgf_b, xga_b, gp=60),
        })
        market, sharp = _pair("team a", "team b")
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is not None

        net_a = xgf_a - xga_a
        net_b = xgf_b - xga_b
        expected_margin = (net_a - net_b) * (MIN_5V5_PER_GAME / 60.0) + HOME_EDGE_GOALS
        expected_prob = _normal_cdf(expected_margin / GOAL_STDEV)
        assert result.true_prob_a == pytest.approx(expected_prob, abs=0.001)


# ── Update is no-op ────────────────────────────────────────────────────────

class TestUpdateIsNoOp:
    def test_update_does_not_mutate_state(self):
        agent = _agent_with({"boston bruins": _team_stats(2.7, 2.3)})
        before = dict(agent._state["nhl"]["teams"]["boston bruins"])
        agent.update("boston bruins", "toronto maple leafs", 4, 2, "nhl", "2025-12-01")
        after = dict(agent._state["nhl"]["teams"]["boston bruins"])
        assert before == after


# ── Generic Elo calibration (MODEL-2 NHL half) ─────────────────────────────

class TestGenericEloNhlCalibration:
    """NHL's generic EloModelAgent (evmax/agents/models/elo_agent.py) is a
    SEPARATE model from NhlXgModelAgent tested above — this class covers the
    K_FACTORS/HOME_ADVANTAGE_ELO calibration shipped 2026-07-18 via
    scripts/backtest_nhl_elo.py (cold-start walk-forward sweep, ranked on
    2023-24, confirmed on 2024-25, reported once on held-out 2025-26).
    """

    def test_nhl_elo_constants_calibrated(self):
        from evmax.agents.models.elo_agent import HOME_ADVANTAGE_ELO, K_FACTORS

        assert "nhl" in K_FACTORS
        assert "nhl" in HOME_ADVANTAGE_ELO
        # Small K befitting an ~82-game season (mirrors baseball's K=6 for a
        # 162-game season) — sanity range, not a re-assertion of the exact
        # sweep winner so a future re-sweep doesn't need a test edit for
        # every minor retune.
        assert 1.0 <= K_FACTORS["nhl"] <= 30.0
        assert HOME_ADVANTAGE_ELO["nhl"] >= 0.0

    def test_nhl_elo_constants_exact_sweep_winner(self):
        """Pins the specific 2026-07-18 sweep result (K=6, home_adv=48)."""
        from evmax.agents.models.elo_agent import HOME_ADVANTAGE_ELO, K_FACTORS

        assert K_FACTORS["nhl"] == 6.0
        assert HOME_ADVANTAGE_ELO["nhl"] == 48.0

    def test_nhl_elo_predicts_home_favorite(self):
        """Basic sanity check on the generic elo formula for nhl using the
        calibrated constants: a higher-rated home team should be favored."""
        from evmax.agents.models.elo_agent import EloModelAgent

        agent = EloModelAgent()
        agent._state = {
            "nhl": {
                "ratings": {"colorado avalanche": 1600.0, "san jose sharks": 1400.0},
                "game_counts": {"colorado avalanche": 50, "san jose sharks": 50},
                "season_games": {"colorado avalanche": 50, "san jose sharks": 50},
                "h2h": {},
            }
        }
        market, sharp = _pair("colorado avalanche", "san jose sharks")
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is not None
        assert result.true_prob_a > 0.5

    def test_nhl_elo_state_shipped_and_populated(self):
        """data/models/elo_state.json must carry a real nhl key post-seed."""
        import json
        import pathlib

        path = pathlib.Path(__file__).resolve().parents[1] / "data" / "models" / "elo_state.json"
        state = json.loads(path.read_text())
        assert "nhl" in state, "nhl key missing from elo_state.json — MODEL-2 gap not closed"
        ratings = state["nhl"]["ratings"]
        # Full NHL is 32 teams; a couple of extra keys are expected from
        # in-window franchise history (Arizona Coyotes relocated to Utah in
        # 2024; "Utah Hockey Club" renamed to "Utah Mammoth" in 2025).
        assert 30 <= len(ratings) <= 40
        assert all(1000.0 < r < 2000.0 for r in ratings.values())
