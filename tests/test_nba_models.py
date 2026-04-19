"""Tests for NBA-specific model agents: efficiency, shot_quality, matchup, calibration, meta_model."""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from evmax.agents.models.efficiency_agent import EfficiencyModelAgent, _normal_cdf
from evmax.agents.models.shot_quality_agent import ShotQualityAgent
from evmax.agents.models.matchup_agent import MatchupAgent
from evmax.agents.models.calibration import ModelCalibrator
from evmax.agents.models.meta_model import MetaModel, _logit, _sigmoid
from evmax.agents.models.possession_sim_agent import PossessionSimAgent
from evmax.models.market import PredictionMarket, MarketSource
from evmax.models.odds import SharpOdds, SharpBook


def _make_pair(home: str, away: str, prob_a: float = 0.6) -> tuple[PredictionMarket, SharpOdds]:
    prob_b = 1.0 - prob_a
    market = PredictionMarket(
        id="t", market_id="t", event_id=f"t_{home}_{away}",
        sector="nba", team_home=home, team_away=away,
        source=MarketSource.kalshi, yes_price=prob_a, no_price=prob_b,
    )
    sharp = SharpOdds(
        event_id=f"t_{home}_{away}", book=SharpBook.pinnacle, sector="nba",
        outcome_a_label=home, outcome_b_label=away,
        outcome_a_decimal=1 / prob_a, outcome_b_decimal=1 / prob_b,
        true_prob_a=prob_a, true_prob_b=prob_b,
    )
    return market, sharp


# ── EfficiencyModelAgent ──────────────────────────────────────────────────


class TestEfficiencyModel:
    def test_normal_cdf_boundary(self):
        assert _normal_cdf(0.0) == pytest.approx(0.5, abs=0.001)
        assert _normal_cdf(6.0) == pytest.approx(1.0, abs=0.001)
        assert _normal_cdf(-6.0) == pytest.approx(0.0, abs=0.001)

    def test_normal_cdf_symmetry(self):
        assert _normal_cdf(1.0) + _normal_cdf(-1.0) == pytest.approx(1.0, abs=0.001)

    def test_predict_pair_non_nba_returns_none(self):
        agent = EfficiencyModelAgent()
        market, sharp = _make_pair("team_a", "team_b")
        market.sector = "soccer"
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is None

    def test_predict_pair_with_state(self):
        agent = EfficiencyModelAgent()
        agent._state = {
            "nba": {
                "league_avg_ortg": 114.0,
                "league_avg_drtg": 114.0,
                "league_avg_pace": 100.0,
                "teams": {
                    "thunder": {"ortg": 118.0, "drtg": 107.0, "pace": 100.0, "net": 11.0, "gp": 82, "full_name": "oklahoma city thunder"},
                    "suns": {"ortg": 114.0, "drtg": 113.0, "pace": 98.0, "net": 1.0, "gp": 82, "full_name": "phoenix suns"},
                },
                "fetched_at": "2026-04-18",
            }
        }
        market, sharp = _make_pair("thunder", "suns", 0.90)
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is not None
        assert result.true_prob_a > 0.70
        assert result.true_prob_b < 0.30
        assert result.confidence == 0.85

    def test_strong_vs_weak_produces_high_prob(self):
        agent = EfficiencyModelAgent()
        agent._state = {
            "nba": {
                "league_avg_ortg": 114.0, "league_avg_drtg": 114.0, "league_avg_pace": 100.0,
                "teams": {
                    "good": {"ortg": 120.0, "drtg": 105.0, "pace": 100.0, "net": 15.0, "gp": 82, "full_name": "good team"},
                    "bad": {"ortg": 108.0, "drtg": 120.0, "pace": 100.0, "net": -12.0, "gp": 82, "full_name": "bad team"},
                },
                "fetched_at": "2026-04-18",
            }
        }
        market, sharp = _make_pair("good", "bad", 0.85)
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is not None
        assert result.true_prob_a > 0.85

    def test_update_is_noop(self):
        agent = EfficiencyModelAgent()
        agent.update("a", "b", 100, 90, "nba")  # should not raise


# ── ShotQualityAgent ─────────────────────────────────────────────────────


class TestShotQuality:
    def test_regression_margin_overperformer(self):
        agent = ShotQualityAgent()
        agent._league_avgs = {"fg3_pct": 0.360, "rim_pct": 0.640}
        stats = {"fg3_pct": 0.400, "rim_pct": 0.640}  # 4% above avg from 3
        margin = agent._regression_margin(stats)
        assert margin > 0  # over-performing

    def test_regression_margin_underperformer(self):
        agent = ShotQualityAgent()
        agent._league_avgs = {"fg3_pct": 0.360, "rim_pct": 0.640}
        stats = {"fg3_pct": 0.320, "rim_pct": 0.640}  # 4% below avg
        margin = agent._regression_margin(stats)
        assert margin < 0

    def test_predict_non_nba_returns_none(self):
        agent = ShotQualityAgent()
        market, sharp = _make_pair("a", "b")
        market.sector = "soccer"
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is None


# ── MatchupAgent ─────────────────────────────────────────────────────────


class TestMatchup:
    def test_predict_non_nba_returns_none(self):
        agent = MatchupAgent()
        market, sharp = _make_pair("a", "b")
        market.sector = "tennis"
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is None

    def test_matchup_margin_capped(self):
        agent = MatchupAgent()
        agent._league_avgs = {"opp_pts_paint": 48.0, "fg3a": 35.0, "tov": 13.0, "blk": 5.0, "stl": 8.0}
        off_a = {"fg3a": 45, "tov": 8, "oreb": 12}
        def_b = {"blk": 3, "stl": 5, "opp_pts_paint": 55, "opp_pts_fb": 14, "drtg": 115}
        off_b = {"fg3a": 30, "tov": 18, "oreb": 8}
        def_a = {"blk": 7, "stl": 10, "opp_pts_paint": 42, "opp_pts_fb": 12, "drtg": 108}
        margin = agent._matchup_margin(off_a, def_b, off_b, def_a)
        assert -4.0 <= margin <= 4.0


# ── Calibration ──────────────────────────────────────────────────────────


class TestCalibration:
    def test_calibrate_without_training(self):
        cal = ModelCalibrator()
        cal._calibrations = {}
        assert cal.calibrate("elo", 0.5) == 0.5

    def test_calibrate_with_breakpoints(self):
        cal = ModelCalibrator()
        cal._calibrations = {
            "test": {
                "x": [0.0, 0.5, 1.0],
                "y": [0.1, 0.5, 0.9],
            }
        }
        assert cal.calibrate("test", 0.0) == pytest.approx(0.1)
        assert cal.calibrate("test", 0.5) == pytest.approx(0.5)
        assert cal.calibrate("test", 1.0) == pytest.approx(0.9)
        assert cal.calibrate("test", 0.25) == pytest.approx(0.3, abs=0.01)

    def test_retrain_insufficient_data(self):
        cal = ModelCalibrator()
        cal._calibrations = {}
        result = cal.retrain("test", [0.5] * 10, [1] * 10)
        assert result is False

    def test_retrain_sufficient_data(self):
        import numpy as np
        np.random.seed(42)
        probs = list(np.random.uniform(0.1, 0.9, 50))
        outcomes = [1 if p > 0.5 else 0 for p in probs]
        cal = ModelCalibrator()
        cal._calibrations = {}
        result = cal.retrain("test", probs, outcomes)
        assert result is True
        assert "test" in cal._calibrations


# ── MetaModel ────────────────────────────────────────────────────────────


class TestMetaModel:
    def test_logit_sigmoid_inverse(self):
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert _sigmoid(_logit(p)) == pytest.approx(p, abs=0.001)

    def test_predict_untrained_raises(self):
        meta = MetaModel()
        meta._trained = False
        with pytest.raises(RuntimeError):
            meta.predict({"sharp_logit": 0})

    def test_build_features(self):
        meta = MetaModel()
        features = meta.build_features(0.7, {"elo": 0.65, "form": 0.6})
        assert "sharp_logit" in features
        assert "elo_logit" in features
        assert "form_logit" in features
        assert features["sharp_prob"] == 0.7
        assert features["efficiency_logit"] == 0.0  # missing model → 0


# ── PossessionSimAgent ──────────────────────────────────────────────────


MOCK_EFF_STATE = {
    "league_avg_ortg": 114.5,
    "teams": {
        "thunder": {
            "ortg": 120.0, "drtg": 107.0, "pace": 101.0,
            "net": 13.0, "gp": 82, "full_name": "oklahoma city thunder",
            "tov_pct": 0.12,
        },
        "suns": {
            "ortg": 114.0, "drtg": 113.0, "pace": 98.0,
            "net": 1.0, "gp": 82, "full_name": "phoenix suns",
            "tov_pct": 0.14,
        },
        "bad": {
            "ortg": 106.0, "drtg": 118.0, "pace": 100.0,
            "net": -12.0, "gp": 82, "full_name": "bad team",
            "tov_pct": 0.16,
        },
    },
    "fetched_at": "2026-04-18",
}


class TestPossessionSim:
    def _make_agent(self, state=None) -> PossessionSimAgent:
        agent = PossessionSimAgent()
        agent._load_efficiency_state = lambda: state if state is not None else MOCK_EFF_STATE
        return agent

    def test_predict_non_nba_returns_none(self):
        agent = self._make_agent()
        market, sharp = _make_pair("thunder", "suns")
        market.sector = "soccer"
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is None

    def test_predict_no_data_returns_none(self):
        agent = self._make_agent(state={})
        market, sharp = _make_pair("thunder", "suns")
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is None

    def test_predict_unknown_team_returns_none(self):
        agent = self._make_agent()
        market, sharp = _make_pair("unknown_team", "suns")
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is None

    def test_strong_vs_weak_favors_strong(self):
        agent = self._make_agent()
        market, sharp = _make_pair("thunder", "bad", 0.85)
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is not None
        assert result.true_prob_a > 0.70

    def test_probabilities_sum_to_one(self):
        agent = self._make_agent()
        market, sharp = _make_pair("thunder", "suns", 0.75)
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is not None
        assert result.true_prob_a + result.true_prob_b == pytest.approx(1.0, abs=0.001)

    def test_deterministic_seeding(self):
        agent1 = self._make_agent()
        agent2 = self._make_agent()
        market, sharp = _make_pair("thunder", "suns", 0.75)
        r1 = asyncio.run(agent1.predict_pair(market, sharp))
        r2 = asyncio.run(agent2.predict_pair(market, sharp))
        assert r1.true_prob_a == r2.true_prob_a

    def test_confidence_high_gp(self):
        agent = self._make_agent()
        market, sharp = _make_pair("thunder", "suns", 0.75)
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result.confidence == 0.80

    def test_confidence_low_gp(self):
        low_gp_state = {
            **MOCK_EFF_STATE,
            "teams": {
                "thunder": {**MOCK_EFF_STATE["teams"]["thunder"], "gp": 30},
                "suns": {**MOCK_EFF_STATE["teams"]["suns"], "gp": 30},
            },
        }
        agent = self._make_agent(state=low_gp_state)
        market, sharp = _make_pair("thunder", "suns", 0.75)
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result.confidence == 0.65

    def test_notes_contain_sim_info(self):
        agent = self._make_agent()
        market, sharp = _make_pair("thunder", "suns", 0.75)
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert "sim=10000" in result.notes
        assert "margin=" in result.notes
        assert "total=" in result.notes

    def test_update_is_noop(self):
        agent = PossessionSimAgent()
        agent.update("a", "b", 100, 90, "nba")

    def test_insufficient_games_returns_none(self):
        low_gp_state = {
            **MOCK_EFF_STATE,
            "teams": {
                "thunder": {**MOCK_EFF_STATE["teams"]["thunder"], "gp": 10},
                "suns": MOCK_EFF_STATE["teams"]["suns"],
            },
        }
        agent = self._make_agent(state=low_gp_state)
        market, sharp = _make_pair("thunder", "suns")
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is None
