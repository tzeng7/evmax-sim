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
from evmax.agents.models._nba_playoff_blend import (
    parse_team_dash_rows,
    gp_weighted_blend,
)
from evmax.clients.nba_season import nba_season_label
from evmax.models.market import PredictionMarket, MarketSource
from evmax.models.odds import SharpOdds, SharpBook


# NBA states must be stamped with the current season to price a game.
_SEASON = nba_season_label()


async def _async_true(*_args, **_kwargs) -> bool:
    """Stub that mimics evmax.agents.models._nba_freshness.state_is_fresh."""
    return True


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
                "season": _SEASON,
            }
        }
        market, sharp = _make_pair("thunder", "suns", 0.90)
        with patch("evmax.agents.models._nba_freshness.state_is_fresh", new=_async_true):
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
                "season": _SEASON,
            }
        }
        market, sharp = _make_pair("good", "bad", 0.85)
        with patch("evmax.agents.models._nba_freshness.state_is_fresh", new=_async_true):
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


# ── NBA playoff blend helpers ───────────────────────────────────────────


class TestPlayoffBlend:
    def test_parse_handles_empty_payload(self):
        assert parse_team_dash_rows(None) == {}
        assert parse_team_dash_rows({}) == {}
        assert parse_team_dash_rows({"resultSets": []}) == {}

    def test_parse_normalizes_two_word_keys(self):
        payload = {"resultSets": [{
            "headers": ["TEAM_NAME", "GP", "OFF_RATING"],
            "rowSet": [
                ["Portland Trail Blazers", 80, 110.0],
                ["LA Clippers", 80, 115.0],
                ["Phoenix Suns", 80, 113.0],
            ],
        }]}
        out = parse_team_dash_rows(payload)
        assert "trail blazers" in out
        assert "la clippers" in out
        assert "suns" in out

    def test_blend_falls_back_to_rs_when_no_playoffs(self):
        rs = {"thunder": {"GP": 82, "OFF_RATING": 118.0, "TEAM_NAME": "Oklahoma City Thunder"}}
        po = {}
        out = gp_weighted_blend(rs, po, ["OFF_RATING"])
        assert out["thunder"]["OFF_RATING"] == 118.0
        assert out["thunder"]["gp"] == 82
        assert out["thunder"]["po_gp"] == 0

    def test_blend_falls_back_to_po_when_no_rs(self):
        # Playoff-only row (off-season scenario where RS payload missing)
        rs = {}
        po = {"thunder": {"GP": 5, "OFF_RATING": 120.0, "TEAM_NAME": "Oklahoma City Thunder"}}
        out = gp_weighted_blend(rs, po, ["OFF_RATING"])
        assert out["thunder"]["OFF_RATING"] == 120.0
        assert out["thunder"]["rs_gp"] == 0

    def test_blend_weights_by_gp(self):
        rs = {"thunder": {"GP": 80, "OFF_RATING": 115.0, "TEAM_NAME": "OKC"}}
        po = {"thunder": {"GP": 4, "OFF_RATING": 125.0, "TEAM_NAME": "OKC"}}
        out = gp_weighted_blend(rs, po, ["OFF_RATING"])
        # Weighted mean: (115*80 + 125*4) / 84
        expected = (115.0 * 80 + 125.0 * 4) / 84
        assert out["thunder"]["OFF_RATING"] == pytest.approx(expected, abs=1e-6)
        assert out["thunder"]["gp"] == 84
        assert out["thunder"]["rs_gp"] == 80
        assert out["thunder"]["po_gp"] == 4

    def test_blend_drops_zero_gp_teams(self):
        rs = {"team_a": {"GP": 0, "OFF_RATING": 0, "TEAM_NAME": "x"}}
        po = {"team_a": {"GP": 0, "OFF_RATING": 0, "TEAM_NAME": "x"}}
        out = gp_weighted_blend(rs, po, ["OFF_RATING"])
        assert out == {}

    def test_blend_unions_keys_from_both_inputs(self):
        rs = {"a": {"GP": 82, "X": 1.0, "TEAM_NAME": "A"}}
        po = {"b": {"GP": 5, "X": 2.0, "TEAM_NAME": "B"}}
        out = gp_weighted_blend(rs, po, ["X"])
        assert set(out) == {"a", "b"}


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
        pytest.importorskip("sklearn")  # retrain() returns False without it
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
    "season": _SEASON,
}


class TestPossessionSim:
    @pytest.fixture(autouse=True)
    def _fresh_state(self, monkeypatch):
        """Keep the freshness guard off the network. Unpatched, state_is_fresh
        asks live ESPN for the last completed NBA game: off-season that is None
        (→ fresh), but once preseason games complete MOCK_EFF_STATE's fetched_at
        is behind it and every prediction abstains (CI broke 2026-10-04). The
        guard's own behavior is pinned by the two explicit-patch tests below,
        whose inner patch() overrides this default."""
        monkeypatch.setattr(
            "evmax.agents.models._nba_freshness.state_is_fresh", _async_true,
        )

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

    def test_predict_stale_state_abstains(self):
        """Family-2 guard: a stale efficiency state (fetched_at behind the last
        completed NBA game) must drop possession_sim from the blend, not size a
        live bet on frozen ratings. Mirrors EfficiencyAgent's freshness guard on
        the shared state."""
        async def _async_false(*_a, **_k):
            return False

        agent = self._make_agent()  # MOCK_EFF_STATE has teams + a fetched_at
        market, sharp = _make_pair("thunder", "suns")
        with patch("evmax.agents.models._nba_freshness.state_is_fresh", new=_async_false):
            result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is None

    def test_predict_fresh_state_produces_prediction(self):
        """With a fresh state the guard passes and a prediction is produced."""
        agent = self._make_agent()
        market, sharp = _make_pair("thunder", "suns")
        with patch("evmax.agents.models._nba_freshness.state_is_fresh", new=_async_true):
            result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is not None
        assert result.model_name == "possession_sim"

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

    def test_cover_probability_favorite(self):
        agent = self._make_agent()
        market, sharp = _make_pair("thunder", "suns", 0.75)
        asyncio.run(agent.predict_pair(market, sharp))
        event_id = sharp.event_id
        # Thunder are strong favorites — should cover small spreads often
        p_cover_3 = agent.cover_probability(event_id, -3.0)
        p_cover_15 = agent.cover_probability(event_id, -15.0)
        assert p_cover_3 is not None
        assert p_cover_15 is not None
        assert p_cover_3 > p_cover_15  # easier to cover small spread

    def test_cover_probability_unknown_event(self):
        agent = self._make_agent()
        assert agent.cover_probability("nonexistent", -5.0) is None

    def test_cover_probability_uses_12_5_sigma(self):
        """The NBA possession-sim cover_probability hard-codes sigma=12.5
        (bumped 11.5→12.5 on 2026-05-07). Must stay in lockstep with
        spread_distribution._SECTOR_SIGMA["nba"] — both layers feed the
        same blended cover prob in ev_gap_agent step 2a."""
        from evmax.models_ml.spread_distribution import _SECTOR_SIGMA
        import inspect
        from evmax.agents.models import possession_sim_agent
        src = inspect.getsource(possession_sim_agent.PossessionSimAgent.cover_probability)
        assert "sigma = 12.5" in src, (
            "possession_sim cover_probability must use sigma=12.5 to match "
            f"_SECTOR_SIGMA['nba']={_SECTOR_SIGMA['nba']}"
        )
        assert _SECTOR_SIGMA["nba"] == 12.5

    def test_total_probability(self):
        agent = self._make_agent()
        market, sharp = _make_pair("thunder", "suns", 0.75)
        asyncio.run(agent.predict_pair(market, sharp))
        event_id = sharp.event_id
        p_over_200 = agent.total_probability(event_id, 200.0, is_over=True)
        p_over_280 = agent.total_probability(event_id, 280.0, is_over=True)
        assert p_over_200 is not None
        assert p_over_280 is not None
        assert p_over_200 > p_over_280  # lower bar → more likely to go over

    def test_sigma_in_notes(self):
        agent = self._make_agent()
        market, sharp = _make_pair("thunder", "suns", 0.75)
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert "sigma=" in result.notes


class TestNBAPropsESPNFallback:
    """ESPN gamelog fallback path used when stats.nba.com is unreachable.

    Stats.nba.com aggressively blocks scrapers during playoffs. When that
    happens the existing nba_api pipe fails wholesale, the daily refresh
    saves an empty cache, and every prop scan logs zero rows. The ESPN path
    + cache-safety guard recover the pipeline.
    """

    @staticmethod
    def _espn_search_payload(player_id: str = "4065648") -> dict:
        return {
            "items": [{
                "id": player_id, "displayName": "Jayson Tatum",
                "type": "player", "sport": "basketball", "league": "nba",
            }]
        }

    @staticmethod
    def _espn_gamelog_payload() -> dict:
        # 6 events worth of stats — enough to clear _MIN_GAMES=5.
        # labels: ['MIN','FG','FG%','3PT','3P%','FT','FT%','REB','AST','BLK','STL','PF','TO','PTS']
        rows = [
            ["35", "10-20", "50.0", "3-7", "42.9", "5-6", "83.3", "8", "5", "1", "2", "3", "2", "28"],
            ["32", "8-18",  "44.4", "2-5", "40.0", "4-4", "100",  "7", "6", "0", "1", "2", "3", "22"],
            ["38", "12-22", "54.5", "4-9", "44.4", "6-7", "85.7", "9", "8", "1", "1", "1", "1", "34"],
            ["30", "9-19",  "47.4", "1-6", "16.7", "3-3", "100",  "11","4", "2", "1", "4", "2", "22"],
            ["36", "11-21", "52.4", "3-8", "37.5", "7-8", "87.5", "6", "7", "0", "2", "3", "3", "32"],
            ["33", "10-19", "52.6", "2-6", "33.3", "4-5", "80.0", "10","5", "1", "1", "2", "1", "26"],
        ]
        events_meta = {
            f"e{i}": {"gameDate": f"2026-04-{20-i:02d}T00:00:00Z",
                      "team": {"abbreviation": "BOS"}}
            for i in range(6)
        }
        return {
            "labels": ["MIN","FG","FG%","3PT","3P%","FT","FT%","REB","AST","BLK","STL","PF","TO","PTS"],
            "events": events_meta,
            "seasonTypes": [{
                "displayName": "2025-26 Postseason",
                "categories": [{
                    "displayName": "Conference Quarterfinals",
                    "events": [{"eventId": f"e{i}", "stats": rows[i]} for i in range(6)],
                }],
            }],
        }

    def test_fetch_via_espn_returns_player_dict(self):
        from evmax.clients import nba_props_cache as npc

        npc._espn_id_cache.clear()

        def fake_get(url, params=None, **kwargs):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if "search" in url:
                resp.json = MagicMock(return_value=self._espn_search_payload())
            else:
                resp.json = MagicMock(return_value=self._espn_gamelog_payload())
            return resp

        with patch("httpx.get", side_effect=fake_get):
            data = npc._fetch_player_via_espn_sync("jayson_tatum")

        assert data is not None
        assert data["team"] == "BOS"
        assert data["n_games"] == 6
        # Most recent date first (April 20 → April 15)
        assert data["stats"]["PTS"][0] == 28.0
        assert data["stats"]["FG3M"][0] == 3.0  # parsed from "3-7"
        assert len(data["stats"]["MIN"]) == 6
        assert data["source"] == "espn"

    def test_fetch_via_espn_returns_none_when_search_misses(self):
        from evmax.clients import nba_props_cache as npc

        npc._espn_id_cache.clear()

        def fake_get(url, params=None, **kwargs):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json = MagicMock(return_value={"items": []})
            return resp

        with patch("httpx.get", side_effect=fake_get):
            assert npc._fetch_player_via_espn_sync("unknown_player") is None
        # negative cache entry written
        assert npc._espn_id_cache.get("unknown_player") == ""

    def test_refresh_keeps_prior_cache_when_fetch_returns_empty(self, tmp_path, monkeypatch):
        """Death-spiral guard: empty fetch must not clobber a non-empty cache."""
        from evmax.clients import nba_props_cache as npc

        # Redirect cache + reset memo
        cache_path = tmp_path / "nba_props_cache.json"
        monkeypatch.setattr(npc, "_CACHE_PATH", cache_path)
        npc._mem_cache = None
        npc._mem_cache_time = 0

        # Seed prior cache with one player
        prior = {
            "fetched_at": 1.0,  # ancient → triggers refresh path, but still loadable any-age
            "date": "2026-05-01",
            "players": {"some_player": {"player_name": "Some Player", "n_games": 12, "stats": {}}},
            "team_stats": {},
            "league_avg": {},
            "schedule": [],
        }
        cache_path.write_text(json.dumps(prior))

        async def fake_fetch(_player_names=None):
            return {}  # network blocked, nothing fetched

        # Post-2026-05-10 the cache no longer fetches team-stats /
        # schedules — those were inputs to the deleted L15 model. Only
        # _fetch_player_stats_async needs stubbing to prove the
        # keep-prior-cache logic.
        monkeypatch.setattr(npc, "_fetch_player_stats_async", fake_fetch)

        n = asyncio.run(npc.refresh_props_cache(force=True, player_names=["new_player"]))

        # Returned the prior count, did NOT overwrite the cache file.
        assert n == 1
        on_disk = json.loads(cache_path.read_text())
        assert "some_player" in on_disk["players"]
        assert on_disk["fetched_at"] == 1.0  # untouched


# ── Season rollover (2026-10-13) ─────────────────────────────────────────
# The NBA stack shipped with season="2025-26" hard-coded at every fetch. At
# the 2026-27 rollover that kept fetching last season's complete stats
# (gp 82-105) and fired them at full confidence into new-season games. These
# pin the fix: fetches follow the current season, states carry the season,
# and another season's (or an unstamped) state never prices a game — not even
# when the refetch fails, which it does until opening night (nba_api returns
# no regular-season rows for a season that has not started).

from datetime import date as _date, datetime as _datetime, timezone as _tz

from evmax.clients.nba_season import nba_season_start_year

_START = nba_season_start_year()
_PRIOR_SEASON = f"{_START - 1}-{_START % 100:02d}"
_PRIOR_SEASON_GAME = _datetime(_START, 2, 15, 12, tzinfo=_tz.utc)
_CURRENT_SEASON_GAME = _datetime(_START + 1, 2, 15, 12, tzinfo=_tz.utc)


class _FakeTeamDash:
    """Stands in for nba_api's LeagueDashTeamStats (no network): records the
    season each call requests and serves ``rows`` for the regular season."""

    calls: list = []
    rows: list = []
    headers = ["TEAM_ID", "TEAM_NAME", "GP", "OFF_RATING", "DEF_RATING", "PACE",
               "NET_RATING", "EFG_PCT", "TS_PCT", "TM_TOV_PCT", "OREB_PCT"]

    def __init__(self, **kwargs):
        _FakeTeamDash.calls.append(kwargs)
        self._season_type = kwargs.get("season_type_all_star")

    def get_dict(self):
        rows = _FakeTeamDash.rows if self._season_type == "Regular Season" else []
        return {"resultSets": [{"headers": _FakeTeamDash.headers, "rowSet": rows}]}


@pytest.fixture
def fake_team_dash(monkeypatch):
    _FakeTeamDash.calls = []
    _FakeTeamDash.rows = [
        [1, "Oklahoma City Thunder", 25, 118.0, 107.0, 100.0, 11.0, 0.55, 0.59, 0.12, 0.28],
        [2, "Phoenix Suns", 25, 114.0, 113.0, 98.0, 1.0, 0.53, 0.57, 0.13, 0.26],
    ]
    monkeypatch.setattr("nba_api.stats.endpoints.LeagueDashTeamStats", _FakeTeamDash)
    return _FakeTeamDash


@pytest.fixture
def nba_api_reachable(monkeypatch):
    """Fresh-state check passes and the circuit breaker is open."""
    monkeypatch.setattr("evmax.agents.models._nba_freshness.state_is_fresh", _async_true)
    monkeypatch.setattr("evmax.agents.models._nba_freshness.nba_api_in_backoff", lambda: False)


def _eff_state(season, gp=82):
    return {
        "league_avg_ortg": 114.0, "league_avg_drtg": 114.0, "league_avg_pace": 100.0,
        "teams": {
            "thunder": {"ortg": 118.0, "drtg": 107.0, "pace": 100.0, "net": 11.0, "gp": gp, "full_name": "oklahoma city thunder"},
            "suns": {"ortg": 114.0, "drtg": 113.0, "pace": 98.0, "net": 1.0, "gp": gp, "full_name": "phoenix suns"},
        },
        "fetched_at": "2026-10-12",
        **({"season": season} if season else {}),
    }


def _shot_state(season, gp=82):
    team = {"fg3_pct": 0.37, "fg3a": 38.0, "rim_pct": 0.66, "rim_fga": 27.0,
            "mid_pct": 0.43, "mid_fga": 12.0, "gp": gp}
    return {
        "fetched_date": "2026-10-12",
        "teams": {"thunder": dict(team), "suns": {**team, "fg3_pct": 0.34}},
        "league_avgs": {"fg3_pct": 0.36, "rim_pct": 0.64, "mid_pct": 0.42, "fg3a": 36.0, "rim_fga": 26.0},
        **({"season": season} if season else {}),
    }


def _matchup_state(season, gp=82):
    off = {"pts": 115.0, "fg3a": 38.0, "fg3_pct": 0.37, "tov": 12.0, "oreb": 11.0, "gp": gp}
    dfn = {"drtg": 110.0, "blk": 5.0, "stl": 8.0, "opp_pts_paint": 46.0, "opp_pts_fb": 13.0, "gp": gp}
    return {
        "fetched_date": "2026-10-12",
        "offense": {"thunder": dict(off), "suns": {**off, "tov": 15.0}},
        "defense": {"thunder": dict(dfn), "suns": {**dfn, "blk": 4.0}},
        "league_avgs": {"fg3a": 36.0, "tov": 13.0, "blk": 5.0, "stl": 8.0,
                        "opp_pts_paint": 48.0, "pts_paint": 48.0, "opp_pts_fb": 14.0},
        **({"season": season} if season else {}),
    }


async def _fetch_fails(self, season):
    raise RuntimeError(f"nba_api returned no team rows for either season type ({season})")


class TestSeasonRollover:
    # ── efficiency ──
    def test_efficiency_fetch_requests_and_stamps_current_season(self, fake_team_dash, nba_api_reachable):
        agent = EfficiencyModelAgent()
        agent._state = {"nba": _eff_state(_PRIOR_SEASON)}
        stats = asyncio.run(agent._ensure_stats("nba"))
        assert {c["season"] for c in fake_team_dash.calls} == {_SEASON}
        assert stats["season"] == _SEASON
        assert stats["teams"]["thunder"]["gp"] == 25

    def test_efficiency_fetch_before_opening_night_prices_nothing(self, fake_team_dash, nba_api_reachable):
        # nba_api's regular-season query has no rows until opening night.
        fake_team_dash.rows = []
        agent = EfficiencyModelAgent()
        agent._state = {"nba": _eff_state(_PRIOR_SEASON)}
        market, sharp = _make_pair("thunder", "suns", 0.70)
        assert asyncio.run(agent.predict_pair(market, sharp)) is None
        assert {c["season"] for c in fake_team_dash.calls} == {_SEASON}

    @pytest.mark.parametrize("season", [_PRIOR_SEASON, None], ids=["prior-season", "unstamped"])
    def test_efficiency_other_season_state_never_prices(self, season, nba_api_reachable, monkeypatch):
        monkeypatch.setattr(EfficiencyModelAgent, "_fetch_from_nba_api", _fetch_fails)
        agent = EfficiencyModelAgent()
        agent._state = {"nba": _eff_state(season)}
        market, sharp = _make_pair("thunder", "suns", 0.70)
        assert asyncio.run(agent.predict_pair(market, sharp)) is None

    def test_efficiency_backoff_does_not_fall_back_to_other_season(self, monkeypatch):
        monkeypatch.setattr("evmax.agents.models._nba_freshness.state_is_fresh", _async_true)
        monkeypatch.setattr("evmax.agents.models._nba_freshness.nba_api_in_backoff", lambda: True)
        monkeypatch.setattr(EfficiencyModelAgent, "_fetch_from_nba_api", _fetch_fails)
        agent = EfficiencyModelAgent()
        agent._state = {"nba": _eff_state(_PRIOR_SEASON)}
        assert asyncio.run(agent._ensure_stats("nba")) == {}

    def test_efficiency_current_season_state_prices_only_its_season(self, nba_api_reachable):
        agent = EfficiencyModelAgent()
        agent._state = {"nba": _eff_state(_SEASON)}
        market, sharp = _make_pair("thunder", "suns", 0.70)
        market.event_date = _CURRENT_SEASON_GAME
        assert asyncio.run(agent.predict_pair(market, sharp)) is not None
        market.event_date = _PRIOR_SEASON_GAME
        assert asyncio.run(agent.predict_pair(market, sharp)) is None

    # ── shot_quality ──
    @pytest.mark.parametrize("season", [_PRIOR_SEASON, None], ids=["prior-season", "unstamped"])
    def test_shot_quality_other_season_state_never_prices(self, season, nba_api_reachable, monkeypatch):
        monkeypatch.setattr(ShotQualityAgent, "_fetch_shooting_data", _fetch_fails)
        agent = ShotQualityAgent()
        agent._state = {"nba": _shot_state(season)}
        market, sharp = _make_pair("thunder", "suns", 0.70)
        assert asyncio.run(agent.predict_pair(market, sharp)) is None
        assert agent._team_shooting == {}

    def test_shot_quality_current_season_state_prices(self, nba_api_reachable):
        agent = ShotQualityAgent()
        agent._state = {"nba": _shot_state(_SEASON)}
        market, sharp = _make_pair("thunder", "suns", 0.70)
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is not None
        assert result.sample_size == 82

    def test_shot_quality_needs_min_games(self, nba_api_reachable):
        from evmax.agents.models.shot_quality_agent import MIN_GAMES
        agent = ShotQualityAgent()
        agent._state = {"nba": _shot_state(_SEASON, gp=MIN_GAMES - 1)}
        market, sharp = _make_pair("thunder", "suns", 0.70)
        assert asyncio.run(agent.predict_pair(market, sharp)) is None

    def test_shot_quality_stale_memory_never_prices_another_season(self, nba_api_reachable):
        # A long-running process (dashboard, Discord bot) that loaded a state
        # before the rollover keeps it in memory — predict_pair re-checks.
        agent = ShotQualityAgent()
        agent._state = {"nba": _shot_state(_SEASON)}
        market, sharp = _make_pair("thunder", "suns", 0.70)
        market.event_date = _PRIOR_SEASON_GAME
        assert asyncio.run(agent.predict_pair(market, sharp)) is None

    # ── matchup ──
    @pytest.mark.parametrize("season", [_PRIOR_SEASON, None], ids=["prior-season", "unstamped"])
    def test_matchup_other_season_state_never_prices(self, season, nba_api_reachable, monkeypatch):
        monkeypatch.setattr(MatchupAgent, "_fetch_data", _fetch_fails)
        agent = MatchupAgent()
        agent._state = {"nba": _matchup_state(season)}
        market, sharp = _make_pair("thunder", "suns", 0.70)
        assert asyncio.run(agent.predict_pair(market, sharp)) is None
        assert agent._off_stats == {}

    def test_matchup_current_season_state_prices(self, nba_api_reachable):
        agent = MatchupAgent()
        agent._state = {"nba": _matchup_state(_SEASON)}
        market, sharp = _make_pair("thunder", "suns", 0.70)
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is not None
        assert result.sample_size == 82

    def test_matchup_needs_min_games(self, nba_api_reachable):
        from evmax.agents.models.matchup_agent import MIN_GAMES
        agent = MatchupAgent()
        agent._state = {"nba": _matchup_state(_SEASON, gp=MIN_GAMES - 1)}
        market, sharp = _make_pair("thunder", "suns", 0.70)
        assert asyncio.run(agent.predict_pair(market, sharp)) is None

    # ── possession_sim (also feeds NBA spread/total pricing + projections) ──
    @pytest.mark.parametrize("season", [_PRIOR_SEASON, None], ids=["prior-season", "unstamped"])
    def test_possession_sim_other_season_state_never_prices(self, season, nba_api_reachable):
        state = {**MOCK_EFF_STATE}
        state.pop("season")
        if season:
            state["season"] = season
        agent = PossessionSimAgent()
        agent._load_efficiency_state = lambda: state
        market, sharp = _make_pair("thunder", "suns", 0.75)
        assert asyncio.run(agent.predict_pair(market, sharp)) is None
        assert agent.simulate_matchup("thunder", "suns") is None

    def test_possession_sim_prices_only_the_state_season(self, nba_api_reachable):
        agent = PossessionSimAgent()
        agent._load_efficiency_state = lambda: MOCK_EFF_STATE
        market, sharp = _make_pair("thunder", "suns", 0.75)
        market.event_date = _CURRENT_SEASON_GAME
        assert asyncio.run(agent.predict_pair(market, sharp)) is not None
        market.event_date = _PRIOR_SEASON_GAME
        assert asyncio.run(agent.predict_pair(market, sharp)) is None
