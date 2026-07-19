"""Tests for the pitcher_v2 components (WS2b, 2026-07-19).

Covers the xERA blend orderings, the park-normalization math — including the
regression that locks in the Pythagenpat cancellation insight (a SYMMETRIC
venue multiplier must not move the win probability; only per-pitcher sample
normalization does) — the offense clamp, and predict_pair integration with
every v2 feed stubbed.
"""

from __future__ import annotations

import math
from datetime import datetime

import pytest

import evmax.agents.models.pitcher_agent as pitcher_mod
from evmax.agents.models.pitcher_agent import (
    ERA_BLEND_WEIGHT_V2,
    FIP_BLEND_WEIGHT_V2,
    OFFENSE_CLAMP_HI,
    OFFENSE_CLAMP_LO,
    PYTHAG_EXP,
    XERA_BLEND_WEIGHT,
    XERA_THIN_BIP,
    PitcherModelAgent,
    _effective_era,
    _park_factor_for,
)
from evmax.models.market import MarketSource, MarketType, PredictionMarket
from evmax.models.odds import SharpBook, SharpOdds


class TestEffectiveEraXera:
    def test_full_v2_blend(self):
        p = {"fip": 3.80, "era": 4.20, "xera": 3.50, "xera_bip": 200}
        total = XERA_BLEND_WEIGHT + FIP_BLEND_WEIGHT_V2 + ERA_BLEND_WEIGHT_V2
        expected = (
            XERA_BLEND_WEIGHT * 3.50
            + FIP_BLEND_WEIGHT_V2 * 3.80
            + ERA_BLEND_WEIGHT_V2 * 4.20
        ) / total
        assert _effective_era(p, 4.08) == pytest.approx(expected)

    def test_thin_bip_substitutes_prior_at_half_weight(self):
        p = {"fip": 3.80, "era": 4.20, "xera": 3.50,
             "xera_bip": XERA_THIN_BIP - 1, "xera_prior": 3.90}
        half = XERA_BLEND_WEIGHT * 0.5
        total = half + FIP_BLEND_WEIGHT_V2 + ERA_BLEND_WEIGHT_V2
        expected = (
            half * 3.90 + FIP_BLEND_WEIGHT_V2 * 3.80 + ERA_BLEND_WEIGHT_V2 * 4.20
        ) / total
        assert _effective_era(p, 4.08) == pytest.approx(expected)

    def test_thin_bip_without_prior_drops_xera(self):
        p = {"fip": 3.80, "era": 4.20, "xera": 3.50, "xera_bip": 10}
        # No xera_prior → falls to the v1 60/40 blend.
        assert _effective_era(p, 4.08) == pytest.approx(0.6 * 3.80 + 0.4 * 4.20)

    def test_prior_only_path(self):
        # No current xera at all, prior present with thin bip → half weight.
        p = {"fip": 3.80, "era": 4.20, "xera_prior": 3.90, "xera_bip": 0}
        half = XERA_BLEND_WEIGHT * 0.5
        total = half + FIP_BLEND_WEIGHT_V2 + ERA_BLEND_WEIGHT_V2
        expected = (
            half * 3.90 + FIP_BLEND_WEIGHT_V2 * 3.80 + ERA_BLEND_WEIGHT_V2 * 4.20
        ) / total
        assert _effective_era(p, 4.08) == pytest.approx(expected)

    def test_v1_paths_unchanged(self):
        assert _effective_era({"fip": 3.5, "era": 4.5}, 4.08) == pytest.approx(
            0.6 * 3.5 + 0.4 * 4.5
        )
        assert _effective_era({"fip": 3.5}, 4.08) == 3.5
        assert _effective_era({"era": 4.5}, 4.08) == 4.5
        assert _effective_era({}, 4.08) == 4.08
        # Placeholder zeros are missing, not rates.
        assert _effective_era({"era": 0.0}, 4.08) == 4.08


class TestParkNormalization:
    def test_results_components_divided_by_sqrt(self):
        p = {"fip": 4.00, "era": 4.00}
        adjusted = _effective_era(p, 4.08, park_factor=1.21)
        assert adjusted == pytest.approx(4.00 / math.sqrt(1.21))

    def test_xera_never_park_adjusted(self):
        # xERA is park-neutral by construction — only FIP/ERA move.
        p = {"fip": 4.00, "era": 4.00, "xera": 4.00, "xera_bip": 200}
        adjusted = _effective_era(p, 4.08, park_factor=1.21)
        div = math.sqrt(1.21)
        total = XERA_BLEND_WEIGHT + FIP_BLEND_WEIGHT_V2 + ERA_BLEND_WEIGHT_V2
        expected = (
            XERA_BLEND_WEIGHT * 4.00
            + FIP_BLEND_WEIGHT_V2 * (4.00 / div)
            + ERA_BLEND_WEIGHT_V2 * (4.00 / div)
        ) / total
        assert adjusted == pytest.approx(expected)

    def test_symmetric_venue_multiplier_cancels_in_pythag(self):
        """The cancellation regression: scaling BOTH teams' RS/RA by the same
        venue constant k must leave the Pythagenpat win prob unchanged —
        which is why v2 applies park to each pitcher's SAMPLE instead of to
        today's venue."""
        e = PYTHAG_EXP
        home_rate, away_rate = 3.70, 4.30

        def _pythag(rs, ra):
            return rs ** e / (rs ** e + ra ** e)

        base = _pythag(away_rate, home_rate)
        for k in (0.80, 1.0, 1.24):
            scaled = _pythag(away_rate * k, home_rate * k)
            assert scaled == pytest.approx(base, abs=1e-12)

    def test_park_factor_lookup(self):
        assert _park_factor_for({"team": "rockies"}) == pytest.approx(1.24)
        assert _park_factor_for({"team": "mariners"}) == pytest.approx(0.92)
        assert _park_factor_for({"team": "yankees"}) == 1.0   # unlisted → neutral
        assert _park_factor_for({}) == 1.0


class TestOffenseFactor:
    def test_clamped(self, monkeypatch):
        import evmax.clients.baseball_props_cache as cache_mod

        monkeypatch.setattr(
            cache_mod, "get_team_offense",
            lambda tid: {"runs_per_game": 9.0, "games": 60},
        )
        monkeypatch.setattr(cache_mod, "league_runs_per_game", lambda: 4.5)
        assert PitcherModelAgent._offense_factor("colorado rockies") == OFFENSE_CLAMP_HI

        monkeypatch.setattr(
            cache_mod, "get_team_offense",
            lambda tid: {"runs_per_game": 2.0, "games": 60},
        )
        assert PitcherModelAgent._offense_factor("colorado rockies") == OFFENSE_CLAMP_LO

    def test_defaults_to_neutral_without_data(self, monkeypatch):
        import evmax.clients.baseball_props_cache as cache_mod

        monkeypatch.setattr(cache_mod, "get_team_offense", lambda tid: None)
        monkeypatch.setattr(cache_mod, "league_runs_per_game", lambda: None)
        assert PitcherModelAgent._offense_factor("new york yankees") == 1.0
        assert PitcherModelAgent._offense_factor("not a team") == 1.0


class TestTeamNickname:
    def test_multiword_nickname_suffix(self):
        assert PitcherModelAgent._team_nickname("boston red sox") == "red sox"
        assert PitcherModelAgent._team_nickname("chicago white sox") == "white sox"
        assert PitcherModelAgent._team_nickname("toronto blue jays") == "blue jays"

    def test_exact_and_unknown(self):
        assert PitcherModelAgent._team_nickname("yankees") == "yankees"
        assert PitcherModelAgent._team_nickname("real madrid") is None


def _market():
    return PredictionMarket(
        id="kalshi:test", source=MarketSource.kalshi, sector="baseball",
        market_type=MarketType.moneyline, title="yankees vs red sox",
        yes_price=0.5, no_price=0.5,
        team_home="yankees", team_away="red sox",
    )


def _sharp():
    return SharpOdds(
        event_id="baseball::2026-07-19::yankees_vs_red_sox",
        book=SharpBook.pinnacle, sector="baseball",
        market_type=MarketType.moneyline,
        outcome_a_label="yankees", outcome_b_label="red sox",
        outcome_a_decimal=1.9, outcome_b_decimal=1.9,
        true_prob_a=0.5, true_prob_b=0.5, margin=0.03,
        event_date=datetime(2026, 7, 19, 19, 0),
    )


class TestPredictPairV2Integration:
    @pytest.fixture
    def agent(self, tmp_path, monkeypatch):
        monkeypatch.setattr("evmax.agents.models.base.STATE_DIR", tmp_path)
        a = PitcherModelAgent()
        a._state_path = tmp_path / "pitcher_state.json"

        async def _no_live():
            return {}

        monkeypatch.setattr(pitcher_mod, "_fetch_probable_starters", _no_live)

        async def _no_appearances(*args, **kwargs):
            return {}

        monkeypatch.setattr(
            "evmax.clients.mlb_statsapi.fetch_reliever_appearances",
            _no_appearances,
        )
        monkeypatch.setattr(
            PitcherModelAgent, "_offense_factor", staticmethod(lambda team: 1.0)
        )
        return a

    @pytest.mark.asyncio
    async def test_model_name_is_v2(self, agent):
        agent.seed_pitchers({
            "Starter A": {"era": 3.5, "fip": 3.6, "ip": 100, "team": "yankees"},
            "Starter B": {"era": 4.5, "fip": 4.4, "ip": 100, "team": "red sox"},
        })
        pred = await agent.predict_pair(_market(), _sharp())
        assert pred is not None
        assert pred.model_name == "pitcher_v2"

    @pytest.mark.asyncio
    async def test_pen_shifts_matchup(self, agent):
        """A strong seeded pen behind the home starter must improve the
        home win prob vs the pen-less baseline."""
        starters = {
            "Starter A": {"era": 4.0, "fip": 4.0, "ip": 100, "team": "yankees"},
            "Starter B": {"era": 4.0, "fip": 4.0, "ip": 100, "team": "red sox"},
        }
        agent.seed_pitchers(dict(starters))
        base = await agent.predict_pair(_market(), _sharp())

        # Elite yankees pen (3 arms, high strikeouts), no red sox pen.
        agent._state["relievers"] = {
            f"yankee arm {i}": {"team": "yankees", "ip": 30.0, "er": 6,
                                "bb": 8, "so": 45, "hr": 1}
            for i in range(3)
        }
        agent._state["league_pen_cfip"] = 3.10
        with_pen = await agent.predict_pair(_market(), _sharp())

        assert with_pen is not None and base is not None
        assert with_pen.true_prob_a > base.true_prob_a
        assert "pen=" in with_pen.notes

    @pytest.mark.asyncio
    async def test_degrades_to_v1_without_v2_state(self, agent):
        """No relievers, no xera, neutral offense/park stubs → identical to
        the v1 starter-only Pythag + home bonus."""
        agent.seed_pitchers({
            "Starter A": {"era": 4.0, "ip": 160, "team": "yankees"},
            "Starter B": {"era": 4.0, "ip": 160, "team": "red sox"},
        })
        # Neutralize park (Fenway is in the table) to isolate the v1 math.
        import evmax.agents.models.pitcher_agent as pm
        old = pm._park_factor_for
        pm._park_factor_for = lambda p: 1.0
        try:
            pred = await agent.predict_pair(_market(), _sharp())
        finally:
            pm._park_factor_for = old
        assert pred is not None
        assert pred.true_prob_a == pytest.approx(0.5 + agent._home_advantage(), abs=1e-3)
