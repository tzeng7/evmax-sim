"""Tests for MLB player-prop plumbing (added 2026-06-27).

Increment 1 wires baseball props end to end: Kalshi series → stat mapping +
title parse, Pinnacle special-description parse (with the "(must start)"
qualifier), accent-folded player normalization so Pinnacle/Kalshi names
collide, and per-stat distribution pricing in ev/prop_pricing.py.

These tests lock the parsing/pricing/normalization contracts that the live
matching depends on. Network-touching end-to-end matching is covered by the
manual integration probe, not here.
"""
from __future__ import annotations

import pytest

from evmax.clients.esports_pinnacle import parse_prop_description
from evmax.clients.kalshi import KalshiClient, _PROP_SERIES_TO_STAT
from evmax.ev.prop_pricing import price_kalshi_threshold, supported_stats
from evmax.players import normalize_player_name


# ===========================================================================
# Pinnacle special-description parsing
# ===========================================================================


class TestPinnacleMlbPropParse:
    @pytest.mark.parametrize(
        "desc,expected",
        [
            ("Rafael Devers (Home Runs)(must start)", ("Rafael Devers", "home_runs")),
            ("Michael Lorenzen (Total Strikeouts)(must start)", ("Michael Lorenzen", "strikeouts")),
            ("Kyle Tucker (Total Bases)(must start)", ("Kyle Tucker", "total_bases")),
            ("Andre Pallante (Pitching Outs)(must start)", ("Andre Pallante", "pitching_outs")),
            # no qualifier suffix should still parse
            ("Kyle Tucker (Total Bases)", ("Kyle Tucker", "total_bases")),
        ],
    )
    def test_mlb_descriptions_parse(self, desc, expected):
        assert parse_prop_description(desc) == expected

    def test_unmapped_mlb_stat_returns_none(self):
        # Pinnacle posts these but we have no matching Kalshi series — drop them.
        assert parse_prop_description("Michael Wacha (Hits Allowed)(must start)") is None
        assert parse_prop_description("Michael Lorenzen (Earned Runs)(must start)") is None

    def test_nba_formats_still_parse(self):
        # Regression: the reorder + qualifier strip must not break NBA props.
        assert parse_prop_description("Luka Doncic (Points)") == ("Luka Doncic", "points")
        assert parse_prop_description("Jalen Brunson Total Assists") == ("Jalen Brunson", "assists")
        assert parse_prop_description("LeBron James (PRA)") == ("LeBron James", "points_rebounds_assists")


# ===========================================================================
# Player normalization — accent fold so Pinnacle (accented) matches Kalshi (ascii)
# ===========================================================================


class TestBaseballNameNormalization:
    def test_accent_fold_collides_pinnacle_and_kalshi(self):
        # Pinnacle: "José Ramírez"; Kalshi title: "Jose Ramirez". Must match.
        assert normalize_player_name("José Ramírez", "baseball") == normalize_player_name(
            "Jose Ramirez", "baseball"
        )
        assert normalize_player_name("José Ramírez", "baseball") == "jose_ramirez"

    def test_suffix_stripped(self):
        assert normalize_player_name("Bobby Witt Jr.", "baseball") == "bobby_witt"
        assert normalize_player_name("Luis Robert Jr.", "baseball") == "luis_robert"

    def test_apostrophe_preserved_consistently(self):
        assert normalize_player_name("Tyler O'Neill", "baseball") == "tyler_o'neill"

    def test_other_sectors_not_accent_folded(self):
        # NBA/NFL must keep their existing behavior (no fold applied).
        assert normalize_player_name("LeBron James", "nba") == "lebron_james"
        assert normalize_player_name("Patrick Mahomes", "nfl") == "patrick_mahomes"


# ===========================================================================
# Kalshi series → stat mapping + title parse
# ===========================================================================


class TestKalshiMlbPropSeries:
    @pytest.mark.parametrize(
        "series,stat",
        [
            ("KXMLBKS", "strikeouts"),
            ("KXMLBOUTS", "pitching_outs"),
            ("KXMLBTB", "total_bases"),
            ("KXMLBHR", "home_runs"),
            ("KXMLBHIT", "hits"),
            ("KXMLBHRR", "hits_runs_rbis"),
            ("KXMLBRBI", "rbis"),
        ],
    )
    def test_series_to_stat(self, series, stat):
        assert _PROP_SERIES_TO_STAT[series] == stat

    def test_hr_and_hrr_do_not_collide(self):
        # KXMLBHR is a prefix of KXMLBHRR; the lookup is exact (per series
        # fetch), so they must map to distinct stats.
        assert _PROP_SERIES_TO_STAT["KXMLBHR"] != _PROP_SERIES_TO_STAT["KXMLBHRR"]

    @pytest.mark.parametrize(
        "title,player,thr",
        [
            ("Andre Pallante: 9+ strikeouts?", "Andre Pallante", 9.0),
            ("Tyler O'Neill: 5+ total bases?", "Tyler O'Neill", 5.0),
            ("Tyler O'Neill: 2+ home runs?", "Tyler O'Neill", 2.0),
            ("Tyler O'Neill: 5+ hits + runs + RBIs?", "Tyler O'Neill", 5.0),
            ("Logan Gilbert: 18+ Outs Recorded?", "Logan Gilbert", 18.0),
        ],
    )
    def test_title_parse(self, title, player, thr):
        client = KalshiClient.__new__(KalshiClient)  # no network / auth
        name, threshold = client._parse_prop_title(title, "strikeouts")
        assert name == player
        assert threshold == thr


# ===========================================================================
# Distribution pricing for MLB stats
# ===========================================================================


class TestBaseballPropPricing:
    def test_all_mlb_stats_supported(self):
        for stat in ["strikeouts", "pitching_outs", "total_bases", "home_runs",
                     "hits", "hits_runs_rbis", "rbis"]:
            assert stat in supported_stats()

    def test_threshold_probability_monotone_decreasing(self):
        # P(>= k) must fall as k rises off a fixed anchor.
        probs = [
            price_kalshi_threshold("strikeouts", 5.5, 0.52, k)
            for k in (4, 5, 6, 7, 8, 9)
        ]
        assert all(p is not None for p in probs)
        assert all(probs[i] >= probs[i + 1] for i in range(len(probs) - 1))

    def test_anchor_recovered_at_its_own_cutoff(self):
        # Anchor strikeouts line 5.5 @ 0.52 over → P(>=6) ≈ 0.52.
        p = price_kalshi_threshold("strikeouts", 5.5, 0.52, 6)
        assert p == pytest.approx(0.52, abs=1e-2)

    def test_home_runs_rare_event_tail(self):
        # HR anchor 0.5 @ 0.30 over → P(>=1)=0.30, P(>=2) small but positive.
        assert price_kalshi_threshold("home_runs", 0.5, 0.30, 1) == pytest.approx(0.30, abs=1e-2)
        p2 = price_kalshi_threshold("home_runs", 0.5, 0.30, 2)
        assert 0.0 < p2 < 0.12

    def test_pitching_outs_normal(self):
        p = price_kalshi_threshold("pitching_outs", 17.5, 0.50, 18)
        assert p == pytest.approx(0.50, abs=1e-2)


# ===========================================================================
# Projection model (evmax/models_ml/baseball_props.py)
# ===========================================================================

from evmax.models_ml.baseball_props import (  # noqa: E402
    HitterProfile,
    MatchupContext,
    PitcherProfile,
    project_mean,
    project_prob,
    project_pitcher_strikeouts,
)


def _k_pitcher() -> PitcherProfile:
    # 146 K / 376 BF over 16 starts → ~0.39 K/BF, ~23.5 BF/start.
    return PitcherProfile(
        batters_faced=376, outs=297, strike_outs=146, home_runs=10,
        hits=70, games_started=16, innings_pitched=99.0,
    )


def _hitter() -> HitterProfile:
    return HitterProfile(
        plate_appearances=336, at_bats=292, hits=98, doubles=15, triples=1,
        home_runs=12, rbi=52, runs=43, strike_outs=44, total_bases=165, games=80,
    )


class TestProjectionModel:
    def test_pitcher_k_expected_in_range(self):
        m = project_pitcher_strikeouts(_k_pitcher(), MatchupContext())
        # ~0.388 K/BF × 23.5 BF/start ≈ 9.1 with neutral opponent.
        assert 8.0 < m < 10.5

    def test_high_k_opponent_raises_projection(self):
        p = _k_pitcher()
        low = project_pitcher_strikeouts(p, MatchupContext(opp_team_k_rate=0.18))
        high = project_pitcher_strikeouts(p, MatchupContext(opp_team_k_rate=0.26))
        assert high > low

    def test_opponent_multiplier_is_clamped(self):
        p = _k_pitcher()
        # An absurd opponent K-rate must not blow up the projection >1.4×.
        base = project_pitcher_strikeouts(p, MatchupContext())
        extreme = project_pitcher_strikeouts(p, MatchupContext(opp_team_k_rate=0.99))
        assert extreme <= base * 1.41

    def test_coors_inflates_home_runs(self):
        h = _hitter()
        neutral = project_mean("home_runs", MatchupContext(home_team_id=137), hitter=h)  # Oracle
        coors = project_mean("home_runs", MatchupContext(home_team_id=115), hitter=h)
        assert coors > neutral

    def test_threshold_prob_monotone(self):
        h = _hitter()
        ctx = MatchupContext()
        probs = [project_prob("total_bases", k, ctx, hitter=h) for k in (1, 2, 3, 4)]
        assert all(p is not None for p in probs)
        assert all(probs[i] >= probs[i + 1] for i in range(len(probs) - 1))

    def test_wrong_player_type_returns_none(self):
        # Pitcher stat with no pitcher profile, hitter stat with no hitter.
        assert project_mean("strikeouts", MatchupContext(), hitter=_hitter()) is None
        assert project_mean("home_runs", MatchupContext(), pitcher=_k_pitcher()) is None

    def test_zero_sample_returns_none(self):
        empty_h = HitterProfile(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        assert project_mean("hits", MatchupContext(), hitter=empty_h) is None


# ===========================================================================
# Coordinator model-blend hook
# ===========================================================================

from types import SimpleNamespace  # noqa: E402

from evmax.agents.coordinator import AgentCoordinator, BASEBALL_PROP_MODEL_WEIGHT  # noqa: E402
from evmax.models_ml.baseball_props import project_prob as _project_prob  # noqa: E402


class TestCoordinatorBaseballBlend:
    def _coord(self):
        # Bare instance — the helper uses no instance state.
        return AgentCoordinator.__new__(AgentCoordinator)

    def test_pitcher_prob_matches_projection(self, monkeypatch):
        import evmax.clients.baseball_props_cache as bbc
        monkeypatch.setattr(bbc, "get_pitcher_profile", lambda n: _k_pitcher())
        market = SimpleNamespace(player_name="x", stat_type="strikeouts", threshold=6.0)
        got = self._coord()._baseball_prop_model_prob(market)
        expected = _project_prob("strikeouts", 6.0, MatchupContext(), pitcher=_k_pitcher())
        assert got == pytest.approx(expected)
        assert 0.0 < got < 1.0

    def test_missing_profile_returns_none(self, monkeypatch):
        import evmax.clients.baseball_props_cache as bbc
        monkeypatch.setattr(bbc, "get_pitcher_profile", lambda n: None)
        market = SimpleNamespace(player_name="x", stat_type="strikeouts", threshold=6.0)
        assert self._coord()._baseball_prop_model_prob(market) is None

    def test_unanchored_stat_still_projects(self, monkeypatch):
        # hits has no Pinnacle anchor but the model can still price it.
        import evmax.clients.baseball_props_cache as bbc
        monkeypatch.setattr(bbc, "get_hitter_profile", lambda n: _hitter())
        market = SimpleNamespace(player_name="x", stat_type="hits", threshold=1.0)
        got = self._coord()._baseball_prop_model_prob(market)
        assert got is not None and 0.0 < got < 1.0

    def test_tb_bridge_prices_unanchored_hits(self, monkeypatch):
        # Hits has no Pinnacle line; the bridge prices it off the player's
        # Total-Bases anchor and returns a synthetic SharpOdds for the hits stat.
        from evmax.models.odds import SharpBook, SharpOdds
        import evmax.clients.baseball_props_cache as bbc
        monkeypatch.setattr(bbc, "get_hitter_profile", lambda n: _hitter())
        tb_anchor = SharpOdds(
            event_id="baseball::2026-06-27_ATL_at_SF::prop::x::total_bases::1.5",
            book=SharpBook.pinnacle, sector="baseball",
            outcome_a_decimal=1.9, outcome_b_decimal=1.9,
            total_line=1.5, true_prob_over=0.50,
            prop_player_name="x", prop_stat_type="total_bases",
        )
        market = SimpleNamespace(player_name="x", stat_type="hits", threshold=2.0)
        synth = self._coord()._baseball_bridged_prop_sharp(market, tb_anchor, "baseball")
        assert synth is not None
        assert synth.prop_stat_type == "hits"
        assert synth.total_line == 2.0
        assert 0.0 < synth.true_prob_over < 1.0
        assert "::prop::x::hits::2.0" in synth.event_id

    def test_tb_bridge_none_without_profile(self, monkeypatch):
        from evmax.models.odds import SharpBook, SharpOdds
        import evmax.clients.baseball_props_cache as bbc
        monkeypatch.setattr(bbc, "get_hitter_profile", lambda n: None)
        tb_anchor = SharpOdds(
            event_id="baseball::g::prop::x::total_bases::1.5",
            book=SharpBook.pinnacle, sector="baseball",
            outcome_a_decimal=1.9, outcome_b_decimal=1.9,
            total_line=1.5, true_prob_over=0.50,
            prop_player_name="x", prop_stat_type="total_bases",
        )
        market = SimpleNamespace(player_name="x", stat_type="hits", threshold=2.0)
        assert self._coord()._baseball_bridged_prop_sharp(market, tb_anchor, "baseball") is None

    def test_blend_weight_is_conservative(self):
        # Guardrail: a rough neutral-context model should nudge, not dominate.
        assert 0.0 < BASEBALL_PROP_MODEL_WEIGHT <= 0.5


# ===========================================================================
# Dashboard surfacing
# ===========================================================================


class TestDashboardEndpoint:
    def test_mlb_props_endpoint_shape(self):
        from fastapi.testclient import TestClient

        from evmax.web.app import app

        r = TestClient(app).get("/api/mlb-props")
        assert r.status_code == 200
        body = r.json()
        assert "calibration" in body and "shadow" in body
        # shadow tally is always present (empty until a scan persists rows)
        assert set(body["shadow"]) >= {"by_stat", "total", "resolved"}
        assert isinstance(body["shadow"]["by_stat"], list)
