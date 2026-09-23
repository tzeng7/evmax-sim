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
        # Exactly the 32 current franchises: the stale "arizona coyotes" /
        # "utah hockey" keys and the 4-Nations/All-Star exhibition keys were
        # pruned 2026-09-22 (scripts/offseason_regress.py --prune-only).
        assert len(ratings) == 32
        for stale in ("arizona coyotes", "utah hockey", "canada", "usa", "mcdavid"):
            assert stale not in ratings
            assert stale not in state["nhl"]["season_games"]
        assert all(1000.0 < r < 2000.0 for r in ratings.values())

    def test_nhl_form_state_has_no_exhibition_keys(self):
        import json
        import pathlib

        path = pathlib.Path(__file__).resolve().parents[1] / "data" / "models" / "form_state.json"
        nhl = json.loads(path.read_text())["nhl"]
        assert len(nhl) == 32
        for stale in ("canada", "finland", "sweden", "usa", "mcdavid", "matthews",
                      "mackinnon", "hughes", "arizona coyotes", "utah hockey"):
            assert stale not in nhl


# ── Preseason-prior ramp + staleness guard (2026-09-22) ────────────────────

from datetime import date, datetime, timezone  # noqa: E402

from evmax.agents.models.nhl_xg_agent import (  # noqa: E402
    PRIOR_RAMP_K,
    PRIOR_REGRESS_RHO,
    RAMP_CONFIDENCE,
    nhl_season_for,
    ramp_rate,
    regress_prior_rate,
)


def _dated_pair(home: str, away: str, day: date) -> tuple[PredictionMarket, SharpOdds]:
    market, sharp = _pair(home, away)
    market = market.model_copy(update={
        "event_date": datetime(day.year, day.month, day.day, 12, tzinfo=timezone.utc),
    })
    return market, sharp


def _v2_agent(teams: dict, prior_teams: dict, season: int = 2026, lg: float = 2.50) -> NhlXgModelAgent:
    agent = NhlXgModelAgent()
    agent._state = {"nhl": {
        "schema_version": 2,
        "season_start_year": season,
        "league_avg_xg_per_60": None if not teams else lg,
        "teams": teams,
        "prior": {"season_start_year": season - 1, "league_avg_xg_per_60": lg,
                  "regress_rho": PRIOR_REGRESS_RHO, "teams": prior_teams},
    }}
    return agent


def _expected_prob(xf_a, xa_a, xf_b, xa_b) -> float:
    margin = ((xf_a - xa_a) - (xf_b - xa_b)) * (MIN_5V5_PER_GAME / 60.0) + HOME_EDGE_GOALS
    return _normal_cdf(margin / GOAL_STDEV)


class TestRampMath:
    def test_validated_constants(self):
        # The walk-forward validated exactly K=20, rho=0.7 — pin them.
        assert PRIOR_RAMP_K == 20.0
        assert PRIOR_REGRESS_RHO == 0.7
        assert RAMP_CONFIDENCE == 0.70

    def test_regress_prior_rate(self):
        assert regress_prior_rate(3.0, 2.5) == pytest.approx(2.5 + 0.7 * 0.5)
        assert regress_prior_rate(2.5, 2.5) == pytest.approx(2.5)

    def test_gp0_is_the_regressed_prior(self):
        assert ramp_rate(None, 0, 2.85) == 2.85
        assert ramp_rate(3.4, 0, 2.85) == 2.85  # gp=0 ignores any in-season number

    def test_gp20_is_fifty_fifty(self):
        assert ramp_rate(3.0, 20, 2.6) == pytest.approx(2.8)

    def test_gp82_keeps_twenty_percent_prior(self):
        got = ramp_rate(3.0, 82, 2.6)
        assert got == pytest.approx((82 * 3.0 + 20 * 2.6) / 102)
        assert (got - 2.6) / (3.0 - 2.6) == pytest.approx(82 / 102)

    def test_season_rollover_is_september(self):
        assert nhl_season_for(date(2026, 9, 29)) == 2026   # 2026-27 opener
        assert nhl_season_for(date(2026, 8, 31)) == 2025
        assert nhl_season_for(date(2027, 4, 10)) == 2026   # 2026-27 finale
        assert nhl_season_for(date(2026, 6, 14)) == 2025   # 2025-26 Cup final


class TestPriorRamp:
    PRIOR = {
        "boston bruins": {"xgf_per_60": 2.70, "xga_per_60": 2.40},
        "toronto maple leafs": {"xgf_per_60": 2.45, "xga_per_60": 2.60},
    }

    def test_opening_night_prior_only_fires_at_ramp_confidence(self):
        agent = _v2_agent({}, self.PRIOR)
        market, sharp = _dated_pair("boston bruins", "toronto maple leafs", date(2026, 9, 29))
        r = asyncio.run(agent.predict_pair(market, sharp))
        assert r is not None
        assert r.confidence == RAMP_CONFIDENCE
        assert r.sample_size == 0
        assert r.true_prob_a == pytest.approx(_expected_prob(2.70, 2.40, 2.45, 2.60), abs=1e-9)
        assert r.notes.startswith("prior_only")

    def test_gp20_blends_fifty_fifty(self):
        teams = {
            "boston bruins": _team_stats(3.10, 2.20, gp=20),
            "toronto maple leafs": _team_stats(2.25, 2.80, gp=20),
        }
        agent = _v2_agent(teams, self.PRIOR)
        market, sharp = _dated_pair("boston bruins", "toronto maple leafs", date(2026, 11, 20))
        r = asyncio.run(agent.predict_pair(market, sharp))
        exp = _expected_prob((3.10 + 2.70) / 2, (2.20 + 2.40) / 2, (2.25 + 2.45) / 2, (2.80 + 2.60) / 2)
        assert r.true_prob_a == pytest.approx(exp, abs=1e-9)
        assert r.confidence == RAMP_CONFIDENCE
        assert r.notes.startswith("ramp")

    def test_gp82_full_confidence_prior_still_weighs(self):
        teams = {
            "boston bruins": _team_stats(3.10, 2.20, gp=82),
            "toronto maple leafs": _team_stats(2.25, 2.80, gp=82),
        }
        agent = _v2_agent(teams, self.PRIOR)
        market, sharp = _dated_pair("boston bruins", "toronto maple leafs", date(2027, 4, 10))
        r = asyncio.run(agent.predict_pair(market, sharp))
        w = 82 / 102
        exp = _expected_prob(w * 3.10 + (1 - w) * 2.70, w * 2.20 + (1 - w) * 2.40,
                             w * 2.25 + (1 - w) * 2.45, w * 2.80 + (1 - w) * 2.60)
        assert r.true_prob_a == pytest.approx(exp, abs=1e-9)
        assert r.confidence == 0.85

    def test_team_with_prior_is_never_blanked_below_min_games(self):
        teams = {"boston bruins": _team_stats(3.0, 2.2, gp=MIN_GAMES - 7)}  # toronto: gp 0
        agent = _v2_agent(teams, self.PRIOR)
        market, sharp = _dated_pair("boston bruins", "toronto maple leafs", date(2026, 10, 5))
        r = asyncio.run(agent.predict_pair(market, sharp))
        assert r is not None
        assert r.sample_size == 0
        assert r.confidence == RAMP_CONFIDENCE

    def test_team_without_prior_needs_min_games(self):
        prior = {"boston bruins": self.PRIOR["boston bruins"]}  # expansion-style gap
        teams = {"toronto maple leafs": _team_stats(2.5, 2.5, gp=MIN_GAMES - 1)}
        agent = _v2_agent(teams, prior)
        market, sharp = _dated_pair("boston bruins", "toronto maple leafs", date(2026, 10, 20))
        assert asyncio.run(agent.predict_pair(market, sharp)) is None

    def test_prior_from_the_wrong_season_is_ignored(self):
        agent = _v2_agent({}, self.PRIOR)
        agent._state["nhl"]["prior"]["season_start_year"] = 2023  # not season - 1
        market, sharp = _dated_pair("boston bruins", "toronto maple leafs", date(2026, 9, 29))
        assert asyncio.run(agent.predict_pair(market, sharp)) is None


class TestStalenessGuard:
    FINAL = {  # last season's FINAL ratings, gp=82
        "boston bruins": _team_stats(2.90, 2.30, gp=82),
        "toronto maple leafs": _team_stats(2.40, 2.70, gp=82),
    }

    def _legacy(self, season: int) -> NhlXgModelAgent:
        agent = NhlXgModelAgent()
        agent._state = {"nhl": {"teams": self.FINAL, "league_avg_xg_per_60": 2.50,
                                "season_start_year": season}}
        return agent

    def test_previous_season_block_is_prior_only_not_current(self):
        agent = self._legacy(2025)
        market, sharp = _dated_pair("boston bruins", "toronto maple leafs", date(2026, 9, 29))
        r = asyncio.run(agent.predict_pair(market, sharp))
        assert r is not None
        assert r.confidence == RAMP_CONFIDENCE          # NOT 0.85
        assert r.sample_size == 0
        exp = _expected_prob(regress_prior_rate(2.90, 2.5), regress_prior_rate(2.30, 2.5),
                             regress_prior_rate(2.40, 2.5), regress_prior_rate(2.70, 2.5))
        assert r.true_prob_a == pytest.approx(exp, abs=1e-9)

    def test_same_block_is_current_inside_its_own_season(self):
        agent = self._legacy(2025)
        market, sharp = _dated_pair("boston bruins", "toronto maple leafs", date(2026, 4, 10))
        r = asyncio.run(agent.predict_pair(market, sharp))
        assert r.confidence == 0.85
        assert r.true_prob_a == pytest.approx(_expected_prob(2.90, 2.30, 2.40, 2.70), abs=1e-9)

    def test_two_seasons_stale_returns_none(self):
        agent = self._legacy(2024)
        market, sharp = _dated_pair("boston bruins", "toronto maple leafs", date(2026, 10, 1))
        assert asyncio.run(agent.predict_pair(market, sharp)) is None

    def test_legacy_st_louis_key_still_resolves(self):
        # Old states keyed St. Louis with a dot; the canonical is now dot-free.
        agent = NhlXgModelAgent()
        agent._state = {"nhl": {"league_avg_xg_per_60": 2.5, "season_start_year": 2025, "teams": {
            "st. louis blues": _team_stats(2.4, 2.5, gp=82),
            "boston bruins": _team_stats(2.7, 2.3, gp=82),
        }}}
        for label in ("St. Louis Blues", "st louis blues", "blues", "STL"):
            market, sharp = _dated_pair(label, "boston bruins", date(2026, 9, 29))
            assert asyncio.run(agent.predict_pair(market, sharp)) is not None, label


class TestShippedOpeningNightState:
    """data/models/nhl_xg_state.json as committed for the 2026-27 opener."""

    def _state(self) -> dict:
        import json
        import pathlib

        path = pathlib.Path(__file__).resolve().parents[1] / "data" / "models" / "nhl_xg_state.json"
        return json.loads(path.read_text())["nhl"]

    def test_prior_only_shape(self):
        st = self._state()
        assert st["schema_version"] == 2
        assert st["season_start_year"] >= 2026
        prior = st["prior"]
        assert prior["season_start_year"] == st["season_start_year"] - 1
        assert prior["regress_rho"] == PRIOR_REGRESS_RHO
        assert sorted(prior["teams"]) == sorted(NHL_ABBREV_TO_NAME.values())
        for t in prior["teams"].values():
            assert t["xgf_per_60"] == pytest.approx(
                regress_prior_rate(t["raw_xgf_per_60"], prior["league_avg_xg_per_60"]), abs=1e-4)

    def test_agent_fires_on_opening_night_from_shipped_state(self):
        agent = NhlXgModelAgent()  # loads the real state file
        market, sharp = _dated_pair("Tampa Bay Lightning", "St. Louis Blues", date(2026, 9, 29))
        r = asyncio.run(agent.predict_pair(market, sharp))
        assert r is not None
        assert r.confidence == RAMP_CONFIDENCE


# ── Ensemble weights + guards (2026-09-22) ─────────────────────────────────

class TestNhlBlendWiring:
    def test_nhl_weights(self):
        from evmax.agents.models.ensemble_agent import EnsembleModelAgent

        assert EnsembleModelAgent.SECTOR_WEIGHT_OVERRIDES["nhl"] == {
            "nhl_xg": 0.30, "elo": 0.15, "form": 0.0, "poisson": 0.0,
        }

    def test_nhl_sharp_only_moneyline_guard(self):
        from evmax.agents.odds.ev_gap_agent import MIN_NONSHARP_MODELS, REQUIRED_BLEND_MODELS

        assert MIN_NONSHARP_MODELS["nhl"]["min_count"] == 1
        assert MIN_NONSHARP_MODELS["nhl"]["market_types"] == frozenset({"moneyline"})
        # All-of gate would shadow-demote every play whenever one model is dark.
        assert "nhl" not in REQUIRED_BLEND_MODELS

    def test_categories_models_list(self):
        from evmax.categories import get_category

        models = get_category("nhl").models
        assert "elo" in models and "nhl_xg" in models
        assert "form" not in models  # weight 0 → does not contribute


# ── Seed script: prior block + prior-only fallback ─────────────────────────

class TestSeedScript:
    ROWS_2025 = None

    @staticmethod
    def _rows(gp: int) -> list[dict]:
        rows = []
        for i, abbr in enumerate(sorted(NHL_ABBREV_TO_NAME)):
            rows.append({"team": abbr, "situation": "5on5", "games_played": gp,
                         "iceTime": 3600 * 40, "xGoalsFor": 100 + i, "xGoalsAgainst": 110 - i / 2,
                         "goalsFor": 95, "goalsAgainst": 100})
        rows.append({"team": "ATL", "situation": "5on5", "games_played": 1, "iceTime": 3600,
                     "xGoalsFor": 1, "xGoalsAgainst": 1, "goalsFor": 1, "goalsAgainst": 1})
        return rows

    def test_rollover_month(self):
        from scripts.seed_nhl_xg import _current_nhl_season_start_year

        assert _current_nhl_season_start_year(date(2026, 9, 22)) == 2026
        assert _current_nhl_season_start_year(date(2026, 8, 31)) == 2025

    def test_current_season_404_writes_prior_only(self, monkeypatch):
        import httpx
        import scripts.seed_nhl_xg as seed

        def fake_fetch(season):
            if season == 2026:
                req = httpx.Request("GET", "https://x")
                raise httpx.HTTPStatusError("404", request=req, response=httpx.Response(404, request=req))
            return self._rows(82)

        monkeypatch.setattr(seed, "fetch_moneypuck_teams_csv", fake_fetch)
        st = seed.build_state(2026, today=date(2026, 9, 22))["nhl"]
        assert st["mode"] == "prior_only"
        assert st["teams"] == {} and st["league_avg_xg_per_60"] is None
        assert st["season_start_year"] == 2026
        assert st["prior"]["season_start_year"] == 2025
        assert len(st["prior"]["teams"]) == 32
        lg = st["prior"]["league_avg_xg_per_60"]
        t = st["prior"]["teams"]["boston bruins"]
        assert t["xgf_per_60"] == pytest.approx(regress_prior_rate(t["raw_xgf_per_60"], lg), abs=1e-4)

    def test_in_season_state_carries_both_blocks(self, monkeypatch):
        import scripts.seed_nhl_xg as seed

        monkeypatch.setattr(seed, "fetch_moneypuck_teams_csv",
                            lambda s: self._rows(12 if s == 2026 else 82))
        st = seed.build_state(2026, today=date(2026, 10, 26))["nhl"]
        assert st["mode"] == "in_season"
        assert len(st["teams"]) == 32 and st["teams"]["boston bruins"]["gp"] == 12
        assert len(st["prior"]["teams"]) == 32

    def test_prior_failure_aborts_without_state(self, monkeypatch):
        import httpx
        import scripts.seed_nhl_xg as seed

        def boom(season):
            req = httpx.Request("GET", "https://x")
            raise httpx.HTTPStatusError("403", request=req, response=httpx.Response(403, request=req))

        monkeypatch.setattr(seed, "fetch_moneypuck_teams_csv", boom)
        with pytest.raises(httpx.HTTPStatusError):
            seed.build_state(2026)

    def test_thin_prior_aborts(self, monkeypatch):
        import scripts.seed_nhl_xg as seed

        monkeypatch.setattr(seed, "fetch_moneypuck_teams_csv", lambda s: self._rows(82)[:5])
        with pytest.raises(RuntimeError):
            seed.build_state(2026)
