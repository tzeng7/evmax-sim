"""Tests for WNBA sector handler and model integration.

Covers:
- WNBAHandler registration and basic behavior
- Model parameter validation (Elo, Form, Poisson have WNBA entries)
- Kalshi/Pinnacle config entries
- Injury agent URL presence
- Resolver ESPN_SPORT_MAP entry
"""

from __future__ import annotations

import pytest

from evmax.sectors.registry import get_handler, ALL_SECTORS


class TestWNBASectorRegistration:
    """Verify WNBA is registered and handler works."""

    def test_wnba_in_all_sectors(self):
        assert "wnba" in ALL_SECTORS

    def test_get_handler_returns_wnba(self):
        handler = get_handler("wnba")
        assert handler.name == "wnba"
        assert handler.sharp_source == "pinnacle"

    def test_market_types_supported(self):
        handler = get_handler("wnba")
        types = handler.market_types_supported()
        from evmax.models.market import MarketType
        assert MarketType.moneyline in types
        assert MarketType.spread in types
        assert MarketType.total in types


class TestWNBAModelParameters:
    """Verify all model agents have WNBA-specific parameters."""

    def test_elo_k_factor(self):
        from evmax.agents.models.elo_agent import K_FACTORS
        assert "wnba" in K_FACTORS
        assert K_FACTORS["wnba"] == 20.0

    def test_elo_home_advantage(self):
        from evmax.agents.models.elo_agent import HOME_ADVANTAGE_ELO
        assert "wnba" in HOME_ADVANTAGE_ELO
        assert HOME_ADVANTAGE_ELO["wnba"] == 60.0

    def test_elo_rest_adjustment(self):
        from evmax.agents.models.elo_agent import REST_ELO_ADJ
        assert "wnba" in REST_ELO_ADJ
        assert REST_ELO_ADJ["wnba"][0] == -50.0  # back-to-back penalty

    def test_form_home_adj(self):
        from evmax.agents.models.form_agent import HOME_ADJ
        assert "wnba" in HOME_ADJ
        assert HOME_ADJ["wnba"] == 0.025

    def test_poisson_league_avg(self):
        from evmax.agents.models.poisson_agent import LEAGUE_AVG_DEFAULTS, SUPPORTED_SECTORS
        assert "wnba" in LEAGUE_AVG_DEFAULTS
        assert LEAGUE_AVG_DEFAULTS["wnba"]["home"] == 83.0
        assert LEAGUE_AVG_DEFAULTS["wnba"]["away"] == 81.0
        # Poisson is soccer-only now (the WNBA ensemble override already zeroed
        # it; keeping it out of SUPPORTED_SECTORS stops the wasted prediction
        # pass and the misleading model_sources entry). League-avg constants are
        # retained for reference but the agent no longer runs on WNBA.
        assert "wnba" not in SUPPORTED_SECTORS

    def test_poisson_max_score_and_bucket(self):
        from evmax.agents.models.poisson_agent import MAX_SCORE, BUCKET_SIZE
        assert "wnba" in MAX_SCORE
        assert MAX_SCORE["wnba"] == 20
        assert "wnba" in BUCKET_SIZE
        assert BUCKET_SIZE["wnba"] == 5


class TestWNBAClientConfig:
    """Verify Kalshi and Pinnacle configs include WNBA."""

    def test_kalshi_series_map(self):
        from evmax.clients.kalshi import SECTOR_SERIES_MAP
        assert "wnba" in SECTOR_SERIES_MAP
        assert "KXWNBAGAME" in SECTOR_SERIES_MAP["wnba"]

    def test_pinnacle_sport_leagues(self):
        from evmax.clients.esports_pinnacle import SECTOR_SPORT_LEAGUES
        assert "wnba" in SECTOR_SPORT_LEAGUES
        sport_id, league_ids = SECTOR_SPORT_LEAGUES["wnba"]
        assert sport_id == 4  # Basketball
        assert 578 in league_ids


class TestWNBAInjuryAndResolver:
    """Verify injury agent and resolver have WNBA entries."""

    def test_injury_url_present(self):
        from evmax.agents.intelligence.injury_agent import SECTOR_INJURY_URLS
        assert "wnba" in SECTOR_INJURY_URLS
        assert any("wnba" in url for url in SECTOR_INJURY_URLS["wnba"])

    def test_resolver_espn_sport_map(self):
        from evmax.agents.cleanup.resolver import ESPN_SPORT_MAP
        assert "wnba" in ESPN_SPORT_MAP
        sport, league, _ = ESPN_SPORT_MAP["wnba"]
        assert sport == "basketball"
        assert league == "wnba"


class TestWNBASeedConfig:
    """Verify seed script has WNBA configuration."""

    def test_sector_config_exists(self):
        # Import inline to avoid side effects
        import importlib
        import sys
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
        # We can't easily import the script, but we can check the file exists
        from pathlib import Path
        seed_path = Path(__file__).resolve().parents[1] / "scripts" / "seed_espn.py"
        content = seed_path.read_text()
        assert '"wnba"' in content
        assert '"basketball"' in content
        assert '"wnba"' in content


class TestWNBAEloPredict:
    """Smoke test: Elo agent can predict a WNBA matchup."""

    @pytest.mark.asyncio
    async def test_predict_pair_wnba(self):
        from evmax.agents.models.elo_agent import EloModelAgent
        agent = EloModelAgent()
        # Seed two teams
        agent._state = {"wnba": {
            "ratings": {"aces": 1600.0, "liberty": 1550.0},
            "game_counts": {"aces": 20, "liberty": 20},
            "h2h": {},
        }}
        from evmax.models.market import MarketSource, MarketType, PredictionMarket
        from evmax.models.odds import SharpBook, SharpOdds

        market = PredictionMarket(
            id="test-wnba-001",
            event_id="wnba::2026-06-01::aces_vs_liberty",
            ticker="KXWNBAGAME-26JUN01LVKNY-LVA",
            title="Las Vegas Aces vs New York Liberty",
            yes_price=0.55,
            no_price=0.45,
            sector="wnba",
            source=MarketSource.kalshi,
            market_type=MarketType.moneyline,
            team_home="aces",
            team_away="liberty",
            yes_team="aces",
        )
        sharp = SharpOdds(
            event_id="wnba::2026-06-01::aces_vs_liberty",
            book=SharpBook.pinnacle,
            sector="wnba",
            team_a="aces",
            team_b="liberty",
            true_prob_a=0.58,
            true_prob_b=0.42,
            outcome_a_decimal=1.72,
            outcome_b_decimal=2.38,
        )
        pred = await agent.predict_pair(market, sharp)
        assert pred is not None
        # Aces rated higher + home advantage → should favor aces
        assert pred.true_prob_a > 0.5
        assert pred.true_prob_b < 0.5
        assert abs(pred.true_prob_a + pred.true_prob_b - 1.0) < 0.01


class TestWNBAEfficiencyStalenessGuard:
    """Regression: prior-season state must not leak into in-season predictions.

    Background: the 2026 season opened on 2025-end-of-season ratings because
    `scripts/wnba_offseason_regress.py` only touches Elo, not efficiency.
    The result was +24pp chalk bias over the opening month. The staleness
    guard returns None whenever source_season < current calendar year during
    May–Oct, forcing the ensemble to fall back to sharp until a fresh seed.
    """

    def test_stale_state_returns_true_during_season(self):
        from datetime import date
        from evmax.agents.models.wnba_efficiency_agent import state_is_stale_for_today

        # May–Oct, source_season behind current year → stale
        stale = {"source_season": "2025"}
        assert state_is_stale_for_today(stale, today=date(2026, 5, 30)) is True
        assert state_is_stale_for_today(stale, today=date(2026, 10, 1)) is True

    def test_fresh_state_returns_false(self):
        from datetime import date
        from evmax.agents.models.wnba_efficiency_agent import state_is_stale_for_today

        fresh = {"source_season": "2026"}
        assert state_is_stale_for_today(fresh, today=date(2026, 5, 30)) is False

    def test_offseason_does_not_trigger_guard(self):
        from datetime import date
        from evmax.agents.models.wnba_efficiency_agent import state_is_stale_for_today

        # In Jan / Mar there are no games, so prior-year state is fine.
        stale = {"source_season": "2025"}
        assert state_is_stale_for_today(stale, today=date(2026, 1, 15)) is False
        assert state_is_stale_for_today(stale, today=date(2026, 4, 30)) is False

    def test_missing_or_bad_source_season_does_not_block(self):
        from datetime import date
        from evmax.agents.models.wnba_efficiency_agent import state_is_stale_for_today

        # Backward compat: state files without a source_season key (or with
        # a non-numeric value) should not be treated as stale.
        assert state_is_stale_for_today({}, today=date(2026, 6, 1)) is False
        assert state_is_stale_for_today({"source_season": None}, today=date(2026, 6, 1)) is False
        assert state_is_stale_for_today({"source_season": "unknown"}, today=date(2026, 6, 1)) is False

    def test_shrinkage_pulls_extreme_ratings_toward_mean(self):
        from evmax.agents.models.wnba_efficiency_agent import shrink_team_stats, SHRINK_K

        league = {"league_avg_ortg": 102.0, "league_avg_drtg": 102.0, "league_avg_pace": 82.0}
        # Extreme team with small sample (mirrors Sun's 2026 -15 NET at gp=11)
        stats = {"ortg": 89.0, "drtg": 105.0, "pace": 80.0, "gp": 11}
        out = shrink_team_stats(stats, league)
        # Sample weight = 11/(11+8) ≈ 0.579, prior weight ≈ 0.421
        expected_ortg = 0.579 * 89.0 + 0.421 * 102.0
        assert abs(out["ortg"] - expected_ortg) < 0.1
        # ORTG should be pulled UP toward 102 (away from 89)
        assert 89.0 < out["ortg"] < 102.0

    def test_shrinkage_disappears_with_large_sample(self):
        from evmax.agents.models.wnba_efficiency_agent import shrink_team_stats

        league = {"league_avg_ortg": 102.0, "league_avg_drtg": 102.0, "league_avg_pace": 82.0}
        stats = {"ortg": 110.0, "drtg": 95.0, "pace": 84.0, "gp": 100}
        out = shrink_team_stats(stats, league)
        # At gp=100 weight is 100/108 ≈ 0.926 — raw dominates
        assert out["ortg"] > 109.0     # mostly 110
        assert out["drtg"] < 96.0       # mostly 95

    def test_shrinkage_handles_zero_gp(self):
        from evmax.agents.models.wnba_efficiency_agent import shrink_team_stats

        league = {"league_avg_ortg": 102.0, "league_avg_drtg": 102.0, "league_avg_pace": 82.0}
        out = shrink_team_stats({"ortg": 110.0, "drtg": 95.0, "pace": 82.0, "gp": 0}, league)
        # No data → no opinion to update; pass through unchanged
        assert out["ortg"] == 110.0

    def test_smooth_confidence_ramps_with_sample_size(self):
        from evmax.agents.models.wnba_efficiency_agent import smooth_confidence

        assert smooth_confidence(0) == pytest.approx(0.40)
        assert smooth_confidence(40) == pytest.approx(0.80)
        assert smooth_confidence(100) == pytest.approx(0.80)  # capped
        # Clears the ensemble's 0.45 gate at gp >= 6
        assert smooth_confidence(6) > 0.45
        # Strictly monotone between the endpoints
        assert smooth_confidence(10) < smooth_confidence(20) < smooth_confidence(30)

    @pytest.mark.asyncio
    async def test_efficiency_agent_returns_none_on_stale_state(self, monkeypatch):
        """End-to-end: stale state in agent makes predict_pair return None."""
        from datetime import date
        from evmax.agents.models import wnba_efficiency_agent as mod
        from evmax.agents.models.wnba_efficiency_agent import WNBAEfficiencyModelAgent
        from evmax.models.market import MarketSource, MarketType, PredictionMarket
        from evmax.models.odds import SharpBook, SharpOdds

        # Freeze "today" inside the agent module's date class
        class FrozenDate(date):
            @classmethod
            def today(cls):
                return date(2026, 6, 1)
        monkeypatch.setattr(mod, "date", FrozenDate)

        agent = WNBAEfficiencyModelAgent()
        agent._state = {
            "source_season": "2025",
            "league_avg_ortg": 100.6,
            "league_avg_drtg": 100.6,
            "teams": {
                "aces": {"ortg": 110.0, "drtg": 95.0, "pace": 82.0, "gp": 50,
                         "efg": 0.52, "tov_pct": 0.15, "oreb_pct": 0.25, "ftr": 0.28,
                         "full_name": "las vegas aces"},
                "liberty": {"ortg": 100.0, "drtg": 100.0, "pace": 82.0, "gp": 50,
                            "efg": 0.50, "tov_pct": 0.16, "oreb_pct": 0.22, "ftr": 0.28,
                            "full_name": "new york liberty"},
            },
        }

        market = PredictionMarket(
            id="test-stale-001",
            event_id="wnba::2026-06-01::aces_vs_liberty",
            ticker="KXWNBAGAME-26JUN01LVKNY-LVA",
            title="Las Vegas Aces vs New York Liberty",
            yes_price=0.6, no_price=0.4, sector="wnba",
            source=MarketSource.kalshi, market_type=MarketType.moneyline,
            team_home="aces", team_away="liberty", yes_team="aces",
        )
        sharp = SharpOdds(
            event_id="wnba::2026-06-01::aces_vs_liberty",
            book=SharpBook.pinnacle, sector="wnba",
            team_a="aces", team_b="liberty",
            true_prob_a=0.6, true_prob_b=0.4,
            outcome_a_decimal=1.67, outcome_b_decimal=2.50,
            outcome_a_label="aces", outcome_b_label="liberty",
        )
        pred = await agent.predict_pair(market, sharp)
        assert pred is None, "stale source_season should silence the agent in-season"
