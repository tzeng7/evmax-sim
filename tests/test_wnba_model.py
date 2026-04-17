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
        assert "wnba" in SUPPORTED_SECTORS

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
