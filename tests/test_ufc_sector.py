"""UFC sector wiring tests — Kalshi parsing, name matching, resolver routing.

Kalshi fixture captured live 2026-07-11 (see tests/fixtures/ufc/README.md).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from evmax.clients.kalshi import KalshiClient, SECTOR_SERIES_MAP
from evmax.clients.time_util import kalshi_game_day
from evmax.matching.engine import MatchingEngine
from evmax.models.market import MarketSource, MarketType, PredictionMarket
from evmax.models.odds import SharpBook, SharpOdds
from evmax.sectors.registry import get_handler

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "ufc"


def _load(name: str) -> dict:
    with open(FIXTURE_DIR / name) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Sector handler — surname normalization
# ---------------------------------------------------------------------------


class TestUFCHandler:
    def test_full_name_to_surname(self):
        h = get_handler("ufc")
        assert h.normalize_team("Conor McGregor") == "mcgregor"
        assert h.normalize_team("Ian Machado Garry") == "garry"

    def test_hyphenated_surname_survives(self):
        h = get_handler("ufc")
        assert h.normalize_team("Benoit Saint-Denis") == "saint-denis"
        assert h.normalize_team("Saint-Denis") == "saint-denis"

    def test_east_asian_name_order_aliases(self):
        h = get_handler("ufc")
        assert h.normalize_team("Zhang Weili") == "zhang"
        assert h.normalize_team("Weili Zhang") == "zhang"

    def test_bare_surname_passthrough(self):
        h = get_handler("ufc")
        assert h.normalize_team("Pimblett") == "pimblett"

    def test_moneyline_only(self):
        assert get_handler("ufc").market_types_supported() == [MarketType.moneyline]


# ---------------------------------------------------------------------------
# Kalshi title parsing
# ---------------------------------------------------------------------------


class TestUFCTitleParsing:
    TITLE_SURNAMES = (
        "Will Benoit Saint-Denis win the Saint-Denis vs Pimblett professional "
        "MMA fight scheduled for Jul 11, 2026?"
    )
    TITLE_FULL_NAMES = (
        "Will Conor McGregor win the Conor McGregor vs Max Holloway "
        "professional MMA fight scheduled for Jul 11, 2026?"
    )

    def test_surname_form(self):
        client = KalshiClient()
        first, second = client._extract_ufc_fighters_from_title(self.TITLE_SURNAMES)
        assert (first, second) == ("Saint-Denis", "Pimblett")

    def test_full_name_form(self):
        client = KalshiClient()
        first, second = client._extract_ufc_fighters_from_title(self.TITLE_FULL_NAMES)
        assert (first, second) == ("Conor McGregor", "Max Holloway")

    def test_yes_fighter_normalized(self):
        client = KalshiClient()
        assert client._extract_tennis_yes_player(self.TITLE_SURNAMES, "ufc") == "saint-denis"
        assert client._extract_tennis_yes_player(self.TITLE_FULL_NAMES, "ufc") == "mcgregor"

    def test_garbage_title(self):
        client = KalshiClient()
        assert client._extract_ufc_fighters_from_title("Will it rain tomorrow?") == (None, None)


class TestKalshiUFCFixture:
    """End-to-end parse of the captured KXUFCFIGHT payload."""

    @pytest.mark.asyncio
    async def test_get_markets_parses_ufc_fights(self, monkeypatch):
        fixture = _load("kxufcfight_markets.json")

        async def fake_get(path, params=None):
            if path == "/markets":
                return fixture
            return {}

        from evmax import settings as settings_module
        settings = settings_module.get_settings()
        monkeypatch.setattr(settings, "offline_mode", False)
        monkeypatch.setattr(settings, "cache_ttl_secs", 0)
        monkeypatch.setattr(settings, "kalshi_ws_enabled", False)

        async with KalshiClient() as client:
            with patch.object(client, "_get", side_effect=fake_get):
                markets = await client.get_markets(sector="ufc")

        assert len(markets) == 4
        by_ticker = {m.ticker: m for m in markets}
        sai = by_ticker["KXUFCFIGHT-26JUL11SAIPIM-SAI"]
        assert sai.market_type == MarketType.moneyline
        assert sai.yes_team == "saint-denis"
        # title order away-first → home = second-listed fighter
        assert sai.team_home == "Pimblett"
        assert sai.team_away == "Saint-Denis"
        assert sai.event_date.date().isoformat() == "2026-07-11"

        mcg = by_ticker["KXUFCFIGHT-26JUL11MCGHOL-MCG"]
        assert mcg.yes_team == "mcgregor"
        assert mcg.team_home == "Max Holloway"
        assert mcg.team_away == "Conor McGregor"

    def test_ufc_series_registered(self):
        assert SECTOR_SERIES_MAP["ufc"] == ["KXUFCFIGHT"]


# ---------------------------------------------------------------------------
# Game-day alignment + matching
# ---------------------------------------------------------------------------


class TestUFCMatching:
    def test_us_calendar_day_for_late_utc_start(self):
        """Pinnacle startTime 2026-07-12T02:50Z is a Jul 11 ET card — the
        game-day must match Kalshi's 26JUL11 ticker date."""
        dt = datetime(2026, 7, 12, 2, 50, tzinfo=timezone.utc)
        assert kalshi_game_day(dt, "ufc") == "2026-07-11"

    def _kalshi_market(self) -> PredictionMarket:
        return PredictionMarket(
            id="kalshi:KXUFCFIGHT-26JUL11SAIPIM-SAI",
            source=MarketSource.kalshi,
            sector="ufc",
            market_type=MarketType.moneyline,
            title="Will Benoit Saint-Denis win the Saint-Denis vs Pimblett professional MMA fight scheduled for Jul 11, 2026?",
            ticker="KXUFCFIGHT-26JUL11SAIPIM-SAI",
            yes_price=0.58,
            no_price=0.45,
            volume_usd=1500.0,
            team_home="Pimblett",
            team_away="Saint-Denis",
            event_date=datetime(2026, 7, 11, 12, tzinfo=timezone.utc),
            yes_team="saint-denis",
        )

    def _pinnacle_sharp(self, home="Paddy Pimblett", away="Benoit Saint-Denis") -> SharpOdds:
        event_date = datetime(2026, 7, 12, 2, 50, tzinfo=timezone.utc)
        date_str = kalshi_game_day(event_date, "ufc")
        from evmax.matching.normalizer import NameNormalizer
        norm = NameNormalizer("ufc")
        home_n = norm.normalize(home).replace(" ", "_")
        away_n = norm.normalize(away).replace(" ", "_")
        return SharpOdds(
            event_id=f"ufc::{date_str}::{home_n}_vs_{away_n}",
            book=SharpBook.pinnacle,
            sector="ufc",
            outcome_a_label=home,
            outcome_b_label=away,
            outcome_a_decimal=1.8,
            outcome_b_decimal=2.1,
            true_prob_a=0.55,
            true_prob_b=0.45,
            margin=0.04,
            event_date=event_date,
        )

    def test_exact_key_match(self):
        engine = MatchingEngine()
        market = self._kalshi_market()
        sharp = self._pinnacle_sharp()
        assert engine.build_market_key(market) == sharp.event_id
        result = engine.match(market, [sharp])
        assert result is not None
        assert result[1] == 100.0

    def test_swapped_home_away_still_matches_fuzzy(self):
        """If Kalshi ever lists the fighters in the opposite order, the
        token-sort fuzzy fallback must still land the match."""
        engine = MatchingEngine()
        market = self._kalshi_market().model_copy(
            update={"team_home": "Saint-Denis", "team_away": "Pimblett"}
        )
        result = engine.match(market, [self._pinnacle_sharp()])
        assert result is not None

    def test_wrong_fight_does_not_match(self):
        engine = MatchingEngine()
        market = self._kalshi_market()
        other = self._pinnacle_sharp(home="Max Holloway", away="Conor McGregor")
        assert engine.match(market, [other]) is None


# ---------------------------------------------------------------------------
# Resolver routing + registry consistency
# ---------------------------------------------------------------------------


class TestUFCWiring:
    def test_resolver_routes_to_kalshi_settlement(self):
        from evmax.agents.cleanup.resolver import (
            ESPN_SPORT_MAP,
            KALSHI_SETTLEMENT_SECTORS,
        )
        assert "ufc" in KALSHI_SETTLEMENT_SECTORS
        # Must NOT be in the ESPN scoreboard path — that branch dispatches
        # first and resolves by score comparison, meaningless for fights.
        assert "ufc" not in ESPN_SPORT_MAP

    def test_category_registered_shadow_kalshi_settlement(self):
        from evmax.categories import get_category
        spec = get_category("ufc")
        assert spec.mode == "shadow"
        assert spec.resolver == "kalshi_settlement"
        assert spec.market_types == (MarketType.moneyline,)
        assert "ufc_rating" in spec.models and "sharp" in spec.models

    def test_ensemble_override_pins_generic_models_to_zero(self):
        from evmax.agents.models.ensemble_agent import EnsembleModelAgent
        override = EnsembleModelAgent.SECTOR_WEIGHT_OVERRIDES["ufc"]
        assert override["ufc_rating"] > 0
        assert override["elo"] == 0.0
        assert override["form"] == 0.0

    def test_no_required_blend_gate_for_ufc(self):
        """A zero-weighted model must never shadow-demote UFC plays — that
        only happens for sectors with a REQUIRED_BLEND_MODELS entry."""
        from evmax.agents.odds.ev_gap_agent import REQUIRED_BLEND_MODELS, has_full_blend
        assert "ufc" not in REQUIRED_BLEND_MODELS
        assert has_full_blend("ufc", "sharp") is True

    def test_coordinator_registers_ufc_agent(self):
        from evmax.agents.coordinator import AgentCoordinator
        coord = AgentCoordinator(sectors=["ufc"], respect_season_window=False)
        assert any(
            type(m).__name__ == "UFCRatingAgent"
            for m in coord.ensemble_agent._models
        )
