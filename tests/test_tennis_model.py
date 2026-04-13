"""Tests for TennisModelAgent — surface-aware Elo for ATP/WTA matches.

Covers:
  - Sector gate (non-tennis sectors are a no-op)
  - Surface detection from event titles (clay/grass/indoor/hard fallback)
  - Player resolution: surname match, multi-word surnames, apostrophes, normalization
  - Rating fallback chain: surface → overall → ATP rank → WTA rank → default 1500
  - Predict probabilities sum to 1.0 and favor the higher-rated player
  - Confidence tiering (sufficient surface games / overall only / rank-only / unranked)
  - Elo update math: winner gains, loser loses, zero-sum across many matches
  - Seeding helpers: seed_rankings, seed_surface_ratings
  - ranking_to_elo monotonicity and floor

Also includes a test that *documents* the MODEL-1 bug from TODO.md: when a Kalshi
title contains only player names (no tournament context), surface detection silently
defaults to "hard". The test currently asserts the buggy behavior so that fixing
MODEL-1 will require updating this test alongside the fix — making the regression
visible at code-review time.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from evmax.agents.models.tennis_model_agent import (
    DEFAULT_ELO,
    DEFAULT_SURFACE,
    K_FACTOR,
    MIN_OVERALL_GAMES,
    MIN_SURFACE_GAMES,
    TennisModelAgent,
    ranking_to_elo,
)
from evmax.models.market import MarketSource, MarketType, PredictionMarket
from evmax.models.odds import SharpBook, SharpOdds


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def agent(tmp_path, monkeypatch):
    """A fresh TennisModelAgent backed by a tmp_path state dir."""
    monkeypatch.setattr("evmax.agents.models.base.STATE_DIR", tmp_path)
    a = TennisModelAgent()
    a._state_path = tmp_path / "tennis_surface_state.json"
    a._state = {}
    return a


def make_tennis_market(
    title: str = "Sinner vs Alcaraz",
    player_home: str = "sinner",
    player_away: str = "alcaraz",
) -> PredictionMarket:
    return PredictionMarket(
        id=f"kalshi:{player_home}_vs_{player_away}",
        source=MarketSource.kalshi,
        sector="tennis",
        market_type=MarketType.moneyline,
        title=title,
        ticker="KXATPMATCH-TEST",
        yes_price=0.5,
        no_price=0.5,
        volume_usd=10_000.0,
        team_home=player_home,
        team_away=player_away,
        yes_team=player_home,
        event_date=datetime(2026, 6, 5, 14, 0, tzinfo=timezone.utc),
    )


def make_tennis_sharp(
    player_a: str = "sinner",
    player_b: str = "alcaraz",
    prob_a: float = 0.55,
) -> SharpOdds:
    prob_b = 1.0 - prob_a
    return SharpOdds(
        event_id=f"tennis::2026-06-05::{player_a}_vs_{player_b}",
        book=SharpBook.pinnacle,
        sector="tennis",
        outcome_a_label=player_a,
        outcome_b_label=player_b,
        outcome_a_decimal=1.0 / prob_a,
        outcome_b_decimal=1.0 / prob_b,
        true_prob_a=prob_a,
        true_prob_b=prob_b,
        margin=0.04,
        event_date=datetime(2026, 6, 5, 14, 0, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# Sector gate
# ---------------------------------------------------------------------------

class TestSectorGate:
    @pytest.mark.asyncio
    async def test_non_tennis_sector_returns_none(self, agent):
        market = make_tennis_market()
        market = market.model_copy(update={"sector": "nba"})
        sharp = make_tennis_sharp()
        sharp = sharp.model_copy(update={"sector": "nba"})

        pred = await agent.predict_pair(market, sharp)
        assert pred is None

    @pytest.mark.asyncio
    async def test_empty_player_names_returns_none(self, agent):
        market = make_tennis_market(player_home="", player_away="")
        sharp = make_tennis_sharp()
        sharp = sharp.model_copy(
            update={"outcome_a_label": "", "outcome_b_label": ""}
        )

        pred = await agent.predict_pair(market, sharp)
        assert pred is None

    def test_update_non_tennis_is_noop(self, agent):
        agent.update("lakers", "celtics", 2, 1, "nba")
        # No state should have been written
        assert agent._state == {}


# ---------------------------------------------------------------------------
# Surface detection
# ---------------------------------------------------------------------------

class TestSurfaceDetection:
    def test_default_surface_is_hard(self):
        assert DEFAULT_SURFACE == "hard"

    def test_roland_garros_is_clay(self):
        assert TennisModelAgent._detect_surface("French Open 2026") == "clay"
        assert TennisModelAgent._detect_surface("Roland Garros R2") == "clay"

    def test_madrid_monte_carlo_rome_are_clay(self):
        assert TennisModelAgent._detect_surface("Madrid Open Final") == "clay"
        assert TennisModelAgent._detect_surface("Monte Carlo Masters") == "clay"
        assert TennisModelAgent._detect_surface("Rome ATP 1000") == "clay"

    def test_wimbledon_is_grass(self):
        assert TennisModelAgent._detect_surface("Wimbledon Day 5") == "grass"
        assert TennisModelAgent._detect_surface("Queen's Club Final") == "grass"
        assert TennisModelAgent._detect_surface("Halle Open") == "grass"

    def test_atp_finals_is_indoor(self):
        assert TennisModelAgent._detect_surface("ATP Finals") == "indoor"
        assert TennisModelAgent._detect_surface("Paris Masters R3") == "indoor"
        assert TennisModelAgent._detect_surface("Rotterdam Open") == "indoor"

    def test_us_open_falls_back_to_hard(self):
        # US Open is hard but not in keyword list; default fallback handles it
        assert TennisModelAgent._detect_surface("US Open Day 1") == "hard"

    def test_australian_open_falls_back_to_hard(self):
        assert TennisModelAgent._detect_surface("Australian Open R4") == "hard"

    def test_case_insensitive(self):
        assert TennisModelAgent._detect_surface("WIMBLEDON FINAL") == "grass"
        assert TennisModelAgent._detect_surface("french OPEN") == "clay"

    def test_empty_title_defaults_to_hard(self):
        assert TennisModelAgent._detect_surface("") == "hard"

    def test_unknown_tournament_defaults_to_hard(self):
        assert TennisModelAgent._detect_surface("Some Random Event") == "hard"

    def test_known_bug_title_without_tournament_silently_defaults_to_hard(self):
        """MODEL-1: when Kalshi titles are just 'Player A vs Player B' with no
        tournament context (the common case), the surface always defaults to hard.

        This means surface-specific Elo (e.g. Nadal's clay rating) is rarely used
        in practice. Documented here so the fix shows up as a test diff.
        """
        assert TennisModelAgent._detect_surface("Nadal vs Alcaraz") == "hard"
        # Even when a Kalshi ticker is in the title, no surface signal:
        assert TennisModelAgent._detect_surface("Sinner v. Djokovic ML YES") == "hard"


# ---------------------------------------------------------------------------
# Player resolution (surname / apostrophe / multi-word / normalization)
# ---------------------------------------------------------------------------

class TestPlayerResolution:
    def test_exact_match(self, agent):
        store = {"sinner": 1800.0, "alcaraz": 1820.0}
        assert agent._resolve_player("sinner", store) == "sinner"

    def test_surname_match_long_to_short(self, agent):
        # "jannik sinner" → "sinner j." (tennis-data short form)
        store = {"sinner j.": 1850.0, "alcaraz c.": 1860.0}
        assert agent._resolve_player("jannik sinner", store) == "sinner j."

    def test_surname_match_short_to_long(self, agent):
        store = {"jannik sinner": 1850.0}
        assert agent._resolve_player("sinner", store) == "jannik sinner"

    def test_multi_word_surname_prefix(self, agent):
        # 'de minaur' should match 'de minaur a.'
        store = {"de minaur a.": 1700.0}
        assert agent._resolve_player("de minaur", store) == "de minaur a."

    def test_apostrophe_normalization(self, agent):
        # 'oconnell' should match "o'connell" (or vice versa)
        store = {"o'connell c.": 1500.0}
        # _normalize_name strips apostrophes and spaces
        # _surname_key on "o'connell c." → "o'connell" → normalized "oconnell"
        # _normalize_name("oconnell") → "oconnell" → match
        assert agent._resolve_player("oconnell", store) == "o'connell c."

    def test_no_match_returns_none(self, agent):
        store = {"sinner": 1800.0}
        assert agent._resolve_player("federer", store) is None

    def test_duplicate_surname_picks_higher_game_count(self, agent):
        """When two state keys share the same surname, prefer the one with more games."""
        agent._state["game_counts"] = {
            "adrian mannarino": {"hard": 5, "overall": 5},
            "mannarino a.": {"hard": 50, "overall": 50},
        }
        store = {"adrian mannarino": 1500.0, "mannarino a.": 1620.0}
        resolved = agent._resolve_player("mannarino", store)
        assert resolved == "mannarino a."


# ---------------------------------------------------------------------------
# Rating fallback chain
# ---------------------------------------------------------------------------

class TestRatingFallback:
    def test_unknown_player_returns_default(self, agent):
        assert agent.get_rating("nobody", "hard") == DEFAULT_ELO

    def test_surface_specific_rating_used_when_present(self, agent):
        agent.seed_surface_ratings("clay", {"nadal": 1950.0})
        assert agent.get_rating("nadal", "clay") == 1950.0

    def test_falls_back_to_overall_when_no_surface_rating(self, agent):
        agent.seed_surface_ratings("overall", {"djokovic": 1880.0})
        # Asking for clay rating but only overall exists
        assert agent.get_rating("djokovic", "clay") == 1880.0

    def test_falls_back_to_atp_rank_when_no_elo(self, agent):
        agent.seed_rankings({"sinner": 1}, tour="atp")
        # No Elo at all — should use ranking prior
        rating = agent.get_rating("sinner", "hard")
        assert rating == ranking_to_elo(1)
        assert rating == 1900.0

    def test_falls_back_to_wta_rank_when_atp_missing(self, agent):
        agent.seed_rankings({"swiatek": 1}, tour="wta")
        rating = agent.get_rating("swiatek", "clay")
        assert rating == 1900.0

    def test_default_elo_when_no_signal(self, agent):
        rating = agent.get_rating("totally_unknown_player", "hard")
        assert rating == DEFAULT_ELO == 1500.0


# ---------------------------------------------------------------------------
# ranking_to_elo
# ---------------------------------------------------------------------------

class TestRankingToElo:
    def test_none_returns_default(self):
        assert ranking_to_elo(None) == DEFAULT_ELO

    def test_rank_1_is_highest(self):
        assert ranking_to_elo(1) == 1900.0

    def test_monotonically_decreasing(self):
        # Each step from rank 1 to rank 200 should be non-increasing
        prev = ranking_to_elo(1)
        for r in range(2, 201):
            cur = ranking_to_elo(r)
            assert cur <= prev, f"rank {r}: {cur} > rank {r-1}: {prev}"
            prev = cur

    def test_rank_5_step(self):
        assert ranking_to_elo(5) == 1860.0

    def test_rank_100_floor(self):
        # rank 100 lands at 1500 per the formula
        assert ranking_to_elo(100) == pytest.approx(1500.0, abs=1.0)

    def test_deep_rank_has_floor(self):
        # rank 9999 should still be ≥ 1350 (the hard floor)
        assert ranking_to_elo(9999) == 1350.0


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

class TestPrediction:
    @pytest.mark.asyncio
    async def test_probs_sum_to_one(self, agent):
        agent.seed_surface_ratings("hard", {"sinner": 1800.0, "alcaraz": 1820.0})
        agent.seed_surface_ratings(
            "overall", {"sinner": 1800.0, "alcaraz": 1820.0}
        )
        market = make_tennis_market()
        sharp = make_tennis_sharp()

        pred = await agent.predict_pair(market, sharp)
        assert pred is not None
        assert abs(pred.true_prob_a + pred.true_prob_b - 1.0) < 1e-6
        assert pred.true_prob_draw is None

    @pytest.mark.asyncio
    async def test_higher_elo_player_is_favored(self, agent):
        agent.seed_surface_ratings("hard", {"sinner": 1900.0, "alcaraz": 1700.0})
        market = make_tennis_market()
        sharp = make_tennis_sharp()

        pred = await agent.predict_pair(market, sharp)
        assert pred is not None
        assert pred.true_prob_a > pred.true_prob_b
        assert pred.true_prob_a > 0.5

    @pytest.mark.asyncio
    async def test_equal_elo_gives_50_50(self, agent):
        agent.seed_surface_ratings("hard", {"sinner": 1800.0, "alcaraz": 1800.0})
        market = make_tennis_market()
        sharp = make_tennis_sharp()

        pred = await agent.predict_pair(market, sharp)
        assert pred is not None
        assert abs(pred.true_prob_a - 0.5) < 1e-6

    @pytest.mark.asyncio
    async def test_elo_400_gap_gives_about_91_percent(self, agent):
        """Standard Elo: 400-point gap → ~90.9% favorite."""
        agent.seed_surface_ratings("hard", {"sinner": 2000.0, "alcaraz": 1600.0})
        market = make_tennis_market()
        sharp = make_tennis_sharp()

        pred = await agent.predict_pair(market, sharp)
        assert pred is not None
        assert pred.true_prob_a == pytest.approx(0.909, abs=0.01)

    @pytest.mark.asyncio
    async def test_surface_used_when_title_has_tournament(self, agent):
        # Stack the deck: clay rating heavily favors player_a, hard rating reverses
        agent.seed_surface_ratings("clay", {"nadal": 2100.0, "djokovic": 1700.0})
        agent.seed_surface_ratings("hard", {"nadal": 1700.0, "djokovic": 2100.0})

        clay_market = make_tennis_market(
            title="Roland Garros Quarterfinal",
            player_home="nadal",
            player_away="djokovic",
        )
        clay_sharp = make_tennis_sharp(player_a="nadal", player_b="djokovic")

        pred_clay = await agent.predict_pair(clay_market, clay_sharp)
        assert pred_clay is not None
        assert pred_clay.true_prob_a > 0.85  # clay → Nadal heavily favored
        assert "surface=clay" in pred_clay.notes

        hard_market = clay_market.model_copy(
            update={"title": "US Open Quarterfinal"}
        )
        pred_hard = await agent.predict_pair(hard_market, clay_sharp)
        assert pred_hard is not None
        assert pred_hard.true_prob_a < 0.15  # hard → Djokovic heavily favored
        assert "surface=hard" in pred_hard.notes


# ---------------------------------------------------------------------------
# Confidence tiering
# ---------------------------------------------------------------------------

class TestConfidenceTiering:
    @pytest.mark.asyncio
    async def test_unranked_unknown_players_below_gate(self, agent):
        market = make_tennis_market(player_home="nobody1", player_away="nobody2")
        sharp = make_tennis_sharp(player_a="nobody1", player_b="nobody2")

        pred = await agent.predict_pair(market, sharp)
        assert pred is not None
        # 0.30 is below the 0.45 ensemble gate → ensemble will exclude this
        assert pred.confidence == 0.30

    @pytest.mark.asyncio
    async def test_ranked_players_above_gate(self, agent):
        agent.seed_rankings({"sinner": 1, "alcaraz": 2}, tour="atp")
        market = make_tennis_market()
        sharp = make_tennis_sharp()

        pred = await agent.predict_pair(market, sharp)
        assert pred is not None
        # 0.48 is just above the 0.45 ensemble gate
        assert pred.confidence == 0.48

    @pytest.mark.asyncio
    async def test_overall_data_only_gives_50(self, agent):
        # Both players have enough overall games but no surface games
        for _ in range(MIN_OVERALL_GAMES + 1):
            agent.update(
                "sinner", "alcaraz", 2, 1, "tennis", surface="overall"
            )

        market = make_tennis_market()
        sharp = make_tennis_sharp()
        pred = await agent.predict_pair(market, sharp)
        assert pred is not None
        assert pred.confidence == 0.50

    @pytest.mark.asyncio
    async def test_sufficient_surface_data_above_55(self, agent):
        # Need MIN_SURFACE_GAMES on the detected surface (hard, since title has no tournament)
        for _ in range(MIN_SURFACE_GAMES + 2):
            agent.update("sinner", "alcaraz", 2, 1, "tennis", surface="hard")

        market = make_tennis_market(title="Some hard court tournament")
        sharp = make_tennis_sharp()
        pred = await agent.predict_pair(market, sharp)
        assert pred is not None
        assert pred.confidence > 0.55
        assert pred.confidence <= 0.80


# ---------------------------------------------------------------------------
# Update / Elo math
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_winner_gains_loser_loses(self, agent):
        agent.update("sinner", "alcaraz", 2, 0, "tennis", surface="clay")
        sinner = agent.get_rating("sinner", "clay")
        alcaraz = agent.get_rating("alcaraz", "clay")
        assert sinner > DEFAULT_ELO
        assert alcaraz < DEFAULT_ELO

    def test_zero_sum_per_match(self, agent):
        agent.update("sinner", "alcaraz", 2, 0, "tennis", surface="clay")
        sinner = agent.get_rating("sinner", "clay")
        alcaraz = agent.get_rating("alcaraz", "clay")
        # Zero-sum on the surface ratings (rounded to 2 decimals)
        assert sinner + alcaraz == pytest.approx(3000.0, abs=0.01)

    def test_overall_also_updated(self, agent):
        agent.update("sinner", "alcaraz", 2, 0, "tennis", surface="clay")
        # Overall rating should also have moved (smaller K)
        sinner_overall = agent.get_rating("sinner", "overall")
        alcaraz_overall = agent.get_rating("alcaraz", "overall")
        assert sinner_overall > DEFAULT_ELO
        assert alcaraz_overall < DEFAULT_ELO
        # Surface delta should be larger than overall delta (overall uses 0.6 × K)
        sinner_clay = agent.get_rating("sinner", "clay")
        clay_delta = sinner_clay - DEFAULT_ELO
        overall_delta = sinner_overall - DEFAULT_ELO
        assert clay_delta > overall_delta

    def test_game_counts_incremented(self, agent):
        agent.update("sinner", "alcaraz", 2, 0, "tennis", surface="clay")
        agent.update("sinner", "alcaraz", 2, 1, "tennis", surface="clay")

        counts = agent._state["game_counts"]
        assert counts["sinner"]["clay"] == 2
        assert counts["sinner"]["overall"] == 2
        assert counts["alcaraz"]["clay"] == 2

    def test_max_k_factor_per_match(self, agent):
        # An upset (lower-rated wins) should produce a delta close to but ≤ K_FACTOR
        agent.seed_surface_ratings(
            "hard", {"underdog": 1500.0, "favorite": 1900.0}
        )
        agent.update("underdog", "favorite", 2, 0, "tennis", surface="hard")
        # Expected = ~0.09 for underdog → delta ≈ 24 * (1 - 0.09) ≈ 21.8
        new_underdog = agent.get_rating("underdog", "hard")
        delta = new_underdog - 1500.0
        assert 0 < delta <= K_FACTOR


# ---------------------------------------------------------------------------
# Seeding helpers
# ---------------------------------------------------------------------------

class TestSeeding:
    def test_seed_atp_rankings_persist(self, agent):
        agent.seed_rankings({"sinner": 1, "alcaraz": 2}, tour="atp")
        assert agent._state["atp_rankings"]["sinner"] == 1
        assert agent._state["atp_rankings"]["alcaraz"] == 2

    def test_seed_wta_rankings_separate_from_atp(self, agent):
        agent.seed_rankings({"sinner": 1}, tour="atp")
        agent.seed_rankings({"swiatek": 1}, tour="wta")
        assert agent._state["atp_rankings"]["sinner"] == 1
        assert agent._state["wta_rankings"]["swiatek"] == 1
        assert "swiatek" not in agent._state["atp_rankings"]

    def test_seed_surface_ratings_normalize_lowercase(self, agent):
        agent.seed_surface_ratings("CLAY", {"NADAL": 1950.0})
        # Both surface and player should be lowercased
        assert agent.get_rating("nadal", "clay") == 1950.0

    def test_all_ratings_returns_copy(self, agent):
        agent.seed_surface_ratings("hard", {"sinner": 1800.0})
        snapshot = agent.all_ratings("hard")
        snapshot["sinner"] = 9999.0
        # Mutation should not leak back into agent state
        assert agent.get_rating("sinner", "hard") == 1800.0
