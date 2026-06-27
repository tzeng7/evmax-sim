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
# Surface resolution (MODEL-1)
#
# The resolver replaces the old _detect_surface() keyword scan. It reads
# Kalshi's structured event.product_metadata.competition field (e.g. "ATP
# Munich", "WTA Rouen") and falls back to title only as a safety net.
# Returns a (surface, is_indoor) tuple where is_indoor is a seam for MODEL-6.
# ---------------------------------------------------------------------------

class TestResolveSurface:
    def test_default_surface_constant(self):
        assert DEFAULT_SURFACE == "hard"

    # --- Primary signal: Kalshi competition string ---

    @pytest.mark.parametrize(
        "competition,expected_surface",
        [
            # Clay — observed live on Kalshi 2026-04-13
            ("ATP Munich", "clay"),
            ("ATP Barcelona", "clay"),
            ("WTA Rouen", "clay"),
            ("WTA Stuttgart", "clay"),
            # Clay — major tour stops
            ("ATP Roland Garros", "clay"),
            ("WTA French Open", "clay"),
            ("ATP Monte Carlo", "clay"),
            ("ATP Madrid", "clay"),
            ("ATP Rome", "clay"),
            ("ATP Hamburg", "clay"),
            # Grass
            ("ATP Wimbledon", "grass"),
            ("WTA Wimbledon", "grass"),
            ("ATP Queen's Club", "grass"),
            ("ATP Halle", "grass"),
            ("WTA Eastbourne", "grass"),
            # Hard (default fallback — not in dict)
            ("ATP Australian Open", "hard"),
            ("US Open", "hard"),
            ("ATP Indian Wells", "hard"),
            ("ATP Miami", "hard"),
            ("ATP Cincinnati", "hard"),
            ("WTA Dubai", "hard"),
        ],
    )
    def test_competition_maps_to_surface(self, competition, expected_surface):
        surface, _is_indoor = TennisModelAgent._resolve_surface(competition=competition)
        assert surface == expected_surface, (
            f"expected {competition!r} → {expected_surface}, got {surface}"
        )

    # --- Indoor flag (seam for MODEL-6) ---

    @pytest.mark.parametrize(
        "competition,expected_indoor",
        [
            ("ATP Paris Masters", True),
            ("ATP Paris Bercy", True),
            ("ATP Rotterdam", True),
            ("ATP Vienna", True),
            ("ATP Basel", True),
            ("Nitto ATP Finals", True),
            ("ATP Finals", True),
            ("WTA Finals", True),
            # Outdoor hard — not indoor
            ("US Open", False),
            ("ATP Australian Open", False),
            ("ATP Miami", False),
            # Clay/grass → never indoor
            ("ATP Roland Garros", False),
            ("ATP Wimbledon", False),
            ("ATP Munich", False),
        ],
    )
    def test_is_indoor_flag(self, competition, expected_indoor):
        _surface, is_indoor = TennisModelAgent._resolve_surface(competition=competition)
        assert is_indoor == expected_indoor

    def test_indoor_flag_requires_hard_surface(self):
        """Clay/grass events never get is_indoor=True, even if the competition
        string contains an indoor-city substring. On tour, clay/grass are
        always outdoor — the resolver enforces this invariant."""
        # A synthetic edge case: if someone constructed "ATP Vienna Roland Garros"
        # (nonsense but tests the gate), clay wins and indoor stays False.
        surface, indoor = TennisModelAgent._resolve_surface(
            competition="ATP Vienna Roland Garros"
        )
        assert surface == "clay"
        assert indoor is False

    # --- Null / empty inputs ---

    def test_none_competition_defaults_to_hard(self):
        assert TennisModelAgent._resolve_surface(competition=None) == ("hard", False)

    def test_empty_competition_defaults_to_hard(self):
        assert TennisModelAgent._resolve_surface(competition="") == ("hard", False)

    def test_both_none(self):
        assert TennisModelAgent._resolve_surface(competition=None, title=None) == ("hard", False)

    # --- Title fallback (secondary signal) ---

    def test_title_fallback_when_competition_missing(self):
        """If competition is None (non-tennis sector or older cached data),
        the resolver falls back to scanning the market title."""
        surface, _ = TennisModelAgent._resolve_surface(
            competition=None,
            title="French Open 2026 — Round 2",
        )
        assert surface == "clay"

    def test_competition_takes_precedence_over_title(self):
        """Competition is the primary signal; title is only a fallback."""
        surface, _ = TennisModelAgent._resolve_surface(
            competition="ATP Wimbledon",   # grass
            title="something about clay",  # misleading
        )
        assert surface == "grass"

    # --- Paris ambiguity (Roland Garros vs Paris Masters) ---

    def test_paris_masters_is_indoor_hard(self):
        surface, indoor = TennisModelAgent._resolve_surface(competition="ATP Paris Masters")
        assert (surface, indoor) == ("hard", True)

    def test_roland_garros_is_outdoor_clay_even_though_in_paris(self):
        """Roland Garros is played in Paris but is outdoor clay. The
        resolver must not flip it to indoor via a 'paris' substring match.
        This is enforced by omitting bare 'paris' from INDOOR_CITIES."""
        surface, indoor = TennisModelAgent._resolve_surface(competition="ATP Roland Garros")
        assert surface == "clay"
        assert indoor is False

    # --- MODEL-1 flipped regression test ---

    def test_resolver_uses_competition_not_title(self):
        """MODEL-1 flipped regression: the old _detect_surface scanned only
        the market title, and Kalshi titles are generic ("Will X win ...")
        with no tournament context, so surface always defaulted to 'hard'.

        The fix routes surface detection through Kalshi's structured
        event.product_metadata.competition field. With the fix, a generic
        title plus a populated competition string resolves correctly —
        proving that surface-specific Elo is now exercised.
        """
        generic_title = "Will Jannik Sinner win the Sinner vs Alcaraz : Round Of 128 match?"
        surface, indoor = TennisModelAgent._resolve_surface(
            competition="ATP Roland Garros",
            title=generic_title,
        )
        assert surface == "clay"
        assert indoor is False

        # Without competition, the title-only path still defaults to hard
        # (generic title has no tournament keywords) — confirming that the
        # competition signal is what unlocks correct classification.
        title_only_surface, _ = TennisModelAgent._resolve_surface(
            competition=None,
            title=generic_title,
        )
        assert title_only_surface == "hard"

    # --- Totality (never raises) ---

    def test_fuzz_never_raises(self):
        """Resolver must be total: 1000 random strings, all return a valid
        (surface, is_indoor) tuple, never raise.
        """
        import random
        import string

        random.seed(42)
        for _ in range(1000):
            length = random.randint(0, 80)
            s = "".join(random.choices(string.printable, k=length))
            surface, indoor = TennisModelAgent._resolve_surface(
                competition=s,
                title=s,
            )
            assert surface in {"hard", "clay", "grass"}
            assert isinstance(indoor, bool)

    def test_case_insensitive(self):
        assert TennisModelAgent._resolve_surface(competition="ATP MUNICH")[0] == "clay"
        assert TennisModelAgent._resolve_surface(competition="atp wimbledon")[0] == "grass"
        assert TennisModelAgent._resolve_surface(competition="Atp Paris Masters") == ("hard", True)


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


class TestSeedPrecomputedSurfaceElo:
    """Seeding from Tennis Abstract's pre-computed Elo leaderboards."""

    RATINGS = {
        "overall": {"Jannik Sinner": 2320.0, "Carlos Alcaraz": 2162.0},
        "hard": {"Jannik Sinner": 2263.0, "Carlos Alcaraz": 2088.0},
        "clay": {"Jannik Sinner": 2216.0, "Carlos Alcaraz": 2102.0},
        "grass": {"Jannik Sinner": 2088.0, "Carlos Alcaraz": 2029.0},
        "indoor": {"Jannik Sinner": 2263.0, "Carlos Alcaraz": 2088.0},
    }

    def test_seeds_all_surfaces_and_resolves_by_surname(self, agent):
        seeded = agent.seed_precomputed_surface_elo(self.RATINGS)
        assert seeded["hard"] == 2
        # Resolver handles full name AND surname-only (live Kalshi style)
        assert agent.get_rating("jannik sinner", "hard") == 2263.0
        assert agent.get_rating("sinner", "clay") == 2216.0
        assert agent.get_rating("alcaraz", "grass") == 2029.0

    def test_stamps_synthetic_counts_to_clear_confidence_gate(self, agent):
        agent.seed_precomputed_surface_elo(self.RATINGS, games_per_player=12)
        # Counts are stamped so predict_pair's surface-confidence branch fires
        assert agent._get_count("jannik sinner", "hard") == 12
        assert agent._get_count("jannik sinner", "clay") == 12
        assert agent._get_overall_count("jannik sinner") == 12
        assert 12 >= MIN_SURFACE_GAMES  # guards the gate assumption

    def test_idempotent_rebuild_no_compounding(self, agent):
        agent.seed_precomputed_surface_elo(self.RATINGS)
        first = agent.all_ratings("hard").copy()
        first_counts = dict(agent._counts("jannik sinner"))
        # Re-seeding must produce identical state, not double the counts/ratings
        agent.seed_precomputed_surface_elo(self.RATINGS)
        assert agent.all_ratings("hard") == first
        assert dict(agent._counts("jannik sinner")) == first_counts

    def test_reset_drops_stale_players_but_keeps_ranking_priors(self, agent):
        # Pre-existing junk rating + a ranking prior
        agent.seed_surface_ratings("hard", {"retired guy": 1700.0})
        agent.seed_rankings({"jannik sinner": 1}, tour="atp")
        agent.seed_precomputed_surface_elo(self.RATINGS)
        # Stale rating cleared by the idempotent reset...
        assert "retired guy" not in agent.all_ratings("hard")
        # ...but ranking priors survive (used as the unrated-player fallback)
        assert agent._state["atp_rankings"]["jannik sinner"] == 1

    def test_indoor_maps_to_hard(self, agent):
        agent.seed_precomputed_surface_elo(self.RATINGS)
        assert agent.get_rating("sinner", "indoor") == agent.get_rating("sinner", "hard")
