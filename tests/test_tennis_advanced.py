"""Tests for TennisAdvancedStatsAgent — logistic model on BP, RPW, UE, W/UE.

Covers:
  - Sector gating (non-tennis returns None)
  - Low sample size exclusion (< MIN_MATCHES, < MIN_BP_FACED, < MIN_RETURN_PTS)
  - Full model (4-feature) vs reduced model (RPW-only) selection
  - Probability direction: higher RPW/BP conv player favored
  - Probability bounds (0.02–0.98)
  - Confidence scaling with match count
  - State seeding round-trip
  - Surname fallback lookup
"""

from __future__ import annotations

import pytest

from evmax.agents.models.tennis_advanced_stats_agent import TennisAdvancedStatsAgent
from evmax.models.market import PredictionMarket, MarketSource
from evmax.models.odds import SharpOdds, SharpBook


def _make_market(
    team_home: str = "sinner",
    team_away: str = "alcaraz",
    sector: str = "tennis",
    title: str = "ATP Test Match",
) -> PredictionMarket:
    return PredictionMarket(
        id="kalshi:TEST",
        source=MarketSource.kalshi,
        title=title,
        team_home=team_home,
        team_away=team_away,
        sector=sector,
        yes_price=0.55,
        no_price=0.45,
    )


def _make_sharp(
    outcome_a: str = "sinner",
    outcome_b: str = "alcaraz",
    prob_a: float = 0.55,
) -> SharpOdds:
    return SharpOdds(
        event_id="tennis::test::sinner_vs_alcaraz",
        book=SharpBook.pinnacle,
        sector="tennis",
        true_prob_a=prob_a,
        true_prob_b=1.0 - prob_a,
        outcome_a_label=outcome_a,
        outcome_b_label=outcome_b,
        outcome_a_decimal=1.0 / prob_a if prob_a > 0 else 2.0,
        outcome_b_decimal=1.0 / (1.0 - prob_a) if prob_a < 1 else 2.0,
    )


def _make_player_stats(
    bp_won: int = 150,
    bp_faced: int = 400,
    return_pts: int = 3000,
    return_pts_won: int = 1200,
    total_pts: int = 7000,
    matches: int = 50,
    unforced: int = 0,
    winners: int = 0,
) -> dict:
    return {
        "bp_won": bp_won,
        "bp_faced": bp_faced,
        "return_pts": return_pts,
        "return_pts_won": return_pts_won,
        "total_pts": total_pts,
        "matches": matches,
        "unforced": unforced,
        "winners": winners,
    }


@pytest.fixture
def agent(tmp_path):
    a = TennisAdvancedStatsAgent()
    a._state_path = tmp_path / "tennis_advanced_state.json"
    a._state = {"stats": {
        "sinner": _make_player_stats(
            bp_won=180, bp_faced=400, return_pts=3500,
            return_pts_won=1470, total_pts=8000, matches=60,
            unforced=320, winners=480,
        ),
        "alcaraz": _make_player_stats(
            bp_won=160, bp_faced=400, return_pts=3200,
            return_pts_won=1280, total_pts=7500, matches=55,
            unforced=350, winners=420,
        ),
        "djokovic": _make_player_stats(
            bp_won=200, bp_faced=500, return_pts=4000,
            return_pts_won=1680, total_pts=9000, matches=70,
        ),
        "unknown player": _make_player_stats(matches=5),
        "low bp": _make_player_stats(bp_faced=10),
        "low rpts": _make_player_stats(return_pts=50),
    }}
    return a


class TestSectorGating:
    @pytest.mark.asyncio
    async def test_non_tennis_returns_none(self, agent):
        market = _make_market(sector="nba")
        result = await agent.predict_pair(market, _make_sharp())
        assert result is None


class TestSampleSizeGating:
    @pytest.mark.asyncio
    async def test_below_min_matches(self, agent):
        market = _make_market(team_home="unknown player", team_away="sinner")
        result = await agent.predict_pair(market, _make_sharp("unknown player", "sinner"))
        assert result is None

    @pytest.mark.asyncio
    async def test_below_min_bp_faced(self, agent):
        market = _make_market(team_home="low bp", team_away="sinner")
        result = await agent.predict_pair(market, _make_sharp("low bp", "sinner"))
        assert result is None

    @pytest.mark.asyncio
    async def test_below_min_return_pts(self, agent):
        market = _make_market(team_home="low rpts", team_away="sinner")
        result = await agent.predict_pair(market, _make_sharp("low rpts", "sinner"))
        assert result is None


class TestModelSelection:
    @pytest.mark.asyncio
    async def test_full_model_when_both_have_ue_data(self, agent):
        """Sinner and Alcaraz both have UE/winners → use full 4-feature model."""
        market = _make_market()
        result = await agent.predict_pair(market, _make_sharp())
        assert result is not None
        assert "variant=full" in result.notes

    @pytest.mark.asyncio
    async def test_reduced_model_when_missing_ue_data(self, agent):
        """Djokovic has no UE/winners (zeros) → fall back to RPW-only."""
        market = _make_market(team_home="djokovic", team_away="sinner")
        result = await agent.predict_pair(market, _make_sharp("djokovic", "sinner"))
        assert result is not None
        assert "variant=reduced" in result.notes


class TestProbabilityDirection:
    @pytest.mark.asyncio
    async def test_better_returner_is_favored(self, agent):
        """Sinner has higher RPW (1470/3500=0.420) than Alcaraz (1280/3200=0.400).
        Model should favor Sinner."""
        market = _make_market()
        result = await agent.predict_pair(market, _make_sharp())
        assert result is not None
        assert result.true_prob_a > 0.50

    @pytest.mark.asyncio
    async def test_much_better_returner_strong_favorite(self, agent):
        """Djokovic RPW=0.420 vs Alcaraz RPW=0.400 → Djokovic should be favored in reduced model."""
        market = _make_market(team_home="djokovic", team_away="alcaraz")
        result = await agent.predict_pair(market, _make_sharp("djokovic", "alcaraz"))
        assert result is not None
        assert result.true_prob_a > 0.50


class TestProbabilityBounds:
    @pytest.mark.asyncio
    async def test_prob_bounded(self, agent):
        result = await agent.predict_pair(_make_market(), _make_sharp())
        assert result is not None
        assert 0.02 <= result.true_prob_a <= 0.98
        assert 0.02 <= result.true_prob_b <= 0.98
        assert abs(result.true_prob_a + result.true_prob_b - 1.0) < 1e-9


class TestConfidence:
    @pytest.mark.asyncio
    async def test_confidence_scales_with_matches(self, agent):
        result = await agent.predict_pair(_make_market(), _make_sharp())
        assert result is not None
        min_matches = min(60, 55)  # sinner=60, alcaraz=55
        expected = min(0.80, 0.50 + min_matches / 500.0)
        assert result.confidence == pytest.approx(expected)


class TestSeedRoundTrip:
    def test_seed_and_lookup(self, tmp_path):
        state_path = tmp_path / "tennis_advanced_state.json"
        agent = TennisAdvancedStatsAgent()
        agent._state_path = state_path
        stats = {
            "nadal": _make_player_stats(
                bp_won=100, bp_faced=300,
                return_pts=2500, return_pts_won=1050,
                total_pts=6000, matches=40,
            ),
        }
        agent.seed_stats(stats)

        # Re-load from disk
        agent2 = TennisAdvancedStatsAgent()
        agent2._state_path = state_path
        agent2.load_state()
        rec = agent2._lookup("nadal")
        assert rec is not None
        assert rec["bp_won"] == 100
        assert rec["matches"] == 40


class TestSurnameFallback:
    @pytest.mark.asyncio
    async def test_surname_matches(self, agent):
        """'jannik sinner' should find 'sinner' via surname fallback."""
        market = _make_market(team_home="jannik sinner", team_away="carlos alcaraz")
        sharp = _make_sharp("jannik sinner", "carlos alcaraz")
        result = await agent.predict_pair(market, sharp)
        assert result is not None


class TestSharedResolver:
    """_lookup swapped to tennis_common.resolve_player (2026-07-19).

    The old endswith-surname scan had no dehyphenation and silently returned
    the FIRST store key on same-surname collisions — a wrong player's stats.
    """

    def test_hyphenated_query_resolves_space_form_key(self, tmp_path):
        a = TennisAdvancedStatsAgent()
        a._state_path = tmp_path / "s.json"
        a._state = {"stats": {"felix auger aliassime": _make_player_stats(matches=40)}}
        rec = a._lookup("felix auger-aliassime")
        assert rec is not None
        assert rec["matches"] == 40

    def test_same_surname_ambiguity_returns_none(self, tmp_path):
        # Genuinely different players sharing a surname, queried by bare
        # surname: wrong-player data is worse than no data.
        a = TennisAdvancedStatsAgent()
        a._state_path = tmp_path / "s.json"
        a._state = {"stats": {
            "francisco cerundolo": _make_player_stats(matches=50),
            "juan manuel cerundolo": _make_player_stats(matches=30),
        }}
        assert a._lookup("cerundolo") is None

    def test_given_name_narrows_same_surname(self, tmp_path):
        a = TennisAdvancedStatsAgent()
        a._state_path = tmp_path / "s.json"
        a._state = {"stats": {
            "xin yu wang": _make_player_stats(matches=44),
            "xiyu wang": _make_player_stats(matches=33),
        }}
        rec = a._lookup("xinyu wang")
        assert rec is not None
        assert rec["matches"] == 44

    def test_duplicate_same_person_picks_higher_match_count(self, tmp_path):
        a = TennisAdvancedStatsAgent()
        a._state_path = tmp_path / "s.json"
        a._state = {"stats": {
            "adrian mannarino": _make_player_stats(matches=60),
            "mannarino a.": _make_player_stats(matches=4),
        }}
        rec = a._lookup("mannarino")
        assert rec is not None
        assert rec["matches"] == 60


class TestComputeFeatures:
    def test_full_features_with_ue(self):
        rec = _make_player_stats(
            bp_won=150, bp_faced=400,
            return_pts=3000, return_pts_won=1200,
            unforced=200, winners=300,
            total_pts=7000, matches=50,
        )
        feats, extended = TennisAdvancedStatsAgent._compute_features(rec)
        assert extended is True
        assert len(feats) == 4
        assert feats[0] == pytest.approx(150 / 400)  # bp_conv
        assert feats[1] == pytest.approx(1200 / 3000)  # rpw
        assert feats[2] == pytest.approx(200 / 7000)  # ue_rate
        assert feats[3] == pytest.approx(300 / 200)  # w_ue

    def test_reduced_features_without_ue(self):
        rec = _make_player_stats(
            bp_won=150, bp_faced=400,
            return_pts=3000, return_pts_won=1200,
            total_pts=7000, matches=50,
        )
        feats, extended = TennisAdvancedStatsAgent._compute_features(rec)
        assert extended is False
        assert len(feats) == 2  # bp_conv + rpw

    def test_empty_features_on_low_bp(self):
        rec = _make_player_stats(bp_faced=5)
        feats, _ = TennisAdvancedStatsAgent._compute_features(rec)
        assert feats == []
