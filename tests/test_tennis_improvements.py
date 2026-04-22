"""Tests for tennis model improvements: recency-weighted SPW, surface-specific
serve stats, bo5 amplification, opponent-adjusted form, and fatigue signal.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import AsyncMock

import pytest

from evmax.agents.models.tennis_serve_return_agent import (
    TennisServeReturnAgent,
    _weighted_spw,
    _match_win_prob,
    _HALF_LIFE_DAYS,
    _MIN_WEIGHTED_PTS,
    _MIN_SURFACE_PTS,
)
from evmax.agents.models.tennis_form_agent import (
    TennisFormAgent,
    _opp_quality,
    _FORM_WINDOW,
    _MIN_RECENT_MATCHES,
    _MAX_FATIGUE_PENALTY,
)
from evmax.agents.models.tennis_model_agent import TennisModelAgent
from evmax.models.market import PredictionMarket, MarketType
from evmax.models.odds import SharpOdds, SharpBook


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_market(
    title: str = "Will Sinner win?",
    competition: str = "ATP Miami",
    team_home: str = "sinner",
    team_away: str = "alcaraz",
    sector: str = "tennis",
    event_date: datetime | None = None,
) -> PredictionMarket:
    return PredictionMarket(
        id="test-market",
        title=title,
        sector=sector,
        team_home=team_home,
        team_away=team_away,
        yes_team=team_home,
        yes_price=0.55,
        no_price=0.45,
        volume=1000.0,
        market_type=MarketType.moneyline,
        source="kalshi",
        competition=competition,
        event_date=event_date or datetime(2026, 4, 15, tzinfo=timezone.utc),
    )


def _make_sharp(
    event_id: str = "tennis::2026-04-15::sinner_vs_alcaraz",
    outcome_a: str = "sinner",
    outcome_b: str = "alcaraz",
) -> SharpOdds:
    return SharpOdds(
        event_id=event_id,
        book=SharpBook.pinnacle,
        sector="tennis",
        event_date=datetime(2026, 4, 15, tzinfo=timezone.utc),
        outcome_a_label=outcome_a,
        outcome_b_label=outcome_b,
        outcome_a_decimal=1.83,
        outcome_b_decimal=2.00,
        true_prob_a=0.55,
        true_prob_b=0.45,
        margin=0.03,
    )


# ---------------------------------------------------------------------------
# Recency-weighted SPW tests
# ---------------------------------------------------------------------------

class TestRecencyWeightedSPW:
    def test_recent_matches_weighted_more(self):
        """Recent matches should dominate the weighted SPW."""
        ref = date(2026, 4, 15)
        entries = [
            # Recent: 80% SPW
            {"date": "2026-04-10", "surface": "hard", "won": 60, "svpt": 75},
            {"date": "2026-04-05", "surface": "hard", "won": 55, "svpt": 70},
            # Old: 50% SPW (1 year ago)
            {"date": "2025-04-10", "surface": "hard", "won": 40, "svpt": 80},
            {"date": "2025-04-05", "surface": "hard", "won": 35, "svpt": 70},
        ]
        result = _weighted_spw(entries, None, ref)
        assert result is not None
        spw, _ = result
        # Should be closer to 80% than 50% due to recency
        assert spw > 0.70

    def test_surface_filter(self):
        """Surface-specific SPW should only use matching entries."""
        ref = date(2026, 4, 15)
        entries = [
            {"date": "2026-04-01", "surface": "hard", "won": 55, "svpt": 80},
            {"date": "2026-03-20", "surface": "hard", "won": 50, "svpt": 75},
            {"date": "2026-03-15", "surface": "clay", "won": 35, "svpt": 80},  # clay = worse
            {"date": "2026-03-01", "surface": "hard", "won": 52, "svpt": 78},
        ]

        overall = _weighted_spw(entries, None, ref)
        hard_only = _weighted_spw(entries, "hard", ref)

        assert overall is not None
        assert hard_only is not None
        # Hard-only should be higher (excludes bad clay match)
        assert hard_only[0] > overall[0]

    def test_insufficient_data_returns_none(self):
        """Below min weighted points → None."""
        ref = date(2026, 4, 15)
        entries = [
            {"date": "2026-04-01", "surface": "hard", "won": 5, "svpt": 8},
        ]
        result = _weighted_spw(entries, None, ref)
        assert result is None

    def test_undated_entries_low_weight(self):
        """Entries without dates get low recency weight (treated as ~1 year old)."""
        ref = date(2026, 4, 15)
        entries = [
            {"date": None, "surface": "hard", "won": 50, "svpt": 75},  # undated
            {"date": None, "surface": "hard", "won": 45, "svpt": 70},  # undated
            {"date": "2026-04-10", "surface": "hard", "won": 60, "svpt": 75},  # recent
        ]
        result = _weighted_spw(entries, None, ref)
        assert result is not None
        spw, _ = result
        # Recent 80% match should dominate over undated ~67% matches
        assert spw > 0.72


class TestServeReturnAgent:
    def test_legacy_format_compat(self):
        """Agent should still work with old flat {spw, n} format."""
        agent = TennisServeReturnAgent()
        agent._state = {
            "spw": {
                "sinner j.": {"spw": 0.68, "n": 5000},
                "alcaraz c.": {"spw": 0.67, "n": 4000},
            }
        }
        entries = agent._lookup_entries("sinner")
        assert entries is not None
        assert len(entries) == 1
        assert entries[0]["svpt"] == 5000

    def test_new_format_preferred(self):
        """When both formats exist, match entries are used."""
        agent = TennisServeReturnAgent()
        agent._state = {
            "matches": {
                "sinner j.": [
                    {"date": "2026-04-01", "surface": "hard", "won": 50, "svpt": 75},
                    {"date": "2026-03-15", "surface": "hard", "won": 48, "svpt": 72},
                ],
            },
            "spw": {
                "sinner j.": {"spw": 0.50, "n": 100},  # lower, but should be ignored
            },
        }
        entries = agent._lookup_entries("sinner")
        assert entries is not None
        assert len(entries) == 2
        assert entries[0]["date"] == "2026-04-01"

    @pytest.mark.asyncio
    async def test_predict_uses_surface(self):
        """Prediction should use surface context from market."""
        agent = TennisServeReturnAgent()
        agent._state = {
            "matches": {
                "sinner": [
                    {"date": "2026-04-01", "surface": "hard", "won": 55, "svpt": 80},
                    {"date": "2026-03-15", "surface": "hard", "won": 50, "svpt": 75},
                    {"date": "2026-03-01", "surface": "clay", "won": 35, "svpt": 70},
                ],
                "alcaraz": [
                    {"date": "2026-04-01", "surface": "hard", "won": 50, "svpt": 78},
                    {"date": "2026-03-15", "surface": "hard", "won": 48, "svpt": 75},
                    {"date": "2026-03-01", "surface": "clay", "won": 45, "svpt": 72},
                ],
            }
        }
        market = _make_market(competition="ATP Miami")
        sharp = _make_sharp()
        pred = await agent.predict_pair(market, sharp)
        assert pred is not None
        assert "surface=" in pred.notes


# ---------------------------------------------------------------------------
# Bo5 amplification tests
# ---------------------------------------------------------------------------

class TestBo5Amplification:
    @pytest.mark.asyncio
    async def test_bo5_amplifies_favorite(self):
        """Grand Slam (bo5) should amplify the favorite's probability."""
        agent = TennisModelAgent()
        agent._state = {
            "ratings": {
                "hard": {"sinner": 1850.0, "alcaraz": 1750.0},
                "overall": {"sinner": 1840.0, "alcaraz": 1740.0},
            },
            "game_counts": {
                "sinner": {"hard": 50, "overall": 200},
                "alcaraz": {"hard": 45, "overall": 180},
            },
        }

        # Bo3 match (ATP Miami)
        market_bo3 = _make_market(
            title="Will Sinner win? ATP Miami",
            competition="ATP Miami",
        )
        sharp = _make_sharp()
        pred_bo3 = await agent.predict_pair(market_bo3, sharp)

        # Bo5 match (Australian Open)
        market_bo5 = _make_market(
            title="Will Sinner win? Australian Open",
            competition="ATP Melbourne",
        )
        pred_bo5 = await agent.predict_pair(market_bo5, sharp)

        assert pred_bo3 is not None
        assert pred_bo5 is not None
        # Bo5 should amplify the favorite (Sinner has higher Elo)
        assert pred_bo5.true_prob_a > pred_bo3.true_prob_a

    @pytest.mark.asyncio
    async def test_indoor_bonus(self):
        """Indoor hard court should give a small bonus to the favorite."""
        agent = TennisModelAgent()
        agent._state = {
            "ratings": {
                "hard": {"sinner": 1800.0, "alcaraz": 1750.0},
                "overall": {"sinner": 1790.0, "alcaraz": 1740.0},
            },
            "game_counts": {
                "sinner": {"hard": 50, "overall": 200},
                "alcaraz": {"hard": 45, "overall": 180},
            },
        }

        market_outdoor = _make_market(competition="ATP Miami")
        market_indoor = _make_market(competition="ATP Rotterdam")
        sharp = _make_sharp()

        pred_outdoor = await agent.predict_pair(market_outdoor, sharp)
        pred_indoor = await agent.predict_pair(market_indoor, sharp)

        assert pred_outdoor is not None
        assert pred_indoor is not None
        # Indoor should slightly amplify the favorite
        assert pred_indoor.true_prob_a > pred_outdoor.true_prob_a
        assert "indoor=True" in pred_indoor.notes


# ---------------------------------------------------------------------------
# Form agent tests
# ---------------------------------------------------------------------------

class TestTennisFormAgent:
    def test_opponent_quality_weights(self):
        """Higher-ranked opponents should have higher quality weights."""
        assert _opp_quality(1) > _opp_quality(50)
        assert _opp_quality(50) > _opp_quality(200)
        assert _opp_quality(None) == 0.5

    def test_form_computation(self):
        """Form should reflect recent quality-weighted win rate."""
        agent = TennisFormAgent()
        ref = date(2026, 4, 15)

        # Player with strong recent form (beating top players)
        strong_matches = [
            {"date": "2026-04-10", "won": True, "opp_rank": 5, "surface": "hard"},
            {"date": "2026-04-08", "won": True, "opp_rank": 10, "surface": "hard"},
            {"date": "2026-04-01", "won": True, "opp_rank": 20, "surface": "hard"},
            {"date": "2026-03-25", "won": False, "opp_rank": 2, "surface": "hard"},
            {"date": "2026-03-20", "won": True, "opp_rank": 30, "surface": "hard"},
        ]

        # Player with weak recent form
        weak_matches = [
            {"date": "2026-04-10", "won": False, "opp_rank": 50, "surface": "hard"},
            {"date": "2026-04-08", "won": False, "opp_rank": 80, "surface": "hard"},
            {"date": "2026-04-01", "won": True, "opp_rank": 150, "surface": "hard"},
            {"date": "2026-03-25", "won": False, "opp_rank": 40, "surface": "hard"},
            {"date": "2026-03-20", "won": False, "opp_rank": 60, "surface": "hard"},
        ]

        strong_form = agent._compute_form(strong_matches, ref)
        weak_form = agent._compute_form(weak_matches, ref)
        assert strong_form is not None
        assert weak_form is not None
        assert strong_form > weak_form
        assert strong_form > 0.5  # mostly winning
        assert weak_form < 0.5   # mostly losing

    def test_form_insufficient_data(self):
        """Fewer than MIN_RECENT_MATCHES → None."""
        agent = TennisFormAgent()
        ref = date(2026, 4, 15)
        matches = [
            {"date": "2026-04-10", "won": True, "opp_rank": 5, "surface": "hard"},
            {"date": "2026-04-08", "won": True, "opp_rank": 10, "surface": "hard"},
        ]
        result = agent._compute_form(matches, ref)
        assert result is None

    def test_fatigue_penalty(self):
        """Recent matches within 7 days should produce fatigue."""
        agent = TennisFormAgent()
        ref = date(2026, 4, 15)

        # Two matches in last week
        matches = [
            {"date": "2026-04-13", "won": True, "opp_rank": 20, "surface": "hard", "minutes": 90},
            {"date": "2026-04-11", "won": True, "opp_rank": 30, "surface": "hard", "minutes": 150},
        ]
        fatigue = agent._compute_fatigue(matches, ref)
        assert fatigue > 0
        # Two matches + one long match (150 min > 120 threshold)
        assert fatigue > 2 * 0.008  # base + long match bonus

    def test_no_fatigue_when_rested(self):
        """No recent matches → zero fatigue."""
        agent = TennisFormAgent()
        ref = date(2026, 4, 15)
        matches = [
            {"date": "2026-03-01", "won": True, "opp_rank": 20, "surface": "hard", "minutes": 90},
        ]
        fatigue = agent._compute_fatigue(matches, ref)
        assert fatigue == 0.0

    def test_fatigue_capped(self):
        """Fatigue should never exceed MAX_FATIGUE_PENALTY."""
        agent = TennisFormAgent()
        ref = date(2026, 4, 15)
        # 7 matches in 7 days, all long
        matches = [
            {"date": f"2026-04-{8+i:02d}", "won": True, "opp_rank": 20, "surface": "hard", "minutes": 200}
            for i in range(7)
        ]
        fatigue = agent._compute_fatigue(matches, ref)
        assert fatigue == _MAX_FATIGUE_PENALTY

    def test_surface_preference_in_form(self):
        """When enough surface-specific matches exist, form should prefer them."""
        agent = TennisFormAgent()
        ref = date(2026, 4, 15)

        matches = [
            # Hard court: all wins
            {"date": "2026-04-10", "won": True, "opp_rank": 20, "surface": "hard"},
            {"date": "2026-04-08", "won": True, "opp_rank": 30, "surface": "hard"},
            {"date": "2026-04-05", "won": True, "opp_rank": 25, "surface": "hard"},
            {"date": "2026-04-01", "won": True, "opp_rank": 40, "surface": "hard"},
            {"date": "2026-03-28", "won": True, "opp_rank": 15, "surface": "hard"},
            # Clay: all losses
            {"date": "2026-03-25", "won": False, "opp_rank": 50, "surface": "clay"},
            {"date": "2026-03-20", "won": False, "opp_rank": 60, "surface": "clay"},
            {"date": "2026-03-15", "won": False, "opp_rank": 70, "surface": "clay"},
            {"date": "2026-03-10", "won": False, "opp_rank": 80, "surface": "clay"},
            {"date": "2026-03-05", "won": False, "opp_rank": 90, "surface": "clay"},
        ]

        form_hard = agent._compute_form(matches, ref, "hard")
        form_all = agent._compute_form(matches, ref)
        assert form_hard is not None
        assert form_all is not None
        # Hard-only form should be much higher (all wins)
        assert form_hard > form_all


# ---------------------------------------------------------------------------
# Match-win probability tests
# ---------------------------------------------------------------------------

class TestMatchWinProb:
    def test_equal_spw_gives_50_50(self):
        """Equal SPW → exactly 50% for both bo3 and bo5."""
        assert _match_win_prob(0.65, 0.65, best_of=3) == 0.5
        assert _match_win_prob(0.65, 0.65, best_of=5) == 0.5

    def test_bo5_amplifies(self):
        """Bo5 should give a higher probability to the favorite."""
        spw_a, spw_b = 0.68, 0.64
        prob_bo3 = _match_win_prob(spw_a, spw_b, best_of=3)
        prob_bo5 = _match_win_prob(spw_a, spw_b, best_of=5)
        assert prob_bo5 > prob_bo3
        assert prob_bo3 > 0.5

    def test_symmetry(self):
        """Swapping players should give complementary probability."""
        prob = _match_win_prob(0.68, 0.62)
        prob_swap = _match_win_prob(0.62, 0.68)
        assert abs(prob + prob_swap - 1.0) < 1e-10
