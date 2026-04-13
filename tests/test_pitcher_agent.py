"""Tests for PitcherModelAgent (MLB starter-based win probability)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from evmax.agents.models import pitcher_agent as pitcher_mod
from evmax.agents.models.pitcher_agent import (
    HOME_BONUS,
    PYTHAG_EXP,
    PitcherModelAgent,
)
from evmax.models.market import MarketSource, MarketType, PredictionMarket
from evmax.models.odds import SharpBook, SharpOdds


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def agent(tmp_path, monkeypatch):
    """Fresh PitcherModelAgent pointing at a temp state dir."""
    monkeypatch.setattr("evmax.agents.models.base.STATE_DIR", tmp_path)
    a = PitcherModelAgent()
    a._state_path = tmp_path / "pitcher_state.json"
    # Disable the ESPN probables cache so tests don't hit the network.
    monkeypatch.setattr(pitcher_mod, "_probable_cache", {})
    monkeypatch.setattr(pitcher_mod, "_probable_cache_ts", 0.0)

    async def _no_live():
        return {}

    monkeypatch.setattr(pitcher_mod, "_fetch_probable_starters", _no_live)
    return a


def _market(team_home="yankees", team_away="red sox", sector="baseball"):
    return PredictionMarket(
        id=f"kalshi:{team_home}_vs_{team_away}",
        source=MarketSource.kalshi,
        sector=sector,
        market_type=MarketType.moneyline,
        title=f"{team_home} vs {team_away}",
        ticker="KXMLBGAME-TEST",
        yes_price=0.50,
        no_price=0.50,
        volume_usd=10_000.0,
        open_interest_usd=5_000.0,
        team_home=team_home,
        team_away=team_away,
        event_date=datetime(2026, 5, 10, 23, 0, tzinfo=timezone.utc),
    )


def _sharp(team_a="yankees", team_b="red sox"):
    return SharpOdds(
        event_id=f"baseball::2026-05-10::{team_a}_vs_{team_b}",
        book=SharpBook.pinnacle,
        sector="baseball",
        outcome_a_label=team_a,
        outcome_b_label=team_b,
        outcome_a_decimal=2.0,
        outcome_b_decimal=2.0,
        true_prob_a=0.50,
        true_prob_b=0.50,
        event_date=datetime(2026, 5, 10, 23, 0, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# Sector gating
# ---------------------------------------------------------------------------


class TestSectorGating:
    @pytest.mark.asyncio
    async def test_non_baseball_sector_returns_none(self, agent):
        m = _market(sector="nba")
        s = _sharp()
        pred = await agent.predict_pair(m, s)
        assert pred is None

    @pytest.mark.asyncio
    async def test_missing_pitcher_data_returns_none(self, agent):
        # No pitchers seeded → can't find starters → None
        m = _market()
        s = _sharp()
        pred = await agent.predict_pair(m, s)
        assert pred is None


# ---------------------------------------------------------------------------
# Seeding + state
# ---------------------------------------------------------------------------


class TestSeedPitchers:
    def test_seed_populates_state(self, agent):
        agent.seed_pitchers(
            {
                "Gerrit Cole": {"era": 3.10, "ip": 180, "team": "yankees"},
                "Chris Sale": {"era": 2.95, "ip": 170, "team": "red sox"},
            },
            league_avg_era=4.20,
        )
        assert agent._state["league_avg_era"] == 4.20
        assert "gerrit cole" in agent._state["pitchers"]
        assert agent._state["pitchers"]["gerrit cole"]["era"] == 3.10
        # First pitcher per team becomes the team_starters entry
        assert agent._state["team_starters"]["yankees"] == "gerrit cole"
        assert agent._state["team_starters"]["red sox"] == "chris sale"

    def test_seed_preserves_first_starter_per_team(self, agent):
        """Second pitcher for the same team should not overwrite the starter."""
        agent.seed_pitchers(
            {
                "Gerrit Cole": {"era": 3.10, "ip": 180, "team": "yankees"},
                "Carlos Rodon": {"era": 4.00, "ip": 150, "team": "yankees"},
            }
        )
        assert agent._state["team_starters"]["yankees"] == "gerrit cole"


# ---------------------------------------------------------------------------
# Pythagorean prediction
# ---------------------------------------------------------------------------


class TestPythagPrediction:
    @pytest.mark.asyncio
    async def test_known_bug_better_home_pitcher_is_less_favored(self, agent):
        """**Known bug — pins current (inverted) behavior.**

        The agent assigns ``home_ra = away_era`` (and vice versa), treating
        "runs the home team allows" as a function of the *opposing* starter.
        That's backwards: a team's runs allowed is driven by its own pitcher.
        The consequence is that the side with the better starter ends up
        *less* favored, which is the opposite of reality.

        This test locks the current output so a fix will visibly break it.
        """
        agent.seed_pitchers(
            {
                "Ace Pitcher": {"era": 2.50, "ip": 180, "team": "yankees"},
                "Weak Starter": {"era": 5.50, "ip": 160, "team": "red sox"},
            },
            league_avg_era=4.20,
        )
        m = _market(team_home="yankees", team_away="red sox")
        s = _sharp(team_a="yankees", team_b="red sox")
        pred = await agent.predict_pair(m, s)

        assert pred is not None
        # Home has the ace (2.50 ERA) vs away's 5.50 ERA — yet under current
        # (buggy) semantics the home team is *under* 50/50 even after the
        # home bonus. A proper fix should flip this to >0.80.
        assert pred.true_prob_a < 0.50
        assert pred.true_prob_a == pytest.approx(1.0 - pred.true_prob_b, abs=1e-6)

    @pytest.mark.asyncio
    async def test_equal_era_home_bonus_applied(self, agent):
        """With identical ERAs, Pythag gives 50/50; HOME_BONUS should tilt
        the home side to ~54%."""
        agent.seed_pitchers(
            {
                "Pitcher A": {"era": 4.00, "ip": 160, "team": "yankees"},
                "Pitcher B": {"era": 4.00, "ip": 160, "team": "red sox"},
            }
        )
        m = _market(team_home="yankees", team_away="red sox")
        s = _sharp(team_a="yankees", team_b="red sox")
        pred = await agent.predict_pair(m, s)

        assert pred is not None
        assert pred.true_prob_a == pytest.approx(0.5 + HOME_BONUS, abs=1e-3)

    @pytest.mark.asyncio
    async def test_probabilities_sum_to_one(self, agent):
        agent.seed_pitchers(
            {
                "A": {"era": 3.20, "ip": 170, "team": "yankees"},
                "B": {"era": 4.50, "ip": 150, "team": "red sox"},
            }
        )
        m = _market()
        s = _sharp()
        pred = await agent.predict_pair(m, s)
        assert pred is not None
        assert pred.true_prob_a + pred.true_prob_b == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.asyncio
    async def test_probabilities_clamped_to_bounds(self, agent):
        """Extreme ERA gap should still be clamped inside [0.10, 0.90]
        (after home bonus). Pythag 1.00 vs 10.00 would otherwise exceed 0.90."""
        agent.seed_pitchers(
            {
                "Ace": {"era": 1.00, "ip": 200, "team": "yankees"},
                "BP": {"era": 10.00, "ip": 150, "team": "red sox"},
            }
        )
        m = _market()
        s = _sharp()
        pred = await agent.predict_pair(m, s)
        assert pred is not None
        assert 0.10 <= pred.true_prob_a <= 0.90
        assert 0.10 <= pred.true_prob_b <= 0.90


# ---------------------------------------------------------------------------
# Confidence tiers
# ---------------------------------------------------------------------------


class TestConfidence:
    @pytest.mark.asyncio
    async def test_deep_ip_history_high_confidence(self, agent):
        agent.seed_pitchers(
            {
                "A": {"era": 3.10, "ip": 180, "team": "yankees"},
                "B": {"era": 3.80, "ip": 170, "team": "red sox"},
            }
        )
        pred = await agent.predict_pair(_market(), _sharp())
        assert pred is not None
        # Static starters + min_ip >= 150 → confidence = 0.65
        assert pred.confidence == pytest.approx(0.65, abs=1e-6)

    @pytest.mark.asyncio
    async def test_thin_data_below_gate(self, agent):
        """Very low IP on one side → confidence below the 0.45 blend gate."""
        agent.seed_pitchers(
            {
                "A": {"era": 3.10, "ip": 20, "team": "yankees"},
                "B": {"era": 3.80, "ip": 15, "team": "red sox"},
            }
        )
        pred = await agent.predict_pair(_market(), _sharp())
        assert pred is not None
        # Static + min_ip < 100 → 0.35 (below the 0.45 gate)
        assert pred.confidence < 0.45

    @pytest.mark.asyncio
    async def test_live_starter_mid_confidence(self, agent, monkeypatch):
        """When both starters come from ESPN probables (no stored IP) we
        fall into the 'live but no IP context' bucket."""
        agent.seed_pitchers(
            {
                "Static A": {"era": 3.10, "ip": 180, "team": "yankees"},
                "Static B": {"era": 3.80, "ip": 170, "team": "red sox"},
            }
        )

        async def _live():
            return {
                "yankees": {"name": "live ace", "era": 2.80, "ip": 0},
                "red sox": {"name": "live ace 2", "era": 3.40, "ip": 0},
            }

        monkeypatch.setattr(pitcher_mod, "_fetch_probable_starters", _live)
        pred = await agent.predict_pair(_market(), _sharp())
        assert pred is not None
        assert pred.confidence == pytest.approx(0.55, abs=1e-6)
        # Notes should mention live.
        assert "live" in pred.notes


# ---------------------------------------------------------------------------
# Team name fallbacks
# ---------------------------------------------------------------------------


class TestTeamNameFallback:
    @pytest.mark.asyncio
    async def test_full_city_name_resolves_via_last_word(self, agent):
        agent.seed_pitchers(
            {
                "A": {"era": 3.10, "ip": 180, "team": "yankees"},
                "B": {"era": 3.80, "ip": 170, "team": "red sox"},
            }
        )
        # Market uses the long form "new york yankees" — must still resolve.
        m = _market(team_home="new york yankees", team_away="boston red sox")
        s = SharpOdds(
            event_id="baseball::2026-05-10::nyy_bos",
            book=SharpBook.pinnacle,
            sector="baseball",
            outcome_a_label="new york yankees",
            outcome_b_label="boston red sox",
            outcome_a_decimal=2.0,
            outcome_b_decimal=2.0,
            true_prob_a=0.50,
            true_prob_b=0.50,
        )
        pred = await agent.predict_pair(m, s)
        assert pred is not None


# ---------------------------------------------------------------------------
# Sanity: Pythag formula alignment
# ---------------------------------------------------------------------------


class TestPythagMath:
    def test_pythag_exponent_constant(self):
        assert PYTHAG_EXP == 1.83

    @pytest.mark.asyncio
    async def test_matches_manual_pythag_calculation(self, agent):
        """Derive the expected probability by hand and compare."""
        league = 4.20
        home_era = 3.00
        away_era = 4.50
        agent.seed_pitchers(
            {
                "H": {"era": home_era, "ip": 180, "team": "yankees"},
                "A": {"era": away_era, "ip": 170, "team": "red sox"},
            },
            league_avg_era=league,
        )

        e = PYTHAG_EXP
        home_wp = league ** e / (league ** e + away_era ** e)
        away_wp = league ** e / (league ** e + home_era ** e)
        expected = home_wp / (home_wp + away_wp) + HOME_BONUS

        pred = await agent.predict_pair(_market(), _sharp())
        assert pred is not None
        assert pred.true_prob_a == pytest.approx(round(expected, 4), abs=1e-3)
