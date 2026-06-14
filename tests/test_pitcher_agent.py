"""Tests for PitcherModelAgent (MLB starter-based win probability)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from evmax.agents.models import pitcher_agent as pitcher_mod
from evmax.agents.models.pitcher_agent import (
    ERA_BLEND_WEIGHT,
    FIP_BLEND_WEIGHT,
    HOME_BONUS,
    HOME_BONUS_MAX,
    HOME_BONUS_MIN,
    PYTHAG_EXP,
    PitcherModelAgent,
    _effective_era,
    _has_real_rate,
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
# Unseeded-starter handling (regression: ZeroDivisionError wiped the batch)
# ---------------------------------------------------------------------------


def _live_starters(monkeypatch, mapping):
    """Patch the agent's probable-starter fetch to return `mapping`."""
    async def _fetch():
        return mapping
    monkeypatch.setattr(pitcher_mod, "_fetch_probable_starters", _fetch)


class TestEffectiveEraNonPositive:
    def test_zero_era_falls_back_to_league_avg(self):
        # 0.0 is the MLB-API placeholder for an unseeded starter — not a rate.
        assert _effective_era({"era": 0.0}, league_avg=4.08) == 4.08

    def test_negative_rate_falls_back_to_league_avg(self):
        assert _effective_era({"era": -1.0, "fip": -2.0}, league_avg=4.08) == 4.08

    def test_zero_fip_with_real_era_uses_era(self):
        # Bad fip is ignored; the real era still drives the rate.
        assert _effective_era({"fip": 0.0, "era": 3.50}, league_avg=4.08) == 3.50


class TestHasRealRate:
    def test_true_when_positive_era(self):
        assert _has_real_rate({"era": 3.5}) is True

    def test_true_when_positive_fip_only(self):
        assert _has_real_rate({"fip": 3.9}) is True

    def test_false_when_placeholder_zero(self):
        assert _has_real_rate({"era": 0.0, "ip": 0}) is False

    def test_false_when_empty(self):
        assert _has_real_rate({}) is False


class TestUnseededStarterAbstains:
    @pytest.mark.asyncio
    async def test_both_unseeded_starters_do_not_crash(self, agent, monkeypatch):
        """Regression: two live starters absent from the seeded DB (era=0.0)
        made the Pythag denominator 0**e + 0**e == 0 → ZeroDivisionError, which
        killed the whole pitcher batch and dropped every baseball moneyline.
        The agent must now abstain (return None), not raise."""
        _live_starters(monkeypatch, {
            "yankees": {"name": "rookie a", "era": 0.0, "ip": 0},
            "red sox": {"name": "rookie b", "era": 0.0, "ip": 0},
        })
        pred = await agent.predict_pair(_market(), _sharp())
        assert pred is None  # abstained, no exception

    @pytest.mark.asyncio
    async def test_one_unseeded_starter_abstains(self, agent, monkeypatch):
        # Yankees ace is seeded; Red Sox starter is an unseeded live call-up.
        agent.seed_pitchers(
            {"Ace": {"era": 2.80, "ip": 180, "team": "yankees"}},
            league_avg_era=4.20,
        )
        _live_starters(monkeypatch, {
            "yankees": {"name": "ace", "era": 2.80, "ip": 180},
            "red sox": {"name": "rookie b", "era": 0.0, "ip": 0},
        })
        pred = await agent.predict_pair(_market(), _sharp())
        assert pred is None


# ---------------------------------------------------------------------------
# Pythagorean prediction
# ---------------------------------------------------------------------------


class TestPythagPrediction:
    @pytest.mark.asyncio
    async def test_lower_era_team_is_favored(self, agent):
        """Regression for BUG-9: the side with the better starter must be
        favored. Prior to the fix the Pythag assigned ``home_ra = away_era``
        (and vice versa), producing the inverted result that a 2.50 ERA ace
        opposing a 5.50 ERA starter won only ~38% of the time."""
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
        # With the fix, 2.50 vs 5.50 ERA produces home_wp ≈ 0.80, and after
        # the +0.04 home bonus the clamp in [0.10, 0.90] still leaves this
        # comfortably above 0.80.
        assert pred.true_prob_a > 0.80
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

    @pytest.mark.asyncio
    async def test_multiword_nickname_resolves_live_not_static(self, agent, monkeypatch):
        """Regression: ESPN keys probable starters by nickname ("red sox",
        "blue jays") while Pinnacle passes full names ("boston red sox").

        The old last-word fallback ("boston red sox" -> "sox") missed the
        multi-word nicknames, so Red Sox / White Sox / Blue Jays games silently
        lost the pitcher model (predict_pair returns None if either starter is
        unresolved) and fell to a generic blend that went -23% ROI live. The
        suffix match must resolve them from the LIVE feed (confidence 0.55 +
        "live" note), not the stale static fallback.
        """
        # Static fallback intentionally has DIFFERENT (stale) pitchers so a
        # live match is observably distinct from the fallback.
        agent.seed_pitchers(
            {
                "Stale Sox": {"era": 9.99, "ip": 200, "team": "red sox"},
                "Stale Jays": {"era": 9.99, "ip": 200, "team": "blue jays"},
            }
        )

        async def _live():
            return {
                "red sox": {"name": "live sox arm", "era": 2.90, "ip": 0},
                "blue jays": {"name": "live jays arm", "era": 3.10, "ip": 0},
            }

        monkeypatch.setattr(pitcher_mod, "_fetch_probable_starters", _live)
        m = _market(team_home="boston red sox", team_away="toronto blue jays")
        s = SharpOdds(
            event_id="baseball::2026-05-16::bos_tor",
            book=SharpBook.pinnacle,
            sector="baseball",
            outcome_a_label="boston red sox",
            outcome_b_label="toronto blue jays",
            outcome_a_decimal=2.0,
            outcome_b_decimal=2.0,
            true_prob_a=0.50,
            true_prob_b=0.50,
        )
        pred = await agent.predict_pair(m, s)
        assert pred is not None
        # Live path → 0.55 confidence tier and a "live" note (the stale static
        # entries carry ip=200 which would land a different tier).
        assert pred.confidence == pytest.approx(0.55, abs=1e-6)
        assert "live" in pred.notes

    def test_match_live_starter_disambiguates_red_vs_white_sox(self, agent):
        """Suffix match must not confuse the two Sox: "chicago white sox" has
        to resolve "white sox", never "red sox". The last-word ("sox") is in
        neither key, so the full-nickname suffix is what disambiguates."""
        live = {
            "red sox": {"name": "boston arm", "era": 3.0, "ip": 0},
            "white sox": {"name": "chicago arm", "era": 4.0, "ip": 0},
        }
        assert agent._match_live_starter("chicago white sox", "sox", live)["name"] == "chicago arm"
        assert agent._match_live_starter("boston red sox", "sox", live)["name"] == "boston arm"


# ---------------------------------------------------------------------------
# FIP support
# ---------------------------------------------------------------------------


class TestEffectiveEra:
    def test_fip_only_returns_fip(self):
        assert _effective_era({"fip": 3.20}, league_avg=4.20) == 3.20

    def test_era_only_returns_era(self):
        assert _effective_era({"era": 3.80}, league_avg=4.20) == 3.80

    def test_neither_returns_league_avg(self):
        assert _effective_era({}, league_avg=4.20) == 4.20

    def test_both_blends_60_40(self):
        # 0.60 * 3.00 + 0.40 * 4.00 = 1.80 + 1.60 = 3.40
        assert _effective_era({"fip": 3.00, "era": 4.00}, league_avg=4.20) == pytest.approx(3.40)

    def test_blend_weights_sum_to_one(self):
        assert FIP_BLEND_WEIGHT + ERA_BLEND_WEIGHT == pytest.approx(1.0)

    def test_fip_weight_dominates(self):
        """FIP should be weighted more than ERA in the blend (it's more predictive)."""
        assert FIP_BLEND_WEIGHT > ERA_BLEND_WEIGHT


class TestFipPrediction:
    @pytest.mark.asyncio
    async def test_fip_changes_predicted_probability(self, agent):
        """A pitcher with bad ERA but good FIP should be projected stronger
        than the ERA-only model would suggest. Compare two seedings: same
        ERA pair, but only one carries FIP data."""
        # Baseline: both have only ERA
        agent.seed_pitchers(
            {
                "Lucky Home": {"era": 3.00, "ip": 180, "team": "yankees"},
                "Unlucky Away": {"era": 5.00, "ip": 170, "team": "red sox"},
            }
        )
        m = _market(team_home="yankees", team_away="red sox")
        s = _sharp(team_a="yankees", team_b="red sox")
        pred_era_only = await agent.predict_pair(m, s)
        assert pred_era_only is not None

        # Now flip the FIP picture: home pitcher's FIP says he's actually
        # mediocre (4.20), away pitcher's FIP says he's actually decent (3.50).
        # The blended rate brings them much closer, so home favoritism should drop.
        agent.seed_pitchers(
            {
                "Lucky Home": {"era": 3.00, "fip": 4.20, "ip": 180, "team": "yankees"},
                "Unlucky Away": {"era": 5.00, "fip": 3.50, "ip": 170, "team": "red sox"},
            }
        )
        pred_with_fip = await agent.predict_pair(m, s)
        assert pred_with_fip is not None

        assert pred_with_fip.true_prob_a < pred_era_only.true_prob_a

    @pytest.mark.asyncio
    async def test_fip_only_pitcher_uses_fip_directly(self, agent):
        """If FIP is present and ERA is not, the agent should use FIP."""
        agent.seed_pitchers(
            {
                "FipOnly Home": {"fip": 3.00, "ip": 180, "team": "yankees"},
                "FipOnly Away": {"fip": 5.00, "ip": 170, "team": "red sox"},
            }
        )
        pred = await agent.predict_pair(_market(), _sharp())
        assert pred is not None
        # 3.00 vs 5.00 should heavily favor home — same as if those were ERAs.
        assert pred.true_prob_a > 0.65

    @pytest.mark.asyncio
    async def test_fip_bumps_confidence_above_era_only(self, agent):
        """When both starters carry FIP, confidence lands in the FIP tier
        (0.75 for IP >= 100), strictly above the ERA-only equivalent (0.65)."""
        agent.seed_pitchers(
            {
                "A": {"era": 3.10, "ip": 180, "team": "yankees"},
                "B": {"era": 3.80, "ip": 170, "team": "red sox"},
            }
        )
        pred_era = await agent.predict_pair(_market(), _sharp())
        assert pred_era is not None
        assert pred_era.confidence == pytest.approx(0.65, abs=1e-6)

        agent.seed_pitchers(
            {
                "A": {"era": 3.10, "fip": 3.30, "ip": 180, "team": "yankees"},
                "B": {"era": 3.80, "fip": 3.60, "ip": 170, "team": "red sox"},
            }
        )
        pred_fip = await agent.predict_pair(_market(), _sharp())
        assert pred_fip is not None
        assert pred_fip.confidence == pytest.approx(0.75, abs=1e-6)
        assert pred_fip.confidence > pred_era.confidence

    @pytest.mark.asyncio
    async def test_fip_only_one_side_falls_to_era_tier(self, agent):
        """The FIP tier requires BOTH starters to carry FIP; partial coverage
        falls back to the ERA-only tier."""
        agent.seed_pitchers(
            {
                "A": {"era": 3.10, "fip": 3.30, "ip": 180, "team": "yankees"},
                "B": {"era": 3.80, "ip": 170, "team": "red sox"},  # no FIP
            }
        )
        pred = await agent.predict_pair(_market(), _sharp())
        assert pred is not None
        assert pred.confidence == pytest.approx(0.65, abs=1e-6)

    @pytest.mark.asyncio
    async def test_fip_fires_in_early_season_with_low_ip(self, agent):
        """Critical for usefulness: with FIP data, even 30-50 IP of current-
        season stats should produce confidence above the 0.45 ensemble gate.
        Pre-FIP, 30 IP would have been static-thin (0.35, dropped)."""
        agent.seed_pitchers(
            {
                "A": {"era": 2.70, "fip": 2.30, "ip": 40, "team": "yankees"},
                "B": {"era": 2.90, "fip": 2.50, "ip": 35, "team": "red sox"},
            }
        )
        pred = await agent.predict_pair(_market(), _sharp())
        assert pred is not None
        assert pred.confidence >= 0.45
        assert pred.confidence == pytest.approx(0.60, abs=1e-6)

    @pytest.mark.asyncio
    async def test_fip_fires_with_thin_15_ip_sample(self, agent):
        """Even thinner samples (15-30 IP, ~2-3 starts) get a 0.50 tier so
        the model contributes through the first month of the season instead
        of waiting until early May. Validated by backtest: pitcher is the
        best single MLB model where it fires, so coverage > exact precision."""
        agent.seed_pitchers(
            {
                "A": {"era": 2.50, "fip": 2.20, "ip": 18, "team": "yankees"},
                "B": {"era": 2.80, "fip": 2.40, "ip": 16, "team": "red sox"},
            }
        )
        pred = await agent.predict_pair(_market(), _sharp())
        assert pred is not None
        assert pred.confidence >= 0.45
        assert pred.confidence == pytest.approx(0.50, abs=1e-6)

    @pytest.mark.asyncio
    async def test_fip_drops_below_15_ip(self, agent):
        """Below 15 IP, FIP-armed predictions still drop to the ERA-only
        thin tier (0.35), below the ensemble gate."""
        agent.seed_pitchers(
            {
                "A": {"era": 2.50, "fip": 2.20, "ip": 8, "team": "yankees"},
                "B": {"era": 2.80, "fip": 2.40, "ip": 6, "team": "red sox"},
            }
        )
        pred = await agent.predict_pair(_market(), _sharp())
        assert pred is not None
        assert pred.confidence < 0.45

    @pytest.mark.asyncio
    async def test_notes_show_fip_when_present(self, agent):
        agent.seed_pitchers(
            {
                "A": {"era": 3.10, "fip": 3.30, "ip": 180, "team": "yankees"},
                "B": {"era": 3.80, "fip": 3.60, "ip": 170, "team": "red sox"},
            }
        )
        pred = await agent.predict_pair(_market(), _sharp())
        assert pred is not None
        assert "fip" in pred.notes.lower()
        assert "effective_home" in pred.notes


# ---------------------------------------------------------------------------
# Sanity: Pythag formula alignment
# ---------------------------------------------------------------------------


class TestAdaptiveHomeAdvantage:
    """Adaptive HOME_BONUS pulled from running estimate of league home WP."""

    def test_no_running_state_uses_default(self, agent):
        assert agent._home_advantage() == HOME_BONUS

    def test_thin_sample_uses_default(self, agent):
        # Below 200 games: even a wildly skewed sample shouldn't override default.
        agent._state["home_wp_running"] = {"games": 100, "home_wins": 80}
        assert agent._home_advantage() == HOME_BONUS

    def test_low_hwp_year_lowers_bonus(self, agent):
        # 2024-style: 51.6% home WP → bonus = 0.016, clamped to floor 0.01.
        agent._state["home_wp_running"] = {"games": 1300, "home_wins": int(1300 * 0.516)}
        hfa = agent._home_advantage()
        assert hfa < HOME_BONUS
        assert hfa >= HOME_BONUS_MIN

    def test_high_hwp_year_raises_bonus_with_ceiling(self, agent):
        # 60% home WP → raw bonus 0.10, capped at HOME_BONUS_MAX (0.07).
        agent._state["home_wp_running"] = {"games": 1300, "home_wins": int(1300 * 0.60)}
        assert agent._home_advantage() == pytest.approx(HOME_BONUS_MAX, abs=1e-6)

    def test_floor_applied_for_very_low_hwp(self, agent):
        # 49% home WP → raw -0.01, clamped to floor 0.01.
        agent._state["home_wp_running"] = {"games": 1300, "home_wins": int(1300 * 0.49)}
        assert agent._home_advantage() == pytest.approx(HOME_BONUS_MIN, abs=1e-6)

    @pytest.mark.asyncio
    async def test_adaptive_bonus_flows_into_prediction(self, agent):
        """Two predictions on identical pitcher data: one with default HFA,
        one with low-HWP running state. The low-HFA prediction should
        favor home less."""
        seed = {
            "Equal A": {"era": 4.00, "fip": 4.00, "ip": 100, "team": "yankees"},
            "Equal B": {"era": 4.00, "fip": 4.00, "ip": 100, "team": "red sox"},
        }
        agent.seed_pitchers(seed)
        pred_default = await agent.predict_pair(_market(), _sharp())

        # Now apply 2024-style low-HWP override
        agent._state["home_wp_running"] = {"games": 1300, "home_wins": int(1300 * 0.516)}
        pred_adapted = await agent.predict_pair(_market(), _sharp())

        assert pred_default is not None and pred_adapted is not None
        assert pred_adapted.true_prob_a < pred_default.true_prob_a


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
        # Post-BUG-9: each team scores at the opposing starter's rate and
        # allows at its own starter's rate.
        home_wp = away_era ** e / (away_era ** e + home_era ** e)
        away_wp = home_era ** e / (home_era ** e + away_era ** e)
        expected = home_wp / (home_wp + away_wp) + HOME_BONUS

        pred = await agent.predict_pair(_market(), _sharp())
        assert pred is not None
        assert pred.true_prob_a == pytest.approx(round(expected, 4), abs=1e-3)
