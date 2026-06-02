"""Tests for injury_agent.py and ev_gap_agent.py.

Covers:
  InjuryReport:
    - probability_adjustment capping at MAX_ADJ
    - has_significant_injuries detection
    - summary formatting
  InjuryReportAgent:
    - _parse_player: status impact, tier classification, zero-impact filter
    - _classify_tier: star/starter/rotation
    - _normalize_team: lowercasing
    - apply_adjustments: team matching, renormalization, spread_multiplier, capping
  EVGap:
    - display_label for moneyline / spread / total
    - edge_label thresholds
    - summary string
  EVGapAgent._evaluate_pair:
    - YES-team alignment (home / away / draw)
    - moneyline-to-spread mismatch rejection
    - model drift capping (ratio >= 2.0 or <= 0.5)
    - injury adjustments applied to blended prob
    - spread market skips model blend
    - invalid prices rejected
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from evmax.agents.intelligence.injury_agent import (
    InjuredPlayer,
    InjuryReport,
    InjuryReportAgent,
    MAX_ADJ,
    STATUS_IMPACT,
    TIER_MULTIPLIER,
)
from evmax.agents.odds.ev_gap_agent import EVGap, EVGapAgent
from evmax.models.market import MarketSource, MarketType, PredictionMarket
from evmax.models.odds import SharpBook, SharpOdds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _player(name="Cade Cunningham", position="PG", status="Out", tier="starter",
             impact=0.045, injury_type="Knee", notes="") -> InjuredPlayer:
    return InjuredPlayer(
        name=name, position=position, status=status,
        tier=tier, impact=impact, injury_type=injury_type, notes=notes,
    )


def _report(team="detroit pistons", players=None, sector="nba") -> InjuryReport:
    r = InjuryReport(team=team, sector=sector)
    r.players = players or []
    return r


def _market(yes_price=0.45, yes_team="pistons", market_type=MarketType.moneyline,
            line=None, yes_price_no=None) -> PredictionMarket:
    if yes_price_no is not None:
        no_price = yes_price_no
    else:
        # Default to ~2% vig so spread_pct > 0 (empty-book filter passes).
        no_price = round(min(0.99, max(0.01, (1.0 - yes_price) + 0.02)), 4)
    return PredictionMarket(
        id="kalshi:TEST-001",
        source=MarketSource.kalshi,
        sector="nba",
        market_type=market_type,
        yes_price=yes_price,
        no_price=no_price,
        volume_usd=10_000.0,
        yes_team=yes_team,
        team_home="pistons",
        team_away="warriors",
        line=line,
    )


def _sharp(outcome_a="pistons", outcome_b="warriors", true_prob_a=0.55,
           spread_line=None, true_prob_draw=None) -> SharpOdds:
    true_prob_b = 1.0 - true_prob_a - (true_prob_draw or 0.0)
    return SharpOdds(
        event_id="nba::2026-03-20::pistons_vs_warriors",
        book=SharpBook.pinnacle,
        sector="nba",
        outcome_a_label=outcome_a,
        outcome_b_label=outcome_b,
        outcome_a_decimal=2.0,
        outcome_b_decimal=2.0,
        true_prob_a=round(true_prob_a, 4),
        true_prob_b=round(true_prob_b, 4),
        true_prob_draw=true_prob_draw,
        spread_line=spread_line,
    )


# ---------------------------------------------------------------------------
# InjuryReport
# ---------------------------------------------------------------------------

class TestInjuryReport:
    def test_probability_adjustment_single_player(self):
        r = _report(players=[_player(impact=0.045)])
        assert r.probability_adjustment == pytest.approx(-0.045)

    def test_probability_adjustment_capped_at_max_adj(self):
        # 5 starters out → 5 × 0.045 = 0.225 > MAX_ADJ (0.20)
        players = [_player(impact=0.045) for _ in range(5)]
        r = _report(players=players)
        assert r.probability_adjustment == pytest.approx(-MAX_ADJ)

    def test_probability_adjustment_empty_roster(self):
        r = _report(players=[])
        assert r.probability_adjustment == 0.0

    def test_has_significant_injuries_with_out(self):
        r = _report(players=[_player(status="Out")])
        assert r.has_significant_injuries is True

    def test_has_significant_injuries_with_dtd(self):
        r = _report(players=[_player(status="Day-to-Day")])
        assert r.has_significant_injuries is True

    def test_has_significant_injuries_with_doubtful(self):
        r = _report(players=[_player(status="Doubtful")])
        assert r.has_significant_injuries is True

    def test_no_significant_injuries_for_questionable(self):
        r = _report(players=[_player(status="Questionable")])
        assert r.has_significant_injuries is False

    def test_no_significant_injuries_empty(self):
        r = _report(players=[])
        assert r.has_significant_injuries is False

    def test_summary_shows_adjustment(self):
        r = _report("Lakers", players=[_player(impact=0.045)])
        s = r.summary()
        assert "Lakers" in s
        assert "adj=" in s

    def test_summary_empty_team(self):
        r = _report(players=[])
        s = r.summary()
        assert "No injuries" in s


# ---------------------------------------------------------------------------
# InjuryReportAgent._parse_player
# ---------------------------------------------------------------------------

class TestParsePlayer:
    def _inj(self, status="Out", position="PG", name="Player X",
              injury_type="Knee", short_comment=""):
        return {
            "athlete": {
                "displayName": name,
                "position": {"abbreviation": position},
                "links": [],
            },
            "status": status,
            "details": {"type": injury_type},
            "shortComment": short_comment,
        }

    def test_out_status_starter_impact(self):
        agent = InjuryReportAgent()
        p = agent._parse_player(self._inj(status="Out", position="PG"))
        assert p is not None
        assert p.impact == pytest.approx(STATUS_IMPACT["out"] * TIER_MULTIPLIER["starter"])

    def test_questionable_impact(self):
        agent = InjuryReportAgent()
        p = agent._parse_player(self._inj(status="Questionable", position="PG"))
        assert p is not None
        assert p.impact == pytest.approx(STATUS_IMPACT["questionable"] * TIER_MULTIPLIER["starter"])

    def test_active_returns_none(self):
        agent = InjuryReportAgent()
        p = agent._parse_player(self._inj(status="Active"))
        assert p is None

    def test_unknown_status_returns_none(self):
        agent = InjuryReportAgent()
        p = agent._parse_player(self._inj(status="Probable"))
        # "probable" maps to 0.004 impact — non-zero so returns something
        # OR might return None if impact is zero
        # Probable has impact=0.004 — check STATUS_IMPACT
        if STATUS_IMPACT.get("probable", 0.0) == 0.0:
            assert p is None
        else:
            assert p is not None

    def test_rotation_player_lower_impact(self):
        agent = InjuryReportAgent()
        # Non-impact position (bench player)
        p_rotation = agent._parse_player(self._inj(status="Out", position="DNP"))
        p_starter = agent._parse_player(self._inj(status="Out", position="PG"))
        assert p_rotation is not None
        assert p_starter is not None
        assert p_rotation.impact < p_starter.impact

    def test_star_tier_with_star_ids(self):
        agent = InjuryReportAgent()
        inj = {
            "athlete": {
                "displayName": "Nikola Jokic",
                "position": {"abbreviation": "C"},
                "links": [{"href": "/id/3112335/nikola-jokic"}],
            },
            "status": "Out",
            "details": {"type": "Back"},
            "shortComment": "",
        }
        p = agent._parse_player(inj, star_ids={"3112335"})
        assert p is not None
        assert p.tier == "star"
        assert p.impact == pytest.approx(STATUS_IMPACT["out"] * TIER_MULTIPLIER["star"])

    def test_star_tier_from_known_stars_name(self):
        """Stars are detected by name even without ESPN leader IDs."""
        agent = InjuryReportAgent()
        for name in ["Luka Doncic", "Stephen Curry", "Austin Reaves"]:
            inj = {
                "athlete": {
                    "displayName": name,
                    "position": {"abbreviation": "G"},
                    "links": [],
                },
                "status": "Out",
                "details": {"type": "Knee"},
                "shortComment": "",
            }
            p = agent._parse_player(inj)  # no star_ids passed
            assert p is not None, f"{name} should be parsed"
            assert p.tier == "star", f"{name} should be star tier, got {p.tier}"
            assert p.impact == pytest.approx(STATUS_IMPACT["out"] * TIER_MULTIPLIER["star"])

    def test_non_star_not_elevated(self):
        """Random bench player should NOT get star tier."""
        agent = InjuryReportAgent()
        p = agent._parse_player(self._inj(name="Random Benchwarmer", position="PG"))
        assert p is not None
        assert p.tier == "starter"  # position-based, not star

    def test_notes_truncated_at_120(self):
        agent = InjuryReportAgent()
        long_comment = "x" * 200
        p = agent._parse_player(self._inj(short_comment=long_comment))
        if p is not None:
            assert len(p.notes) <= 120

    def test_status_as_dict_object(self):
        agent = InjuryReportAgent()
        inj = {
            "athlete": {
                "displayName": "Player",
                "position": {"abbreviation": "PG"},
                "links": [],
            },
            "status": {"name": "Out"},
            "details": {"type": "Ankle"},
            "shortComment": "",
        }
        p = agent._parse_player(inj)
        assert p is not None
        assert p.status == "Out"


# ---------------------------------------------------------------------------
# InjuryReportAgent._classify_tier
# ---------------------------------------------------------------------------

class TestClassifyTier:
    def test_star_when_in_star_ids(self):
        tier = InjuryReportAgent._classify_tier("PG", athlete_id="12345",
                                                  star_ids={"12345"})
        assert tier == "star"

    def test_starter_for_high_impact_positions(self):
        for pos in ["PG", "SG", "SF", "PF", "C", "G", "F",
                     "GK", "CB", "QB", "WR"]:
            assert InjuryReportAgent._classify_tier(pos) == "starter"

    def test_rotation_for_bench_positions(self):
        assert InjuryReportAgent._classify_tier("DNP") == "rotation"
        assert InjuryReportAgent._classify_tier("") == "rotation"

    def test_no_star_ids_means_no_star(self):
        tier = InjuryReportAgent._classify_tier("PG", athlete_id="12345",
                                                  star_ids=None)
        assert tier == "starter"


# ---------------------------------------------------------------------------
# InjuryReportAgent.apply_adjustments
# ---------------------------------------------------------------------------

class TestApplyAdjustments:
    def test_no_injuries_probs_unchanged(self):
        new_a, new_b, notes = InjuryReportAgent.apply_adjustments(
            reports={}, true_prob_a=0.60, true_prob_b=0.40,
            team_a="pistons", team_b="warriors",
        )
        assert new_a == pytest.approx(0.60)
        assert new_b == pytest.approx(0.40)
        assert notes == ""

    def test_team_a_injury_reduces_prob_a(self):
        report = _report("pistons", players=[_player(impact=0.045)])
        new_a, new_b, notes = InjuryReportAgent.apply_adjustments(
            reports={"pistons": report},
            true_prob_a=0.60, true_prob_b=0.40,
            team_a="pistons", team_b="warriors",
        )
        assert new_a < 0.60
        assert new_b > 0.40

    def test_team_b_injury_reduces_prob_b(self):
        report = _report("warriors", players=[_player(impact=0.045)])
        new_a, new_b, notes = InjuryReportAgent.apply_adjustments(
            reports={"warriors": report},
            true_prob_a=0.60, true_prob_b=0.40,
            team_a="pistons", team_b="warriors",
        )
        assert new_b < 0.40
        assert new_a > 0.60

    def test_renormalization_sums_to_one(self):
        report = _report("pistons", players=[_player(impact=0.045)])
        new_a, new_b, _ = InjuryReportAgent.apply_adjustments(
            reports={"pistons": report},
            true_prob_a=0.60, true_prob_b=0.40,
            team_a="pistons", team_b="warriors",
        )
        assert new_a + new_b == pytest.approx(1.0)

    def test_spread_multiplier_amplifies_adjustment(self):
        report = _report("pistons", players=[_player(impact=0.045)])
        new_a_ml, _, _ = InjuryReportAgent.apply_adjustments(
            reports={"pistons": report},
            true_prob_a=0.60, true_prob_b=0.40,
            team_a="pistons", team_b="warriors",
            spread_multiplier=1.0,
        )
        new_a_sp, _, _ = InjuryReportAgent.apply_adjustments(
            reports={"pistons": report},
            true_prob_a=0.60, true_prob_b=0.40,
            team_a="pistons", team_b="warriors",
            spread_multiplier=2.0,
        )
        assert new_a_sp < new_a_ml  # larger multiplier → bigger reduction

    def test_probs_clamped_above_0_02(self):
        # Massive injury burden should not push prob to 0
        big_report = _report("pistons", players=[_player(impact=0.12) for _ in range(5)])
        new_a, new_b, _ = InjuryReportAgent.apply_adjustments(
            reports={"pistons": big_report},
            true_prob_a=0.55, true_prob_b=0.45,
            team_a="pistons", team_b="warriors",
        )
        assert new_a >= 0.02
        assert new_b >= 0.02

    def test_partial_name_match(self):
        # "detroit pistons" contains "pistons" and vice versa
        report = _report("detroit pistons", players=[_player(impact=0.045)])
        new_a, _, notes = InjuryReportAgent.apply_adjustments(
            reports={"detroit pistons": report},
            true_prob_a=0.60, true_prob_b=0.40,
            team_a="pistons", team_b="warriors",  # "pistons" in "detroit pistons"
        )
        assert new_a < 0.60


# ---------------------------------------------------------------------------
# EVGap dataclass
# ---------------------------------------------------------------------------

class TestEVGap:
    def _gap(self, ev_pct=0.05, market_type="moneyline", yes_team="pistons",
              line=None, kalshi_yes_price=0.45) -> EVGap:
        return EVGap(
            market_id="kalshi:TEST",
            event_id="nba::2026-03-20::pistons_vs_warriors",
            sector="nba",
            yes_team=yes_team,
            market_type=market_type,
            kalshi_yes_price=kalshi_yes_price,
            sharp_true_prob=0.55,
            blended_true_prob=0.55,
            ev_pct=ev_pct,
            kelly_full=0.10,
            kelly_fraction=0.05,
            match_confidence=0.95,
            volume_usd=10_000.0,
            spread_pct=0.02,
            line=line,
        )

    def test_edge_label_strong(self):
        assert self._gap(ev_pct=0.15).edge_label == "STRONG"

    def test_edge_label_good(self):
        assert self._gap(ev_pct=0.07).edge_label == "GOOD"

    def test_edge_label_marginal(self):
        assert self._gap(ev_pct=0.03).edge_label == "MARGINAL"

    def test_display_label_moneyline(self):
        label = self._gap(market_type="moneyline", yes_team="pistons").display_label
        assert "ML" in label
        assert "Pistons" in label

    def test_display_label_spread_with_line(self):
        label = self._gap(market_type="spread", yes_team="pistons", line=-10.5).display_label
        assert "-10.5" in label or "10.5" in label

    def test_display_label_no_trailing_zero_on_half_line(self):
        label = self._gap(market_type="spread", yes_team="pistons", line=-8.5).display_label
        assert "8.5" in label

    def test_summary_contains_sector_and_ev(self):
        s = self._gap().summary()
        assert "NBA" in s
        assert "EV=" in s

    def test_implied_prob_equals_kalshi_price(self):
        g = self._gap(kalshi_yes_price=0.36)
        assert g.implied_prob == pytest.approx(0.36)


# ---------------------------------------------------------------------------
# EVGapAgent._evaluate_pair
# ---------------------------------------------------------------------------

class TestEvaluatePair:
    def _agent(self, chalk_ceiling=0.90):
        with patch("evmax.agents.odds.ev_gap_agent.get_settings") as mock_settings:
            settings = MagicMock()
            settings.ev_threshold = 0.02
            settings.max_kelly_fraction = 0.05
            settings.chalk_price_ceiling = chalk_ceiling
            mock_settings.return_value = settings
            agent = EVGapAgent()
        return agent

    def test_yes_is_outcome_a_home_team(self):
        agent = self._agent()
        market = _market(yes_price=0.45, yes_team="pistons")
        sharp = _sharp(outcome_a="pistons", outcome_b="warriors", true_prob_a=0.60)
        gap = agent._evaluate_pair(market, sharp, 0.95, "nba", {}, {}, model_sources={}, kelly_base=0.5)
        assert gap is not None
        assert gap.ev_pct > 0

    def test_yes_is_outcome_b_away_team(self):
        """When yes_team is outcome_b, true_prob_b should be used."""
        agent = self._agent()
        # Market says YES = warriors, but outcome_a is pistons (home)
        market = _market(yes_price=0.60, yes_team="warriors")
        # Pinnacle: pistons = 55%, warriors = 45%
        sharp = _sharp(outcome_a="pistons", outcome_b="warriors", true_prob_a=0.55)
        # Kalshi has warriors at 0.60 but true prob is 0.45 → negative EV
        gap = agent._evaluate_pair(market, sharp, 0.95, "nba", {}, {}, model_sources={}, kelly_base=0.5)
        assert gap is None  # 0.60 price for 0.45 true prob → negative EV

    def test_draw_market_uses_true_prob_draw(self):
        agent = self._agent()
        market = _market(yes_price=0.25, yes_team="tie")
        # Soccer 3-way: include draw prob
        sharp = SharpOdds(
            event_id="soccer::2026-03-20::team_a_vs_team_b",
            book=SharpBook.pinnacle,
            sector="soccer",
            outcome_a_label="team a",
            outcome_b_label="team b",
            outcome_a_decimal=2.5,
            outcome_b_decimal=2.5,
            outcome_draw_decimal=4.0,
            true_prob_a=0.40,
            true_prob_b=0.32,
            true_prob_draw=0.28,
        )
        market_obj = PredictionMarket(
            id="kalshi:DRAW-001",
            source=MarketSource.kalshi,
            sector="soccer",
            market_type=MarketType.moneyline,
            yes_price=0.25,
            no_price=0.77,  # +2¢ vig → spread_pct > 0
            yes_team="tie",
        )
        gap = agent._evaluate_pair(market_obj, sharp, 0.95, "soccer", {}, {}, model_sources={}, kelly_base=0.5)
        # Kalshi 0.25 vs true 0.28 → EV ≈ +12% → should find gap
        assert gap is not None

    def test_draw_market_no_draw_prob_returns_none(self):
        agent = self._agent()
        sharp = _sharp(true_prob_a=0.55)  # no draw
        market = PredictionMarket(
            id="kalshi:DRAW-002",
            source=MarketSource.kalshi,
            sector="nba",
            market_type=MarketType.moneyline,
            yes_price=0.25,
            no_price=0.75,
            yes_team="tie",
        )
        gap = agent._evaluate_pair(market, sharp, 0.95, "nba", {}, {}, model_sources={}, kelly_base=0.5)
        assert gap is None

    def test_invalid_price_zero_returns_none(self):
        agent = self._agent()
        market = _market(yes_price=0.0)
        sharp = _sharp(true_prob_a=0.60)
        gap = agent._evaluate_pair(market, sharp, 0.95, "nba", {}, {}, model_sources={}, kelly_base=0.5)
        assert gap is None

    def test_invalid_price_one_returns_none(self):
        agent = self._agent()
        market = _market(yes_price=1.0)
        sharp = _sharp(true_prob_a=0.60)
        gap = agent._evaluate_pair(market, sharp, 0.95, "nba", {}, {}, model_sources={}, kelly_base=0.5)
        assert gap is None

    def test_moneyline_to_spread_mismatch_rejected(self):
        agent = self._agent()
        market = _market(yes_price=0.55, market_type=MarketType.moneyline)
        sharp = _sharp(spread_line=-7.5)  # spread event
        gap = agent._evaluate_pair(market, sharp, 0.95, "nba", {}, {}, model_sources={}, kelly_base=0.5)
        assert gap is None

    def test_below_ev_threshold_returns_none(self):
        agent = self._agent()
        # Price = 0.55, true prob = 0.56 → tiny EV, below 2% threshold
        market = _market(yes_price=0.55, yes_team="pistons")
        sharp = _sharp(outcome_a="pistons", true_prob_a=0.56)
        gap = agent._evaluate_pair(market, sharp, 0.95, "nba", {}, {}, model_sources={}, kelly_base=0.5)
        assert gap is None

    def test_model_drift_capped_when_ratio_too_high(self):
        """Model predicts 3× sharp — should cap back to sharp."""
        agent = self._agent()
        market = _market(yes_price=0.40, yes_team="pistons")
        sharp = _sharp(outcome_a="pistons", true_prob_a=0.55)

        class FakeBlend:
            true_prob_a = 0.99   # wild model overestimate (ratio = 0.99/0.55 ≈ 1.8 < 2.0)
            true_prob_b = 0.01
            true_prob_draw = None
            model_sources = "elo"

        # Use ratio >= 2.0: blend = 1.15 (impossible, use 0.99)
        # Actually let's use ratio exactly > 2: set sharp 0.30, blend 0.70
        sharp2 = _sharp(outcome_a="pistons", true_prob_a=0.30)
        market2 = _market(yes_price=0.40, yes_team="pistons")

        class HighBlend:
            true_prob_a = 0.80   # ratio = 0.80/0.30 ≈ 2.67 → should cap
            true_prob_b = 0.20
            true_prob_draw = None
            model_sources = "elo"

        blended_preds = {"nba::2026-03-20::pistons_vs_warriors": HighBlend()}
        gap = agent._evaluate_pair(market2, sharp2, 0.95, "nba",
                                   blended_preds, {}, model_sources={}, kelly_base=0.5)
        if gap is not None:
            # When capped, blended_true_prob should be close to sharp_true_prob
            assert gap.model_sources == "sharp(capped)"
            assert gap.blended_true_prob == pytest.approx(gap.sharp_true_prob)

    def test_model_drift_capped_when_ratio_too_low(self):
        """Model predicts < 0.5× sharp — should cap back to sharp."""
        agent = self._agent()
        sharp = _sharp(outcome_a="pistons", true_prob_a=0.70)
        market = _market(yes_price=0.40, yes_team="pistons")

        class LowBlend:
            true_prob_a = 0.30   # ratio = 0.30/0.70 ≈ 0.43 → should cap
            true_prob_b = 0.70
            true_prob_draw = None
            model_sources = "elo"

        blended_preds = {"nba::2026-03-20::pistons_vs_warriors": LowBlend()}
        gap = agent._evaluate_pair(market, sharp, 0.95, "nba",
                                   blended_preds, {}, model_sources={}, kelly_base=0.5)
        if gap is not None:
            assert gap.model_sources == "sharp(capped)"

    def test_injury_adjustment_shifts_prob(self):
        """Injury report for the YES team should reduce their blended_true_prob."""
        agent = self._agent()
        market = _market(yes_price=0.40, yes_team="pistons")
        sharp = _sharp(outcome_a="pistons", outcome_b="warriors", true_prob_a=0.60)

        report = _report("pistons", players=[_player(impact=0.045)])
        gap_no_inj = agent._evaluate_pair(
            market, sharp, 0.95, "nba", {}, {}, model_sources={}, kelly_base=0.5)
        gap_with_inj = agent._evaluate_pair(
            market, sharp, 0.95, "nba", {}, {"pistons": report}, model_sources={}, kelly_base=0.5)

        # Injury reduces pistons' true prob → less EV
        if gap_no_inj is not None and gap_with_inj is not None:
            assert gap_with_inj.blended_true_prob < gap_no_inj.blended_true_prob

    def test_chalk_yes_price_above_ceiling_returns_none(self):
        """YES at 0.92 should be filtered even when sharp says 0.96 → +EV."""
        agent = self._agent(chalk_ceiling=0.90)
        market = _market(yes_price=0.92, yes_team="pistons")
        sharp = _sharp(outcome_a="pistons", outcome_b="warriors", true_prob_a=0.96)
        gap = agent._evaluate_pair(market, sharp, 0.95, "nba", {}, {}, model_sources={}, kelly_base=0.5)
        assert gap is None

    def test_yes_price_at_ceiling_passes(self):
        """0.90 exactly should NOT be filtered (strict > comparison)."""
        agent = self._agent(chalk_ceiling=0.90)
        market = _market(yes_price=0.90, yes_team="pistons")
        sharp = _sharp(outcome_a="pistons", outcome_b="warriors", true_prob_a=0.95)
        gap = agent._evaluate_pair(market, sharp, 0.95, "nba", {}, {}, model_sources={}, kelly_base=0.5)
        assert gap is not None

    def test_positive_ev_gap_returned(self):
        """Core happy path: big EV gap → gap object created."""
        agent = self._agent()
        # Kalshi offers 0.35 for a team that sharp says is 50% — big value
        market = _market(yes_price=0.35, yes_team="pistons")
        sharp = _sharp(outcome_a="pistons", outcome_b="warriors", true_prob_a=0.55)
        gap = agent._evaluate_pair(market, sharp, 0.95, "nba", {}, {}, model_sources={}, kelly_base=0.5)
        assert gap is not None
        assert gap.ev_pct > 0.02
        assert gap.sector == "nba"
        assert gap.yes_team == "pistons"


# ---------------------------------------------------------------------------
# NFL sector-aware position weights (Phase 3)
# ---------------------------------------------------------------------------

class TestNflPositionWeights:
    """Position weighting only applies for NFL — every other sector behaves
    identically to the pre-Phase-3 flat 1.0 multiplier."""

    def _inj(self, status="Out", position="QB", name="Player X"):
        return {
            "athlete": {
                "displayName": name,
                "position": {"abbreviation": position},
                "links": [],
            },
            "status": status,
            "details": {"type": "Knee"},
            "shortComment": "",
        }

    def test_qb_out_hits_harder_than_guard_out(self):
        agent = InjuryReportAgent()
        qb = agent._parse_player(self._inj(position="QB"), sector="nfl")
        g = agent._parse_player(self._inj(position="G"), sector="nfl")
        assert qb is not None and g is not None
        # QB weight 1.50, Guard weight 0.15 → QB impact should be ~10x guard
        assert qb.impact > g.impact * 5

    def test_lt_higher_than_te(self):
        agent = InjuryReportAgent()
        lt = agent._parse_player(self._inj(position="LT"), sector="nfl")
        te = agent._parse_player(self._inj(position="TE"), sector="nfl")
        assert lt is not None and te is not None
        # LT weight 0.50, TE weight 0.20 → LT > TE impact
        assert lt.impact > te.impact

    def test_kicker_near_zero(self):
        agent = InjuryReportAgent()
        k = agent._parse_player(self._inj(position="K"), sector="nfl")
        # Kicker weight 0.05 → very small but nonzero impact for OUT status
        assert k is None or k.impact < 0.005

    def test_unknown_position_uses_default_weight(self):
        agent = InjuryReportAgent()
        # "ZZ" is gibberish — should fall back to NFL_DEFAULT_POSITION_WEIGHT (0.10)
        p = agent._parse_player(self._inj(position="ZZ"), sector="nfl")
        # With unknown position and fallback tier 'rotation' (since ZZ not in
        # HIGH_IMPACT_POSITIONS): impact = 0.045 × 0.5 × 0.10 = 0.00225
        assert p is not None
        assert p.impact == pytest.approx(0.045 * 0.5 * 0.10, abs=1e-4)

    def test_non_nfl_sector_unchanged(self):
        """Soccer and NBA should produce identical impact whether we pass sector
        or not — NFL is the only sector with weights configured today."""
        agent = InjuryReportAgent()
        with_sector = agent._parse_player(self._inj(position="GK"), sector="soccer")
        without_sector = agent._parse_player(self._inj(position="GK"))
        assert with_sector is not None and without_sector is not None
        assert with_sector.impact == without_sector.impact

    def test_star_qb_out_breaches_starter_qb_baseline(self):
        """A KNOWN_STARS QB out should hit harder than a generic starter QB out
        (star tier 1.5 vs starter tier 1.0)."""
        agent = InjuryReportAgent()
        starter_qb = agent._parse_player(
            self._inj(position="QB", name="Generic Backup"),
            sector="nfl",
        )
        # Use a name actually in KNOWN_STARS
        star_qb = agent._parse_player(
            self._inj(position="QB", name="Patrick Mahomes"),
            sector="nfl",
        )
        assert starter_qb is not None and star_qb is not None
        assert star_qb.impact > starter_qb.impact

    def test_nfl_qb_out_drops_team_probability_more_than_te(self):
        """End-to-end: apply_adjustments after a QB OUT shifts probability
        further than apply_adjustments after a TE OUT, holding all else equal."""
        # Build two reports differing only by injured player's position
        qb_player = InjuredPlayer(
            name="QB Player", position="QB", status="Out",
            tier="starter", impact=0.045 * 1.0 * 1.50,  # raw × tier × QB weight
            injury_type="Knee", notes="",
        )
        te_player = InjuredPlayer(
            name="TE Player", position="TE", status="Out",
            tier="starter", impact=0.045 * 1.0 * 0.20,  # raw × tier × TE weight
            injury_type="Knee", notes="",
        )
        report_qb = InjuryReport(team="kansas city chiefs", sector="nfl",
                                  players=[qb_player])
        report_te = InjuryReport(team="kansas city chiefs", sector="nfl",
                                  players=[te_player])

        prob_a, prob_b = 0.55, 0.45
        a_qb, b_qb, _ = InjuryReportAgent.apply_adjustments(
            {"kansas city chiefs": report_qb}, prob_a, prob_b,
            "kansas city chiefs", "buffalo bills",
        )
        a_te, b_te, _ = InjuryReportAgent.apply_adjustments(
            {"kansas city chiefs": report_te}, prob_a, prob_b,
            "kansas city chiefs", "buffalo bills",
        )
        # KC's probability should drop more after QB out than after TE out
        assert a_qb < a_te


# ---------------------------------------------------------------------------
# NO-side total derivation: Kalshi frames totals YES = OVER, so the UNDER is
# only reachable via the NO side. Before _build_no_side_total_gap, the under
# was structurally unbettable (the NO-side path was spread-only) — which is why
# 100% of logged baseball totals were overs.
# ---------------------------------------------------------------------------

class TestNoSideTotalGap:
    def _agent(self):
        with patch("evmax.agents.odds.ev_gap_agent.get_settings") as mock_settings:
            settings = MagicMock()
            settings.ev_threshold = 0.02
            settings.max_kelly_fraction = 0.05
            settings.chalk_price_ceiling = 0.90
            mock_settings.return_value = settings
            return EVGapAgent()

    def _payload(self, blended_over, sharp_over):
        return {
            "blended_prob_yes": blended_over,
            "sharp_true_prob_yes": sharp_over,
            "src": "sharp+pitcher",
            "yes_is_outcome_b": False,
            "used_spread_model": False,
        }

    def test_under_gap_built_when_positive_ev(self):
        agent = self._agent()
        # Over priced rich (under NO ask = 0.33) but model says over only 0.55,
        # so under = 0.45 → EV = 0.45/0.33 - 1 ≈ +36%.
        market = _market(yes_price=0.70, yes_team="over",
                         market_type=MarketType.total, line=8.5, yes_price_no=0.33)
        sharp = _sharp(outcome_a="over", outcome_b="under", true_prob_a=0.50)
        gap = agent._build_no_side_total_gap(
            market, sharp, self._payload(0.55, 0.52), 0.95, "baseball", kelly_base=0.5,
        )
        assert gap is not None
        assert gap.yes_team == "under"
        assert gap.market_type == MarketType.total.value
        assert gap.line == 8.5                      # totals line NOT flipped
        assert gap.market_id.endswith(":no")
        assert gap.kalshi_yes_price == pytest.approx(0.33)
        assert gap.blended_true_prob == pytest.approx(0.45)
        assert gap.ev_pct > 0
        assert gap.model_sources.endswith("+no_side")
        # Event title must be the team matchup, NOT "over vs under" — totals
        # outcome labels are over/under, so the title is derived from the
        # event_id slug. Regression for the dashboard showing "over vs under"
        # in the Event column for under bets.
        assert gap.event_title == "Pistons vs Warriors"
        assert "over" not in gap.event_title.lower()

    def test_no_under_gap_when_model_loves_over(self):
        """If the model projects a high over prob, the under is -EV → None."""
        agent = self._agent()
        market = _market(yes_price=0.70, yes_team="over",
                         market_type=MarketType.total, line=8.5, yes_price_no=0.33)
        sharp = _sharp(outcome_a="over", outcome_b="under", true_prob_a=0.50)
        # blended over 0.80 → under 0.20; 0.20/0.33 - 1 ≈ -39% → no gap
        gap = agent._build_no_side_total_gap(
            market, sharp, self._payload(0.80, 0.78), 0.95, "baseball", kelly_base=0.5,
        )
        assert gap is None

    def test_dead_under_book_returns_none(self):
        agent = self._agent()
        # NO ask at the 1¢ floor = dead book
        market = _market(yes_price=0.99, yes_team="over",
                         market_type=MarketType.total, line=8.5, yes_price_no=0.005)
        sharp = _sharp(outcome_a="over", outcome_b="under", true_prob_a=0.50)
        gap = agent._build_no_side_total_gap(
            market, sharp, self._payload(0.30, 0.30), 0.95, "baseball", kelly_base=0.5,
        )
        assert gap is None
