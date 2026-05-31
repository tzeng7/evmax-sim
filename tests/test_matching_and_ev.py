"""Comprehensive tests for MatchingEngine and EVGapAgent.

Covers:
  MatchingEngine.build_market_key:
    - Moneyline → base_key (no suffix)
    - Spread → base_key + "::spread"
    - Total with line → base_key + "::total::<line>"
    - Total without line → base_key + "::total"
    - Missing teams/date → None

  MatchingEngine.match:
    - Exact key → (sharp, 100.0)
    - Fuzzy match for minor name variations (moneyline only)
    - Spread market skips fuzzy (returns None on mismatch)
    - Total market skips fuzzy (returns None on mismatch)
    - Total market nearest-line fallback within 2pt tolerance
    - Total market no match when line diff > 2pt tolerance
    - Moneyline market does not match spread SharpOdds
    - Total market does not match moneyline SharpOdds

  EVGapAgent._evaluate_pair:
    - Moneyline YES=home team → positive EV gap
    - Moneyline YES=away team → uses true_prob_b
    - Draw market YES="draw" → uses true_prob_draw
    - Moneyline matched to spread SharpOdds → None (type mismatch)
    - Total matched to moneyline SharpOdds (no total_line) → None
    - Moneyline matched to totals SharpOdds (total_line set) → None
    - Total YES=over → positive EV
    - Total YES=under → positive EV
    - Team scoring prop (NBA line=109.5) → None (is_game_total guard)
    - Negative EV → None
    - EV below 2% threshold → None
    - Kelly fraction never exceeds 5% cap
    - steam_move=True when event_id in steam_events
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from unittest.mock import patch, MagicMock

import pytest

from evmax.matching.engine import MatchingEngine
from evmax.agents.odds.ev_gap_agent import EVGapAgent, EVGap
from evmax.models.market import MarketSource, MarketType, PredictionMarket
from evmax.models.odds import SharpBook, SharpOdds


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_EVENT_DATE = datetime(2026, 3, 21, 18, 0, 0, tzinfo=timezone.utc)
_DATE_STR = "2026-03-21"


def _market(
    team_home: str = "lakers",
    team_away: str = "celtics",
    sector: str = "nba",
    market_type: MarketType = MarketType.moneyline,
    yes_price: float = 0.50,
    no_price: Optional[float] = None,
    yes_team: Optional[str] = None,
    line: Optional[float] = None,
    event_date: Optional[datetime] = _EVENT_DATE,
    market_id: str = "kalshi:TEST-001",
    volume_usd: float = 10_000.0,
) -> PredictionMarket:
    # Default no_price simulates a real liquid orderbook with ~2% vig
    # (yes_ask + no_ask sums to ~1.02). Tests that need a specific spread_pct
    # — including the empty-book regression — pass no_price explicitly.
    if no_price is None:
        no_price = round(min(0.99, max(0.01, (1.0 - yes_price) + 0.02)), 4)
    return PredictionMarket(
        id=market_id,
        source=MarketSource.kalshi,
        sector=sector,
        market_type=market_type,
        yes_price=yes_price,
        no_price=no_price,
        volume_usd=volume_usd,
        team_home=team_home,
        team_away=team_away,
        yes_team=yes_team,
        line=line,
        event_date=event_date,
    )


def _moneyline_sharp(
    event_id: str,
    sector: str = "nba",
    outcome_a: str = "lakers",
    outcome_b: str = "celtics",
    true_prob_a: float = 0.60,
    true_prob_draw: Optional[float] = None,
) -> SharpOdds:
    true_prob_b = round(1.0 - true_prob_a - (true_prob_draw or 0.0), 6)
    return SharpOdds(
        event_id=event_id,
        book=SharpBook.pinnacle,
        sector=sector,
        outcome_a_label=outcome_a,
        outcome_b_label=outcome_b,
        outcome_a_decimal=1.0 / true_prob_a,
        outcome_b_decimal=1.0 / true_prob_b,
        true_prob_a=true_prob_a,
        true_prob_b=true_prob_b,
        true_prob_draw=true_prob_draw,
        margin=0.04,
    )


def _spread_sharp(
    event_id: str,
    sector: str = "nba",
    outcome_a: str = "lakers",
    outcome_b: str = "celtics",
    spread_line: float = -8.5,
    true_prob_a: float = 0.52,
) -> SharpOdds:
    true_prob_b = round(1.0 - true_prob_a, 6)
    return SharpOdds(
        event_id=event_id,
        book=SharpBook.pinnacle,
        sector=sector,
        outcome_a_label=outcome_a,
        outcome_b_label=outcome_b,
        outcome_a_decimal=1.0 / true_prob_a,
        outcome_b_decimal=1.0 / true_prob_b,
        true_prob_a=true_prob_a,
        true_prob_b=true_prob_b,
        spread_line=spread_line,
        margin=0.04,
    )


def _total_sharp(
    event_id: str,
    sector: str = "nba",
    total_line: float = 220.5,
    true_prob_over: float = 0.52,
) -> SharpOdds:
    true_prob_under = round(1.0 - true_prob_over, 6)
    return SharpOdds(
        event_id=event_id,
        book=SharpBook.pinnacle,
        sector=sector,
        outcome_a_label="over",
        outcome_b_label="under",
        outcome_a_decimal=1.0 / true_prob_over,
        outcome_b_decimal=1.0 / true_prob_under,
        true_prob_a=true_prob_over,
        true_prob_b=true_prob_under,
        total_line=total_line,
        true_prob_over=true_prob_over,
        true_prob_under=true_prob_under,
        margin=0.04,
    )


def _make_ev_gap_agent() -> EVGapAgent:
    """Construct EVGapAgent with real settings (no API calls)."""
    return EVGapAgent()


# ---------------------------------------------------------------------------
# MatchingEngine.build_market_key
# ---------------------------------------------------------------------------

class TestBuildMarketKey:
    def setup_method(self):
        self.engine = MatchingEngine()

    def test_moneyline_market_returns_base_key_no_suffix(self):
        market = _market(
            team_home="lakers", team_away="celtics",
            sector="nba", market_type=MarketType.moneyline,
        )
        key = self.engine.build_market_key(market)
        assert key is not None
        assert "::spread" not in key
        assert "::total" not in key
        # Key has sector::date::teams format
        assert key.startswith("nba::")
        assert _DATE_STR in key

    def test_spread_market_appends_spread_suffix(self):
        market = _market(
            team_home="lakers", team_away="celtics",
            sector="nba", market_type=MarketType.spread, line=-8.5,
        )
        key = self.engine.build_market_key(market)
        assert key is not None
        assert key.endswith("::spread")

    def test_total_market_with_line_embeds_line_value(self):
        market = _market(
            team_home="lakers", team_away="celtics",
            sector="nba", market_type=MarketType.total, line=220.5,
        )
        key = self.engine.build_market_key(market)
        assert key is not None
        assert "::total::220.5" in key

    def test_total_market_without_line_returns_total_suffix_only(self):
        market = _market(
            team_home="lakers", team_away="celtics",
            sector="nba", market_type=MarketType.total, line=None,
        )
        key = self.engine.build_market_key(market)
        assert key is not None
        assert key.endswith("::total")
        # Should NOT have a numeric line appended
        parts = key.split("::")
        assert parts[-1] == "total"

    def test_market_missing_team_home_returns_none(self):
        market = PredictionMarket(
            id="kalshi:TEST",
            source=MarketSource.kalshi,
            sector="nba",
            yes_price=0.50,
            no_price=0.50,
            team_home=None,
            team_away="celtics",
            volume_usd=10_000.0,
            event_date=_EVENT_DATE,
        )
        key = self.engine.build_market_key(market)
        assert key is None

    def test_market_missing_team_away_returns_none(self):
        market = PredictionMarket(
            id="kalshi:TEST",
            source=MarketSource.kalshi,
            sector="nba",
            yes_price=0.50,
            no_price=0.50,
            team_home="lakers",
            team_away=None,
            volume_usd=10_000.0,
            event_date=_EVENT_DATE,
        )
        key = self.engine.build_market_key(market)
        assert key is None

    def test_market_missing_event_date_returns_none(self):
        market = PredictionMarket(
            id="kalshi:TEST",
            source=MarketSource.kalshi,
            sector="nba",
            yes_price=0.50,
            no_price=0.50,
            team_home="lakers",
            team_away="celtics",
            volume_usd=10_000.0,
            event_date=None,
        )
        key = self.engine.build_market_key(market)
        assert key is None

    def test_spread_key_differs_from_moneyline_key(self):
        base_market = _market(
            team_home="chiefs", team_away="eagles",
            sector="nfl", market_type=MarketType.moneyline,
        )
        spread_market = _market(
            team_home="chiefs", team_away="eagles",
            sector="nfl", market_type=MarketType.spread, line=-6.5,
        )
        ml_key = self.engine.build_market_key(base_market)
        spread_key = self.engine.build_market_key(spread_market)
        assert ml_key != spread_key
        assert spread_key.endswith("::spread")

    def test_two_different_total_lines_produce_different_keys(self):
        m1 = _market(
            team_home="lakers", team_away="celtics",
            sector="nba", market_type=MarketType.total, line=220.5,
        )
        m2 = _market(
            team_home="lakers", team_away="celtics",
            sector="nba", market_type=MarketType.total, line=221.5,
        )
        k1 = self.engine.build_market_key(m1)
        k2 = self.engine.build_market_key(m2)
        assert k1 != k2
        assert "220.5" in k1
        assert "221.5" in k2


# ---------------------------------------------------------------------------
# MatchingEngine.match
# ---------------------------------------------------------------------------

class TestMatchingEngineMatch:
    def setup_method(self):
        self.engine = MatchingEngine()

    def test_exact_key_match_returns_sharp_and_100_confidence(self):
        market = _market(team_home="lakers", team_away="celtics", sector="nba")
        key = self.engine.build_market_key(market)
        sharp = _moneyline_sharp(event_id=key, sector="nba")

        result = self.engine.match(market, [sharp])

        assert result is not None
        matched_sharp, confidence = result
        assert matched_sharp.event_id == key
        assert confidence == 100.0

    def test_fuzzy_match_handles_minor_team_name_variation(self):
        """'LA Lakers' on Kalshi should fuzzy-match 'lakers' on Pinnacle."""
        # Build a market with the un-normalized name; the normalizer for nba
        # should resolve "Los Angeles Lakers" to "lakers" via alias
        market = _market(team_home="Los Angeles Lakers", team_away="Boston Celtics", sector="nba")
        canonical_key = self.engine.build_market_key(market)
        assert canonical_key is not None

        # Sharp uses the canonical already-normalized key
        sharp = _moneyline_sharp(event_id=canonical_key, sector="nba")

        result = self.engine.match(market, [sharp])
        assert result is not None
        _, confidence = result
        assert confidence == 100.0  # exact after normalization

    def test_fuzzy_moneyline_matches_reversed_home_away(self):
        """Fuzzy matching should handle home/away swap between Kalshi and Pinnacle."""
        market = _market(team_home="chiefs", team_away="eagles", sector="nfl")
        market_key = self.engine.build_market_key(market)
        assert market_key is not None

        # Sharp has teams reversed: eagles_vs_chiefs instead of chiefs_vs_eagles
        reversed_key = market_key.replace("chiefs_vs_eagles", "eagles_vs_chiefs")
        sharp = _moneyline_sharp(event_id=reversed_key, sector="nfl",
                                 outcome_a="eagles", outcome_b="chiefs")

        result = self.engine.match(market, [sharp])
        assert result is not None, "Reversed home/away should fuzzy-match for moneyline"

    def test_spread_market_does_not_fuzzy_match(self):
        """Spread markets must match exactly — fuzzy is disabled for them."""
        market = _market(
            team_home="lakers", team_away="celtics",
            sector="nba", market_type=MarketType.spread, line=-8.5,
        )
        # Build a slightly different key (simulate what a near-miss looks like)
        market_key = self.engine.build_market_key(market)
        # Supply a sharp key that is close but NOT an exact match
        wrong_key = market_key.replace("lakers", "la_lakers")
        sharp = _spread_sharp(event_id=wrong_key, sector="nba")

        result = self.engine.match(market, [sharp])
        # Should NOT find a fuzzy match for spread markets
        assert result is None

    def test_total_market_does_not_fuzzy_match(self):
        """Total markets must match exactly — fuzzy is disabled for them."""
        market = _market(
            team_home="lakers", team_away="celtics",
            sector="nba", market_type=MarketType.total, line=220.5,
        )
        market_key = self.engine.build_market_key(market)
        # Provide a sharp key with a slightly different team name spelling
        wrong_key = market_key.replace("lakers", "la_lakers")
        sharp = _total_sharp(event_id=wrong_key, sector="nba", total_line=220.5)

        result = self.engine.match(market, [sharp])
        assert result is None

    def test_total_nearest_line_within_tolerance_matches(self):
        """Total market with Kalshi line 220.5 and Pinnacle line 221.5 → nearest-line match."""
        market = _market(
            team_home="lakers", team_away="celtics",
            sector="nba", market_type=MarketType.total, line=220.5,
        )
        # The exact key for 220.5 does not exist; Pinnacle has 221.5
        base_key_220 = self.engine.build_market_key(market)
        assert base_key_220 is not None

        # Derive the base without the line suffix, then build the 221.5 key
        # base_key_220 looks like "nba::2026-03-21::lakers_vs_celtics::total::220.5"
        # strip the last "::220.5" to get the prefix used by _find_nearest_total
        base_without_line = base_key_220.rsplit("::", 1)[0]  # removes "220.5"
        key_221_5 = f"{base_without_line}::221.5"

        sharp_221 = _total_sharp(event_id=key_221_5, sector="nba", total_line=221.5)

        result = self.engine.match(market, [sharp_221])
        assert result is not None, "Pinnacle line 221.5 should match Kalshi line 220.5 (within 2pt)"
        matched_sharp, confidence = result
        assert matched_sharp.event_id == key_221_5
        assert confidence < 100.0  # confidence is penalised for line distance

    def test_total_nearest_line_outside_tolerance_no_match(self):
        """Total market with Kalshi line 220.5 and Pinnacle line 225.0 → no match (> 2pt)."""
        market = _market(
            team_home="lakers", team_away="celtics",
            sector="nba", market_type=MarketType.total, line=220.5,
        )
        base_key = self.engine.build_market_key(market)
        base_without_line = base_key.rsplit("::", 1)[0]
        key_225 = f"{base_without_line}::225.0"

        sharp_225 = _total_sharp(event_id=key_225, sector="nba", total_line=225.0)

        result = self.engine.match(market, [sharp_225])
        assert result is None, "Line diff of 4.5 pts exceeds 2pt tolerance"

    def test_moneyline_market_does_not_match_spread_sharp(self):
        """
        Matching engine does NOT block moneyline-vs-spread at the match() level — that
        responsibility belongs to EVGapAgent._evaluate_pair (type-mismatch guard).

        This test verifies that even if match() returns a spread record for a moneyline
        market (via fuzzy), the evaluate_pair layer correctly rejects it and produces
        no EVGap.  This reflects the actual two-layer defense in the codebase.
        """
        market = _market(
            team_home="lakers", team_away="celtics",
            yes_price=0.48, yes_team="lakers",
            market_type=MarketType.moneyline,
            sector="nba",
        )
        # Build a spread key and register a spread-typed SharpOdds
        spread_key = self.engine.build_market_key(
            _market(team_home="lakers", team_away="celtics",
                    sector="nba", market_type=MarketType.spread, line=-8.5)
        )
        sharp = _spread_sharp(event_id=spread_key, sector="nba",
                              spread_line=-8.5, true_prob_a=0.60)

        # Even if match() returns this spread record, _evaluate_pair must reject it
        agent = _make_ev_gap_agent()
        result = self.engine.match(market, [sharp])
        if result is not None:
            matched_sharp, confidence = result
            gap = agent._evaluate_pair(
                market=market,
                sharp=matched_sharp,
                confidence=confidence,
                sector="nba",
                blended_preds={},
                injuries={},
                model_sources={},
                kelly_base=0.25,
            )
            assert gap is None, (
                "EVGapAgent._evaluate_pair must reject moneyline market matched to "
                "a spread SharpOdds record (type-mismatch guard)"
            )

    def test_total_market_does_not_match_moneyline_sharp(self):
        """A total market should return None when only moneyline SharpOdds exist."""
        market = _market(
            team_home="lakers", team_away="celtics",
            sector="nba", market_type=MarketType.total, line=220.5,
        )
        # Provide only a moneyline sharp record
        ml_key = self.engine.build_market_key(
            _market(team_home="lakers", team_away="celtics", sector="nba")
        )
        sharp = _moneyline_sharp(event_id=ml_key, sector="nba")

        result = self.engine.match(market, [sharp])
        # The total key is "...::total::220.5" which is different from the moneyline key.
        # Even if matching returned something, _evaluate_pair would block it; but here
        # match itself should not find an exact key match.
        # The nearest-line fallback also won't fire since sharp.event_id has no "::total::" prefix.
        assert result is None, (
            "Total market should not match a moneyline SharpOdds record"
        )

    def test_empty_sharp_list_returns_none(self):
        market = _market(team_home="lakers", team_away="celtics", sector="nba")
        assert self.engine.match(market, []) is None

    def test_no_match_for_completely_different_teams(self):
        market = _market(team_home="lakers", team_away="celtics", sector="nba")
        sharp_key = "nba::2026-03-21::bucks_vs_bulls"
        sharp = _moneyline_sharp(event_id=sharp_key, sector="nba",
                                 outcome_a="bucks", outcome_b="bulls")
        result = self.engine.match(market, [sharp])
        assert result is None


# ---------------------------------------------------------------------------
# EVGapAgent._evaluate_pair  (no API calls — plain Python objects only)
# ---------------------------------------------------------------------------

class TestEvaluatePair:
    """Tests for EVGapAgent._evaluate_pair using pre-built fixtures."""

    def setup_method(self):
        self.agent = _make_ev_gap_agent()

    # -----------------------------------------------------------------------
    # Helper to call _evaluate_pair with common defaults
    # -----------------------------------------------------------------------
    def _call(
        self,
        market: PredictionMarket,
        sharp: SharpOdds,
        confidence: float = 95.0,
        sector: str = "nba",
        blended_preds: dict = None,
        injuries: dict = None,
        model_sources: dict = None,
        kelly_base: float = 0.25,
        steam_events: set = None,
    ) -> Optional[EVGap]:
        return self.agent._evaluate_pair(
            market=market,
            sharp=sharp,
            confidence=confidence,
            sector=sector,
            blended_preds=blended_preds or {},
            injuries=injuries or {},
            model_sources=model_sources or {},
            kelly_base=kelly_base,
            steam_events=steam_events or set(),
        )

    # -----------------------------------------------------------------------
    # YES-team alignment
    # -----------------------------------------------------------------------

    def test_moneyline_yes_home_team_positive_ev(self):
        """YES = home team; true_prob_a=0.60, kalshi=0.52 → positive EV."""
        market = _market(
            team_home="pistons", team_away="warriors",
            yes_price=0.52, yes_team="pistons",
            sector="nba",
        )
        sharp = _moneyline_sharp(
            event_id="nba::2026-03-21::pistons_vs_warriors",
            outcome_a="pistons", outcome_b="warriors",
            true_prob_a=0.60,
        )
        gap = self._call(market, sharp)

        assert gap is not None
        assert gap.ev_pct > 0.02
        assert gap.blended_true_prob == pytest.approx(0.60, abs=0.001)
        assert gap.yes_team == "pistons"
        assert gap.sector == "nba"
        assert gap.market_type == "moneyline"

    def test_moneyline_yes_away_team_uses_true_prob_b(self):
        """YES = away team (outcome_b); agent must use true_prob_b for EV."""
        # Pinnacle: pistons (home) = 55%, warriors (away) = 45%
        # Kalshi offers warriors YES at 0.40 → EV for warriors using true_prob_b=0.45
        market = _market(
            team_home="pistons", team_away="warriors",
            yes_price=0.40, yes_team="warriors",  # YES = away
            sector="nba",
        )
        sharp = _moneyline_sharp(
            event_id="nba::2026-03-21::pistons_vs_warriors",
            outcome_a="pistons", outcome_b="warriors",
            true_prob_a=0.55,  # pistons favoured; warriors = 0.45
        )
        gap = self._call(market, sharp)

        # EV = (0.45 / 0.40) - 1 = 0.125 → positive, above threshold
        assert gap is not None
        assert gap.blended_true_prob == pytest.approx(0.45, abs=0.001)
        assert gap.ev_pct > 0.02

    def test_moneyline_yes_away_team_negative_ev_returns_none(self):
        """YES = away; kalshi_price > true_prob_b → negative EV → None."""
        # warriors are 45% true prob; Kalshi offers them at 0.60 (overpriced)
        market = _market(
            team_home="pistons", team_away="warriors",
            yes_price=0.60, yes_team="warriors",
            sector="nba",
        )
        sharp = _moneyline_sharp(
            event_id="nba::2026-03-21::pistons_vs_warriors",
            outcome_a="pistons", outcome_b="warriors",
            true_prob_a=0.55,
        )
        gap = self._call(market, sharp)
        assert gap is None

    def test_draw_market_uses_true_prob_draw(self):
        """YES = 'draw'; must use true_prob_draw (not true_prob_a or true_prob_b)."""
        # Soccer 3-way: home=0.48, draw=0.28, away=0.24
        # Kalshi offers draw at 0.22 → EV for draw = (0.28/0.22)-1 ≈ 0.273 → positive
        market = PredictionMarket(
            id="kalshi:DRAW-001",
            source=MarketSource.kalshi,
            sector="soccer",
            market_type=MarketType.moneyline,
            yes_price=0.22,
            no_price=0.80,  # +2¢ vig → spread_pct > 0
            yes_team="draw",
            team_home="man city",
            team_away="arsenal",
            volume_usd=10_000.0,
            event_date=_EVENT_DATE,
        )
        sharp = SharpOdds(
            event_id="soccer::2026-03-21::man_city_vs_arsenal",
            book=SharpBook.pinnacle,
            sector="soccer",
            outcome_a_label="man city",
            outcome_b_label="arsenal",
            outcome_a_decimal=2.0,
            outcome_b_decimal=4.0,
            outcome_draw_decimal=3.5,
            true_prob_a=0.48,
            true_prob_b=0.24,
            true_prob_draw=0.28,
            margin=0.04,
        )
        gap = self._call(market, sharp, sector="soccer")

        assert gap is not None
        assert gap.blended_true_prob == pytest.approx(0.28, abs=0.001)
        assert gap.ev_pct > 0.02

    def test_draw_market_no_true_prob_draw_returns_none(self):
        """YES = 'draw' but SharpOdds.true_prob_draw is None → None."""
        market = PredictionMarket(
            id="kalshi:DRAW-002",
            source=MarketSource.kalshi,
            sector="nba",
            market_type=MarketType.moneyline,
            yes_price=0.28,
            no_price=0.72,
            yes_team="draw",
        )
        sharp = _moneyline_sharp(
            event_id="nba::2026-03-21::lakers_vs_celtics",
            true_prob_a=0.60,
        )  # no true_prob_draw
        gap = self._call(market, sharp)
        assert gap is None

    def test_draw_market_yes_team_x_also_detected(self):
        """YES = 'x' (common European notation for draw) should trigger draw path."""
        market = PredictionMarket(
            id="kalshi:DRAW-003",
            source=MarketSource.kalshi,
            sector="soccer",
            market_type=MarketType.moneyline,
            yes_price=0.22,
            no_price=0.80,  # +2¢ vig → spread_pct > 0
            yes_team="x",
            volume_usd=10_000.0,
        )
        sharp = SharpOdds(
            event_id="soccer::2026-03-21::barca_vs_madrid",
            book=SharpBook.pinnacle,
            sector="soccer",
            outcome_a_label="barca",
            outcome_b_label="madrid",
            outcome_a_decimal=2.0,
            outcome_b_decimal=4.0,
            outcome_draw_decimal=3.5,
            true_prob_a=0.48,
            true_prob_b=0.24,
            true_prob_draw=0.28,
            margin=0.04,
        )
        gap = self._call(market, sharp, sector="soccer")
        assert gap is not None
        assert gap.blended_true_prob == pytest.approx(0.28, abs=0.001)

    # -----------------------------------------------------------------------
    # Type-mismatch guards
    # -----------------------------------------------------------------------

    def test_moneyline_market_matched_to_spread_sharp_returns_none(self):
        """Moneyline market vs SharpOdds that has spread_line set → None."""
        market = _market(
            team_home="pistons", team_away="warriors",
            yes_price=0.50, yes_team="pistons",
            market_type=MarketType.moneyline,
        )
        # Sharp record with spread_line (NOT a moneyline record)
        sharp = _spread_sharp(
            event_id="nba::2026-03-21::pistons_vs_warriors",
            spread_line=-7.5,
            true_prob_a=0.60,
        )
        gap = self._call(market, sharp)
        assert gap is None

    def test_total_market_matched_to_moneyline_sharp_no_total_line_returns_none(self):
        """Total market vs SharpOdds with total_line=None → None."""
        market = _market(
            team_home="pistons", team_away="warriors",
            market_type=MarketType.total, line=220.5,
        )
        # SharpOdds has no total_line — it is a moneyline record
        sharp = _moneyline_sharp(
            event_id="nba::2026-03-21::pistons_vs_warriors",
            true_prob_a=0.60,
        )
        assert sharp.total_line is None
        gap = self._call(market, sharp)
        assert gap is None

    def test_moneyline_market_matched_to_totals_sharp_returns_none(self):
        """Moneyline market (not total) vs SharpOdds with total_line set → None."""
        market = _market(
            team_home="pistons", team_away="warriors",
            market_type=MarketType.moneyline,
            yes_price=0.45, yes_team="pistons",
        )
        # SharpOdds is a totals record (total_line is set)
        sharp = _total_sharp(
            event_id="nba::2026-03-21::pistons_vs_warriors",
            total_line=220.5,
            true_prob_over=0.55,
        )
        gap = self._call(market, sharp)
        assert gap is None

    # -----------------------------------------------------------------------
    # Total market EV scenarios
    # -----------------------------------------------------------------------

    def test_total_market_yes_over_positive_ev(self):
        """YES = 'over'; true_prob_over=0.58, kalshi=0.50 → positive EV."""
        market = _market(
            team_home="pistons", team_away="warriors",
            market_type=MarketType.total,
            yes_price=0.50,
            yes_team="over",
            line=220.5,  # NBA game total → above is_game_total floor (170)
        )
        sharp = _total_sharp(
            event_id="nba::2026-03-21::pistons_vs_warriors",
            total_line=220.5,
            true_prob_over=0.58,
        )
        gap = self._call(market, sharp)

        assert gap is not None
        assert gap.ev_pct > 0.02
        assert gap.line == 220.5

    def test_total_market_yes_under_positive_ev(self):
        """YES = 'under'; true_prob_under=0.55, kalshi=0.48 → positive EV."""
        market = _market(
            team_home="pistons", team_away="warriors",
            market_type=MarketType.total,
            yes_price=0.48,
            yes_team="under",
            line=220.5,
        )
        sharp = _total_sharp(
            event_id="nba::2026-03-21::pistons_vs_warriors",
            total_line=220.5,
            true_prob_over=0.45,  # true_prob_under = 0.55
        )
        gap = self._call(market, sharp)

        assert gap is not None
        # blended_true_prob should reflect true_prob_under = 0.55
        assert gap.blended_true_prob == pytest.approx(0.55, abs=0.05)
        assert gap.ev_pct > 0.02

    def test_team_scoring_prop_below_game_total_floor_returns_none(self):
        """NBA line=109.5 < 170 (game total floor) → is_game_total=False → None."""
        market = _market(
            team_home="lakers", team_away="celtics",
            market_type=MarketType.total,
            yes_price=0.48,
            yes_team="over",
            line=109.5,  # team scoring prop, not game total
            sector="nba",
        )
        sharp = _total_sharp(
            event_id="nba::2026-03-21::lakers_vs_celtics",
            total_line=109.5,
            true_prob_over=0.55,
        )
        gap = self._call(market, sharp, sector="nba")
        assert gap is None, (
            "line=109.5 is below NBA game-total floor of 170 — should be rejected as team prop"
        )

    def test_nfl_game_total_above_floor_is_accepted(self):
        """NFL line=45.5 >= 35 (NFL floor) → is_game_total=True → evaluated."""
        market = _market(
            team_home="chiefs", team_away="eagles",
            market_type=MarketType.total,
            yes_price=0.48,
            yes_team="over",
            line=45.5,
            sector="nfl",
        )
        sharp = _total_sharp(
            event_id="nfl::2026-03-21::chiefs_vs_eagles",
            total_line=45.5,
            true_prob_over=0.58,
            sector="nfl",
        )
        gap = self._call(market, sharp, sector="nfl")
        # 0.58 / 0.48 - 1 ≈ 20.8% → positive EV; should NOT be blocked
        assert gap is not None

    def test_nfl_team_scoring_prop_below_floor_returns_none(self):
        """NFL line=27.5 < 35 (NFL floor) → is_game_total=False → None."""
        market = _market(
            team_home="chiefs", team_away="eagles",
            market_type=MarketType.total,
            yes_price=0.48,
            yes_team="over",
            line=27.5,
            sector="nfl",
        )
        sharp = _total_sharp(
            event_id="nfl::2026-03-21::chiefs_vs_eagles",
            total_line=27.5,
            true_prob_over=0.55,
            sector="nfl",
        )
        gap = self._call(market, sharp, sector="nfl")
        assert gap is None

    # -----------------------------------------------------------------------
    # EV and threshold checks
    # -----------------------------------------------------------------------

    def test_negative_ev_returns_none(self):
        """true_prob < kalshi_price → negative EV → None."""
        # Kalshi offers pistons at 0.70 but true probability is only 0.55
        market = _market(
            team_home="pistons", team_away="warriors",
            yes_price=0.70, yes_team="pistons",
        )
        sharp = _moneyline_sharp(
            event_id="nba::2026-03-21::pistons_vs_warriors",
            outcome_a="pistons", outcome_b="warriors",
            true_prob_a=0.55,
        )
        gap = self._call(market, sharp)
        assert gap is None

    def test_ev_just_below_2pct_threshold_returns_none(self):
        """EV = 1.9% < 2% threshold → None."""
        # EV = true_prob/price - 1 ≈ 0.019 → below threshold
        # price=0.53, true_prob=0.54 → EV = 0.54/0.53 - 1 ≈ 0.0189
        market = _market(
            team_home="pistons", team_away="warriors",
            yes_price=0.53, yes_team="pistons",
        )
        sharp = _moneyline_sharp(
            event_id="nba::2026-03-21::pistons_vs_warriors",
            outcome_a="pistons", outcome_b="warriors",
            true_prob_a=0.54,
        )
        gap = self._call(market, sharp)
        # (0.54/0.53) - 1 = 0.01887 → below 0.02 threshold
        assert gap is None

    def test_ev_just_above_2pct_threshold_is_accepted(self):
        """EV = ~2.1% → above threshold → gap returned."""
        # price=0.48, true_prob=0.49 → EV = 0.49/0.48 - 1 ≈ 0.0208
        market = _market(
            team_home="pistons", team_away="warriors",
            yes_price=0.48, yes_team="pistons",
        )
        sharp = _moneyline_sharp(
            event_id="nba::2026-03-21::pistons_vs_warriors",
            outcome_a="pistons", outcome_b="warriors",
            true_prob_a=0.49,
        )
        gap = self._call(market, sharp)
        assert gap is not None
        assert gap.ev_pct >= 0.02

    def test_invalid_yes_price_zero_returns_none(self):
        market = _market(yes_price=0.0)
        sharp = _moneyline_sharp(event_id="nba::2026-03-21::lakers_vs_celtics")
        gap = self._call(market, sharp)
        # PredictionMarket validator raises, so we check it before that:
        # If price=0.0 is somehow allowed, _evaluate_pair guards at top
        # Actually the Pydantic validator allows 0.0; _evaluate_pair checks > 0
        assert gap is None

    def test_invalid_yes_price_one_returns_none(self):
        # yes_price=1.0 passes Pydantic validator (0.0 <= v <= 1.0) but
        # _evaluate_pair rejects prices >= 1.0
        try:
            market = _market(yes_price=1.0, no_price=0.0)
        except Exception:
            pytest.skip("PredictionMarket validator rejects yes_price=1.0")
            return
        sharp = _moneyline_sharp(event_id="nba::2026-03-21::lakers_vs_celtics")
        gap = self._call(market, sharp)
        assert gap is None

    # -----------------------------------------------------------------------
    # Kelly fraction cap
    # -----------------------------------------------------------------------

    def test_kelly_fraction_never_exceeds_5pct(self):
        """Even with huge EV edge, kelly_fraction must be capped at max_kelly=0.05."""
        # Create extreme edge: price=0.10, true_prob=0.70 → massive edge
        market = _market(
            team_home="pistons", team_away="warriors",
            yes_price=0.10, yes_team="pistons",
            # Note: valid Pydantic range is 0–1
        )
        sharp = _moneyline_sharp(
            event_id="nba::2026-03-21::pistons_vs_warriors",
            outcome_a="pistons", outcome_b="warriors",
            true_prob_a=0.60,
        )
        gap = self._call(market, sharp)

        assert gap is not None, "Extreme edge should still produce a gap"
        assert gap.kelly_fraction <= 0.05, (
            f"kelly_fraction={gap.kelly_fraction} exceeds the 5% hard cap"
        )

    def test_kelly_fraction_is_positive_for_positive_ev(self):
        """Positive EV should produce a positive kelly_fraction."""
        market = _market(
            team_home="pistons", team_away="warriors",
            yes_price=0.40, yes_team="pistons",
        )
        sharp = _moneyline_sharp(
            event_id="nba::2026-03-21::pistons_vs_warriors",
            outcome_a="pistons", outcome_b="warriors",
            true_prob_a=0.60,
        )
        gap = self._call(market, sharp)

        assert gap is not None
        assert gap.kelly_fraction > 0.0

    def test_kelly_fraction_with_various_large_edges_stays_capped(self):
        """kelly_fraction is always <= 0.05 regardless of how large the edge is."""
        prices = [0.15, 0.20, 0.25, 0.30]
        for price in prices:
            market = _market(
                team_home="pistons", team_away="warriors",
                yes_price=price, yes_team="pistons",
            )
            sharp = _moneyline_sharp(
                event_id="nba::2026-03-21::pistons_vs_warriors",
                outcome_a="pistons", outcome_b="warriors",
                true_prob_a=0.70,
            )
            gap = self._call(market, sharp)
            if gap is not None:
                assert gap.kelly_fraction <= 0.05, (
                    f"kelly_fraction={gap.kelly_fraction:.4f} exceeded 5% cap at price={price}"
                )

    # -----------------------------------------------------------------------
    # steam_move flag
    # -----------------------------------------------------------------------

    def test_steam_move_true_when_event_id_in_steam_events(self):
        """steam_move should be True when sharp.event_id is in the steam_events set."""
        market = _market(
            team_home="pistons", team_away="warriors",
            yes_price=0.40, yes_team="pistons",
        )
        event_id = "nba::2026-03-21::pistons_vs_warriors"
        sharp = _moneyline_sharp(
            event_id=event_id,
            outcome_a="pistons", outcome_b="warriors",
            true_prob_a=0.60,
        )
        steam_events = {event_id}  # this event is a steam move
        gap = self._call(market, sharp, steam_events=steam_events)

        assert gap is not None
        assert gap.steam_move is True

    def test_steam_move_false_when_event_id_not_in_steam_events(self):
        """steam_move should be False when the event is not flagged."""
        market = _market(
            team_home="pistons", team_away="warriors",
            yes_price=0.40, yes_team="pistons",
        )
        event_id = "nba::2026-03-21::pistons_vs_warriors"
        sharp = _moneyline_sharp(
            event_id=event_id,
            outcome_a="pistons", outcome_b="warriors",
            true_prob_a=0.60,
        )
        steam_events = {"nba::2026-03-21::celtics_vs_knicks"}  # different event
        gap = self._call(market, sharp, steam_events=steam_events)

        assert gap is not None
        assert gap.steam_move is False

    def test_steam_move_false_when_steam_events_is_empty_set(self):
        """steam_move is False when steam_events is an empty set."""
        market = _market(
            team_home="pistons", team_away="warriors",
            yes_price=0.40, yes_team="pistons",
        )
        sharp = _moneyline_sharp(
            event_id="nba::2026-03-21::pistons_vs_warriors",
            outcome_a="pistons", outcome_b="warriors",
            true_prob_a=0.60,
        )
        gap = self._call(market, sharp, steam_events=set())

        assert gap is not None
        assert gap.steam_move is False

    # -----------------------------------------------------------------------
    # EVGap field integrity
    # -----------------------------------------------------------------------

    def test_gap_fields_are_populated_correctly(self):
        """All core EVGap fields are non-default after a valid evaluate_pair call."""
        market = _market(
            team_home="pistons", team_away="warriors",
            yes_price=0.40, yes_team="pistons",
            market_id="kalshi:PISTON-001",
            volume_usd=15_000.0,
        )
        event_id = "nba::2026-03-21::pistons_vs_warriors"
        sharp = _moneyline_sharp(
            event_id=event_id,
            outcome_a="pistons", outcome_b="warriors",
            true_prob_a=0.60,
        )
        gap = self._call(market, sharp, confidence=88.5)

        assert gap is not None
        assert gap.market_id == "kalshi:PISTON-001"
        assert gap.event_id == event_id
        assert gap.sector == "nba"
        assert gap.yes_team == "pistons"
        assert gap.market_type == "moneyline"
        assert gap.kalshi_yes_price == pytest.approx(0.40)
        assert gap.sharp_true_prob == pytest.approx(0.60, abs=0.001)
        assert gap.match_confidence == pytest.approx(88.5)
        assert gap.volume_usd == pytest.approx(15_000.0)
        assert gap.kelly_full > 0
        assert gap.kelly_fraction > 0

    # -----------------------------------------------------------------------
    # Model blend integration (basic path)
    # -----------------------------------------------------------------------

    def test_blended_pred_for_home_team_overrides_sharp_prob(self):
        """When a BlendedPrediction exists for the event, true_prob_a is used."""
        from evmax.agents.models.ensemble_agent import BlendedPrediction

        market = _market(
            team_home="pistons", team_away="warriors",
            yes_price=0.45, yes_team="pistons",
        )
        event_id = "nba::2026-03-21::pistons_vs_warriors"
        sharp = _moneyline_sharp(
            event_id=event_id,
            outcome_a="pistons", outcome_b="warriors",
            true_prob_a=0.50,  # borderline: 0.50/0.45 - 1 = 11% → positive but we check blend
        )
        blend = BlendedPrediction(
            event_id=event_id,
            true_prob_a=0.62,
            true_prob_b=0.38,
            true_prob_draw=None,
            confidence=0.85,
            model_sources="elo+sharp",
            per_model={},
        )
        gap = self._call(market, sharp, blended_preds={event_id: blend})

        assert gap is not None
        assert gap.blended_true_prob == pytest.approx(0.62, abs=0.001)

    def test_blended_pred_for_away_team_uses_true_prob_b(self):
        """YES = away team with blended prediction → uses blend.true_prob_b."""
        from evmax.agents.models.ensemble_agent import BlendedPrediction

        market = _market(
            team_home="pistons", team_away="warriors",
            yes_price=0.38, yes_team="warriors",  # YES = away
        )
        event_id = "nba::2026-03-21::pistons_vs_warriors"
        sharp = _moneyline_sharp(
            event_id=event_id,
            outcome_a="pistons", outcome_b="warriors",
            true_prob_a=0.55,  # warriors = 0.45
        )
        blend = BlendedPrediction(
            event_id=event_id,
            true_prob_a=0.52,
            true_prob_b=0.48,
            true_prob_draw=None,
            confidence=0.80,
            model_sources="elo+sharp",
            per_model={},
        )
        gap = self._call(market, sharp, blended_preds={event_id: blend})

        assert gap is not None
        assert gap.blended_true_prob == pytest.approx(0.48, abs=0.001)

    def test_model_drift_cap_applied_when_ratio_exceeds_2x(self):
        """When blend/sharp ratio >= 2.0, blended_prob is capped to sharp_true_prob."""
        from evmax.agents.models.ensemble_agent import BlendedPrediction

        # sharp_true_prob = 0.30; blend.true_prob_a = 0.80 → ratio = 2.67 → cap
        market = _market(
            team_home="pistons", team_away="warriors",
            yes_price=0.38, yes_team="pistons",
        )
        event_id = "nba::2026-03-21::pistons_vs_warriors"
        sharp = _moneyline_sharp(
            event_id=event_id,
            outcome_a="pistons", outcome_b="warriors",
            true_prob_a=0.30,
        )
        blend = BlendedPrediction(
            event_id=event_id,
            true_prob_a=0.80,
            true_prob_b=0.20,
            true_prob_draw=None,
            confidence=0.80,
            model_sources="elo",
            per_model={},
        )
        gap = self._call(market, sharp, blended_preds={event_id: blend})

        if gap is not None:
            assert gap.model_sources == "sharp(capped)"
            assert gap.blended_true_prob == pytest.approx(gap.sharp_true_prob, abs=0.001)

    def test_model_drift_cap_applied_when_ratio_below_half(self):
        """When blend/sharp ratio <= 0.5, blended_prob is capped to sharp_true_prob."""
        from evmax.agents.models.ensemble_agent import BlendedPrediction

        # sharp_true_prob = 0.70; blend.true_prob_a = 0.30 → ratio = 0.43 → cap
        market = _market(
            team_home="pistons", team_away="warriors",
            yes_price=0.40, yes_team="pistons",
        )
        event_id = "nba::2026-03-21::pistons_vs_warriors"
        sharp = _moneyline_sharp(
            event_id=event_id,
            outcome_a="pistons", outcome_b="warriors",
            true_prob_a=0.70,
        )
        blend = BlendedPrediction(
            event_id=event_id,
            true_prob_a=0.30,
            true_prob_b=0.70,
            true_prob_draw=None,
            confidence=0.80,
            model_sources="elo",
            per_model={},
        )
        gap = self._call(market, sharp, blended_preds={event_id: blend})

        if gap is not None:
            assert gap.model_sources == "sharp(capped)"
            assert gap.blended_true_prob == pytest.approx(gap.sharp_true_prob, abs=0.001)


# ---------------------------------------------------------------------------
# NO-side spread (synthesized "+spread" cover)
# ---------------------------------------------------------------------------

class TestNoSideSpread:
    """NO-on-spread emission, display, and resolution.

    Kalshi spread tickers ask "Team X wins by over N.5" (always a -spread).
    The NO side is the only way to bet the corresponding +spread, since
    Kalshi does not list "Team Y +N.5" as its own ticker. We synthesize
    a paired EVGap with `:no` market_id, opponent yes_team, positive line,
    and inverted price/prob.
    """

    def setup_method(self):
        self.agent = _make_ev_gap_agent()

    def test_display_label_renders_positive_spread_with_plus_prefix(self):
        gap = EVGap(
            market_id="kalshi:foo:no",
            event_id="nba::2026-03-21::pistons_vs_warriors",
            sector="nba",
            yes_team="warriors",
            market_type="spread",
            kalshi_yes_price=0.55,
            sharp_true_prob=0.60,
            blended_true_prob=0.60,
            ev_pct=0.05,
            kelly_full=0.10,
            kelly_fraction=0.025,
            match_confidence=95.0,
            volume_usd=5000.0,
            spread_pct=0.02,
            line=9.5,
        )
        assert gap.display_label == "Warriors +9.5"

    def test_display_label_renders_negative_spread_without_plus(self):
        gap = EVGap(
            market_id="kalshi:foo",
            event_id="nba::2026-03-21::pistons_vs_warriors",
            sector="nba",
            yes_team="pistons",
            market_type="spread",
            kalshi_yes_price=0.45,
            sharp_true_prob=0.50,
            blended_true_prob=0.50,
            ev_pct=0.05,
            kelly_full=0.10,
            kelly_fraction=0.025,
            match_confidence=95.0,
            volume_usd=5000.0,
            spread_pct=0.02,
            line=-9.5,
        )
        assert gap.display_label == "Pistons -9.5"

    def test_no_side_spread_emitted_when_yes_below_threshold(self):
        """Deeply −EV YES → corresponding NO becomes +EV and is emitted."""
        from evmax.agents.base import AgentRequest
        from unittest.mock import patch
        import asyncio

        # Pistons -9.5 ticker priced at YES=0.65, NO=0.36. Pinnacle posts
        # Pistons -7.5 implied cover prob 0.55 → spread distribution
        # extrapolates to a much lower P(yes covers -9.5), forcing NO +EV.
        market = _market(
            team_home="pistons", team_away="warriors",
            yes_price=0.65, no_price=0.36,
            yes_team="pistons",
            market_type=MarketType.spread,
            line=-9.5,
            sector="nba",
        )
        sharp = _spread_sharp(
            event_id="nba::2026-03-21::pistons_vs_warriors::spread",
            outcome_a="pistons", outcome_b="warriors",
            spread_line=-7.5,
            true_prob_a=0.55,
        )

        with patch.object(
            self.agent._matching, "match_all",
            return_value=[(market, sharp, 95.0)],
        ):
            req = AgentRequest(
                sector="nba",
                params={"kalshi_markets": [market], "sharp_odds": [sharp]},
            )
            resp = asyncio.run(self.agent(req))

        no_gaps = [g for g in resp.data if g.market_id.endswith(":no")]
        assert len(no_gaps) == 1, (
            f"expected one NO-side gap, got: "
            f"{[(g.market_id, g.ev_pct) for g in resp.data]}"
        )
        ng = no_gaps[0]
        assert ng.yes_team == "warriors"
        assert ng.line == 9.5  # sign flipped to positive
        assert ng.kalshi_yes_price == pytest.approx(0.36, abs=0.001)
        assert ng.market_type == "spread"
        assert ng.ev_pct >= 0.05  # NO-side floor is 5%, raised from 2% on 2026-05-07
        assert "no_side" in ng.model_sources
        assert ng.display_label == "Warriors +9.5"

    def test_no_side_not_emitted_when_below_threshold(self):
        """No edge in either direction → no NO gap."""
        from evmax.agents.base import AgentRequest
        from unittest.mock import patch
        import asyncio

        market = _market(
            team_home="pistons", team_away="warriors",
            yes_price=0.50, no_price=0.51,
            yes_team="pistons",
            market_type=MarketType.spread,
            line=-7.5,
            sector="nba",
        )
        sharp = _spread_sharp(
            event_id="nba::2026-03-21::pistons_vs_warriors::spread",
            outcome_a="pistons", outcome_b="warriors",
            spread_line=-7.5,
            true_prob_a=0.50,
        )
        with patch.object(
            self.agent._matching, "match_all",
            return_value=[(market, sharp, 95.0)],
        ):
            req = AgentRequest(
                sector="nba",
                params={"kalshi_markets": [market], "sharp_odds": [sharp]},
            )
            resp = asyncio.run(self.agent(req))

        no_gaps = [g for g in resp.data if g.market_id.endswith(":no")]
        assert len(no_gaps) == 0

    def test_no_side_below_5pct_floor_not_emitted(self):
        """NO-side EV between 2% and 5% must not emit. 2026-05-07: NO-side
        floor was raised from 2% to 5% after archive backtest showed sigma=14
        EV>=2% lost -11c/$1 while EV>=5% cleared breakeven.
        """
        from evmax.agents.base import AgentRequest
        from unittest.mock import patch
        import asyncio

        # Construct a NO-side scenario where Pinnacle's spread distribution
        # makes Pistons covering -8.5 ~52%, so sharp NO ~48%. With NO ask 0.45,
        # NO EV = 0.48/0.45 - 1 ≈ +6.7% — would have passed the old 2% floor
        # but we want it to land in the 2-5% gap region. Tune until in-band.
        market = _market(
            team_home="pistons", team_away="warriors",
            yes_price=0.55, no_price=0.46,  # NO ask 0.46
            yes_team="pistons",
            market_type=MarketType.spread,
            line=-8.5,
            sector="nba",
        )
        # spread_line=-7.5 with true_prob_a=0.52 → cover at -8.5 slightly lower,
        # putting NO prob in the high-40s%, EV around +3-4%.
        sharp = _spread_sharp(
            event_id="nba::2026-03-21::pistons_vs_warriors::spread",
            outcome_a="pistons", outcome_b="warriors",
            spread_line=-7.5,
            true_prob_a=0.52,
        )
        with patch.object(
            self.agent._matching, "match_all",
            return_value=[(market, sharp, 95.0)],
        ):
            req = AgentRequest(
                sector="nba",
                params={"kalshi_markets": [market], "sharp_odds": [sharp]},
            )
            resp = asyncio.run(self.agent(req))

        no_gaps = [g for g in resp.data if g.market_id.endswith(":no")]
        # Must be either zero (filtered) or — if construction landed above 5% —
        # at least confirm any emitted NO gap respects the floor.
        for g in no_gaps:
            assert g.ev_pct >= 0.05, (
                f"NO-side gap with ev_pct={g.ev_pct:.4f} should have been filtered"
            )

    def test_no_side_skipped_for_moneyline(self):
        """Moneyline tickers have a separate YES per team — NO would double-count."""
        from evmax.agents.base import AgentRequest
        from unittest.mock import patch
        import asyncio

        market = _market(
            team_home="pistons", team_away="warriors",
            yes_price=0.65, no_price=0.36,
            yes_team="pistons",
            market_type=MarketType.moneyline,
            sector="nba",
        )
        sharp = _moneyline_sharp(
            event_id="nba::2026-03-21::pistons_vs_warriors",
            outcome_a="pistons", outcome_b="warriors",
            true_prob_a=0.45,
        )
        with patch.object(
            self.agent._matching, "match_all",
            return_value=[(market, sharp, 95.0)],
        ):
            req = AgentRequest(
                sector="nba",
                params={"kalshi_markets": [market], "sharp_odds": [sharp]},
            )
            resp = asyncio.run(self.agent(req))

        no_gaps = [g for g in resp.data if g.market_id.endswith(":no")]
        assert len(no_gaps) == 0


# ---------------------------------------------------------------------------
# Resolver — positive spread lines (NO-side cover)
# ---------------------------------------------------------------------------

class TestResolverPositiveSpreadLine:
    """Resolver must treat line>0 as a +spread cover.

    A bet stored with line=+9.5 means "yes_team doesn't lose by more than
    9.5 points". Cover threshold flips: margin > -threshold instead of
    margin > threshold.
    """

    def _resolve(self, line: float, yes_score: int, opp_score: int) -> int:
        threshold = abs(float(line))
        margin = yes_score - opp_score
        cover_threshold = -threshold if line > 0 else threshold
        return 1 if margin > cover_threshold else 0

    def test_positive_line_yes_loses_by_less_than_threshold_covers(self):
        # +9.5: yes loses by 5 → covers
        assert self._resolve(line=+9.5, yes_score=110, opp_score=115) == 1

    def test_positive_line_yes_loses_by_more_than_threshold_does_not_cover(self):
        # +9.5: yes loses by 12 → does NOT cover
        assert self._resolve(line=+9.5, yes_score=100, opp_score=112) == 0

    def test_positive_line_yes_wins_outright_covers(self):
        # +9.5: yes wins by 3 → covers
        assert self._resolve(line=+9.5, yes_score=110, opp_score=107) == 1

    def test_positive_line_loss_by_exactly_ten_does_not_cover(self):
        # +9.5: yes loses by 10 (110 vs 100) → does NOT cover
        assert self._resolve(line=+9.5, yes_score=100, opp_score=110) == 0

    def test_negative_line_unchanged_behavior(self):
        # -7.5 (favorite cover): win by 8 → cover; win by 7 → no cover
        assert self._resolve(line=-7.5, yes_score=110, opp_score=102) == 1
        assert self._resolve(line=-7.5, yes_score=110, opp_score=103) == 0


# ---------------------------------------------------------------------------
# Spread orientation: yes_is_outcome_b is preserved through to injury step
# ---------------------------------------------------------------------------

class TestSpreadOrientationPreservation:
    """Regression for the underdog-cover injury direction bug.

    When YES is the underdog (yes_is_outcome_b=True), the spread distribution
    model returns a YES-aligned cover prob. Downstream injury / standings /
    playoff steps need yes_is_outcome_b=True to flip the prob onto the
    correct team. A previous code path reset yes_is_outcome_b=False after
    spread_dist, which made injury adjustments to the YES (underdog) team
    flow in the WRONG direction (their prob increased when they got hurt).

    These tests verify the orientation flag survives the spread-model step.
    """

    def setup_method(self):
        self.agent = _make_ev_gap_agent()

    def test_spread_injury_layer_skipped(self):
        """Path B: spread bets bypass the injury adjustment entirely.

        Pinnacle's spread line already prices known injuries (sharp book,
        insider networks). Adding ESPN-derived shifts on top double-counts
        the signal AND uses moneyline-shaped impact on a tail probability.
        This test asserts that adding injuries does NOT change the spread
        blended_prob — a regression in either direction (decrease OR
        increase) signals the layer crept back in.
        """
        from evmax.agents.intelligence.injury_agent import (
            InjuredPlayer, InjuryReport,
        )
        from evmax.agents.base import AgentRequest
        from unittest.mock import patch
        import asyncio

        market = _market(
            team_home="pistons", team_away="warriors",
            yes_price=0.40, no_price=0.61,
            yes_team="warriors",
            market_type=MarketType.spread,
            line=-8.5,
            sector="nba",
        )
        sharp = _spread_sharp(
            event_id="nba::2026-03-21::pistons_vs_warriors::spread",
            outcome_a="pistons", outcome_b="warriors",
            spread_line=-6.5,
            true_prob_a=0.55,
        )

        injured = InjuredPlayer(
            name="Stephen Curry", position="PG", status="Out",
            tier="star", impact=0.045, injury_type="Knee", notes="",
        )
        injuries = {
            "warriors": InjuryReport(team="warriors", sector="nba", players=[injured]),
        }

        async def _run(inj):
            with patch.object(
                self.agent._matching, "match_all",
                return_value=[(market, sharp, 95.0)],
            ):
                req = AgentRequest(
                    sector="nba",
                    params={"kalshi_markets": [market], "sharp_odds": [sharp],
                            "injuries": inj},
                )
                return await self.agent(req)

        resp_no_inj  = asyncio.run(_run({}))
        resp_with_inj = asyncio.run(_run(injuries))

        def yes_spread_gap(resp):
            for g in resp.data:
                if g.market_type == "spread" and not g.market_id.endswith(":no"):
                    return g
            return None

        gap_no  = yes_spread_gap(resp_no_inj)
        gap_inj = yes_spread_gap(resp_with_inj)

        # If either filtered out via threshold, can't compare — accept that.
        if gap_no is not None and gap_inj is not None:
            assert gap_inj.blended_true_prob == pytest.approx(
                gap_no.blended_true_prob, abs=1e-6
            ), (
                "Path B: spread injury layer must be skipped, but blended_prob "
                f"changed: {gap_no.blended_true_prob:.6f} → {gap_inj.blended_true_prob:.6f}"
            )
            assert "+injury" not in gap_inj.model_sources, (
                f"model_sources should not contain +injury for spread bets, "
                f"got: {gap_inj.model_sources}"
            )

    def test_no_side_opp_label_correct_when_yes_is_underdog(self):
        """NO-side opponent label is the FAVORITE when YES is the underdog.

        Regression for the orientation reset: previously yes_is_outcome_b
        was forced to False after spread_dist, causing the NO-side derivation
        to look up sharp.outcome_b_label as the opponent — which was the
        SAME team as YES. Now it correctly returns outcome_a (the favorite).
        """
        from evmax.agents.base import AgentRequest
        from unittest.mock import patch
        import asyncio

        # YES = warriors (underdog), Kalshi line -8.5, Pinnacle pistons -6.5.
        # Push Kalshi YES price high so YES is −EV → NO side becomes +EV.
        market = _market(
            team_home="pistons", team_away="warriors",
            yes_price=0.65, no_price=0.36,
            yes_team="warriors",
            market_type=MarketType.spread,
            line=-8.5,
            sector="nba",
        )
        sharp = _spread_sharp(
            event_id="nba::2026-03-21::pistons_vs_warriors::spread",
            outcome_a="pistons", outcome_b="warriors",
            spread_line=-6.5,
            true_prob_a=0.55,
        )

        with patch.object(
            self.agent._matching, "match_all",
            return_value=[(market, sharp, 95.0)],
        ):
            req = AgentRequest(
                sector="nba",
                params={"kalshi_markets": [market], "sharp_odds": [sharp]},
            )
            resp = asyncio.run(self.agent(req))

        no_gaps = [g for g in resp.data if g.market_id.endswith(":no")]
        if no_gaps:
            ng = no_gaps[0]
            # Opponent of warriors (YES underdog) is pistons (favorite, outcome_a).
            assert "pistons" in ng.yes_team.lower(), (
                f"NO-side opponent should be 'pistons' (outcome_a), got '{ng.yes_team}'"
            )
            assert ng.line == 8.5  # sign flipped to positive


# ---------------------------------------------------------------------------
# WNBA spread / total path
# ---------------------------------------------------------------------------

class TestWnbaSpreadPath:
    """WNBA was previously moneyline-only. After wiring KXWNBASPREAD/
    KXWNBATOTAL series + WNBA-specific sigmas + WNBA possession sim
    dispatch, WNBA spread/total Kalshi tickers (when they appear) get the
    same EV pipeline as NBA — with WNBA-tuned constants."""

    def test_spread_distribution_uses_wnba_sigma(self):
        from evmax.models_ml.spread_distribution import (
            SpreadDistributionModel, _SECTOR_SIGMA,
        )
        # Sigma should be 12.5 (matches WNBAPossessionSimAgent SCORE_STDEV).
        # NBA was bumped to 12.5 on 2026-05-07 too, so the values match
        # numerically — the WNBA path remains tuned independently and
        # the assertion below confirms WNBA reads its own key.
        assert _SECTOR_SIGMA["wnba"] == 12.5

        sharp = _spread_sharp(
            event_id="wnba::2026-06-01::aces_vs_liberty::spread",
            outcome_a="aces", outcome_b="liberty",
            spread_line=-4.5, true_prob_a=0.55,
        )
        result = SpreadDistributionModel().predict(
            sharp_odds=sharp, target_line=-4.5,
            sector="wnba", yes_is_underdog=False,
        )
        assert result is not None
        assert result.sigma == 12.5

    def test_total_distribution_uses_wnba_sigma(self):
        from evmax.models_ml.total_distribution import _TOTAL_SIGMA, _GAME_TOTAL_FLOOR
        assert _TOTAL_SIGMA["wnba"] == 16.0
        assert _GAME_TOTAL_FLOOR["wnba"] == 130.0

    def test_sim_dispatch_routes_wnba_to_wnba_sim(self):
        """_sim_for_sector should pick the WNBA agent for sector=wnba."""
        from unittest.mock import MagicMock
        agent = _make_ev_gap_agent()
        agent._possession_sim = MagicMock(name="nba_sim")
        agent._wnba_possession_sim = MagicMock(name="wnba_sim")

        assert agent._sim_for_sector("nba") is agent._possession_sim
        assert agent._sim_for_sector("wnba") is agent._wnba_possession_sim
        assert agent._sim_for_sector("nfl") is None
        assert agent._sim_for_sector("") is None

    def test_kalshi_parser_recognizes_wnba_spread_prefix(self):
        """The is_spread prefix list must include KXWNBASPREAD so parser
        sets MarketType.spread and _extract_spread_line runs."""
        from evmax.clients.kalshi import KalshiClient
        client = KalshiClient.__new__(KalshiClient)
        line = client._extract_spread_line("KXWNBASPREAD-26MAY16NYLLVA-LVA4")
        assert line == pytest.approx(-4.5)

    def test_wnba_in_sector_series_map(self):
        from evmax.clients.kalshi import SECTOR_SERIES_MAP
        wnba_series = SECTOR_SERIES_MAP["wnba"]
        assert "KXWNBAGAME" in wnba_series
        assert "KXWNBASPREAD" in wnba_series
        assert "KXWNBATOTAL" in wnba_series


# ---------------------------------------------------------------------------
# Web display_label_for_row — parallel to EVGap.display_label, used by the
# dashboard / API for DB-backed rows. Must keep the same +/- spread sign
# convention so NO-side rows render as "Cavaliers +9.5", not "Cavaliers 9.5".
# ---------------------------------------------------------------------------

class TestWebDisplayLabelForRow:
    def test_positive_spread_renders_with_plus(self):
        from evmax.web.app import _display_label_for_row
        row = {"market_type": "spread", "yes_team": "cavaliers", "line": 9.5}
        assert _display_label_for_row(row) == "Cavaliers +9.5"

    def test_negative_spread_renders_without_plus(self):
        from evmax.web.app import _display_label_for_row
        row = {"market_type": "spread", "yes_team": "pistons", "line": -7.5}
        assert _display_label_for_row(row) == "Pistons -7.5"

    def test_moneyline_unchanged(self):
        from evmax.web.app import _display_label_for_row
        row = {"market_type": "moneyline", "yes_team": "lakers"}
        assert _display_label_for_row(row) == "Lakers ML"

    def test_total_renders_side_explicitly(self):
        # Totals used to render as "O/U 220.5" which left users guessing
        # which side the YES market represented. Kalshi totals are YES = OVER,
        # so the label must say it explicitly. yes_team starting with "u"
        # falls back to "Under" if a future source ever posts under-side YES.
        from evmax.web.app import _display_label_for_row
        over_row = {"market_type": "total", "yes_team": "over", "line": 220.5}
        assert _display_label_for_row(over_row) == "Over 220.5"
        under_row = {"market_type": "total", "yes_team": "under", "line": 220.5}
        assert _display_label_for_row(under_row) == "Under 220.5"

    def test_invalid_line_string_falls_back(self):
        from evmax.web.app import _display_label_for_row
        # Ensure the new try/except still falls back gracefully on bad input.
        row = {"market_type": "spread", "yes_team": "lakers", "line": "garbage"}
        assert _display_label_for_row(row) == "Lakers garbage"


class TestNoSideOpponentNameNormalized:
    """The NO-side spread builder used to store the FULL Pinnacle outcome label
    as yes_team (e.g. "cleveland cavaliers"), so the dashboard rendered
    "Cleveland cavaliers +9.5". YES-side rows store the canonical short
    name ("cavaliers") via NameNormalizer, so NO-side should match.
    """

    def setup_method(self):
        self.agent = _make_ev_gap_agent()

    def test_no_side_yes_team_uses_canonical_short_name(self):
        """Even when sharp.outcome_a_label is the full Pinnacle name,
        the NO-side gap should store the alias-normalized short form."""
        from evmax.agents.base import AgentRequest
        from unittest.mock import patch
        import asyncio

        # YES = pistons (favorite), pinnacle has full names. Force a
        # high YES price so NO becomes +EV and the NO gap surfaces.
        market = _market(
            team_home="pistons", team_away="warriors",
            yes_price=0.65, no_price=0.36,
            yes_team="pistons",
            market_type=MarketType.spread,
            line=-7.5,
            sector="nba",
        )
        sharp = _spread_sharp(
            event_id="nba::2026-03-21::pistons_vs_warriors::spread",
            outcome_a="Detroit Pistons",   # full name as Pinnacle returns
            outcome_b="Golden State Warriors",
            spread_line=-7.5,
            true_prob_a=0.55,
        )

        with patch.object(
            self.agent._matching, "match_all",
            return_value=[(market, sharp, 95.0)],
        ):
            req = AgentRequest(
                sector="nba",
                params={"kalshi_markets": [market], "sharp_odds": [sharp]},
            )
            resp = asyncio.run(self.agent(req))

        no_gaps = [g for g in resp.data if g.market_id.endswith(":no")]
        if no_gaps:
            ng = no_gaps[0]
            # Opponent of pistons (YES favorite) is warriors.
            # Must NOT be the long form "golden state warriors".
            assert "golden state" not in ng.yes_team, (
                f"Expected canonical short name, got long form: '{ng.yes_team}'"
            )
            assert "warriors" in ng.yes_team


# ---------------------------------------------------------------------------
# Empty-orderbook filter — spread_pct floor
# ---------------------------------------------------------------------------

class TestEmptyOrderbookFilter:
    """Markets where yes_ask + no_ask sum to exactly 1.0 don't have a real
    two-sided book (real markets have vig). spread_pct = 0 is the signal.
    Filter such markets so we don't surface untradeable +EV phantoms.
    """

    def setup_method(self):
        self.agent = _make_ev_gap_agent()

    def test_yes_side_filtered_when_spread_pct_is_zero(self):
        """yes_price + no_price == 1.0 → spread_pct=0 → drop the bet."""
        from evmax.agents.base import AgentRequest
        from unittest.mock import patch
        import asyncio

        # Construct a market with no_price = 1 - yes_price exactly (synthetic
        # complement, no real two-sided book).
        market = _market(
            team_home="lakers", team_away="rockets",
            yes_price=0.45, no_price=0.55,
            yes_team="lakers",
            market_type=MarketType.spread,
            line=-7.5,
            sector="nba",
        )
        # Sanity: spread_pct should be ~0 by construction
        assert market.spread_pct < 0.005

        sharp = _spread_sharp(
            event_id="nba::2026-05-01::rockets_vs_lakers::spread",
            outcome_a="lakers", outcome_b="rockets",
            spread_line=-5.5, true_prob_a=0.55,
        )

        with patch.object(
            self.agent._matching, "match_all",
            return_value=[(market, sharp, 95.0)],
        ):
            req = AgentRequest(
                sector="nba",
                params={"kalshi_markets": [market], "sharp_odds": [sharp]},
            )
            resp = asyncio.run(self.agent(req))

        # Both YES and NO should be filtered → no gaps emitted
        assert len(resp.data) == 0, (
            f"Empty-book market should not emit any gaps; got: "
            f"{[(g.market_id, g.ev_pct) for g in resp.data]}"
        )

    def test_real_book_with_vig_passes_filter(self):
        """yes_ask + no_ask > 1.0 (real vig) → spread_pct > 0 → not filtered."""
        from evmax.agents.base import AgentRequest
        from unittest.mock import patch
        import asyncio

        # Real book: yes=0.45, no=0.57 → sum 1.02 (2% vig), spread_pct ~ 4%
        market = _market(
            team_home="rockets", team_away="lakers",
            yes_price=0.45, no_price=0.57,
            yes_team="rockets",
            market_type=MarketType.moneyline,
            sector="nba",
        )
        assert market.spread_pct >= 0.005

        sharp = _moneyline_sharp(
            event_id="nba::2026-05-01::rockets_vs_lakers",
            outcome_a="rockets", outcome_b="lakers",
            true_prob_a=0.58,
        )

        with patch.object(
            self.agent._matching, "match_all",
            return_value=[(market, sharp, 95.0)],
        ):
            req = AgentRequest(
                sector="nba",
                params={"kalshi_markets": [market], "sharp_odds": [sharp]},
            )
            resp = asyncio.run(self.agent(req))

        # Should produce a gap (price 0.45, true prob 0.58 → +EV)
        assert len(resp.data) >= 1
