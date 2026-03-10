"""Tests for the market matching engine."""

import pytest
from datetime import datetime, timezone

from evmax.matching.engine import MatchingEngine
from evmax.matching.fuzzy import fuzzy_match, fuzzy_match_event_keys
from evmax.matching.normalizer import NameNormalizer
from evmax.models.market import MarketSource, MarketType, PredictionMarket
from evmax.models.odds import SharpBook, SharpOdds


def make_market(
    team_home: str,
    team_away: str,
    sector: str = "nfl",
    event_date: datetime = None,
) -> PredictionMarket:
    return PredictionMarket(
        id=f"kalshi:TEST-{team_home[:3]}",
        source=MarketSource.kalshi,
        sector=sector,
        yes_price=0.50,
        no_price=0.50,
        team_home=team_home,
        team_away=team_away,
        event_date=event_date or datetime(2026, 2, 22, tzinfo=timezone.utc),
    )


def make_sharp(
    event_id: str,
    sector: str = "nfl",
) -> SharpOdds:
    return SharpOdds(
        event_id=event_id,
        book=SharpBook.pinnacle,
        sector=sector,
        outcome_a_label="Team A",
        outcome_b_label="Team B",
        outcome_a_decimal=1.91,
        outcome_b_decimal=2.05,
        true_prob_a=0.52,
        true_prob_b=0.48,
        margin=0.03,
    )


class TestNameNormalizer:
    def test_nfl_alias_resolution(self):
        """NFL team aliases should resolve correctly."""
        norm = NameNormalizer("nfl")
        assert norm.normalize("Kansas City Chiefs") == "chiefs"
        assert norm.normalize("KC") == "chiefs"

    def test_unknown_sector_no_crash(self):
        """Unknown sector should not crash, just return cleaned name."""
        norm = NameNormalizer("unknown_sector")
        result = norm.normalize("Team Name")
        assert result  # Should return something

    def test_cleanup(self):
        """Should strip noise words and lowercase."""
        norm = NameNormalizer("nfl")
        result = norm.normalize("  SOME FC TEAM  ")
        assert result == result.lower()
        assert "fc" not in result


class TestFuzzyMatch:
    def test_exact_match(self):
        """Exact match should return 100."""
        result = fuzzy_match("chiefs", ["chiefs", "49ers", "eagles"])
        assert result is not None
        match, score = result
        assert match == "chiefs"
        assert score == 100.0

    def test_close_match(self):
        """Close fuzzy match should succeed above threshold."""
        result = fuzzy_match("chefs", ["chiefs", "49ers", "eagles"], threshold=70)
        assert result is not None

    def test_no_match_below_threshold(self):
        """No match below threshold should return None."""
        result = fuzzy_match("random", ["chiefs", "49ers", "eagles"], threshold=90)
        assert result is None

    def test_empty_query(self):
        result = fuzzy_match("", ["chiefs"])
        assert result is None

    def test_empty_candidates(self):
        result = fuzzy_match("chiefs", [])
        assert result is None


class TestMatchingEngine:
    def test_exact_match(self):
        """Markets with matching canonical keys should be found."""
        engine = MatchingEngine()
        market = make_market("chiefs", "eagles", sector="nfl")

        # Build the key manually
        key = engine.build_market_key(market)
        assert key is not None

        sharp = make_sharp(key, sector="nfl")
        result = engine.match(market, [sharp])

        assert result is not None
        matched_odds, confidence = result
        assert confidence == 100.0

    def test_no_date_returns_none(self):
        """Market without event_date cannot build key → no match."""
        engine = MatchingEngine()
        market = PredictionMarket(
            id="kalshi:TEST",
            source=MarketSource.kalshi,
            sector="nfl",
            yes_price=0.50,
            no_price=0.50,
            team_home="chiefs",
            team_away="eagles",
            event_date=None,  # No date
        )
        result = engine.match(market, [make_sharp("nfl::2026-02-22::chiefs_vs_eagles")])
        assert result is None

    def test_no_teams_returns_none(self):
        """Market without team names cannot match."""
        engine = MatchingEngine()
        market = PredictionMarket(
            id="kalshi:TEST",
            source=MarketSource.kalshi,
            sector="nfl",
            yes_price=0.50,
            no_price=0.50,
            event_date=datetime(2026, 2, 22, tzinfo=timezone.utc),
        )
        result = engine.match(market, [make_sharp("nfl::2026-02-22::chiefs_vs_eagles")])
        assert result is None

    def test_empty_sharp_returns_none(self):
        """No sharp odds → no match."""
        engine = MatchingEngine()
        market = make_market("chiefs", "eagles")
        result = engine.match(market, [])
        assert result is None

    def test_fuzzy_home_away_swap(self):
        """Fuzzy match should handle reversed home/away team order (e.g. UCL games)."""
        result = fuzzy_match_event_keys(
            "soccer::2026-02-25::monaco_vs_psg",
            ["soccer::2026-02-25::psg_vs_monaco"],
            threshold=88,
        )
        assert result is not None, "Reversed home/away teams should fuzzy-match"
        matched_key, score = result
        assert matched_key == "soccer::2026-02-25::psg_vs_monaco"
        assert score >= 88

    def test_fuzzy_ucl_brugge_atletico(self):
        """UCL game with reversed home/away should match at score 100."""
        result = fuzzy_match_event_keys(
            "soccer::2026-02-24::brugge_vs_atletico",
            ["soccer::2026-02-24::atletico_vs_brugge"],
            threshold=88,
        )
        assert result is not None
        _, score = result
        assert score == 100.0

    def test_match_all(self):
        """match_all returns results for all matchable markets."""
        engine = MatchingEngine()
        markets = [
            make_market("chiefs", "eagles"),
            make_market("bills", "jets"),
        ]

        key1 = engine.build_market_key(markets[0])
        key2 = engine.build_market_key(markets[1])

        sharp_odds = [
            make_sharp(key1),
            make_sharp(key2),
        ]

        results = engine.match_all(markets, sharp_odds)
        assert len(results) == 2
