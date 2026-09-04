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

    def test_mls_full_name_aliases(self):
        """MLS clubs added 2026-07-09 resolve to the same canonicals the
        Kalshi series code map (kalshi.py) produces, so cross-venue and
        Kalshi↔Pinnacle matching agree on names."""
        norm = NameNormalizer("soccer")
        assert norm.normalize("Vancouver Whitecaps FC") == "vancouver"
        assert norm.normalize("St. Louis City") == "st louis"
        assert norm.normalize("Saint Louis") == "st louis"
        assert norm.normalize("San Jose Earthquakes") == "san jose"
        assert norm.normalize("Orlando City") == "orlando"
        assert norm.normalize("D.C. United") == "dc united"
        # The flat Serie A code alias is untouched — Toronto FC only resolves
        # correctly via the KXMLSGAME series-scoped map, not this namespace.
        assert norm.normalize("tor") == "torino"
        assert norm.normalize("Toronto FC") == "toronto"

    def test_mlb_athletics_diamondbacks_short_codes(self):
        """Kalshi's current MLB tickers use 'ATH' for the Athletics (post-
        Oakland rebrand) and 'AZ' for the Diamondbacks — neither short code
        had an alias entry, so every Athletics/Diamondbacks moneyline/spread/
        total market silently match_failed against Pinnacle every day while
        the legacy 'oak'/'ari' codes (and Polymarket US's full-name feed)
        kept working, masking the gap."""
        norm = NameNormalizer("baseball")
        assert norm.normalize("ath") == "athletics"
        assert norm.normalize("az") == "diamondbacks"
        # Legacy codes still resolve (ESPN / older Kalshi feeds).
        assert norm.normalize("oak") == "athletics"
        assert norm.normalize("ari") == "diamondbacks"

    def test_wnba_portland_fire_pdx_alias(self):
        """Kalshi's WNBA moneyline/spread tickers encode Portland Fire as
        PDX (airport code), not POR — moneyline/spread markets parse team
        codes straight from the ticker (unlike totals, which parse full
        names from the title), so an unresolved 'pdx' silently broke
        Kalshi↔Pinnacle matching for that market type while totals and
        Polymarket US (full-name feed) matched fine."""
        norm = NameNormalizer("wnba")
        assert norm.normalize("pdx") == "fire"
        assert norm.normalize("por") == "fire"
        assert norm.normalize("Portland Fire") == "fire"


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

    def test_build_market_key_uses_et_game_day_for_us_sectors(self):
        """PolymarketUS supplies the precise UTC first-pitch time (unlike
        Kalshi, whose ticker date is anchored to noon UTC as a workaround).
        A night game like an 8:05pm ET start carries a UTC timestamp that
        has already rolled to the next calendar day (2026-07-21T00:05:00Z).
        build_market_key used to strftime that UTC datetime directly,
        producing the wrong (next-day) date and silently breaking matching
        against Pinnacle's ET-anchored key for every US-sector night game —
        this is exactly the off-by-one time_util.kalshi_game_day exists to
        prevent, but build_market_key never called it."""
        engine = MatchingEngine()
        market = make_market(
            "tex",
            "cws",
            sector="baseball",
            event_date=datetime(2026, 7, 21, 0, 5, tzinfo=timezone.utc),
        )
        key = engine.build_market_key(market)
        assert key == "baseball::2026-07-20::rangers_vs_white_sox"

        sharp = make_sharp(key, sector="baseball")
        result = engine.match(market, [sharp])
        assert result is not None

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

    def test_match_all_dedup_playoff_series(self):
        """Multiple Kalshi contracts for the same matchup → keep closest date."""
        engine = MatchingEngine()

        # Three Kalshi markets for Lakers vs Rockets at different game dates
        m1 = make_market("lakers", "rockets", sector="nba",
                         event_date=datetime(2026, 4, 18, 12, tzinfo=timezone.utc))
        m1.id = "kalshi:KXNBAGAME-26APR18HOULAL-LAL"
        m2 = make_market("lakers", "rockets", sector="nba",
                         event_date=datetime(2026, 4, 24, 12, tzinfo=timezone.utc))
        m2.id = "kalshi:KXNBAGAME-26APR24LALHOU-LAL"
        m3 = make_market("lakers", "rockets", sector="nba",
                         event_date=datetime(2026, 4, 26, 12, tzinfo=timezone.utc))
        m3.id = "kalshi:KXNBAGAME-26APR26LALHOU-LAL"

        # Single Pinnacle event for the APR19 game
        sharp_key = engine.build_market_key(
            make_market("lakers", "rockets", sector="nba",
                        event_date=datetime(2026, 4, 19, 12, tzinfo=timezone.utc))
        )
        sharp = make_sharp(sharp_key, sector="nba")
        sharp.event_date = datetime(2026, 4, 19, 12, tzinfo=timezone.utc)

        results = engine.match_all([m1, m2, m3], [sharp])
        assert len(results) == 1
        winner_market = results[0][0]
        assert winner_market.id == "kalshi:KXNBAGAME-26APR18HOULAL-LAL"


class TestNormalizerIdempotence:
    """normalize(normalize(x)) == normalize(x) for every canonical.

    Regression for 2026-09-04: the noise-word strip ran on already-canonical
    names, so "man united" → "man" on the Kalshi ticker-code path while
    Pinnacle's "Manchester United" → "man united", and the Champions League
    Man United vs Sabah market fuzzy-scored 77 < 88 against its sharp line.
    """

    def test_united_canonicals_survive_second_pass(self):
        from evmax.matching.normalizer import NameNormalizer
        n = NameNormalizer("soccer")
        assert n.normalize("Manchester United") == "man united"
        assert n.normalize("man united") == "man united"
        assert n.normalize("dc united") == "dc united"
        assert n.normalize_event_key("sabah", "man united", "2026-09-10", "soccer") == (
            "soccer::2026-09-10::sabah_vs_man_united"
        )

    def test_every_soccer_canonical_is_fixed_point(self):
        from evmax.matching.normalizer import NameNormalizer
        from evmax.sectors.registry import get_handler
        n = NameNormalizer("soccer")
        bad = sorted({
            c for c in set(get_handler("soccer")._aliases.values())
            if n.normalize(c) != c
        })
        assert bad == [], bad

    def test_non_canonical_noise_still_stripped(self):
        from evmax.matching.normalizer import NameNormalizer
        n = NameNormalizer("soccer")
        # Unknown club: noise words are still removed.
        assert n.normalize("Sabah FK") == "sabah"
        assert n.normalize("Newcastle United") == "newcastle"
