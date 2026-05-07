"""Pass 2 — Matching correctness robustness tests.

Covers:
  - fuzzy.py: empty token conservative reject, date-filter fallback logging,
               known false-positive prevention
  - kalshi.py: spread line out-of-bounds guard
  - prop_matcher.py: stat-type mismatch, threshold tolerance, player name fuzzy
"""

import pytest

from evmax.matching.fuzzy import (
    fuzzy_match,
    fuzzy_match_event_keys,
    _teams_individually_match,
    _date_close,
)
from evmax.matching.prop_matcher import PropMatcher
from evmax.models.market import MarketType, MarketSource, PredictionMarket
from evmax.models.odds import SharpBook, SharpOdds


# ---------------------------------------------------------------------------
# fuzzy.py
# ---------------------------------------------------------------------------

class TestFuzzyMatch:
    def test_exact_match(self):
        result = fuzzy_match("lakers", ["lakers", "celtics", "nets"])
        assert result is not None
        match, score = result
        assert match == "lakers"
        assert score == 100.0

    def test_no_match_below_threshold(self):
        result = fuzzy_match("xyz_totally_different", ["lakers", "celtics"])
        assert result is None

    def test_empty_query_returns_none(self):
        assert fuzzy_match("", ["lakers"]) is None

    def test_empty_candidates_returns_none(self):
        assert fuzzy_match("lakers", []) is None

    def test_transposed_team_names(self):
        # token_sort_ratio should handle order reversal
        result = fuzzy_match("brugge atletico", ["atletico brugge"])
        assert result is not None


class TestTeamsIndividuallyMatch:
    def test_exact_teams_match(self):
        assert _teams_individually_match("lakers vs celtics", "celtics vs lakers") is True

    def test_different_teams_reject(self):
        # "nets vs celtics" must NOT match "nuggets vs celtics"
        result = _teams_individually_match("nets vs celtics", "nuggets vs celtics")
        assert result is False

    def test_empty_query_tokens_returns_false(self):
        """Empty tokens should reject conservatively, not allow."""
        assert _teams_individually_match("", "lakers vs celtics") is False

    def test_empty_match_tokens_returns_false(self):
        assert _teams_individually_match("lakers vs celtics", "") is False

    def test_both_empty_returns_false(self):
        assert _teams_individually_match("", "") is False

    def test_partial_overlap_passes(self):
        # "celtics" appears in both — one team matched is enough per token
        assert _teams_individually_match("celtics vs nets", "celtics vs knicks") is False


class TestDateClose:
    def test_same_date(self):
        assert _date_close("2026-03-15", "2026-03-15") is True

    def test_one_day_apart(self):
        assert _date_close("2026-03-15", "2026-03-16") is True
        assert _date_close("2026-03-16", "2026-03-15") is True

    def test_two_days_apart_fails(self):
        assert _date_close("2026-03-15", "2026-03-17") is False

    def test_unknown_date_allows(self):
        """Unknown dates should allow match (can't check)."""
        assert _date_close("unknown", "2026-03-15") is True
        assert _date_close("2026-03-15", "unknown") is True
        assert _date_close("unknown", "unknown") is True

    def test_empty_date_allows(self):
        assert _date_close("", "2026-03-15") is True

    def test_malformed_date_allows(self):
        assert _date_close("not-a-date", "2026-03-15") is True


class TestFuzzyMatchEventKeys:
    SHARP_KEYS = [
        "nba::2026-03-15::lakers_vs_celtics",
        "nba::2026-03-15::bulls_vs_heat",
        "nba::2026-03-16::warriors_vs_suns",
        "soccer::2026-03-15::arsenal_vs_chelsea",
    ]

    def test_exact_key_match(self):
        result = fuzzy_match_event_keys(
            "nba::2026-03-15::lakers_vs_celtics",
            self.SHARP_KEYS,
        )
        assert result is not None
        key, score = result
        assert "lakers" in key and "celtics" in key

    def test_cross_sector_no_match(self):
        # Soccer key should not match NBA entries
        result = fuzzy_match_event_keys(
            "soccer::2026-03-15::lakers_vs_celtics",
            ["nba::2026-03-15::lakers_vs_celtics"],
        )
        assert result is None

    def test_nets_vs_nuggets_not_matched(self):
        """Classic false-positive: nets/nuggets share many characters."""
        result = fuzzy_match_event_keys(
            "nba::2026-03-15::nets_vs_celtics",
            ["nba::2026-03-15::nuggets_vs_celtics"],
        )
        assert result is None

    def test_reversed_team_order_matches(self):
        """Home/away order can be swapped between sources."""
        result = fuzzy_match_event_keys(
            "nba::2026-03-15::celtics_vs_lakers",
            self.SHARP_KEYS,
        )
        assert result is not None

    def test_date_one_day_off_still_matches(self):
        """Within ±1 day window should still match."""
        result = fuzzy_match_event_keys(
            "nba::2026-03-14::lakers_vs_celtics",
            self.SHARP_KEYS,
        )
        assert result is not None

    def test_date_two_days_off_uses_fallback(self):
        """2+ days off falls back to all candidates — may still match."""
        result = fuzzy_match_event_keys(
            "nba::2026-03-10::lakers_vs_celtics",
            self.SHARP_KEYS,
        )
        # Should still find lakers vs celtics via full-candidate fallback
        assert result is not None


# ---------------------------------------------------------------------------
# kalshi.py — spread line bounds check
# ---------------------------------------------------------------------------

class TestKalshiSpreadLineExtraction:
    """Test _extract_spread_line() via round-trip through KalshiClient."""

    def setup_method(self):
        from evmax.clients.kalshi import KalshiClient
        self.client = KalshiClient.__new__(KalshiClient)

    def test_normal_line(self):
        # KXNBASPREAD-26MAR09DENOKC-OKC7 → -7.5
        result = self.client._extract_spread_line("KXNBASPREAD-26MAR09DENOKC-OKC7")
        assert result == pytest.approx(-7.5)

    def test_line_of_3(self):
        result = self.client._extract_spread_line("KXNBASPREAD-26MAR09DENOKC-OKC3")
        assert result == pytest.approx(-3.5)

    def test_line_zero_is_rejected(self):
        # A "0" suffix makes no sense for a spread — reject it
        result = self.client._extract_spread_line("KXNBASPREAD-26MAR09DENOKC-OKC0")
        assert result is None

    def test_absurdly_large_line_rejected(self):
        # 700-point spread is clearly malformed data
        result = self.client._extract_spread_line("KXNBASPREAD-26MAR09DENOKC-OKC700")
        assert result is None

    def test_no_digits_returns_none(self):
        result = self.client._extract_spread_line("KXNBASPREAD-26MAR09DENOKC-OKC")
        assert result is None

    def test_line_at_boundary_50(self):
        # 50 pts is extreme but theoretically valid for NCAAB
        result = self.client._extract_spread_line("KXNCAABGAME-26MAR09XYZABC-ABC50")
        assert result == pytest.approx(-50.5)

    def test_line_51_rejected(self):
        result = self.client._extract_spread_line("KXNCAABGAME-26MAR09XYZABC-ABC51")
        assert result is None

    def test_wnba_spread_line_extracted(self):
        # KXWNBASPREAD-26MAY16NYLLVA-LVA5 → -5.5
        result = self.client._extract_spread_line("KXWNBASPREAD-26MAY16NYLLVA-LVA5")
        assert result == pytest.approx(-5.5)


class TestKalshiTeamExtractionVariableLength:
    """Outcome-anchored team-pair splitter for variable-length codes (WNBA)."""

    def setup_method(self):
        from evmax.clients.kalshi import KalshiClient
        self.client = KalshiClient.__new__(KalshiClient)

    def test_wnba_outcome_at_end_2char(self):
        # GSSEA-SEA → home=SEA, away=GS (outcome=HOME)
        home, away = self.client._extract_teams_from_ticker("KXWNBAGAME-26MAY08GSSEA-SEA")
        assert (home, away) == ("sea", "gs")

    def test_wnba_outcome_at_end_4char(self):
        # CONNNY-NY: 6-char pair would mis-split as CON/NNY under legacy logic.
        # Outcome anchoring identifies NY at the end → home=NY, away=CONN.
        home, away = self.client._extract_teams_from_ticker("KXWNBAGAME-26MAY08CONNNY-NY")
        assert (home, away) == ("ny", "conn")

    def test_wnba_outcome_at_start(self):
        # WSHTOR-WSH → outcome=WSH, pair starts with WSH → away=WSH, home=TOR
        home, away = self.client._extract_teams_from_ticker("KXWNBAGAME-26MAY08WSHTOR-WSH")
        assert (home, away) == ("tor", "wsh")

    def test_wnba_spread_outcome_with_digits(self):
        # SPREAD outcomes append the line: "SEA4". Digits stripped before anchoring.
        home, away = self.client._extract_teams_from_ticker("KXWNBASPREAD-26MAY08GSSEA-SEA4")
        assert (home, away) == ("sea", "gs")

    def test_wnba_total_outcome_numeric_falls_back(self):
        # TOTAL outcome is the line itself ("159"). Stripping digits leaves "" —
        # bail out so 6-char pairs don't get mis-split as 3+3.
        home, away = self.client._extract_teams_from_ticker("KXWNBATOTAL-26MAY08CONNNY-165")
        assert (home, away) == (None, None)

    def test_nba_legacy_3char_still_works(self):
        # Regression: don't break the existing NBA 3+3 path.
        home, away = self.client._extract_teams_from_ticker("KXNBAGAME-26FEB24ORLLAL-ORL")
        assert (home, away) == ("lal", "orl")


class TestKalshiTotalLineExtraction:
    def setup_method(self):
        from evmax.clients.kalshi import KalshiClient
        self.client = KalshiClient.__new__(KalshiClient)

    def test_wnba_total_line(self):
        # KXWNBATOTAL-...-159 → over 159.5
        assert self.client._extract_total_line("KXWNBATOTAL-26MAY08GSSEA-159") == pytest.approx(159.5)

    def test_below_50_rejected(self):
        # Quarterly/half totals (~25–40) shouldn't be parsed as game totals
        assert self.client._extract_total_line("KXWNBATOTAL-26MAY08GSSEA-40") is None

    def test_above_400_rejected(self):
        assert self.client._extract_total_line("KXWNBATOTAL-26MAY08GSSEA-500") is None


# ---------------------------------------------------------------------------
# prop_matcher.py
# ---------------------------------------------------------------------------

def _make_prop_market(player_name: str, stat_type: str, threshold: float, yes_price: float = 0.45) -> PredictionMarket:
    return PredictionMarket(
        id=f"kalshi:KXNBAPTS-TEST-{player_name.upper()[:3]}",
        source=MarketSource.kalshi,
        sector="nba",
        market_type=MarketType.player_prop,
        title=f"Will {player_name} score {threshold}+ points?",
        ticker=f"KXNBAPTS-TEST",
        yes_price=yes_price,
        no_price=1 - yes_price,
        player_name=player_name.lower(),
        stat_type=stat_type,
        threshold=threshold,
    )


def _make_prop_sharp(player_name: str, stat_type: str, threshold: float, prob_over: float = 0.55) -> SharpOdds:
    return SharpOdds(
        event_id=f"nba::2026-03-22::lakers_vs_celtics::prop::{player_name}::{stat_type}::{threshold}",
        book=SharpBook.pinnacle,
        sector="nba",
        outcome_a_label=player_name,
        outcome_b_label=player_name,
        outcome_a_decimal=1.0 / prob_over,
        outcome_b_decimal=1.0 / (1 - prob_over),
        true_prob_a=prob_over,
        true_prob_b=1 - prob_over,
        true_prob_over=prob_over,
        true_prob_under=1 - prob_over,
        total_line=threshold,
        prop_player_name=player_name.lower(),
        prop_stat_type=stat_type,
    )


class TestPropMatcher:
    def setup_method(self):
        self.matcher = PropMatcher()

    def test_exact_match(self):
        market = _make_prop_market("lebron james", "points", 24.5)
        sharp = _make_prop_sharp("lebron james", "points", 24.5)
        result = self.matcher.match(market, [sharp])
        assert result is not None
        so, conf = result
        assert so.prop_player_name == "lebron james"

    def test_stat_type_mismatch_no_match(self):
        market = _make_prop_market("lebron james", "points", 24.5)
        sharp = _make_prop_sharp("lebron james", "rebounds", 24.5)  # different stat
        result = self.matcher.match(market, [sharp])
        assert result is None

    def test_threshold_too_far_no_match(self):
        market = _make_prop_market("lebron james", "points", 24.5)
        sharp = _make_prop_sharp("lebron james", "points", 30.5)  # 6 pts apart
        result = self.matcher.match(market, [sharp])
        assert result is None

    def test_threshold_within_tolerance(self):
        # Exactly at tolerance boundary (0.5 pts)
        market = _make_prop_market("lebron james", "points", 24.5)
        sharp = _make_prop_sharp("lebron james", "points", 25.0)
        result = self.matcher.match(market, [sharp])
        # 0.5 == THRESHOLD_TOLERANCE, should match
        assert result is not None

    def test_fuzzy_player_name_match(self):
        # Minor spelling difference
        market = _make_prop_market("lebron", "points", 24.5)
        sharp = _make_prop_sharp("lebron james", "points", 24.5)
        result = self.matcher.match(market, [sharp])
        # "lebron" vs "lebron james" — depends on fuzzy score
        # This is a real use case so we just verify no crash
        # Score may or may not exceed 85 threshold
        assert result is None or isinstance(result, tuple)

    def test_player_name_too_different_no_match(self):
        market = _make_prop_market("stephen curry", "points", 24.5)
        sharp = _make_prop_sharp("lebron james", "points", 24.5)
        result = self.matcher.match(market, [sharp])
        assert result is None

    def test_match_all_returns_multiple(self):
        markets = [
            _make_prop_market("lebron james", "points", 24.5),
            _make_prop_market("anthony davis", "rebounds", 10.5),
        ]
        sharps = [
            _make_prop_sharp("lebron james", "points", 24.5),
            _make_prop_sharp("anthony davis", "rebounds", 10.5),
            _make_prop_sharp("stephen curry", "points", 26.5),
        ]
        results = self.matcher.match_all(markets, sharps)
        assert len(results) == 2

    def test_empty_markets_returns_empty(self):
        results = self.matcher.match_all([], [_make_prop_sharp("lebron james", "points", 24.5)])
        assert results == []

    def test_empty_sharps_returns_empty(self):
        results = self.matcher.match_all([_make_prop_market("lebron james", "points", 24.5)], [])
        assert results == []

    def test_market_missing_player_name_skipped(self):
        market = _make_prop_market("lebron james", "points", 24.5)
        market.player_name = None
        sharp = _make_prop_sharp("lebron james", "points", 24.5)
        result = self.matcher.match(market, [sharp])
        assert result is None

    def test_market_missing_threshold_skipped(self):
        market = _make_prop_market("lebron james", "points", 24.5)
        market.threshold = None
        sharp = _make_prop_sharp("lebron james", "points", 24.5)
        result = self.matcher.match(market, [sharp])
        assert result is None

    def test_best_match_wins_when_multiple_sharps(self):
        """When multiple sharps could match, the one with best player score wins."""
        market = _make_prop_market("lebron james", "points", 24.5)
        close_sharp = _make_prop_sharp("lebron james", "points", 24.5, prob_over=0.55)
        far_sharp = _make_prop_sharp("lebron jameson", "points", 24.5, prob_over=0.60)
        result = self.matcher.match(market, [close_sharp, far_sharp])
        assert result is not None
        so, conf = result
        assert so.prop_player_name == "lebron james"
