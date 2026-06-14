"""Tests for evmax/agents/cleanup/resolver.py.

Covers:
  - _slug_teams: slug parsing including suffix stripping and edge cases
  - _to_fuzz: punctuation normalisation (., &, -, ', _)
  - _fuzzy_team_match: token_set_ratio vs mascot names and abbreviations
  - _match_espn: event-identity gate, YES-team alignment, home/away logic,
                 best-slug tiebreaker for abbreviated yes_team
  - _match_bo3: event gate and team1/team2 swap
  - _fetch_espn_scores: JSON parsing, completed-only filtering, bad score handling
  - _resolve_via_kalshi: settlement price classification
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evmax.agents.cleanup.resolver import (
    _fuzzy_team_match,
    _match_bo3,
    _match_espn,
    _resolve_via_kalshi,
    _slug_teams,
    _to_fuzz,
    _void_prediction,
    _write_outcome,
    _fetch_espn_scores,
    _norm_prop_player,
    _resolve_prop_from_espn,
    FUZZY_THRESHOLD,
    _ACRONYM_EXPAND,
)


# ---------------------------------------------------------------------------
# _slug_teams
# ---------------------------------------------------------------------------

class TestSlugTeams:
    def test_standard_event_id(self):
        a, b = _slug_teams("nba::2026-03-19::pistons_vs_warriors")
        assert a == "pistons"
        assert b == "warriors"

    def test_multiword_teams(self):
        a, b = _slug_teams("soccer::2026-03-18::atletico_madrid_vs_real_madrid")
        assert a == "atletico madrid"
        assert b == "real madrid"

    def test_spread_suffix_stripped(self):
        # event_id with ::spread suffix — parts[2] still has _vs_
        a, b = _slug_teams("nba::2026-03-19::hornets_vs_magic::spread")
        assert a == "hornets"
        assert b == "magic"

    def test_texas_am_slug(self):
        a, b = _slug_teams("ncaab::2026-03-19::saint_marys_ca_vs_texas_a_m")
        assert a == "saint marys ca"
        assert b == "texas a m"

    def test_missing_vs_returns_empty(self):
        a, b = _slug_teams("nba::2026-03-19::unknown")
        assert a == ""
        assert b == ""

    def test_too_few_parts_returns_empty(self):
        a, b = _slug_teams("nba::2026-03-19")
        assert a == ""
        assert b == ""

    def test_empty_string_returns_empty(self):
        a, b = _slug_teams("")
        assert a == ""
        assert b == ""

    def test_single_word_teams(self):
        a, b = _slug_teams("ncaab::2026-03-19::nebraska_vs_troy")
        assert a == "nebraska"
        assert b == "troy"


# ---------------------------------------------------------------------------
# _to_fuzz
# ---------------------------------------------------------------------------

class TestToFuzz:
    def test_lowercase(self):
        assert _to_fuzz("PISTONS") == "pistons"

    def test_underscores_to_spaces(self):
        assert _to_fuzz("real_madrid") == "real madrid"

    def test_dot_removed(self):
        assert _to_fuzz("St. Louis") == "st louis"

    def test_hyphen_to_space(self):
        assert _to_fuzz("San-Antonio") == "san antonio"

    def test_ampersand_to_space(self):
        # Critical: "Texas A&M" must not produce "a&m" (one token)
        result = _to_fuzz("Texas A&M Aggies")
        assert "&" not in result
        assert "texas" in result and "aggies" in result

    def test_apostrophe_removed(self):
        # Saint Mary's → "saint marys"
        assert _to_fuzz("Saint Mary's Gaels") == "saint marys gaels"

    def test_strip_whitespace(self):
        assert _to_fuzz("  warriors  ") == "warriors"

    def test_combined_punctuation(self):
        result = _to_fuzz("Texas A&M's 1st.")
        assert "&" not in result
        assert "'" not in result
        assert "." not in result

    # -- Unicode / umlaut normalization --

    def test_umlaut_o_stripped(self):
        """ö → o so 'köln' normalises to 'koln'."""
        assert _to_fuzz("köln") == "koln"

    def test_umlaut_u_stripped(self):
        assert _to_fuzz("münchen") == "munchen"

    def test_accent_e_stripped(self):
        assert _to_fuzz("atlético") == "atletico"

    def test_umlaut_in_full_name(self):
        result = _to_fuzz("1. FC Köln")
        assert "ö" not in result
        assert "koln" in result

    def test_unicode_normalization_round_trip(self):
        """Umlaut variants of the same city must produce identical fuzz strings."""
        assert _to_fuzz("köln") == _to_fuzz("koln")
        assert _to_fuzz("münchen") == _to_fuzz("munchen")

    # -- Acronym expansion --

    def test_psg_expanded(self):
        assert _to_fuzz("psg") == "paris saint germain"

    def test_mgladbach_expanded(self):
        assert _to_fuzz("mgladbach") == "borussia monchengladbach"

    def test_m_gladbach_with_apostrophe_expanded(self):
        # After apostrophe removal "m'gladbach" → "mgladbach"
        assert _to_fuzz("m'gladbach") == "borussia monchengladbach"


# ---------------------------------------------------------------------------
# _fuzzy_team_match
# ---------------------------------------------------------------------------

class TestFuzzyTeamMatch:
    """Key invariant: short city/school slugs must match ESPN's full "City Mascots" names."""

    def test_exact_match(self):
        assert _fuzzy_team_match("pistons", "pistons") == 100.0

    def test_mascot_names_pass_threshold(self):
        """NCAAB/NBA slugs that were failing with token_sort_ratio."""
        pairs = [
            ("nebraska", "Nebraska Cornhuskers"),
            ("byu", "BYU Cougars"),
            ("tcu", "TCU Horned Frogs"),
            ("vcu", "VCU Rams"),
            ("troy", "Troy Trojans"),
            ("houston", "Houston Cougars"),
            ("wisconsin", "Wisconsin Badgers"),
            ("duke", "Duke Blue Devils"),
        ]
        for slug, espn_name in pairs:
            score = _fuzzy_team_match(slug, espn_name)
            assert score >= FUZZY_THRESHOLD, (
                f"{slug!r} vs {espn_name!r} scored {score:.1f} < {FUZZY_THRESHOLD}"
            )

    def test_texas_am_with_ampersand(self):
        """texas a m (from slug) vs Texas A&M Aggies must pass threshold."""
        score = _fuzzy_team_match("texas a m", "Texas A&M Aggies")
        assert score >= FUZZY_THRESHOLD

    def test_saint_marys_passes(self):
        score = _fuzzy_team_match("saint marys ca", "Saint Mary's Gaels")
        assert score >= FUZZY_THRESHOLD

    def test_no_match_different_teams(self):
        # Clearly different teams should score below threshold
        score = _fuzzy_team_match("warriors", "Charlotte Hornets")
        assert score < FUZZY_THRESHOLD

    def test_case_insensitive(self):
        score_lower = _fuzzy_team_match("pistons", "Detroit Pistons")
        score_upper = _fuzzy_team_match("PISTONS", "DETROIT PISTONS")
        assert score_lower == score_upper

    def test_abbreviation_matches_parent(self):
        """HP (High Point) should beat Wisconsin with token_set_ratio."""
        hp_vs_hp = _fuzzy_team_match("hp", "high point")
        hp_vs_wis = _fuzzy_team_match("hp", "wisconsin")
        # hp has a better (even if below-threshold) match with high point
        assert hp_vs_hp > hp_vs_wis


# ---------------------------------------------------------------------------
# _match_espn
# ---------------------------------------------------------------------------

def _make_score(home: str, home_score: int, away: str, away_score: int) -> dict:
    return {
        "home_name": home,
        "away_name": away,
        "home_score": home_score,
        "away_score": away_score,
        "home_won": home_score > away_score,
    }


def _make_pred(event_id: str, yes_team: str, **extra) -> dict:
    base = {"event_id": event_id, "yes_team": yes_team}
    base.update(extra)
    return base


class TestMatchEspn:
    # ------------------------------------------------------------------
    # Basic YES = home-team cases
    # ------------------------------------------------------------------

    def test_yes_team_is_home_wins(self):
        scores = [_make_score("Detroit Pistons", 117, "Washington Wizards", 95)]
        pred = _make_pred("nba::2026-03-19::pistons_vs_wizards", "pistons")
        assert _match_espn(pred, scores) == 1

    def test_yes_team_is_home_loses(self):
        scores = [_make_score("Charlotte Hornets", 100, "Orlando Magic", 115)]
        pred = _make_pred("nba::2026-03-19::hornets_vs_magic", "hornets")
        assert _match_espn(pred, scores) == 0

    # ------------------------------------------------------------------
    # YES = away-team cases
    # ------------------------------------------------------------------

    def test_yes_team_is_away_wins(self):
        scores = [_make_score("Washington Wizards", 95, "Detroit Pistons", 117)]
        pred = _make_pred("nba::2026-03-19::pistons_vs_wizards", "pistons")
        assert _match_espn(pred, scores) == 1

    def test_yes_team_is_away_loses(self):
        scores = [_make_score("Orlando Magic", 115, "Charlotte Hornets", 100)]
        pred = _make_pred("nba::2026-03-19::hornets_vs_magic", "hornets")
        assert _match_espn(pred, scores) == 0

    # ------------------------------------------------------------------
    # Event identity gate — wrong game must not match
    # ------------------------------------------------------------------

    def test_wrong_game_not_matched(self):
        # Completely different teams — should not match
        scores = [_make_score("Los Angeles Lakers", 120, "Golden State Warriors", 115)]
        pred = _make_pred("nba::2026-03-19::pistons_vs_wizards", "pistons")
        assert _match_espn(pred, scores) is None

    def test_partial_team_name_match_blocked_by_gate(self):
        # "lakers" shares "l" sounds but should not match pistons event
        scores = [_make_score("Chicago Bulls", 110, "Cleveland Cavaliers", 105)]
        pred = _make_pred("nba::2026-03-19::pistons_vs_celtics", "pistons")
        assert _match_espn(pred, scores) is None

    # ------------------------------------------------------------------
    # Mascot name matching (NCAA)
    # ------------------------------------------------------------------

    def test_ncaab_mascot_names_match(self):
        scores = [_make_score("Houston Cougars", 78, "Idaho Vandals", 47)]
        pred = _make_pred("ncaab::2026-03-20::houston_vs_idaho", "houston")
        assert _match_espn(pred, scores) == 1

    def test_ncaab_upset_mascot_names(self):
        # Wisconsin loses to High Point
        scores = [_make_score("Wisconsin Badgers", 82, "High Point Panthers", 83)]
        pred = _make_pred("ncaab::2026-03-19::wisconsin_vs_high_point", "wisconsin")
        assert _match_espn(pred, scores) == 0

    # ------------------------------------------------------------------
    # Abbreviated yes_team (best-slug tiebreaker)
    # ------------------------------------------------------------------

    def test_abbreviated_yes_team_hp(self):
        # yes_team = "hp" (Kalshi abbreviation for High Point)
        # should resolve via best-slug tiebreaker since 0 < hp_vs_hp > hp_vs_wisconsin
        scores = [_make_score("Wisconsin Badgers", 82, "High Point Panthers", 83)]
        pred = _make_pred("ncaab::2026-03-19::wisconsin_vs_high_point", "hp")
        result = _match_espn(pred, scores)
        # HP won (home_won=False because High Point is away with score 83 > 82)
        assert result == 1  # hp (High Point) won

    def test_abbreviated_yes_team_hp_loses(self):
        # Reverse: Wisconsin wins
        scores = [_make_score("Wisconsin Badgers", 90, "High Point Panthers", 75)]
        pred = _make_pred("ncaab::2026-03-19::wisconsin_vs_high_point", "hp")
        result = _match_espn(pred, scores)
        assert result == 0  # hp (High Point) lost

    # ------------------------------------------------------------------
    # Texas A&M (& in team name)
    # ------------------------------------------------------------------

    def test_texas_am_slug_matches(self):
        scores = [_make_score("Texas A&M Aggies", 63, "Saint Mary's Gaels", 50)]
        pred = _make_pred("ncaab::2026-03-19::saint_marys_ca_vs_texas_a_m", "texas a&m")
        result = _match_espn(pred, scores)
        assert result == 1  # Texas A&M won

    def test_saint_marys_loses(self):
        scores = [_make_score("Texas A&M Aggies", 63, "Saint Mary's Gaels", 50)]
        pred = _make_pred("ncaab::2026-03-19::saint_marys_ca_vs_texas_a_m", "saint marys")
        result = _match_espn(pred, scores)
        assert result == 0  # Saint Mary's lost

    # ------------------------------------------------------------------
    # Empty / missing scores
    # ------------------------------------------------------------------

    def test_empty_scores_returns_none(self):
        pred = _make_pred("nba::2026-03-19::pistons_vs_wizards", "pistons")
        assert _match_espn(pred, []) is None

    def test_malformed_event_id_no_slug(self):
        pred = _make_pred("nba::2026-03-19::unknown", "pistons")
        scores = [_make_score("Detroit Pistons", 117, "Washington Wizards", 95)]
        # Without slug, falls through to direct yes_team match
        result = _match_espn(pred, scores)
        # pistons fuzzy-matches "Detroit Pistons" as home — should resolve
        assert result == 1

    # ------------------------------------------------------------------
    # Soccer 3-way (draw detection happens upstream, not in _match_espn)
    # ------------------------------------------------------------------

    def test_soccer_home_win(self):
        # Use teams where slugs fuzzy-match ESPN names above threshold
        scores = [_make_score("Chelsea", 3, "Arsenal", 1)]
        pred = _make_pred("soccer::2026-03-19::chelsea_vs_arsenal", "chelsea")
        assert _match_espn(pred, scores) == 1

    # ------------------------------------------------------------------
    # Umlaut / unicode matching (regression: köln, atlético)
    # ------------------------------------------------------------------

    def test_koln_umlaut_matches_espn(self):
        """Event slug 'koln' must match ESPN's '1. FC Köln'."""
        scores = [_make_score("1. FC Köln", 2, "Borussia Mönchengladbach", 1)]
        pred = _make_pred("soccer::2026-03-21::koln_vs_m'gladbach", "koln")
        assert _match_espn(pred, scores) == 1

    def test_mgladbach_umlaut_matches_espn(self):
        """Event slug 'm'gladbach' must match ESPN's 'Borussia Mönchengladbach'."""
        scores = [_make_score("1. FC Köln", 2, "Borussia Mönchengladbach", 1)]
        pred = _make_pred("soccer::2026-03-21::koln_vs_m'gladbach", "m'gladbach")
        assert _match_espn(pred, scores) == 0

    def test_atletico_accent_matches_espn(self):
        """'atletico madrid' slug must match ESPN's 'Atlético de Madrid'."""
        scores = [_make_score("Real Madrid", 2, "Atlético de Madrid", 1)]
        pred = _make_pred("soccer::2026-03-22::real_madrid_vs_atletico", "real madrid")
        assert _match_espn(pred, scores) == 1

    # ------------------------------------------------------------------
    # PSG acronym expansion (regression: psg vs Paris Saint-Germain)
    # ------------------------------------------------------------------

    def test_psg_slug_matches_espn_full_name(self):
        """'psg' slug must match ESPN's 'Paris Saint-Germain' above threshold."""
        scores = [_make_score("Paris Saint-Germain", 3, "OGC Nice", 0)]
        pred = _make_pred("soccer::2026-03-21::nice_vs_psg", "psg")
        assert _match_espn(pred, scores) == 1

    def test_psg_loses(self):
        scores = [_make_score("OGC Nice", 2, "Paris Saint-Germain", 1)]
        pred = _make_pred("soccer::2026-03-21::nice_vs_psg", "psg")
        assert _match_espn(pred, scores) == 0

    def test_psg_fuzzy_threshold(self):
        """Directly verify PSG expansion meets threshold against ESPN full name."""
        score = _fuzzy_team_match("psg", "Paris Saint-Germain")
        assert score >= FUZZY_THRESHOLD, f"psg vs Paris Saint-Germain scored {score:.1f}"

    def test_multiple_scores_correct_event_selected(self):
        """Only the matching game should be used, not unrelated ones."""
        scores = [
            _make_score("Lakers", 120, "Warriors", 110),
            _make_score("Detroit Pistons", 117, "Washington Wizards", 95),
            _make_score("Bulls", 98, "Celtics", 102),
        ]
        pred = _make_pred("nba::2026-03-19::pistons_vs_wizards", "pistons")
        assert _match_espn(pred, scores) == 1

    # ------------------------------------------------------------------
    # Spread resolution — regression for BUG where _match_espn ignored
    # market_type entirely and graded every bet as a moneyline. The
    # example that surfaced it: Hornets won by 1 vs Heat on 2026-04-14,
    # a Hornets -9.5 spread bet was marked WON.
    # ------------------------------------------------------------------

    def test_spread_yes_team_wins_but_does_not_cover(self):
        scores = [_make_score("Charlotte Hornets", 127, "Miami Heat", 126)]
        pred = _make_pred(
            "nba::2026-04-14::hornets_vs_heat::spread",
            "hornets", market_type="spread", line=-9.5,
        )
        # Hornets won by 1, threshold is 9.5 → did NOT cover.
        assert _match_espn(pred, scores) == 0

    def test_spread_yes_team_covers(self):
        scores = [_make_score("Boston Celtics", 130, "Washington Wizards", 95)]
        pred = _make_pred(
            "nba::2026-03-19::celtics_vs_wizards::spread",
            "celtics", market_type="spread", line=-10.5,
        )
        # Margin 35 > 10.5 → cover.
        assert _match_espn(pred, scores) == 1

    def test_spread_yes_team_away_covers(self):
        scores = [_make_score("Washington Wizards", 95, "Boston Celtics", 130)]
        pred = _make_pred(
            "nba::2026-03-19::celtics_vs_wizards::spread",
            "celtics", market_type="spread", line=-10.5,
        )
        assert _match_espn(pred, scores) == 1

    def test_spread_exact_margin_does_not_cover(self):
        # Kalshi phrasing is "wins by OVER 9.5 points" — exactly 10 covers
        # a -9.5 line, but exactly 9 does not.
        scores = [_make_score("Boston Celtics", 109, "Washington Wizards", 100)]
        pred = _make_pred(
            "nba::2026-03-19::celtics_vs_wizards::spread",
            "celtics", market_type="spread", line=-9.5,
        )
        assert _match_espn(pred, scores) == 0  # margin 9 < 9.5

    def test_moneyline_unaffected_by_spread_branch(self):
        """Moneyline bets must still resolve on simple win/loss."""
        scores = [_make_score("Charlotte Hornets", 127, "Miami Heat", 126)]
        pred = _make_pred(
            "nba::2026-04-14::hornets_vs_heat",
            "hornets", market_type="moneyline", line=None,
        )
        assert _match_espn(pred, scores) == 1

    # ------------------------------------------------------------------
    # Total / over-under resolution — regression for BUG where _match_espn
    # required resolving the YES *side* before grading. yes_team for totals
    # is "over"/"under", which never fuzzy-matches a team slug; whenever
    # over/under tied against both team slugs (e.g. "guardians"/"nationals",
    # "red sox"/"orioles") the side resolver set yes_is_team_a=None and
    # `continue`d past the already-matched game, leaving every team-name-
    # symmetric total permanently unresolved. Totals are side-independent:
    # the combined score decides them once the event gate confirms the game.
    # ------------------------------------------------------------------

    def test_total_over_wins(self):
        scores = [_make_score("Houston Astros", 6, "Los Angeles Angels", 7)]
        pred = _make_pred(
            "baseball::2026-06-10::angels_vs_astros::total::6.5",
            "over", market_type="total", line=12.5,
        )
        assert _match_espn(pred, scores) == 1  # total 13 > 12.5

    def test_total_over_loses(self):
        scores = [_make_score("Houston Astros", 2, "Los Angeles Angels", 3)]
        pred = _make_pred(
            "baseball::2026-06-10::angels_vs_astros::total::6.5",
            "over", market_type="total", line=6.5,
        )
        assert _match_espn(pred, scores) == 0  # total 5 < 6.5

    def test_total_under_wins(self):
        scores = [_make_score("Baltimore Orioles", 3, "Boston Red Sox", 3)]
        pred = _make_pred(
            "baseball::2026-06-02::red_sox_vs_orioles::total::10.5",
            "under", market_type="total", line=10.5,
        )
        assert _match_espn(pred, scores) == 1  # total 6 < 10.5

    def test_total_under_loses(self):
        scores = [_make_score("Baltimore Orioles", 8, "Boston Red Sox", 7)]
        pred = _make_pred(
            "baseball::2026-06-02::red_sox_vs_orioles::total::10.5",
            "under", market_type="total", line=10.5,
        )
        assert _match_espn(pred, scores) == 0  # total 15 > 10.5

    def test_total_resolves_when_over_under_ties_both_slugs(self):
        # The exact regression case: "over" fuzzy-scores identically against
        # "guardians" and "nationals", which used to set yes_is_team_a=None
        # and silently drop the bet. Must now resolve from the combined score.
        scores = [_make_score("Cleveland Guardians", 2, "Washington Nationals", 10)]
        pred = _make_pred(
            "baseball::2026-05-25::guardians_vs_nationals::total::7.0",
            "over", market_type="total", line=6.5,
        )
        assert _match_espn(pred, scores) == 1  # total 12 > 6.5

    def test_total_wnba_portland_fire_resolves(self):
        # WNBA expansion team "Portland Fire" — same over/under-tie path that
        # left "Fire vs Aces" totals unresolved in the user's portfolio.
        scores = [_make_score("Portland Fire", 89, "Las Vegas Aces", 105)]
        pred = _make_pred(
            "wnba::2026-06-11::fire_vs_aces::total::171.5",
            "over", market_type="total", line=169.5,
        )
        assert _match_espn(pred, scores) == 1  # total 194 > 169.5

    def test_total_missing_line_returns_none(self):
        scores = [_make_score("Cleveland Guardians", 2, "Washington Nationals", 10)]
        pred = _make_pred(
            "baseball::2026-05-25::guardians_vs_nationals::total::7.0",
            "over", market_type="total", line=None,
        )
        assert _match_espn(pred, scores) is None

    # ------------------------------------------------------------------
    # Cross-day series — the right game in a multi-day series must win.
    # Regression: a Mon-night game whose UTC comp.date rolled forward to Tue
    # collided with the Tue game of the same series; the first one in the
    # fetch list won and graded the bet against the wrong score. _match_espn
    # now evaluates the game closest to the prediction date first.
    # ------------------------------------------------------------------

    def test_series_matches_closest_date_game(self):
        mon = {**_make_score("Colorado Rockies", 1, "Milwaukee Brewers", 7),
               "game_date": "2026-06-06"}   # total 8
        tue = {**_make_score("Colorado Rockies", 4, "Milwaukee Brewers", 12),
               "game_date": "2026-06-07"}   # total 16
        pred = _make_pred(
            "baseball::2026-06-07::rockies_vs_brewers::total::9.5",
            "over", market_type="total", line=9.5, event_date="2026-06-07",
        )
        # Tue game (total 16) is ours — over 9.5 → 1. The Mon game (total 8)
        # appearing first in the list must NOT win the match.
        assert _match_espn(pred, [mon, tue]) == 1
        assert _match_espn(pred, [tue, mon]) == 1  # order-independent


# ---------------------------------------------------------------------------
# _match_bo3
# ---------------------------------------------------------------------------

def _make_bo3_score(t1: str, t1s: int, t2: str, t2s: int) -> dict:
    return {
        "team1_name": t1,
        "team2_name": t2,
        "team1_score": t1s,
        "team2_score": t2s,
        "team1_won": t1s > t2s,
    }


class TestMatchBo3:
    def test_team1_wins(self):
        scores = [_make_bo3_score("NaVi", 2, "Team Vitality", 0)]
        pred = {"event_id": "cs2::2026-03-19::navi_vs_vitality", "yes_team": "navi"}
        assert _match_bo3(pred, scores) == 1

    def test_team2_wins(self):
        scores = [_make_bo3_score("NaVi", 0, "Team Vitality", 2)]
        pred = {"event_id": "cs2::2026-03-19::navi_vs_vitality", "yes_team": "navi"}
        assert _match_bo3(pred, scores) == 0

    def test_team1_team2_swapped_in_score(self):
        # slug_a=navi is team2 in the score, slug_b=vitality is team1
        scores = [_make_bo3_score("Team Vitality", 0, "NaVi", 2)]
        pred = {"event_id": "cs2::2026-03-19::navi_vs_vitality", "yes_team": "navi"}
        assert _match_bo3(pred, scores) == 1

    def test_wrong_game_not_matched(self):
        scores = [_make_bo3_score("Team Liquid", 2, "G2 Esports", 1)]
        pred = {"event_id": "cs2::2026-03-19::navi_vs_vitality", "yes_team": "navi"}
        assert _match_bo3(pred, scores) is None

    def test_empty_scores_returns_none(self):
        pred = {"event_id": "cs2::2026-03-19::navi_vs_vitality", "yes_team": "navi"}
        assert _match_bo3(pred, []) is None

    def test_missing_slug_returns_none(self):
        pred = {"event_id": "cs2::2026-03-19::unknown", "yes_team": "navi"}
        scores = [_make_bo3_score("NaVi", 2, "Vitality", 0)]
        assert _match_bo3(pred, scores) is None

    def test_yes_team_is_team_b_wins(self):
        scores = [_make_bo3_score("NaVi", 1, "Team Vitality", 2)]
        pred = {"event_id": "cs2::2026-03-19::navi_vs_vitality", "yes_team": "vitality"}
        assert _match_bo3(pred, scores) == 1


# ---------------------------------------------------------------------------
# _fetch_espn_scores (async, mocked)
# ---------------------------------------------------------------------------

def _build_espn_response(events: list[dict]) -> dict:
    return {"events": events}


def _build_espn_event(home: str, home_score: int, away: str, away_score: int,
                       completed: bool = True) -> dict:
    return {
        "competitions": [{
            "status": {"type": {"completed": completed}},
            "competitors": [
                {
                    "homeAway": "home",
                    "score": str(home_score),
                    "team": {"displayName": home},
                },
                {
                    "homeAway": "away",
                    "score": str(away_score),
                    "team": {"displayName": away},
                },
            ],
        }]
    }


class TestFetchEspnScores:
    def _make_mock_client(self, response_data: dict, status_code: int = 200):
        mock_resp = MagicMock()
        mock_resp.json.return_value = response_data
        mock_resp.raise_for_status.return_value = None
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        return mock_client

    def test_returns_completed_games(self):
        data = _build_espn_response([
            _build_espn_event("Detroit Pistons", 117, "Washington Wizards", 95),
            _build_espn_event("Charlotte Hornets", 130, "Orlando Magic", 111),
        ])
        client = self._make_mock_client(data)

        results = asyncio.run(_fetch_espn_scores(client, "basketball", "nba", "20260319"))
        assert len(results) == 2
        assert results[0]["home_name"] == "Detroit Pistons"
        assert results[0]["home_score"] == 117
        assert results[0]["home_won"] is True

    def test_game_date_uses_queried_slate_not_utc(self):
        # comp.date is UTC and rolls a night game to the next day; the stored
        # game_date must instead be the queried US slate date, otherwise a
        # Mon-night game collides with the Tue game of the same series and the
        # wrong game resolves a bet. Regression for cross-day series mismatch.
        event = _build_espn_event("Colorado Rockies", 7, "Milwaukee Brewers", 9)
        event["competitions"][0]["date"] = "2026-06-07T01:10Z"  # UTC = next day
        client = self._make_mock_client(_build_espn_response([event]))
        results = asyncio.run(_fetch_espn_scores(client, "baseball", "mlb", "20260606"))
        assert results[0]["game_date"] == "2026-06-06"

    def test_skips_incomplete_games(self):
        data = _build_espn_response([
            _build_espn_event("Team A", 80, "Team B", 75, completed=True),
            _build_espn_event("Team C", 0, "Team D", 0, completed=False),
        ])
        client = self._make_mock_client(data)

        results = asyncio.run(_fetch_espn_scores(client, "basketball", "nba", "20260319"))
        assert len(results) == 1
        assert results[0]["home_name"] == "Team A"

    def test_skips_missing_home_away(self):
        # Event with no homeAway designation
        data = {
            "events": [{
                "competitions": [{
                    "status": {"type": {"completed": True}},
                    "competitors": [
                        {"score": "100", "team": {"displayName": "Team X"}},
                        {"score": "90", "team": {"displayName": "Team Y"}},
                    ],
                }]
            }]
        }
        client = self._make_mock_client(data)
        results = asyncio.run(_fetch_espn_scores(client, "basketball", "nba", "20260319"))
        assert results == []

    def test_skips_bad_score_values(self):
        data = {
            "events": [{
                "competitions": [{
                    "status": {"type": {"completed": True}},
                    "competitors": [
                        {"homeAway": "home", "score": "not-a-number", "team": {"displayName": "A"}},
                        {"homeAway": "away", "score": "80", "team": {"displayName": "B"}},
                    ],
                }]
            }]
        }
        client = self._make_mock_client(data)
        results = asyncio.run(_fetch_espn_scores(client, "basketball", "nba", "20260319"))
        assert results == []

    def test_http_error_returns_empty(self):
        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("Connection error")
        results = asyncio.run(_fetch_espn_scores(mock_client, "basketball", "nba", "20260319"))
        assert results == []

    def test_extra_params_passed_through(self):
        data = _build_espn_response([])
        client = self._make_mock_client(data)
        asyncio.run(_fetch_espn_scores(client, "basketball", "mens-college-basketball",
                                        "20260319", extra_params={"groups": "50"}))
        call_kwargs = client.get.call_args[1]
        assert call_kwargs["params"]["groups"] == "50"

    def test_home_won_flag_correct(self):
        data = _build_espn_response([
            _build_espn_event("Underdog", 50, "Favorite", 90),  # away wins
        ])
        client = self._make_mock_client(data)
        results = asyncio.run(_fetch_espn_scores(client, "basketball", "nba", "20260319"))
        assert results[0]["home_won"] is False


# ---------------------------------------------------------------------------
# _write_outcome
# ---------------------------------------------------------------------------

class TestWriteOutcome:
    def _make_conn(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE ev_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT UNIQUE,
                event_id TEXT,
                event_date TEXT,
                sector TEXT,
                yes_team TEXT,
                outcome INTEGER,
                sharp_true_prob REAL,
                blended_true_prob REAL,
                resolved_at TEXT,
                result_source TEXT
            )
        """)
        return conn

    def test_writes_yes_outcome(self):
        conn = self._make_conn()
        pred = {
            "market_id": "kalshi:TEST-123",
            "event_id": "nba::2026-03-19::pistons_vs_wizards",
            "sector": "nba",
            "yes_team": "pistons",
            "event_date": "2026-03-19",
            "sharp_true_prob": 0.65,
            "blended_true_prob": 0.70,
        }
        _write_outcome(conn, pred, 1, "espn")
        row = conn.execute("SELECT * FROM ev_outcomes WHERE market_id = ?",
                           ("kalshi:TEST-123",)).fetchone()
        assert row is not None
        assert row["outcome"] == 1
        assert row["result_source"] == "espn"

    def test_writes_no_outcome(self):
        conn = self._make_conn()
        pred = {
            "market_id": "kalshi:TEST-456",
            "event_id": "nba::2026-03-19::test",
            "sector": "nba",
            "yes_team": "wizards",
            "event_date": "2026-03-19",
            "sharp_true_prob": 0.35,
            "blended_true_prob": 0.30,
        }
        _write_outcome(conn, pred, 0, "espn")
        row = conn.execute("SELECT outcome FROM ev_outcomes WHERE market_id = ?",
                           ("kalshi:TEST-456",)).fetchone()
        assert row["outcome"] == 0

    def test_replace_on_conflict(self):
        conn = self._make_conn()
        pred = {
            "market_id": "kalshi:TEST-DUPE",
            "event_id": "nba::2026-03-19::test",
            "sector": "nba",
            "yes_team": "pistons",
            "event_date": "2026-03-19",
            "sharp_true_prob": 0.65,
            "blended_true_prob": 0.70,
        }
        _write_outcome(conn, pred, 0, "espn")
        _write_outcome(conn, pred, 1, "espn_retry")  # override
        rows = conn.execute("SELECT * FROM ev_outcomes WHERE market_id = ?",
                            ("kalshi:TEST-DUPE",)).fetchall()
        assert len(rows) == 1
        assert rows[0]["outcome"] == 1

    def test_commit_per_write_releases_lock(self, tmp_path):
        """Each _write_outcome commits, so a second connection can immediately read.

        Regression for 'database is locked' from portfolios.sync_portfolio_outcomes:
        if the resolver held one giant transaction across all writes + HTTP work,
        a concurrent writer (the dashboard sync button) would hit the 5s
        busy_timeout and 500 with OperationalError.
        """
        db_path = tmp_path / "test.db"
        # Writer connection initializes the schema.
        writer = sqlite3.connect(str(db_path))
        writer.row_factory = sqlite3.Row
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("""
            CREATE TABLE ev_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT UNIQUE,
                event_id TEXT,
                event_date TEXT,
                sector TEXT,
                yes_team TEXT,
                outcome INTEGER,
                sharp_true_prob REAL,
                blended_true_prob REAL,
                resolved_at TEXT,
                result_source TEXT
            )
        """)
        writer.commit()

        pred = {
            "market_id": "kalshi:LOCKTEST",
            "event_id": "nba::2026-05-10::test",
            "sector": "nba",
            "yes_team": "lakers",
            "event_date": "2026-05-10",
            "sharp_true_prob": 0.55,
            "blended_true_prob": 0.58,
        }
        _write_outcome(writer, pred, 1, "espn_boxscore")

        # Independent reader connection — would see no row if the writer hadn't
        # committed (and would block on a non-WAL DB until busy_timeout expires).
        reader = sqlite3.connect(str(db_path), timeout=0.5)
        reader.row_factory = sqlite3.Row
        row = reader.execute(
            "SELECT outcome, result_source FROM ev_outcomes WHERE market_id = ?",
            ("kalshi:LOCKTEST",),
        ).fetchone()
        reader.close()
        writer.close()
        assert row is not None
        assert row["outcome"] == 1
        assert row["result_source"] == "espn_boxscore"


# ---------------------------------------------------------------------------
# Pending deduplication helper (unit test for the dedup logic)
# ---------------------------------------------------------------------------

class TestPendingDedup:
    """Verify that duplicate market_ids in pending produce only one entry each."""

    def _dedup(self, rows: list[dict]) -> list[dict]:
        """Replicates the dedup logic from resolve_outcomes_for_date."""
        seen: set[str] = set()
        out: list[dict] = []
        for d in rows:
            if d["market_id"] not in seen:
                seen.add(d["market_id"])
                out.append(d)
        return out

    def test_no_duplicates_unchanged(self):
        rows = [
            {"market_id": "kalshi:A", "event_id": "nba::2026-03-21::rockets_vs_hawks"},
            {"market_id": "kalshi:B", "event_id": "nba::2026-03-21::nuggets_vs_raptors"},
        ]
        assert len(self._dedup(rows)) == 2

    def test_duplicates_reduced(self):
        rows = [
            {"market_id": "kalshi:A", "event_id": "soccer::2026-03-21::nice_vs_psg"},
            {"market_id": "kalshi:A", "event_id": "soccer::2026-03-21::nice_vs_psg"},
            {"market_id": "kalshi:A", "event_id": "soccer::2026-03-21::nice_vs_psg"},
            {"market_id": "kalshi:B", "event_id": "soccer::2026-03-21::koln_vs_mgladbach"},
        ]
        result = self._dedup(rows)
        assert len(result) == 2
        assert result[0]["market_id"] == "kalshi:A"
        assert result[1]["market_id"] == "kalshi:B"

    def test_first_occurrence_kept(self):
        rows = [
            {"market_id": "kalshi:A", "event_id": "first"},
            {"market_id": "kalshi:A", "event_id": "second"},
        ]
        result = self._dedup(rows)
        assert result[0]["event_id"] == "first"


# ---------------------------------------------------------------------------
# Off-by-one date matching (2-day window)
# ---------------------------------------------------------------------------

class TestOffByOneDateMatching:
    """Verify that a game stored with the wrong event_date (off by 1) still resolves.

    The resolver fetches ESPN scores for both event_date AND event_date-1.
    These tests exercise _match_espn directly against the combined score list,
    simulating the merged window that resolve_outcomes_for_date now builds.
    """

    def test_game_found_on_prev_day(self):
        """Event stored as 3-22 but game was on 3-21 — prev-day scores contain it."""
        prev_day_scores = [_make_score("Golden State Warriors", 115, "Atlanta Hawks", 102)]
        stored_day_scores = []  # empty for 3-22
        combined = stored_day_scores + prev_day_scores
        pred = _make_pred("nba::2026-03-22::hawks_vs_warriors", "hawks")
        # Hawks are away and lost — outcome = 0
        assert _match_espn(pred, combined) == 0

    def test_game_found_on_prev_day_home_wins(self):
        prev_day_scores = [_make_score("San Antonio Spurs", 110, "Indiana Pacers", 95)]
        pred = _make_pred("nba::2026-03-22::spurs_vs_pacers", "spurs")
        assert _match_espn(pred, prev_day_scores) == 1

    def test_stored_date_takes_precedence_when_both_have_game(self):
        """If the same matchup somehow appears in both windows, first match is used."""
        prev_day_score = _make_score("Detroit Pistons", 100, "Los Angeles Lakers", 98)
        stored_day_score = _make_score("Detroit Pistons", 105, "Los Angeles Lakers", 100)
        combined = [stored_day_score, prev_day_score]
        pred = _make_pred("nba::2026-03-23::pistons_vs_lakers", "pistons")
        assert _match_espn(pred, combined) == 1  # pistons won in both, either is fine

    def test_soccer_barcelona_prev_day(self):
        """Barcelona match stored as 3-22 resolves against prev-day scores."""
        prev_scores = [_make_score("FC Barcelona", 3, "Rayo Vallecano", 0)]
        pred = _make_pred("soccer::2026-03-22::barcelona_vs_rayo_vallecano", "barcelona")
        assert _match_espn(pred, prev_scores) == 1


# ---------------------------------------------------------------------------
# Prop resolution — ESPN-first path keeps stats.nba.com out of the hot loop.
# ---------------------------------------------------------------------------

class TestNormPropPlayer:
    def test_lowercase_underscores(self):
        assert _norm_prop_player("Jalen Duren") == "jalen_duren"

    def test_strips_accents(self):
        assert _norm_prop_player("Luka Dončić") == "luka_doncic"

    def test_strips_periods(self):
        assert _norm_prop_player("P.J. Washington") == "pj_washington"


class TestResolvePropFromEspn:
    """ESPN lookup hands back (outcome, value) when player+stat are present.

    These directly exercise the helper that replaced per-bet stats.nba.com
    calls. The 30s timeouts in the logs were one PlayerGameLogs call PER
    (player, stat); this path lets one ESPN summary call serve every prop
    for that player on the slate.
    """

    @staticmethod
    def _stats():
        return {
            "donovan_mitchell": {"PTS": 28.0, "REB": 6.0, "AST": 9.0, "3PT": 4.0,
                                 "STL": 2.0, "BLK": 0.0, "TO": 3.0},
            "jalen_duren":      {"PTS": 18.0, "REB": 14.0, "AST": 1.0, "3PT": 0.0,
                                 "STL": 1.0, "BLK": 2.0, "TO": 2.0},
        }

    def test_points_over(self):
        out = _resolve_prop_from_espn(self._stats(), "donovan_mitchell", "points", 24.5)
        assert out == (1, 28.0)

    def test_points_under(self):
        out = _resolve_prop_from_espn(self._stats(), "jalen_duren", "points", 19.5)
        assert out == (0, 18.0)

    def test_threes(self):
        out = _resolve_prop_from_espn(self._stats(), "donovan_mitchell", "threes", 2.5)
        assert out == (1, 4.0)

    def test_points_rebounds_assists_derived(self):
        out = _resolve_prop_from_espn(
            self._stats(), "donovan_mitchell", "points_rebounds_assists", 40.5,
        )
        assert out == (1, 43.0)

    def test_blocks_steals_derived(self):
        out = _resolve_prop_from_espn(
            self._stats(), "jalen_duren", "blocks_steals", 2.5,
        )
        assert out == (1, 3.0)

    def test_last_name_fallback(self):
        # Query name doesn't match the cache exactly but last token does.
        out = _resolve_prop_from_espn(self._stats(), "d_mitchell", "points", 25.0)
        assert out == (1, 28.0)

    def test_player_not_in_espn_returns_none(self):
        # Caller falls back to nba_api when ESPN doesn't have the player.
        assert _resolve_prop_from_espn(self._stats(), "ghost_player", "points", 10.0) is None

    def test_unknown_stat_type_returns_none(self):
        assert _resolve_prop_from_espn(self._stats(), "donovan_mitchell", "fouls", 2.5) is None


class TestPropResolutionEspnFirst:
    """End-to-end: when ESPN has the data, nba_api is never called.

    Regression for the 2026-05-11 timeout flood — per-bet PlayerGameLogs
    calls were exhausting the stats.nba.com quota and 30s-timing-out on
    multi-stat players like Mitchell, Duren, Hachimura.
    """

    class _NoCloseConn:
        """Wraps a sqlite3 connection so `.close()` is a no-op. resolve_outcomes_for_date
        closes its connection at the end; in-memory DBs die with the connection,
        so this lets the test still query ev_outcomes after the resolver returns."""
        def __init__(self, conn):
            self._conn = conn
        def __getattr__(self, item):
            return getattr(self._conn, item)
        def close(self):
            pass

    def _make_db(self):
        from evmax.agents.cleanup.db import SCHEMA
        raw = sqlite3.connect(":memory:")
        raw.row_factory = sqlite3.Row
        raw.executescript(SCHEMA)
        return self._NoCloseConn(raw)

    def _seed_prop_pred(self, conn, market_id, player, stat, threshold, event_date="2026-05-10"):
        conn.execute(
            """INSERT INTO ev_predictions
               (scan_date, market_id, event_id, sector, yes_team, market_type,
                event_title, event_date, kalshi_yes_price, sharp_true_prob,
                blended_true_prob, ev_pct, kelly_fraction, bankroll_used, line)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event_date, market_id,
                f"nba::{event_date}::prop::{player}::{stat}::{threshold}",
                "nba", "yes", "player_prop",
                f"{player} {stat}", event_date,
                0.40, 0.45, 0.48, 0.05, 0.01, 500.0, threshold,
            ),
        )
        conn.commit()

    def test_espn_resolves_without_nba_api(self):
        from evmax.agents.cleanup import resolver

        conn = self._make_db()
        self._seed_prop_pred(conn, "kalshi:M1", "donovan_mitchell", "points", 24.5)
        self._seed_prop_pred(conn, "kalshi:M2", "donovan_mitchell", "rebounds", 5.5)
        self._seed_prop_pred(conn, "kalshi:M3", "donovan_mitchell", "assists", 7.5)

        espn_payload = {
            "donovan_mitchell": {"PTS": 28.0, "REB": 6.0, "AST": 9.0,
                                 "3PT": 4.0, "STL": 2.0, "BLK": 0.0, "TO": 3.0},
        }

        with patch.object(resolver, "get_connection", return_value=conn), \
             patch.object(resolver, "_fetch_espn_nba_player_stats", return_value=espn_payload) as espn_fetch, \
             patch.object(resolver, "_resolve_prop_outcome") as nba_api_fallback:
            asyncio.run(resolver.resolve_outcomes_for_date(date(2026, 5, 10)))

        # ESPN is called exactly once per (sector, date), not once per (player, stat).
        assert espn_fetch.call_count == 1
        # nba_api fallback should never fire when ESPN resolves all rows.
        assert nba_api_fallback.call_count == 0

        outcomes = conn.execute(
            "SELECT market_id, outcome, result_source FROM ev_outcomes ORDER BY market_id"
        ).fetchall()
        assert len(outcomes) == 3
        for row in outcomes:
            assert row["outcome"] == 1
            assert row["result_source"] == "espn_boxscore"

    def test_falls_back_to_nba_api_when_espn_misses(self):
        from evmax.agents.cleanup import resolver

        conn = self._make_db()
        self._seed_prop_pred(conn, "kalshi:M9", "obscure_rookie", "points", 9.5)

        with patch.object(resolver, "get_connection", return_value=conn), \
             patch.object(resolver, "_fetch_espn_nba_player_stats", return_value={}), \
             patch.object(resolver, "_resolve_prop_outcome", return_value=1) as nba_api_fallback:
            asyncio.run(resolver.resolve_outcomes_for_date(date(2026, 5, 10)))

        assert nba_api_fallback.call_count == 1
        row = conn.execute(
            "SELECT outcome, result_source FROM ev_outcomes WHERE market_id = ?",
            ("kalshi:M9",),
        ).fetchone()
        assert row["outcome"] == 1
        assert row["result_source"] == "nba_api"


# ---------------------------------------------------------------------------
# Scalar settlement → void (cancelled match / walkover before play)
#
# Regression for Paul vs Mpetshi Perricard (2026 ATP Stuttgart R32): Tommy Paul
# withdrew with a neck injury before a ball was played, so Kalshi finalized the
# binary market to a scalar fair-price refund (result="scalar") instead of a
# Yes/No. The old resolver only handled result in {"yes","no"} and left these
# rows unresolved forever. They must now be voided.
# ---------------------------------------------------------------------------

class _FakeKalshiClient:
    """Async-context stub exposing get_market_settlement keyed by ticker."""

    def __init__(self, verdicts: dict[str, str | None]):
        self._verdicts = verdicts

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get_market_settlement(self, ticker: str):
        return self._verdicts.get(ticker)


class TestGetMarketSettlement:
    """KalshiClient.get_market_settlement verdict classification."""

    def _verdict(self, market: dict) -> str | None:
        from evmax.clients.kalshi import KalshiClient

        client = KalshiClient.__new__(KalshiClient)  # skip RSA-auth __init__
        client._get = AsyncMock(return_value={"market": market})
        return asyncio.run(client.get_market_settlement("KXSOME-TICKER"))

    def test_result_yes(self):
        assert self._verdict({"result": "yes", "status": "finalized"}) == "yes"

    def test_result_no(self):
        assert self._verdict({"result": "no", "status": "finalized"}) == "no"

    def test_scalar_is_void(self):
        # Real shape from the Paul/Perricard market: finalized + result="scalar".
        market = {
            "result": "scalar",
            "status": "finalized",
            "settlement_value_dollars": "0.3300",
            "yes_bid_dollars": "0.0000",
            "yes_ask_dollars": "1.0000",
        }
        assert self._verdict(market) == "void"

    def test_finalized_without_binary_result_is_void(self):
        assert self._verdict({"result": "", "status": "finalized"}) == "void"

    def test_open_near_one_is_yes(self):
        market = {"result": "", "status": "active",
                  "yes_bid_dollars": "0.99", "yes_ask_dollars": "1.0000"}
        assert self._verdict(market) == "yes"

    def test_open_near_zero_is_no(self):
        market = {"result": "", "status": "active",
                  "yes_bid_dollars": "0.0000", "yes_ask_dollars": "0.0100"}
        assert self._verdict(market) == "no"

    def test_open_mid_is_none(self):
        market = {"result": "", "status": "active",
                  "yes_bid_dollars": "0.40", "yes_ask_dollars": "0.42"}
        assert self._verdict(market) is None

    def test_legacy_cents_fields(self):
        market = {"result": "", "status": "active", "yes_bid": 99, "yes_ask": 100}
        assert self._verdict(market) == "yes"

    def test_fetch_error_is_none(self):
        from evmax.clients.kalshi import KalshiClient

        client = KalshiClient.__new__(KalshiClient)
        client._get = AsyncMock(side_effect=RuntimeError("boom"))
        assert asyncio.run(client.get_market_settlement("KXSOME")) is None


class TestResolveViaKalshi:
    """_resolve_via_kalshi splits outcomes from scalar-void settlements."""

    def test_classifies_yes_no_void_open(self):
        preds = [
            {"market_id": "kalshi:T_YES"},
            {"market_id": "kalshi:T_NO"},
            {"market_id": "kalshi:T_VOID"},
            {"market_id": "kalshi:T_OPEN"},
        ]
        verdicts = {"T_YES": "yes", "T_NO": "no", "T_VOID": "void", "T_OPEN": None}
        with patch("evmax.clients.kalshi.KalshiClient",
                   lambda: _FakeKalshiClient(verdicts)):
            out, voided = asyncio.run(_resolve_via_kalshi(preds))

        assert out["kalshi:T_YES"] == 1
        assert out["kalshi:T_NO"] == 0
        assert out["kalshi:T_OPEN"] is None
        assert "kalshi:T_VOID" not in out  # voids never become a binary outcome
        assert voided == {"kalshi:T_VOID"}


class TestScalarSettlementVoidIntegration:
    """End-to-end: a scalar settlement voids the row and writes no outcome."""

    class _NoCloseConn:
        def __init__(self, conn):
            self._conn = conn

        def __getattr__(self, item):
            return getattr(self._conn, item)

        def close(self):
            pass

    def _make_db(self):
        from evmax.agents.cleanup.db import SCHEMA
        raw = sqlite3.connect(":memory:")
        raw.row_factory = sqlite3.Row
        raw.executescript(SCHEMA)
        return self._NoCloseConn(raw)

    def _seed_tennis_pred(self, conn, market_id, event_date="2026-06-08"):
        conn.execute(
            """INSERT INTO ev_predictions
               (scan_date, market_id, event_id, sector, yes_team, market_type,
                event_title, event_date, kalshi_yes_price, sharp_true_prob,
                blended_true_prob, ev_pct, kelly_fraction, bankroll_used, line)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event_date, market_id,
                "tennis::2026-06-08::paul_vs_perricard",
                "tennis", "perricard", "moneyline",
                "Tommy Paul vs Giovanni Mpetshi Perricard", event_date,
                0.32, 0.33, 0.33, 0.03, 0.01, 500.0, None,
            ),
        )
        conn.commit()

    def test_tennis_scalar_settlement_voids_row(self):
        from evmax.agents.cleanup import resolver

        conn = self._make_db()
        mid = "kalshi:KXATPMATCH-26JUN08PAUMPE-MPE"
        self._seed_tennis_pred(conn, mid)

        with patch.object(resolver, "get_connection", return_value=conn), \
             patch.object(resolver, "_resolve_via_kalshi",
                          AsyncMock(return_value=({}, {mid}))):
            result = asyncio.run(resolver.resolve_outcomes_for_date(date(2026, 6, 8)))

        row = conn.execute(
            "SELECT voided FROM ev_predictions WHERE market_id = ?", (mid,)
        ).fetchone()
        assert row["voided"] == 1
        # No binary outcome is written for a voided match.
        assert conn.execute("SELECT COUNT(*) AS c FROM ev_outcomes").fetchone()["c"] == 0
        assert result["voided"] == 1
        assert result["resolved"] == 0

    def test_void_prediction_is_idempotent(self):
        conn = self._make_db()
        mid = "kalshi:X1"
        self._seed_tennis_pred(conn, mid)

        _void_prediction(conn, {"market_id": mid})
        _void_prediction(conn, {"market_id": mid})  # second call must not error

        row = conn.execute(
            "SELECT voided FROM ev_predictions WHERE market_id = ?", (mid,)
        ).fetchone()
        assert row["voided"] == 1
        # No ev_outcomes side effect.
        assert conn.execute("SELECT COUNT(*) AS c FROM ev_outcomes").fetchone()["c"] == 0
