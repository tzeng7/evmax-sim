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
from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evmax.agents.cleanup.resolver import (
    _fuzzy_team_match,
    _is_polymarket_us_row,
    _match_bo3,
    _match_espn,
    _mlb_outs_from_innings_pitched,
    _polyus_bet_is_long,
    _fetch_json_concurrent,
    _resolve_baseball_prop_observations,
    _resolve_prop_observations,
    _resolve_via_kalshi,
    PROP_RESOLVE_LOOKBACK_DAYS,
    _resolve_via_polymarket_us,
    _slug_teams,
    _split_polymarket_us_id,
    _to_fuzz,
    _void_prediction,
    _write_outcome,
    _fetch_espn_scores,
    _norm_prop_player,
    _resolve_prop_from_espn,
    clv_entry_price,
    FUZZY_THRESHOLD,
    _ACRONYM_EXPAND,
)


# ---------------------------------------------------------------------------
# clv_entry_price — CLV is anchored to the price we actually get in at
# ---------------------------------------------------------------------------
class TestClvEntryPrice:
    def test_placed_uses_fill_price(self):
        # A placed bet anchors CLV to the real fill, not the scan ask.
        assert clv_entry_price(placed=1, placed_price=0.22, scan_price=0.30) == 0.22

    def test_unplaced_uses_scan_price(self):
        assert clv_entry_price(placed=0, placed_price=None, scan_price=0.30) == 0.30

    def test_placed_but_no_fill_falls_back_to_scan(self):
        # placed=1 yet placed_price missing → scan price.
        assert clv_entry_price(placed=1, placed_price=None, scan_price=0.30) == 0.30

    def test_fill_out_of_range_falls_back_to_scan(self):
        assert clv_entry_price(placed=1, placed_price=0.0, scan_price=0.30) == 0.30
        assert clv_entry_price(placed=1, placed_price=1.0, scan_price=0.30) == 0.30

    def test_null_placed_uses_scan_price(self):
        # legacy rows where placed is NULL behave like unplaced.
        assert clv_entry_price(placed=None, placed_price=0.22, scan_price=0.30) == 0.30


# ---------------------------------------------------------------------------
# backfill_clv — Kalshi CLV must be tracked identically for YES and NO bets
# ---------------------------------------------------------------------------
# Regression for the bug where NO-side bets (under totals / opponent +spread)
# carry a ":no" market_id suffix. The raw ticker (KX...:no) never matched the
# archived clean ticker, so EVERY no-side bet silently got NULL kalshi_clv_pct —
# a sector-uneven gap (baseball totals, NBA/WNBA spreads were ~all no-side).
class TestBackfillClvNoSide:
    def _setup_dbs(self, tmp_path, monkeypatch):
        """Build a minimal predictions.db + archive.db and point the code at them."""
        import evmax.archiver as archiver_mod
        from evmax.agents.cleanup import db as cleanup_db

        pred_path = tmp_path / "predictions.db"
        arch_path = tmp_path / "archive.db"
        # get_connection() reads DB_PATH at call time and auto-migrates schema.
        monkeypatch.setattr(cleanup_db, "DB_PATH", pred_path)
        monkeypatch.setattr(archiver_mod, "DB_PATH", arch_path)

        conn = cleanup_db.get_connection()  # creates + migrates predictions schema

        event_id = "baseball::2026-06-02::reds_vs_royals::total::5.5"
        ticker_yes = "KXMLBTOTAL-26JUN021910KCCIN-7"  # YES = over
        # YES-side total (over) and NO-side total (under) on the same Kalshi market.
        for mid, yes_team, mtype, entry in [
            (f"kalshi:{ticker_yes}", "over", "total", 0.40),
            (f"kalshi:{ticker_yes}:no", "under", "total", 0.62),
        ]:
            conn.execute(
                """INSERT INTO ev_predictions
                   (scan_date, market_id, event_id, sector, yes_team, market_type,
                    event_date, kalshi_yes_price, sharp_true_prob, blended_true_prob,
                    ev_pct, kelly_fraction)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                ("2026-06-01", mid, event_id, "baseball", yes_team, mtype,
                 "2026-06-02", entry, 0.5, 0.5, 5.0, 0.02),
            )
            conn.execute(
                """INSERT INTO ev_outcomes
                   (market_id, event_id, event_date, sector, yes_team, outcome)
                   VALUES (?,?,?,?,?,?)""",
                (mid, event_id, "2026-06-02", "baseball", yes_team, 1),
            )
        conn.commit()
        conn.close()

        # archive: tipoff anchor (Pinnacle) + Kalshi YES close snapshot @ T-60.
        arch = sqlite3.connect(str(arch_path))
        arch.execute(
            """CREATE TABLE archived_sharp_odds (
                   id INTEGER PRIMARY KEY, session_id TEXT, fetched_at TEXT,
                   sector TEXT, event_id TEXT, book TEXT,
                   outcome_a_label TEXT, outcome_b_label TEXT,
                   outcome_a_decimal REAL, outcome_b_decimal REAL,
                   true_prob_a REAL, true_prob_b REAL, true_prob_draw REAL,
                   margin REAL, spread_line REAL, event_date TEXT,
                   true_prob_over REAL, true_prob_under REAL, total_line REAL)"""
        )
        arch.execute(
            """CREATE TABLE archived_kalshi_markets (
                   id INTEGER PRIMARY KEY, session_id TEXT, fetched_at TEXT,
                   sector TEXT, ticker TEXT, market_type TEXT,
                   yes_price REAL, no_price REAL, event_date TEXT, event_id TEXT)"""
        )
        arch.execute(
            """INSERT INTO archived_sharp_odds
               (session_id, fetched_at, sector, event_id, book,
                outcome_a_decimal, outcome_b_decimal, true_prob_a, true_prob_b,
                margin, event_date)
               VALUES ('s','2026-06-02T22:00:00+00:00','baseball',?, 'pinnacle',
                       2.0, 2.0, 0.5, 0.5, 0.0, '2026-06-02T23:10:00+00:00')""",
            (event_id,),
        )
        # YES (over) close = 0.55, well before the 23:10 tipoff.
        arch.execute(
            """INSERT INTO archived_kalshi_markets
               (session_id, fetched_at, sector, ticker, market_type,
                yes_price, no_price, event_date, event_id)
               VALUES ('s','2026-06-02T22:00:00+00:00','baseball',?, 'total',
                       0.55, 0.45, '2026-06-02', ?)""",
            (ticker_yes, event_id),
        )
        arch.commit()
        arch.close()
        return pred_path

    def test_no_side_clv_is_populated_and_flipped(self, tmp_path, monkeypatch):
        from evmax.agents.cleanup.resolver import backfill_clv
        from evmax.agents.cleanup import db as cleanup_db

        pred_path = self._setup_dbs(tmp_path, monkeypatch)
        result = backfill_clv()

        conn = sqlite3.connect(str(pred_path))
        conn.row_factory = sqlite3.Row
        rows = {
            r["market_id"]: r["kalshi_clv_pct"]
            for r in conn.execute(
                "SELECT market_id, kalshi_clv_pct FROM ev_predictions"
            ).fetchall()
        }
        conn.close()

        ticker = "kalshi:KXMLBTOTAL-26JUN021910KCCIN-7"
        # YES (over): close 0.55 − entry 0.40 = +15.0pp
        assert rows[ticker] == pytest.approx(15.0, abs=0.01)
        # NO (under): close is FLIPPED to 1 − 0.55 = 0.45; 0.45 − entry 0.62 = −17.0pp.
        # Pre-fix this was NULL because the ":no" ticker never matched the archive.
        assert rows[f"{ticker}:no"] is not None
        assert rows[f"{ticker}:no"] == pytest.approx(-17.0, abs=0.01)
        assert result["updated"] == 2


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
# _fetch_espn_scores: bounded concurrency + per-run cache (perf refactor)
# ---------------------------------------------------------------------------

class TestEspnFetchConcurrencyAndCache:
    def _resp(self, data):
        r = MagicMock()
        r.json.return_value = data
        r.raise_for_status.return_value = None
        return r

    def test_bounded_concurrency_never_exceeds_semaphore(self):
        """Twenty concurrent fetches must be throttled by _ESPN_FETCH_SEM to at
        most 6 in-flight — the single guardrail against bursting ESPN."""
        import evmax.agents.cleanup.resolver as rmod

        in_flight = 0
        max_in_flight = 0

        async def _get(url, params=None):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.01)  # hold the slot so overlap can build
            in_flight -= 1
            return self._resp({"events": []})

        client = MagicMock()
        client.get = _get

        async def _run():
            await asyncio.gather(*(
                rmod._fetch_espn_scores(client, "basketball", "nba", f"202603{i:02d}")
                for i in range(20)
            ))

        asyncio.run(_run())
        assert max_in_flight <= 6, f"burst of {max_in_flight} exceeded the semaphore"
        assert max_in_flight > 1, "sanity: fetches should actually overlap"

    def test_fetch_survives_multiple_event_loops(self):
        """`evmax cleanup resolve` calls asyncio.run() twice (resolve phase, then
        the model-update hook) — two loops. A single module-level asyncio.Semaphore
        binds to the first loop and raises 'bound to a different event loop' on the
        second, silently killing every fetch in the model-update phase. Regression:
        both loops must succeed."""
        import evmax.agents.cleanup.resolver as rmod

        async def _one():
            client = AsyncMock()
            client.get.return_value = self._resp(_build_espn_response([]))
            return await rmod._fetch_espn_scores(client, "basketball", "nba", "20260319")

        assert asyncio.run(_one()) == []
        assert asyncio.run(_one()) == []  # pre-fix: RuntimeError here

    def test_cache_hit_skips_second_fetch(self):
        import evmax.agents.cleanup.resolver as rmod

        client = AsyncMock()
        client.get.return_value = self._resp(
            _build_espn_response([_build_espn_event("A", 1, "B", 0)])
        )
        cache: dict = {}
        r1 = asyncio.run(
            rmod._fetch_espn_scores(client, "basketball", "nba", "20260319", cache=cache)
        )
        r2 = asyncio.run(
            rmod._fetch_espn_scores(client, "basketball", "nba", "20260319", cache=cache)
        )
        assert client.get.call_count == 1, "cache hit must not re-fetch"
        assert r1 == r2
        assert len(cache) == 1

    def test_cache_different_date_fetches_again(self):
        import evmax.agents.cleanup.resolver as rmod

        client = AsyncMock()
        client.get.return_value = self._resp(_build_espn_response([]))
        cache: dict = {}
        asyncio.run(rmod._fetch_espn_scores(client, "basketball", "nba", "20260319", cache=cache))
        asyncio.run(rmod._fetch_espn_scores(client, "basketball", "nba", "20260320", cache=cache))
        assert client.get.call_count == 2
        assert len(cache) == 2

    def test_cache_key_includes_extra_params(self):
        """Same sport/league/date but different extra_params are distinct keys."""
        import evmax.agents.cleanup.resolver as rmod

        client = AsyncMock()
        client.get.return_value = self._resp(_build_espn_response([]))
        cache: dict = {}
        asyncio.run(rmod._fetch_espn_scores(
            client, "basketball", "ncaab", "20260319", extra_params={"groups": "50"}, cache=cache))
        asyncio.run(rmod._fetch_espn_scores(
            client, "basketball", "ncaab", "20260319", extra_params={"groups": "1"}, cache=cache))
        assert client.get.call_count == 2
        assert len(cache) == 2

    def test_cache_none_bypasses(self):
        import evmax.agents.cleanup.resolver as rmod

        client = AsyncMock()
        client.get.return_value = self._resp(_build_espn_response([]))
        asyncio.run(rmod._fetch_espn_scores(client, "basketball", "nba", "20260319", cache=None))
        asyncio.run(rmod._fetch_espn_scores(client, "basketball", "nba", "20260319", cache=None))
        assert client.get.call_count == 2, "cache=None must never memoize"

    def test_fetch_completed_scores_soccer_gathers_every_league(self):
        """The soccer path must fetch every ESPN league for the sector and
        combine them — concurrently, but result-equivalent to the old loop."""
        import evmax.agents.cleanup.resolver as rmod

        leagues = rmod.ESPN_SOCCER_LIKE_LEAGUES["soccer"]

        async def _fake(client, sport, league, espn_date, extra_params=None, cache=None):
            return [{"league": league}]

        with patch.object(rmod, "_fetch_espn_scores", side_effect=_fake):
            out = asyncio.run(rmod.fetch_completed_scores("soccer", date(2026, 6, 9)))

        assert len(out) == len(leagues)
        assert {r["league"] for r in out} == set(leagues)


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

    def test_polymarket_us_rows_are_dropped(self):
        # A PolyUS market_id is not a Kalshi ticker — it must never reach the
        # Kalshi API (garbage 404 lookups). With only PolyUS rows the resolver
        # returns empty without even opening a client.
        preds = [{"market_id": "polymarket_us:aec-atp-a-b-2026-07-08:abc"}]
        with patch("evmax.clients.kalshi.KalshiClient", lambda: _FakeKalshiClient({})):
            out, voided = asyncio.run(_resolve_via_kalshi(preds))
        assert out == {} and voided == set()


# ---------------------------------------------------------------------------
# Polymarket US settlement resolution
# ---------------------------------------------------------------------------
class _FakePolymarketUSClient:
    """Async-context stub for PolymarketUSClient settlement lookups."""

    def __init__(self, settlements: dict, sides: dict | None = None):
        self._settlements = settlements
        self._sides = sides or {}
        self.settlement_calls: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get_market_settlement(self, slug: str):
        self.settlement_calls.append(slug)
        return self._settlements.get(slug)

    async def get_market_sides(self, slug: str):
        return self._sides.get(slug, [])


def _pm_sides(long_abbrev: str, short_abbrev: str) -> list[dict]:
    return [
        {"long": True, "team": {"abbreviation": long_abbrev}, "description": "A"},
        {"long": False, "team": {"abbreviation": short_abbrev}, "description": "B"},
    ]


class TestSplitPolymarketUsId:
    def test_moneyline_side_suffix(self):
        slug, side = _split_polymarket_us_id("polymarket_us:aec-wnba-min-conn-2026-07-08:conn")
        assert slug == "aec-wnba-min-conn-2026-07-08"
        assert side == "conn"

    def test_bare_slug(self):
        slug, side = _split_polymarket_us_id("polymarket_us:aec-epl-ars-che-2026-08-01")
        assert slug == "aec-epl-ars-che-2026-08-01"
        assert side is None

    def test_is_polymarket_us_row(self):
        assert _is_polymarket_us_row({"market_id": "polymarket_us:x:y"})
        assert not _is_polymarket_us_row({"market_id": "kalshi:KXATPMATCH-X"})
        assert not _is_polymarket_us_row({"market_id": None})


class TestPolyusBetIsLong:
    def test_side_key_matches_long(self):
        assert _polyus_bet_is_long(_pm_sides("car", "gb"), "car", None) is True

    def test_side_key_matches_short(self):
        assert _polyus_bet_is_long(_pm_sides("car", "gb"), "gb", None) is False

    def test_side_key_case_insensitive(self):
        assert _polyus_bet_is_long(_pm_sides("CAR", "gb"), "car", None) is True

    def test_ab_fallback(self):
        # sides without abbreviations use the "a" (long) / "b" (short) ids
        sides = [{"long": True, "team": {}}, {"long": False, "team": {}}]
        assert _polyus_bet_is_long(sides, "a", None) is True
        assert _polyus_bet_is_long(sides, "b", None) is False

    def test_unknown_side_key_returns_none(self):
        assert _polyus_bet_is_long(_pm_sides("car", "gb"), "nyj", None) is None

    def test_total_over_maps_by_description(self):
        sides = [
            {"long": False, "description": "Over"},
            {"long": True, "description": "Under"},
        ]
        assert _polyus_bet_is_long(sides, None, "over") is False

    def test_bare_slug_defaults_to_long(self):
        # drawable_outcome / spread markets emit YES = long by construction
        assert _polyus_bet_is_long([], None, "arsenal") is True


class TestResolveViaPolymarketUs:
    SLUG = "aec-atp-pla-plb-2026-07-08"

    def _run(self, preds, settlements, sides=None):
        fake = _FakePolymarketUSClient(settlements, sides)
        with patch(
            "evmax.clients.polymarket_us.PolymarketUSClient", lambda: fake
        ):
            out, voided = asyncio.run(_resolve_via_polymarket_us(preds))
        return out, voided, fake

    def test_long_side_win_and_loss(self):
        preds = [
            {"market_id": f"polymarket_us:{self.SLUG}:pla", "yes_team": "Player A"},
            {"market_id": f"polymarket_us:{self.SLUG}:plb", "yes_team": "Player B"},
        ]
        out, voided, _ = self._run(
            preds, {self.SLUG: 1.0}, {self.SLUG: _pm_sides("pla", "plb")}
        )
        # settlement 1 → long (pla) won → YES on pla wins, YES on plb loses
        assert out[f"polymarket_us:{self.SLUG}:pla"] == 1
        assert out[f"polymarket_us:{self.SLUG}:plb"] == 0
        assert voided == set()

    def test_short_side_win(self):
        preds = [
            {"market_id": f"polymarket_us:{self.SLUG}:plb", "yes_team": "Player B"},
        ]
        out, _, _ = self._run(
            preds, {self.SLUG: 0.0}, {self.SLUG: _pm_sides("pla", "plb")}
        )
        assert out[f"polymarket_us:{self.SLUG}:plb"] == 1

    def test_fractional_settlement_voids(self):
        # tie 50-50 / cancellation at last-traded / walkover — no binary outcome
        preds = [
            {"market_id": f"polymarket_us:{self.SLUG}:pla", "yes_team": "Player A"},
        ]
        out, voided, _ = self._run(preds, {self.SLUG: 0.5})
        assert out == {}
        assert voided == {f"polymarket_us:{self.SLUG}:pla"}

    def test_unsettled_returns_none(self):
        preds = [
            {"market_id": f"polymarket_us:{self.SLUG}:pla", "yes_team": "Player A"},
        ]
        out, voided, _ = self._run(preds, {})
        assert out[f"polymarket_us:{self.SLUG}:pla"] is None
        assert voided == set()

    def test_unmappable_side_left_open(self):
        # settlement exists but the side suffix matches neither abbreviation —
        # leave unresolved rather than guess a direction
        preds = [
            {"market_id": f"polymarket_us:{self.SLUG}:xyz", "yes_team": "Player A"},
        ]
        out, voided, _ = self._run(
            preds, {self.SLUG: 1.0}, {self.SLUG: _pm_sides("pla", "plb")}
        )
        assert out[f"polymarket_us:{self.SLUG}:xyz"] is None
        assert voided == set()

    def test_no_side_suffix_skipped(self):
        # synthesized ":no" spread/total rows resolve via scores, not settlement
        preds = [{"market_id": f"polymarket_us:{self.SLUG}:no", "yes_team": "under"}]
        out, voided, fake = self._run(preds, {self.SLUG: 1.0})
        assert out[f"polymarket_us:{self.SLUG}:no"] is None
        assert fake.settlement_calls == []

    def test_kalshi_rows_ignored(self):
        preds = [
            {"market_id": "kalshi:KXATPMATCH-X", "yes_team": "Player A"},
            {"market_id": f"polymarket_us:{self.SLUG}:pla", "yes_team": "Player A"},
        ]
        out, _, _ = self._run(
            preds, {self.SLUG: 1.0}, {self.SLUG: _pm_sides("pla", "plb")}
        )
        assert "kalshi:KXATPMATCH-X" not in out
        assert out[f"polymarket_us:{self.SLUG}:pla"] == 1


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


# ---------------------------------------------------------------------------
# backfill_clv — NO-side ticker handling + placement-aware close anchor.
#
# NO-side bets are synthesized with a ":no" market_id suffix and store the
# NO-side ask in kalshi_yes_price. Before the fix, removeprefix("kalshi:") left
# the ticker as "KX...:no" which never matched the archive's clean ticker, so
# EVERY no-side bet got NULL kalshi_clv_pct. The fix strips ":no" and flips the
# archived YES close to the NO side (1 - yes_close).
# ---------------------------------------------------------------------------

class TestBackfillClvNoSide:
    class _NoCloseConn:
        def __init__(self, conn):
            self._conn = conn
        def __getattr__(self, item):
            return getattr(self._conn, item)
        def close(self):
            pass

    def _make_predictions_db(self, tmp_path, monkeypatch):
        # Use the real migrated get_connection so the CLV columns
        # (kalshi_clv_pct, pinnacle_drift_pct, placed_at, ...) exist — they're
        # added by ALTER migrations, not the base SCHEMA.
        import evmax.agents.cleanup.db as dbmod
        monkeypatch.setattr(dbmod, "DB_PATH", tmp_path / "predictions.db")
        return self._NoCloseConn(dbmod.get_connection())

    def _seed(self, conn, market_id, yes_team, market_type, entry, line, event_id,
              event_date="2026-05-25", placed=0, placed_at=None, placed_price=None):
        conn.execute(
            """INSERT INTO ev_predictions
               (scan_date, market_id, event_id, sector, yes_team, market_type,
                event_title, event_date, kalshi_yes_price, sharp_true_prob,
                blended_true_prob, ev_pct, kelly_fraction, bankroll_used, line,
                placed, placed_at, placed_price)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (event_date, market_id, event_id, "baseball", yes_team, market_type,
             "Yankees vs Red Sox", event_date, entry, 0.55, 0.55, 0.05, 0.01, 500.0, line,
             placed, placed_at, placed_price),
        )
        conn.execute(
            """INSERT INTO ev_outcomes
               (market_id, event_id, event_date, sector, yes_team, outcome)
               VALUES (?,?,?,?,?,?)""",
            (market_id, event_id, event_date, "baseball", yes_team, 1),
        )
        conn.commit()

    def _seed_archive(self, db_path, event_id, yes_close, fetched_at, tipoff):
        from evmax.archiver import DataArchiver
        from evmax.models.odds import SharpBook, SharpOdds
        archiver = DataArchiver()
        # Pinnacle row supplies the tipoff anchor for the close window.
        archiver.open_session("so", ["baseball"], "test")
        archiver.archive_sharp_odds("so", "baseball", [SharpOdds(
            event_id=event_id, book=SharpBook.pinnacle, sector="baseball",
            outcome_a_label="yankees", outcome_b_label="redsox",
            outcome_a_decimal=1.9, outcome_b_decimal=1.9,
            true_prob_a=0.5, true_prob_b=0.5, margin=0.04,
            event_date=tipoff, fetched_at=tipoff - timedelta(hours=4),
        )])
        archiver.archive_kalshi_snapshot(
            "k1", "baseball",
            [{"ticker": "KXMLBGAME-T", "yes_price": yes_close,
              "event_id": event_id, "market_type": "total"}],
            fetched_at=fetched_at,
        )

    def test_no_side_strips_suffix_and_flips_close(self, tmp_path, monkeypatch):
        from datetime import datetime, timezone
        from evmax.agents.cleanup import resolver
        import evmax.archiver as archiver_mod

        monkeypatch.setattr(archiver_mod, "DB_PATH", tmp_path / "archive.db")
        event_id = "baseball::2026-05-25::yankees_vs_redsox::total::8.5"
        tip = datetime(2026, 5, 25, 23, 0, tzinfo=timezone.utc)
        # YES close 0.30 → NO close 0.70; NO entry 0.60 → CLV = +10.0pp.
        self._seed_archive(tmp_path / "archive.db", event_id, 0.30, tip - timedelta(hours=1), tip)

        conn = self._make_predictions_db(tmp_path, monkeypatch)
        self._seed(conn, "kalshi:KXMLBGAME-T:no", "under", "total", 0.60, 8.5, event_id)

        with patch.object(resolver, "get_connection", return_value=conn):
            resolver.backfill_clv()

        row = conn.execute(
            "SELECT kalshi_clv_pct FROM ev_predictions WHERE market_id = ?",
            ("kalshi:KXMLBGAME-T:no",),
        ).fetchone()
        assert row["kalshi_clv_pct"] == pytest.approx(10.0)

    def test_no_side_was_null_without_suffix_strip(self, tmp_path, monkeypatch):
        """The YES ticker exists in the archive; a NO-side bet must resolve to a
        (flipped) CLV rather than NULL — the regression this fix closes."""
        from datetime import datetime, timezone
        from evmax.agents.cleanup import resolver
        import evmax.archiver as archiver_mod

        monkeypatch.setattr(archiver_mod, "DB_PATH", tmp_path / "archive.db")
        event_id = "baseball::2026-05-25::yankees_vs_redsox::total::8.5"
        tip = datetime(2026, 5, 25, 23, 0, tzinfo=timezone.utc)
        self._seed_archive(tmp_path / "archive.db", event_id, 0.30, tip - timedelta(hours=1), tip)

        conn = self._make_predictions_db(tmp_path, monkeypatch)
        self._seed(conn, "kalshi:KXMLBGAME-T:no", "under", "total", 0.60, 8.5, event_id)
        with patch.object(resolver, "get_connection", return_value=conn):
            resolver.backfill_clv()

        row = conn.execute(
            "SELECT kalshi_clv_pct FROM ev_predictions WHERE market_id = ?",
            ("kalshi:KXMLBGAME-T:no",),
        ).fetchone()
        assert row["kalshi_clv_pct"] is not None

    def test_placed_at_anchors_close_after_fill(self, tmp_path, monkeypatch):
        """A placed YES bet whose fill is AFTER the only archived snapshot gets no
        forward close (NULL), instead of a backward CLV against a pre-entry price."""
        from datetime import datetime, timezone
        from evmax.agents.cleanup import resolver
        import evmax.archiver as archiver_mod

        monkeypatch.setattr(archiver_mod, "DB_PATH", tmp_path / "archive.db")
        event_id = "baseball::2026-05-25::yankees_vs_redsox::ml"
        tip = datetime(2026, 5, 25, 23, 0, tzinfo=timezone.utc)
        # Only snapshot is at tip-1h; fill is at tip-30m (after it) → no forward close.
        self._seed_archive(tmp_path / "archive.db", event_id, 0.40, tip - timedelta(hours=1), tip)
        placed_at = (tip - timedelta(minutes=30)).isoformat()

        conn = self._make_predictions_db(tmp_path, monkeypatch)
        self._seed(conn, "kalshi:KXMLBGAME-Y", "yankees", "moneyline", 0.45, None, event_id,
                   placed=1, placed_at=placed_at, placed_price=0.45)
        with patch.object(resolver, "get_connection", return_value=conn):
            resolver.backfill_clv()

        row = conn.execute(
            "SELECT kalshi_clv_pct FROM ev_predictions WHERE market_id = ?",
            ("kalshi:KXMLBGAME-Y",),
        ).fetchone()
        assert row["kalshi_clv_pct"] is None


# ---------------------------------------------------------------------------
# close_lookup_ticker — venue-aware market_id → archive ticker mapping.
#
# Kalshi ids lose their prefix (the archive stores raw tickers); Polymarket US
# ids KEEP it (watch-closes archives PolyUS snapshots under the full prefixed
# id). Before this helper, removeprefix("kalshi:") was a no-op on PolyUS ids
# and the archive lookup never hit — every PolyUS bet silently got NULL
# kalshi_clv_pct, the same failure mode as the old NO-side suffix bug.
# ---------------------------------------------------------------------------

class TestCloseLookupTicker:
    def test_kalshi_plain(self):
        from evmax.agents.cleanup.resolver import close_lookup_ticker
        assert close_lookup_ticker("kalshi:KXWNBAGAME-X") == ("KXWNBAGAME-X", False)

    def test_kalshi_no_side(self):
        from evmax.agents.cleanup.resolver import close_lookup_ticker
        assert close_lookup_ticker("kalshi:KXMLBGAME-T:no") == ("KXMLBGAME-T", True)

    def test_polymarket_us_keeps_prefix(self):
        from evmax.agents.cleanup.resolver import close_lookup_ticker
        assert close_lookup_ticker("polymarket_us:lva-sea-2026-07-08:LVA") == (
            "polymarket_us:lva-sea-2026-07-08:LVA", False,
        )

    def test_polymarket_us_no_side(self):
        from evmax.agents.cleanup.resolver import close_lookup_ticker
        assert close_lookup_ticker("polymarket_us:slug-total:no") == (
            "polymarket_us:slug-total", True,
        )

    def test_empty_and_none(self):
        from evmax.agents.cleanup.resolver import close_lookup_ticker
        assert close_lookup_ticker(None) == (None, False)
        assert close_lookup_ticker("") == (None, False)


class TestBackfillClvPolymarketUS:
    """PolyUS rows must get venue CLV from snapshots archived under their
    prefixed id. Borrows the NO-side class's seed helpers (not inherited,
    so the parent's tests aren't collected twice)."""

    _NoCloseConn = TestBackfillClvNoSide._NoCloseConn
    _make_predictions_db = TestBackfillClvNoSide._make_predictions_db
    _seed = TestBackfillClvNoSide._seed

    MID = "polymarket_us:nyy-bos-2026-05-25:NYY"

    def _seed_pmus_archive(self, tmp_path, event_id, yes_close, fetched_at, tipoff):
        from evmax.archiver import DataArchiver
        from evmax.models.odds import SharpBook, SharpOdds
        archiver = DataArchiver()
        archiver.open_session("so", ["baseball"], "test")
        archiver.archive_sharp_odds("so", "baseball", [SharpOdds(
            event_id=event_id, book=SharpBook.pinnacle, sector="baseball",
            outcome_a_label="yankees", outcome_b_label="redsox",
            outcome_a_decimal=1.9, outcome_b_decimal=1.9,
            true_prob_a=0.5, true_prob_b=0.5, margin=0.04,
            event_date=tipoff, fetched_at=tipoff - timedelta(hours=4),
        )])
        # watch-closes archives PolyUS snapshots under the PREFIXED id.
        archiver.archive_kalshi_snapshot(
            "pmus1", "baseball",
            [{"ticker": self.MID, "yes_price": yes_close, "event_id": event_id}],
            fetched_at=fetched_at,
        )

    def test_polymarket_us_row_gets_clv(self, tmp_path, monkeypatch):
        from datetime import datetime, timezone
        from evmax.agents.cleanup import resolver
        import evmax.archiver as archiver_mod

        monkeypatch.setattr(archiver_mod, "DB_PATH", tmp_path / "archive.db")
        event_id = "baseball::2026-05-25::yankees_vs_redsox"
        tip = datetime(2026, 5, 25, 23, 0, tzinfo=timezone.utc)
        # entry 0.55 → close 0.60 = +5.0pp CLV
        self._seed_pmus_archive(tmp_path, event_id, 0.60, tip - timedelta(hours=1), tip)

        conn = self._make_predictions_db(tmp_path, monkeypatch)
        self._seed(conn, self.MID, "yankees", "moneyline", 0.55, None, event_id)

        with patch.object(resolver, "get_connection", return_value=conn):
            resolver.backfill_clv()

        row = conn.execute(
            "SELECT kalshi_clv_pct FROM ev_predictions WHERE market_id = ?",
            (self.MID,),
        ).fetchone()
        assert row["kalshi_clv_pct"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# baseball_props resolution (MLB Stats API boxscore)
# ---------------------------------------------------------------------------

class TestMlbOutsFromInningsPitched:
    def test_whole_innings(self):
        assert _mlb_outs_from_innings_pitched("6") == 18.0

    def test_one_out_fraction(self):
        assert _mlb_outs_from_innings_pitched("6.1") == 19.0

    def test_two_out_fraction(self):
        assert _mlb_outs_from_innings_pitched("6.2") == 20.0

    def test_none_returns_none(self):
        assert _mlb_outs_from_innings_pitched(None) is None

    def test_garbage_returns_none(self):
        assert _mlb_outs_from_innings_pitched("--") is None


def _mlb_boxscore(pitcher_line=None, batter_line=None):
    players = {}
    if pitcher_line is not None:
        players["ID1"] = {
            "person": {"fullName": pitcher_line["name"]},
            "stats": {"pitching": pitcher_line["stats"], "batting": {}},
        }
    if batter_line is not None:
        players["ID2"] = {
            "person": {"fullName": batter_line["name"]},
            "stats": {"batting": batter_line["stats"], "pitching": {}},
        }
    return {"teams": {"home": {"players": players}, "away": {"players": {}}}}


class TestResolveBaseballPropObservations:
    def _make_conn(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE prop_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_date TEXT,
                sector TEXT,
                player_name TEXT,
                stat_type TEXT,
                line REAL,
                event_id TEXT,
                actual_value REAL,
                outcome INTEGER
            )
        """)
        return conn

    def _seed_row(self, conn, player, stat_type, line, event_id="baseball::2026-07-15::yankees_vs_redsox"):
        conn.execute(
            """INSERT INTO prop_observations
               (scan_date, sector, player_name, stat_type, line, event_id)
               VALUES ('2026-07-15', 'baseball', ?, ?, ?, ?)""",
            (player, stat_type, line, event_id),
        )
        conn.commit()
        return conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def _fake_client(self, schedule_payload, boxscore_payload):
        client = MagicMock()

        def _get(url, params=None, **kwargs):
            resp = MagicMock()
            resp.raise_for_status.return_value = None
            if "schedule" in url:
                resp.json.return_value = schedule_payload
            else:
                resp.json.return_value = boxscore_payload
            return resp

        client.get.side_effect = _get
        return client

    def test_pitcher_strikeouts_and_outs_resolve(self):
        conn = self._make_conn()
        self._seed_row(conn, "Gerrit Cole", "strikeouts", 6.5)
        self._seed_row(conn, "Gerrit Cole", "pitching_outs", 18.5)

        schedule = {"dates": [{"games": [{"gamePk": 1, "status": {"abstractGameState": "Final"}}]}]}
        box = _mlb_boxscore(
            pitcher_line={"name": "Gerrit Cole", "stats": {"strikeOuts": 8, "inningsPitched": "6.1"}}
        )
        fake = self._fake_client(schedule, box)
        with patch("evmax.agents.cleanup.resolver.httpx.Client", return_value=fake):
            n = _resolve_baseball_prop_observations(
                conn, {"2026-07-15": [dict(r) for r in conn.execute(
                    "SELECT id, scan_date, player_name, stat_type, line, event_id FROM prop_observations"
                ).fetchall()]},
            )

        assert n == 2
        rows = {r["stat_type"]: r for r in conn.execute("SELECT * FROM prop_observations").fetchall()}
        assert rows["strikeouts"]["outcome"] == 1  # 8 >= 6.5
        assert rows["strikeouts"]["actual_value"] == 8.0
        assert rows["pitching_outs"]["outcome"] == 1  # 19 >= 18.5
        assert rows["pitching_outs"]["actual_value"] == 19.0

    def test_batter_total_bases_and_hits_runs_rbis(self):
        conn = self._make_conn()
        self._seed_row(conn, "Aaron Judge", "total_bases", 2.5)
        self._seed_row(conn, "Aaron Judge", "hits_runs_rbis", 3.5)

        schedule = {"dates": [{"games": [{"gamePk": 1, "status": {"abstractGameState": "Final"}}]}]}
        box = _mlb_boxscore(
            batter_line={
                "name": "Aaron Judge",
                "stats": {"totalBases": 4, "homeRuns": 1, "hits": 2, "runs": 1, "rbi": 2},
            }
        )
        fake = self._fake_client(schedule, box)
        with patch("evmax.agents.cleanup.resolver.httpx.Client", return_value=fake):
            n = _resolve_baseball_prop_observations(
                conn, {"2026-07-15": [dict(r) for r in conn.execute(
                    "SELECT id, scan_date, player_name, stat_type, line, event_id FROM prop_observations"
                ).fetchall()]},
            )

        assert n == 2
        rows = {r["stat_type"]: r for r in conn.execute("SELECT * FROM prop_observations").fetchall()}
        assert rows["total_bases"]["actual_value"] == 4.0
        assert rows["total_bases"]["outcome"] == 1
        # hits(2) + runs(1) + rbi(2) = 5 >= 3.5
        assert rows["hits_runs_rbis"]["actual_value"] == 5.0
        assert rows["hits_runs_rbis"]["outcome"] == 1

    def test_unfinished_games_skipped(self):
        conn = self._make_conn()
        self._seed_row(conn, "Gerrit Cole", "strikeouts", 6.5)

        schedule = {"dates": [{"games": [{"gamePk": 1, "status": {"abstractGameState": "Live"}}]}]}
        fake = self._fake_client(schedule, {})
        with patch("evmax.agents.cleanup.resolver.httpx.Client", return_value=fake):
            n = _resolve_baseball_prop_observations(
                conn, {"2026-07-15": [dict(r) for r in conn.execute(
                    "SELECT id, scan_date, player_name, stat_type, line, event_id FROM prop_observations"
                ).fetchall()]},
            )
        assert n == 0
        assert fake.get.call_count == 1  # only the schedule call, no boxscore fetch

    def test_player_not_in_boxscore_left_unresolved(self):
        conn = self._make_conn()
        self._seed_row(conn, "Nobody Special", "strikeouts", 6.5)

        schedule = {"dates": [{"games": [{"gamePk": 1, "status": {"abstractGameState": "Final"}}]}]}
        box = _mlb_boxscore(
            pitcher_line={"name": "Gerrit Cole", "stats": {"strikeOuts": 8, "inningsPitched": "6.1"}}
        )
        fake = self._fake_client(schedule, box)
        with patch("evmax.agents.cleanup.resolver.httpx.Client", return_value=fake):
            n = _resolve_baseball_prop_observations(
                conn, {"2026-07-15": [dict(r) for r in conn.execute(
                    "SELECT id, scan_date, player_name, stat_type, line, event_id FROM prop_observations"
                ).fetchall()]},
            )
        assert n == 0
        row = conn.execute("SELECT outcome FROM prop_observations").fetchone()
        assert row["outcome"] is None


class TestFetchJsonConcurrent:
    """_fetch_json_concurrent: order-preserving bounded thread-pool map."""

    def test_empty_input_returns_empty_no_pool(self):
        calls = []
        assert _fetch_json_concurrent(lambda x: calls.append(x), []) == []
        assert calls == []

    def test_single_item_runs_inline(self):
        assert _fetch_json_concurrent(lambda x: x * 2, [21]) == [42]

    def test_preserves_order_across_workers(self):
        # Even with out-of-order completion, results align to input order.
        items = list(range(20))
        out = _fetch_json_concurrent(lambda x: x * x, items)
        assert out == [x * x for x in items]

    def test_failed_fetch_none_is_passed_through(self):
        # fetch_fn is expected to swallow its own errors and return None.
        def fetch(x):
            return None if x % 2 else x
        assert _fetch_json_concurrent(fetch, [0, 1, 2, 3]) == [0, None, 2, None]


class TestResolvePropObservationsWindow:
    """_resolve_prop_observations only touches games within the lookback window.

    A prop still pending long after its game finished is permanently
    unresolvable; re-fetching its slate every run is what made resolve slow.
    """

    def _make_conn(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            """CREATE TABLE prop_observations (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   scan_date TEXT, sector TEXT, player_name TEXT,
                   stat_type TEXT, line REAL, event_id TEXT,
                   actual_value REAL, outcome INTEGER)"""
        )
        return conn

    def _seed(self, conn, *, scan_date, player, stat_type, line, event_id, sector="nba"):
        conn.execute(
            """INSERT INTO prop_observations
               (scan_date, sector, player_name, stat_type, line, event_id)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (scan_date, sector, player, stat_type, line, event_id),
        )
        conn.commit()

    def _fake_nba_client(self, recording):
        """Fake httpx.Client that records the ESPN scoreboard dates it is asked
        for and returns a one-completed-game slate for any date."""
        client = MagicMock()

        def _get(url, params=None, **kwargs):
            resp = MagicMock()
            resp.raise_for_status.return_value = None
            if "scoreboard" in url:
                recording.append(params.get("dates"))
                resp.json.return_value = {
                    "events": [
                        {"id": "401", "competitions": [
                            {"status": {"type": {"completed": True}}}]}
                    ]
                }
            else:  # summary/boxscore
                resp.json.return_value = {
                    "boxscore": {"players": [
                        {"statistics": [
                            {"labels": ["PTS", "REB"], "athletes": [
                                {"athlete": {"displayName": "Recent Player"},
                                 "stats": ["30", "10"]}
                            ]}
                        ]}
                    ]}
                }
            return resp

        client.get.side_effect = _get
        return client

    def test_stale_game_skipped_recent_resolved(self):
        conn = self._make_conn()
        target = date(2026, 7, 15)
        window_lo = target - timedelta(days=PROP_RESOLVE_LOOKBACK_DAYS)  # 2026-07-12
        stale_gd = (window_lo - timedelta(days=4)).isoformat()          # 2026-07-08

        # Recent prop: game inside window → should resolve.
        self._seed(
            conn, scan_date=target.isoformat(), player="Recent Player",
            stat_type="points", line=25.0,
            event_id=f"nba::{target.isoformat()}::a_vs_b::prop::recent_player::points::25.0",
        )
        # Stale prop: scan_date passes the coarse SQL floor, but the GAME is
        # before the window → must be skipped and never fetched.
        self._seed(
            conn, scan_date=stale_gd, player="Recent Player",
            stat_type="points", line=25.0,
            event_id=f"nba::{stale_gd}::a_vs_b::prop::recent_player::points::25.0",
        )

        recording: list = []
        fake = self._fake_nba_client(recording)
        with patch("evmax.agents.cleanup.resolver.httpx.Client", return_value=fake):
            n = _resolve_prop_observations(conn, target)

        assert n == 1
        # Only the in-window date's scoreboard was fetched.
        assert recording == [target.isoformat().replace("-", "")]

        rows = conn.execute(
            "SELECT scan_date, outcome FROM prop_observations ORDER BY scan_date"
        ).fetchall()
        by_date = {r["scan_date"]: r["outcome"] for r in rows}
        assert by_date[target.isoformat()] == 1     # resolved
        assert by_date[stale_gd] is None            # left pending, not fetched

    def test_ancient_rows_below_sql_floor_not_scanned(self):
        conn = self._make_conn()
        target = date(2026, 7, 15)
        # scan_date far below the (window_lo - 7d) floor: never selected at all.
        ancient = "2026-01-01"
        self._seed(
            conn, scan_date=ancient, player="Old Player",
            stat_type="points", line=10.0,
            event_id=f"nba::{ancient}::a_vs_b::prop::old_player::points::10.0",
        )
        recording: list = []
        fake = self._fake_nba_client(recording)
        with patch("evmax.agents.cleanup.resolver.httpx.Client", return_value=fake):
            n = _resolve_prop_observations(conn, target)
        assert n == 0
        assert recording == []  # no network fetch at all
        assert fake.get.call_count == 0


# ---------------------------------------------------------------------------
# ESPN User-Agent — regression guard (2026-08-05)
#
# ESPN's public-API WAF began 403-ing our identifying "evmax-*" User-Agents on
# every scoreboard/summary request. `_fetch_espn_scores` swallows the 403 and
# returns [], so resolution silently matched nothing: `evmax cleanup resolve`
# reported 0 resolved / all-unmatched for otherwise-final games. The fix routes
# every ESPN client through the neutral `_ESPN_HTTP_UA` tool string.
# ---------------------------------------------------------------------------
class TestEspnUserAgent:
    def test_ua_constant_is_neutral_tool_ua(self):
        from evmax.agents.cleanup.resolver import _ESPN_HTTP_UA
        assert _ESPN_HTTP_UA, "UA must be non-empty"
        # Our identifying prefix is exactly what ESPN blocklisted.
        assert not _ESPN_HTTP_UA.lower().startswith("evmax")
        # A browser-impersonator string is ALSO blocked (scraper heuristic).
        assert "mozilla" not in _ESPN_HTTP_UA.lower()

    def test_fetch_completed_scores_sends_neutral_ua(self):
        """The async ESPN client must construct with the neutral UA header.

        Fails before the fix: fetch_completed_scores built its AsyncClient with
        headers={"User-Agent": "evmax-update/1.0"}.
        """
        from evmax.agents.cleanup import resolver as R

        captured: dict = {}

        class _FakeResp:
            def raise_for_status(self):
                return None

            def json(self):
                return {"events": []}

        class _FakeAsyncClient:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def get(self, url, params=None, **kw):
                return _FakeResp()

        with patch.object(R.httpx, "AsyncClient", _FakeAsyncClient):
            asyncio.run(R.fetch_completed_scores("nba", date(2026, 8, 4)))

        ua = captured.get("headers", {}).get("User-Agent")
        assert ua == R._ESPN_HTTP_UA
        assert not ua.lower().startswith("evmax")

    def test_no_blocklisted_ua_literals_in_source(self):
        """Guard against reintroducing any of the three blocklisted UA strings."""
        import inspect
        from evmax.agents.cleanup import resolver as R

        src = inspect.getsource(R)
        for blocked in ("evmax-live/1.0", "evmax-update/1.0", "evmax-cleanup/1.0"):
            assert blocked not in src, f"blocklisted ESPN UA reintroduced: {blocked}"
