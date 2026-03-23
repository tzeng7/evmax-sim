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
    _slug_teams,
    _to_fuzz,
    _write_outcome,
    _fetch_espn_scores,
    FUZZY_THRESHOLD,
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


def _make_pred(event_id: str, yes_team: str) -> dict:
    return {"event_id": event_id, "yes_team": yes_team}


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

    def test_multiple_scores_correct_event_selected(self):
        """Only the matching game should be used, not unrelated ones."""
        scores = [
            _make_score("Lakers", 120, "Warriors", 110),
            _make_score("Detroit Pistons", 117, "Washington Wizards", 95),
            _make_score("Bulls", 98, "Celtics", 102),
        ]
        pred = _make_pred("nba::2026-03-19::pistons_vs_wizards", "pistons")
        assert _match_espn(pred, scores) == 1


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
