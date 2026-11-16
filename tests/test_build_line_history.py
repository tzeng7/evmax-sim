"""Tests for the closing-line collector's parsers (scripts/build_line_history.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "build_line_history", Path(__file__).resolve().parent.parent / "scripts" / "build_line_history.py")
blh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(blh)


def _event(eid, home, away, hs, as_, completed=True, period=4):
    return {"id": eid, "season": {"type": 2}, "competitions": [{
        "status": {"period": period, "type": {"completed": completed}},
        "competitors": [
            {"homeAway": "home", "score": hs, "team": {"displayName": home}},
            {"homeAway": "away", "score": as_, "team": {"displayName": away}},
        ]}]}


def test_parse_scoreboard_keeps_completed_games_with_scores():
    doc = {"events": [
        _event("1", "Philadelphia 76ers", "New York Knicks", "119", "125", period=5),
        _event("2", "A", "B", "10", "8", completed=False),
        _event("3", "A", "B", None, "8"),
    ]}
    (g,) = blh.parse_scoreboard(doc)
    assert g == {"event_id": "1", "home": "Philadelphia 76ers", "away": "New York Knicks",
                 "home_score": 119.0, "away_score": 125.0, "periods": 5, "season_type": 2}
    assert blh.parse_scoreboard({}) == []


def test_parse_odds_reads_the_home_handicap_and_skips_live_books():
    doc = {"items": [
        {"provider": {"name": "ESPN Bet - Live Odds"}, "spread": -1.5, "overUnder": 240.5},
        {"provider": {"name": "ESPN BET"}, "spread": 5.5, "overUnder": 219.5},
    ]}
    assert blh.parse_odds(doc) == (5.5, 219.5, "ESPN BET")


def test_parse_odds_ignores_placeholders():
    assert blh.parse_odds({"items": [{"provider": {"name": "numberfire"}, "spread": 0.0,
                                      "overUnder": 0.0}]}) == (None, None, None)
    assert blh.parse_odds({"items": [{"provider": {"name": "x"}, "spread": "abc"}]}) == (
        None, None, None)
    assert blh.parse_odds(None) == (None, None, None)
    # a spread with no total is still a line
    assert blh.parse_odds({"items": [{"provider": {"name": "x"}, "spread": -3.5}]}) == (
        -3.5, None, "x")


def test_season_days_span_the_season_and_stop_before_today():
    days = blh.season_days(2024, (11, 1), (4, 10))
    assert days[0] == "20231101" and days[-1] == "20240410"
    future = blh.season_days(2999, (11, 1), (4, 10))
    assert future == []
