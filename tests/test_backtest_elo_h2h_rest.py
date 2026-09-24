"""scripts/backtest_elo_h2h_rest.py — the walk-forward that gates the Elo
H2H / rest read-key change (docs/elo-h2h-rest-eval.md).

Pins the pieces the verdict rests on: point-in-time rest measurement, the ET
calendar day (live rest is measured between ET days — the UTC stamp turns a
Sat-night → Sun-matinee pair into a phantom "0 days"), the variant toggles
sharing one rating trajectory, the blend-eligibility filter, and the stats.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from datetime import date
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "backtest_elo_h2h_rest.py"
_spec = importlib.util.spec_from_file_location("backtest_elo_h2h_rest", _PATH)
bt = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = bt  # dataclasses need the module registered
_spec.loader.exec_module(bt)


# --- ScheduleTracker ---------------------------------------------------------


def test_tracker_days_of_rest_is_point_in_time():
    t = bt.ScheduleTracker()
    assert t.days_of_rest("lakers", date(2026, 1, 5)) is None
    t.record("lakers", date(2026, 1, 3))
    assert t.days_of_rest("lakers", date(2026, 1, 4)) == 1      # back-to-back
    assert t.days_of_rest("lakers", date(2026, 1, 6)) == 3


def test_tracker_keeps_dates_sorted_when_recorded_out_of_order():
    t = bt.ScheduleTracker()
    t.record("x", date(2026, 1, 10))
    t.record("x", date(2026, 1, 4))
    assert t.days_of_rest("x", date(2026, 1, 11)) == 1        # latest game wins


def test_tracker_window_matches_live_cutoff_semantics():
    # Live: count prior records with date >= reference - days (the game being
    # predicted is not in form_state yet).
    t = bt.ScheduleTracker()
    for d in (date(2026, 3, 1), date(2026, 3, 4), date(2026, 3, 7), date(2026, 3, 8)):
        t.record("city", d)
    assert t.games_in_window("city", date(2026, 3, 8), 7) == 3   # 3/1..3/7; 3/8 itself excluded
    assert t.games_in_window("city", date(2026, 3, 9), 7) == 3   # 3/2..3/8
    assert t.games_in_window("nobody", date(2026, 3, 9), 7) == 0


# --- ESPN parse ----------------------------------------------------------------


def _event(stamp: str, *, completed=True, season_type=2, home="Boston Celtics", away="Los Angeles Lakers"):
    return {
        "date": stamp,
        "season": {"type": season_type},
        "competitions": [{
            "status": {"type": {"completed": completed}},
            "competitors": [
                {"homeAway": "home", "score": "110", "team": {"displayName": home}},
                {"homeAway": "away", "score": "99", "team": {"displayName": away}},
            ],
        }],
    }


def test_parse_espn_uses_the_et_calendar_day():
    # 00:30Z Jan 6 is 7:30 pm ET Jan 5 — the live rest layer's day.
    games = bt._parse_espn({"events": [_event("2024-01-06T00:30Z")]}, "nba")
    assert [g["date"] for g in games] == ["2024-01-05"]
    # A 3:30 pm ET matinee stays on its own day.
    games = bt._parse_espn({"events": [_event("2024-01-07T20:30Z")]}, "nba")
    assert [g["date"] for g in games] == ["2024-01-07"]


def test_parse_espn_drops_preseason_and_unfinished_games():
    events = [
        _event("2024-10-10T23:00Z", season_type=1),
        _event("2024-10-11T23:00Z", completed=False),
        _event("2024-10-25T23:00Z"),
    ]
    games = bt._parse_espn({"events": events}, "nba")
    assert len(games) == 1 and games[0]["hs"] == 110.0 and games[0]["as"] == 99.0


def test_season_labels():
    assert bt.season_of("nba", date(2025, 10, 22)) == 2025
    assert bt.season_of("nba", date(2026, 3, 1)) == 2025
    assert bt.season_of("wnba", date(2026, 5, 20)) == 2026
    assert bt.season_of("baseball", date(2026, 9, 1)) == 2026
    assert bt.season_of("soccer", date(2026, 1, 10)) == 2025


# --- replay agent toggles ------------------------------------------------------


def test_replay_agent_toggles_share_one_trajectory():
    t = bt.ScheduleTracker()
    agent = bt.make_replay_agent(t)
    for _ in range(4):
        agent.update("lakers", "celtics", 110, 100, "nba", event_date="2026-01-01")
    t.record("lakers", date(2026, 1, 9))          # lakers on a back-to-back on 1/10
    t.record("celtics", date(2026, 1, 7))         # celtics on 3 days
    ref = date(2026, 1, 10)
    probs = {}
    for v, (h2h, rest) in {"off": (0, 0), "h2h": (1, 0), "rest": (0, 1), "on": (1, 1)}.items():
        agent.use_h2h, agent.use_rest = bool(h2h), bool(rest)
        probs[v] = agent._win_probs("nba", "lakers", "celtics", ref)[0]
    assert probs["h2h"] > probs["off"]           # lakers won all 4 meetings
    assert probs["rest"] < probs["off"]          # lakers (B2B) vs rested celtics
    assert probs["on"] != probs["off"]
    # toggling never touched the ratings
    agent.use_h2h = agent.use_rest = False
    assert agent._win_probs("nba", "lakers", "celtics", ref)[0] == probs["off"]


def test_replay_scores_only_blend_eligible_games():
    # Elo with < LOW_DATA_THRESHOLD games on a side is gated out of the live
    # blend; the replay must not score it (NCAAB's one-off non-D1 opponents
    # otherwise let the rest layer pose as a "has-a-rating" signal).
    games = [bt.Game(date(2025, 11, 1 + i), "a", "b", 1, 0, season=2025) for i in range(3)]
    games += [bt.Game(date(2025, 11, 10 + i), "a", "b", 1, 0, season=2025) for i in range(6)]
    cfg = {"eval": [2025], "holdout": 2026}
    saved = bt.SECTORS.get("toy")
    bt.SECTORS["toy"] = cfg
    try:
        recs = bt.replay("toy", games)
    finally:
        if saved is None:
            bt.SECTORS.pop("toy")
    n_eligible = len(games) - bt.LOW_DATA_THRESHOLD
    assert len(recs) == n_eligible
    assert all(set(r.elo) == set(bt.VARIANTS) for r in recs)


# --- stats -----------------------------------------------------------------------


def test_paired_delta():
    m, se, z, n = bt.paired_delta([0.20, 0.30, 0.25, 0.25], [0.25, 0.30, 0.30, 0.25])
    assert n == 4
    assert m == pytest.approx(-25.0)             # ×1000
    assert z < 0 and se > 0
    assert all(math.isnan(x) for x in bt.paired_delta([0.1], [0.2])[:3])


def test_ols_slope():
    x = [0.0, 1.0, 2.0, 3.0, 4.0]
    slope, t, n = bt.ols_slope(x, [1 + 2 * v for v in x])
    assert slope == pytest.approx(2.0) and t == math.inf and n == 5
    slope, t, _ = bt.ols_slope(x, [1.0, 3.2, 4.8, 7.1, 9.0])
    assert slope == pytest.approx(1.99) and t > 10
    assert math.isnan(bt.ols_slope([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])[0])


def test_brier_multiclass():
    assert bt.brier((0.5, 0.3, 0.2), (1.0, 0.0, 0.0)) == pytest.approx(0.25 + 0.09 + 0.04)
    assert bt.brier((0.7,), (1.0,)) == pytest.approx(0.09)
