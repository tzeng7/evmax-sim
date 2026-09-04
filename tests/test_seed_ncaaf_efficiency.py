"""Tests for scripts/seed_ncaaf_efficiency.py — the in-season FBS-universe path.

Network-free: the ESPN client calls are monkeypatched. Guards the 2026-09-03
early-season seed bug: the in-season FBS universe used to be derived from
COMPLETED games only, so until ~week 6 no team reached FBS_MIN_APPEARANCES,
the universe was EMPTY, every team was FCS-pooled and `gp` stayed 0 for all
138 teams — the prior→in-season ramp never engaged. The fix derives the
universe from the FULL SCHEDULE (completed + upcoming) and parses plays only
for the completed rows.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "seed_ncaaf_efficiency",
    Path(__file__).resolve().parents[1] / "scripts" / "seed_ncaaf_efficiency.py",
)
seed = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(seed)


def _game(gid: str, home: str, away: str, completed: bool) -> dict:
    return {
        "game_id": gid, "date": "2026-09-05", "neutral": False, "completed": completed,
        "home": {"id": home, "abbr": home.upper(), "location": f"Team {home}", "score": 21},
        "away": {"id": away, "abbr": away.upper(), "location": f"Team {away}", "score": 14},
    }


def _schedule() -> list[dict]:
    """Two FBS teams (a, b) with a full 12-game slate each, one FCS side (fcs)
    that appears once as a buy-game opponent. Only week-1 games are complete."""
    games = []
    # a and b play 12 games each against a rotating cast (c1..c11 are FBS too
    # but only need to appear >= FBS_MIN_APPEARANCES times to count; give them
    # enough games via a round-robin so the universe is realistic).
    fbs = [f"t{i}" for i in range(12)]
    gid = 0
    for i, h in enumerate(fbs):
        for j, a in enumerate(fbs):
            if i < j:
                gid += 1
                games.append(_game(f"g{gid}", h, a, completed=(gid <= 3)))
    games.append(_game("buy", "t0", "fcs", completed=True))  # FCS buy game
    return games


def test_fbs_universe_from_schedule_includes_teams_with_no_completed_games():
    schedule = _schedule()
    completed_only = [g for g in schedule if g["completed"]]

    fbs_from_completed, _, _ = seed._fbs_universe(completed_only)
    fbs_from_schedule, name, abbr = seed._fbs_universe(schedule)

    # The bug: early in the season nobody has FBS_MIN_APPEARANCES completed games.
    assert fbs_from_completed == set()
    # The fix: the full schedule identifies every FBS team, excludes the FCS side.
    assert fbs_from_schedule == {f"t{i}" for i in range(12)}
    assert "fcs" not in fbs_from_schedule
    assert name["t0"] == "team t0" and abbr["t0"] == "T0"


def test_season_ratings_in_season_uses_schedule_and_completed_plays(monkeypatch):
    schedule = _schedule()
    calls: dict[str, object] = {}

    def fake_fetch_season_games(season, only_completed=True):
        calls["only_completed"] = only_completed
        return schedule if not only_completed else [g for g in schedule if g["completed"]]

    def fake_fetch_season_plays(season, games=None, **kw):
        calls["games_passed"] = games is not None
        done = [g for g in (games or []) if g.get("completed")]
        return [], done  # no plays needed — the universe/gp wiring is what's under test

    captured: dict[str, object] = {}

    def fake_build_team_ratings(plays, fbs, games_by_id, table, ridge=1.0):
        captured["fbs"] = set(fbs)
        captured["games"] = set(games_by_id)
        return {"teams": {t: {"off_epa_adj": 0.0, "def_epa_adj": 0.0, "gp": 0,
                              "off_success_rate": 0.0, "def_success_rate": 0.0} for t in fbs},
                "league_mean_epa": 0.0, "hfa_epa": 0.0}

    monkeypatch.setattr(seed.C, "fetch_season_games", fake_fetch_season_games)
    monkeypatch.setattr(seed.C, "fetch_season_plays", fake_fetch_season_plays)
    monkeypatch.setattr(seed.E, "build_team_ratings", fake_build_team_ratings)

    ratings, name, abbr, table, n_games = seed._season_ratings(
        2026, ep_table={"dummy": 1}, ridge=1.0, in_season=True
    )

    assert calls["only_completed"] is False          # full schedule requested
    assert calls["games_passed"] is True             # schedule handed to the plays fetch
    assert captured["fbs"] == {f"t{i}" for i in range(12)}   # universe from the schedule
    assert captured["games"] == {"g1", "g2", "g3", "buy"}    # ratings see completed games only
    assert n_games == 4
    assert set(ratings["teams"]) == captured["fbs"]


def test_season_ratings_default_path_unchanged(monkeypatch):
    """The prior-season (full-season) path still derives the universe from the
    completed games it fetched — the backtest/prior behaviour is untouched."""
    schedule = _schedule()
    completed = [g for g in schedule if g["completed"]]

    def fake_fetch_season_games(*a, **k):
        raise AssertionError("default path must not walk the schedule separately")

    def fake_fetch_season_plays(season, games=None, **kw):
        assert games is None
        return [], completed

    seen: dict[str, object] = {}

    def fake_build_team_ratings(plays, fbs, games_by_id, table, ridge=1.0):
        seen["fbs"] = set(fbs)
        return {"teams": {}, "league_mean_epa": 0.0, "hfa_epa": 0.0}

    monkeypatch.setattr(seed.C, "fetch_season_games", fake_fetch_season_games)
    monkeypatch.setattr(seed.C, "fetch_season_plays", fake_fetch_season_plays)
    monkeypatch.setattr(seed.E, "build_team_ratings", fake_build_team_ratings)

    seed._season_ratings(2025, ep_table={"dummy": 1}, ridge=1.0)
    assert seen["fbs"] == set()   # 4 completed games < FBS_MIN_APPEARANCES → same as before


def test_fbs_universe_excludes_espn_tbd_placeholder_sides():
    """Not-yet-set December championship/bowl slots carry ESPN's 'TBD' sides
    (ids -1 / -2). On a full-schedule walk they appear 50+ times and would
    otherwise pass FBS_MIN_APPEARANCES and land a junk 'tbd' team in the state."""
    schedule = _schedule()
    for i in range(8):
        g = _game(f"tbd{i}", "t0", "-1", completed=False)
        g["away"]["location"] = "TBD"
        g["away"]["abbr"] = "TBD"
        schedule.append(g)
    # A positively-numbered id whose name is TBD is a placeholder too.
    g = _game("tbdx", "-2", "t1", completed=False)
    g["home"]["location"] = "TBD"
    schedule.append(g)

    fbs, name, _ = seed._fbs_universe(schedule)

    assert "-1" not in fbs and "-2" not in fbs
    assert "tbd" not in name.values()
    assert fbs == {f"t{i}" for i in range(12)}
    assert seed._is_placeholder_team("-1", "Alabama") is True      # negative id
    assert seed._is_placeholder_team("2000", "TBD") is True        # TBD name
    assert seed._is_placeholder_team("2000", "Alabama") is False
