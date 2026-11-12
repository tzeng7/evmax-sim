"""Tests for scripts/seed_ncaab_efficiency.py's in-season behaviour.

Regressions for the November 2026 reseed failures:
- 2026-11-02: a same-day fetch cached opening night (182 men's D1 finals) as a
  2-game day file that was never refetched (`day_is_complete`,
  `cached_day_is_complete`).
- 2026-11-05: the appearance-based D1 filter needs 8 completed games per team,
  so every early-season seed aborted with "no D1 games after filtering"
  (`build_state(..., d1_team_ids=...)`, `filter_games_to_teams`,
  `early_season_min_appearances`).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import pytest

from evmax.agents.models._college_efficiency import (
    MIN_D1_APPEARANCES,
    early_season_min_appearances,
    filter_games_to_teams,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "seed_ncaab_efficiency.py"


def _load():
    spec = importlib.util.spec_from_file_location("seed_ncaab_efficiency", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # @dataclass resolves its module through sys.modules.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


seed = _load()

DAY = date(2026, 11, 2)


def _fetch(**kw):
    base = dict(games=[], scoreboard_ok=True, n_pending=0, n_missing_box=0)
    base.update(kw)
    return seed.DayFetch(**base)


# ---------------------------------------------------------------------------
# Day-file completeness
# ---------------------------------------------------------------------------


class TestDayIsComplete:
    def test_same_day_fetch_is_never_complete(self):
        # The 2026-11-02 08:17 run: no events pending yet would still be wrong.
        assert not seed.day_is_complete(DAY, DAY, _fetch())

    def test_next_day_settled_fetch_is_complete(self):
        assert seed.day_is_complete(DAY, date(2026, 11, 3), _fetch())

    def test_pending_event_or_missing_box_keeps_it_incomplete(self):
        nxt = date(2026, 11, 3)
        assert not seed.day_is_complete(DAY, nxt, _fetch(n_pending=1))
        assert not seed.day_is_complete(DAY, nxt, _fetch(n_missing_box=1))

    def test_scoreboard_failure_is_never_complete(self):
        assert not seed.day_is_complete(DAY, date(2026, 12, 30), _fetch(scoreboard_ok=False))

    def test_grace_period_stops_endless_refetch(self):
        late = date(2026, 11, 2 + seed.SETTLE_GRACE_DAYS + 1)
        assert seed.day_is_complete(DAY, late, _fetch(n_pending=1, n_missing_box=2))


class TestCachedDayIsComplete:
    def test_missing_file(self, tmp_path):
        assert not seed.cached_day_is_complete(tmp_path / "x.json", DAY)

    def test_flagged_file(self, tmp_path):
        f = tmp_path / "d.json"
        f.write_text(json.dumps({"games": [], "complete": False}))
        assert not seed.cached_day_is_complete(f, DAY)
        f.write_text(json.dumps({"games": [], "complete": True}))
        assert seed.cached_day_is_complete(f, DAY)

    def test_legacy_file_trusted_only_when_written_well_after_the_day(self, tmp_path):
        f = tmp_path / "legacy.json"
        f.write_text(json.dumps({"date": DAY.isoformat(), "games": []}))

        def _set_mtime(d: date) -> None:
            ts = time.mktime(d.timetuple()) + 12 * 3600
            os.utime(f, (ts, ts))

        _set_mtime(DAY)  # written the same morning — the frozen opening night
        assert not seed.cached_day_is_complete(f, DAY)
        _set_mtime(date(2026, 11, 2 + seed.LEGACY_TRUST_DAYS))
        assert seed.cached_day_is_complete(f, DAY)

    def test_corrupt_file_is_refetched(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("{not json")
        assert not seed.cached_day_is_complete(f, DAY)


# ---------------------------------------------------------------------------
# D1 universe
# ---------------------------------------------------------------------------


def _row(h, a):
    return {"home_id": h, "away_id": a, "neutral": False, "home_pts": 70,
            "away_pts": 60, "home_poss": 68.0, "away_poss": 68.0}


def test_filter_games_to_teams_drops_non_d1_side():
    rows = [_row("1", "2"), _row("1", "99")]
    kept, seen = filter_games_to_teams(rows, {"1", "2", "3"})
    assert kept == [rows[0]]
    assert seen == {"1", "2"}


@pytest.mark.parametrize(
    "today, expected",
    [
        (date(2026, 11, 2), 2),          # opening week: floor of 2
        (date(2026, 11, 22), 3),         # three weeks in
        (date(2027, 1, 20), MIN_D1_APPEARANCES),  # capped at the full-season bar
    ],
)
def test_early_season_min_appearances(today, expected):
    assert early_season_min_appearances(today) == expected


def _cached_game(day, hid, aid, hs, as_):
    def side(tid, score):
        return {"id": tid, "location": f"School {tid}", "display": f"School {tid} Team",
                "abbr": f"S{tid}", "score": score, "fgm": 25, "fga": 58, "tpm": 7,
                "ftm": 12, "fta": 17, "oreb": 9, "dreb": 24, "tov": 12}
    return {"event_id": f"{day}-{hid}-{aid}", "date": day, "neutral": False,
            "home": side(hid, hs), "away": side(aid, as_)}


def test_build_state_early_season_with_d1_list():
    # Three teams with 2 games each: the 8-appearance filter finds nothing,
    # the D1 list keeps every D1-vs-D1 game and drops the exhibition.
    games = [
        _cached_game("2026-11-02", "1", "2", 80, 70),
        _cached_game("2026-11-04", "2", "3", 75, 72),
        _cached_game("2026-11-06", "3", "1", 66, 71),
        _cached_game("2026-11-06", "1", "900", 110, 50),  # non-D1 exhibition
    ]
    with pytest.raises(RuntimeError, match="no D1 games"):
        seed.build_state(games, 2026, "ncaab")
    state = seed.build_state(games, 2026, "ncaab", d1_team_ids={"1", "2", "3"})
    assert state["source_season"] == "2026"
    assert state["n_games_aggregated"] == 3
    assert len(state["teams"]) == 3
    assert all(t["gp"] == 2 for t in state["teams"].values())


def test_fetch_d1_team_ids_rejects_short_list(monkeypatch):
    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            teams = [{"team": {"id": i}} for i in range(5)]
            return {"sports": [{"leagues": [{"teams": teams}]}]}

    monkeypatch.setattr(seed.httpx, "get", lambda *a, **k: _Resp())
    assert seed.fetch_d1_team_ids("mens-college-basketball") is None


def test_fetch_d1_team_ids_happy_path(monkeypatch):
    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            teams = [{"team": {"id": i}} for i in range(365)]
            return {"sports": [{"leagues": [{"teams": teams}]}]}

    monkeypatch.setattr(seed.httpx, "get", lambda *a, **k: _Resp())
    ids = seed.fetch_d1_team_ids("mens-college-basketball")
    assert ids is not None and len(ids) == 365 and "0" in ids
