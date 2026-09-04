"""evmax/clients/nfl_depth_charts.py — pre-game QB starters from nflverse depth charts."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from evmax.clients import nfl_depth_charts as dc
from evmax.clients.nfl_depth_charts import (
    QbChartRow,
    normalize_person,
    pbp_passer_name,
    qb_depth_as_of,
    resolve_pregame_starters,
    team_key,
)


@pytest.mark.parametrize("full, expected", [
    ("Patrick Mahomes", "P.Mahomes"),
    ("Michael Penix Jr.", "M.Penix"),
    ("Gardner Minshew II", "G.Minshew"),
    ("Aidan O'Connell", "A.O'Connell"),
    ("Amon-Ra St. Brown", "A.St. Brown"),
    ("Anthony Richardson Sr.", "A.Richardson"),
    ("Tommy DeVito", "T.DeVito"),
    ("Cher", "Cher"),
    ("", ""),
])
def test_pbp_passer_name_matches_nflverse_format(full, expected):
    assert pbp_passer_name(full) == expected


def test_normalize_person_strips_suffix_punctuation_and_case():
    assert normalize_person("Michael Penix Jr.") == "michael penix"
    assert normalize_person("MICHAEL PENIX") == "michael penix"
    assert normalize_person("Aidan O'Connell") == "aidan oconnell"
    assert normalize_person("Gardner Minshew II") == normalize_person("Gardner Minshew")


def test_team_key_maps_codes_and_aliases():
    assert team_key("KC") == "kansas city chiefs"
    assert team_key("LAR") == team_key("LA") == "los angeles rams"
    assert team_key("WSH") == team_key("WAS") == "washington commanders"
    assert team_key("OAK") == "las vegas raiders"
    assert team_key("XXX") is None


def _snap(team, player, rank, when):
    return QbChartRow(team, player, rank, datetime.fromisoformat(when).replace(tzinfo=timezone.utc), None)


KC = "kansas city chiefs"
ATL = "atlanta falcons"

SNAPSHOT_ROWS = [
    _snap(KC, "Patrick Mahomes", 1, "2025-12-10T12:00:00"),
    _snap(KC, "Gardner Minshew II", 2, "2025-12-10T12:00:00"),
    _snap(KC, "Gardner Minshew II", 1, "2025-12-16T12:00:00"),   # swap listed 5 days pre-game
    _snap(KC, "Chris Oladokun", 2, "2025-12-16T12:00:00"),
    _snap(KC, "Chris Oladokun", 1, "2025-12-21T18:30:00"),      # game-day (post-kick) snapshot
    _snap(ATL, "Michael Penix Jr.", 1, "2025-12-16T12:00:00"),
    _snap(ATL, "Kirk Cousins", 2, "2025-12-16T12:00:00"),
]


def test_snapshot_uses_latest_chart_strictly_before_game_day():
    depth = qb_depth_as_of(SNAPSHOT_ROWS, as_of=date(2025, 12, 21))
    assert depth[KC] == ["Gardner Minshew II", "Chris Oladokun"]   # 12-16 chart, not 12-21
    assert depth[ATL] == ["Michael Penix Jr.", "Kirk Cousins"]
    # a week earlier the 12-10 chart applies
    assert qb_depth_as_of(SNAPSHOT_ROWS, as_of=date(2025, 12, 14))[KC][0] == "Patrick Mahomes"
    # before any chart → nothing for that team
    assert KC not in qb_depth_as_of(SNAPSHOT_ROWS, as_of=date(2025, 12, 1))


def test_snapshot_none_cutoff_is_the_live_latest():
    assert qb_depth_as_of(SNAPSHOT_ROWS)[KC][0] == "Chris Oladokun"


WEEKLY_ROWS = [
    QbChartRow(KC, "Patrick Mahomes", 1, None, 1),
    QbChartRow(KC, "Carson Wentz", 2, None, 1),
    QbChartRow(KC, "Carson Wentz", 1, None, 2),
]


def test_weekly_schema_is_keyed_by_week():
    assert qb_depth_as_of(WEEKLY_ROWS, week=1)[KC] == ["Patrick Mahomes", "Carson Wentz"]
    assert qb_depth_as_of(WEEKLY_ROWS, week=2)[KC] == ["Carson Wentz"]
    assert qb_depth_as_of(WEEKLY_ROWS, week=None) == {}
    assert qb_depth_as_of([]) == {}


def test_resolve_skips_qbs_the_injury_report_rules_out():
    res = resolve_pregame_starters(
        2025, as_of=date(2025, 12, 21), rows=SNAPSHOT_ROWS,
        out_players={ATL: ["Michael Penix Jr."], "Kansas City Chiefs": {"gardner minshew"}},
    )
    assert res[KC]["starter"] == "C.Oladokun" and res[KC]["skipped"] == ["Gardner Minshew II"]
    assert res[ATL]["starter"] == "K.Cousins" and res[ATL]["rank"] == 2
    # nobody healthy → team absent (caller falls back to last-game starter)
    res2 = resolve_pregame_starters(
        2025, as_of=date(2025, 12, 21), rows=SNAPSHOT_ROWS,
        out_players={ATL: ["Michael Penix Jr.", "Kirk Cousins"]},
    )
    assert ATL not in res2 and res2[KC]["starter"] == "G.Minshew"


def test_rows_from_frame_handles_both_schemas():
    pl = pytest.importorskip("polars")
    snap = pl.DataFrame({
        "dt": ["2026-09-03T11:53:47Z", "2026-09-03T11:53:47Z", "2026-09-03T11:53:47Z"],
        "team": ["ARI", "ARI", "ARI"],
        "player_name": ["Jacoby Brissett", "Gardner Minshew II", "Kyler Murray"],
        "pos_abb": ["QB", "QB", "WR"],
        "pos_rank": [1, 2, 1],
    })
    rows = dc.rows_from_frame(snap)
    assert [(r.team, r.player, r.rank) for r in rows] == [
        ("arizona cardinals", "Jacoby Brissett", 1), ("arizona cardinals", "Gardner Minshew II", 2)]
    assert rows[0].as_of.tzinfo is not None and rows[0].week is None
    weekly = pl.DataFrame({
        "season": [2024, 2024, 2024], "club_code": ["KC", "KC", "ZZZ"], "week": [1, 1, 1],
        "game_type": ["REG"] * 3, "depth_team": ["2", "1", "1"], "position": ["QB", "QB", "QB"],
        "full_name": ["Carson Wentz", "Patrick Mahomes", "Nobody"],
    })
    rows = dc.rows_from_frame(weekly)
    assert sorted((r.player, r.rank, r.week) for r in rows) == [("Carson Wentz", 2, 1), ("Patrick Mahomes", 1, 1)]


def test_load_is_fail_soft(monkeypatch):
    import sys, types
    fake = types.ModuleType("nflreadpy")
    def boom(seasons): raise RuntimeError("offline")
    fake.load_depth_charts = boom
    monkeypatch.setitem(sys.modules, "nflreadpy", fake)
    dc._cache.clear()
    assert dc.load_qb_chart_rows(2099, refresh=True) == []
