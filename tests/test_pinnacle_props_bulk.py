"""Pinnacle player-prop pricing off the bulk sport-level straight-markets index.

Background (2026-09-02 NFL season-start audit): from the US the guest API
403s (``BAD_LOCATION``) the per-matchup ``/matchups/{id}/markets/related/
straight`` fetch for SPECIAL matchups only — the parent game's id serves
fine — which is why player props were the sector the intermittent geo-block
"hit hardest". The bare ``/sports/{sport_id}/markets/straight`` call still
carries every special's over/under market keyed by ``matchupId``, so
``get_prop_odds`` now prices props off ONE bulk fetch and only falls back to
the per-matchup endpoint for specials the index lacks.

Also pins the NFL league id (258 → 889 for the 2026 season): a stale id
returns zero matchups, i.e. no sharp anchor and no NFL EV all season.
"""

from __future__ import annotations

import asyncio

import pytest

from evmax.clients import esports_pinnacle as ep


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


_START = "2026-09-11T00:35:00Z"  # Thu SF@LAR kickoff, 8:35pm ET → ET game day 09-10

_MATCHUPS = [
    {
        "id": 100, "type": "matchup", "parentId": None, "startTime": _START,
        "league": {"id": 889, "name": "NFL"},
        "participants": [
            {"name": "Los Angeles Rams", "alignment": "home"},
            {"name": "San Francisco 49ers", "alignment": "away"},
        ],
    },
    {
        "id": 201, "type": "special", "parentId": 100, "startTime": _START,
        "league": {"id": 889, "name": "NFL"},
        "special": {"category": "Player Props", "description": "Brock Purdy Total Passing Yards"},
    },
    {
        "id": 202, "type": "special", "parentId": 100, "startTime": _START,
        "league": {"id": 889, "name": "NFL"},
        "special": {"category": "Player Props", "description": "Christian McCaffrey Total Receptions"},
    },
    # Not in the bulk index → must take the per-matchup fallback path.
    {
        "id": 203, "type": "special", "parentId": 100, "startTime": _START,
        "league": {"id": 889, "name": "NFL"},
        "special": {"category": "Player Props", "description": "Puka Nacua Total Receiving Yards"},
    },
    # Wrong league → filtered out before any pricing.
    {
        "id": 301, "type": "special", "parentId": 150, "startTime": _START,
        "league": {"id": 880, "name": "NCAA"},
        "special": {"category": "Player Props", "description": "Some Kid Total Passing Yards"},
    },
    # Not a player prop → filtered out.
    {
        "id": 401, "type": "special", "parentId": 100, "startTime": _START,
        "league": {"id": 889, "name": "NFL"},
        "special": {"category": "Futures", "description": "Regular Season MVP"},
    },
]


def _total(mid: int, line: float, over: int, under: int) -> dict:
    # Real bulk payload shape: prop prices carry no `designation`, so the
    # parser's positional fallback (index 0 = over, 1 = under) is exercised.
    return {
        "matchupId": mid, "type": "total", "period": 0, "key": "s;0;ou",
        "prices": [
            {"participantId": mid * 10 + 1, "points": line, "price": over},
            {"participantId": mid * 10 + 2, "points": line, "price": under},
        ],
    }


_BULK = [
    {"matchupId": 100, "type": "moneyline", "period": 0, "prices": []},
    _total(201, 249.5, -115, -105),
    _total(202, 5.5, -120, 100),
]


class _Recorder:
    """Fake `_logged_get` that serves the fixtures and records every path."""

    def __init__(self, *, bulk_error: Exception | None = None):
        self.calls: list[str] = []
        self.bulk_error = bulk_error

    async def __call__(self, path, params=None, *, sector, purpose):
        self.calls.append(path)
        if path == "/sports/15/matchups":
            return _MATCHUPS
        if path == "/sports/15/markets/straight":
            assert params is None, "withSpecials on the bulk endpoint trips the geo-block"
            if self.bulk_error is not None:
                raise self.bulk_error
            return _BULK
        if path == "/matchups/203/markets/related/straight":
            return [_total(100, 47.5, -110, -110), _total(203, 79.5, -110, -110)]
        if path.startswith("/matchups/20") and path.endswith("/related/straight"):
            # 201/202 are priced by the bulk index; a per-matchup call for them
            # is exactly the geo-blocked request the bulk path exists to avoid.
            return [_total(int(path.split("/")[2]), 1.5, -110, -110)]
        raise AssertionError(f"unexpected path {path}")


def _client(rec: _Recorder) -> ep.PinnacleGuestClient:
    c = ep.PinnacleGuestClient()
    c._logged_get = rec  # type: ignore[method-assign]
    return c


def test_nfl_league_id_pinned_to_2026_value():
    """258 was the 2025 id; Pinnacle re-cut it to 889 for 2026. A stale id
    yields zero matchups (no sharp anchor) — change deliberately, re-verify
    via GET /sports/15/leagues at every season boundary."""
    assert ep.SECTOR_SPORT_LEAGUES["nfl"] == (15, [889])
    assert ep.SECTOR_SPORT_LEAGUES["ncaaf"] == (15, [880])


def test_ucl_league_id_pinned_to_2026_27_value():
    """Same drift class, caught by scripts/check_pinnacle_leagues.py the day it
    was written: UCL moved 2186 → 2627 (2186 served 0 matchups). The draw set
    must follow so the 3-way devig still fires for UCL."""
    soccer_ids = ep.SECTOR_SPORT_LEAGUES["soccer"][1]
    assert 2627 in soccer_ids and 2186 not in soccer_ids
    assert 2627 in ep.SOCCER_DRAW_LEAGUES and 2186 not in ep.SOCCER_DRAW_LEAGUES


def test_props_priced_from_bulk_index_without_per_matchup_calls():
    rec = _Recorder()
    props = _run(_client(rec).get_prop_odds("nfl"))

    by_stat = {p.prop_stat_type: p for p in props}
    assert set(by_stat) == {"passing_yards", "receptions", "receiving_yards"}
    assert by_stat["passing_yards"].total_line == 249.5
    assert by_stat["receptions"].total_line == 5.5
    assert by_stat["receiving_yards"].total_line == 79.5
    # Over/under devig sanity: -115/-105 → over favoured.
    assert by_stat["passing_yards"].true_prob_over > 0.5
    assert by_stat["receptions"].true_prob_over > by_stat["receptions"].true_prob_under
    # Event ids land on the ET game day (8:35pm ET Thursday kickoff → 09-10).
    assert all(p.event_id.startswith("nfl::2026-09-10::prop::") for p in props)

    assert rec.calls.count("/sports/15/markets/straight") == 1
    per_matchup = [c for c in rec.calls if "/related/straight" in c]
    assert per_matchup == ["/matchups/203/markets/related/straight"], per_matchup


def test_bulk_failure_degrades_to_per_matchup_path():
    rec = _Recorder(bulk_error=RuntimeError("403 BAD_LOCATION"))
    props = _run(_client(rec).get_prop_odds("nfl"))

    assert {p.prop_stat_type for p in props} == {"passing_yards", "receptions", "receiving_yards"}
    per_matchup = sorted(c for c in rec.calls if "/related/straight" in c)
    assert per_matchup == [
        "/matchups/201/markets/related/straight",
        "/matchups/202/markets/related/straight",
        "/matchups/203/markets/related/straight",
    ]


def test_parse_prop_matchup_is_pure_and_filters_to_own_total():
    c = ep.PinnacleGuestClient()
    matchup = _MATCHUPS[1]
    # Parent-game total first — must NOT be picked up as the prop's line.
    own = [_total(201, 249.5, -110, -110)]
    odds = c._parse_prop_matchup(matchup, own, "nfl")
    assert odds is not None and odds.total_line == 249.5
    assert c._parse_prop_matchup(matchup, [], "nfl") is None
    unparsable = dict(matchup, special={"category": "Player Props", "description": "Brock Purdy Total Dunks"})
    assert c._parse_prop_matchup(unparsable, own, "nfl") is None
