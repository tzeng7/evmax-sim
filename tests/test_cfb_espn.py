"""Tests for the ESPN college-football play-by-play parser (evmax/clients/cfb_espn.py).

The network paths (fetch_*) are not exercised here — only parse_game_plays, which
turns a raw ESPN summary into the EPA-input rows the model core consumes. This is
where the load-bearing logic lives: score-delta attribution, half derivation,
offense/defense tagging, OT exclusion.
"""

from __future__ import annotations

from evmax.clients import cfb_espn as C


def _summary():
    """A minimal 2-drive ESPN summary: home (id 1) scores a TD; away (id 2)
    turns it over. Includes an OT play that must be dropped."""
    return {
        "drives": {
            "previous": [
                {
                    "team": {"id": "1"},
                    "plays": [
                        {  # 1st-and-10 rush, no score yet
                            "period": {"number": 1}, "type": {"text": "Rush"},
                            "homeScore": 0, "awayScore": 0, "statYardage": 6,
                            "start": {"down": 1, "distance": 10, "yardsToEndzone": 75,
                                      "team": {"id": "1"}},
                            "end": {"down": 2, "distance": 4, "yardsToEndzone": 69,
                                    "team": {"id": "1"}},
                        },
                        {  # touchdown: home score 0 -> 7
                            "period": {"number": 2}, "type": {"text": "Passing Touchdown"},
                            "homeScore": 7, "awayScore": 0, "statYardage": 69,
                            "start": {"down": 2, "distance": 4, "yardsToEndzone": 69,
                                      "team": {"id": "1"}},
                            "end": {"down": -1, "distance": 0, "yardsToEndzone": 0,
                                    "team": {"id": "1"}},
                        },
                    ],
                },
                {
                    "team": {"id": "2"},
                    "plays": [
                        {  # interception: possession flips to id 1, no score
                            "period": {"number": 3}, "type": {"text": "Pass Interception Return"},
                            "homeScore": 7, "awayScore": 0, "statYardage": 0, "isTurnover": True,
                            "start": {"down": 1, "distance": 10, "yardsToEndzone": 60,
                                      "team": {"id": "2"}},
                            "end": {"down": 1, "distance": 10, "yardsToEndzone": 40,
                                    "team": {"id": "1"}},
                        },
                        {  # overtime play — must be dropped
                            "period": {"number": 5}, "type": {"text": "Rush"},
                            "homeScore": 7, "awayScore": 0, "statYardage": 3,
                            "start": {"down": 1, "distance": 10, "yardsToEndzone": 25,
                                      "team": {"id": "2"}},
                            "end": {"down": 2, "distance": 7, "yardsToEndzone": 22,
                                    "team": {"id": "2"}},
                        },
                    ],
                },
            ]
        }
    }


def _meta():
    return {"game_id": "g1", "neutral": False,
            "home": {"id": "1", "abbr": "HOM"}, "away": {"id": "2", "abbr": "AWY"}}


def test_parse_drops_overtime():
    rows = C.parse_game_plays(_summary(), _meta())
    assert all(r["period"] < 5 for r in rows)
    # 3 non-OT plays (2 home drive + 1 away interception)
    assert len(rows) == 3


def test_parse_offense_defense_tags():
    rows = C.parse_game_plays(_summary(), _meta())
    r0 = rows[0]
    assert r0["off_team"] == "1" and r0["def_team"] == "2"
    r_int = rows[2]
    assert r_int["off_team"] == "2" and r_int["def_team"] == "1"


def test_parse_score_attribution():
    rows = C.parse_game_plays(_summary(), _meta())
    td = rows[1]
    assert td["score_points"] == 7
    assert td["score_team"] == "1"
    assert td["score_off"] is True     # home offense scored its own TD


def test_parse_half_derivation():
    rows = C.parse_game_plays(_summary(), _meta())
    assert rows[0]["half"] == 1   # period 1
    assert rows[1]["half"] == 1   # period 2
    assert rows[2]["half"] == 2   # period 3


def test_parse_turnover_end_state_flips_team():
    rows = C.parse_game_plays(_summary(), _meta())
    r_int = rows[2]
    # possession ends with the OTHER team (id 1) — play_epa uses this to sign EPA
    assert r_int["end_team"] == "1"
    assert r_int["off_team"] == "2"


def test_parse_empty_drives():
    assert C.parse_game_plays({"drives": {"previous": []}}, _meta()) == []
    assert C.parse_game_plays({}, _meta()) == []


def test_half_helper():
    assert C._half(1) == 1
    assert C._half(2) == 1
    assert C._half(3) == 2
    assert C._half(5) == 2   # OT folds into half 2 (dropped downstream)
    assert C._half(None) == 1


class _FakeResp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._p


class _FakeClient:
    def __init__(self, payload):
        self._p = payload

    def get(self, *a, **k):
        return _FakeResp(self._p)


def test_fetch_scoreboard_day_skips_events_without_competitions():
    """A placeholder event with no `competitions` block must not abort the day
    walk (it killed the 2026-09-03 reseed with a KeyError)."""
    import datetime as dt

    payload = {"events": [
        {"id": "1"},  # placeholder — no competitions
        {"id": "2", "date": "2026-09-05T20:00Z", "competitions": [{
            "neutralSite": False, "status": {"type": {"completed": True}},
            "competitors": [
                {"homeAway": "home", "score": "31", "team": {"id": "10", "abbreviation": "A", "location": "Alpha"}},
                {"homeAway": "away", "score": "10", "team": {"id": "20", "abbreviation": "B", "location": "Beta"}},
            ]}]},
    ]}
    games = C.fetch_scoreboard_day(_FakeClient(payload), dt.date(2026, 9, 5))
    assert [g["game_id"] for g in games] == ["2"]
    assert games[0]["home"]["id"] == "10" and games[0]["away"]["score"] == 10


def _sched_game(gid: str, completed: bool) -> dict:
    return {"game_id": gid, "date": "2026-09-05", "neutral": False, "completed": completed,
            "home": {"id": "1", "abbr": "HOM", "location": "Home"},
            "away": {"id": "2", "abbr": "AWY", "location": "Away"}}


def test_fetch_season_plays_with_schedule_skips_walk_and_keeps_completed_only(monkeypatch):
    """Passing a pre-fetched schedule must bypass the scoreboard walk and parse
    only COMPLETED rows — upcoming games have no plays to fetch."""
    def _boom(*a, **k):
        raise AssertionError("fetch_season_games must not be called when games= is given")

    fetched: list[str] = []

    def _fake_summary(gid, client=None, use_cache=True):
        fetched.append(gid)
        return {}  # parse_game_plays({}) -> [] — the parser is covered above

    monkeypatch.setattr(C, "fetch_season_games", _boom)
    monkeypatch.setattr(C, "fetch_game_summary", _fake_summary)

    schedule = [_sched_game("done", True), _sched_game("future", False)]
    rows, games = C.fetch_season_plays(2026, games=schedule, max_workers=1)

    assert rows == []
    assert [g["game_id"] for g in games] == ["done"]   # upcoming row dropped
    assert fetched == ["done"]                          # only the completed game fetched


# --- scoreboard-delta validation (2026-09 EPA corruption) --------------------
# ESPN's running scoreboard is not reliable enough to difference blindly. Game
# 401866418 carried a "14" typed as 1414 on a 0-yard rush, which the old parser
# turned into a 1400-point play (+21 EPA/play for one game-side, then a 2-hop
# rating corruption through the ridge solve). 720 illegal deltas were found over
# 353 cached games 2021-2026. Each case below reproduces one observed mode.


def _play(home, away, period=2, ptype="Rush", ytg=50):
    return {
        "period": {"number": period}, "type": {"text": ptype},
        "homeScore": home, "awayScore": away, "statYardage": 0,
        "start": {"down": 1, "distance": 10, "yardsToEndzone": ytg, "team": {"id": "1"}},
        "end": {"down": 2, "distance": 10, "yardsToEndzone": ytg, "team": {"id": "1"}},
    }


def _one_drive(*plays):
    return {"drives": {"previous": [{"team": {"id": "1"}, "plays": list(plays)}]}}


def test_parse_scoreboard_typo_spike_is_not_a_score():
    # (7, 14) -> (7, 1414) -> (7, 14): the observed game-401866418 pattern.
    rows = C.parse_game_plays(
        _one_drive(_play(7, 14), _play(7, 1414), _play(7, 14), _play(14, 14)), _meta()
    )
    assert [r["score_points"] for r in rows] == [0, 0, 0, 7]
    # the spike never reaches the garbage-time margin either
    assert rows[2]["off_margin_pre"] == 7 - 14


def test_parse_missing_score_carries_forward():
    # A play with no score fields used to read as 0-0, so the next play's delta
    # became the whole cumulative score (the "pts 14/16/20" rows).
    p_missing = _play(None, None)
    rows = C.parse_game_plays(
        _one_drive(_play(14, 7), p_missing, _play(14, 7), _play(21, 7)), _meta()
    )
    assert [r["score_points"] for r in rows] == [0, 0, 0, 7]
    assert rows[1]["off_margin_pre"] == 14 - 7


def test_parse_zero_zero_reset_mid_game_is_ignored():
    rows = C.parse_game_plays(
        _one_drive(_play(10, 3), _play(0, 0), _play(10, 3), _play(13, 3)), _meta()
    )
    assert [r["score_points"] for r in rows] == [0, 0, 0, 3]


def test_parse_multi_score_jump_is_not_credited():
    # A skipped play makes BOTH teams' scores move at once (7 + 3 = 10 in one
    # delta). That is not one play's score — credit nothing, then resync.
    rows = C.parse_game_plays(
        _one_drive(_play(7, 14), _play(14, 17), _play(14, 24)), _meta()
    )
    assert [r["score_points"] for r in rows] == [0, 0, 7]
    assert rows[2]["score_team"] == "2" and rows[2]["score_off"] is False


def test_parse_single_team_illegal_jump_is_not_credited():
    # One team +14 in one row (two TDs folded together) is not a legal play score.
    rows = C.parse_game_plays(_one_drive(_play(0, 0), _play(14, 0), _play(16, 0)), _meta())
    assert [r["score_points"] for r in rows] == [0, 0, 2]


def test_parse_emits_only_legal_score_points():
    rows = C.parse_game_plays(
        _one_drive(_play(0, 0), _play(1400, 0), _play(-3, 0), _play("x", 0),
                   _play(6, 0), _play(7, 0), _play(7, 2), _play(10, 2)),
        _meta(),
    )
    pts = [r["score_points"] for r in rows]
    assert pts == [0, 0, 0, 0, 6, 1, 2, 3]
    assert all(p == 0 or p in C.LEGAL_PLAY_POINTS for p in pts)


def test_parse_possession_from_play_start_team_not_drive():
    """ESPN game 401856784 labels Baylor (home, id 1 here) drives as the
    visitor's. The play's own start.team is the reliable possession field."""
    mislabeled = {
        "team": {"id": "2"},   # drive says AWAY has the ball…
        "plays": [
            {  # …but the snap is HOME's (start.team 1) and home keeps the ball
                "period": {"number": 2}, "type": {"text": "Rush"},
                "homeScore": 0, "awayScore": 0, "statYardage": 6,
                "start": {"down": 1, "distance": 10, "yardsToEndzone": 60, "team": {"id": "1"}},
                "end": {"down": 2, "distance": 4, "yardsToEndzone": 54, "team": {"id": "1"}},
            },
            {  # home TD on the same mislabeled drive
                "period": {"number": 2}, "type": {"text": "Rushing Touchdown"},
                "homeScore": 7, "awayScore": 0, "statYardage": 54,
                "start": {"down": 2, "distance": 4, "yardsToEndzone": 54, "team": {"id": "1"}},
                "end": {"down": -1, "distance": 0, "yardsToEndzone": 0, "team": {"id": "1"}},
            },
            {  # kickoff: start.team is the KICKING side — keep the drive team
                "period": {"number": 2}, "type": {"text": "Kickoff"},
                "homeScore": 7, "awayScore": 0, "statYardage": 0,
                "start": {"down": 1, "distance": 10, "yardsToEndzone": 65, "team": {"id": "1"}},
                "end": {"down": 1, "distance": 10, "yardsToEndzone": 75, "team": {"id": "2"}},
            },
        ],
    }
    rows = C.parse_game_plays({"drives": {"previous": [mislabeled]}}, _meta())
    assert rows[0]["off_team"] == "1" and rows[0]["def_team"] == "2"
    assert rows[0]["end_team"] == rows[0]["off_team"]      # no phantom turnover
    assert rows[1]["score_off"] is True                   # offense scored its own TD
    assert rows[2]["off_team"] == "2"                      # kickoff keeps drive team


def test_parse_possession_falls_back_to_drive_team():
    """No start.team, or one naming neither side → the drive team."""
    drive = {"team": {"id": "1"}, "plays": [
        _play(0, 0),
        {**_play(0, 0), "start": {"down": 1, "distance": 10, "yardsToEndzone": 50}},
        {**_play(0, 0), "start": {"down": 1, "distance": 10, "yardsToEndzone": 50,
                                   "team": {"id": "999"}}},
    ]}
    rows = C.parse_game_plays({"drives": {"previous": [drive]}}, _meta())
    assert [r["off_team"] for r in rows] == ["1", "1", "1"]


def test_parse_kickoff_scoreboard_dip_does_not_recredit_the_score():
    """ESPN shows the PRE-score scoreboard on the kickoff row after a TD
    ((10,21) → kickoff (10,14) → next snap (10,21)). The dip must not become the
    baseline, or the next ordinary snap is re-credited with the touchdown
    (game 401403853: a Hawaii incompletion credited +7 to Vanderbilt)."""
    rows = C.parse_game_plays(
        _one_drive(_play(10, 14), _play(10, 21), _play(10, 14, ptype="Kickoff"),
                   _play(10, 21), _play(10, 21)),
        _meta(),
    )
    assert [r["score_points"] for r in rows] == [0, 7, 0, 0, 0]
