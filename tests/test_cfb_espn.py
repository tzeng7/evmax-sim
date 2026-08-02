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
