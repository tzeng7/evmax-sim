"""Tests for the MLB Stats API client (evmax/clients/mlb_statsapi.py).

Covers the pure parser against a fixture schedule payload — the part that
matters for correctness: stable team-id → nickname mapping, accent-folding to
match the seeded pitcher DB, and TBD-starter handling.
"""

from __future__ import annotations

import pytest

from evmax.clients.mlb_statsapi import (
    MLB_TEAM_NICKNAME,
    ascii_fold,
    _parse_probables,
)


def _game(home_id, home_pitcher, away_id, away_pitcher):
    def _team(tid, pitcher):
        t = {"team": {"id": tid}}
        if pitcher is not None:
            t["probablePitcher"] = {"id": 1, "fullName": pitcher}
        return t
    return {"teams": {"home": _team(home_id, home_pitcher), "away": _team(away_id, away_pitcher)}}


def _payload(*games):
    return {"dates": [{"games": list(games)}]}


class TestAsciiFold:
    def test_lowercases(self):
        assert ascii_fold("Sonny Gray") == "sonny gray"

    def test_strips_accents(self):
        # Must match scripts/seed_pitcher_fip.py::_ascii_fold exactly.
        assert ascii_fold("Cristopher Sánchez") == "cristopher sanchez"
        assert ascii_fold("José Berríos") == "jose berrios"

    def test_handles_empty(self):
        assert ascii_fold("") == ""
        assert ascii_fold(None) == ""


class TestParseProbables:
    def test_maps_team_ids_to_nicknames(self):
        # 111 = red sox, 147 = yankees — the multi-word nickname that broke
        # the old ESPN name match resolves by stable id here.
        out = _parse_probables(_payload(_game(147, "Ryan Weathers", 111, "Sonny Gray")))
        assert set(out) == {"yankees", "red sox"}
        assert out["red sox"]["name"] == "sonny gray"
        assert out["yankees"]["name"] == "ryan weathers"

    def test_folds_pitcher_name_for_seed_lookup(self):
        out = _parse_probables(_payload(_game(141, "José Berríos", 142, "Pablo López")))
        assert out["blue jays"]["name"] == "jose berrios"
        assert out["twins"]["name"] == "pablo lopez"

    def test_tbd_starter_skipped(self):
        # No probablePitcher announced for the home side yet → omit that team,
        # keep the side that has one.
        out = _parse_probables(_payload(_game(111, None, 147, "Ryan Weathers")))
        assert "red sox" not in out
        assert out["yankees"]["name"] == "ryan weathers"

    def test_unknown_team_id_skipped(self):
        # Defensive: an unrecognised id (e.g. an All-Star/exhibition club)
        # must not crash or leak a None key.
        out = _parse_probables(_payload(_game(999, "Nobody", 147, "Ryan Weathers")))
        assert out == {"yankees": {"name": "ryan weathers", "era": 0.0, "ip": 0,
                                   "mlb_id": 1, "mlb_team_id": 147}}

    def test_carries_stable_ids(self):
        out = _parse_probables(_payload(_game(147, "Ryan Weathers", 111, "Sonny Gray")))
        assert out["yankees"]["mlb_team_id"] == 147
        assert out["yankees"]["mlb_id"] == 1

    def test_empty_payload(self):
        assert _parse_probables({}) == {}
        assert _parse_probables({"dates": []}) == {}


class TestTeamMap:
    def test_thirty_teams(self):
        assert len(MLB_TEAM_NICKNAME) == 30

    def test_nicknames_unique(self):
        assert len(set(MLB_TEAM_NICKNAME.values())) == 30

    def test_multiword_nicknames_present(self):
        # The three that broke the ESPN last-word match.
        vals = set(MLB_TEAM_NICKNAME.values())
        assert {"red sox", "white sox", "blue jays"} <= vals


# ---------------------------------------------------------------------------
# Reliever appearances (WS2a) — pure parsers against a boxscore fixture
# ---------------------------------------------------------------------------

from evmax.clients.mlb_statsapi import (  # noqa: E402
    parse_innings_pitched,
    parse_reliever_appearances,
)


class TestParseInningsPitched:
    def test_thirds_notation(self):
        assert parse_innings_pitched("1.2") == pytest.approx(1 + 2 / 3)
        assert parse_innings_pitched("0.1") == pytest.approx(1 / 3)
        assert parse_innings_pitched("6.0") == 6.0
        assert parse_innings_pitched("7") == 7.0

    def test_garbage_is_zero(self):
        assert parse_innings_pitched(None) == 0.0
        assert parse_innings_pitched("abc") == 0.0


def _boxscore_fixture() -> dict:
    def _pitcher(pid, name, gs, ip, pitches):
        return f"ID{pid}", {
            "person": {"id": pid, "fullName": name},
            "stats": {"pitching": {
                "gamesStarted": gs,
                "inningsPitched": ip,
                "numberOfPitches": pitches,
            }},
        }

    home_players = dict([
        _pitcher(1, "Jake Bennett", 1, "6.0", 88),
        _pitcher(2, "Alec Gamboa", 0, "2.1", 34),
        _pitcher(3, "José Reliever", 0, "0.2", 12),
    ])
    away_players = dict([
        _pitcher(4, "Away Starter", 1, "5.0", 95),
        _pitcher(5, "Away Closer", 0, "1.0", 15),
    ])
    return {
        "teams": {
            "home": {"team": {"id": 111}, "pitchers": [1, 2, 3], "players": home_players},
            "away": {"team": {"id": 147}, "pitchers": [4, 5], "players": away_players},
        }
    }


class TestParseRelieverAppearances:
    def test_starters_excluded_relievers_extracted(self):
        out = parse_reliever_appearances(_boxscore_fixture(), "2026-07-18")
        assert "jake bennett" not in out
        assert "away starter" not in out
        assert out["alec gamboa"] == ("2026-07-18", pytest.approx(2 + 1 / 3), 34)
        assert out["away closer"] == ("2026-07-18", 1.0, 15)

    def test_names_accent_folded(self):
        out = parse_reliever_appearances(_boxscore_fixture(), "2026-07-18")
        assert "jose reliever" in out  # 'José' folded

    def test_empty_payload_safe(self):
        assert parse_reliever_appearances({}, "2026-07-18") == {}
