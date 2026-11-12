"""scripts/check_pinnacle_leagues.py — league-id drift detector (pure classifier)."""

from __future__ import annotations

from scripts.check_pinnacle_leagues import classify_league_ids


def test_ok_and_stale_split_preserves_order():
    listed = {876: "Canadian Football", 880: "NCAA", 889: "NFL"}
    v = classify_league_ids([258, 880], listed)
    assert v["stale"] == [258]
    assert v["ok"] == [(880, "NCAA")]


def test_all_ok_when_every_id_is_served():
    v = classify_league_ids([889], {889: "NFL"})
    assert v == {"ok": [(889, "NFL")], "stale": []}


def test_empty_served_map_marks_everything_stale():
    v = classify_league_ids([487], {})
    assert v == {"ok": [], "stale": [487]}


def test_id_in_catalog_but_not_listed_is_idle_not_stale():
    # WNCAA 583, 2026-11-11: valid id, zero open matchups, so the default
    # (active-only) league list omits it.
    v = classify_league_ids([493, 583], {493: "NCAA"}, {493: "NCAA", 583: "WNCAA"})
    assert v == {"ok": [(493, "NCAA")], "stale": [], "idle": [(583, "WNCAA")]}


def test_id_absent_from_catalog_stays_stale():
    v = classify_league_ids([258], {889: "NFL"}, {889: "NFL", 880: "NCAA"})
    assert v == {"ok": [], "stale": [258], "idle": []}
