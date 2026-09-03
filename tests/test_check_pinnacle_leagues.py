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
