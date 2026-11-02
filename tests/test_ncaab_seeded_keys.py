"""Pinnacle NCAAB spellings must land on the Elo keys seeded from ESPN.

Elo is seeded from the full D1 scoreboard (seed_espn.py, groups=50), so its
keys are ESPN displayNames ("little rock trojans"). Before the 2026-11-02
aliases, the word-boundary fallback priced "Arkansas Little Rock" off Arkansas
and "Texas San Antonio" off Texas, and nine other Pinnacle teams resolved to
nothing.
"""

from __future__ import annotations

import pytest

from evmax.agents.models._team_lookup import resolve_team_key, write_team_key

# Keys exactly as seed_espn.py writes them from ESPN displayNames.
SEEDED = {k: 1500.0 for k in (
    "little rock trojans", "arkansas", "utsa", "texas", "east tennessee state buccaneers",
    "miami oh redhawks", "miami", "san josé state spartans", "saint peters peacocks",
    "ut martin skyhawks", "tennessee", "louisiana ragin cajuns", "louisiana tech bulldogs",
    "lafayette", "ul monroe warhawks", "kansas city roos", "kansas", "milwaukee panthers",
)}


@pytest.mark.parametrize("label,key", [
    ("Arkansas Little Rock", "little rock trojans"),
    ("Texas San Antonio", "utsa"),
    ("East Tenn State", "east tennessee state buccaneers"),
    ("Miami Ohio", "miami oh redhawks"),
    ("San Jose State", "san josé state spartans"),
    ("St. Peter's", "saint peters peacocks"),
    ("Tennessee Martin", "ut martin skyhawks"),
    ("UL - Lafayette", "louisiana ragin cajuns"),
    ("UL - Monroe", "ul monroe warhawks"),
    ("UMKC", "kansas city roos"),
    ("Wisc Milwaukee", "milwaukee panthers"),
])
def test_pinnacle_spelling_resolves_to_the_seeded_key(label, key):
    assert resolve_team_key("ncaab", label.lower().strip(), SEEDED) == key


@pytest.mark.parametrize("label,key", [
    ("Arkansas", "arkansas"), ("Texas", "texas"), ("Miami Florida", "miami"),
    ("Lafayette", "lafayette"), ("Louisiana Tech", "louisiana tech bulldogs"),
    ("Tennessee", "tennessee"), ("Kansas", "kansas"),
])
def test_namesake_schools_are_not_captured(label, key):
    assert resolve_team_key("ncaab", label.lower().strip(), SEEDED) == key


def test_daily_feed_writes_onto_the_seeded_key_not_a_new_one():
    # ESPN displayName through the feed's write path: no split entity.
    assert write_team_key("ncaab", "little rock trojans", SEEDED) == "little rock trojans"
    assert write_team_key("ncaab", "kansas city roos", SEEDED) == "kansas city roos"
