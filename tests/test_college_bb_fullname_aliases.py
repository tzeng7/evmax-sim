"""College-basketball Elo/Form writes land on the stored full-name record.

The ncaab / ncaaw Elo and Form stores are keyed by ESPN full names
("idaho state bengals"), while logged predictions and Pinnacle use the school
name ("idaho state"). Reads bridged the two through the fuzzy fallback; writes
are identity-only, so before the generated full-name aliases a result fed under
the short name created a second "idaho state" record at the default rating, and
later reads preferred that 1-game record (Idaho State Elo n=1, Form missing on
the 2026-11-19 slate). Every OLD note is what the pre-alias code did.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evmax.agents.models._team_lookup import resolve_team_key, write_team_key
from evmax.agents.models.elo_agent import EloModelAgent
from evmax.agents.models.form_agent import FormModelAgent
from evmax.matching.normalizer import NameNormalizer

ROOT = Path(__file__).resolve().parent.parent


def _elo(sector: str, ratings: dict[str, float]) -> EloModelAgent:
    agent = EloModelAgent()
    agent._state = {
        sector: {
            "ratings": dict(ratings),
            "game_counts": {k: 90 for k in ratings},
            "season_games": {k: 3 for k in ratings},
            "h2h": {},
        }
    }
    return agent


@pytest.mark.parametrize(
    "sector,short,full",
    [
        ("ncaab", "idaho state", "idaho state bengals"),
        ("ncaab", "cornell", "cornell big red"),
        ("ncaab", "eastern kentucky", "eastern kentucky colonels"),
        ("ncaab", "liberty", "liberty flames"),
        ("ncaaw", "western carolina", "western carolina catamounts"),
    ],
)
def test_short_name_write_finds_full_name_record(sector, short, full):
    store = {full: 1600.0, "some other school": 1500.0}
    # OLD: write_team_key returned the short name (a new record).
    assert write_team_key(sector, short, store) == full
    assert resolve_team_key(sector, short, store, allow_fuzzy=False) == full


def test_elo_update_under_short_name_updates_existing_record():
    agent = _elo("ncaab", {"idaho state bengals": 1429.0, "montana state bobcats": 1550.0})
    agent.update("idaho state", "montana state", 80, 70, "ncaab", event_date="2026-11-20")
    ratings = agent._state["ncaab"]["ratings"]
    # OLD: a new "idaho state" key born at 1500 took the win.
    assert "idaho state" not in ratings
    assert ratings["idaho state bengals"] > 1429.0
    assert agent._state["ncaab"]["game_counts"]["idaho state bengals"] == 91


def test_form_update_under_short_name_appends_to_existing_record():
    agent = FormModelAgent()
    agent._state = {"ncaab": {"liberty flames": [
        {"date": "2026-11-13", "won": True, "opp": "x", "home": True, "drew": False, "margin": 3.0},
    ]}}
    agent.update("liberty", "rhode island", 70, 60, "ncaab", event_date="2026-11-20")
    store = agent._state["ncaab"]
    # OLD: a separate "liberty" list held the new game.
    assert "liberty" not in store
    assert [r["date"] for r in store["liberty flames"]] == ["2026-11-20", "2026-11-13"]


@pytest.mark.parametrize("sector", ["ncaab", "ncaaw"])
def test_every_espn_full_name_reaches_its_school_key(sector):
    """Pins the generated block against the shipped efficiency state (same ESPN team id)."""
    teams = json.loads((ROOT / "data" / "models" / f"{sector}_efficiency_state.json").read_text())["teams"]
    norm = NameNormalizer(sector)
    misses = [
        (key, rec["full_name"])
        for key, rec in teams.items()
        if rec.get("full_name") and norm.normalize(rec["full_name"]) != norm.normalize(key)
    ]
    assert misses == []


def test_distinct_schools_stay_distinct():
    norm = NameNormalizer("ncaab")
    pairs = [
        ("idaho state bengals", "idaho vandals"),
        ("south dakota state jackrabbits", "south dakota coyotes"),
        ("montana state bobcats", "montana grizzlies"),
        ("texas a m aggies", "east texas a m lions"),
        ("northwestern wildcats", "northwestern state demons"),
    ]
    for a, b in pairs:
        assert norm.normalize(a) != norm.normalize(b), (a, b)
