"""Regression tests: model-side team-name resolution (2026-09-22 audit).

Every test here drives an agent through an API that already existed before the
shared lookup rule (``evmax/agents/models/_team_lookup.py``) and FAILS on the
pre-fix code:

* Poisson priced Paris Saint-Germain as Paris FC (no normalizer + first prefix
  match), and every soccer model missed clubs whose ESPN seed spelling differs
  from the live Pinnacle canonical (accents: "montréal"; spellings: "red bull
  new york", "union st.-gilloise", …).
* NCAAF v2 priced FCS opponents as their FBS namesake (unique-prefix fallback);
  elo/form did the same through last-word / endswith fallbacks without a word
  boundary ("west georgia" → "georgia").
* ``EloModelAgent.update`` read a rating through the fuzzy fallback but wrote the
  raw key, so a new FCS team was born with the FBS namesake's rating.
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from evmax.agents.cleanup import model_updater
from evmax.agents.models.elo_agent import DEFAULT_ELO, EloModelAgent
from evmax.agents.models.form_agent import FormModelAgent
from evmax.agents.models.ncaaf_efficiency_agent import NcaafEfficiencyModelAgent
from evmax.agents.models.poisson_agent import PoissonModelAgent
from evmax.agents.models.soccer_xg_agent import SoccerXgAgent
from evmax.matching.normalizer import NameNormalizer

# ---------------------------------------------------------------------------
# Builders — every agent gets an in-memory state; nothing touches data/models/.
# ---------------------------------------------------------------------------


def _elo(sector: str, ratings: dict[str, float], counts: dict[str, int] | None = None) -> EloModelAgent:
    agent = EloModelAgent()
    counts = counts or {k: 30 for k in ratings}
    agent._state = {
        sector: {
            "ratings": dict(ratings),
            "game_counts": dict(counts),
            "season_games": dict(counts),
            "h2h": {},
        }
    }
    return agent


def _records(n: int = 6, start_day: int = 1) -> list[dict]:
    return [
        {"date": f"2026-09-{start_day + i:02d}", "won": i % 2 == 0, "opp": f"opp{i}",
         "home": True, "drew": False, "margin": 1.0}
        for i in range(n)
    ]


def _form(sector: str, teams: dict[str, list[dict]]) -> FormModelAgent:
    agent = FormModelAgent()
    agent._state = {sector: {k: list(v) for k, v in teams.items()}}
    return agent


def _poisson(sector: str, teams: dict[str, dict]) -> PoissonModelAgent:
    agent = PoissonModelAgent()
    agent._state = {sector: {"league_avg": {"home": 1.55, "away": 1.2},
                             "teams": {k: dict(v) for k, v in teams.items()}}}
    return agent


def _xg(tmp_path, teams: dict[str, int]) -> SoccerXgAgent:
    agent = SoccerXgAgent()
    agent._state_path = tmp_path / "soccer_xg_state.json"
    agent._state = {"teams": {
        k: {"matches": [{"date": f"2026-09-{i + 1:02d}", "goals_for": 1, "goals_against": 1,
                         "sot": 4, "shots": 10, "xg": 1.3, "xga": 1.1, "is_home": True}
                        for i in range(n)]}
        for k, n in teams.items()
    }}
    return agent


# ---------------------------------------------------------------------------
# Soccer — Poisson PSG vs Paris FC
# ---------------------------------------------------------------------------

PSG = {"attack": 1.6361, "defense": 0.7490, "games": 57}
PARIS_FC = {"attack": 1.0099, "defense": 0.9581, "games": 39}


def test_poisson_prices_psg_not_paris_fc():
    agent = _poisson("soccer", {"paris": PARIS_FC, "psg": PSG})
    assert agent._team_stats("soccer", "paris saint-germain") == PSG


def test_poisson_expected_goals_use_psg_strengths():
    agent = _poisson("soccer", {"paris": PARIS_FC, "psg": PSG, "lyon": {"attack": 1.0, "defense": 1.0, "games": 40}})
    lam_psg, _ = agent._expected_goals("soccer", "paris saint-germain", "lyon")
    lam_pfc, _ = agent._expected_goals("soccer", "paris fc", "lyon")
    assert lam_psg > lam_pfc * 1.4  # 1.64 vs 1.01 attack (shrunk), not identical


def test_poisson_resolves_pinnacle_labels_through_the_alias_map():
    # Poisson had NO normalizer step: "Manchester City" never reached "man city".
    row = {"attack": 1.5, "defense": 0.8, "games": 50}
    agent = _poisson("soccer", {"man city": row, "man united": {"attack": 1.1, "defense": 1.0, "games": 50}})
    assert agent._team_stats("soccer", "manchester city") == row


# ---------------------------------------------------------------------------
# Soccer — accents: ESPN keeps them, Pinnacle / Kalshi drop them
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("espn", "pinnacle", "canonical"),
    [
        ("CF Montréal", "CF Montreal", "montreal"),
        ("Deportivo Alavés", "Alaves", "alaves"),
        ("Málaga", "Malaga", "malaga"),
        ("Malmö FF", "Malmo FF", "malmo ff"),
        ("F.C. København", "FC Copenhagen", "copenhagen"),
    ],
)
def test_soccer_normalizer_is_accent_insensitive(espn, pinnacle, canonical):
    norm = NameNormalizer("soccer")
    assert norm.normalize(espn) == norm.normalize(pinnacle) == canonical


def test_soccer_event_keys_match_across_accents():
    norm = NameNormalizer("soccer")
    assert norm.normalize_event_key("CF Montréal", "Columbus Crew", "2026-09-19", "soccer") == \
        norm.normalize_event_key("CF Montreal", "Columbus Crew", "2026-09-19", "soccer")


ACCENTED_SEED_KEYS = {"montréal": ("CF Montreal", 1373.05), "alavés": ("Alaves", 1486.28),
                      "málaga": ("Malaga", 1449.18)}


@pytest.mark.parametrize("seed_key", sorted(ACCENTED_SEED_KEYS))
def test_accented_seed_keys_resolve_for_every_soccer_model(seed_key, tmp_path):
    label, rating = ACCENTED_SEED_KEYS[seed_key]
    elo = _elo("soccer", {seed_key: rating, "arsenal": 1700.0})
    form = _form("soccer", {seed_key: _records()})
    pois = _poisson("soccer", {seed_key: PSG})
    xg = _xg(tmp_path, {seed_key: 6})

    assert elo.get_rating("soccer", label) == pytest.approx(rating)
    assert len(form._team_records("soccer", label.lower())) == 6
    assert pois._team_stats("soccer", label.lower()) == PSG
    assert xg._resolve_team_key(label, "soccer") == seed_key
    assert xg._team_xg_stats(label, "soccer") is not None


# ESPN (seed / resolve) spelling → the Pinnacle label every live lookup uses.
SEED_SPELLINGS = [
    ("red bull new york", "New York Red Bulls"),
    ("brighton hove albion", "Brighton and Hove Albion"),
    ("union st.-gilloise", "Union Saint-Gilloise"),
    ("rb salzburg", "Salzburg"),
    ("tsg hoffenheim", "Hoffenheim"),
    ("vfl wolfsburg", "Wolfsburg"),
]


@pytest.mark.parametrize(("seed_key", "label"), SEED_SPELLINGS)
def test_espn_seed_spellings_resolve_for_every_soccer_model(seed_key, label, tmp_path):
    elo = _elo("soccer", {seed_key: 1555.0, "arsenal": 1700.0})
    form = _form("soccer", {seed_key: _records()})
    pois = _poisson("soccer", {seed_key: PSG})
    xg = _xg(tmp_path, {seed_key: 6})

    resolved = {
        "elo": elo.get_rating("soccer", label) == pytest.approx(1555.0),
        "form": len(form._team_records("soccer", label.lower())) == 6,
        "poisson": pois._team_stats("soccer", label.lower()) == PSG,
        "xg": xg._resolve_team_key(label, "soccer") == seed_key,
    }
    assert all(resolved.values()), resolved


def test_soccer_updates_do_not_fork_a_seeded_accented_club(tmp_path):
    """A resolve-time update under the live spelling must land on the seeded
    record, not create a fresh 1500 / 1.0 / empty-history twin."""
    elo = _elo("soccer", {"montréal": 1373.05, "columbus": 1510.0}, {"montréal": 60, "columbus": 60})
    elo.update("montreal", "columbus", 0, 2, "soccer", "2026-09-19")
    assert "montreal" not in elo._state["soccer"]["ratings"]
    assert elo._state["soccer"]["game_counts"]["montréal"] == 61

    form = _form("soccer", {"montréal": _records(), "columbus": _records()})
    form.update("montreal", "columbus", 0, 2, "soccer", "2026-09-19")
    assert "montreal" not in form._state["soccer"]
    assert len(form._state["soccer"]["montréal"]) == 7

    pois = _poisson("soccer", {"montréal": dict(PARIS_FC), "columbus": dict(PSG)})
    pois.update("montreal", "columbus", 0, 2, "soccer", "2026-09-19")
    assert "montreal" not in pois._state["soccer"]["teams"]
    assert pois._state["soccer"]["teams"]["montréal"]["games"] == PARIS_FC["games"] + 1

    xg = _xg(tmp_path, {"montréal": 6})
    xg.record_match("montreal", 0, 2, 3, 9, 6, 14, "2026-09-19", True, sector="soccer")
    assert "montreal" not in xg._state["teams"]
    assert len(xg._state["teams"]["montréal"]["matches"]) == 7


# ---------------------------------------------------------------------------
# NCAAF — FCS teams must never price as their FBS namesake
# ---------------------------------------------------------------------------

FBS_TEAMS = {k: {"gp": 3} for k in (
    "alabama", "florida", "north carolina", "utah", "houston", "north dakota state",
    "texas", "illinois", "georgia", "tennessee", "kansas", "kansas state",
    "middle tennessee",
)}


@pytest.mark.parametrize(
    "fcs_label",
    ["Alabama State", "Florida A&M", "North Carolina Central", "Utah Tech",
     "Houston Christian", "North Dakota", "Texas Southern", "Illinois State"],
)
def test_ncaaf_v2_leaves_fcs_namesakes_unresolved(fcs_label):
    agent = NcaafEfficiencyModelAgent()
    assert agent._resolve_team(FBS_TEAMS, fcs_label) is None


@pytest.mark.parametrize(
    ("label", "store_key", "fbs_key"),
    [
        ("West Georgia", "west georgia wolves", "georgia"),   # alias → canonical-equality tier
        ("Tennessee Martin", "ut martin", "tennessee"),       # alias → canonical tier
    ],
)
def test_ncaaf_elo_and_form_resolve_fcs_team_not_fbs_namesake(label, store_key, fbs_key):
    # FBS namesake listed FIRST so the old first-match loop hits it.
    elo = _elo("ncaaf", {fbs_key: 1861.9, store_key: 1438.7})
    assert elo.get_rating("ncaaf", label) == pytest.approx(1438.7)

    form = _form("ncaaf", {fbs_key: _records(6, 1), store_key: _records(4, 11)})
    recs = form._team_records("ncaaf", label.lower())
    assert [r.date for r in recs] == [r["date"] for r in _records(4, 11)]


@pytest.mark.parametrize(("label", "fbs_key"), [("West Georgia", "georgia"), ("Utah Tech", "utah"),
                                                 ("Florida A&M", "florida"), ("Alabama State", "alabama")])
def test_ncaaf_elo_unknown_fcs_team_gets_default_not_fbs_rating(label, fbs_key):
    elo = _elo("ncaaf", {fbs_key: 1861.9})
    assert elo.get_rating("ncaaf", label) == DEFAULT_ELO


def test_elo_update_writes_the_key_it_reads():
    """The FCS rating-inheritance bug: 'alabama state hornets' was READ as
    Alabama (1807) through the fallback and WRITTEN under its own name."""
    elo = _elo("ncaaf", {"alabama": 1807.1, "jackson state": 1450.0}, {"alabama": 73, "jackson state": 20})
    elo.update("alabama state hornets", "jackson state", 7, 35, "ncaaf", "2026-09-13")
    ratings = elo._state["ncaaf"]["ratings"]
    assert ratings["alabama"] == pytest.approx(1807.1)
    assert elo._state["ncaaf"]["game_counts"]["alabama"] == 73
    assert "alabama state hornets" not in ratings
    # New team starts from the default rating (under its alias canonical) and lost.
    assert 1400.0 < ratings["alabama state"] < DEFAULT_ELO


def test_elo_update_merges_into_canonical_not_the_split_key():
    elo = _elo(
        "ncaaf",
        {"middle tennessee": 1392.5, "middle tennessee state": 1412.8, "wku": 1500.0},
        {"middle tennessee": 64, "middle tennessee state": 1, "wku": 60},
    )
    assert elo.get_rating("ncaaf", "Middle Tennessee State") == pytest.approx(1392.5)
    elo.update("middle tennessee state", "wku", 30, 10, "ncaaf", "2026-09-19")
    counts = elo._state["ncaaf"]["game_counts"]
    assert counts["middle tennessee"] == 65
    assert counts["middle tennessee state"] == 1


# ---------------------------------------------------------------------------
# Kansas / Kansas State (fuzzy-enabled college sector)
# ---------------------------------------------------------------------------

def test_kansas_state_never_borrows_kansas():
    elo = _elo("ncaab", {"kansas": 1700.0})
    assert elo.get_rating("ncaab", "Kansas State") == DEFAULT_ELO


def test_state_school_mascot_form_picks_the_specific_school():
    # ncaab's alias map has no "georgia state panthers" entry, so this reaches
    # the fuzzy tier. The namesake is listed FIRST so the old first-match loop
    # took it ("georgia state panthers".startswith("georgia ")).
    elo = _elo("ncaab", {"georgia": 1700.0, "georgia state": 1550.0})
    assert elo.get_rating("ncaab", "Georgia State Panthers") == pytest.approx(1550.0)


@pytest.mark.parametrize(("label", "wrong_key"), [("George Washington", "washington"),
                                                   ("Central Florida", "florida"),
                                                   ("Miami Ohio", "miami"),
                                                   ("Illinois State", "illinois")])
def test_college_basketball_never_maps_to_a_shorter_different_school(label, wrong_key):
    elo = _elo("ncaab", {wrong_key: 1700.0})
    assert elo.get_rating("ncaab", label) == DEFAULT_ELO


# ---------------------------------------------------------------------------
# Pro leagues — split keys and seed spellings
# ---------------------------------------------------------------------------

def test_nba_canonical_outranks_a_stale_split_key():
    elo = _elo("nba", {"celtics": 1657.66, "boston celtics": 1725.0}, {"celtics": 87, "boston celtics": 1})
    assert elo.get_rating("nba", "Boston Celtics") == pytest.approx(1657.66)
    assert elo._get_count("nba", "boston celtics") == 87


def test_nba_clippers_espn_seed_key_resolves():
    elo = _elo("nba", {"la clippers": 1522.53, "lakers": 1600.0})
    assert elo.get_rating("nba", "Los Angeles Clippers") == pytest.approx(1522.53)


# ---------------------------------------------------------------------------
# Esports — no substring matches, no academy / second-team borrowing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("sector", "label", "main_key"),
    [("cs2", "CarritoSpain", "pain"), ("cs2", "Matrix", "atrix"), ("lol", "GD", "lgd"),
     ("cs2", "ENCE Academy", "ence"), ("cs2", "FURIA Female", "furia"),
     ("lol", "T1 Academy", "t1"), ("lol", "Hanwha Life Challengers", "hanwha life"),
     ("cs2", "MOUZ NXT", "mouz")],
)
def test_esports_never_borrows_a_substring_or_main_roster(sector, label, main_key):
    elo = _elo(sector, {main_key: 1650.0})
    assert elo.get_rating(sector, label) == DEFAULT_ELO


# ---------------------------------------------------------------------------
# Resolve-time ledger — canonical comparison
# ---------------------------------------------------------------------------

def _ledger_db(tmp_path, rows: list[tuple[str, str, str, str]]):
    db = tmp_path / "predictions.db"
    seed = sqlite3.connect(str(db))
    seed.execute("CREATE TABLE ev_predictions (id INTEGER PRIMARY KEY, event_id TEXT, sector TEXT, event_date TEXT)")
    seed.execute(
        """CREATE TABLE applied_model_games (
            id INTEGER PRIMARY KEY, applied_at TEXT DEFAULT (datetime('now')),
            sector TEXT NOT NULL, event_date TEXT NOT NULL,
            team_a TEXT NOT NULL, team_b TEXT NOT NULL,
            UNIQUE(sector, event_date, team_a, team_b))"""
    )
    seed.executemany(
        "INSERT INTO applied_model_games (sector, event_date, team_a, team_b) VALUES (?, ?, ?, ?)", rows
    )
    seed.commit()
    seed.close()

    def _open():
        conn = sqlite3.connect(str(db))
        conn.row_factory = sqlite3.Row
        return conn

    return _open


def _run_update(sector, day, scores, opener):
    coord = MagicMock()
    with patch.object(model_updater, "fetch_completed_scores", return_value=scores), \
         patch.object(model_updater, "get_connection", side_effect=opener):
        result = asyncio.run(model_updater.update_models_for_date([sector], day, coordinator=coord))
    return result, coord


def test_ledger_matches_a_game_recorded_under_another_spelling(tmp_path):
    """A game ledgered under one spelling of the pair must not be fed again
    when the same game arrives under another spelling (the trailing 7-day
    backfill re-reads the date after every alias change)."""
    day = date(2026, 9, 19)
    opener = _ledger_db(tmp_path, [("nba", day.isoformat(), "boston celtics", "miami heat")])
    scores = [{"home_name": "Boston Celtics", "away_name": "Miami Heat", "home_score": 110, "away_score": 102}]
    result, coord = _run_update("nba", day, scores, opener)
    assert result.skipped == 1
    assert coord.update_models.call_count == 0
