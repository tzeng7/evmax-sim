"""Unit + guard tests for the shared model team-lookup rule (_team_lookup.py).

The agent-level regressions (each failing on the pre-fix code) live in
tests/test_model_name_resolution.py. This file pins the rule itself, the
behaviour that must NOT change for ordinary names, and the integrity of the
alias data the rule depends on.
"""

from __future__ import annotations

import collections
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from evmax.agents.models._team_lookup import (
    IDENTITY_ONLY_SECTORS,
    _canonical,
    identity_team_key,
    resolve_team_key,
    write_team_key,
)
from evmax.matching.normalizer import NameNormalizer
from evmax.sectors.base import fold_accents

ROOT = Path(__file__).resolve().parents[1]
ALIASES = ROOT / "evmax" / "sectors" / "aliases"


# ---------------------------------------------------------------------------
# Ordinary names keep resolving exactly as before
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("sector", "label", "store", "expected"),
    [
        ("nba", "Los Angeles Lakers", {"lakers": 1, "clippers": 2}, "lakers"),
        ("nfl", "Kansas City Chiefs", {"chiefs": 1, "cardinals": 2}, "chiefs"),
        ("baseball", "G1 Boston Red Sox", {"red sox": 1, "white sox": 2}, "red sox"),
        ("baseball", "G2 Colorado Rockies", {"rockies": 1}, "rockies"),       # state name in the head is a city
        ("baseball", "G1 St. Louis Cardinals", {"cardinals": 1}, "cardinals"),  # "st." in the head is "Saint"
        ("lol", "Kiwoom DRX", {"drx": 1}, "drx"),
        ("ncaab", "VCU", {"vcu rams": 1}, "vcu rams"),
        ("ncaab", "Duke Blue Devils", {"duke": 1}, "duke"),
        ("ufc", "Levi Rodrigues Jr.", {"levi rodrigues": 1}, "levi rodrigues"),
        ("soccer", "Manchester City", {"man city": 1, "man united": 2}, "man city"),
        ("soccer", "Paris FC", {"paris": 1, "psg": 2}, "paris"),
        ("ncaaf", "Kansas State", {"kansas": 1, "kansas state": 2}, "kansas state"),
        ("ncaaf", "Miami Ohio", {"miami": 1, "miami oh": 2}, "miami oh"),
        ("worldcup", "USA", {"united states": 1}, "united states"),
    ],
)
def test_ordinary_names_resolve(sector, label, store, expected):
    assert resolve_team_key(sector, label, store) == expected


def test_empty_inputs_resolve_to_none():
    assert resolve_team_key("nba", "", {"lakers": 1}) is None
    assert resolve_team_key("nba", None, {"lakers": 1}) is None
    assert resolve_team_key("nba", "lakers", {}) is None


# ---------------------------------------------------------------------------
# The rule's guards
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("sector", "label", "store"),
    [
        ("soccer", "Inter Miami", {"inter": 1}),              # identity-only sector
        ("soccer", "Inter Turku", {"inter": 1}),
        ("worldcup", "Equatorial Guinea", {"guinea": 1}),
        ("ncaaf", "Alabama State", {"alabama": 1}),
        # Unaliased mascot form: resolves in ncaab (fuzzy tier), never in ncaaf.
        ("ncaaf", "Zeta College Hawks", {"zeta college": 1}),
    ],
)
def test_identity_only_sectors_never_fuzzy_match(sector, label, store):
    assert sector in IDENTITY_ONLY_SECTORS
    assert resolve_team_key(sector, label, store) is None


def test_the_same_mascot_form_resolves_where_fuzzy_is_allowed():
    assert resolve_team_key("ncaab", "Zeta College Hawks", {"zeta college": 1}) == "zeta college"


def test_ambiguous_fuzzy_match_is_refused():
    assert resolve_team_key("cs2", "alpha", {"alpha one": 1, "alpha two": 2}) is None


def test_nested_candidates_collapse_to_the_most_specific():
    store = {"georgia": 1, "georgia state": 2}
    assert resolve_team_key("ncaab", "georgia state panthers", store) == "georgia state"


def test_registered_team_may_match_a_longer_key_but_never_a_shorter_one():
    # "texas a&m" is a registered ncaab canonical: it must not borrow "texas".
    assert NameNormalizer("ncaab").is_known_team("texas a&m")
    assert resolve_team_key("ncaab", "Texas A&M", {"texas": 1}) is None
    assert resolve_team_key("ncaab", "Texas A&M", {"texas": 1, "texas a m aggies": 2}) == "texas a m aggies"


def test_person_name_sectors_do_not_fuzzy_on_the_surname():
    # UFC normalizes "Apollo Gomes" to the bare surname "gomes".
    assert resolve_team_key("ufc", "Apollo Gomes", {"denise gomes": 1}) is None
    # Exact full name still wins over a stored surname-only key.
    assert resolve_team_key("ufc", "Jon Jones", {"jon jones": 1, "jones": 2}) == "jon jones"


def test_identity_resolution_never_uses_the_fuzzy_tier():
    store = {"georgia state": 1}
    assert resolve_team_key("ncaab", "georgia state panthers", store) == "georgia state"
    assert identity_team_key("ncaab", "georgia state panthers", store) is None


def test_canonical_equality_index_sees_keys_added_later():
    store: dict[str, int] = {"arsenal": 1}
    assert resolve_team_key("soccer", "CF Montreal", store) is None
    store["montréal"] = 2
    assert resolve_team_key("soccer", "CF Montreal", store) == "montréal"


def test_write_key_prefers_existing_then_alias_canonical_then_raw():
    assert write_team_key("soccer", "montreal", {"montréal": 1}) == "montréal"
    assert write_team_key("ncaaf", "Alabama State Hornets", {}) == "alabama state"
    # "team a" normalizes to "a" (noise word) — not a registered team, keep raw.
    assert write_team_key("lol", "Team A", {}) == "team a"


# ---------------------------------------------------------------------------
# Accent folding — soccer-like sectors only
# ---------------------------------------------------------------------------

def test_fold_accents_handles_non_decomposing_letters():
    assert fold_accents("Bodø/Glimt") == "Bodo/Glimt"
    assert fold_accents("københavn") == "kobenhavn"
    assert fold_accents("Łódź") == "Lodz"
    assert fold_accents("Montréal") == "Montreal"      # case is preserved
    assert fold_accents("ascii stays") == "ascii stays"


def test_accent_folding_is_scoped_to_soccer_like_sectors():
    # NCAAF's canonical keeps ESPN's accent; UFC fighter keys are accented.
    assert NameNormalizer("ncaaf").normalize("San José State") == "san josé state"
    assert NameNormalizer("ncaaf").normalize("San Jose State") == "san josé state"
    assert NameNormalizer("ufc").normalize("Jiří Procházka") == "procházka"
    assert NameNormalizer("worldcup").normalize("Curaçao") == NameNormalizer("worldcup").normalize("Curacao")


@pytest.mark.parametrize("sector", ["soccer", "worldcup"])
def test_alias_keys_fold_without_collision(sector):
    """Two alias keys that differ only by accents must point at one club."""
    raw = yaml.safe_load((ALIASES / f"{sector}.yaml").read_text())["aliases"]
    folded: dict[str, set[str]] = collections.defaultdict(set)
    for key, target in raw.items():
        folded[fold_accents(str(key))].add(fold_accents(str(target)))
    clashes = {k: v for k, v in folded.items() if len(v) > 1}
    assert clashes == {}


@pytest.mark.parametrize("sector", ["soccer", "worldcup"])
def test_shipped_state_keys_do_not_merge_under_folding(sector):
    """No two distinct clubs in shipped Elo state may share a canonical once
    accents are folded — that would silently merge their ratings."""
    state = json.loads((ROOT / "data" / "models" / "elo_state.json").read_text())
    ratings = state.get(sector, {}).get("ratings", {})
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for key in ratings:
        groups[_canonical(sector, key)].append(key)
    merged = {c: ks for c, ks in groups.items() if len(ks) > 1}
    assert merged == {}


# ---------------------------------------------------------------------------
# Alias data integrity
# ---------------------------------------------------------------------------

def _load_ncaaf_builder():
    spec = importlib.util.spec_from_file_location(
        "build_ncaaf_aliases", ROOT / "scripts" / "build_ncaaf_aliases.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_curated_ncaaf_aliases_are_present_in_the_generated_yaml():
    """ncaaf.yaml is generated (network-only builder). Curated overrides added
    by hand must match what a regeneration would emit, or the next rebuild
    silently changes behaviour."""
    builder = _load_ncaaf_builder()
    aliases = yaml.safe_load((ALIASES / "ncaaf.yaml").read_text())["aliases"]
    missing = {
        builder._norm_key(k): builder._norm_key(v)
        for k, v in builder.CURATED_ALIASES.items()
        if aliases.get(builder._norm_key(k)) != builder._norm_key(v)
    }
    assert missing == {}


@pytest.mark.parametrize(
    ("label", "canonical"),
    [
        ("Middle Tennessee State", "middle tennessee"),
        ("Tennessee Martin", "ut martin"),
        ("West Georgia", "west georgia"),
        ("Gardner Webb", "gardner-webb"),
        ("Bethune Cookman", "bethune-cookman"),
        ("Nicholls State", "nicholls"),
        ("Albany", "ualbany"),
        ("Alabama State Hornets", "alabama state"),
        ("Dixie State", "utah tech"),
    ],
)
def test_ncaaf_split_spellings_share_one_canonical(label, canonical):
    assert NameNormalizer("ncaaf").normalize(label) == canonical


@pytest.mark.parametrize(
    ("espn", "live"),
    [
        ("Brighton & Hove Albion", "Brighton"),
        ("Red Bull New York", "New York Red Bulls"),
        ("Union St.-Gilloise", "Union Saint-Gilloise"),
        ("RB Salzburg", "Salzburg"),
        ("TSG Hoffenheim", "Hoffenheim"),
        ("VfL Wolfsburg", "Wolfsburg"),
        ("CF Montréal", "CF Montreal"),
    ],
)
def test_soccer_seed_and_live_spellings_share_one_canonical(espn, live):
    norm = NameNormalizer("soccer")
    assert norm.normalize(espn) == norm.normalize(live)


def test_nba_clippers_espn_spelling_shares_the_pinnacle_canonical():
    norm = NameNormalizer("nba")
    assert norm.normalize("LA Clippers") == norm.normalize("Los Angeles Clippers") == "clippers"
