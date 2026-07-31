"""Tests for the shared tennis player-name resolver (tennis_common.py).

Regression context (2026-07-01 audit):

1. Hyphenated surnames never resolved: Tennis Abstract state keys use the
   space form ('felix auger aliassime') while Kalshi/Pinnacle send
   'Felix Auger-Aliassime'. Every FAA match since May logged
   model_sources='tennis_surface+sharp' — serve/return, form, and advanced
   all silently dropped, so the full-blend gate shadow-demoted the ATP #8.

2. Same-surname collisions resolved to an arbitrary wrong player: a
   'xinyu wang' query returned Xiyu Wang's data because the duplicate-seed
   tie-break (max game count) is a dead heat on Tennis Abstract's uniform
   synthetic counts. Now the given name narrows candidates, and genuinely
   distinct players (different first initials) return None instead of a
   coin-flip identity.
"""

from __future__ import annotations

from evmax.agents.models.tennis_common import (
    given_name,
    normalize_player,
    resolve_player,
    surname,
)


class TestGivenName:
    def test_western_full_name(self):
        assert given_name("jannik sinner") == "jannik"

    def test_multi_token_given_name(self):
        assert given_name("xin yu wang") == "xin yu"

    def test_trailing_initial_form(self):
        assert given_name("mannarino a.") == "a"

    def test_surname_only(self):
        assert given_name("sinner") == ""


class TestHyphenatedSurnames:
    STORE = {"felix auger aliassime": 2000.0, "jan lennard struff": 1800.0}

    def test_hyphenated_full_name_resolves_space_form(self):
        assert (
            resolve_player("felix auger-aliassime", self.STORE)
            == "felix auger aliassime"
        )

    def test_hyphenated_bare_surname_resolves(self):
        # TennisHandler.normalize_team collapses to the last token, keeping
        # the hyphen: 'Felix Auger-Aliassime' → 'auger-aliassime'
        assert resolve_player("auger-aliassime", self.STORE) == "felix auger aliassime"

    def test_hyphen_in_given_name_still_resolves(self):
        assert resolve_player("jan-lennard struff", self.STORE) == "jan lennard struff"

    def test_hyphenated_store_key_matches_space_query(self):
        store = {"felix auger-aliassime": 2000.0}
        assert resolve_player("auger aliassime", store) == "felix auger-aliassime"


class TestSameSurnameCollisions:
    WANGS = {
        "xiyu wang": 1700.0,
        "xin yu wang": 1710.0,
        "yafan wang": 1690.0,
        "jiaqi wang": 1650.0,
    }

    def test_given_name_narrows_to_correct_player(self):
        # 'xinyu' normalizes equal to 'xin yu' — must NOT return Xiyu Wang
        assert resolve_player("xinyu wang", self.WANGS) == "xin yu wang"

    def test_bare_surname_with_distinct_players_returns_none(self):
        # Four different people — an arbitrary pick is wrong-player data
        assert resolve_player("wang", self.WANGS) is None

    def test_bare_surname_distinct_players_none_even_with_weight_fn(self):
        assert resolve_player("wang", self.WANGS, weight_fn=lambda k: 1.0) is None

    def test_sibling_pair_bare_surname_returns_none(self):
        store = {"francisco cerundolo": 1900.0, "juan manuel cerundolo": 1750.0}
        assert resolve_player("cerundolo", store) is None
        assert resolve_player("francisco cerundolo", store) == "francisco cerundolo"
        assert (
            resolve_player("juan manuel cerundolo", store) == "juan manuel cerundolo"
        )

    def test_incompatible_given_name_returns_none(self):
        # Query names a Wang the store doesn't have — never guess a sibling
        store = {"xiyu wang": 1700.0, "yafan wang": 1690.0}
        assert resolve_player("xinyu wang", store) is None


class TestSamePlayerDuplicates:
    def test_duplicate_seed_entries_pick_by_weight(self):
        # 'adrian mannarino' + 'mannarino a.' are the same person (initials
        # agree) — the weight_fn tie-break must still fire
        store = {"adrian mannarino": 1500.0, "mannarino a.": 1620.0}
        games = {"adrian mannarino": 5, "mannarino a.": 50}
        assert (
            resolve_player("mannarino", store, weight_fn=lambda k: games[k])
            == "mannarino a."
        )

    def test_duplicate_without_weight_fn_returns_first(self):
        store = {"adrian mannarino": 1500.0, "mannarino a.": 1620.0}
        assert resolve_player("mannarino", store) == "adrian mannarino"

    def test_initial_query_matches_full_given_name(self):
        store = {"jannik sinner": 1900.0}
        assert resolve_player("sinner j.", store) == "jannik sinner"


class TestExistingCascade:
    """The pre-fix cascade must keep working unchanged."""

    def test_exact_match(self):
        assert resolve_player("sinner", {"sinner": 1.0}) == "sinner"

    def test_surname_long_to_short(self):
        assert resolve_player("jannik sinner", {"sinner j.": 1.0}) == "sinner j."

    def test_surname_short_to_long(self):
        assert resolve_player("sinner", {"jannik sinner": 1.0}) == "jannik sinner"

    def test_multi_word_surname(self):
        assert resolve_player("de minaur", {"de minaur a.": 1.0}) == "de minaur a."

    def test_apostrophe_normalization(self):
        assert resolve_player("oconnell", {"o'connell c.": 1.0}) == "o'connell c."
        assert normalize_player("o'connell") == "oconnell"

    def test_no_match_returns_none(self):
        assert resolve_player("federer", {"sinner": 1.0}) is None

    def test_surname_helper(self):
        assert surname("jannik sinner") == "sinner"
        assert surname("sinner j.") == "sinner"


class TestAccentFolding:
    """Regression (2026-07-30): ESPN-seeded UFC state keys carry diacritics
    ('duško todorović', 'uroš medić') while Kalshi/Pinnacle send ASCII, so
    every lookup for an accented fighter returned None and the whole card
    went sharp-only (model_diagnostics: missing=[ufc_rating])."""

    STORE = {
        "duško todorović": 1,
        "uroš medić": 2,
        "julianna peña": 3,
        "jiří procházka": 4,
        "hailey cowan": 5,
    }

    def test_ascii_full_name_resolves_diacritic_key(self):
        assert resolve_player("dusko todorovic", self.STORE) == "duško todorović"
        assert resolve_player("uros medic", self.STORE) == "uroš medić"

    def test_ascii_surname_resolves_diacritic_key(self):
        assert resolve_player("prochazka", self.STORE) == "jiří procházka"

    def test_tilde_and_caron_fold(self):
        assert resolve_player("julianna pena", self.STORE) == "julianna peña"

    def test_accented_query_resolves_ascii_key(self):
        store = {"dusko todorovic": 1}
        assert resolve_player("duško todorović", store) == "dusko todorovic"

    def test_stroked_letters_survive_folding(self):
        # đ/ø/ł don't decompose under NFKD; a bare ascii-ignore would delete
        # them ('đorđe' → 'ore') and break the surname comparison.
        assert normalize_player("Đorđe Petrović") == "dordepetrovic"
        store = {"đorđe petrović": 1}
        assert resolve_player("dorde petrovic", store) == "đorđe petrović"

    def test_unaccented_names_unaffected(self):
        assert resolve_player("hailey cowan", self.STORE) == "hailey cowan"

    def test_no_match_still_returns_none(self):
        assert resolve_player("nina milosevic", self.STORE) is None
