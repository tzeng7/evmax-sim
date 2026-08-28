"""Unit tests for the Kalshi-code derivation in scripts/build_ncaaf_aliases.py.

The generator turns Kalshi's live KXNCAAFGAME markets into code→canonical alias
entries by normalizing each market's ``yes_sub_title``. Two pieces carry real
logic and must be guarded:

  * ``_expand_state_abbrev`` — Kalshi abbreviates "State" as "St."; the canonical
    map spells it out, so a trailing "St." must expand before lookup.
  * ``derive_kalshi_code_aliases`` — resolves codes against the built map, drops
    non-FBS buy-game opponents, and NEVER guesses a code Kalshi reuses across two
    schools (the Miami FL/OH · same-surname rule).
"""

from __future__ import annotations

from scripts.build_ncaaf_aliases import (
    _expand_state_abbrev,
    derive_kalshi_code_aliases,
)


class TestExpandStateAbbrev:
    def test_trailing_period_form(self):
        assert _expand_state_abbrev("Penn St.") == "Penn State"

    def test_trailing_no_period_form(self):
        assert _expand_state_abbrev("Ohio St") == "Ohio State"

    def test_multiword_school(self):
        assert _expand_state_abbrev("San Jose St.") == "San Jose State"

    def test_no_trailing_st_unchanged(self):
        assert _expand_state_abbrev("Alabama") == "Alabama"
        assert _expand_state_abbrev("Georgia Tech") == "Georgia Tech"

    def test_leading_saint_not_expanded(self):
        # A leading "St." means "Saint", not "State" — must be left alone.
        assert _expand_state_abbrev("St. Francis") == "St. Francis"

    def test_embedded_state_word_unchanged(self):
        # Only a word-final "St." is expanded; "State" already spelled is a no-op.
        assert _expand_state_abbrev("Colorado State Rams") == "Colorado State Rams"


class TestDeriveKalshiCodeAliases:
    # A minimal built map: keys are variant→canonical; the canonical set is the
    # VALUES. Mirrors how build_alias_map emits entries (canonical never a key).
    ALIASES = {
        "penn state nittany lions": "penn state",
        "colorado state rams": "colorado state",
        "georgia tech yellow jackets": "georgia tech",
        # Curated hyphen spelling Kalshi uses for ULM.
        "louisiana-monroe": "ul monroe",
    }

    def test_clean_fbs_code_resolves(self):
        resolved, skipped, ambiguous, collisions = derive_kalshi_code_aliases(
            {"psu": {"Penn St."}}, dict(self.ALIASES)
        )
        assert resolved == {"psu": "penn state"}
        assert not skipped and not ambiguous and not collisions

    def test_hyphen_curated_code_resolves(self):
        resolved, _, _, _ = derive_kalshi_code_aliases(
            {"ulm": {"Louisiana-Monroe"}}, dict(self.ALIASES)
        )
        assert resolved == {"ulm": "ul monroe"}

    def test_reused_code_is_ambiguous_never_resolved(self):
        # csu → Colorado St. (known/FBS) AND Central State (OH) (unknown/D2):
        # two distinct canonicals → must be flagged ambiguous, never guessed.
        resolved, skipped, ambiguous, collisions = derive_kalshi_code_aliases(
            {"csu": {"Colorado St.", "Central State (OH) Marauders"}},
            dict(self.ALIASES),
        )
        assert "csu" not in resolved
        assert [c for c, _ in ambiguous] == ["csu"]

    def test_non_fbs_code_is_skipped(self):
        # A code whose only name resolves to nothing the map knows is dropped —
        # it can never match Pinnacle's FBS-only feed.
        resolved, skipped, ambiguous, collisions = derive_kalshi_code_aliases(
            {"chs": {"Chicago State Cougars"}}, dict(self.ALIASES)
        )
        assert resolved == {}
        assert [c for c, _ in skipped] == ["chs"]
        assert not ambiguous

    def test_two_name_forms_same_canonical_is_not_ambiguous(self):
        # Two spellings of the SAME school (both → colorado state) are safe:
        # one distinct canonical, so it resolves rather than flagging ambiguous.
        resolved, skipped, ambiguous, collisions = derive_kalshi_code_aliases(
            {"cst": {"Colorado St.", "Colorado State Rams"}}, dict(self.ALIASES)
        )
        assert resolved == {"cst": "colorado state"}
        assert not ambiguous

    def test_mixed_batch(self):
        resolved, skipped, ambiguous, collisions = derive_kalshi_code_aliases(
            {
                "psu": {"Penn St."},
                "gt": {"Georgia Tech"},
                "ulm": {"Louisiana-Monroe"},
                "csu": {"Colorado St.", "Central State (OH) Marauders"},
                "chs": {"Chicago State Cougars"},
            },
            dict(self.ALIASES),
        )
        assert resolved == {
            "psu": "penn state",
            "gt": "georgia tech",
            "ulm": "ul monroe",
        }
        assert [c for c, _ in ambiguous] == ["csu"]
        assert [c for c, _ in skipped] == ["chs"]
        assert not collisions
