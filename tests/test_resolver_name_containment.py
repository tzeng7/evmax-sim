"""Golden tests: outcome resolution when one team's name contains the other's.

Regression for the 2026-09-22 resolver audit. ``_match_espn`` used
``rapidfuzz.token_set_ratio`` to place the YES team on the event's slug teams
and broke ties toward team A; ``token_set_ratio`` scores every token-subset
pair at 100 ("utah" vs "utah state"), so the containing team's bet was graded
on the contained team's result. Four NCAAF rows were graded WON on games the
favorite won (Western Michigan, Florida Atlantic, Washington State, Utah State).
``yes_aligned_close_prob`` had the same class of bug through substring
matching (side A checked first).

Every pair is exercised in both YES directions and both home/away orders, and
with the event slug in both orders — the old code was right by luck whenever
the YES team happened to be the slug's first team.
"""

from __future__ import annotations

import itertools

import pytest

from evmax.agents.cleanup.resolver import (
    _match_bo3,
    _match_espn,
    yes_aligned_close_prob,
)
from evmax.matching.alignment import YesOutcome, resolve_team


def _score(home: str, home_score: int, away: str, away_score: int, **extra) -> dict:
    d = {
        "home_name": home,
        "away_name": away,
        "home_score": home_score,
        "away_score": away_score,
        "home_won": home_score > away_score,
        "game_date": "2026-09-19",
    }
    d.update(extra)
    return d


def _pred(sector: str, slug_a: str, slug_b: str, yes: str, **extra) -> dict:
    d = {
        "market_id": f"kalshi:TEST-{yes}",
        "event_id": f"{sector}::2026-09-19::{slug_a.replace(' ', '_')}_vs_{slug_b.replace(' ', '_')}",
        "sector": sector,
        "yes_team": yes,
        "event_date": "2026-09-19",
        "market_type": "moneyline",
    }
    d.update(extra)
    return d


# (sector, short canonical, long canonical, ESPN name of short, ESPN name of long)
NESTED_PAIRS = [
    ("ncaaf", "utah", "utah state", "Utah Utes", "Utah State Aggies"),
    ("ncaab", "kansas", "kansas state", "Kansas Jayhawks", "Kansas State Wildcats"),
    ("ncaab", "iowa", "iowa state", "Iowa Hawkeyes", "Iowa State Cyclones"),
    ("ncaaf", "florida", "florida atlantic", "Florida Gators", "Florida Atlantic Owls"),
    ("ncaaf", "michigan", "western michigan", "Michigan Wolverines", "Western Michigan Broncos"),
    ("ncaaf", "washington", "washington state", "Washington Huskies", "Washington State Cougars"),
    # Soccer: "psg" expands to "paris saint germain" (resolver _ACRONYM_EXPAND),
    # which CONTAINS Paris FC's canonical "paris" — the Ligue 1 Paris derby.
    ("soccer", "paris", "psg", "Paris FC", "Paris Saint-Germain"),
]

_CASES = [
    pytest.param(
        sector, short, long_, espn_short, espn_long, yes_is_short, short_is_home, short_slug_first,
        id=f"{sector}-{short.replace(' ', '_')}|{long_.replace(' ', '_')}"
           f"-yes_{'short' if yes_is_short else 'long'}"
           f"-{'short' if short_is_home else 'long'}_home"
           f"-{'short' if short_slug_first else 'long'}_slug_first",
    )
    for (sector, short, long_, espn_short, espn_long), yes_is_short, short_is_home, short_slug_first
    in itertools.product(NESTED_PAIRS, (True, False), (True, False), (True, False))
]


class TestMatchEspnNestedNames:
    @pytest.mark.parametrize(
        "sector,short,long_,espn_short,espn_long,yes_is_short,short_is_home,short_slug_first",
        _CASES,
    )
    @pytest.mark.parametrize("short_won", [True, False], ids=["short_won", "long_won"])
    def test_moneyline_graded_on_the_yes_team(
        self, sector, short, long_, espn_short, espn_long,
        yes_is_short, short_is_home, short_slug_first, short_won,
    ):
        s_pts, l_pts = (31, 10) if short_won else (10, 31)
        if short_is_home:
            score = _score(espn_short, s_pts, espn_long, l_pts)
        else:
            score = _score(espn_long, l_pts, espn_short, s_pts)
        slug_a, slug_b = (short, long_) if short_slug_first else (long_, short)
        yes = short if yes_is_short else long_
        pred = _pred(sector, slug_a, slug_b, yes)

        expected = 1 if (yes_is_short == short_won) else 0
        assert _match_espn(pred, [score]) == expected

    @pytest.mark.parametrize("yes_is_short", [True, False])
    @pytest.mark.parametrize("short_is_home", [True, False])
    def test_spread_graded_on_the_yes_team(self, yes_is_short, short_is_home):
        # Utah beat Utah State by 32; each side's -3.5 covers only for Utah.
        if short_is_home:
            score = _score("Utah Utes", 42, "Utah State Aggies", 10)
        else:
            score = _score("Utah State Aggies", 10, "Utah Utes", 42)
        yes = "utah" if yes_is_short else "utah state"
        pred = _pred("ncaaf", "utah", "utah state", yes, market_type="spread", line=-3.5)
        assert _match_espn(pred, [score]) == (1 if yes_is_short else 0)


class TestMeasuredIncidents:
    """The exact rows the audit found graded WON on a favorite's win."""

    @pytest.mark.parametrize(
        "event_id,yes,home,hs,away,as_",
        [
            ("ncaaf::2026-09-19::utah_vs_utah_state", "utah state",
             "Utah Utes", 45, "Utah State Aggies", 14),
            ("ncaaf::2026-09-05::michigan_vs_western_michigan", "western michigan",
             "Michigan Wolverines", 41, "Western Michigan Broncos", 10),
            ("ncaaf::2026-09-05::florida_vs_florida_atlantic", "florida atlantic",
             "Florida Gators", 38, "Florida Atlantic Owls", 7),
            ("ncaaf::2026-09-06::washington_vs_washington_state", "washington state",
             "Washington Huskies", 30, "Washington State Cougars", 13),
        ],
    )
    def test_underdog_yes_lost(self, event_id, yes, home, hs, away, as_):
        date_ = event_id.split("::")[1]
        pred = {
            "market_id": "kalshi:X", "event_id": event_id, "sector": "ncaaf",
            "yes_team": yes, "event_date": date_, "market_type": "moneyline",
        }
        score = dict(_score(home, hs, away, as_), game_date=date_)
        assert _match_espn(pred, [score]) == 0


class TestNoAliasMapStillNeverDefaultsToTeamA:
    """Without a sector alias map the more specific name still decides; a
    truly undecidable pair stays unresolved instead of grading team A."""

    @pytest.mark.parametrize("yes,expected", [("utah state", 0), ("utah", 1)])
    @pytest.mark.parametrize("utah_home", [True, False])
    def test_specific_name_decides_the_bijection(self, yes, expected, utah_home):
        if utah_home:
            score = _score("Utah Utes", 42, "Utah State Aggies", 10)
        else:
            score = _score("Utah State Aggies", 10, "Utah Utes", 42)
        # unknown sector ⇒ no alias map; slug order puts the contained name first
        pred = _pred("zz", "utah", "utah state", yes)
        assert _match_espn(pred, [score]) == expected

    def test_ambiguous_yes_label_is_unresolved(self):
        # "michigan" is a token subset of BOTH slug teams.
        score = _score("Michigan State Spartans", 20, "Western Michigan Broncos", 17)
        pred = _pred("zz", "michigan state", "western michigan", "michigan")
        assert _match_espn(pred, [score]) is None

    def test_cross_game_containment_is_ambiguous_without_alias_map(self):
        # Every slug token is contained in BOTH games; with no alias map to say
        # which is which the event is ambiguous — never "first game wins".
        scores = [
            _score("Washington State Cougars", 24, "Oregon State Beavers", 21),
            _score("Washington Huskies", 10, "Oregon Ducks", 35),
        ]
        pred = _pred("zz", "washington", "oregon", "washington")
        assert _match_espn(pred, scores) is None

    def test_cross_game_known_teams_rejected_by_alias_map(self):
        score = _score("Washington State Cougars", 24, "Oregon State Beavers", 21)
        pred = _pred("ncaaf", "washington", "oregon", "washington")
        assert _match_espn(pred, [score]) is None

    def test_real_game_chosen_over_contained_game(self):
        scores = [
            _score("Washington State Cougars", 24, "Oregon State Beavers", 21),
            _score("Washington Huskies", 10, "Oregon Ducks", 35),
        ]
        pred = _pred("ncaaf", "washington", "oregon", "washington")
        assert _match_espn(pred, scores) == 0


class TestNonNestedStillResolve:
    def test_nfl_home_and_away(self):
        score = _score("Kansas City Chiefs", 27, "Buffalo Bills", 24)
        assert _match_espn(_pred("nfl", "chiefs", "bills", "chiefs"), [score]) == 1
        assert _match_espn(_pred("nfl", "chiefs", "bills", "bills"), [score]) == 0
        assert _match_espn(_pred("nfl", "bills", "chiefs", "bills"), [score]) == 0

    def test_nba_mascot_slugs(self):
        score = _score("Los Angeles Lakers", 99, "LA Clippers", 101)
        assert _match_espn(_pred("nba", "lakers", "clippers", "clippers"), [score]) == 1
        assert _match_espn(_pred("nba", "lakers", "clippers", "lakers"), [score]) == 0

    def test_polymarket_location_label_resolves_through_alias_map(self):
        score = _score("Minnesota Lynx", 80, "Connecticut Sun", 70)
        pred = _pred("wnba", "sun", "lynx", "connecticut", venue="polymarket_us")
        assert _match_espn(pred, [score]) == 0

    def test_ticker_code_yes_label(self):
        score = _score("Wisconsin Badgers", 82, "High Point Panthers", 83)
        assert _match_espn(_pred("ncaab", "wisconsin", "high point", "hp"), [score]) == 1

    def test_espn_abbreviation_fallback(self):
        score = _score("New England Revolution", 1, "Columbus Crew", 2,
                       home_abbr="NE", away_abbr="CLB")
        assert _match_espn(_pred("soccer", "new england", "columbus", "clb"), [score]) == 1

    def test_soccer_draw_market(self):
        score = _score("Paris Saint-Germain", 1, "Paris FC", 1)
        assert _match_espn(_pred("soccer", "psg", "paris", "tie"), [score]) == 1

    def test_total_is_side_independent(self):
        score = _score("Utah Utes", 42, "Utah State Aggies", 10)
        pred = _pred("ncaaf", "utah", "utah state", "over", market_type="total", line=49.5)
        assert _match_espn(pred, [score]) == 1

    def test_series_same_matchup_picks_closest_date(self):
        scores = [
            dict(_score("Boston Celtics", 100, "New York Knicks", 90), game_date="2026-09-18"),
            dict(_score("Boston Celtics", 90, "New York Knicks", 100), game_date="2026-09-19"),
        ]
        assert _match_espn(_pred("nba", "celtics", "knicks", "celtics"), scores) == 0


class TestMatchBo3Nested:
    @pytest.mark.parametrize("academy_first", [True, False])
    @pytest.mark.parametrize("yes,expected", [("mibr", 0), ("mibr academy", 1)])
    def test_academy_team_contains_parent_name(self, academy_first, yes, expected):
        # MIBR Academy beat MIBR 2-1.
        if academy_first:
            score = {"team1_name": "MIBR Academy", "team2_name": "MIBR",
                     "team1_score": 2, "team2_score": 1, "team1_won": True}
        else:
            score = {"team1_name": "MIBR", "team2_name": "MIBR Academy",
                     "team1_score": 1, "team2_score": 2, "team1_won": False}
        pred = {"event_id": "cs2::2026-09-19::mibr_vs_mibr_academy", "sector": "cs2",
                "yes_team": yes}
        assert _match_bo3(pred, [score]) == expected


class TestYesAlignedCloseProbNested:
    @pytest.mark.parametrize(
        "short,long_",
        [
            ("Utah", "Utah State"),
            ("Kansas", "Kansas State"),
            ("Iowa", "Iowa State"),
            ("Florida", "Florida Atlantic"),
            ("Michigan", "Western Michigan"),
            ("Washington", "Washington State"),
        ],
    )
    @pytest.mark.parametrize("short_is_a", [True, False])
    @pytest.mark.parametrize("yes_is_short", [True, False])
    @pytest.mark.parametrize("sector", [None, "ncaaf"])
    def test_close_prob_is_the_yes_teams(self, short, long_, short_is_a, yes_is_short, sector):
        a, b = (short, long_) if short_is_a else (long_, short)
        p_a, p_b = 0.8, 0.2
        yes = (short if yes_is_short else long_).lower()
        want = p_a if (yes_is_short == short_is_a) else p_b
        # sector=None exercises the call shape every pre-fix caller used.
        kwargs = {"sector": sector} if sector else {}
        got = yes_aligned_close_prob(yes, a, b, p_a, p_b, **kwargs)
        assert got == pytest.approx(want)

    def test_soccer_paris_derby(self):
        # Without the alias map "paris" is contained in both labels: unresolved
        # (the old substring test graded it as PSG, the first label).
        assert yes_aligned_close_prob("paris", "Paris Saint-Germain", "Paris FC", 0.7, 0.1, 0.2) is None
        # "psg" vs label "Paris Saint-Germain" and "paris" vs "Paris FC".
        assert yes_aligned_close_prob(
            "paris", "Paris Saint-Germain", "Paris FC", 0.7, 0.1, 0.2, sector="soccer",
        ) == pytest.approx(0.1)
        assert yes_aligned_close_prob(
            "psg", "Paris Saint-Germain", "Paris FC", 0.7, 0.1, 0.2, sector="soccer",
        ) == pytest.approx(0.7)

    def test_tennis_same_surname_is_unresolved(self):
        # A bare surname contained in BOTH players' names picks neither.
        assert yes_aligned_close_prob(
            "pliskova", "Karolina Pliskova", "Kristyna Pliskova", 0.6, 0.4,
        ) is None
        assert yes_aligned_close_prob(
            "pliskova", "Karolina Pliskova", "Kristyna Pliskova", 0.6, 0.4, sector="tennis",
        ) is None

    def test_tennis_full_name_picks_the_player(self):
        assert yes_aligned_close_prob(
            "kristyna pliskova", "Karolina Pliskova", "Kristyna Pliskova", 0.6, 0.4,
            sector="tennis",
        ) == pytest.approx(0.4)
        assert yes_aligned_close_prob(
            "sinner", "Carlos Alcaraz", "Jannik Sinner", 0.45, 0.55, sector="tennis",
        ) == pytest.approx(0.55)

    def test_tennis_first_name_equals_opponent_surname(self):
        # Live placed row 8319 (KXWTAMATCH-26JUN09SAKMAR-MAR, YES = Tatjana MARIA):
        # "maria" ⊂ "maria sakkari" handed the bet Sakkari's close.
        # Without the alias map the name is contained in both labels: unresolved.
        assert yes_aligned_close_prob("maria", "Maria Sakkari", "Tatjana Maria", 0.514, 0.486) is None
        assert yes_aligned_close_prob(
            "maria", "Maria Sakkari", "Tatjana Maria", 0.514, 0.486, sector="tennis",
        ) == pytest.approx(0.486)

    @pytest.mark.parametrize(
        "yes,a,b,want",
        [
            # substring accidents of the old aligner (rows 14659 and 2397)
            ("ly", "FlyQuest", "LYON", 0.43),        # "ly" ⊂ "fLYquest"
            ("m", "Hoffenheim", "Mainz 05", 0.43),   # "m" ⊂ "hoffenheiM"
            # ticker codes that must keep resolving
            ("g", "G2", "Karmine Corp", 0.57),
            ("phi", "New England Revolution", "Philadelphia Union", 0.43),
        ],
    )
    def test_ticker_code_labels(self, yes, a, b, want):
        assert yes_aligned_close_prob(yes, a, b, 0.57, 0.43) == pytest.approx(want)

    def test_ambiguous_ticker_code_is_unresolved(self):
        # "m" prefixes a word of BOTH labels.
        assert yes_aligned_close_prob("m", "Mainz 05", "Bayern Munich", 0.2, 0.6, 0.2) is None

    def test_label_drift_still_absorbed(self):
        # yes ⊆ label and label ⊆ yes both still work when unambiguous.
        assert yes_aligned_close_prob("fever", "Indiana Fever", "Chicago Sky", 0.6, 0.4) == 0.6
        assert yes_aligned_close_prob("indiana fever", "Fever", "Sky", 0.6, 0.4) == 0.6
        assert yes_aligned_close_prob("sky", "Indiana Fever", "Chicago Sky", 0.6, 0.4) == 0.4

    def test_alias_map_used_when_sector_given(self):
        assert yes_aligned_close_prob(
            "man city", "Manchester City", "Arsenal", 0.5, 0.3, 0.2, sector="soccer",
        ) == pytest.approx(0.5)

    def test_shortcuts_unchanged(self):
        assert yes_aligned_close_prob("draw", "A", "B", 0.4, 0.3, 0.3) == 0.3
        assert yes_aligned_close_prob("under", "over", "under", 0.45, 0.55) == 0.55
        assert yes_aligned_close_prob(None, "A", "B", 0.5, 0.5) is None
        assert yes_aligned_close_prob("zzz", "Alpha", "Beta", 0.5, 0.5) is None


class TestResolveTeamHelper:
    def test_exact_beats_subset(self):
        assert resolve_team("utah", "utah state", "utah") == (YesOutcome.B, "canonical")

    def test_unique_subset(self):
        assert resolve_team("utah state", "Utah State Aggies", "Utah Utes")[0] is YesOutcome.A

    def test_double_subset_is_none(self):
        assert resolve_team("utah", "Utah State Aggies", "Utah Utes") is None

    def test_reverse_subset(self):
        assert resolve_team("cleveland cavaliers", "raptors", "cavaliers") == (
            YesOutcome.B, "tokens_rev",
        )

    def test_codes_only_when_allowed(self):
        assert resolve_team("hp", "wisconsin", "high point") is None
        assert resolve_team("hp", "wisconsin", "high point", allow_codes=True) == (
            YesOutcome.B, "code",
        )

    def test_ambiguous_code_is_none(self):
        assert resolve_team("m", "m'gladbach", "mainz 05", allow_codes=True) is None
