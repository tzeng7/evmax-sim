"""Polymarket US soccer club names reach the canonical Pinnacle's label does.

A read-only check on 2026-10-13 found 8 of the 18 UCL games for 10-13/10-14
unmatched on Polymarket US (Kalshi matched all 18). PolyUS lists clubs under
their registered names ("FC Bayern München", "Lille OSC", "BV Borussia 09
Dortmund"), which missed every soccer alias. Two of them were worse than a
missing alias: the normalizer's noise-word strip dropped "united", so
"Manchester United FC" became a bare "manchester" (a city with two clubs) and
"D.C. United SC" became "d.c.".

Spellings below are the raw PolyUS ``team.name`` / event-title halves and the
raw Pinnacle labels, both captured live on 2026-10-13.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from evmax.clients.polymarket_us import (
    _LEAGUE_TEAM_NAME_MAPS,
    PolymarketUSClient,
    _canon_team,
)
from evmax.matching.alignment import YesOutcome, align_yes_side
from evmax.matching.engine import MatchingEngine
from evmax.matching.normalizer import NameNormalizer
from evmax.models.market import MarketType
from evmax.models.odds import SharpBook, SharpOdds
from evmax.sectors.registry import get_handler


def _nn(name: str) -> str:
    return NameNormalizer("soccer").normalize(name)


# (PolyUS name, Pinnacle label, canonical). PolyUS event titles drop the
# accents ("FC Bayern Munchen"), its team.name keeps them ("München") — both
# are pinned.
@pytest.mark.parametrize(
    "polyus, pinnacle, canonical",
    [
        # UCL league phase — the 8 games unmatched on 2026-10-13/14
        ("Lille OSC", "Lille", "lille"),
        ("Club Atlético de Madrid", "Atletico Madrid", "atletico"),
        ("Club Atletico de Madrid", "Atletico Madrid", "atletico"),
        ("Manchester United FC", "Manchester United", "man united"),
        ("FC Internazionale Milano", "Internazionale", "inter"),
        ("Club Brugge KV", "Club Brugge", "brugge"),
        ("FC Bayern München", "Bayern Munich", "bayern"),
        ("FC Bayern Munchen", "Bayern Munich", "bayern"),
        ("Racing Club de Lens", "Lens", "lens"),
        ("BV Borussia 09 Dortmund", "Borussia Dortmund", "dortmund"),
        ("Real Betis Seville", "Real Betis", "betis"),
        # UCL spellings that already matched — pinned so they stay matched
        ("FK Bodø/Glimt", "Bodo Glimt", "bodo glimt"),
        ("FK Bodo/Glimt", "Bodo Glimt", "bodo glimt"),
        ("Como 1907", "Como", "como"),
        ("Feyenoord Rotterdam", "Feyenoord", "feyenoord"),
        ("Galatasaray SK", "Galatasaray", "galatasaray"),
        ("Fenerbahçe SK", "Fenerbahce", "fenerbahce"),
        ("SK Slavia Praha", "Slavia Prague", "slavia prague"),
        ("ŠK Slovan Bratislava", "Slovan Bratislava", "slovan bratislava"),
        ("Manchester City FC", "Manchester City", "man city"),
        ("Paris Saint-Germain FC", "Paris Saint-Germain", "psg"),
        # MLS / EPL / Championship — the same "united" strip
        ("D.C. United SC", "D.C. United", "dc united"),
        ("Sheffield United FC", "Sheffield United", "sheffield utd"),
        # Liga MX
        ("Pumas de la UNAM", "Pumas UNAM", "pumas unam"),
        # La Liga / Bundesliga / Serie A (PolyUS lal / bun / sea — arb only)
        ("RC Celta de Vigo", "Celta Vigo", "celta vigo"),
        ("RC Deportivo A Coruña", "Deportivo La Coruna", "deportivo la coruna"),
        ("RCD Espanyol de Barcelona", "Espanyol", "espanyol"),
        ("Real Racing Club", "Racing Santander", "racing santander"),
        ("Real Sociedad de Fútbol", "Real Sociedad", "sociedad"),
        ("1. FSV Mainz 05", "Mainz 05", "mainz"),
        ("Bayer 04 Leverkusen", "Bayer Leverkusen", "leverkusen"),
        ("SV 07 Elversberg", "Elversberg", "elversberg"),
        ("ACF Fiorentina", "Fiorentina", "fiorentina"),
        ("Bologna FC 1909", "Bologna", "bologna"),
        ("Cagliari Calcio", "Cagliari", "cagliari"),
        ("Frosinone Calcio", "Frosinone", "frosinone"),
        ("Genoa CFC", "Genoa", "genoa"),
        ("US Lecce", "Lecce", "lecce"),
        ("US Sassuolo Calcio", "Sassuolo", "sassuolo"),
        ("Udinese Calcio", "Udinese", "udinese"),
    ],
)
def test_polyus_name_and_pinnacle_label_share_a_canonical(polyus, pinnacle, canonical):
    assert _nn(polyus) == _nn(pinnacle) == canonical


class TestManchesterClubsNeverMerge:
    UNITED = ["Manchester United FC", "Manchester United", "Man Utd", "Man United", "MUN"]
    CITY = ["Manchester City FC", "Manchester City", "Man City", "MCI"]

    def test_every_spelling_reaches_its_own_club(self):
        assert {_nn(n) for n in self.UNITED} == {"man united"}
        assert {_nn(n) for n in self.CITY} == {"man city"}

    def test_bare_manchester_is_never_aliased(self):
        """A bare "manchester" names two clubs, so no alias may claim it, and
        the normalizer must not produce it from either club's full name."""
        aliases = get_handler("soccer")._aliases
        assert "manchester" not in aliases
        assert _nn("Manchester") not in {"man united", "man city"}
        assert "manchester" not in {_nn(n) for n in self.UNITED + self.CITY}


class TestUnitedStripIsLastResort:
    """The normalizer retries the alias map with only the club-form affixes
    removed before it drops "united"/"utd"."""

    @pytest.mark.parametrize(
        "name, canonical",
        [
            # alias on "<city> united" now reached from the "...FC" form
            ("Newcastle United FC", "newcastle"),
            ("Minnesota United FC", "minnesota"),
            ("Atlanta United FC", "atlanta"),
            ("West Ham United FC", "west ham"),
            # no alias on "leeds united": still falls through to the old strip
            ("Leeds United FC", "leeds"),
            ("Leeds United", "leeds"),
        ],
    )
    def test_united_clubs(self, name, canonical):
        assert _nn(name) == canonical

    def test_names_without_united_are_unchanged(self):
        assert _nn("Arsenal FC") == "arsenal"
        assert _nn("FC Tokyo") == "tokyo"
        assert _nn("Qairat FK") == "qairat"


class TestLeagueScopedAek:
    """PolyUS lists AEK Athens as a bare "AEK". AEK Larnaca also exists, so the
    name resolves only inside the PolyUS ucl league, like Kalshi's
    KXUCLGAME-scoped "aek" code."""

    def test_flat_alias_map_does_not_claim_bare_aek(self):
        assert "aek" not in get_handler("soccer")._aliases
        assert _nn("AEK") == "aek"
        # the scoped target is a registered canonical
        assert get_handler("soccer").is_canonical("aek athens")
        assert _LEAGUE_TEAM_NAME_MAPS["ucl"]["aek"] == "aek athens"

    def test_canon_team_resolves_aek_only_in_ucl(self):
        side = {"team": {"name": "AEK", "alias": "Enosis"}}
        assert _canon_team("soccer", side, "ucl") == "aek athens"
        assert _canon_team("soccer", side, "epl") == "aek"
        assert _canon_team("soccer", side) == "aek"

    def test_parse_event_resolves_title_and_outcome_sides(self):
        ev = _ucl_event("FC Shakhtar Donetsk", "AEK", "2026-10-14T19:00:00Z")
        markets = PolymarketUSClient()._parse_event(ev, "soccer", "ucl")
        assert {(m.team_home, m.team_away) for m in markets} == {
            ("shakhtar donetsk", "aek athens")
        }
        assert {m.yes_team for m in markets} == {"shakhtar donetsk", "draw", "aek athens"}

    def test_other_league_keeps_bare_aek(self):
        ev = _ucl_event("FC Shakhtar Donetsk", "AEK", "2026-10-14T19:00:00Z")
        markets = PolymarketUSClient()._parse_event(ev, "soccer", "epl")
        assert {m.team_away for m in markets} == {"aek"}


class TestNicknameAppendNeverDisplacesACanonical:
    """_canon_team also normalizes "<name> <alias>" and keeps the shorter
    result. "D.C. United SC" + "DCU" loses "united" to the noise strip and
    comes out "d.c. dcu" — shorter than "dc united" but not a club."""

    def test_dc_united_keeps_its_canonical(self):
        side = {"team": {"name": "D.C. United SC", "alias": "DCU"}}
        assert _canon_team("soccer", side) == "dc united"

    def test_nickname_still_resolves_an_ambiguous_city(self):
        side = {"team": {"name": "Los Angeles", "alias": "Lakers"}}
        assert _canon_team("nba", side) == "lakers"

    def test_marketing_alias_text_is_still_ignored(self):
        side = {"team": {"name": "Qairat FK", "alias": "Халық командасы"}}
        assert _canon_team("soccer", side) == "qairat"


# ---------------------------------------------------------------------------
# End to end: PolyUS UCL event → MatchingEngine → YES alignment
# ---------------------------------------------------------------------------

def _outcome_market(slug, team_name, yes_q, no_q, ordering="home"):
    yes = {"description": "Yes", "long": True, "tradable": True,
           "quote": {"value": yes_q, "currency": "USD"}}
    no = {"description": "No", "long": False, "tradable": True,
          "quote": {"value": no_q, "currency": "USD"}}
    if team_name:
        yes["team"] = {"name": team_name, "ordering": ordering}
        no["team"] = {"name": team_name, "ordering": "away"}
    return {
        "slug": slug, "marketType": "drawable_outcome",
        "sportsMarketType": "soccer_team_full_time_winner",
        "active": True, "closed": False, "volume": 100,
        "title": f"{team_name or 'Draw'} (Reg. Time)",
        "marketSides": [yes, no],
    }


def _ucl_event(home, away, start):
    """Shape of a live PolyUS UCL event: one Yes/No market per outcome, each
    team side claiming ordering=home, so the title fallback names the teams."""
    slug = "ucl-test-" + start[:10]
    return {
        "slug": slug, "ticker": slug, "title": f"{home} vs. {away}",
        "startDate": start, "startTime": start,
        "markets": [
            _outcome_market(f"{slug}-home", home, "0.4500", "0.56"),
            _outcome_market(f"{slug}-draw", None, "0.2600", "0.75"),
            _outcome_market(f"{slug}-away", away, "0.3100", "0.70"),
        ],
    }


def _pinnacle(event_id, label_a, label_b):
    return SharpOdds(
        event_id=event_id, book=SharpBook.pinnacle, sector="soccer",
        market_type=MarketType.moneyline,
        outcome_a_label=label_a, outcome_b_label=label_b,
        outcome_a_decimal=2.2, outcome_b_decimal=3.3, outcome_draw_decimal=3.6,
        true_prob_a=0.43, true_prob_b=0.29, true_prob_draw=0.28, margin=0.03,
        event_date=datetime(2026, 10, 13, 19, 0, tzinfo=timezone.utc),
    )


@pytest.mark.parametrize(
    "home, away, start, event_id, label_a, label_b, yes_sides",
    [
        ("Club Atletico de Madrid", "Manchester United FC", "2026-10-13T19:00:00Z",
         "soccer::2026-10-13::atletico_vs_man_united", "Atletico Madrid", "Manchester United",
         {"atletico": YesOutcome.A, "man united": YesOutcome.B}),
        ("Viking FK", "FC Bayern Munchen", "2026-10-13T19:00:00Z",
         "soccer::2026-10-13::viking_vs_bayern", "Viking", "Bayern Munich",
         {"viking": YesOutcome.A, "bayern": YesOutcome.B}),
        ("FC Shakhtar Donetsk", "AEK", "2026-10-14T19:00:00Z",
         "soccer::2026-10-14::shakhtar_donetsk_vs_aek_athens", "Shakhtar Donetsk", "AEK Athens",
         {"shakhtar donetsk": YesOutcome.A, "aek athens": YesOutcome.B}),
    ],
)
def test_polyus_ucl_event_matches_pinnacle_and_aligns(
    home, away, start, event_id, label_a, label_b, yes_sides
):
    markets = PolymarketUSClient()._parse_event(_ucl_event(home, away, start), "soccer", "ucl")
    sharp = _pinnacle(event_id, label_a, label_b)
    engine = MatchingEngine()
    matched = engine.match_all(markets, [sharp])
    assert len(matched) == 3
    got = {
        m.yes_team: align_yes_side(m, so, engine.normalizer_for("soccer")).outcome
        for m, so, _ in matched
    }
    assert got == {**yes_sides, "draw": YesOutcome.DRAW}
