"""NHL alias map (evmax/sectors/aliases/nhl.yaml) — the opening-night matching blocker.

Before this file existed, Kalshi NHL tickers carried team codes (bos, la, nj,
sj, tb, vgk, …) while Pinnacle sent full names, so the production
MatchingEngine matched 0/236 archived moneylines, 0/480 spreads and 0/896
totals (Mar–Jun 2026 replay) and zero NHL rows were ever logged. The golden
cases below are real archived Kalshi tickers/titles and Pinnacle labels
(archive.db, 2026-03-28 → 2026-09-22).
"""

from __future__ import annotations

import json
import pathlib
from datetime import datetime, timezone

import pytest
import yaml

from evmax.agents.models.nhl_xg_agent import NHL_ABBREV_TO_NAME
from evmax.clients.esports_pinnacle import PinnacleGuestClient
from evmax.clients.kalshi import KalshiClient
from evmax.matching.alignment import YesOutcome, align_yes_side
from evmax.matching.engine import MatchingEngine
from evmax.matching.normalizer import NameNormalizer
from evmax.models.market import MarketType
from evmax.models.odds import SharpBook, SharpOdds

REPO = pathlib.Path(__file__).resolve().parents[1]
ALIAS_FILE = REPO / "evmax" / "sectors" / "aliases" / "nhl.yaml"

CANONICALS = sorted(set(NHL_ABBREV_TO_NAME.values()))

# Kalshi ticker code (as seen in archived KXNHLGAME tickers) → canonical.
# LA / NJ / SJ / TB are Kalshi's 2-letter codes; the NHL tricodes differ.
KALSHI_CODES = {
    "ana": "anaheim ducks", "bos": "boston bruins", "buf": "buffalo sabres",
    "car": "carolina hurricanes", "cbj": "columbus blue jackets",
    "cgy": "calgary flames", "chi": "chicago blackhawks",
    "col": "colorado avalanche", "dal": "dallas stars", "det": "detroit red wings",
    "edm": "edmonton oilers", "fla": "florida panthers", "la": "los angeles kings",
    "min": "minnesota wild", "mtl": "montreal canadiens", "nj": "new jersey devils",
    "nsh": "nashville predators", "nyi": "new york islanders",
    "nyr": "new york rangers", "ott": "ottawa senators", "phi": "philadelphia flyers",
    "pit": "pittsburgh penguins", "sea": "seattle kraken", "sj": "san jose sharks",
    "stl": "st louis blues", "tb": "tampa bay lightning", "tor": "toronto maple leafs",
    "uta": "utah mammoth", "van": "vancouver canucks", "vgk": "vegas golden knights",
    "wpg": "winnipeg jets", "wsh": "washington capitals",
}

# Kalshi title city labels (every one observed in archived NHL titles).
KALSHI_TITLE_CITIES = {
    "Anaheim": "anaheim ducks", "Boston": "boston bruins", "Buffalo": "buffalo sabres",
    "Calgary": "calgary flames", "Carolina": "carolina hurricanes",
    "Chicago": "chicago blackhawks", "Colorado": "colorado avalanche",
    "Columbus": "columbus blue jackets", "Dallas": "dallas stars",
    "Detroit": "detroit red wings", "Edmonton": "edmonton oilers",
    "Florida": "florida panthers", "Los Angeles": "los angeles kings",
    "Minnesota": "minnesota wild", "Montreal": "montreal canadiens",
    "Nashville": "nashville predators", "New Jersey": "new jersey devils",
    "New York I": "new york islanders", "New York R": "new york rangers",
    "Ottawa": "ottawa senators", "Philadelphia": "philadelphia flyers",
    "Pittsburgh": "pittsburgh penguins", "San Jose": "san jose sharks",
    "Seattle": "seattle kraken", "St. Louis": "st louis blues",
    "Tampa Bay": "tampa bay lightning", "Toronto": "toronto maple leafs",
    "Utah": "utah mammoth", "Vancouver": "vancouver canucks",
    "Vegas": "vegas golden knights", "Washington": "washington capitals",
    "Winnipeg": "winnipeg jets",
}

# Pinnacle / ESPN full names (Pinnacle archive labels + the two teams with no
# archived Pinnacle moneyline; ESPN's accented Montréal spelling).
FULL_NAMES = {
    "Anaheim Ducks": "anaheim ducks", "Boston Bruins": "boston bruins",
    "Buffalo Sabres": "buffalo sabres", "Calgary Flames": "calgary flames",
    "Carolina Hurricanes": "carolina hurricanes", "Chicago Blackhawks": "chicago blackhawks",
    "Colorado Avalanche": "colorado avalanche", "Columbus Blue Jackets": "columbus blue jackets",
    "Dallas Stars": "dallas stars", "Detroit Red Wings": "detroit red wings",
    "Edmonton Oilers": "edmonton oilers", "Florida Panthers": "florida panthers",
    "Los Angeles Kings": "los angeles kings", "Minnesota Wild": "minnesota wild",
    "Montreal Canadiens": "montreal canadiens", "Montréal Canadiens": "montreal canadiens",
    "Nashville Predators": "nashville predators", "New Jersey Devils": "new jersey devils",
    "New York Islanders": "new york islanders", "New York Rangers": "new york rangers",
    "Ottawa Senators": "ottawa senators", "Philadelphia Flyers": "philadelphia flyers",
    "Pittsburgh Penguins": "pittsburgh penguins", "San Jose Sharks": "san jose sharks",
    "Seattle Kraken": "seattle kraken", "St. Louis Blues": "st louis blues",
    "Tampa Bay Lightning": "tampa bay lightning", "Toronto Maple Leafs": "toronto maple leafs",
    "Utah Mammoth": "utah mammoth", "Vancouver Canucks": "vancouver canucks",
    "Vegas Golden Knights": "vegas golden knights", "Washington Capitals": "washington capitals",
    "Winnipeg Jets": "winnipeg jets",
}


@pytest.fixture(scope="module")
def norm() -> NameNormalizer:
    return NameNormalizer("nhl")


def _aliases() -> dict[str, str]:
    return yaml.safe_load(ALIAS_FILE.read_text())["aliases"]


def _kalshi_market(ticker: str, title: str, *, floor_strike: float | None = None):
    raw = {
        "ticker": ticker, "title": title,
        "yes_ask_dollars": "0.55", "no_ask_dollars": "0.47",
        "yes_bid_dollars": "0.53", "no_bid_dollars": "0.45",
    }
    if floor_strike is not None:
        raw["floor_strike"] = floor_strike
    kc = KalshiClient.__new__(KalshiClient)  # parser only — no network/auth
    return kc._parse_market(raw, "nhl")


# ── The alias map itself ───────────────────────────────────────────────────

class TestAliasMap:
    def test_file_exists_and_loads_into_the_handler(self):
        from evmax.sectors.registry import get_handler

        assert ALIAS_FILE.exists()
        assert get_handler("nhl").normalize_team("vgk") == "vegas golden knights"

    def test_exactly_32_canonicals_matching_the_xg_team_map(self):
        assert sorted(set(_aliases().values())) == CANONICALS
        assert len(CANONICALS) == 32

    def test_canonicals_are_idempotent(self, norm):
        for canon in CANONICALS:
            assert norm.normalize(canon) == canon

    def test_canonicals_are_dot_free(self):
        # Pinnacle event keys strip "." (esports_pinnacle._normalize) while the
        # Kalshi key builder does not; a dotted canonical would never
        # exact-match a spread/total record.
        assert all("." not in c for c in CANONICALS)

    def test_no_alias_key_is_another_teams_canonical(self):
        for key, canon in _aliases().items():
            if key in CANONICALS:
                assert canon == key, f"{key!r} is a canonical but aliases to {canon!r}"

    def test_bare_new_york_is_not_resolved_to_either_team(self, norm):
        assert "new york" not in _aliases()
        assert norm.normalize("New York") not in CANONICALS


class TestNameForms:
    @pytest.mark.parametrize("code,canon", sorted(KALSHI_CODES.items()))
    def test_kalshi_ticker_codes(self, norm, code, canon):
        assert norm.normalize(code) == canon
        assert norm.normalize(code.upper()) == canon

    @pytest.mark.parametrize("tricode,canon", sorted(NHL_ABBREV_TO_NAME.items()))
    def test_nhl_tricodes(self, norm, tricode, canon):
        assert norm.normalize(tricode) == canon

    @pytest.mark.parametrize("short,tri", [("la", "lak"), ("nj", "njd"), ("sj", "sjs"), ("tb", "tbl")])
    def test_two_letter_and_tricode_forms_agree(self, norm, short, tri):
        assert norm.normalize(short) == norm.normalize(tri)

    @pytest.mark.parametrize("city,canon", sorted(KALSHI_TITLE_CITIES.items()))
    def test_kalshi_title_cities(self, norm, city, canon):
        assert norm.normalize(city) == canon

    @pytest.mark.parametrize("name,canon", sorted(FULL_NAMES.items()))
    def test_pinnacle_and_espn_full_names(self, norm, name, canon):
        assert norm.normalize(name) == canon

    @pytest.mark.parametrize("name,canon", sorted(FULL_NAMES.items()))
    def test_pinnacle_event_key_part_equals_kalshi_key_part(self, norm, name, canon):
        # The two sides build event keys differently; they must still agree.
        assert PinnacleGuestClient._normalize(name, "nhl") == canon.replace(" ", "_")
        assert norm.normalize(canon).replace(" ", "_") == canon.replace(" ", "_")

    def test_rangers_vs_islanders(self, norm):
        assert norm.normalize("New York R") == "new york rangers"
        assert norm.normalize("New York I") == "new york islanders"
        assert norm.normalize("NYR") == "new york rangers"
        assert norm.normalize("NYI") == "new york islanders"
        assert norm.normalize("Rangers") != norm.normalize("Islanders")

    def test_utah_franchise_names(self, norm):
        for name in ("Utah", "UTA", "Utah Mammoth", "Utah Hockey Club", "Mammoth"):
            assert norm.normalize(name) == "utah mammoth"
        # The deactivated Coyotes are not Utah's rating history.
        assert norm.normalize("Arizona Coyotes") != "utah mammoth"

    def test_st_louis_spellings(self, norm):
        for name in ("St. Louis Blues", "St Louis Blues", "St. Louis", "STL", "Blues"):
            assert norm.normalize(name) == "st louis blues"


# ── Kalshi parser → canonicals (golden archived tickers) ───────────────────

GOLDEN_ML = [
    # (ticker, title, home, away, yes)
    ("KXNHLGAME-26APR14LAVAN-LA", "Los Angeles at Vancouver Winner?",
     "vancouver canucks", "los angeles kings", "los angeles kings"),
    ("KXNHLGAME-26APR14NJBOS-NJ", "New Jersey at Boston Winner?",
     "boston bruins", "new jersey devils", "new jersey devils"),
    ("KXNHLGAME-26APR15SJCHI-SJ", "San Jose at Chicago Winner?",
     "chicago blackhawks", "san jose sharks", "san jose sharks"),
    ("KXNHLGAME-26APR15NYRTB-TB", "New York R at Tampa Bay Winner?",
     "tampa bay lightning", "new york rangers", "tampa bay lightning"),
    ("KXNHLGAME-26APR14CARNYI-NYI", "Carolina at New York I Winner?",
     "new york islanders", "carolina hurricanes", "new york islanders"),
    ("KXNHLGAME-26APR14PITSTL-STL", "Pittsburgh at St. Louis Winner?",
     "st louis blues", "pittsburgh penguins", "st louis blues"),
    ("KXNHLGAME-26APR14WPGUTA-UTA", "Winnipeg at Utah Winner?",
     "utah mammoth", "winnipeg jets", "utah mammoth"),
    ("KXNHLGAME-26SEP19VGKLA-LA", "Los Angeles wins",
     "los angeles kings", "vegas golden knights", "los angeles kings"),
]


class TestKalshiParserGolden:
    @pytest.mark.parametrize("ticker,title,home,away,yes", GOLDEN_ML)
    def test_moneyline_tickers(self, ticker, title, home, away, yes):
        m = _kalshi_market(ticker, title)
        assert m is not None
        assert m.market_type == MarketType.moneyline
        norm = NameNormalizer("nhl")
        assert norm.normalize(m.team_home) == home
        assert norm.normalize(m.team_away) == away
        assert m.yes_team == yes

    def test_spread_ticker_with_two_letter_code(self):
        m = _kalshi_market("KXNHLSPREAD-26MAR28UTALA-LA1", "Los Angeles wins by over 1.5 goals?",
                           floor_strike=1.5)
        assert m.market_type == MarketType.spread
        assert m.yes_team == "los angeles kings"
        assert {m.team_home, m.team_away} == {"la", "uta"}

    def test_totals_title_fallback_uses_city_aliases(self):
        # 5-char pair + numeric outcome → no ticker split; the title is parsed.
        m = _kalshi_market("KXNHLTOTAL-26APR15NYRTB-6", "New York R vs Tampa Bay: Total Goals",
                           floor_strike=6.5)
        norm = NameNormalizer("nhl")
        assert m.market_type == MarketType.total
        assert norm.normalize(m.team_home) == "tampa bay lightning"
        assert norm.normalize(m.team_away) == "new york rangers"

    def test_every_archived_code_is_known(self):
        for code in KALSHI_CODES:
            assert code in _aliases(), f"Kalshi code {code!r} missing from nhl.yaml"


# ── MatchingEngine: a Kalshi NHL market pairs with its Pinnacle record ─────

def _pinnacle(home: str, away: str, day: str, *, spread: float | None = None,
              p_home: float = 0.58) -> SharpOdds:
    base = (f"nhl::{day}::{PinnacleGuestClient._normalize(home, 'nhl')}"
            f"_vs_{PinnacleGuestClient._normalize(away, 'nhl')}")
    if spread is None:
        return SharpOdds(
            event_id=base, book=SharpBook.pinnacle, sector="nhl",
            outcome_a_label=home, outcome_b_label=away,
            outcome_a_decimal=1.7, outcome_b_decimal=2.2,
            true_prob_a=p_home, true_prob_b=1 - p_home,
        )
    return SharpOdds(
        event_id=f"{base}::spread", book=SharpBook.pinnacle, sector="nhl",
        outcome_a_label=away, outcome_b_label=home,
        outcome_a_decimal=1.9, outcome_b_decimal=1.9,
        true_prob_a=0.35, true_prob_b=0.65, spread_line=spread,
    )


class TestMatchingEngine:
    def _match(self, market, sharp):
        from evmax.sectors.registry import get_handler

        market = get_handler("nhl").enrich_market(market)
        eng = MatchingEngine()
        return market, eng, eng.match(market, sharp)

    def test_moneyline_two_letter_code_exact_match_and_alignment(self):
        m = _kalshi_market("KXNHLGAME-26APR14LAVAN-LA", "Los Angeles at Vancouver Winner?")
        sharp = [
            _pinnacle("Vancouver Canucks", "Los Angeles Kings", "2026-04-14"),
            _pinnacle("Winnipeg Jets", "Utah Mammoth", "2026-04-14"),
        ]
        market, eng, res = self._match(m, sharp)
        assert res is not None
        so, score = res
        assert so.event_id == "nhl::2026-04-14::vancouver_canucks_vs_los_angeles_kings"
        assert score == 100.0
        al = align_yes_side(market, so, eng.normalizer_for("nhl"))
        assert al is not None and al.outcome is YesOutcome.B  # YES = LA = away

    def test_st_louis_spread_matches_exactly(self):
        # Spreads have no fuzzy fallback — the dot-free canonical is what makes
        # the Kalshi key equal Pinnacle's dot-stripped key.
        m = _kalshi_market("KXNHLSPREAD-26APR14PITSTL-STL1", "St. Louis wins by over 1.5 goals?",
                           floor_strike=1.5)
        sharp = [_pinnacle("St. Louis Blues", "Pittsburgh Penguins", "2026-04-14", spread=-1.5)]
        market, eng, res = self._match(m, sharp)
        assert res is not None
        assert res[0].event_id == "nhl::2026-04-14::st_louis_blues_vs_pittsburgh_penguins::spread"
        al = align_yes_side(market, res[0], eng.normalizer_for("nhl"))
        assert al is not None and al.outcome is YesOutcome.B  # YES = STL = outcome_b

    def test_rangers_game_does_not_match_islanders_game(self):
        m = _kalshi_market("KXNHLGAME-26APR15NYRTB-NYR", "New York R at Tampa Bay Winner?")
        sharp = [_pinnacle("Tampa Bay Lightning", "New York Islanders", "2026-04-15")]
        _, _, res = self._match(m, sharp)
        assert res is None


# ── Shipped state is keyed by the same canonicals ──────────────────────────

def test_shipped_elo_state_uses_the_canonicals():
    state = json.loads((REPO / "data" / "models" / "elo_state.json").read_text())
    ratings = state["nhl"]["ratings"]
    assert sorted(ratings) == CANONICALS
