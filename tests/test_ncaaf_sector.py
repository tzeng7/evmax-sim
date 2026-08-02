"""NCAAF sector wiring + alias reconciliation tests (Phase 1).

Covers the launch-critical guarantees:
  - the sector is registered and its category validates
  - Kalshi/Pinnacle/resolver maps carry ncaaf
  - the generated alias map reconciles the notorious cross-source name
    disagreements to the ESPN-canonical form that resolution reads
"""

from __future__ import annotations

import pytest

from evmax.categories import all_categories
from evmax.clients.esports_pinnacle import SECTOR_SPORT_LEAGUES, TOTALS_SECTORS
from evmax.clients.kalshi import SECTOR_SERIES_MAP
from evmax.clients.time_util import _US_SECTORS, kalshi_game_day
from evmax.matching.normalizer import NameNormalizer
from evmax.models.market import MarketType, PredictionMarket, MarketSource
from evmax.sectors.registry import get_handler


class TestNcaafWiring:
    def test_registered_in_registry(self):
        h = get_handler("ncaaf")
        assert h.name == "ncaaf"
        assert h.sharp_source == "pinnacle"

    def test_kalshi_series_map(self):
        assert SECTOR_SERIES_MAP["ncaaf"] == [
            "KXNCAAFGAME",
            "KXNCAAFSPREAD",
            "KXNCAAFTOTAL",
        ]

    def test_pinnacle_league(self):
        # sport 15 (football), league 880 (NCAA)
        assert SECTOR_SPORT_LEAGUES["ncaaf"] == (15, [880])
        assert "ncaaf" in TOTALS_SECTORS

    def test_us_gameday_convention(self):
        # ncaaf uses the ET game-day convention so late West-coast kickoffs with
        # next-day-UTC Pinnacle timestamps land on the Kalshi ticker's ET day.
        assert "ncaaf" in _US_SECTORS
        import datetime as dt

        # 2026-08-30 02:30 UTC == 2026-08-29 22:30 ET (a Fri-night kickoff)
        late = dt.datetime(2026, 8, 30, 2, 30, tzinfo=dt.timezone.utc)
        assert kalshi_game_day(late, "ncaaf") == "2026-08-29"

    def test_resolver_espn_map(self):
        from evmax.agents.cleanup.resolver import ESPN_SPORT_MAP

        assert ESPN_SPORT_MAP["ncaaf"] == ("football", "college-football", {"groups": "80"})

    def test_category_entry_valid(self):
        cats = {c.key: c for c in all_categories()}
        assert "ncaaf" in cats
        n = cats["ncaaf"]
        assert n.mode == "shadow"
        assert n.resolver == "espn_scoreboard"
        # spread/total ship disabled (capture machinery not accrued yet)
        assert set(n.disabled_market_types) == {"spread", "total"}

    def test_min_nonsharp_guard(self):
        from evmax.agents.odds.ev_gap_agent import MIN_NONSHARP_MODELS

        assert MIN_NONSHARP_MODELS["ncaaf"]["min_count"] == 1
        assert "moneyline" in MIN_NONSHARP_MODELS["ncaaf"]["market_types"]

    def test_handler_supported_market_types(self):
        h = get_handler("ncaaf")
        types = h.market_types_supported()
        assert MarketType.moneyline in types
        assert MarketType.spread in types
        assert MarketType.total in types


class TestNcaafAliasReconciliation:
    """Every source spelling must collapse to the ESPN-canonical location the
    resolver keys on. Regression guards for the documented ambiguity traps."""

    @pytest.fixture(scope="class")
    def nz(self):
        return NameNormalizer("ncaaf")

    @pytest.mark.parametrize(
        "source,canonical",
        [
            # Ole Miss: ESPN location "Ole Miss"; Pinnacle "Mississippi"
            ("Ole Miss", "ole miss"),
            ("Mississippi", "ole miss"),
            ("Mississippi State", "mississippi state"),  # must stay distinct
            # Miami FL vs Miami OH
            ("Miami FL", "miami"),
            ("Miami", "miami"),
            ("Miami (OH)", "miami oh"),
            ("Miami OH", "miami oh"),
            # USC vs South Carolina (share no canonical)
            ("USC", "usc"),
            ("Southern California", "usc"),
            ("South Carolina", "south carolina"),
            # ESPN's non-obvious location spellings
            ("App State", "app state"),
            ("Appalachian State", "app state"),
            ("UConn", "uconn"),
            ("Connecticut", "uconn"),
            ("Southern Mississippi", "southern miss"),
            ("Ohio State Buckeyes", "ohio state"),
            ("LSU", "lsu"),
        ],
    )
    def test_reconciles(self, nz, source, canonical):
        assert nz.normalize(source) == canonical

    def test_accented_san_jose_state(self, nz):
        # ESPN keeps the accent ("San José State"); the normalizer does NOT fold
        # diacritics, so the no-accent source form must alias to the accented
        # canonical or matching silently fails.
        assert nz.normalize("San Jose State") == "san josé state"
        assert nz.normalize("San José State") == "san josé state"

    def test_usc_not_south_carolina(self, nz):
        # The single most dangerous CFB collision — they must never share a key.
        assert nz.normalize("USC") != nz.normalize("South Carolina")

    def test_enrich_market_normalizes(self):
        h = get_handler("ncaaf")
        m = PredictionMarket(
            id="kalshi:KXNCAAFGAME-25AUG30OSUTEX-OSU",
            source=MarketSource.kalshi,
            sector="ncaaf",
            ticker="KXNCAAFGAME-25AUG30OSUTEX-OSU",
            title="Ohio State vs Texas",
            team_home="Ohio State Buckeyes",
            team_away="Texas Longhorns",
            market_type=MarketType.moneyline,
            yes_price=0.55,
            no_price=0.45,
        )
        out = h.enrich_market(m)
        assert out.team_home == "ohio state"
        assert out.team_away == "texas"
