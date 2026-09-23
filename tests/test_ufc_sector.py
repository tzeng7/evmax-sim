"""UFC sector wiring tests — Kalshi parsing, name matching, resolver routing.

Kalshi fixture captured live 2026-07-11 (see tests/fixtures/ufc/README.md).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from evmax.clients.kalshi import KalshiClient, SECTOR_SERIES_MAP
from evmax.clients.time_util import kalshi_game_day
from evmax.matching.engine import MatchingEngine
from evmax.models.market import MarketSource, MarketType, PredictionMarket
from evmax.models.odds import SharpBook, SharpOdds
from evmax.sectors.registry import get_handler

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "ufc"


def _load(name: str) -> dict:
    with open(FIXTURE_DIR / name) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Sector handler — surname normalization
# ---------------------------------------------------------------------------


class TestUFCHandler:
    def test_full_name_to_surname(self):
        h = get_handler("ufc")
        assert h.normalize_team("Conor McGregor") == "mcgregor"
        assert h.normalize_team("Ian Machado Garry") == "garry"

    def test_hyphenated_surname_survives(self):
        h = get_handler("ufc")
        assert h.normalize_team("Benoit Saint-Denis") == "saint-denis"
        assert h.normalize_team("Saint-Denis") == "saint-denis"

    def test_east_asian_name_order_aliases(self):
        h = get_handler("ufc")
        assert h.normalize_team("Zhang Weili") == "zhang"
        assert h.normalize_team("Weili Zhang") == "zhang"

    def test_bare_surname_passthrough(self):
        h = get_handler("ufc")
        assert h.normalize_team("Pimblett") == "pimblett"

    def test_generational_suffix_stripped(self):
        """Kalshi "Raul Rosas Jr" and Pinnacle "Raul Rosas Jr." used to
        normalize to "jr" / "jr." — neither each other nor the right fighter."""
        from evmax.matching.normalizer import NameNormalizer
        h = get_handler("ufc")
        norm = NameNormalizer("ufc")
        assert h.normalize_team("Raul Rosas Jr") == "rosas"
        assert norm.normalize("Raul Rosas Jr") == "rosas"
        assert norm.normalize("Raul Rosas Jr.") == "rosas"
        assert norm.normalize("Kai Kamaka III") == "kamaka"
        assert norm.normalize("Khalil Rountree Jr.") == "rountree"
        assert norm.normalize("Rosas Jr") == "rosas"
        # A bare suffix token is not a name to strip down to nothing.
        assert h.normalize_team("Jr") == "jr"

    def test_venue_spelling_aliases(self):
        """2026-09-22 live gaps: Kalshi spelling vs Pinnacle/ESPN spelling."""
        from evmax.matching.normalizer import NameNormalizer
        norm = NameNormalizer("ufc")
        assert norm.normalize("Norma Dumont Viana") == norm.normalize("Norma Dumont") == "dumont"
        assert norm.normalize("Dumont Viana") == "dumont"
        assert norm.normalize("Alateng Heili") == norm.normalize("Alatengheili") == "alatengheili"
        assert norm.normalize("Heili") == "alatengheili"
        # Canonical targets are idempotent under re-normalization.
        for canon in ("dumont", "alatengheili", "rosas"):
            assert norm.normalize(canon) == canon

    def test_moneyline_only(self):
        assert get_handler("ufc").market_types_supported() == [MarketType.moneyline]


# ---------------------------------------------------------------------------
# Kalshi title parsing
# ---------------------------------------------------------------------------


class TestUFCTitleParsing:
    TITLE_SURNAMES = (
        "Will Benoit Saint-Denis win the Saint-Denis vs Pimblett professional "
        "MMA fight scheduled for Jul 11, 2026?"
    )
    TITLE_FULL_NAMES = (
        "Will Conor McGregor win the Conor McGregor vs Max Holloway "
        "professional MMA fight scheduled for Jul 11, 2026?"
    )

    def test_surname_form(self):
        client = KalshiClient()
        first, second = client._extract_ufc_fighters_from_title(self.TITLE_SURNAMES)
        assert (first, second) == ("Saint-Denis", "Pimblett")

    def test_full_name_form(self):
        client = KalshiClient()
        first, second = client._extract_ufc_fighters_from_title(self.TITLE_FULL_NAMES)
        assert (first, second) == ("Conor McGregor", "Max Holloway")

    def test_yes_fighter_normalized(self):
        client = KalshiClient()
        assert client._extract_tennis_yes_player(self.TITLE_SURNAMES, "ufc") == "saint-denis"
        assert client._extract_tennis_yes_player(self.TITLE_FULL_NAMES, "ufc") == "mcgregor"

    def test_garbage_title(self):
        client = KalshiClient()
        assert client._extract_ufc_fighters_from_title("Will it rain tomorrow?") == (None, None)


class TestKalshiUFCFixture:
    """End-to-end parse of the captured KXUFCFIGHT payload."""

    @pytest.mark.asyncio
    async def test_get_markets_parses_ufc_fights(self, monkeypatch):
        fixture = _load("kxufcfight_markets.json")

        async def fake_get(path, params=None):
            if path == "/markets":
                return fixture
            return {}

        from evmax import settings as settings_module
        settings = settings_module.get_settings()
        monkeypatch.setattr(settings, "offline_mode", False)
        monkeypatch.setattr(settings, "cache_ttl_secs", 0)
        monkeypatch.setattr(settings, "kalshi_ws_enabled", False)

        async with KalshiClient() as client:
            with patch.object(client, "_get", side_effect=fake_get):
                markets = await client.get_markets(sector="ufc")

        assert len(markets) == 4
        by_ticker = {m.ticker: m for m in markets}
        sai = by_ticker["KXUFCFIGHT-26JUL11SAIPIM-SAI"]
        assert sai.market_type == MarketType.moneyline
        assert sai.yes_team == "saint-denis"
        # Ticker order away-first ({AWAY}{HOME} = SAIPIM) → home = PIM. The
        # sibling yes_sub_titles (full names) are the primary fighter source
        # since the 2026-08 short-title switch; the long-form title only
        # backs them up (see TestUFCLegacyTitleFallback).
        assert sai.team_home == "Paddy Pimblett"
        assert sai.team_away == "Benoit Saint-Denis"
        assert sai.event_date.date().isoformat() == "2026-07-11"

        mcg = by_ticker["KXUFCFIGHT-26JUL11MCGHOL-MCG"]
        assert mcg.yes_team == "mcgregor"
        assert mcg.team_home == "Max Holloway"
        assert mcg.team_away == "Conor McGregor"

    def test_ufc_series_registered(self):
        assert SECTOR_SERIES_MAP["ufc"] == ["KXUFCFIGHT"]

    def test_ufc_joins_events(self):
        from evmax.clients.kalshi import _EVENT_TITLE_SECTORS
        assert "ufc" in _EVENT_TITLE_SECTORS


# ---------------------------------------------------------------------------
# 2026-08 short-title format ("{Name} wins") — live capture 2026-09-22
# ---------------------------------------------------------------------------

SHORT_MARKETS = "kxufcfight_short_title_markets.json"
SHORT_EVENTS = "kxufcfight_short_title_events.json"

# event_ticker → (away, home) canonical surnames. Ticker {AWAY}{HOME} order ==
# event sub_title order == Pinnacle's (away, home) on every live fight.
SHORT_TITLE_EXPECTED = {
    "KXUFCFIGHT-26SEP26GALDUM": ("gall", "dumas"),
    "KXUFCFIGHT-26SEP26DUMPER": ("dumont", "perez"),          # "Norma Dumont Viana" alias
    "KXUFCFIGHT-26SEP26CASHEI": ("castaneda", "alatengheili"),  # "Alateng Heili" alias
    "KXUFCFIGHT-26SEP26ROSBAR": ("rosas", "barcelos"),        # "Raul Rosas Jr" suffix
    "KXUFCFIGHT-26OCT03DOSHER": ("anjos", "hernandez"),
    "KXUFCFIGHT-26OCT03SILCON": ("silva", "wang"),            # "Wang Cong" family-first
    "KXUFCFIGHT-26OCT03FIGTAL": ("figueiredo", "talbott"),
}

# Pinnacle (sport 22) labels for the same cards, captured live 2026-09-22:
# (startTime, home, away). Dos Anjos vs Hernandez was not yet posted.
PINNACLE_LIVE = [
    ("2026-09-26T19:00:00+00:00", "Sedriques Dumas", "Mickey Gall"),
    ("2026-09-26T19:00:00+00:00", "Ailin Perez", "Norma Dumont"),
    ("2026-09-26T19:00:00+00:00", "Alatengheili", "John Castaneda"),
    ("2026-09-26T19:00:00+00:00", "Raoni Barcelos", "Raul Rosas Jr."),
    ("2026-10-03T21:00:00+00:00", "Wang Cong", "Natalia Silva"),
    ("2026-10-03T21:00:00+00:00", "Payton Talbott", "Deiveson Figueiredo"),
]


async def _parse_ufc(markets: dict, events=None, events_exc=None) -> list[PredictionMarket]:
    async def fake_get(path, params=None):
        if path == "/markets":
            return markets
        if path == "/events":
            if events_exc is not None:
                raise events_exc
            return events if events is not None else {"events": []}
        return {}

    async with KalshiClient() as client:
        with patch.object(client, "_get", side_effect=fake_get):
            return await client.get_markets(sector="ufc")


def _pinnacle(start: str, home: str, away: str) -> SharpOdds:
    from evmax.matching.normalizer import NameNormalizer
    event_date = datetime.fromisoformat(start)
    norm = NameNormalizer("ufc")
    key = (
        f"ufc::{kalshi_game_day(event_date, 'ufc')}::"
        f"{norm.normalize(home).replace(' ', '_')}_vs_{norm.normalize(away).replace(' ', '_')}"
    )
    return SharpOdds(
        event_id=key, book=SharpBook.pinnacle, sector="ufc",
        outcome_a_label=home, outcome_b_label=away,
        outcome_a_decimal=1.9, outcome_b_decimal=1.95,
        true_prob_a=0.51, true_prob_b=0.49, margin=0.04, event_date=event_date,
    )


class TestUFCShortTitleFormat:
    """Kalshi moved KXUFCFIGHT to the short "{Full Name} wins" title in
    2026-08 (first seen 08-19, every market from 08-23). The old parser read
    fighters only from the long "Will X win the A vs B ... MMA fight" title,
    so every UFC market parsed with EMPTY fighters and matched zero Pinnacle
    events from 09-05 onward (~4 cards missed). Fighters now come per event
    from the sibling markets' yes_sub_title, with the event sub_title and the
    legacy title as fallbacks."""

    @pytest.fixture(autouse=True)
    def _no_cache(self, monkeypatch):
        from evmax import settings as settings_module
        settings = settings_module.get_settings()
        monkeypatch.setattr(settings, "offline_mode", False)
        monkeypatch.setattr(settings, "cache_ttl_secs", 0)
        monkeypatch.setattr(settings, "kalshi_ws_enabled", False)

    def test_short_title_has_no_matchup_for_the_legacy_regex(self):
        """Pins the break itself: the legacy regex finds nothing here."""
        client = KalshiClient()
        assert client._extract_ufc_fighters_from_title("Mickey Gall wins") == (None, None)
        assert client._extract_tennis_yes_player("Mickey Gall wins", "ufc") is None

    @pytest.mark.asyncio
    async def test_every_live_market_yields_both_fighters_and_yes(self):
        from evmax.matching.normalizer import NameNormalizer
        raw = {r["ticker"]: r for r in _load(SHORT_MARKETS)["markets"]}
        markets = await _parse_ufc(_load(SHORT_MARKETS), _load(SHORT_EVENTS))
        assert len(markets) == 14
        norm = NameNormalizer("ufc")
        for m in markets:
            r = raw[m.ticker]
            assert r["title"].endswith(" wins")  # this IS the short format
            away, home = SHORT_TITLE_EXPECTED[r["event_ticker"]]
            assert (norm.normalize(m.team_away), norm.normalize(m.team_home)) == (away, home), m.ticker
            assert m.yes_team == norm.normalize(r["yes_sub_title"])
            assert m.yes_team in (away, home)
            assert m.market_type == MarketType.moneyline
        # Fighters are the FULL names from the sibling yes_sub_titles (the
        # form the alias map is keyed on), oriented by the ticker codes even
        # though Kalshi lists the PER market before the DUM market.
        dum = {m.ticker: m for m in markets}["KXUFCFIGHT-26SEP26DUMPER-DUM"]
        assert (dum.team_away, dum.team_home) == ("Norma Dumont Viana", "Ailin Perez")
        assert dum.yes_team == "dumont"
        assert dum.competition == "Fight Night"

    @pytest.mark.asyncio
    async def test_matches_live_pinnacle_and_aligns_yes(self):
        """End to end through the production MatchingEngine + YES alignment:
        every fight Pinnacle lists matches at the exact key (100) and each
        YES contract prices its own fighter."""
        from evmax.matching.alignment import YesOutcome, align_yes_side
        markets = await _parse_ufc(_load(SHORT_MARKETS), _load(SHORT_EVENTS))
        sharps = [_pinnacle(*row) for row in PINNACLE_LIVE]
        engine = MatchingEngine()
        pairs = engine.match_all(markets, sharps)
        assert len(pairs) == 12  # 6 posted fights × 2 YES markets
        assert {m.ticker for m, _, _ in pairs}.isdisjoint(
            {"KXUFCFIGHT-26OCT03DOSHER-DOS", "KXUFCFIGHT-26OCT03DOSHER-HER"}
        )
        norm = engine.normalizer_for("ufc")
        for market, sharp, conf in pairs:
            assert conf == 100.0, market.ticker
            a = align_yes_side(market, sharp, norm)
            assert a is not None, market.ticker
            label = sharp.outcome_a_label if a.outcome is YesOutcome.A else sharp.outcome_b_label
            assert norm.normalize(label) == market.yes_team, market.ticker

    @pytest.mark.asyncio
    async def test_events_failure_still_recovers_fighters_from_siblings(self):
        markets = await _parse_ufc(
            _load(SHORT_MARKETS), events_exc=RuntimeError("simulated 400 Bad Request"),
        )
        from evmax.matching.normalizer import NameNormalizer
        norm = NameNormalizer("ufc")
        raw = {r["ticker"]: r for r in _load(SHORT_MARKETS)["markets"]}
        assert len(markets) == 14
        for m in markets:
            away, home = SHORT_TITLE_EXPECTED[raw[m.ticker]["event_ticker"]]
            assert (norm.normalize(m.team_away), norm.normalize(m.team_home)) == (away, home)
            assert m.competition is None  # the events join is what's lost

    @pytest.mark.asyncio
    async def test_event_subtitle_fallback_without_yes_sub_title(self):
        """No yes_sub_title anywhere → fighters from the event sub_title
        (Kalshi's surname form) and the YES fighter from the short title."""
        import copy
        mk = copy.deepcopy(_load(SHORT_MARKETS))
        for r in mk["markets"]:
            r.pop("yes_sub_title", None)
        markets = await _parse_ufc(mk, _load(SHORT_EVENTS))
        from evmax.matching.normalizer import NameNormalizer
        norm = NameNormalizer("ufc")
        raw = {r["ticker"]: r for r in mk["markets"]}
        assert len(markets) == 14
        for m in markets:
            et = raw[m.ticker]["event_ticker"]
            got = (norm.normalize(m.team_away), norm.normalize(m.team_home))
            if et == "KXUFCFIGHT-26OCT03SILCON":
                # The sub_title surname form "Silva vs Cong" loses the
                # family-first order the "Wang Cong" alias needs — why the
                # full-name siblings are the primary source.
                assert got == ("silva", "cong")
            else:
                assert got == SHORT_TITLE_EXPECTED[et], m.ticker
            assert m.yes_team  # from "{Name} wins"
        ros = {m.ticker: m for m in markets}["KXUFCFIGHT-26SEP26ROSBAR-ROS"]
        assert (ros.team_away, ros.team_home) == ("Rosas Jr", "Barcelos")
        assert ros.yes_team == "rosas"


class TestUFCLegacyTitleFallback:
    """Old long-form titles ("Will X win the A vs B professional MMA
    fight ...") must keep parsing (cached rows, a Kalshi revert)."""

    @pytest.fixture(autouse=True)
    def _no_cache(self, monkeypatch):
        from evmax import settings as settings_module
        settings = settings_module.get_settings()
        monkeypatch.setattr(settings, "offline_mode", False)
        monkeypatch.setattr(settings, "cache_ttl_secs", 0)
        monkeypatch.setattr(settings, "kalshi_ws_enabled", False)

    @pytest.mark.asyncio
    async def test_long_title_without_subtitles_or_events(self):
        import copy
        mk = copy.deepcopy(_load("kxufcfight_markets.json"))
        for r in mk["markets"]:
            r.pop("yes_sub_title", None)
        markets = await _parse_ufc(mk)
        by_ticker = {m.ticker: m for m in markets}
        assert len(by_ticker) == 4
        sai = by_ticker["KXUFCFIGHT-26JUL11SAIPIM-SAI"]
        assert (sai.team_away, sai.team_home) == ("Saint-Denis", "Pimblett")
        assert sai.yes_team == "saint-denis"
        mcg = by_ticker["KXUFCFIGHT-26JUL11MCGHOL-MCG"]
        assert (mcg.team_away, mcg.team_home) == ("Conor McGregor", "Max Holloway")
        assert mcg.yes_team == "mcgregor"

    def test_direct_parse_market_long_title(self):
        raw = _load("kxufcfight_markets.json")["markets"][1]  # PIM market
        m = KalshiClient()._parse_market(raw, "ufc")
        assert (m.team_away, m.team_home) == ("Saint-Denis", "Pimblett")
        assert m.yes_team == "pimblett"


class TestUFCEventFighterHelpers:
    def test_matchup_from_event_forms(self):
        from evmax.clients.kalshi import _ufc_matchup_from_event as mu
        assert mu({"sub_title": "Demopoulos vs Jauregui"}) == ("Demopoulos", "Jauregui")
        assert mu({"title": "332: McGee vs Nolan"}) == ("McGee", "Nolan")
        assert mu({"title": "Fight Night: Rosas Jr vs Barcelos"}) == ("Rosas Jr", "Barcelos")
        assert mu({"sub_title": "Gall vs. Dumas (Sep 26)"}) == ("Gall", "Dumas")
        assert mu({"title": "UFC 332"}) is None
        assert mu({}) is None

    def test_orient_by_ticker_codes(self):
        from evmax.clients.kalshi import _orient_ufc_siblings as orient
        # Kalshi listed the HOME fighter first; the {AWAY}{HOME} pair fixes it.
        got = orient(
            "KXUFCFIGHT-26SEP26DEMJAU",
            {"JAU": "Yazmin Jauregui", "DEM": "Vanessa Demopoulos"},
            None,
        )
        assert got == ("Vanessa Demopoulos", "Yazmin Jauregui")

    def test_orient_by_matchup_when_codes_do_not_reconcile(self):
        from evmax.clients.kalshi import _orient_ufc_siblings as orient
        sib = {"HER": "Alexander Hernandez", "DOS1": "Rafael Dos Anjos"}
        assert orient("KXUFCFIGHT-26OCT03DOSHER", sib, ("Anjos", "Hernandez")) == (
            "Rafael Dos Anjos", "Alexander Hernandez",
        )
        # Matchup that names neither sibling → None (caller uses the matchup).
        assert orient("KXUFCFIGHT-26OCT03DOSHER", sib, ("Smith", "Jones")) is None
        # No matchup at all → listed order (fuzzy matching is order-insensitive).
        assert orient("KXUFCFIGHT-26OCT03DOSHER", sib, None) == (
            "Alexander Hernandez", "Rafael Dos Anjos",
        )

    def test_orient_rejects_unusable_siblings(self):
        from evmax.clients.kalshi import _orient_ufc_siblings as orient
        assert orient("KXUFCFIGHT-26SEP26GALDUM", {"GAL": "Mickey Gall"}, None) is None
        assert orient(
            "KXUFCFIGHT-26SEP26GALDUM", {"GAL": "Mickey Gall", "DUM": "mickey gall"}, None,
        ) is None

    def test_event_fighters_prefers_siblings_then_matchup(self):
        from evmax.clients.kalshi import _ufc_event_fighters
        markets = [
            {"ticker": "KXUFCFIGHT-26OCT03SILCON-SIL", "event_ticker": "KXUFCFIGHT-26OCT03SILCON",
             "yes_sub_title": "Natalia Silva"},
            {"ticker": "KXUFCFIGHT-26OCT03SILCON-CON", "event_ticker": "KXUFCFIGHT-26OCT03SILCON",
             "yes_sub_title": "Wang Cong"},
            # Second event: only ONE sibling listed → matchup fallback.
            {"ticker": "KXUFCFIGHT-26OCT03GAUKOP-GAU", "event_ticker": "KXUFCFIGHT-26OCT03GAUKOP",
             "yes_sub_title": "Ateba Gautier"},
        ]
        events = [
            {"event_ticker": "KXUFCFIGHT-26OCT03SILCON", "sub_title": "Silva vs Cong"},
            {"event_ticker": "KXUFCFIGHT-26OCT03GAUKOP", "sub_title": "Gautier vs Kopylov"},
        ]
        got = _ufc_event_fighters(markets, events)
        assert got["KXUFCFIGHT-26OCT03SILCON"] == ("Natalia Silva", "Wang Cong")
        assert got["KXUFCFIGHT-26OCT03GAUKOP"] == ("Gautier", "Kopylov")
        assert _ufc_event_fighters(None, None) == {}


# ---------------------------------------------------------------------------
# Game-day alignment + matching
# ---------------------------------------------------------------------------


class TestUFCMatching:
    def test_us_calendar_day_for_late_utc_start(self):
        """Pinnacle startTime 2026-07-12T02:50Z is a Jul 11 ET card — the
        game-day must match Kalshi's 26JUL11 ticker date."""
        dt = datetime(2026, 7, 12, 2, 50, tzinfo=timezone.utc)
        assert kalshi_game_day(dt, "ufc") == "2026-07-11"

    def _kalshi_market(self) -> PredictionMarket:
        return PredictionMarket(
            id="kalshi:KXUFCFIGHT-26JUL11SAIPIM-SAI",
            source=MarketSource.kalshi,
            sector="ufc",
            market_type=MarketType.moneyline,
            title="Will Benoit Saint-Denis win the Saint-Denis vs Pimblett professional MMA fight scheduled for Jul 11, 2026?",
            ticker="KXUFCFIGHT-26JUL11SAIPIM-SAI",
            yes_price=0.58,
            no_price=0.45,
            volume_usd=1500.0,
            team_home="Pimblett",
            team_away="Saint-Denis",
            event_date=datetime(2026, 7, 11, 12, tzinfo=timezone.utc),
            yes_team="saint-denis",
        )

    def _pinnacle_sharp(self, home="Paddy Pimblett", away="Benoit Saint-Denis") -> SharpOdds:
        event_date = datetime(2026, 7, 12, 2, 50, tzinfo=timezone.utc)
        date_str = kalshi_game_day(event_date, "ufc")
        from evmax.matching.normalizer import NameNormalizer
        norm = NameNormalizer("ufc")
        home_n = norm.normalize(home).replace(" ", "_")
        away_n = norm.normalize(away).replace(" ", "_")
        return SharpOdds(
            event_id=f"ufc::{date_str}::{home_n}_vs_{away_n}",
            book=SharpBook.pinnacle,
            sector="ufc",
            outcome_a_label=home,
            outcome_b_label=away,
            outcome_a_decimal=1.8,
            outcome_b_decimal=2.1,
            true_prob_a=0.55,
            true_prob_b=0.45,
            margin=0.04,
            event_date=event_date,
        )

    def test_exact_key_match(self):
        engine = MatchingEngine()
        market = self._kalshi_market()
        sharp = self._pinnacle_sharp()
        assert engine.build_market_key(market) == sharp.event_id
        result = engine.match(market, [sharp])
        assert result is not None
        assert result[1] == 100.0

    def test_swapped_home_away_still_matches_fuzzy(self):
        """If Kalshi ever lists the fighters in the opposite order, the
        token-sort fuzzy fallback must still land the match."""
        engine = MatchingEngine()
        market = self._kalshi_market().model_copy(
            update={"team_home": "Saint-Denis", "team_away": "Pimblett"}
        )
        result = engine.match(market, [self._pinnacle_sharp()])
        assert result is not None

    def test_wrong_fight_does_not_match(self):
        engine = MatchingEngine()
        market = self._kalshi_market()
        other = self._pinnacle_sharp(home="Max Holloway", away="Conor McGregor")
        assert engine.match(market, [other]) is None


# ---------------------------------------------------------------------------
# Resolver routing + registry consistency
# ---------------------------------------------------------------------------


class TestUFCWiring:
    def test_resolver_routes_to_kalshi_settlement(self):
        from evmax.agents.cleanup.resolver import (
            ESPN_SPORT_MAP,
            KALSHI_SETTLEMENT_SECTORS,
        )
        assert "ufc" in KALSHI_SETTLEMENT_SECTORS
        # Must NOT be in the ESPN scoreboard path — that branch dispatches
        # first and resolves by score comparison, meaningless for fights.
        assert "ufc" not in ESPN_SPORT_MAP

    def test_category_registered_shadow_kalshi_settlement(self):
        from evmax.categories import get_category
        spec = get_category("ufc")
        assert spec.mode == "shadow"
        assert spec.resolver == "kalshi_settlement"
        assert spec.market_types == (MarketType.moneyline,)
        assert "ufc_rating" in spec.models and "sharp" in spec.models

    def test_ensemble_override_pins_generic_models_to_zero(self):
        from evmax.agents.models.ensemble_agent import EnsembleModelAgent
        override = EnsembleModelAgent.SECTOR_WEIGHT_OVERRIDES["ufc"]
        assert override["ufc_rating"] > 0
        assert override["elo"] == 0.0
        assert override["form"] == 0.0

    def test_no_required_blend_gate_for_ufc(self):
        """A zero-weighted model must never shadow-demote UFC plays — that
        only happens for sectors with a REQUIRED_BLEND_MODELS entry."""
        from evmax.agents.odds.ev_gap_agent import REQUIRED_BLEND_MODELS, has_full_blend
        assert "ufc" not in REQUIRED_BLEND_MODELS
        assert has_full_blend("ufc", "sharp") is True

    def test_coordinator_registers_ufc_agent(self):
        from evmax.agents.coordinator import AgentCoordinator
        coord = AgentCoordinator(sectors=["ufc"], respect_season_window=False)
        assert any(
            type(m).__name__ == "UFCRatingAgent"
            for m in coord.ensemble_agent._models
        )
