"""Esports (lol / cs2) Kalshi parsing + matching after the 2026-08 title break.

Kalshi moved KXLOLGAME / KXCS2GAME per-team titles to the short "{Team} wins"
form (first seen 2026-08-19, every market from 08-23). The parser keyed
esports events off the ticker codes (KXLOLGAME-26SEP261200FURRED ->
``red_vs_fur``), which never equal Pinnacle's full names
(``furia_vs_red_canids``), so lol/cs2 matched only by coincidence and 0 from
2026-09-21. Teams now come from the per-event join (sibling yes_sub_titles
oriented by the ticker code pair, then the event title), the same approach as
the UFC fix (#325).

Fixtures captured live 2026-09-24 (see tests/fixtures/esports/README.md).
"""

from __future__ import annotations

import collections
import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from evmax.clients.esports_pinnacle import PinnacleGuestClient
from evmax.clients.kalshi import KalshiClient, SECTOR_SERIES_MAP
from evmax.matching.alignment import (
    PRICE_FALLBACK_SECTORS,
    YesOutcome,
    align_yes_side,
    alignment_looks_suspect,
)
from evmax.matching.engine import MatchingEngine
from evmax.matching.normalizer import NameNormalizer
from evmax.models.market import MarketType, PredictionMarket
from evmax.models.odds import SharpOdds
from evmax.sectors.registry import get_handler

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "esports"
ALIASES = Path(__file__).parent.parent / "evmax" / "sectors" / "aliases"
SERIES = {"lol": "KXLOLGAME", "cs2": "KXCS2GAME"}


def _load(name: str) -> dict:
    with open(FIXTURE_DIR / name) as f:
        return json.load(f)


def _markets(sector: str) -> dict:
    return _load(f"{SERIES[sector].lower()}_short_title_markets.json")


def _events(sector: str) -> dict:
    return _load(f"{SERIES[sector].lower()}_short_title_events.json")


def _sharps(sector: str) -> list[SharpOdds]:
    return [
        SharpOdds.model_validate(r)
        for r in _load("pinnacle_esports_odds.json")
        if r["sector"] == sector
    ]


# event_ticker -> (away, home) canonical names. Away = the ticker's first code
# = the event title's first-listed team (226/226 live markets, 2026-09-24).
EXPECTED = {
    "lol": {
        "KXLOLGAME-26SEP261200FURRED": ("furia", "red canids"),           # "FURIA Esports"
        "KXLOLGAME-26SEP240500KTCKOIA": ("kt rolster challengers", "movistar koi fenix"),  # "Fénix"
        "KXLOLGAME-26SEP240300EDGYFUE": ("edward youth", "fuego"),       # "EDward Gaming Youth Team"
        "KXLOLGAME-26SEP251600FLYSR": ("flyquest", "sr"),                # "Shopify Rebellion" alias
        "KXLOLGAME-26SEP240400BLGJCTBCA": ("bilibili junior", "ctbc flying oyster academy"),
        "KXLOLGAME-26SEP271600TLC9": ("liquid", "cloud9"),               # C9 used to parse as "c"
        "KXLOLGAME-26SEP2405009GSGW": ("9gaming", "saigon warriors"),    # time 0500 + code 9G
    },
    "cs2": {
        "KXCS2GAME-26SEP240800JUSFNC": ("just players", "fnatic"),       # "Just_Players" alias
        "KXCS2GAME-26SEP241230KEYDTDP": ("keyd stars", "turma do pagode"),  # "Keyd" alias
        "KXCS2GAME-26SEP240500NEMRUN": ("nemesis", "rune eaters"),       # "Team Nemesis"
        "KXCS2GAME-26SEP241000BIGAEXZ": ("big academy", "ex-zero tenacity"),
        "KXCS2GAME-26SEP240900BHEPAIN": ("bounty hunters", "pain"),
        "KXCS2GAME-26SEP252100OTLAG": ("overtake sector", "lag"),        # "Overtake" alias
        "KXCS2GAME-26SEP250500GROTHU": ("ground zero", "thunder downunder"),
        "KXCS2GAME-26SEP2508003DMAXNIP": ("3dmax", "nip"),               # time 0800 + code 3DMAX
        "KXCS2GAME-26SEP2407154MINF": ("four magic", "infinite talent"),  # time 0715 + code 4M
        "KXCS2GAME-26SEP251400K27GL": ("k27", "gamerlegion"),            # K27 used to parse as "k"
    },
}

# Fixture events Pinnacle also listed at capture (the rest had no Pinnacle
# record — Pinnacle posts esports ~1-2 days out).
PINNACLE_LISTED = {
    "lol": {
        "KXLOLGAME-26SEP261200FURRED", "KXLOLGAME-26SEP240500KTCKOIA",
        "KXLOLGAME-26SEP240300EDGYFUE", "KXLOLGAME-26SEP251600FLYSR",
        "KXLOLGAME-26SEP240400BLGJCTBCA",
    },
    "cs2": {
        "KXCS2GAME-26SEP240800JUSFNC", "KXCS2GAME-26SEP241230KEYDTDP",
        "KXCS2GAME-26SEP240500NEMRUN", "KXCS2GAME-26SEP241000BIGAEXZ",
        "KXCS2GAME-26SEP240900BHEPAIN",
    },
}


async def _parse(sector: str, markets: dict, events=None, events_exc=None) -> list[PredictionMarket]:
    series = SERIES[sector]

    async def fake_get(path, params=None):
        if params and params.get("series_ticker") != series:
            return {"markets": [], "events": []}  # KXCS2GAMES: empty today
        if path == "/markets":
            return markets
        if path == "/events":
            if events_exc is not None:
                raise events_exc
            return events if events is not None else {"events": []}
        return {}

    async with KalshiClient() as client:
        with patch.object(client, "_get", side_effect=fake_get):
            return await client.get_markets(sector=sector)


@pytest.fixture
def no_cache(monkeypatch):
    from evmax import settings as settings_module
    settings = settings_module.get_settings()
    monkeypatch.setattr(settings, "offline_mode", False)
    monkeypatch.setattr(settings, "cache_ttl_secs", 0)
    monkeypatch.setattr(settings, "kalshi_ws_enabled", False)


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


class TestEsportsWiring:
    def test_series_registered(self):
        assert SECTOR_SERIES_MAP["lol"] == ["KXLOLGAME"]
        assert SECTOR_SERIES_MAP["cs2"] == ["KXCS2GAME", "KXCS2GAMES"]

    def test_esports_join_events(self):
        from evmax.clients.kalshi import _ESPORTS_TITLE_SECTORS, _EVENT_TITLE_SECTORS
        assert {"lol", "cs2"} <= set(_EVENT_TITLE_SECTORS)
        assert _ESPORTS_TITLE_SECTORS == {"lol", "cs2"}

    def test_price_fallback_still_scoped_to_esports(self):
        assert PRICE_FALLBACK_SECTORS == {"lol", "cs2"}


# ---------------------------------------------------------------------------
# Name folding + aliases
# ---------------------------------------------------------------------------


class TestEsportsNameNormalization:
    @pytest.mark.parametrize("sector,kalshi,pinnacle", [
        ("lol", "Movistar KOI Fénix", "Movistar KOI Fenix"),        # accent
        ("lol", "FURIA Esports", "FURIA"),                          # noise word
        ("lol", "EDward Gaming Youth Team", "EDward Gaming Youth"),
        ("lol", "Bilibili Gaming Junior", "Bilibili Junior"),
        ("lol", "Bilibili Gaming", "Bilibili"),
        ("lol", "Gen.G", "Gen.G"),                                  # dot
        ("lol", "devils.one x KMT", "devils.one"),
        ("lol", "Ici Japon Corp. Esport", "Ici Japon Corp"),
        ("lol", "Skillcamp Esport", "Skillcamp"),
        ("lol", "Pyramid IV Esports", "Pyramid IV"),
        ("lol", "INTZ e-Sports", "INTZ"),
        ("lol", "KaBuM! Ilha das Lendas", "KaBuM IDL"),
        ("lol", "OKSavingsBank BRION", "HANJIN BRION"),
        ("lol", "Nongshim Esports Academy", "Nongshim RedForce Challengers"),
        ("cs2", "Just_Players", "Just Players"),                    # underscore
        ("cs2", "Keyd", "Keyd Stars"),
        ("cs2", "Overtake", "Overtake Sector"),
        ("cs2", "Team Nemesis", "Nemesis"),
        ("cs2", "KUUSAMO.gg", "KUUSAMO"),
        ("cs2", "BetBoom Team", "BB Team"),
        ("cs2", "1WIN", "1W"),
        ("cs2", "TheMongolz", "The MongolZ"),
        ("cs2", "Esport Academy Copenhagen", "EAC"),
        ("cs2", "BC.Game Esports", "BC.Game"),                      # dot + noise word
    ])
    def test_kalshi_and_pinnacle_spellings_meet(self, sector, kalshi, pinnacle):
        n = NameNormalizer(sector)
        assert n.normalize(kalshi) == n.normalize(pinnacle)
        # ...and the venue-side key equals the key the Pinnacle client builds.
        assert n.normalize(kalshi).replace(" ", "_") == PinnacleGuestClient._normalize(pinnacle, sector)

    @pytest.mark.parametrize("sector,a,b", [
        # Distinct rosters must stay distinct (never guess).
        ("cs2", "WBT Academy", "WBT"),
        ("cs2", "BIG Academy", "BIG"),
        ("lol", "KaBuM", "KaBuM IDL"),
        ("lol", "HANJIN BRION Challengers", "HANJIN BRION"),
        ("lol", "Nongshim RedForce Challengers", "Nongshim Redforce"),
        ("lol", "Bilibili Junior", "Bilibili"),
        ("cs2", "Keyd Stars", "Vivo Keyd Stars Academy"),
    ])
    def test_distinct_teams_stay_distinct(self, sector, a, b):
        n = NameNormalizer(sector)
        assert n.normalize(a) != n.normalize(b)

    def test_dot_and_accent_folds_are_esports_only(self):
        assert get_handler("lol").strip_dots and get_handler("cs2").strip_dots
        assert get_handler("lol").fold_accents and get_handler("cs2").fold_accents
        for sector in ("nba", "nfl", "ncaab", "ncaaf", "soccer", "worldcup", "tennis", "ufc", "nhl"):
            assert get_handler(sector).strip_dots is False, sector
        # Non-esports names keep their dots exactly as before.
        assert NameNormalizer("soccer").normalize("St. Pauli") == "st. pauli"

    def test_fold_name_is_identity_without_flags(self):
        h = get_handler("nba")
        assert h.fold_name("gen.g fénix") == "gen.g fénix"
        assert get_handler("soccer").fold_name("alavés f.c.") == "alaves f.c."
        assert get_handler("lol").fold_name("gen.g fénix") == "geng fenix"

    @pytest.mark.parametrize("sector", ["lol", "cs2"])
    def test_alias_keys_fold_without_collision(self, sector):
        """Two alias keys that differ only by accents/dots must point at one team."""
        h = get_handler(sector)
        raw = yaml.safe_load((ALIASES / f"{sector}.yaml").read_text())["aliases"]
        folded: dict[str, set[str]] = collections.defaultdict(set)
        for key, target in raw.items():
            folded[h.fold_name(str(key))].add(h.fold_name(str(target)))
        assert {k: v for k, v in folded.items() if len(v) > 1} == {}

    @pytest.mark.parametrize("sector", ["lol", "cs2"])
    def test_normalization_is_idempotent_on_canonicals(self, sector):
        n = NameNormalizer(sector)
        for target in set(get_handler(sector)._aliases.values()):
            assert n.normalize(target) == target, target


# ---------------------------------------------------------------------------
# Per-event join helpers
# ---------------------------------------------------------------------------


class TestEsportsEventHelpers:
    @pytest.mark.parametrize("pair,codes,expected", [
        ("1600TLC9", "TLC9", True),        # HHMM + codes
        ("05009GSGW", "9GSGW", True),      # code starting with a digit
        ("07154MINF", "4MINF", True),
        ("DEMJAU", "DEMJAU", True),        # UFC layout (no time)
        ("1600TLC9", "C9TL", False),       # wrong order
        ("123TLC9", "TLC9", False),        # a 3-digit prefix is not a time
        ("X1600TLC9", "TLC9", False),
        ("TLC9", "", False),
    ])
    def test_ticker_pair_is(self, pair, codes, expected):
        from evmax.clients.kalshi import _ticker_pair_is
        assert _ticker_pair_is(pair, codes) is expected

    def test_matchup_from_event_forms(self):
        from evmax.clients.kalshi import _esports_matchup_from_event as mu
        assert mu({"title": "FURIA Esports vs. RED Canids"}) == ("FURIA Esports", "RED Canids")
        assert mu({"sub_title": "LOUD vs. LOS (Sep 27)"}) == ("LOUD", "LOS")
        # A parenthesized part of a team name is kept; only a date is dropped.
        assert mu({"sub_title": "Los Heretics (OLD) vs. Fuego (Sep 6)"}) == ("Los Heretics (OLD)", "Fuego")
        assert mu({"title": "League of Legends"}) is None
        assert mu({}) is None

    def test_label_names_is_equality_not_partial(self):
        from evmax.clients.kalshi import _esports_label_names as same
        assert same("Movistar KOI Fénix", "movistar koi fenix")
        assert same("FURIA  Esports", "FURIA Esports")
        assert not same("Keyd", "Vivo Keyd Stars")
        assert not same("", "Keyd")

    def test_orients_by_ticker_even_when_home_listed_first(self):
        from evmax.clients.kalshi import _esports_event_teams
        rows = [
            {"ticker": "KXLOLGAME-26SEP261200FURRED-RED", "event_ticker": "KXLOLGAME-26SEP261200FURRED",
             "yes_sub_title": "RED Canids"},
            {"ticker": "KXLOLGAME-26SEP261200FURRED-FUR", "event_ticker": "KXLOLGAME-26SEP261200FURRED",
             "yes_sub_title": "FURIA Esports"},
            {"ticker": "KXLOLGAME-26SEP2405009GSGW-SGW", "event_ticker": "KXLOLGAME-26SEP2405009GSGW",
             "yes_sub_title": "Saigon Warriors"},
            {"ticker": "KXLOLGAME-26SEP2405009GSGW-9G", "event_ticker": "KXLOLGAME-26SEP2405009GSGW",
             "yes_sub_title": "9Gaming"},
        ]
        got = _esports_event_teams(rows, None)
        assert got["KXLOLGAME-26SEP261200FURRED"] == ("FURIA Esports", "RED Canids")
        assert got["KXLOLGAME-26SEP2405009GSGW"] == ("9Gaming", "Saigon Warriors")

    def test_event_title_fallback_with_one_sibling(self):
        from evmax.clients.kalshi import _esports_event_teams
        rows = [{"ticker": "KXCS2GAME-26SEP241230KEYDTDP-TDP", "event_ticker": "KXCS2GAME-26SEP241230KEYDTDP",
                 "yes_sub_title": "Turma do Pagode"}]
        events = [{"event_ticker": "KXCS2GAME-26SEP241230KEYDTDP", "title": "Keyd vs. Turma do Pagode"}]
        assert _esports_event_teams(rows, events) == {
            "KXCS2GAME-26SEP241230KEYDTDP": ("Keyd", "Turma do Pagode"),
        }
        # Neither a sibling pair nor an event: nothing to recover.
        assert _esports_event_teams(rows, None) == {}
        assert _esports_event_teams(None, None) == {}


# ---------------------------------------------------------------------------
# Short-title format, live fixture
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("no_cache")
class TestEsportsShortTitleFormat:
    def test_the_old_ticker_path_never_builds_pinnacle_keys(self):
        """Pins the break: the ticker codes are the only teams the old path had."""
        client = KalshiClient()
        n = NameNormalizer("lol")
        home, away = client._extract_teams_from_ticker("KXLOLGAME-26SEP261200FURRED-FUR", "lol")
        assert (n.normalize(home), n.normalize(away)) == ("red", "fur")
        assert client._extract_esports_teams_from_legacy_title("FURIA Esports wins") == (None, None)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("sector", ["lol", "cs2"])
    async def test_every_live_market_yields_both_teams_and_yes(self, sector):
        raw = {r["ticker"]: r for r in _markets(sector)["markets"]}
        markets = await _parse(sector, _markets(sector), _events(sector))
        assert len(markets) == len(raw) == 2 * len(EXPECTED[sector])
        n = NameNormalizer(sector)
        for m in markets:
            r = raw[m.ticker]
            assert r["title"].endswith(" wins")  # this IS the short format
            away, home = EXPECTED[sector][r["event_ticker"]]
            assert (n.normalize(m.team_away), n.normalize(m.team_home)) == (away, home), m.ticker
            assert m.yes_team == n.normalize(r["yes_sub_title"]), m.ticker
            assert m.yes_team in (away, home), m.ticker
            assert m.market_type == MarketType.moneyline, m.ticker
            assert m.competition == {"lol": "League of Legends", "cs2": "CS2"}[sector]

    @pytest.mark.asyncio
    async def test_keyword_trap_names_stay_moneyline(self):
        """Before: "Overtake wins" / "THUNDER dOWNUNDER wins" parsed as totals,
        "ex-Zero Tenacity wins" as a spread, "Ground Zero wins" as a map
        handicap — each silently dropped from moneyline matching."""
        markets = {m.ticker: m for m in await _parse("cs2", _markets("cs2"), _events("cs2"))}
        for t in ("KXCS2GAME-26SEP252100OTLAG-OT", "KXCS2GAME-26SEP250500GROTHU-THU",
                  "KXCS2GAME-26SEP250500GROTHU-GRO", "KXCS2GAME-26SEP241000BIGAEXZ-EXZ"):
            assert markets[t].market_type == MarketType.moneyline, t
            assert markets[t].line is None

    @pytest.mark.asyncio
    async def test_digit_edged_codes_keep_their_digits(self):
        """_extract_yes_team strips trailing digits (a spread line) — C9 was
        "c" and K27 "k". The yes_sub_title path carries the real names."""
        lol = {m.ticker: m for m in await _parse("lol", _markets("lol"), _events("lol"))}
        cs2 = {m.ticker: m for m in await _parse("cs2", _markets("cs2"), _events("cs2"))}
        assert lol["KXLOLGAME-26SEP271600TLC9-C9"].yes_team == "cloud9"
        assert lol["KXLOLGAME-26SEP2405009GSGW-9G"].yes_team == "9gaming"
        assert cs2["KXCS2GAME-26SEP251400K27GL-K27"].yes_team == "k27"
        assert cs2["KXCS2GAME-26SEP2508003DMAXNIP-3DMAX"].yes_team == "3dmax"

    @pytest.mark.asyncio
    async def test_game_date_is_the_ticker_et_day(self):
        markets = {m.ticker: m for m in await _parse("cs2", _markets("cs2"), _events("cs2"))}
        # 9:00 PM EDT Sep 25 = 01:00Z Sep 26; Kalshi dates it Sep 25.
        assert markets["KXCS2GAME-26SEP252100OTLAG-OT"].event_date.date().isoformat() == "2026-09-25"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("sector", ["lol", "cs2"])
    async def test_events_failure_still_recovers_teams_from_siblings(self, sector):
        markets = await _parse(sector, _markets(sector), events_exc=RuntimeError("simulated 400"))
        raw = {r["ticker"]: r for r in _markets(sector)["markets"]}
        n = NameNormalizer(sector)
        assert len(markets) == len(raw)
        for m in markets:
            away, home = EXPECTED[sector][raw[m.ticker]["event_ticker"]]
            assert (n.normalize(m.team_away), n.normalize(m.team_home)) == (away, home), m.ticker
            assert m.competition is None  # the events join is what's lost

    @pytest.mark.asyncio
    @pytest.mark.parametrize("sector", ["lol", "cs2"])
    async def test_event_title_fallback_without_yes_sub_title(self, sector):
        """No yes_sub_title anywhere: teams from the event title, the YES
        team from the short "{Team} wins" title."""
        mk = copy.deepcopy(_markets(sector))
        for r in mk["markets"]:
            r.pop("yes_sub_title", None)
        markets = await _parse(sector, mk, _events(sector))
        raw = {r["ticker"]: r for r in mk["markets"]}
        n = NameNormalizer(sector)
        assert len(markets) == len(raw)
        for m in markets:
            r = raw[m.ticker]
            away, home = EXPECTED[sector][r["event_ticker"]]
            assert (n.normalize(m.team_away), n.normalize(m.team_home)) == (away, home), m.ticker
            assert m.yes_team == n.normalize(r["title"][: -len(" wins")]), m.ticker

    @pytest.mark.asyncio
    async def test_no_names_at_all_falls_back_to_ticker_codes(self):
        """No yes_sub_title, no events, short title: only the ticker is left.
        The old code path still runs (never crashes, never invents a name)."""
        mk = copy.deepcopy(_markets("lol"))
        for r in mk["markets"]:
            r.pop("yes_sub_title", None)
        markets = {m.ticker: m for m in await _parse("lol", mk)}
        fur = markets["KXLOLGAME-26SEP261200FURRED-FUR"]
        assert (fur.team_away, fur.team_home) == ("fur", "red")
        assert fur.yes_team == "furia"  # the short title still names the YES team


class TestEsportsLegacyTitleFallback:
    """Long-form titles ("Will X win the A vs. B <game> match?", archived
    2026-08-17) must keep parsing for cached rows or a Kalshi revert."""

    @pytest.mark.parametrize("sector,ticker,title,away,home,yes", [
        ("lol", "KXLOLGAME-26AUG170400KTCDKC-DKC",
         "Will Dplus KIA Challengers win the KT Rolster Challengers vs. Dplus KIA Challengers League of Legends match?",
         "KT Rolster Challengers", "Dplus KIA Challengers", "dplus kia challengers"),
        ("lol", "KXLOLGAME-26AUG170500T1DNF-T1",
         "Will T1 win the T1 vs. DN Freecs League of Legends match?", "T1", "DN Freecs", "t1"),
        ("cs2", "KXCS2GAME-26AUG170400PHACBEN-BEN",
         "Will Benched gods win the Phantom Academy vs. Benched gods CS2 match?",
         "Phantom Academy", "Benched gods", "benched gods"),
        ("cs2", "KXCS2GAME-26AUG170400UNITYPERM-UNITY",
         "Will UNiTY esports win the UNiTY esports vs. Permitta Esports CS2 match?",
         "UNiTY esports", "Permitta Esports", "unity"),
    ])
    def test_legacy_title(self, sector, ticker, title, away, home, yes):
        raw = {"ticker": ticker, "event_ticker": ticker.rsplit("-", 1)[0], "title": title,
               "yes_ask_dollars": "0.55", "no_ask_dollars": "0.47",
               "yes_bid_dollars": "0.53", "no_bid_dollars": "0.45"}
        m = KalshiClient()._parse_market(raw, sector)
        assert m is not None
        assert (m.team_away, m.team_home) == (away, home)
        assert m.yes_team == yes
        assert m.market_type == MarketType.moneyline


class TestEsportsWinnerOnlyMoneyline:
    """Real archived Kalshi esports rows (ticker + title) that the keyword
    inference turned into totals / spreads / map handicaps — the archived
    market_type is in the comment. ~10k archived lol/cs2 snapshot rows
    carry such types."""

    @pytest.mark.parametrize("sector,ticker,title", [
        ("cs2", "KXCS2GAME-26AUG252100OTVIL-OT", "Overtake wins"),                # total
        ("cs2", "KXCS2GAME-26AUG201900UNKUND-UND", "underw0rld wins"),            # total
        ("lol", "KXLOLGAME-26AUG230300LGDTT-TT", "ThunderTalk Gaming wins"),      # total
        ("lol", "KXLOLGAME-26AUG241900VKSAITZ-ITZ", "INTZ e-Sports wins"),        # spread
        ("cs2", "KXCS2GAME-26AUG231200PURERGG-RGG", "RoundsGG wins"),             # map_handicap
        ("lol", "KXLOLGAME-26AUG080900THGAO-AO",                                  # map_handicap
         "Will Ancient Ones win the Team High Ground vs. Ancient Ones League of Legends match?"),
    ])
    def test_titles_never_change_market_type(self, sector, ticker, title):
        raw = {"ticker": ticker, "title": title, "yes_ask_dollars": "0.45",
               "no_ask_dollars": "0.57", "yes_bid_dollars": "0.43",
               "no_bid_dollars": "0.55", "event_ticker": ticker.rsplit("-", 1)[0]}
        m = KalshiClient()._parse_market(raw, sector)
        assert m is not None
        assert m.market_type == MarketType.moneyline
        assert m.line is None


# ---------------------------------------------------------------------------
# End to end: production MatchingEngine + YES alignment vs live Pinnacle
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("no_cache")
class TestEsportsMatchingEndToEnd:
    @pytest.mark.parametrize("sector", ["lol", "cs2"])
    def test_fixture_event_ids_are_what_the_pinnacle_client_builds_today(self, sector):
        for so in _sharps(sector):
            date = so.event_id.split("::")[1]
            rebuilt = (f"{sector}::{date}::{PinnacleGuestClient._normalize(so.outcome_a_label, sector)}"
                       f"_vs_{PinnacleGuestClient._normalize(so.outcome_b_label, sector)}")
            assert rebuilt == so.event_id

    @pytest.mark.asyncio
    @pytest.mark.parametrize("sector", ["lol", "cs2"])
    async def test_matches_live_pinnacle_and_aligns_yes(self, sector):
        """Before the fix 0 of these markets matched (live capture:
        lol 0/46, cs2 0/180). Now every fixture event Pinnacle lists matches,
        both YES markets of it, and each YES contract prices its own team by
        canonical equality — never the esports price fallback."""
        markets = await _parse(sector, _markets(sector), _events(sector))
        engine = MatchingEngine()
        pairs = engine.match_all(markets, _sharps(sector))
        matched_events = {m.ticker.rsplit("-", 1)[0] for m, _, _ in pairs}
        assert matched_events == PINNACLE_LISTED[sector]
        assert len(pairs) == 2 * len(PINNACLE_LISTED[sector])
        norm = engine.normalizer_for(sector)
        sides = collections.Counter()
        for market, sharp, conf in pairs:
            assert conf == 100.0, market.ticker
            a = align_yes_side(market, sharp, norm)
            assert a is not None, market.ticker
            assert a.method == "canonical", market.ticker
            label = sharp.outcome_a_label if a.outcome is YesOutcome.A else sharp.outcome_b_label
            assert norm.normalize(label) == market.yes_team, market.ticker
            assert not alignment_looks_suspect(a, market.yes_price, sharp), market.ticker
            sides[(sharp.event_id, a.outcome)] += 1
        # The two YES markets of a match price the two DIFFERENT sides.
        assert all(v == 1 for v in sides.values())

    @pytest.mark.asyncio
    async def test_no_false_match_for_events_pinnacle_does_not_list(self):
        markets = await _parse("cs2", _markets("cs2"), _events("cs2"))
        engine = MatchingEngine()
        unlisted = [m for m in markets if m.ticker.rsplit("-", 1)[0] not in PINNACLE_LISTED["cs2"]]
        assert unlisted
        assert engine.match_all(unlisted, _sharps("cs2")) == []
