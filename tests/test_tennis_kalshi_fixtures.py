"""Replay tests against captured Kalshi tennis fixtures.

Exercises three layers of the MODEL-1 change without any network:

1. Resolver replay — feeds each captured event's
   ``product_metadata.competition`` into ``_resolve_surface`` and asserts
   the expected ``(surface, is_indoor)`` tuple.
2. End-to-end Kalshi join — monkey-patches ``KalshiClient._get`` to
   return the captured JSON for ``/markets`` and ``/events``, then invokes
   ``KalshiClient.get_markets(sector='tennis')`` and asserts each returned
   ``PredictionMarket`` has ``competition`` populated via event_ticker join.
3. Indoor-flag verification — since ``is_indoor`` is a seam for MODEL-6 and
   not yet consumed by ``predict_pair``, explicit checks prevent silent drift.

Fixtures live at ``tests/fixtures/kalshi/`` and were captured live on
2026-04-13. See that directory's README for provenance and refresh steps.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from evmax.agents.models.tennis_model_agent import TennisModelAgent
from evmax.clients.kalshi import KalshiClient
from evmax.models.market import PredictionMarket

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "kalshi"


def _load(name: str) -> dict:
    with open(FIXTURE_DIR / name) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Layer 1: resolver replay on real captured event metadata
# ---------------------------------------------------------------------------

class TestResolverReplay:
    """Each captured event's competition string resolves to a known surface."""

    def test_atp_events_resolve_to_clay(self):
        """All 5 captured ATP events are Munich / Barcelona (clay)."""
        data = _load("atp_events.json")
        events = data.get("events", [])
        assert len(events) > 0, "fixture has no events"

        for ev in events:
            comp = ev["product_metadata"]["competition"]
            surface, indoor = TennisModelAgent._resolve_surface(
                competition=comp, title=ev.get("title", "")
            )
            # Live snapshot (2026-04-13): Munich + Barcelona, both clay
            assert surface == "clay", f"{comp} → {surface}, expected clay"
            assert indoor is False

    def test_wta_events_resolve_to_clay(self):
        """All 5 captured WTA events are Rouen / Stuttgart (clay)."""
        data = _load("wta_events.json")
        events = data.get("events", [])
        assert len(events) > 0

        for ev in events:
            comp = ev["product_metadata"]["competition"]
            surface, indoor = TennisModelAgent._resolve_surface(
                competition=comp, title=ev.get("title", "")
            )
            assert surface == "clay", f"{comp} → {surface}, expected clay"
            assert indoor is False

    def test_every_captured_event_has_competition_field(self):
        """The whole MODEL-1 fix depends on this field being present.
        If Kalshi ever ships an event without it, this test fails loudly."""
        for name in ("atp_events.json", "wta_events.json"):
            data = _load(name)
            for ev in data.get("events", []):
                pm = ev.get("product_metadata") or {}
                comp = pm.get("competition")
                assert comp, f"{name}: event {ev.get('event_ticker')} missing competition"


# ---------------------------------------------------------------------------
# Layer 2: end-to-end join through KalshiClient.get_markets
# ---------------------------------------------------------------------------

class TestKalshiEventsJoin:
    """Monkey-patch ``_get`` to return captured fixtures and verify the
    events fetch + event_ticker join actually populates
    ``PredictionMarket.competition`` end-to-end."""

    @pytest.mark.asyncio
    async def test_get_markets_attaches_competition_to_tennis_markets(
        self, monkeypatch
    ):
        atp_markets = _load("atp_markets.json")
        atp_events = _load("atp_events.json")
        wta_markets = _load("wta_markets.json")
        wta_events = _load("wta_events.json")

        # Route /markets and /events calls to the right fixtures based on
        # the series_ticker param.
        async def fake_get(path, params=None):
            series = (params or {}).get("series_ticker", "")
            if path == "/markets":
                if "ATP" in series:
                    return atp_markets
                if "WTA" in series:
                    return wta_markets
            elif path == "/events":
                if "ATP" in series:
                    return atp_events
                if "WTA" in series:
                    return wta_events
            return {}

        # Disable caching so the real parse path is exercised every call.
        from evmax import settings as settings_module
        settings = settings_module.get_settings()
        monkeypatch.setattr(settings, "offline_mode", False)
        monkeypatch.setattr(settings, "cache_ttl_secs", 0)
        monkeypatch.setattr(settings, "kalshi_ws_enabled", False)

        async with KalshiClient() as client:
            with patch.object(client, "_get", side_effect=fake_get):
                markets = await client.get_markets(sector="tennis")

        assert len(markets) > 0, "no markets returned from fixture replay"

        # Every tennis market should have competition populated via the join.
        # Known live competitions in the fixtures: ATP Munich, ATP Barcelona,
        # WTA Rouen, WTA Stuttgart.
        known_competitions = {
            "ATP Munich", "ATP Barcelona", "WTA Rouen", "WTA Stuttgart"
        }
        for m in markets:
            assert m.competition is not None, (
                f"market {m.ticker} missing competition after join"
            )
            assert m.competition in known_competitions, (
                f"unexpected competition {m.competition!r}"
            )
            # A tennis market with no YES player can never align to a Pinnacle
            # matchup — it is silently dropped from every gap. Guard it here so
            # the player-extraction path can't regress unnoticed (it did once,
            # when Kalshi changed the title format — see TestTennisPlayerExtraction).
            assert m.yes_team, f"market {m.ticker} missing yes_team after parse"
            assert m.team_home and m.team_away, (
                f"market {m.ticker} missing competitors after parse"
            )

    @pytest.mark.asyncio
    async def test_non_tennis_sector_gets_none_competition(self, monkeypatch):
        """Non-tennis sectors must not trigger the /events fetch and must
        return PredictionMarket objects with competition=None, preserving
        the pre-MODEL-1 behavior for NBA/NFL/etc."""
        # Minimal fake /markets response for NBA; shape matches the real API.
        fake_nba_markets = {
            "markets": [
                {
                    "ticker": "KXNBAGAME-26APR14LALGSW-GSW",
                    "event_ticker": "KXNBAGAME-26APR14LALGSW",
                    "title": "Will Warriors beat Lakers?",
                    "yes_bid_dollars": 0.55,
                    "yes_ask_dollars": 0.57,
                    "no_bid_dollars": 0.43,
                    "no_ask_dollars": 0.45,
                    "volume_fp": 10000,
                    "open_interest_fp": 5000,
                },
            ]
        }

        events_call_count = {"n": 0}

        async def fake_get(path, params=None):
            if path == "/events":
                events_call_count["n"] += 1
                return {"events": []}
            if path == "/markets":
                return fake_nba_markets
            return {}

        from evmax import settings as settings_module
        settings = settings_module.get_settings()
        monkeypatch.setattr(settings, "offline_mode", False)
        monkeypatch.setattr(settings, "cache_ttl_secs", 0)
        monkeypatch.setattr(settings, "kalshi_ws_enabled", False)

        async with KalshiClient() as client:
            with patch.object(client, "_get", side_effect=fake_get):
                markets = await client.get_markets(sector="nba")

        # /events must never have been called for NBA
        assert events_call_count["n"] == 0
        for m in markets:
            assert m.competition is None


# ---------------------------------------------------------------------------
# Layer 3: is_indoor flag explicit verification (seam for MODEL-6)
# ---------------------------------------------------------------------------

class TestIsIndoorSeam:
    """``is_indoor`` is computed on every resolve but not yet consumed by
    ``predict_pair``. These tests guard the seam so it cannot silently
    drift before MODEL-6 lands."""

    def test_captured_clay_events_are_not_indoor(self):
        """All live fixture events are outdoor clay."""
        for name in ("atp_events.json", "wta_events.json"):
            data = _load(name)
            for ev in data.get("events", []):
                comp = ev["product_metadata"]["competition"]
                _surface, indoor = TennisModelAgent._resolve_surface(competition=comp)
                assert indoor is False, f"{comp}: expected outdoor, got indoor"

    @pytest.mark.parametrize(
        "competition",
        [
            "ATP Paris Masters",
            "ATP Paris Bercy",
            "ATP Rotterdam",
            "ATP Basel",
            "ATP Vienna",
            "Nitto ATP Finals",
            "ATP Finals",
            "WTA Finals",
        ],
    )
    def test_synthetic_indoor_events_flag_correctly(self, competition):
        surface, indoor = TennisModelAgent._resolve_surface(competition=competition)
        assert surface == "hard"
        assert indoor is True


# ---------------------------------------------------------------------------
# Layer 4: player extraction across title formats + events robustness
# ---------------------------------------------------------------------------

def _price_fields() -> dict:
    """Minimal price block so ``_parse_market`` accepts a synthetic market."""
    return {
        "yes_bid_dollars": 0.40, "yes_ask_dollars": 0.42,
        "no_bid_dollars": 0.58, "no_ask_dollars": 0.60,
        "volume_fp": 1000, "open_interest_fp": 500,
    }


# The short "{Full Name} wins" title format Kalshi switched tennis markets to
# in 2026-08. Crucially it carries NO "A vs B" matchup — the opponent must come
# from the parent event's title, and the YES player from ``yes_sub_title``.
_NEW_FORMAT_MARKETS = {
    "markets": [
        {
            "ticker": "KXATPMATCH-26AUG30DIMPOP-POP",
            "event_ticker": "KXATPMATCH-26AUG30DIMPOP",
            "title": "Alexei Popyrin wins",
            "yes_sub_title": "Alexei Popyrin",
            **_price_fields(),
        },
        {
            "ticker": "KXATPMATCH-26AUG30DIMPOP-DIM",
            "event_ticker": "KXATPMATCH-26AUG30DIMPOP",
            "title": "Grigor Dimitrov wins",
            "yes_sub_title": "Grigor Dimitrov",
            **_price_fields(),
        },
    ]
}
_NEW_FORMAT_EVENTS = {
    "events": [
        {
            "event_ticker": "KXATPMATCH-26AUG30DIMPOP",
            "title": "Dimitrov vs Popyrin",
            "sub_title": "Dimitrov vs Popyrin (Aug 30)",
            "product_metadata": {
                "competition": "US Open Men Singles",
                "competition_scope": "Game",
            },
        }
    ]
}


class TestTennisPlayerExtraction:
    """Player extraction must survive the 2026-08 Kalshi tennis title change.

    Old titles were "Will {Name} win the {A} vs {B}: Round ..."; new titles are
    the short "{Name} wins", which carries no matchup. The parser recovers the
    YES player from ``yes_sub_title`` and both competitors from the parent event
    title. A market with no ``yes_team`` can never match Pinnacle, so a silent
    extraction failure zeroes out every Kalshi tennis gap (the original bug).
    """

    @pytest.fixture(autouse=True)
    def _no_cache(self, monkeypatch):
        from evmax import settings as settings_module
        settings = settings_module.get_settings()
        monkeypatch.setattr(settings, "offline_mode", False)
        monkeypatch.setattr(settings, "cache_ttl_secs", 0)
        monkeypatch.setattr(settings, "kalshi_ws_enabled", False)

    @pytest.mark.asyncio
    async def test_new_short_title_format_extracts_players(self):
        # get_markets sweeps both tennis series (KXATPMATCH + KXWTAMATCH);
        # serve the ATP fixture only for the ATP series, empty for WTA.
        async def fake_get(path, params=None):
            is_atp = "ATP" in (params or {}).get("series_ticker", "")
            if path == "/markets":
                return _NEW_FORMAT_MARKETS if is_atp else {"markets": []}
            if path == "/events":
                return _NEW_FORMAT_EVENTS if is_atp else {"events": []}
            return {}

        async with KalshiClient() as client:
            with patch.object(client, "_get", side_effect=fake_get):
                markets = await client.get_markets(sector="tennis")

        by_ticker = {m.ticker: m for m in markets}
        assert len(by_ticker) == 2

        pop = by_ticker["KXATPMATCH-26AUG30DIMPOP-POP"]
        dim = by_ticker["KXATPMATCH-26AUG30DIMPOP-DIM"]
        # YES player comes from yes_sub_title, normalized to the surname.
        assert pop.yes_team == "popyrin"
        assert dim.yes_team == "dimitrov"
        # Both competitors come from the parent event title "Dimitrov vs Popyrin".
        for m in (pop, dim):
            assert {m.team_home.lower(), m.team_away.lower()} == {"dimitrov", "popyrin"}
            assert m.competition == "US Open Men Singles"

    @pytest.mark.asyncio
    async def test_legacy_will_win_the_title_still_parses(self):
        """Old-format markets (still served from cache / other cards, and the
        UFC path shares the helper) must keep extracting players."""
        legacy_markets = {
            "markets": [
                {
                    "ticker": "KXATPMATCH-26APR14COBDED-COB",
                    "event_ticker": "KXATPMATCH-26APR14COBDED",
                    "title": "Will Flavio Cobolli win the Cobolli vs Dedura-Palomero: Round Of 32 match?",
                    "yes_sub_title": "Flavio Cobolli",
                    **_price_fields(),
                },
            ]
        }
        legacy_events = {
            "events": [
                {
                    "event_ticker": "KXATPMATCH-26APR14COBDED",
                    "title": "Cobolli vs Dedura-Palomero",
                    "product_metadata": {"competition": "ATP Munich"},
                }
            ]
        }

        async def fake_get(path, params=None):
            is_atp = "ATP" in (params or {}).get("series_ticker", "")
            if path == "/markets":
                return legacy_markets if is_atp else {"markets": []}
            return legacy_events if is_atp else {"events": []}

        async with KalshiClient() as client:
            with patch.object(client, "_get", side_effect=fake_get):
                markets = await client.get_markets(sector="tennis")

        assert len(markets) == 1
        assert markets[0].yes_team == "cobolli"
        assert {markets[0].team_home.lower(), markets[0].team_away.lower()} == {
            "cobolli", "dedura-palomero",
        }

    @pytest.mark.asyncio
    async def test_events_limit_capped_at_200(self):
        """Kalshi's /events endpoint rejects limit>200 (400). The events call
        must cap its limit independently of the markets limit."""
        seen = {"events_limits": []}

        async def fake_get(path, params=None):
            is_atp = "ATP" in (params or {}).get("series_ticker", "")
            if path == "/events":
                seen["events_limits"].append((params or {}).get("limit"))
                return _NEW_FORMAT_EVENTS if is_atp else {"events": []}
            return _NEW_FORMAT_MARKETS if is_atp else {"markets": []}

        async with KalshiClient() as client:
            with patch.object(client, "_get", side_effect=fake_get):
                markets = await client.get_markets(sector="tennis", limit=1000)

        assert seen["events_limits"]  # /events was called
        assert all(lim <= 200 for lim in seen["events_limits"])
        assert len(markets) == 2  # markets still returned at limit=1000

    @pytest.mark.asyncio
    async def test_events_failure_is_non_fatal(self):
        """An /events failure must not wipe the markets — it only degrades the
        competition/matchup join."""
        async def fake_get(path, params=None):
            is_atp = "ATP" in (params or {}).get("series_ticker", "")
            if path == "/events":
                raise RuntimeError("simulated 400 Bad Request")
            return _NEW_FORMAT_MARKETS if is_atp else {"markets": []}

        async with KalshiClient() as client:
            with patch.object(client, "_get", side_effect=fake_get):
                markets = await client.get_markets(sector="tennis")

        # Markets survive; YES player still comes from yes_sub_title (not events).
        assert len(markets) == 2
        assert {m.yes_team for m in markets} == {"popyrin", "dimitrov"}
