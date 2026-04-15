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
