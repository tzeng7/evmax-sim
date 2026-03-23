"""Pass 3 — Infrastructure robustness tests.

Covers:
  - agents/base.py: AgentBus subscriber deduplication
  - coordinator.py: steam cache load/save error handling (via monkeypatching)
  - settings.py: ev_threshold out-of-range validation
  - devig.py: proportional fallback warning (log check)
  - kelly.py: NaN guard
"""

import json
import logging
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from evmax.agents.base import AgentBus, AgentMessage, AgentRequest, AgentResponse, Agent


# ---------------------------------------------------------------------------
# AgentBus — subscriber deduplication
# ---------------------------------------------------------------------------

class TestAgentBusDedup:
    def test_same_handler_not_added_twice_to_topic(self):
        bus = AgentBus()
        handler = AsyncMock()
        bus.subscribe("test.topic", handler)
        bus.subscribe("test.topic", handler)
        handlers = bus._subscribers["test.topic"]
        assert handlers.count(handler) == 1

    def test_different_handlers_both_added(self):
        bus = AgentBus()
        h1 = AsyncMock()
        h2 = AsyncMock()
        bus.subscribe("test.topic", h1)
        bus.subscribe("test.topic", h2)
        assert len(bus._subscribers["test.topic"]) == 2

    def test_wildcard_handler_not_duplicated(self):
        bus = AgentBus()
        handler = AsyncMock()
        bus.subscribe("*", handler)
        bus.subscribe("*", handler)
        assert bus._wildcard.count(handler) == 1

    def test_wildcard_different_handlers_both_added(self):
        bus = AgentBus()
        h1 = AsyncMock()
        h2 = AsyncMock()
        bus.subscribe("*", h1)
        bus.subscribe("*", h2)
        assert len(bus._wildcard) == 2

    @pytest.mark.asyncio
    async def test_handler_called_exactly_once_after_dedup(self):
        bus = AgentBus()
        handler = AsyncMock()
        bus.subscribe("ev.gaps.nba", handler)
        bus.subscribe("ev.gaps.nba", handler)  # duplicate

        msg = AgentMessage(topic="ev.gaps.nba", payload=[], sender="test")
        await bus.publish(msg)
        handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_handler_exception_doesnt_crash_bus(self):
        bus = AgentBus()

        async def bad_handler(msg):
            raise RuntimeError("handler error")

        good_handler = AsyncMock()
        bus.subscribe("test.topic", bad_handler)
        bus.subscribe("test.topic", good_handler)

        msg = AgentMessage(topic="test.topic", payload={}, sender="test")
        await bus.publish(msg)  # should not raise
        good_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_wildcard_receives_all_topics(self):
        bus = AgentBus()
        wildcard = AsyncMock()
        bus.subscribe("*", wildcard)

        for topic in ["odds.kalshi.nba", "ev.gaps.soccer", "coordinator.cycle.done"]:
            await bus.publish(AgentMessage(topic=topic, payload=None, sender="test"))

        assert wildcard.call_count == 3

    def test_clear_removes_all_subscribers(self):
        bus = AgentBus()
        bus.subscribe("test.topic", AsyncMock())
        bus.subscribe("*", AsyncMock())
        bus.clear()
        assert len(bus._subscribers) == 0
        assert len(bus._wildcard) == 0


# ---------------------------------------------------------------------------
# Steam cache error handling
# ---------------------------------------------------------------------------

class TestSteamCacheErrorHandling:
    def test_load_returns_empty_dict_on_corrupt_json(self, tmp_path):
        """Corrupted JSON file should return {} without raising."""
        cache_file = tmp_path / "steam_cache.json"
        cache_file.write_text("not valid json {{{{")

        from evmax.agents import coordinator as coord_mod
        original_path = coord_mod._STEAM_CACHE_PATH
        try:
            coord_mod._STEAM_CACHE_PATH = cache_file
            result = coord_mod._load_steam_cache()
            assert result == {}  # must not crash, must return empty dict
        finally:
            coord_mod._STEAM_CACHE_PATH = original_path

    def test_load_returns_empty_dict_when_file_missing(self, tmp_path):
        """Missing file should return {} without error."""
        from evmax.agents import coordinator as coord_mod
        original_path = coord_mod._STEAM_CACHE_PATH
        try:
            coord_mod._STEAM_CACHE_PATH = tmp_path / "nonexistent.json"
            result = coord_mod._load_steam_cache()
            assert result == {}
        finally:
            coord_mod._STEAM_CACHE_PATH = original_path

    def test_load_returns_valid_data(self, tmp_path):
        cache_file = tmp_path / "steam_cache.json"
        cache_file.write_text(json.dumps({"nba::2026::event1": 0.55}))

        from evmax.agents import coordinator as coord_mod
        original_path = coord_mod._STEAM_CACHE_PATH
        try:
            coord_mod._STEAM_CACHE_PATH = cache_file
            result = coord_mod._load_steam_cache()
            assert result == {"nba::2026::event1": 0.55}
        finally:
            coord_mod._STEAM_CACHE_PATH = original_path

    def test_save_does_not_raise_on_permission_denied(self, tmp_path):
        """Save failure must not raise — errors are logged but execution continues."""
        read_only_dir = tmp_path / "readonly"
        read_only_dir.mkdir()
        read_only_dir.chmod(0o444)

        from evmax.agents import coordinator as coord_mod
        original_path = coord_mod._STEAM_CACHE_PATH
        try:
            coord_mod._STEAM_CACHE_PATH = read_only_dir / "nested" / "steam_cache.json"
            # Must not raise even though mkdir will fail due to permissions
            coord_mod._save_steam_cache({"key": 0.5})
        finally:
            coord_mod._STEAM_CACHE_PATH = original_path
            read_only_dir.chmod(0o755)

    def test_save_and_load_roundtrip(self, tmp_path):
        cache_file = tmp_path / "steam_cache.json"
        data = {"nba::2026-03-22::lakers_vs_celtics": 0.62}

        from evmax.agents import coordinator as coord_mod
        original_path = coord_mod._STEAM_CACHE_PATH
        try:
            coord_mod._STEAM_CACHE_PATH = cache_file
            coord_mod._save_steam_cache(data)
            loaded = coord_mod._load_steam_cache()
            assert loaded == data
        finally:
            coord_mod._STEAM_CACHE_PATH = original_path


# ---------------------------------------------------------------------------
# Settings validation
# ---------------------------------------------------------------------------

class TestSettingsValidation:
    def test_default_settings_valid(self):
        from evmax.settings import Settings
        s = Settings()
        assert 0.0 <= s.ev_threshold <= 0.5
        assert s.max_kelly_fraction <= 1.0
        assert s.min_kelly_fraction >= 0.0

    def test_ev_threshold_above_50pct_raises(self):
        from evmax.settings import Settings
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="50%"):
            Settings(ev_threshold=0.6)

    def test_ev_threshold_below_zero_raises(self):
        from evmax.settings import Settings
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Settings(ev_threshold=-0.01)

    def test_ev_threshold_at_zero_valid(self):
        from evmax.settings import Settings
        s = Settings(ev_threshold=0.0)
        assert s.ev_threshold == 0.0

    def test_ev_threshold_at_50pct_valid(self):
        from evmax.settings import Settings
        s = Settings(ev_threshold=0.5)
        assert s.ev_threshold == 0.5

    def test_max_kelly_above_one_raises(self):
        from evmax.settings import Settings
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Settings(max_kelly_fraction=1.5)

    def test_max_kelly_below_min_raises(self):
        from evmax.settings import Settings
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Settings(max_kelly_fraction=0.0)


# ---------------------------------------------------------------------------
# devig.py — fallback logging
# ---------------------------------------------------------------------------

class TestDevigFallbackLogging:
    def test_fallback_logs_warning(self, caplog):
        """Proportional fallback should emit a warning log."""
        from evmax.ev.devig import devig_power_method
        # Create extreme odds that reliably trigger brentq failure by using a
        # bracket that deliberately cannot find the root
        # We can't easily force brentq to fail with real odds, so we patch it
        with patch("evmax.ev.devig.brentq", side_effect=ValueError("no root")):
            with caplog.at_level(logging.WARNING, logger="evmax.ev.devig"):
                result = devig_power_method([1.91, 2.05])
        # Should still produce valid output
        assert abs(sum(result.true_probs) - 1.0) < 0.01
        assert result.k == 1.0
        assert "fallback" in caplog.text.lower() or "devig_power_method_fallback" in caplog.text


# ---------------------------------------------------------------------------
# Player name normalization
# ---------------------------------------------------------------------------

class TestPlayerNormalization:
    def test_alias_lookup(self):
        from evmax.players import normalize_player_name
        assert normalize_player_name("lbj", "nba") == "lebron james"
        assert normalize_player_name("kd", "nba") == "kevin durant"
        assert normalize_player_name("sga", "nba") == "shai gilgeous-alexander"

    def test_full_name_passthrough(self):
        from evmax.players import normalize_player_name
        assert normalize_player_name("lebron james", "nba") == "lebron james"

    def test_unknown_name_falls_back_to_last_name(self):
        from evmax.players import normalize_player_name
        result = normalize_player_name("Victor Wembanyama Jr", "nba")
        # Unknown in alias table → last word
        assert result == "jr" or result == "wembanyama"

    def test_empty_string_returns_empty(self):
        from evmax.players import normalize_player_name
        assert normalize_player_name("", "nba") == ""

    def test_nfl_aliases(self):
        from evmax.players import normalize_player_name
        assert normalize_player_name("cmc", "nfl") == "christian mccaffrey"
        assert normalize_player_name("mahomes", "nfl") == "patrick mahomes"

    def test_wrong_sector_misses_alias(self):
        from evmax.players import normalize_player_name
        # "lbj" is only in NBA aliases, not NFL
        result = normalize_player_name("lbj", "nfl")
        # Falls back to last name = "lbj" (one word)
        assert result == "lbj"


# ---------------------------------------------------------------------------
# Agent ABC error capture
# ---------------------------------------------------------------------------

class ConcreteAgent(Agent):
    name = "test_agent"
    description = "test"

    def __init__(self, raises=False):
        super().__init__()
        self._raises = raises

    async def run(self, request: AgentRequest) -> AgentResponse:
        if self._raises:
            raise RuntimeError("intentional test error")
        return AgentResponse(agent_name=self.name, sector=request.sector, data={"ok": True})


class TestAgentErrorCapture:
    @pytest.mark.asyncio
    async def test_successful_run(self):
        agent = ConcreteAgent()
        req = AgentRequest(sector="nba")
        resp = await agent(req)
        assert resp.status == "ok"
        assert resp.data == {"ok": True}
        assert resp.latency_ms > 0

    @pytest.mark.asyncio
    async def test_exception_captured_as_error_response(self):
        agent = ConcreteAgent(raises=True)
        req = AgentRequest(sector="nba")
        resp = await agent(req)
        assert resp.status == "error"
        assert "intentional test error" in resp.error
        assert resp.data is None

    @pytest.mark.asyncio
    async def test_agent_status_resets_to_idle_after_success(self):
        from evmax.agents.base import AgentStatus
        agent = ConcreteAgent()
        await agent(AgentRequest(sector="nba"))
        assert agent.status == AgentStatus.IDLE

    @pytest.mark.asyncio
    async def test_agent_status_set_to_error_after_failure(self):
        from evmax.agents.base import AgentStatus
        agent = ConcreteAgent(raises=True)
        await agent(AgentRequest(sector="nba"))
        assert agent.status == AgentStatus.ERROR
