"""Tests for the NBA model state freshness helper."""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from evmax.agents.models import _nba_freshness as freshness


@pytest.fixture(autouse=True)
def _reset_cache():
    freshness._reset_cache_for_tests()
    yield
    freshness._reset_cache_for_tests()


def _scoreboard_response(completed_dates: set[str]):
    """Return a fake AsyncClient context manager keyed on the requested date."""

    async def _get(url, params=None):
        d = params["dates"]  # YYYYMMDD
        iso = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if iso in completed_dates:
            resp.json = MagicMock(return_value={"events": [{"status": {"type": {"completed": True}}}]})
        else:
            resp.json = MagicMock(return_value={"events": []})
        return resp

    inner = MagicMock()
    inner.get = _get

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=inner)
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm, inner


def test_last_completed_returns_yesterday_when_games_played():
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    cm, _ = _scoreboard_response({yesterday})
    with patch("httpx.AsyncClient", return_value=cm):
        result = asyncio.run(freshness.last_completed_nba_game_date())
    assert result == yesterday


def test_last_completed_returns_none_when_no_games_in_window():
    cm, _ = _scoreboard_response(set())
    with patch("httpx.AsyncClient", return_value=cm):
        result = asyncio.run(freshness.last_completed_nba_game_date())
    assert result is None


def test_state_is_fresh_after_last_game():
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    cm, _ = _scoreboard_response({yesterday})
    with patch("httpx.AsyncClient", return_value=cm):
        # State stamped today is newer than yesterday's last game → fresh
        assert asyncio.run(freshness.state_is_fresh(date.today().isoformat())) is True
        # State stamped exactly when last game finished → fresh (boundary)
        freshness._reset_cache_for_tests()
        assert asyncio.run(freshness.state_is_fresh(yesterday)) is True


def test_state_is_stale_when_games_played_after_fetch():
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    two_days_ago = (date.today() - timedelta(days=2)).isoformat()
    cm, _ = _scoreboard_response({yesterday})
    with patch("httpx.AsyncClient", return_value=cm):
        # State stamped before yesterday's game → stale, must refetch
        assert asyncio.run(freshness.state_is_fresh(two_days_ago)) is False


def test_empty_fetched_at_is_stale():
    cm, _ = _scoreboard_response(set())
    with patch("httpx.AsyncClient", return_value=cm):
        assert asyncio.run(freshness.state_is_fresh("")) is False


def test_no_games_in_window_treats_state_as_fresh():
    """If ESPN has no completed games in 7d (offseason / break), don't refetch."""
    cm, _ = _scoreboard_response(set())
    with patch("httpx.AsyncClient", return_value=cm):
        old_date = (date.today() - timedelta(days=30)).isoformat()
        assert asyncio.run(freshness.state_is_fresh(old_date)) is True


def test_circuit_breaker_starts_closed():
    """Fresh process — no failures yet, breaker is closed."""
    assert freshness.nba_api_in_backoff() is False


def test_circuit_breaker_opens_on_failure():
    freshness.mark_nba_api_failure()
    assert freshness.nba_api_in_backoff() is True


def test_circuit_breaker_resets_via_test_hook():
    freshness.mark_nba_api_failure()
    assert freshness.nba_api_in_backoff() is True
    freshness._reset_cache_for_tests()
    assert freshness.nba_api_in_backoff() is False


def test_circuit_breaker_persists_across_processes(tmp_path, monkeypatch):
    """A second process must see the breaker without re-triggering the failure."""
    breaker_file = tmp_path / "breaker.json"
    monkeypatch.setattr(freshness, "_BREAKER_PATH", breaker_file)

    freshness.mark_nba_api_failure()
    assert breaker_file.exists()

    # Simulate a fresh process: clear in-memory state but keep the file
    freshness._cache_value = None
    freshness._cache_time = 0.0
    freshness._last_failure_time = 0.0

    assert freshness.nba_api_in_backoff() is True
    assert freshness._last_failure_time > 0  # was lifted from disk


def test_cache_avoids_repeat_calls_within_ttl():
    today = date.today().isoformat()
    cm, inner = _scoreboard_response({today})
    call_count = {"n": 0}
    real_get = inner.get

    async def counting_get(url, params=None):
        call_count["n"] += 1
        return await real_get(url, params=params)

    inner.get = counting_get
    with patch("httpx.AsyncClient", return_value=cm):
        asyncio.run(freshness.last_completed_nba_game_date())
        asyncio.run(freshness.last_completed_nba_game_date())
        asyncio.run(freshness.last_completed_nba_game_date())
    # Only one network probe even though we asked three times
    assert call_count["n"] == 1
