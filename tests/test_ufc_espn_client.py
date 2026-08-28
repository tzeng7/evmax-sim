"""ESPN MMA client parsing tests — committed fixtures, no network."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx

from evmax.clients.ufc_espn import (
    UFCESPNClient,
    _ESPN_HTTP_UA,
    normalize_method,
    parse_athlete,
    parse_scoreboard,
    parse_status_result,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "ufc"


def _load(name: str) -> dict:
    with open(FIXTURE_DIR / name) as f:
        return json.load(f)


class TestParseScoreboard:
    def test_parses_both_fights(self):
        fights = parse_scoreboard(_load("espn_scoreboard_trimmed.json"))
        assert len(fights) == 2
        assert all(f.event_name.startswith("UFC 300") for f in fights)
        assert all(f.date == "2024-04-13" for f in fights)

    def test_winner_side_follows_order_not_list_position(self):
        """Competitors arrive winner-first in the raw feed; a/b assignment
        must come from the `order` field so the pairing is stable."""
        fights = parse_scoreboard(_load("espn_scoreboard_trimmed.json"))
        fig = next(f for f in fights if "Figueiredo" in f.fighter_a + f.fighter_b)
        # order=1 → side a = Figueiredo (the winner)
        assert fig.fighter_a == "Deiveson Figueiredo"
        assert fig.fighter_b == "Cody Garbrandt"
        assert fig.winner == "a"
        assert fig.completed is True

    def test_finish_round_and_clock(self):
        fights = parse_scoreboard(_load("espn_scoreboard_trimmed.json"))
        fig = next(f for f in fights if f.fighter_a == "Deiveson Figueiredo")
        assert fig.round == 2
        assert fig.clock == "4:02"

    def test_athlete_ids_extracted(self):
        fights = parse_scoreboard(_load("espn_scoreboard_trimmed.json"))
        fig = next(f for f in fights if f.fighter_a == "Deiveson Figueiredo")
        assert fig.fighter_a_id == "4189320"
        assert fig.fighter_b_id == "3163637"

    def test_empty_payload(self):
        assert parse_scoreboard({}) == []
        assert parse_scoreboard({"events": []}) == []

    def test_single_competitor_bout_skipped(self):
        raw = {"events": [{"id": "1", "name": "X", "date": "2024-01-01T00:00Z",
                           "competitions": [{"id": "9", "competitors": [{"id": "5"}]}]}]}
        assert parse_scoreboard(raw) == []


class TestParseStatusResult:
    def test_submission_method(self):
        method, detail = parse_status_result(_load("espn_status.json"))
        assert method == "submission"
        assert detail == "Rear Naked Choke"

    def test_missing_result(self):
        assert parse_status_result({}) == (None, None)


class TestNormalizeMethod:
    def test_canonical_mapping(self):
        assert normalize_method("KO/TKO") == "ko_tko"
        assert normalize_method("Submission") == "submission"
        assert normalize_method("Unanimous Decision") == "decision"
        assert normalize_method("Split Decision") == "decision"
        assert normalize_method("No Contest") == "nc"
        assert normalize_method("Disqualification") == "dq"
        assert normalize_method("Majority Draw") == "draw"

    def test_unknown_and_empty(self):
        assert normalize_method(None) is None
        assert normalize_method("") is None
        assert normalize_method("something else") is None

    def test_no_contest_beats_ko_substring(self):
        # "no contest (accidental eye poke)" contains neither ko nor tko as
        # a decisive token, but ordering must keep nc ahead of the greedy
        # "ko" substring in words like "poke".
        assert normalize_method("No Contest (accidental clash)") == "nc"


class TestParseAthlete:
    def test_bio_fields(self):
        bio = parse_athlete(_load("espn_athlete_trimmed.json"))
        assert bio.athlete_id == "4189320"
        assert bio.name == "Deiveson Figueiredo"
        assert bio.dob == "1987-12-18"
        assert bio.height_in == 65.0
        assert bio.reach_in == 68.0
        assert bio.stance == "Orthodox"

    def test_missing_fields_are_none(self):
        bio = parse_athlete({"id": 1, "displayName": "X"})
        assert bio.dob is None
        assert bio.height_in is None
        assert bio.reach_in is None
        assert bio.stance is None


class TestEspnUserAgent:
    """ESPN's WAF 403s "evmax-*" and browser-impersonator UAs; only neutral
    tool UAs pass. Regression guard: the client once sent "evmax-ufc-seed/1.0",
    which 403'd every scoreboard request, so the weekly `--fetch` reseed aborted
    and ufc_rating_state.json froze (state stuck at 2026-08-01 for ~4 weeks →
    ufc_rating "missing" on ~half of each live card)."""

    def test_ua_constant_is_not_a_blocked_string(self):
        assert not _ESPN_HTTP_UA.lower().startswith("evmax")
        assert "mozilla" not in _ESPN_HTTP_UA.lower()  # not a browser string

    def test_client_sends_the_neutral_ua(self):
        async def _run():
            client = UFCESPNClient()
            try:
                return client._client.headers["user-agent"]
            finally:
                await client._client.aclose()

        assert asyncio.run(_run()) == _ESPN_HTTP_UA


class _NoRateLimit:
    """No-op async context manager: isolates the cache-fallback tests from the
    module-level AsyncLimiter (which warns when reused across asyncio.run loops)."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class TestRefreshCacheFallback:
    """A FAILED forced-refresh must degrade to last-known-good cache, never
    drop the data — otherwise a transient 403/timeout regresses seeded state."""

    def test_failed_forced_refresh_returns_cached(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "evmax.clients.ufc_espn._ESPN_RATE_LIMITER", _NoRateLimit(),
        )

        async def _run():
            client = UFCESPNClient(cache_dir=tmp_path)
            client._cache_set("scoreboard_202608", {"events": [{"id": "1"}]})

            async def _boom(*args, **kwargs):
                raise httpx.ConnectError("simulated 403/timeout")

            client._client.get = _boom  # type: ignore[assignment]
            try:
                return await client._get_json(
                    "http://x", cache_key="scoreboard_202608", refresh=True,
                )
            finally:
                await client._client.aclose()

        assert asyncio.run(_run()) == {"events": [{"id": "1"}]}

    def test_failed_fetch_with_no_cache_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "evmax.clients.ufc_espn._ESPN_RATE_LIMITER", _NoRateLimit(),
        )

        async def _run():
            client = UFCESPNClient(cache_dir=tmp_path)

            async def _boom(*args, **kwargs):
                raise httpx.ConnectError("simulated 403/timeout")

            client._client.get = _boom  # type: ignore[assignment]
            try:
                return await client._get_json(
                    "http://x", cache_key="never_cached", refresh=True,
                )
            finally:
                await client._client.aclose()

        assert asyncio.run(_run()) is None
