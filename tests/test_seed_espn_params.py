"""seed_espn must request the same ESPN games the daily model feed applies.

Regression for 2026-11-02: the seeder fetched NCAAB/NCAAW without the
resolver's ``groups=50`` filter and with ``limit=100``. ESPN then returns only
its featured games (17 of 132 NCAAB games on 2026-02-14) and ignores ``page``,
so college Elo held ~150 of ~360 D1 teams and was never fully seeded.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _load_seed_espn():
    spec = importlib.util.spec_from_file_location("seed_espn", _ROOT / "scripts" / "seed_espn.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["seed_espn"] = mod
    spec.loader.exec_module(mod)
    return mod


seed = _load_seed_espn()


def test_college_sectors_use_the_resolver_groups_filter():
    from evmax.agents.cleanup.resolver import ESPN_SPORT_MAP

    assert seed.espn_scoreboard_params("ncaab") == {"groups": "50"}
    assert seed.espn_scoreboard_params("ncaaw") == {"groups": "50"}
    for sector, (_, _, extra) in ESPN_SPORT_MAP.items():
        assert seed.espn_scoreboard_params(sector) == extra


def test_unfiltered_sectors_and_unknown_sectors_get_no_params():
    assert seed.espn_scoreboard_params("nba") == {}
    assert seed.espn_scoreboard_params("not_a_sector") == {}


class _FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"events": []}


class _FakeClient:
    def __init__(self):
        self.calls: list[dict] = []

    async def get(self, url, params=None):
        self.calls.append(dict(params or {}))
        return _FakeResponse()


def test_fetch_passes_groups_and_an_uncapped_limit():
    client = _FakeClient()
    asyncio.run(seed.fetch_espn_games(
        client, "basketball", "mens-college-basketball", "20260214",
        extra_params={"groups": "50"},
    ))
    params = client.calls[0]
    assert params["groups"] == "50"
    assert params["dates"] == "20260214"
    assert params["limit"] >= 400  # a 182-game college day must fit in one response


def test_fetch_without_extra_params_is_unfiltered():
    client = _FakeClient()
    asyncio.run(seed.fetch_espn_games(client, "basketball", "nba", "202601"))
    assert "groups" not in client.calls[0]
