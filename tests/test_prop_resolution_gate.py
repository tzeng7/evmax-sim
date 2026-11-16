"""Resolve skips prop rows whose ``{sector}_props`` category is disabled."""

from __future__ import annotations

import sqlite3
from datetime import date

import pytest

from evmax.agents.cleanup import resolver


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE prop_observations (
            id INTEGER PRIMARY KEY, scan_date TEXT, sector TEXT, event_id TEXT,
            player_name TEXT, stat_type TEXT, line REAL, outcome INTEGER,
            actual_value REAL, market_id TEXT, venue TEXT)"""
    )
    today = date.today().isoformat()
    for i, sector in enumerate(("nfl", "baseball", "nba")):
        conn.execute(
            "INSERT INTO prop_observations VALUES (?,?,?,?,?,?,?,NULL,NULL,?,?)",
            (i, today, sector, f"{sector}::{today}::a_vs_b::prop::p::x::1.5",
             "P", "points", 1.5, f"m{i}", "kalshi"),
        )
    return conn


@pytest.mark.parametrize("sector", ["nfl", "baseball", "nba"])
def test_shipped_yaml_disables_prop_resolution(sector):
    assert resolver._props_resolution_enabled(sector) is False


def test_unknown_or_missing_sector_stays_enabled():
    assert resolver._props_resolution_enabled("ncaab") is True
    assert resolver._props_resolution_enabled(None) is True


def test_disabled_props_make_no_network_calls(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("resolver fetched while props are disabled")

    for name in ("_resolve_baseball_prop_observations", "_resolve_nfl_prop_observations"):
        monkeypatch.setattr(resolver, name, boom)
    monkeypatch.setattr(resolver.httpx, "Client", boom)
    assert resolver._resolve_prop_observations(_conn(), date.today()) == 0


def test_enabled_sector_still_routes(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        resolver, "_props_resolution_enabled", lambda sector: sector == "nfl"
    )
    monkeypatch.setattr(
        resolver, "_resolve_nfl_prop_observations",
        lambda conn, by_date: seen.setdefault("nfl", sum(map(len, by_date.values()))),
    )
    monkeypatch.setattr(
        resolver, "_resolve_baseball_prop_observations",
        lambda *a, **k: seen.setdefault("baseball", 1),
    )
    resolver._resolve_prop_observations(_conn(), date.today())
    assert seen == {"nfl": 1}
