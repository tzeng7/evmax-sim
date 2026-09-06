"""Tests for the NFL seed-script correctness fixes (2026-09-06).

Two latent defects fixed:
  1. seed_nfl_efficiency.py wrote the REQUESTED season list into seasons_used
     (not the seasons actually loaded), so the staleness guard could unblank
     nfl_efficiency on data that did not contain the new season.
  2. Neither seed tolerated an unpublished season file — nflreadpy.load_pbp
     raises on a 404, so a reseed run before nflverse posts the current year's
     parquet would hard-fail instead of refreshing the older seasons.

Network-free: nflreadpy is monkeypatched. Also pins the staleness-guard
behavior the seasons_used fix protects.
"""

from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl
import pytest

_spec = importlib.util.spec_from_file_location(
    "seed_nfl_efficiency",
    Path(__file__).resolve().parents[1] / "scripts" / "seed_nfl_efficiency.py",
)
seed = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(seed)

from evmax.agents.models.nfl_efficiency_agent import nfl_state_is_stale_for_today


def _fake_pbp(season: int) -> pl.DataFrame:
    return pl.DataFrame({"season": [season, season], "epa": [0.1, -0.2]})


class TestLoadPbpTolerant:
    def test_skips_unpublished_season_and_concatenates_rest(self, monkeypatch):
        def fake_load(seasons):
            (s,) = seasons
            if s == 2026:
                raise ConnectionError("404 play_by_play_2026.parquet not found")
            return _fake_pbp(s)

        monkeypatch.setattr(seed.nfl, "load_pbp", fake_load)
        df = seed.load_pbp_tolerant([2024, 2025, 2026])
        assert not df.is_empty()
        # 2026 was skipped; only 2024 + 2025 present
        assert sorted(df["season"].unique().to_list()) == [2024, 2025]

    def test_all_unpublished_returns_empty(self, monkeypatch):
        def fake_load(seasons):
            raise ConnectionError("404")

        monkeypatch.setattr(seed.nfl, "load_pbp", fake_load)
        df = seed.load_pbp_tolerant([2026])
        assert df.is_empty()

    def test_seasons_used_derived_from_actual_data(self, monkeypatch):
        """The fix: seasons_used reflects loaded seasons, not the request. Even
        if 2026 is requested but only 2024/2025 load, seasons_used is [2024,2025],
        so the staleness guard stays engaged (no false unblank on stale data)."""
        def fake_load(seasons):
            (s,) = seasons
            if s == 2026:
                raise ConnectionError("404")
            return _fake_pbp(s)

        monkeypatch.setattr(seed.nfl, "load_pbp", fake_load)
        df = seed.load_pbp_tolerant([2024, 2025, 2026])
        seasons_used = sorted(int(s) for s in df["season"].unique().to_list())
        assert seasons_used == [2024, 2025]


class TestStalenessGuardHonorsSeasonsUsed:
    def test_guard_stays_stale_when_2026_absent_in_september(self):
        """seasons_used maxing at 2025 during the Sep 2026 window keeps the
        model blanked — the exact protection the seasons_used fix preserves."""
        state = {"nfl": {"seasons_used": [2020, 2021, 2022, 2023, 2024, 2025]}}
        assert nfl_state_is_stale_for_today(state, today=date(2026, 9, 20)) is True

    def test_guard_releases_once_2026_present(self):
        state = {"nfl": {"seasons_used": [2021, 2022, 2023, 2024, 2025, 2026]}}
        assert nfl_state_is_stale_for_today(state, today=date(2026, 9, 20)) is False
