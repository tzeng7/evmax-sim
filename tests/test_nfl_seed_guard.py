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


class TestSuccessRateOpponentAdjustment:
    """compute_team_stats opponent-adjusts success rate the same way as EPA
    (mirrors NCAAF v2's opponent-adjusted SR). 3-team schedule so the SoS
    subtraction is non-degenerate (a 2-team pair cancels to 0)."""

    def _df(self):
        # (posteam, defteam, success) per play; 3 games among KC/BUF/DEN.
        plays = (
            [("KC", "BUF", 1)] * 4 + [("BUF", "KC", 0)] * 2               # g1 KC@BUF
            + [("KC", "DEN", 1), ("KC", "DEN", 0), ("DEN", "KC", 1), ("DEN", "KC", 1)]  # g2
            + [("BUF", "DEN", 1), ("BUF", "DEN", 1), ("DEN", "BUF", 0), ("DEN", "BUF", 0)]  # g3
        )
        gids = ["g1"] * 6 + ["g2"] * 4 + ["g3"] * 4
        return pl.DataFrame({
            "season": [2025] * len(plays),
            "posteam": [p[0] for p in plays],
            "defteam": [p[1] for p in plays],
            "success": [float(p[2]) for p in plays],
            "epa": [0.0] * len(plays),
            "yards_gained": [4.0] * len(plays),
            "yardline_100": [50] * len(plays),
            "touchdown": [0.0] * len(plays),
            "game_id": gids,
        })

    def test_off_success_is_opponent_adjusted(self):
        stats = seed.compute_team_stats(self._df(), current_season=2025)
        kc = stats["kansas city chiefs"]
        assert "off_success_adj" in kc and "def_success_adj" in kc
        # raw_off_success[KC]=5/6; avg_opp_def_success weighted by KC's plays
        # (4 vs BUF def=4/6, 2 vs DEN def=3/4) = 0.6944 → adj ≈ +0.139
        assert kc["off_success_adj"] == pytest.approx(0.1389, abs=0.005)
        # raw SR is still kept for diagnostics and is NOT the adjusted value
        assert kc["off_success_rate"] == pytest.approx(0.8333, abs=0.005)
