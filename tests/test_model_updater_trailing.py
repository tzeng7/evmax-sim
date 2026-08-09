"""Trailing-window model-feed backfill tests.

The resolve hook feeds only the single date it is given, so any day the resolve
task does not run leaves a PERMANENT hole in Elo/Form state (games are fed once,
in date order — an out-of-order backfill later would corrupt the path-dependent
ratings). `update_models_trailing_window` feeds a short trailing window on every
run so a transient multi-day outage self-heals on the next successful run,
staying idempotent via the `applied_model_games` ledger.
"""

from __future__ import annotations

import asyncio
from datetime import date
from unittest.mock import patch

from evmax.agents.cleanup import model_updater


def _game(sector: str = "nba") -> model_updater.GameUpdate:
    return model_updater.GameUpdate(
        sector=sector, home_name="H", away_name="A",
        team_a="h", team_b="a", score_a=100, score_b=90, applied=True,
    )


def _run(**kwargs):
    """Run the trailing window with update_models_for_date stubbed; return
    (recorded_dates, aggregate_result)."""
    calls: list[date] = []
    seen: dict = {}

    async def fake_update(sectors, d, coordinator=None, *,
                          dry_run=False, force=False, espn_cache=None):
        calls.append(d)
        seen.update(sectors=sectors, coordinator=coordinator,
                    dry_run=dry_run, force=force, espn_cache=espn_cache)
        return model_updater.UpdateResult(games=[_game()], updated=1, skipped=2)

    with patch.object(model_updater, "update_models_for_date", side_effect=fake_update):
        res = asyncio.run(model_updater.update_models_trailing_window(**kwargs))
    return calls, res, seen


class TestTrailingWindow:
    def test_feeds_each_date_oldest_first(self):
        calls, res, _ = _run(sectors=["nba"], end_date=date(2026, 8, 8), lookback_days=3)
        # Chronological order matters — Elo is path-dependent.
        assert calls == [date(2026, 8, 6), date(2026, 8, 7), date(2026, 8, 8)]

    def test_aggregates_updated_skipped_and_games(self):
        _, res, _ = _run(sectors=["nba"], end_date=date(2026, 8, 8), lookback_days=3)
        assert res.updated == 3        # 1 per date × 3 dates
        assert res.skipped == 6        # 2 per date × 3 dates
        assert len(res.games) == 3     # games concatenated across dates

    def test_lookback_one_is_single_date(self):
        """lookback_days=1 reproduces the old single-date behavior exactly."""
        calls, _, _ = _run(sectors=["nba"], end_date=date(2026, 8, 8), lookback_days=1)
        assert calls == [date(2026, 8, 8)]

    def test_lookback_zero_clamps_to_one(self):
        calls, _, _ = _run(sectors=["nba"], end_date=date(2026, 8, 8), lookback_days=0)
        assert calls == [date(2026, 8, 8)]

    def test_default_window_length(self):
        calls, _, _ = _run(sectors=["nba"], end_date=date(2026, 8, 8))
        assert len(calls) == model_updater.DEFAULT_MODEL_BACKFILL_DAYS
        assert calls[-1] == date(2026, 8, 8)          # window ends on end_date
        assert calls == sorted(calls)                 # oldest → newest

    def test_passes_through_kwargs(self):
        cache: dict = {}
        _, _, seen = _run(
            sectors=["soccer"], end_date=date(2026, 8, 8), coordinator="COORD",
            lookback_days=1, dry_run=True, force=True, espn_cache=cache,
        )
        assert seen["sectors"] == ["soccer"]
        assert seen["coordinator"] == "COORD"
        assert seen["dry_run"] is True
        assert seen["force"] is True
        assert seen["espn_cache"] is cache
