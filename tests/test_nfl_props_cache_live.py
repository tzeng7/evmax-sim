"""Tests for the NFL prop cache live-scan layer.

`compute_nfl_prop_prob_cached` is the runtime entry point that MODEL-9
wires into the scan coordinator. The tests here exercise the disk → memory
→ compute path against a small synthetic parquet fixture so CI is
hermetic (no dependency on the real `data/backtest/nfl_props/` files).

The pure probability math is already covered by `test_nfl_props_cache.py`.
This file focuses on the cache layer: parquet loading, name normalization,
point-in-time history selection, schedule-based opponent resolution, and
graceful fallback when inputs are missing.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evmax.clients import nfl_props_cache


def _run(coro):
    """Run a coroutine in a fresh event loop so tests don't share loop state.

    Python 3.10+ deprecated asyncio.get_event_loop() without a running loop;
    creating our own loop avoids "There is no current event loop in thread"
    failures that appear when prior test modules have closed the default.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture
def nfl_parquets(tmp_path: Path, monkeypatch) -> Path:
    """Build a synthetic 2-season parquet set the cache can load.

    Two QBs (high volume, low volume) + one RB, across 10 weeks of a
    single season. Thresholds were chosen so the model returns a range
    of probabilities rather than all 0.99 / 0.01.
    """
    rows = []
    # QB1 — big passer, 280-310 yds/game, 2 TDs/game
    for wk in range(1, 11):
        rows.append({
            "player_id": "QB1_ID",
            "player_display_name": "Big Passer",
            "position": "QB",
            "season": 2025,
            "week": wk,
            "team": "KC",
            "opponent_team": "LV" if wk % 2 else "DEN",
            "passing_yards": 295 + (wk % 3) * 10,
            "rushing_yards": 8 + (wk % 4),
            "receiving_yards": 0,
            "passing_tds": 2 + (wk % 2),
            "rushing_tds": 1 if wk == 3 else 0,
            "receiving_tds": 0,
            "receptions": 0,
        })
    # RB1 — 80 rush yds, 3 rec, 30 rec yds
    for wk in range(1, 11):
        rows.append({
            "player_id": "RB1_ID",
            "player_display_name": "Solid Rusher",
            "position": "RB",
            "season": 2025,
            "week": wk,
            "team": "SF",
            "opponent_team": "SEA" if wk % 2 else "ARI",
            "passing_yards": 0,
            "rushing_yards": 75 + (wk % 5) * 5,
            "receiving_yards": 20 + (wk % 3) * 3,
            "passing_tds": 0,
            "rushing_tds": 1 if wk in (2, 5, 8) else 0,
            "receiving_tds": 0,
            "receptions": 3,
        })
    # QB2 — thin sample (only 3 games — below MIN_GAMES)
    for wk in range(1, 4):
        rows.append({
            "player_id": "QB2_ID",
            "player_display_name": "Rookie QB",
            "position": "QB",
            "season": 2025,
            "week": wk,
            "team": "NYG",
            "opponent_team": "DAL",
            "passing_yards": 200,
            "rushing_yards": 0,
            "receiving_yards": 0,
            "passing_tds": 1,
            "rushing_tds": 0,
            "receiving_tds": 0,
            "receptions": 0,
        })
    weekly = pd.DataFrame(rows)

    rosters = pd.DataFrame([
        {"player_display_name": "Big Passer", "gsis_id": "QB1_ID", "season": 2025},
        {"player_display_name": "Solid Rusher", "gsis_id": "RB1_ID", "season": 2025},
        {"player_display_name": "Rookie QB", "gsis_id": "QB2_ID", "season": 2025},
        # Veteran who wasn't in the weekly table — tests name resolution
        # fallback when the player has no game log.
        {"player_display_name": "Unknown Vet", "gsis_id": "VET_ID", "season": 2025},
    ])

    schedules = pd.DataFrame([
        {"season": 2025, "week": 11, "gameday": "2025-11-17", "home_team": "KC", "away_team": "LV"},
        {"season": 2025, "week": 11, "gameday": "2025-11-17", "home_team": "SF", "away_team": "SEA"},
    ])

    d = tmp_path / "nfl_props"
    d.mkdir()
    weekly.to_parquet(d / "weekly_stats.parquet")
    rosters.to_parquet(d / "rosters.parquet")
    schedules.to_parquet(d / "schedules.parquet")

    monkeypatch.setattr(nfl_props_cache, "_NFL_DATA_DIR", d)
    monkeypatch.setattr(nfl_props_cache, "_WEEKLY_PATH", d / "weekly_stats.parquet")
    monkeypatch.setattr(nfl_props_cache, "_ROSTERS_PATH", d / "rosters.parquet")
    monkeypatch.setattr(nfl_props_cache, "_SCHEDULES_PATH", d / "schedules.parquet")
    nfl_props_cache._reset_cache_for_tests()
    yield d
    nfl_props_cache._reset_cache_for_tests()


class TestCacheLoad:
    def test_is_fresh_true_when_all_parquets_present(self, nfl_parquets):
        assert nfl_props_cache.is_nfl_cache_fresh() is True

    def test_is_fresh_false_when_any_parquet_missing(self, nfl_parquets):
        (nfl_parquets / "rosters.parquet").unlink()
        nfl_props_cache._reset_cache_for_tests()
        assert nfl_props_cache.is_nfl_cache_fresh() is False

    def test_refresh_returns_player_count(self, nfl_parquets):
        n = _run(nfl_props_cache.refresh_nfl_props_cache())
        # Every roster entry resolves — 4 entries in the synthetic fixture
        assert n == 4

    def test_name_normalization_is_case_and_suffix_insensitive(self, nfl_parquets):
        nfl_props_cache._load_tables()
        assert nfl_props_cache._normalize_name("Big Passer") == "big passer"
        assert nfl_props_cache._normalize_name("BIG   PASSER Jr.") == "big passer"
        assert nfl_props_cache._normalize_name("Dončić") == "doncic"


class TestComputeCached:
    def test_returns_none_for_unknown_player(self, nfl_parquets):
        result = nfl_props_cache.compute_nfl_prop_prob_cached(
            "Who Is This", "passing_yards", 249.5, None
        )
        assert result is None

    def test_returns_none_for_thin_sample_player(self, nfl_parquets):
        # Rookie QB only has 3 games; MIN_GAMES = 4
        result = nfl_props_cache.compute_nfl_prop_prob_cached(
            "Rookie QB", "passing_yards", 200, None
        )
        assert result is None

    def test_scores_qb_passing_yards(self, nfl_parquets):
        result = nfl_props_cache.compute_nfl_prop_prob_cached(
            "Big Passer", "passing_yards", 249.5, None
        )
        assert result is not None
        # He averages ~300 yds with a tight spread, so P(>= 249.5) must be
        # very high but clamped under 0.99 by the model.
        assert 0.70 <= result.prob <= 0.99
        assert result.n_games >= 4

    def test_scores_rb_rushing_yards(self, nfl_parquets):
        result = nfl_props_cache.compute_nfl_prop_prob_cached(
            "Solid Rusher", "rushing_yards", 69.5, None
        )
        assert result is not None
        assert 0.50 <= result.prob <= 0.99

    def test_scores_anytime_td_via_poisson_branch(self, nfl_parquets):
        # Solid Rusher scored rushing TDs in 3 / 10 games → λ ≈ 0.3 →
        # P(>= 1) ≈ 1 - e^-0.3 ≈ 0.26. The real answer depends on the
        # poisson tail math; just assert it's in the believable middle.
        result = nfl_props_cache.compute_nfl_prop_prob_cached(
            "Solid Rusher", "anytime_td", 0.5, None
        )
        assert result is not None
        assert 0.10 <= result.prob <= 0.80

    def test_unknown_stat_type_returns_none(self, nfl_parquets):
        result = nfl_props_cache.compute_nfl_prop_prob_cached(
            "Big Passer", "not_a_stat", 100, None
        )
        assert result is None


class TestCoordinatorFetchPropsNfl:
    """Integration test: AgentCoordinator._fetch_props('nfl') must produce
    SharpOdds via the NFL branch. Exercises the MODEL-9 wire-up from Kalshi
    market fetch through to cached probability compute, without hitting
    the network or the real parquets.
    """

    def test_fetch_props_nfl_returns_sharp_odds(self, nfl_parquets, monkeypatch):
        from datetime import datetime
        from unittest.mock import AsyncMock

        from evmax.agents.coordinator import AgentCoordinator
        from evmax.models.market import MarketSource, MarketType, PredictionMarket

        # Build two fake Kalshi prop markets for players in the synthetic
        # parquet fixture so the cache can actually score them.
        markets = [
            PredictionMarket(
                id="KXNFLPASSYDS-25NOV17KC-B299.5",
                source=MarketSource.kalshi,
                sector="nfl",
                market_type=MarketType.player_prop,
                title="Big Passer 299.5 Passing Yards",
                event_id="nfl::2025-11-17::prop::bigpasser::passing_yards::299.5",
                team_home="KC",
                team_away="LV",
                yes_team="Big Passer",
                yes_price=0.50,
                no_price=0.50,
                event_date=datetime(2025, 11, 17, 20, 0),
                player_name="Big Passer",
                stat_type="passing_yards",
                threshold=299.5,
            ),
            PredictionMarket(
                id="KXNFLRSHYDS-25NOV17SF-R79.5",
                source=MarketSource.kalshi,
                sector="nfl",
                market_type=MarketType.player_prop,
                title="Solid Rusher 79.5 Rushing Yards",
                event_id="nfl::2025-11-17::prop::solidrusher::rushing_yards::79.5",
                team_home="SF",
                team_away="SEA",
                yes_team="Solid Rusher",
                yes_price=0.50,
                no_price=0.50,
                event_date=datetime(2025, 11, 17, 20, 0),
                player_name="Solid Rusher",
                stat_type="rushing_yards",
                threshold=79.5,
            ),
        ]

        # Mock KalshiClient so the coordinator doesn't touch the network
        class _FakeKalshi:
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return None
            async def get_markets(self, sector):
                return markets if sector == "nfl_props" else []

        from evmax.clients import kalshi as kalshi_mod
        monkeypatch.setattr(kalshi_mod, "KalshiClient", _FakeKalshi)

        coord = AgentCoordinator(sectors=["nfl"], enable_models=False)
        prop_markets, prop_sharp = _run(coord._fetch_props("nfl"))

        # Both markets make it through dedupe
        assert len(prop_markets) == 2
        # Both players score — the cache fixture has enough history for both
        assert len(prop_sharp) == 2
        names = {s.prop_player_name for s in prop_sharp}
        assert names == {"Big Passer", "Solid Rusher"}

        # Every SharpOdds carries a valid true_prob_over from the model
        for s in prop_sharp:
            assert 0.0 < s.true_prob_over < 1.0
            assert s.true_prob_under == 1.0 - s.true_prob_over
            assert s.prop_l15_games >= 4  # MIN_GAMES
            assert s.sector == "nfl"
            assert s.prop_stat_type in {"passing_yards", "rushing_yards"}

    def test_fetch_props_nfl_skips_unknown_player(self, nfl_parquets, monkeypatch):
        from datetime import datetime

        from evmax.agents.coordinator import AgentCoordinator
        from evmax.models.market import MarketSource, MarketType, PredictionMarket

        markets = [
            PredictionMarket(
                id="KXNFLPASSYDS-MYSTERY",
                source=MarketSource.kalshi,
                sector="nfl",
                market_type=MarketType.player_prop,
                title="Mystery QB 100 Passing Yards",
                event_id="nfl::2025-11-17::prop::mystery::passing_yards::100",
                team_home="XXX",
                team_away="YYY",
                yes_team="Mystery QB",
                yes_price=0.50,
                no_price=0.50,
                event_date=datetime(2025, 11, 17, 20, 0),
                player_name="Mystery QB",
                stat_type="passing_yards",
                threshold=100.0,
            ),
        ]

        class _FakeKalshi:
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return None
            async def get_markets(self, sector):
                return markets if sector == "nfl_props" else []

        from evmax.clients import kalshi as kalshi_mod
        monkeypatch.setattr(kalshi_mod, "KalshiClient", _FakeKalshi)

        coord = AgentCoordinator(sectors=["nfl"], enable_models=False)
        prop_markets, prop_sharp = _run(coord._fetch_props("nfl"))

        # Market dedupes through but nobody can be scored
        assert len(prop_markets) == 1
        assert len(prop_sharp) == 0


class TestScheduleResolution:
    def test_game_date_picks_correct_season_and_week(self, nfl_parquets):
        # 2025-11-17 is week 11 in the synthetic schedule. Big Passer's
        # history should be weeks 1-10 (< 11). With game_date set, the
        # model uses the point-in-time history cleanly.
        result = nfl_props_cache.compute_nfl_prop_prob_cached(
            "Big Passer", "passing_yards", 249.5, "2025-11-17"
        )
        assert result is not None
        assert result.n_games == 8  # LAST_N_GAMES cap

    def test_opponent_adjustment_applies_when_schedule_resolves(self, nfl_parquets):
        # With game_date → schedule lookup, opponent = LV (Big Passer's team
        # is KC and KC plays LV on 2025-11-17 per the fixture). The defense
        # table is sparse in the synthetic data so opp_adj may just be 1.0,
        # but the call must still succeed and return a valid probability.
        result = nfl_props_cache.compute_nfl_prop_prob_cached(
            "Big Passer", "passing_tds", 1.5, "2025-11-17"
        )
        assert result is not None
        assert 0.0 < result.prob < 1.0
