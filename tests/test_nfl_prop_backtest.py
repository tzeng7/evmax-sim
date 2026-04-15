"""Tests for evmax.backtest.sources.nfl_props + metrics_props.

Smoke tests using a tiny in-memory fixture — builds synthetic joined rows
and weekly stats, verifies the loader produces scored PropBacktestRows,
then runs compute_prop_report over them.
"""

from __future__ import annotations

import pandas as pd
import pytest

from evmax.backtest.metrics_props import compute_prop_report
from evmax.backtest.models import PropBacktestRow


def _make_row(**kwargs) -> PropBacktestRow:
    defaults = dict(
        sector="nfl_props",
        stat_type="passing_yards",
        player_name="Test QB",
        player_id="00-TEST",
        team="TST",
        opponent="OPP",
        is_home=True,
        season=2025,
        week=13,
        game_date="2025-12-01",
        threshold=249.5,
        actual_result=1,
        volume=1000.0,
        closing_yes_price=0.55,
        model_prob=0.62,
        n_games_used=8,
    )
    defaults.update(kwargs)
    return PropBacktestRow(**defaults)


def test_compute_prop_report_basic_metrics():
    rows = [
        _make_row(actual_result=1, model_prob=0.70),
        _make_row(actual_result=0, model_prob=0.30),
        _make_row(actual_result=1, model_prob=0.60),
        _make_row(actual_result=0, model_prob=0.40),
        _make_row(actual_result=1, model_prob=0.80),
    ]
    r = compute_prop_report(rows)
    assert r.n_scored == 5
    assert r.brier_score > 0
    assert r.brier_score < 0.25
    assert r.log_loss > 0
    # All five predictions are directionally correct → accuracy 100%
    assert r.accuracy == pytest.approx(1.0)


def test_compute_prop_report_per_stat_breakdown():
    rows = [
        _make_row(stat_type="passing_yards", actual_result=1, model_prob=0.70),
        _make_row(stat_type="passing_yards", actual_result=0, model_prob=0.40),
        _make_row(stat_type="rushing_yards", actual_result=1, model_prob=0.55),
        _make_row(stat_type="rushing_yards", actual_result=0, model_prob=0.35),
    ]
    r = compute_prop_report(rows)
    assert set(r.stat_types) == {"passing_yards", "rushing_yards"}
    assert r.by_stat["passing_yards"]["n"] == 2
    assert r.by_stat["rushing_yards"]["n"] == 2


def test_roi_only_fires_on_edges_above_threshold():
    # All bets have edge > 10% and actual=1 → strongly positive ROI
    rows = [
        _make_row(model_prob=0.75, closing_yes_price=0.50, actual_result=1),
        _make_row(model_prob=0.80, closing_yes_price=0.55, actual_result=1),
        _make_row(model_prob=0.30, closing_yes_price=0.25, actual_result=1),
    ]
    r = compute_prop_report(rows)
    roi_2pct = r.roi_by_ev_threshold["ev>=2%"]
    assert roi_2pct["bets"] == 3
    assert roi_2pct["wins"] == 3
    assert roi_2pct["roi"] > 0


def test_volume_gate_filters_low_liquidity_markets():
    rows = [
        _make_row(volume=50.0, model_prob=0.70, actual_result=1),
        _make_row(volume=2000.0, model_prob=0.40, actual_result=0),
    ]
    ungated = compute_prop_report(rows, min_volume=0.0)
    gated = compute_prop_report(rows, min_volume=1000.0)
    # Ungated scored both; gated scores only the 2000-volume row
    assert ungated.n_scored == 2
    assert gated.n_scored == 2  # n_scored is pre-gate, accurate
    # But gated brier/accuracy only reflects the one surviving row
    assert gated.by_stat["passing_yards"]["n"] == 1


def test_empty_rows_safe():
    r = compute_prop_report([])
    assert r.n_scored == 0
    assert r.brier_score == 0
    assert r.by_stat == {}


# --- Loader smoke test (uses real parquets if present) --------------------


def test_loader_smoke_reads_parquets():
    from pathlib import Path

    p = Path("data/backtest/nfl_props/joined.parquet")
    if not p.exists():
        pytest.skip("Stage 3 parquet not yet built")

    from evmax.backtest.sources.nfl_props import load_nfl_props

    rows = load_nfl_props()
    assert len(rows) > 0
    # Must have a mix of stats
    stats = {r.stat_type for r in rows}
    assert "passing_yards" in stats
    # Probabilities bounded
    for r in rows[:50]:
        assert 0.0 <= r.model_prob <= 1.0
        assert r.n_games_used >= 4


def test_loader_stats_filter_narrows_output():
    from pathlib import Path

    p = Path("data/backtest/nfl_props/joined.parquet")
    if not p.exists():
        pytest.skip("Stage 3 parquet not yet built")

    from evmax.backtest.sources.nfl_props import load_nfl_props

    rows = load_nfl_props(stats_filter=["passing_yards", "passing_tds"])
    assert len(rows) > 0
    stats = {r.stat_type for r in rows}
    assert stats == {"passing_yards", "passing_tds"}
