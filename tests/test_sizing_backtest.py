"""Tests for the sizing replay harness (evmax/backtest/sizing.py)."""

from __future__ import annotations

import numpy as np
import pytest

from evmax.backtest.sizing import (
    ResolvedRow,
    SimResult,
    block_bootstrap_log_growth,
    edge_ratio,
    make_kelly_policy,
    simulate,
    walk_forward_months,
)


def _row(mid, date, blended, price, outcome, sector="nba", event=None, ev=0.05, mt="moneyline"):
    return ResolvedRow(
        market_id=mid, sector=sector, market_type=mt,
        event_id=event or f"{sector}::{date}::{mid}", event_date=date,
        blended=blended, price=price, outcome=outcome, ev_pct=ev,
    )


def test_simulate_single_winning_bet():
    # One bet: flat 10% of bankroll at price 0.5 (payout 2, b=1, ignore fee).
    rows = [_row("m1", "2026-01-01", 0.6, 0.5, 1)]
    res = simulate(rows, lambda r: 0.10, fee_venue=None, event_cap=1.0)
    assert res.n_bets == 1
    # win: wealth *= (1 + 0.10 * 1.0) = 1.10
    assert res.final_multiple == pytest.approx(1.10, abs=1e-9)
    assert res.log_growth == pytest.approx(np.log(1.10), abs=1e-9)


def test_simulate_losing_bet():
    rows = [_row("m1", "2026-01-01", 0.6, 0.5, 0)]
    res = simulate(rows, lambda r: 0.10, fee_venue=None, event_cap=1.0)
    assert res.final_multiple == pytest.approx(0.90, abs=1e-9)


def test_same_day_settlement_is_simultaneous():
    # Two bets same day: a win and a loss, each 10% at b=1. Simultaneous settle:
    # wealth *= (1 + 0.10 - 0.10) = 1.0. Serial would give 1.1*0.9=0.99.
    rows = [
        _row("m1", "2026-01-01", 0.6, 0.5, 1, event="nba::d::g1"),
        _row("m2", "2026-01-01", 0.6, 0.5, 0, event="nba::d::g2"),
    ]
    res = simulate(rows, lambda r: 0.10, fee_venue=None, event_cap=1.0)
    assert res.final_multiple == pytest.approx(1.0, abs=1e-9)


def test_exposure_cap_limits_same_game_stake():
    # Three legs on ONE game, each wants 5%, cap 8% → total staked = 8%.
    rows = [
        _row("m1", "2026-01-01", 0.6, 0.5, 1, event="nba::d::g1", ev=0.09),
        _row("m2", "2026-01-01", 0.6, 0.5, 1, event="nba::d::g1", ev=0.08),
        _row("m3", "2026-01-01", 0.6, 0.5, 1, event="nba::d::g1", ev=0.07),
    ]
    res = simulate(rows, lambda r: 0.05, fee_venue=None, event_cap=0.08)
    # All win at b=1: day_pnl == total staked == 0.08 → wealth 1.08.
    assert res.final_multiple == pytest.approx(1.08, abs=1e-9)


def test_edge_ratio_honest_model_is_one():
    # A model whose blended == realized win rate, priced at cost, gives ratio 1.
    # Deterministic: exactly 60% winners at blended 0.6, price 0.5.
    rows = []
    for i in range(400):
        y = 1 if (i % 10) < 6 else 0
        rows.append(_row(f"m{i}", f"2026-01-{(i%28)+1:02d}", 0.6, 0.50, y))
    r = edge_ratio(rows, fee_venue=None)
    # predicted edge = 0.6-0.5 = 0.1; realized = 0.6-0.5 = 0.1 → exactly 1.
    assert r == pytest.approx(1.0, abs=1e-9)


def test_edge_ratio_below_one_when_model_overstates():
    # Model claims 0.6 but only 50% win → realized edge 0 → ratio 0.
    rows = [_row(f"m{i}", f"2026-01-{(i%28)+1:02d}", 0.6, 0.50, i % 2) for i in range(200)]
    assert edge_ratio(rows, fee_venue=None) == pytest.approx(0.0, abs=1e-9)


def test_kelly_policy_caps_and_floors():
    pol = make_kelly_policy(base_fraction=0.5, max_kelly=0.05, fee_venue=None)
    # Huge edge → capped at 0.05.
    assert pol(_row("m", "2026-01-01", 0.95, 0.50, 1)) == pytest.approx(0.05)
    # -EV → 0.
    assert pol(_row("m", "2026-01-01", 0.40, 0.50, 0)) == 0.0


def test_walk_forward_scores_only_held_out_months():
    # 4 months of coin-flip-ish data; walk-forward with 2-month warmup places
    # months 3 and 4 → n bets == rows in months 3+4.
    rows = []
    for m in range(1, 5):
        for i in range(10):
            rows.append(_row(f"m{m}_{i}", f"2026-0{m}-15", 0.6, 0.5, i % 2))
    res = walk_forward_months(
        rows, lambda tr: make_kelly_policy(base_fraction=0.5, fee_venue=None),
        fee_venue=None, min_train_months=2,
    )
    assert res.n_bets == 20  # months 3 and 4 only


def test_bootstrap_returns_percentiles():
    rows = [_row(f"m{i}", f"2026-01-{(i%28)+1:02d}", 0.6, 0.5, i % 2) for i in range(60)]
    boot = block_bootstrap_log_growth(
        rows, make_kelly_policy(fee_venue=None), fee_venue=None, n_boot=200
    )
    assert set(boot) == {5.0, 50.0, 95.0}
    assert boot[5.0] <= boot[50.0] <= boot[95.0]
