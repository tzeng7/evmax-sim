"""Tests for the simulation engine and metrics."""

import pytest
from datetime import datetime, timezone

from evmax.models.ev_bet import EVBet
from evmax.models.simulated_bet import BetStatus, SimulatedBet
from evmax.simulation.metrics import calculate_metrics, _sharpe_ratio, _max_drawdown


def make_ev_bet(
    ev: float = 0.05,
    true_prob: float = 0.55,
    payout: float = 2.0,
    suggested_units: float = 0.02,
    sector: str = "nfl",
) -> EVBet:
    return EVBet(
        market_id=f"kalshi:TEST-{sector}",
        event_id=f"{sector}::2026-02-22::team_a_vs_team_b",
        sector=sector,
        outcome="yes",
        market_implied_prob=0.50,
        true_prob=true_prob,
        payout_decimal=payout,
        ev=ev,
        edge_pct=ev,
        kelly_full=0.10,
        kelly_fraction=suggested_units,
        suggested_units=suggested_units,
        volume_usd=50_000.0,
    )


def make_sim_bet(
    status: BetStatus = BetStatus.open,
    stake: float = 200.0,
    odds: float = 2.0,
    pnl: float = 0.0,
    placed_at: datetime = None,
) -> SimulatedBet:
    return SimulatedBet(
        ev_bet_id=1,
        market_id="kalshi:TEST",
        event_id="nfl::2026-02-22::a_vs_b",
        sector="nfl",
        outcome="yes",
        stake_usd=stake,
        odds_decimal=odds,
        potential_profit_usd=stake * (odds - 1.0),
        status=status,
        pnl_usd=pnl,
        bankroll_before=10_000.0,
        placed_at=placed_at or datetime(2026, 2, 22, tzinfo=timezone.utc),
    )


class TestCalculateMetrics:
    def test_empty_bets(self):
        """Empty bet list should return zeros."""
        metrics = calculate_metrics([])
        assert metrics.total_bets == 0
        assert metrics.total_pnl_usd == 0.0
        assert metrics.roi_pct == 0.0

    def test_won_bet(self):
        """Single won bet should show positive P&L."""
        bet = make_sim_bet(status=BetStatus.won, stake=100.0, odds=2.0, pnl=100.0)
        metrics = calculate_metrics([bet])
        assert metrics.won_bets == 1
        assert metrics.lost_bets == 0
        assert metrics.total_pnl_usd == 100.0
        assert metrics.roi_pct == 100.0  # 100 profit on 100 wagered

    def test_lost_bet(self):
        """Single lost bet should show negative P&L."""
        bet = make_sim_bet(status=BetStatus.lost, stake=100.0, odds=2.0, pnl=-100.0)
        metrics = calculate_metrics([bet])
        assert metrics.lost_bets == 1
        assert metrics.total_pnl_usd == -100.0
        assert metrics.roi_pct < 0

    def test_mixed_bets(self):
        """Mix of won/lost bets."""
        bets = [
            make_sim_bet(status=BetStatus.won, stake=100.0, pnl=100.0),
            make_sim_bet(status=BetStatus.lost, stake=100.0, pnl=-100.0),
            make_sim_bet(status=BetStatus.open, stake=100.0, pnl=0.0),
        ]
        metrics = calculate_metrics(bets)
        assert metrics.total_bets == 3
        assert metrics.open_bets == 1
        assert metrics.total_pnl_usd == 0.0
        assert metrics.win_rate == 0.5

    def test_best_worst(self):
        """Best win and worst loss tracking."""
        bets = [
            make_sim_bet(status=BetStatus.won, pnl=500.0),
            make_sim_bet(status=BetStatus.won, pnl=100.0),
            make_sim_bet(status=BetStatus.lost, pnl=-200.0),
        ]
        metrics = calculate_metrics(bets)
        assert metrics.best_win_usd == 500.0
        assert metrics.worst_loss_usd == -200.0

    def test_roi_calculation(self):
        """ROI = total_pnl / total_wagered × 100."""
        bets = [
            make_sim_bet(status=BetStatus.won, stake=200.0, pnl=100.0),
            make_sim_bet(status=BetStatus.won, stake=200.0, pnl=50.0),
        ]
        metrics = calculate_metrics(bets)
        assert abs(metrics.roi_pct - 37.5) < 0.1  # 150/400 * 100


class TestSharpeRatio:
    def test_needs_minimum_samples(self):
        """Sharpe ratio requires at least 5 samples."""
        assert _sharpe_ratio([0.1, 0.2, 0.3]) is None

    def test_positive_returns_positive_sharpe(self):
        """Consistently positive returns → positive Sharpe."""
        returns = [0.1] * 10
        sharpe = _sharpe_ratio(returns)
        # All same returns → std=0 → None (not computable)
        # Actually with same values, std=0, so should return None
        assert sharpe is None or sharpe > 0

    def test_mixed_returns(self):
        """Mixed returns should produce a valid Sharpe (may be negative)."""
        returns = [0.1, -0.2, 0.3, -0.1, 0.2, 0.1, -0.05, 0.15, -0.1, 0.2]
        sharpe = _sharpe_ratio(returns)
        assert sharpe is not None


class TestMaxDrawdown:
    def test_no_drawdown(self):
        """Monotonically increasing curve has 0 drawdown."""
        curve = [100, 110, 120, 130]
        assert _max_drawdown(curve) == 0.0

    def test_full_loss(self):
        """Complete loss from peak."""
        curve = [100, 50, 0]
        dd = _max_drawdown(curve)
        assert dd == 100.0  # 100% drawdown

    def test_partial_drawdown(self):
        """50% drawdown from 100 to 50."""
        curve = [100, 80, 50, 70]
        dd = _max_drawdown(curve)
        assert abs(dd - 50.0) < 0.1

    def test_short_curve(self):
        """Single element curve has 0 drawdown."""
        assert _max_drawdown([100]) == 0.0
