"""Metrics calculator — P&L, ROI, Sharpe ratio, max drawdown."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from evmax.models.simulated_bet import BetStatus, SimulatedBet


@dataclass
class PerformanceMetrics:
    total_bets: int
    won_bets: int
    lost_bets: int
    open_bets: int
    win_rate: float  # 0.0–1.0

    total_wagered_usd: float
    total_pnl_usd: float
    roi_pct: float  # as percentage

    sharpe_ratio: Optional[float]
    max_drawdown_pct: float  # as percentage of peak bankroll

    avg_ev: Optional[float]
    avg_edge_pct: Optional[float]
    best_win_usd: float
    worst_loss_usd: float


def calculate_metrics(
    bets: list[SimulatedBet],
    initial_bankroll: float = 10_000.0,
) -> PerformanceMetrics:
    """
    Calculate performance metrics for a list of simulated bets.

    Args:
        bets: All simulated bets (any status).
        initial_bankroll: Starting bankroll for drawdown calculation.

    Returns:
        PerformanceMetrics dataclass.
    """
    settled = [b for b in bets if b.status in (BetStatus.won, BetStatus.lost)]
    won = [b for b in settled if b.status == BetStatus.won]
    lost = [b for b in settled if b.status == BetStatus.lost]
    open_bets = [b for b in bets if b.status == BetStatus.open]

    total_wagered = sum(b.stake_usd for b in settled)
    total_pnl = sum(b.pnl_usd for b in settled)
    roi_pct = (total_pnl / total_wagered * 100) if total_wagered > 0 else 0.0

    win_rate = len(won) / len(settled) if settled else 0.0

    # Per-bet P&L as fraction of stake (unit returns)
    unit_returns = [b.pnl_usd / b.stake_usd for b in settled if b.stake_usd > 0]
    sharpe = _sharpe_ratio(unit_returns) if len(unit_returns) >= 5 else None

    # Max drawdown
    bankroll_curve = _build_bankroll_curve(bets, initial_bankroll)
    max_dd = _max_drawdown(bankroll_curve)

    best_win = max((b.pnl_usd for b in won), default=0.0)
    worst_loss = min((b.pnl_usd for b in lost), default=0.0)

    return PerformanceMetrics(
        total_bets=len(bets),
        won_bets=len(won),
        lost_bets=len(lost),
        open_bets=len(open_bets),
        win_rate=win_rate,
        total_wagered_usd=total_wagered,
        total_pnl_usd=total_pnl,
        roi_pct=roi_pct,
        sharpe_ratio=sharpe,
        max_drawdown_pct=max_dd,
        avg_ev=None,  # Populated by pipeline with ev data
        avg_edge_pct=None,
        best_win_usd=best_win,
        worst_loss_usd=worst_loss,
    )


def _sharpe_ratio(returns: list[float], risk_free: float = 0.0) -> Optional[float]:
    """Annualized Sharpe ratio from per-bet unit returns."""
    if len(returns) < 5:
        return None
    arr = np.array(returns)
    mean = float(np.mean(arr)) - risk_free
    std = float(np.std(arr, ddof=1))
    if std == 0:
        return None
    # Scale: assume ~1000 bets/year for annualization
    return float(mean / std * np.sqrt(1000))


def _build_bankroll_curve(
    bets: list[SimulatedBet],
    initial_bankroll: float,
) -> list[float]:
    """Build bankroll curve over time from resolved bets."""
    # Sort by placed_at
    sorted_bets = sorted(bets, key=lambda b: b.placed_at)
    curve = [initial_bankroll]
    balance = initial_bankroll

    for bet in sorted_bets:
        if bet.status in (BetStatus.won, BetStatus.lost):
            balance += bet.pnl_usd
            curve.append(balance)

    return curve


def _max_drawdown(curve: list[float]) -> float:
    """Compute maximum peak-to-trough drawdown as percentage."""
    if len(curve) < 2:
        return 0.0

    arr = np.array(curve)
    peak = arr[0]
    max_dd = 0.0

    for val in arr[1:]:
        if val > peak:
            peak = val
        dd = (peak - val) / peak * 100 if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    return max_dd
