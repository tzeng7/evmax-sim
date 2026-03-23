"""Backtest data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional


@dataclass
class BacktestRow:
    """One historical match with Pinnacle odds and actual result."""
    sector: str
    league: str                          # "EPL", "La Liga", etc.
    date: date
    team_home: str                       # raw from CSV
    team_away: str                       # raw from CSV
    pinnacle_home_dec: float             # Pinnacle decimal odds
    pinnacle_away_dec: float
    pinnacle_draw_dec: Optional[float]   # None for tennis
    result: str                          # "H"/"D"/"A" (soccer) or "W" (tennis — home=Winner always)
    # Derived after devigging
    true_prob_home: float = 0.0
    true_prob_away: float = 0.0
    true_prob_draw: Optional[float] = None
    # Convenience
    home_won: Optional[bool] = None      # True if home team / winner won
    draw: Optional[bool] = None          # True if result is draw (soccer only)
    # Optional Kalshi join
    kalshi_ticker: Optional[str] = None
    kalshi_last_price: Optional[float] = None
    kalshi_yes_team: Optional[str] = None
    kalshi_result: Optional[str] = None  # "yes" or "no"
    theoretical_ev: Optional[float] = None


@dataclass
class CalibrationBin:
    prob_low: float
    prob_high: float
    mean_pred: float
    actual_rate: float
    n: int
    ci_low: float    # 95% Wilson CI
    ci_high: float


@dataclass
class BacktestReport:
    sector: str
    leagues: list[str]
    seasons: list[str]
    n_games: int
    n_kalshi_matched: int = 0

    # Core metrics
    brier_score: float = 0.0
    log_loss: float = 0.0
    accuracy: float = 0.0
    baseline_accuracy: float = 0.0   # always pick favourite
    mean_pinnacle_margin: float = 0.0

    # Calibration (primary outcome: home/winner)
    calibration_bins: list[CalibrationBin] = field(default_factory=list)

    # Theoretical EV (requires Kalshi match — heavily caveated)
    n_positive_ev: int = 0
    n_above_threshold: int = 0
    ev_threshold: float = 0.02
    mean_ev_positive: float = 0.0
    ev_buckets: dict[str, int] = field(default_factory=dict)

    # League breakdown
    by_league: dict[str, dict] = field(default_factory=dict)

    @property
    def random_brier_baseline(self) -> float:
        return 0.333 if self.sector == "soccer" else 0.250
