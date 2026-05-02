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
    # Goal counts (soccer) — needed for Poisson walk-forward updates
    home_goals: Optional[int] = None
    away_goals: Optional[int] = None
    # Shot counts (soccer) — needed for xG walk-forward updates. Football-data
    # CSVs carry HS/AS (total shots) and HST/AST (shots on target). The live
    # SoccerXgAgent derives xG from these via xg_from_shots(sot, total).
    home_shots: Optional[int] = None
    away_shots: Optional[int] = None
    home_sot: Optional[int] = None
    away_sot: Optional[int] = None
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


@dataclass
class PropBacktestRow:
    """One settled NFL player-prop market with its model prediction."""
    sector: str
    stat_type: str              # passing_yards, rushing_yards, …
    player_name: str
    player_id: str              # gsis_id
    team: str
    opponent: str
    is_home: bool
    season: int
    week: int
    game_date: str
    threshold: float
    actual_result: int          # 1 = YES hit, 0 = NO
    # Market context
    volume: float
    closing_yes_price: Optional[float] = None  # last_price_dollars, [0,1]
    # Model output
    model_prob: float = 0.0     # predicted P(stat ≥ threshold)
    n_games_used: int = 0       # priors actually used
    # Derived
    @property
    def edge(self) -> Optional[float]:
        """Model prob − market implied prob (uses closing price)."""
        if self.closing_yes_price is None:
            return None
        return self.model_prob - self.closing_yes_price


@dataclass
class PropBacktestReport:
    """Per-sector report over NFL player props. Separate from BacktestReport
    because the unit here is a (player, stat, threshold) market, not a game.
    """
    sector: str
    stat_types: list[str]
    n_markets: int
    n_scored: int               # markets where the model produced a probability
    # Core metrics (scored markets only)
    brier_score: float = 0.0
    log_loss: float = 0.0
    accuracy: float = 0.0
    # Naive baselines
    baseline_mean_prior: float = 0.0  # Brier of always-predict-yes-rate
    baseline_market: float = 0.0      # Brier of closing market price (needs closing_yes_price)
    # Calibration
    calibration_bins: list[CalibrationBin] = field(default_factory=list)
    # Per-stat breakdown
    by_stat: dict[str, dict] = field(default_factory=dict)
    # ROI analysis at selected EV thresholds (uses closing price as fair)
    roi_by_ev_threshold: dict[str, dict] = field(default_factory=dict)
    # Volume-gated variants
    min_volume: float = 0.0
