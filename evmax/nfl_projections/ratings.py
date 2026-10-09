"""Point-in-time opponent-adjusted team ratings (recency-weighted ridge).

For a metric ``m`` measured on team-game rows (the team on offense):

    m = mu + hfa * home + off[team] + def[opp] + noise

``off`` is how much the team adds on offense and ``def`` how much it ALLOWS on
defense (higher = worse defense), both shrunk toward 0 by a ridge penalty
``lam``. Rows are weighted by age: ``0.5 ** (days_before_cutoff / half_life)``
— a calendar half-life, so last season's games fade naturally across the
offseason and early-season ratings regress toward the league mean (the
penalty dominates while recent weight is small). Only rows strictly before
``cutoff`` enter a fit, which makes every walk-forward projection leak-free.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RatingFit:
    metric: str
    mu: float
    hfa: float
    off: dict[str, float] = field(default_factory=dict)
    deff: dict[str, float] = field(default_factory=dict)

    def expect(self, team: str, opp: str, home: int) -> float:
        """Expected metric for ``team`` on offense against ``opp``'s defense."""
        return self.mu + self.hfa * home + self.off.get(team, 0.0) + self.deff.get(opp, 0.0)


def recency_weights(gameday: pd.Series, cutoff: pd.Timestamp, half_life_days: float) -> np.ndarray:
    age = (cutoff - gameday).dt.days.to_numpy(dtype=float)
    return 0.5 ** (age / half_life_days)


def fit_rating(rows: pd.DataFrame, metric: str, cutoff: pd.Timestamp,
               half_life_days: float = 70.0, lam: float = 4.0,
               weight_col: str | None = None) -> RatingFit:
    """Fit one metric's ratings on ``rows`` with gameday < cutoff.

    ``weight_col`` (e.g. plays) multiplies the recency weight so per-play
    metrics weight games by their sample size.
    """
    r = rows[(rows["gameday"] < cutoff) & rows[metric].notna()]
    if r.empty:
        return RatingFit(metric, float("nan"), 0.0)
    teams = sorted(set(r["team"]) | set(r["opp"]))
    ix = {t: i for i, t in enumerate(teams)}
    n, k = len(r), len(teams)
    X = np.zeros((n, 2 + 2 * k))
    X[:, 0] = 1.0
    X[:, 1] = r["home"].to_numpy(dtype=float)
    rows_i = np.arange(n)
    X[rows_i, 2 + r["team"].map(ix).to_numpy()] = 1.0
    X[rows_i, 2 + k + r["opp"].map(ix).to_numpy()] = 1.0
    w = recency_weights(r["gameday"], cutoff, half_life_days)
    if weight_col is not None:
        w = w * r[weight_col].to_numpy(dtype=float)
        w = w / np.mean(r[weight_col].to_numpy(dtype=float))
    y = r[metric].to_numpy(dtype=float)
    penalty = np.diag([0.0, 0.0] + [lam] * (2 * k))
    beta = np.linalg.solve(X.T @ (X * w[:, None]) + penalty, X.T @ (w * y))
    return RatingFit(
        metric=metric, mu=float(beta[0]), hfa=float(beta[1]),
        off={t: float(beta[2 + i]) for t, i in ix.items()},
        deff={t: float(beta[2 + k + i]) for t, i in ix.items()},
    )
