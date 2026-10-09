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


@dataclass(frozen=True)
class QBRatings:
    """Point-in-time QB quality (EPA per dropback) and each team's assumed QB level.

    ``delta(team, qb)`` is how much better (+) or worse (-) the game's starter
    is than the QB mix the team's ratings were fit on — ~0 for the usual
    starter, negative when a backup starts.
    """
    ratings: dict[str, float]
    team_level: dict[str, float]
    prior: float

    def rating(self, qb_id) -> float:
        return self.ratings.get(qb_id, self.prior)

    def delta(self, team: str, qb_id) -> float:
        if qb_id is None or (isinstance(qb_id, float) and np.isnan(qb_id)) or team not in self.team_level:
            return 0.0
        return self.rating(qb_id) - self.team_level[team]


def fit_qb_ratings(team_games: pd.DataFrame, cutoff: pd.Timestamp,
                   qb_half_life_days: float = 365.0, team_half_life_days: float = 70.0,
                   k_dropbacks: float = 200.0, unknown_offset: float = 0.05,
                   lookback_days: int = 1095) -> QBRatings:
    """Shrunk, recency-weighted EPA/dropback per starting QB, from games before ``cutoff``.

    Each game credits the team's main passer (``starter_id``, most dropbacks)
    with his dropbacks and QB EPA. A QB with little history is shrunk toward
    ``league mean - unknown_offset`` (unfamiliar starters are usually backups or
    rookies). The team level weights the CURRENT ratings of the QBs who started
    for the team by their recency-weighted dropbacks (team half-life, matching
    the team ratings).
    """
    r = team_games[(team_games["gameday"] < cutoff)
                   & (team_games["gameday"] >= cutoff - pd.Timedelta(days=lookback_days))
                   & team_games["starter_id"].notna() & team_games["starter_epa"].notna()]
    if r.empty:
        return QBRatings({}, {}, 0.0)
    db = r["starter_dropbacks"].to_numpy(dtype=float)
    epa = r["starter_epa"].to_numpy(dtype=float)
    league = float(np.sum(db * epa) / np.sum(db))
    prior = league - unknown_offset
    wq = recency_weights(r["gameday"], cutoff, qb_half_life_days) * db
    agg = pd.DataFrame({"qb": r["starter_id"].to_numpy(), "w": wq, "we": wq * epa}).groupby("qb").sum()
    ratings = ((agg["we"] + k_dropbacks * prior) / (agg["w"] + k_dropbacks)).to_dict()
    wt = recency_weights(r["gameday"], cutoff, team_half_life_days) * db
    qb_r = np.array([ratings[q] for q in r["starter_id"]])
    tl = pd.DataFrame({"team": r["team"].to_numpy(), "w": wt, "wr": wt * qb_r}).groupby("team").sum()
    team_level = (tl["wr"] / tl["w"]).to_dict()
    return QBRatings(ratings=ratings, team_level=team_level, prior=prior)
