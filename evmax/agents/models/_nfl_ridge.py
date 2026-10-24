"""Ridge opponent-adjusted EPA ratings for the NFL efficiency model.

Pure math shared by the weekly seed (`scripts/seed_nfl_efficiency.py`), the
walk-forward harness (`scripts/exp_nfl_eff_ridge.py`) and the blend backtest, so
all three compute the identical rating.

Model (additive, signed EPA — the same family as the NCAAF solver, not the
multiplicative basketball one):

    epa_play = mu + h*[offense is home] + off[posteam] + dfa[defteam]

fitted by weighted ridge over (posteam, defteam, home) cells. `dfa` is EPA
ALLOWED (positive = bad defense), so a team's net rating is `off - dfa`
(higher = better), the same sign convention as the legacy `off_epa_adj -
def_epa_adj`.

Recency is a smooth exponential in play age (half-life `WEEK_HALF_LIFE`
weeks, measured from the newest play in the slice) with NO per-season factor:
the walk-forward lens (scripts/exp_nfl_eff_ridge.py) found that replacing the
legacy 0.45-per-season / flat-within-season weighting with a 26-week half-life
was the whole gain (standalone Brier -3.0/1000 on 2020-24, z -2.45; -5.6/1000
on the untouched 2025 holdout). The ridge solver alone was -0.7/1000, noise.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import polars as pl

RIDGE_LAMBDA = 1000.0
WEEK_HALF_LIFE = 26.0


def ridge_ratings(
    df: pl.DataFrame,
    lam: float = RIDGE_LAMBDA,
    week_half_life: Optional[float] = WEEK_HALF_LIFE,
) -> dict[str, dict[str, float]]:
    """Return {team_abbrev: {"net": ..., "off": ..., "def": ...}}.

    `df` needs: posteam, defteam, home_team, epa, game_date (pl.Date). `def` is
    EPA allowed relative to league (lower = better); `net = off - def`.
    """
    if df.is_empty():
        return {}
    # nflverse ships game_date as a STRING; the weekly seed passes the frame
    # straight from load_pbp. Date arithmetic below needs pl.Date.
    if df.schema["game_date"] != pl.Date:
        df = df.with_columns(pl.col("game_date").cast(pl.Date))
    newest = df["game_date"].max()
    w = pl.lit(1.0)
    if week_half_life:
        age_w = (pl.lit(newest) - pl.col("game_date")).dt.total_days() / 7.0
        w = 0.5 ** (age_w / week_half_life)
    cells = (
        df.with_columns(w.alias("w"), (pl.col("posteam") == pl.col("home_team")).alias("home_off"))
        .group_by(["posteam", "defteam", "home_off"])
        .agg(pl.col("w").sum().alias("sw"), (pl.col("w") * pl.col("epa")).sum().alias("swy"))
    )
    teams = sorted(set(cells["posteam"].to_list()) | set(cells["defteam"].to_list()))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    p = 2 + 2 * n  # mu, h, off[n], dfa[n]
    X = np.zeros((cells.height, p))
    y = np.zeros(cells.height)
    sw = np.zeros(cells.height)
    for r, (pos, dfn, home, w_sum, wy) in enumerate(cells.iter_rows()):
        X[r, 0] = 1.0
        X[r, 1] = float(home)
        X[r, 2 + idx[pos]] = 1.0
        X[r, 2 + n + idx[dfn]] = 1.0
        sw[r] = w_sum
        y[r] = wy / w_sum
    pen = np.full(p, float(lam))
    pen[:2] = 1e-6  # mu and h are not shrunk
    A = (X * sw[:, None]).T @ X + np.diag(pen)
    beta = np.linalg.solve(A, (X * sw[:, None]).T @ y)
    off, dfa = beta[2:2 + n], beta[2 + n:]
    return {
        t: {"net": float(off[idx[t]] - dfa[idx[t]]), "off": float(off[idx[t]]), "def": float(dfa[idx[t]])}
        for t in teams
    }
