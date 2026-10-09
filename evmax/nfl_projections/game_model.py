"""Game projections: each team's points -> margin, total, P(home win).

Two stages:

1. **Ratings** (``fit_ratings``): point-in-time opponent-adjusted ridge ratings
   for several metrics. For a team on offense against an opponent, each fit's
   ``expect`` is one *feature* (expected points, expected EPA/play, expected
   success rate, ...).
2. **Combiner** (``fit_combiner``): an OLS map from those features to the
   team's points, trained ONLY on games from earlier seasons (walk-forward).

Margins and totals come from the two projected team scores, so they are always
consistent (margin = home - away, total = sum).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from evmax.nfl_projections.ratings import RatingFit, fit_rating

MARGIN_SD = 13.5  # Stern-style margin noise (1978-2012 re-estimate 13.45)

# feature name -> (team-game metric, per-play sample-size weight column or None)
RATED_METRICS: dict[str, tuple[str, str | None]] = {
    "pts": ("points_for", None),
    "epa": ("epa_pp", "comp_plays"),
    "sr": ("sr", "comp_plays"),
}


@dataclass(frozen=True)
class GameModelConfig:
    half_life_days: float = 70.0
    lam: float = 4.0
    lookback_days: int = 730
    features: tuple[str, ...] = ("pts", "epa", "sr")
    first_feature_season: int = 2015  # combiner training starts here


@dataclass(frozen=True)
class Combiner:
    """points = coef[0] + sum(coef[i] * feature_i)."""
    features: tuple[str, ...]
    coef: tuple[float, ...]

    def predict(self, feats: dict[str, float]) -> float:
        return self.coef[0] + sum(c * feats[f] for c, f in zip(self.coef[1:], self.features))


@dataclass(frozen=True)
class GameProjection:
    game_id: str
    home_team: str
    away_team: str
    home_points: float
    away_points: float

    @property
    def margin(self) -> float:  # home - away
        return self.home_points - self.away_points

    @property
    def total(self) -> float:
        return self.home_points + self.away_points

    @property
    def p_home_win(self) -> float:
        return float(stats.norm.cdf(self.margin / MARGIN_SD))


def fit_ratings(team_games: pd.DataFrame, cutoff: pd.Timestamp,
                cfg: GameModelConfig = GameModelConfig()) -> dict[str, RatingFit]:
    window = team_games[(team_games["gameday"] < cutoff)
                        & (team_games["gameday"] >= cutoff - pd.Timedelta(days=cfg.lookback_days))]
    return {
        name: fit_rating(window, metric, cutoff, cfg.half_life_days, cfg.lam, weight_col=wcol)
        for name, (metric, wcol) in RATED_METRICS.items() if name in cfg.features
    }


def side_features(fits: dict[str, RatingFit], team: str, opp: str, home: int) -> dict[str, float]:
    return {name: fit.expect(team, opp, home) for name, fit in fits.items()}


def _sides(g) -> list[tuple[str, str, int, str]]:
    """(team, opp, home flag, side) for both teams of a schedule row."""
    h = 0 if g.location == "Neutral" else 1
    return [(g.home_team, g.away_team, h, "home"), (g.away_team, g.home_team, 0, "away")]


def feature_table(team_games: pd.DataFrame, games: pd.DataFrame, seasons: list[int],
                  cfg: GameModelConfig = GameModelConfig()) -> pd.DataFrame:
    """One row per (game, side) with point-in-time features and the actual points.

    Each (season, week) uses one rating snapshot cut at that week's first gameday.
    """
    sched = games[games["season"].isin(seasons) & games["home_score"].notna()]
    rows = []
    for (season, week), wk in sched.groupby(["season", "week"], sort=True):
        fits = fit_ratings(team_games, wk["gameday"].min(), cfg)
        for g in wk.itertuples():
            for team, opp, home, side in _sides(g):
                feats = side_features(fits, team, opp, home)
                points = g.home_score if side == "home" else g.away_score
                rows.append({"game_id": g.game_id, "season": season, "week": week, "side": side,
                             "team": team, "opp": opp, "home": home, "points": points, **feats})
    return pd.DataFrame(rows)


def fit_combiner(rows: pd.DataFrame, features: tuple[str, ...]) -> Combiner:
    r = rows.dropna(subset=list(features) + ["points"])
    X = np.column_stack([np.ones(len(r))] + [r[f].to_numpy(dtype=float) for f in features])
    coef = np.linalg.lstsq(X, r["points"].to_numpy(dtype=float), rcond=None)[0]
    return Combiner(features=features, coef=tuple(float(c) for c in coef))


def walk_forward(team_games: pd.DataFrame, games: pd.DataFrame, seasons: list[int],
                 cfg: GameModelConfig = GameModelConfig()) -> pd.DataFrame:
    """Project every completed game in ``seasons``, leak-free.

    Ratings are point-in-time per week; the combiner for season ``s`` is
    trained on feature rows from seasons ``first_feature_season .. s-1``.
    """
    feat_seasons = list(range(cfg.first_feature_season, max(seasons) + 1))
    ft = feature_table(team_games, games, feat_seasons, cfg)
    out = []
    for s in seasons:
        comb = fit_combiner(ft[ft["season"] < s], cfg.features)
        cur = ft[ft["season"] == s].copy()
        cur["proj"] = [comb.predict(r) for r in cur[list(cfg.features)].to_dict("records")]
        wide = cur.pivot_table(index=["game_id", "season", "week"], columns="side", values="proj").reset_index()
        out.append(wide)
    proj = pd.concat(out, ignore_index=True).rename(columns={"home": "proj_home", "away": "proj_away"})
    proj["proj_margin"] = proj["proj_home"] - proj["proj_away"]
    proj["proj_total"] = proj["proj_home"] + proj["proj_away"]
    proj["p_home"] = stats.norm.cdf(proj["proj_margin"] / MARGIN_SD)
    keep = ["game_id", "game_type", "home_team", "away_team", "home_score", "away_score",
            "spread_line", "total_line", "location", "roof", "wind", "temp", "div_game"]
    return proj.merge(games[keep], on="game_id", how="left")
