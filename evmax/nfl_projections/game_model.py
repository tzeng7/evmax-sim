"""Game projections: each team's points -> margin, total, P(home win).

``project_games`` fits the point-in-time ratings once for a cutoff and projects
every game on or after it. Margins and totals come from the two projected team
scores, so they are always consistent (margin = home - away, total = sum).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from evmax.nfl_projections.ratings import RatingFit, fit_rating

MARGIN_SD = 13.5  # Stern-style margin noise (1978-2012 re-estimate 13.45)


@dataclass(frozen=True)
class GameModelConfig:
    half_life_days: float = 70.0
    lam: float = 4.0
    lookback_days: int = 730


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
        "points": fit_rating(window, "points_for", cutoff, cfg.half_life_days, cfg.lam),
    }


def project_games(team_games: pd.DataFrame, games: pd.DataFrame, cutoff: pd.Timestamp,
                  cfg: GameModelConfig = GameModelConfig()) -> list[GameProjection]:
    """Project ``games`` (schedule rows) with ratings fit on team-games before ``cutoff``."""
    fits = fit_ratings(team_games, cutoff, cfg)
    pts = fits["points"]
    out = []
    for g in games.itertuples():
        neutral = int(g.location == "Neutral")
        h_home = 0 if neutral else 1
        out.append(GameProjection(
            game_id=g.game_id, home_team=g.home_team, away_team=g.away_team,
            home_points=pts.expect(g.home_team, g.away_team, h_home),
            away_points=pts.expect(g.away_team, g.home_team, 0),
        ))
    return out


def walk_forward(team_games: pd.DataFrame, games: pd.DataFrame, seasons: list[int],
                 cfg: GameModelConfig = GameModelConfig()) -> pd.DataFrame:
    """Project every completed game in ``seasons`` week by week, leak-free.

    For each (season, week) the cutoff is that week's first gameday, so a
    Thursday game and the following Monday game share one rating snapshot.
    """
    sched = games[games["season"].isin(seasons) & games["home_score"].notna()]
    rows = []
    for (season, week), wk in sched.groupby(["season", "week"], sort=True):
        cutoff = wk["gameday"].min()
        for p in project_games(team_games, wk, cutoff, cfg):
            rows.append({"game_id": p.game_id, "season": season, "week": week,
                         "proj_home": p.home_points, "proj_away": p.away_points,
                         "proj_margin": p.margin, "proj_total": p.total, "p_home": p.p_home_win})
    proj = pd.DataFrame(rows)
    keep = ["game_id", "game_type", "home_team", "away_team", "home_score", "away_score",
            "spread_line", "total_line", "location", "roof", "wind", "temp", "div_game"]
    return proj.merge(games[keep], on="game_id", how="left")
