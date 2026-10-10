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

from evmax.nfl_projections.ratings import QBRatings, RatingFit, fit_qb_ratings, fit_rating

MARGIN_SD = 13.5  # Stern-style margin noise; walk-forward residual SD 11.8-14.1 (2016-25), Brier flat 13.0-13.5
TOTAL_SD = 13.4   # walk-forward total residual SD (2016-25 seasons: 12.7-14.6)

# feature name -> (team-game metric, per-play sample-size weight column or None)
RATED_METRICS: dict[str, tuple[str, str | None]] = {
    "pts": ("points_for", None),
    "epa": ("epa_pp", "comp_plays"),
    "sr": ("sr", "comp_plays"),
}


# Game-context features (same value for both sides of a game). Outdoor games
# with no recorded wind get the league median.
MEDIAN_OUTDOOR_WIND_MPH = 7.0


def context_features(g) -> dict[str, float]:
    """Roof / weather features for one schedule row (namedtuple from itertuples)."""
    dome = 1.0 if g.roof in ("dome", "closed") else 0.0
    wind = 0.0 if dome else (float(g.wind) if pd.notna(g.wind) else MEDIAN_OUTDOOR_WIND_MPH)
    return {"dome": dome, "wind": wind}


@dataclass(frozen=True)
class GameModelConfig:
    half_life_days: float = 70.0
    lam: float = 4.0
    lookback_days: int = 730
    offseason_days: float = 180.0  # offseason not counted as rating decay
    features: tuple[str, ...] = ("pts", "epa", "sr", "dome", "wind", "qb")
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
        name: fit_rating(window, metric, cutoff, cfg.half_life_days, cfg.lam, weight_col=wcol,
                         offseason_days=cfg.offseason_days)
        for name, (metric, wcol) in RATED_METRICS.items() if name in cfg.features
    }


DRIVE_SIMS = 2000


def drive_points(team_games: pd.DataFrame, games: pd.DataFrame, cutoff: pd.Timestamp,
                 week_games: pd.DataFrame) -> dict[tuple[str, str], float]:
    """(game_id, team) -> mean simulated points from the drive simulator (``drive_model``),
    fitted on drives before ``cutoff``. Each game uses its own fixed seed (deterministic)."""
    import zlib

    from evmax.nfl_projections import drive_model

    seasons = range(int(team_games["season"].min()), int(team_games["season"].max()) + 1)
    drives = drive_model.load_drives(seasons)
    gameday = games.set_index("game_id")["gameday"]
    state = drive_model.fit_drive_state(drives, gameday, cutoff)
    out = {}
    for g in week_games.itertuples():
        rng = np.random.default_rng(zlib.crc32(str(g.game_id).encode()))
        h, a = drive_model.simulate_game(state, g.home_team, g.away_team, neutral=g.location == "Neutral",
                                         n=DRIVE_SIMS, rng=rng)
        out[(g.game_id, g.home_team)] = float(h.mean())
        out[(g.game_id, g.away_team)] = float(a.mean())
    return out


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
    # Pre-game starter of each (game, team): the first-dropback passer. Only the
    # starter's IDENTITY is read from the projected game, never its stats.
    starters = team_games.set_index(["game_id", "team"])["first_qb_id"].to_dict()
    rows = []
    for (season, week), wk in sched.groupby(["season", "week"], sort=True):
        cutoff = wk["gameday"].min()
        fits = fit_ratings(team_games, cutoff, cfg)
        qbr = fit_qb_ratings(team_games, cutoff) if "qb" in cfg.features else None
        dpts = drive_points(team_games, games, cutoff, wk) if "drive" in cfg.features else {}
        for g in wk.itertuples():
            ctx = context_features(g)
            for team, opp, home, side in _sides(g):
                feats = {**side_features(fits, team, opp, home), **ctx}
                if qbr is not None:
                    feats["qb"] = qbr.delta(team, starters.get((g.game_id, team)))
                if dpts:
                    feats["drive"] = dpts.get((g.game_id, team))
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
