"""Project an upcoming NFL week with the walk-forward-validated game model.

Live inputs differ from the backtest in three ways, handled here:

* **Starting QB.** The backtest reads the first-dropback passer of the game
  itself. Live, the starter comes from (in order) an explicit override, the
  nflverse schedule's projected starter (``home_qb_id`` / ``away_qb_id``), or
  the team's most recent starter.
* **Wind.** Unplayed games have no recorded wind; outdoor games use the league
  median (a forecast feed is a follow-up — wind is where the totals gain lives).
* **Roof.** Retractable stadiums are listed with an empty roof until game day;
  they take the stadium's most common historical setting.

The combiner is trained on completed seasons before ``season`` and the ratings
on every game before the week's first kickoff — exactly the walk-forward setup.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from evmax.nfl_projections import data, team_games
from evmax.nfl_projections.game_model import (
    MARGIN_SD, TOTAL_SD, GameModelConfig, context_features, feature_table, fit_combiner,
    fit_ratings, side_features,
)
from evmax.nfl_projections.ratings import fit_qb_ratings


def next_week(games: pd.DataFrame, today: date) -> tuple[int, int]:
    """(season, week) of the earliest regular/post-season week with an unplayed game on or after today."""
    upcoming = games[games["home_score"].isna() & (games["gameday"].dt.date >= today)]
    if upcoming.empty:
        raise ValueError("no unplayed games on or after today in the schedule")
    first = upcoming.sort_values("gameday").iloc[0]
    return int(first["season"]), int(first["week"])


def infer_roof(games: pd.DataFrame, stadium_id: str, roof: str) -> str:
    """The game's roof, or the stadium's most common played roof when unknown ('')."""
    if roof:
        return roof
    hist = games[(games["stadium_id"] == stadium_id) & games["home_score"].notna() & (games["roof"] != "")]
    return hist["roof"].mode().iloc[0] if not hist.empty else "outdoors"


def latest_starters(tg: pd.DataFrame, cutoff: pd.Timestamp) -> dict[str, str]:
    """Each team's most recent pre-game starter (first-dropback passer) before ``cutoff``."""
    past = tg[(tg["gameday"] < cutoff) & tg["first_qb_id"].notna()].sort_values("gameday")
    return past.groupby("team")["first_qb_id"].last().to_dict()


def project_week(season: int, week: int, cfg: GameModelConfig = GameModelConfig(),
                 d: Optional[Path] = None, refresh: bool = True,
                 starters: Optional[dict[str, str]] = None) -> pd.DataFrame:
    """Projections for every game of ``season`` week ``week``.

    ``starters`` maps team abbreviation -> gsis QB id and overrides the
    schedule's projected starter. Returns one row per game, home perspective:
    proj_home / proj_away points, proj_margin (home - away), proj_total,
    p_home_win, the QB ids used, and the schedule's consensus lines (for
    comparison only — never a model input).
    """
    first_season = cfg.first_feature_season - 2  # ratings need two prior seasons
    if refresh:
        data.ensure_games(d)
        data.ensure_pbp(range(first_season, season + 1), d, refresh_seasons=[season])
    tg = team_games.load_team_games(range(first_season, season + 1), d)
    games = data.load_games(d)
    wk = games[(games["season"] == season) & (games["week"] == week)].copy()
    if wk.empty:
        raise ValueError(f"no games for season {season} week {week}")
    cutoff = wk["gameday"].min()

    comb = fit_combiner(feature_table(tg, games, list(range(cfg.first_feature_season, season)), cfg),
                        cfg.features)
    fits = fit_ratings(tg, cutoff, cfg)
    qbr = fit_qb_ratings(tg, cutoff)
    recent = latest_starters(tg, cutoff)
    overrides = starters or {}

    rows = []
    for g in wk.itertuples():
        g = g._replace(roof=infer_roof(games, g.stadium_id, g.roof or ""))
        ctx = context_features(g)
        neutral = g.location == "Neutral"
        sides = {}
        for team, opp, home, qb_col in ((g.home_team, g.away_team, 0 if neutral else 1, "home_qb_id"),
                                         (g.away_team, g.home_team, 0, "away_qb_id")):
            sched_qb = getattr(g, qb_col)
            qb = overrides.get(team) or (sched_qb if isinstance(sched_qb, str) and sched_qb else None) \
                or recent.get(team)
            feats = {**side_features(fits, team, opp, home), **ctx, "qb": qbr.delta(team, qb)}
            sides[team] = (comb.predict(feats), qb)
        (ph, hq), (pa, aq) = sides[g.home_team], sides[g.away_team]
        rows.append({
            "game_id": g.game_id, "gameday": g.gameday.date(), "gametime": g.gametime,
            "home_team": g.home_team, "away_team": g.away_team, "neutral": bool(neutral),
            "roof": g.roof, "proj_home": ph, "proj_away": pa, "proj_margin": ph - pa,
            "proj_total": ph + pa, "p_home_win": float(stats.norm.cdf((ph - pa) / MARGIN_SD)),
            "home_qb_id": hq, "away_qb_id": aq,
            "home_qb_name": g.home_qb_name, "away_qb_name": g.away_qb_name,
            "market_spread_line": g.spread_line, "market_total_line": g.total_line,
        })
    return pd.DataFrame(rows)


def total_over_probability(proj_total: float, line: float) -> float:
    """P(total > line) under the model's total distribution (Normal, SD from walk-forward residuals)."""
    return float(stats.norm.sf(line, loc=proj_total, scale=TOTAL_SD))


def home_cover_probability(proj_margin: float, home_handicap: float) -> Optional[float]:
    """P(home margin + handicap > 0), e.g. handicap -3.5 = home laying 3.5.

    Uses the key-number margin PMF (evmax.models_ml.spread_pmf) located at the
    projected margin; None if the artifact is unavailable. Pushes (integer
    handicaps) are excluded from both sides: P(cover | no push).
    """
    from evmax.models_ml.spread_pmf import load_margin_pmf

    pmf = load_margin_pmf("nfl")
    if pmf is None:
        return None
    fav_home = proj_margin >= 0
    p = pmf.pmf(abs(proj_margin))  # favorite-margin distribution over pmf.ks
    home_margin = pmf.ks if fav_home else -pmf.ks
    net = home_margin + home_handicap
    win, lose = float(p[net > 0].sum()), float(p[net < 0].sum())
    return win / (win + lose) if (win + lose) > 0 else None
