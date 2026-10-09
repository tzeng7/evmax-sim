"""Player stat-line projections: team volume x player share x player efficiency.

At a cutoff (a week's first kickoff), using only games before it:

* **Team volume** — opponent-adjusted ridge ratings (``ratings.fit_rating``)
  for team targets, carries and pass attempts per game.
* **Usage** — each player's recency-weighted share of his team's targets and
  carries in games he played, shrunk toward a position prior. Volume and share
  are the stable parts of a player's line.
* **Efficiency** — catch rate, yards per target, yards per carry and the
  starting QB's yards per attempt, recency-weighted and shrunk hard toward
  position means (per-target/per-carry efficiency is mostly noise).

Means: targets = team targets x share, receptions = targets x
catch rate, receiving yards = targets x yards/target, rushing yards = carries x
yards/carry, passing yards = team attempts x the starter's share of team
attempts x his yards/attempt (both over his starts). The reported ``proj_*``
point projections are MEDIANS (MAE-optimal for right-skewed stats) from a Gamma
(yardage) / negative binomial (receptions) whose dispersion is fitted at each
cutoff from past games only; the means are kept as ``mean_*``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from scipy import stats

from evmax.nfl_projections.ratings import RatingFit, fit_rating, recency_weights

SKILL = ("WR", "TE", "RB", "QB")


@dataclass(frozen=True)
class PlayerModelConfig:
    half_life_days: float = 70.0
    offseason_days: float = 180.0
    lookback_days: int = 730
    lam: float = 4.0
    share_prior_games: float = 3.0     # pseudo-games at the position-mean share
    k_targets: float = 60.0            # pseudo-targets for catch rate / yards per target
    k_carries: float = 80.0            # pseudo-carries for yards per carry
    k_attempts: float = 150.0          # pseudo-attempts for QB yards per attempt
    k_team_attempts: float = 100.0     # pseudo team-attempts for a starter's attempt share
    min_start_attempts: float = 10.0   # a game counts as a start at >= this many attempts


@dataclass
class PlayerState:
    """Point-in-time usage and efficiency for every player seen before the cutoff."""
    usage: pd.DataFrame                 # index player_id: tgt_share, car_share, games, position, team
    fits: dict[str, RatingFit] = field(default_factory=dict)
    # Distribution shape fitted from past games (leak-free): Gamma Var = theta * mean
    # for yardage, NegBin Var = mu + mu^2 / k for receptions.
    theta: dict[str, float] = field(default_factory=dict)
    negbin_k: float = 5.0


def team_volume_rows(player_games: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    """Team-game rows with team targets / carries / attempts and the home flag."""
    tv = player_games.groupby(["game_id", "team"]).agg(
        team_targets=("team_targets", "first"), team_carries=("team_carries", "first"),
        team_attempts=("team_attempts", "first")).reset_index()
    return tv.merge(team_games[["game_id", "team", "opp", "home", "gameday"]], on=["game_id", "team"])


def fit_player_state(player_games: pd.DataFrame, volume_rows: pd.DataFrame, cutoff: pd.Timestamp,
                     cfg: PlayerModelConfig = PlayerModelConfig()) -> PlayerState:
    lo = cutoff - pd.Timedelta(days=cfg.lookback_days)
    vr = volume_rows[(volume_rows["gameday"] < cutoff) & (volume_rows["gameday"] >= lo)]
    fits = {m: fit_rating(vr, m, cutoff, cfg.half_life_days, cfg.lam, offseason_days=cfg.offseason_days)
            for m in ("team_targets", "team_carries", "team_attempts")}

    pg = player_games[(player_games["gameday"] < cutoff) & (player_games["gameday"] >= lo)]
    pg = pg[pg["position"].isin(SKILL)]
    w = recency_weights(pg["gameday"], cutoff, cfg.half_life_days, cfg.offseason_days)
    t_share = np.where(pg["team_targets"] > 0, pg["targets"] / pg["team_targets"].clip(lower=1), 0.0)
    c_share = np.where(pg["team_carries"] > 0, pg["carries"] / pg["team_carries"].clip(lower=1), 0.0)
    a = pd.DataFrame({
        "player_id": pg["player_id"].to_numpy(), "position": pg["position"].to_numpy(),
        "w": w, "wt_share": w * t_share, "wc_share": w * c_share,
        "w_tgt": w * pg["targets"].to_numpy(), "w_rec": w * pg["receptions"].to_numpy(),
        "w_ry": w * pg["receiving_yards"].to_numpy(), "w_car": w * pg["carries"].to_numpy(),
        "w_rush": w * pg["rushing_yards"].to_numpy(), "w_att": w * pg["attempts"].to_numpy(),
        "w_py": w * pg["passing_yards"].to_numpy(), "one": 1.0,
    })
    # QB starts only: a starter's share of team attempts and yards per attempt.
    start = (pg["attempts"] >= cfg.min_start_attempts).to_numpy()
    a["ws_att"] = np.where(start, w * pg["attempts"].to_numpy(), 0.0)
    a["ws_tatt"] = np.where(start, w * pg["team_attempts"].to_numpy(), 0.0)
    a["ws_py"] = np.where(start, w * pg["passing_yards"].to_numpy(), 0.0)
    agg = a.groupby("player_id").agg(
        position=("position", "last"), games=("one", "sum"), w=("w", "sum"),
        wt_share=("wt_share", "sum"), wc_share=("wc_share", "sum"), w_tgt=("w_tgt", "sum"),
        w_rec=("w_rec", "sum"), w_ry=("w_ry", "sum"), w_car=("w_car", "sum"),
        w_rush=("w_rush", "sum"), w_att=("w_att", "sum"), w_py=("w_py", "sum"),
        ws_att=("ws_att", "sum"), ws_tatt=("ws_tatt", "sum"), ws_py=("ws_py", "sum"))
    # Position priors (unweighted league means over the window).
    pos = a.groupby("position").agg(t=("wt_share", "sum"), c=("wc_share", "sum"), w=("w", "sum"),
                                     tgt=("w_tgt", "sum"), rec=("w_rec", "sum"), ry=("w_ry", "sum"),
                                     car=("w_car", "sum"), rush=("w_rush", "sum"),
                                     att=("w_att", "sum"), py=("w_py", "sum"))
    pos_t = (pos["t"] / pos["w"]).to_dict()
    pos_c = (pos["c"] / pos["w"]).to_dict()
    pos_cr = (pos["rec"] / pos["tgt"].clip(lower=1e-9)).to_dict()
    pos_ypt = (pos["ry"] / pos["tgt"].clip(lower=1e-9)).to_dict()
    pos_ypc = (pos["rush"] / pos["car"].clip(lower=1e-9)).to_dict()
    pos_ypa = float(pos.loc["QB", "py"] / pos.loc["QB", "att"]) if "QB" in pos.index else 6.5
    starts_share = float(a["ws_att"].sum() / max(a["ws_tatt"].sum(), 1e-9))
    starts_ypa = float(a["ws_py"].sum() / max(a["ws_att"].sum(), 1e-9))

    k = cfg.share_prior_games
    p = agg["position"]
    agg["tgt_share"] = (agg["wt_share"] + k * p.map(pos_t)) / (agg["w"] + k)
    agg["car_share"] = (agg["wc_share"] + k * p.map(pos_c)) / (agg["w"] + k)
    agg["catch_rate"] = (agg["w_rec"] + cfg.k_targets * p.map(pos_cr)) / (agg["w_tgt"] + cfg.k_targets)
    agg["ypt"] = (agg["w_ry"] + cfg.k_targets * p.map(pos_ypt)) / (agg["w_tgt"] + cfg.k_targets)
    agg["ypc"] = (agg["w_rush"] + cfg.k_carries * p.map(pos_ypc).fillna(4.2)) / (agg["w_car"] + cfg.k_carries)
    agg["att_share"] = ((agg["ws_att"] + cfg.k_team_attempts * starts_share)
                        / (agg["ws_tatt"] + cfg.k_team_attempts))
    agg["ypa"] = (agg["ws_py"] + cfg.k_attempts * starts_ypa) / (agg["ws_att"] + cfg.k_attempts)
    return PlayerState(usage=agg, fits=fits, theta=fit_dispersion(pg), negbin_k=fit_negbin_k(pg))


MIN_GAMES_FOR_DISPERSION = 6
DEFAULT_THETA = {"receiving_yards": 25.0, "rushing_yards": 15.0}   # only if a window has no data


def _player_moments(pg: pd.DataFrame, stat: str, min_mean: float) -> pd.DataFrame:
    m = pg.groupby("player_id")[stat].agg(["mean", "var", "size"])
    return m[(m["size"] >= MIN_GAMES_FOR_DISPERSION) & (m["mean"] >= min_mean)]


def fit_dispersion(pg: pd.DataFrame) -> dict[str, float]:
    """Gamma scale theta (Var = theta * mean) per yardage stat, pooled over players
    with enough games in the window (game-weighted method of moments)."""
    out = {}
    for stat, floor in (("receiving_yards", 5.0), ("rushing_yards", 5.0)):
        m = _player_moments(pg, stat, floor)
        out[stat] = (float((m["var"] * m["size"]).sum() / (m["mean"] * m["size"]).sum())
                     if len(m) else DEFAULT_THETA[stat])
    return out


def fit_negbin_k(pg: pd.DataFrame) -> float:
    """NegBin size k for receptions (Var = mu + mu^2 / k), pooled method of moments."""
    m = _player_moments(pg, "receptions", 0.5)
    excess = ((m["var"] - m["mean"]) * m["size"]).sum()
    if not len(m) or excess <= 0:
        return 50.0
    return float(max((m["mean"] ** 2 * m["size"]).sum() / excess, 1.0))


def gamma_median(mean: np.ndarray, theta: float) -> np.ndarray:
    mean = np.asarray(mean, dtype=float)
    out = np.zeros_like(mean)
    pos = mean > 0
    out[pos] = stats.gamma.ppf(0.5, a=mean[pos] / theta, scale=theta)
    return out


def negbin_median(mean: np.ndarray, k: float) -> np.ndarray:
    mean = np.asarray(mean, dtype=float)
    out = np.zeros_like(mean)
    pos = mean > 0
    out[pos] = stats.nbinom.ppf(0.5, n=k, p=k / (k + mean[pos]))
    return out


def project_players(state: PlayerState, roster: pd.DataFrame) -> pd.DataFrame:
    """Project each (player, team, opp, home, is_starting_qb) row of ``roster``.

    Players never seen before the cutoff are skipped (no basis for a projection).
    """
    u = state.usage
    r = roster[roster["player_id"].isin(u.index)].copy()
    if r.empty:
        return r.assign(proj_targets=[], proj_receptions=[], proj_receiving_yards=[],
                        proj_carries=[], proj_rushing_yards=[], proj_passing_yards=[],
                        mean_receptions=[], mean_receiving_yards=[], mean_rushing_yards=[])
    f = state.fits
    exp = {m: np.array([f[m].expect(t, o, h) for t, o, h in zip(r["team"], r["opp"], r["home"])])
           for m in ("team_targets", "team_carries", "team_attempts")}
    uu = u.loc[r["player_id"]]
    r["proj_targets"] = exp["team_targets"] * uu["tgt_share"].to_numpy()
    r["proj_receptions"] = r["proj_targets"] * uu["catch_rate"].to_numpy()
    r["proj_receiving_yards"] = r["proj_targets"] * uu["ypt"].to_numpy()
    r["proj_carries"] = exp["team_carries"] * uu["car_share"].to_numpy()
    r["proj_rushing_yards"] = r["proj_carries"] * uu["ypc"].to_numpy()
    r["proj_passing_yards"] = np.where(
        r["is_starting_qb"], exp["team_attempts"] * uu["att_share"].to_numpy() * uu["ypa"].to_numpy(), 0.0)
    # Means above; the reported point projection is the MEDIAN (MAE-optimal for
    # skewed stats). Passing yards are near-symmetric for starters: median = mean.
    for stat in ("receptions", "receiving_yards", "rushing_yards"):
        r[f"mean_{stat}"] = r[f"proj_{stat}"]
    theta = {**DEFAULT_THETA, **state.theta}
    r["proj_receiving_yards"] = gamma_median(r["mean_receiving_yards"], theta["receiving_yards"])
    r["proj_rushing_yards"] = gamma_median(r["mean_rushing_yards"], theta["rushing_yards"])
    r["proj_receptions"] = negbin_median(r["mean_receptions"], state.negbin_k)
    return r
