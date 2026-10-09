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

Projections are means: targets = team targets x share, receptions = targets x
catch rate, receiving yards = targets x yards/target, rushing yards = carries x
yards/carry, passing yards = team attempts x QB yards/attempt.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

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


@dataclass
class PlayerState:
    """Point-in-time usage and efficiency for every player seen before the cutoff."""
    usage: pd.DataFrame                 # index player_id: tgt_share, car_share, games, position, team
    fits: dict[str, RatingFit] = field(default_factory=dict)


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
    agg = a.groupby("player_id").agg(
        position=("position", "last"), games=("one", "sum"), w=("w", "sum"),
        wt_share=("wt_share", "sum"), wc_share=("wc_share", "sum"), w_tgt=("w_tgt", "sum"),
        w_rec=("w_rec", "sum"), w_ry=("w_ry", "sum"), w_car=("w_car", "sum"),
        w_rush=("w_rush", "sum"), w_att=("w_att", "sum"), w_py=("w_py", "sum"))
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

    k = cfg.share_prior_games
    p = agg["position"]
    agg["tgt_share"] = (agg["wt_share"] + k * p.map(pos_t)) / (agg["w"] + k)
    agg["car_share"] = (agg["wc_share"] + k * p.map(pos_c)) / (agg["w"] + k)
    agg["catch_rate"] = (agg["w_rec"] + cfg.k_targets * p.map(pos_cr)) / (agg["w_tgt"] + cfg.k_targets)
    agg["ypt"] = (agg["w_ry"] + cfg.k_targets * p.map(pos_ypt)) / (agg["w_tgt"] + cfg.k_targets)
    agg["ypc"] = (agg["w_rush"] + cfg.k_carries * p.map(pos_ypc).fillna(4.2)) / (agg["w_car"] + cfg.k_carries)
    agg["ypa"] = (agg["w_py"] + cfg.k_attempts * pos_ypa) / (agg["w_att"] + cfg.k_attempts)
    return PlayerState(usage=agg, fits=fits)


def project_players(state: PlayerState, roster: pd.DataFrame) -> pd.DataFrame:
    """Project each (player, team, opp, home, is_starting_qb) row of ``roster``.

    Players never seen before the cutoff are skipped (no basis for a projection).
    """
    u = state.usage
    r = roster[roster["player_id"].isin(u.index)].copy()
    if r.empty:
        return r.assign(proj_targets=[], proj_receptions=[], proj_receiving_yards=[],
                        proj_carries=[], proj_rushing_yards=[], proj_passing_yards=[])
    f = state.fits
    exp = {m: np.array([f[m].expect(t, o, h) for t, o, h in zip(r["team"], r["opp"], r["home"])])
           for m in ("team_targets", "team_carries", "team_attempts")}
    uu = u.loc[r["player_id"]]
    r["proj_targets"] = exp["team_targets"] * uu["tgt_share"].to_numpy()
    r["proj_receptions"] = r["proj_targets"] * uu["catch_rate"].to_numpy()
    r["proj_receiving_yards"] = r["proj_targets"] * uu["ypt"].to_numpy()
    r["proj_carries"] = exp["team_carries"] * uu["car_share"].to_numpy()
    r["proj_rushing_yards"] = r["proj_carries"] * uu["ypc"].to_numpy()
    r["proj_passing_yards"] = np.where(r["is_starting_qb"], exp["team_attempts"] * uu["ypa"].to_numpy(), 0.0)
    return r
