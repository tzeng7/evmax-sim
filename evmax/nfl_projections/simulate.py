"""Joint Monte Carlo of one NFL game's player stat lines (top-down, vectorized).

``player_model.project_players`` projects each player on his own; their
medians need not add up (receivers' yards vs the QB's passing yards). This
module draws whole team box scores, so the sum identities hold by
construction and teammates' lines are correlated the way games are:

1. Team targets and carries: Poisson around the projected team volume, mixed
   with a Gamma game factor (``team_vol_cv``).
2. Allocation: a Dirichlet draw of each game's shares around the projected
   shares (concentration ``target_kappa`` / ``carry_kappa``), then a multinomial
   split. Shares not owned by a projected player go to an "other" slot.
3. Catches: binomial on targets with the player's catch rate.
4. Yards: per-catch Gamma (shape ``rec_shape``) and per-carry Normal
   (``rush_sd``), both scaled by a team efficiency factor shared by every
   player of the team in that game (lognormal, ``pass_eff_sd`` / ``rush_eff_sd``).
5. The starting QB's passing yards = the sum of his receivers' yards (exact).
6. Touchdowns: team rushing / receiving TDs are Poisson around the projection,
   scaled by the same efficiency factor, and split by expected-TD share. The
   QB's passing TDs = the team's receiving TDs.

Parameters come from ``fit_sim_params`` (method of moments on past
player-games). Validation: ``scripts/eval_nfl_joint_sim.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimParams:
    target_kappa: float = 20.0
    carry_kappa: float = 20.0
    team_vol_cv: float = 0.15
    rec_shape: float = 1.0
    rush_sd: float = 6.0
    pass_eff_sd: float = 0.15
    rush_eff_sd: float = 0.15


MIN_GAMES = 6


def _kappa(count: np.ndarray, team: np.ndarray, share: np.ndarray) -> float:
    """Dirichlet-multinomial concentration from extra-binomial share variance."""
    ok = (team > 1) & (share > 0) & (share < 1)
    c, t, s = count[ok], team[ok], share[ok]
    if not ok.any():
        return SimParams.target_kappa
    phi = float(((c - t * s) ** 2).sum() / (t * s * (1 - s)).sum())
    if phi <= 1.0:
        return 1e4
    return max(float((t.mean() - 1) / (phi - 1) - 1), 1.0)


def fit_sim_params(player_games: pd.DataFrame) -> SimParams:
    """Method-of-moments fit on ``player_games`` (use past seasons only).

    Each player-season's own mean share / rate is the reference, so only
    players with >= MIN_GAMES games in a season enter.
    """
    pg = player_games[player_games["position"].isin(["WR", "TE", "RB", "QB"])].copy()
    pg["ps"] = pg["season"].astype(str) + "|" + pg["player_id"].astype(str)
    n = pg.groupby("ps")["game_id"].transform("size")
    pg = pg[n >= MIN_GAMES]
    g = pg.groupby("ps")
    s_t = g["targets"].transform("sum") / g["team_targets"].transform("sum").clip(lower=1)
    s_c = g["carries"].transform("sum") / g["team_carries"].transform("sum").clip(lower=1)
    target_kappa = _kappa(pg["targets"].to_numpy(float), pg["team_targets"].to_numpy(float), s_t.to_numpy())
    carry_kappa = _kappa(pg["carries"].to_numpy(float), pg["team_carries"].to_numpy(float), s_c.to_numpy())

    tv = player_games.groupby(["season", "team", "game_id"])[["team_targets", "team_carries"]].first()
    cvs = []
    for col in ("team_targets", "team_carries"):
        m = tv.groupby(["season", "team"])[col].transform("mean")
        cvs.append(max(float((((tv[col] - m) ** 2).mean() - m.mean()) / (m ** 2).mean()), 0.0))
    team_vol_cv = float(np.sqrt(np.mean(cvs)))

    # Receiving: expected yards given catches = catches x the player-season's yards per catch.
    ypr = g["receiving_yards"].transform("sum") / g["receptions"].transform("sum").clip(lower=1)
    mu = pg["receptions"] * ypr
    r = pg["receiving_yards"] - mu
    pass_var, rec_shape = _shared_factor(pg, mu, r, per_unit=pg["receptions"], unit_scale=ypr)
    ypc = g["rushing_yards"].transform("sum") / g["carries"].transform("sum").clip(lower=1)
    mu_r = pg["carries"] * ypc
    rr = pg["rushing_yards"] - mu_r
    rush_var, rush_k = _shared_factor(pg, mu_r, rr, per_unit=pg["carries"], unit_scale=None)
    return SimParams(target_kappa=target_kappa, carry_kappa=carry_kappa, team_vol_cv=team_vol_cv,
                     rec_shape=rec_shape, rush_sd=float(np.sqrt(rush_k)), pass_eff_sd=float(np.sqrt(pass_var)),
                     rush_eff_sd=float(np.sqrt(rush_var)))


def _shared_factor(pg: pd.DataFrame, mu: pd.Series, r: pd.Series, per_unit: pd.Series,
                   unit_scale: pd.Series | None) -> tuple[float, float]:
    """Split residual variance into a team-game factor and per-unit noise.

    Shared factor: cov(r_i, r_j) = mu_i mu_j s2 for teammates i != j in a game.
    Per-unit: the remaining variance per catch (Gamma shape, when ``unit_scale``
    = yards per catch) or per carry (variance, when None).
    """
    d = pd.DataFrame({"k": pg["game_id"] + "|" + pg["team"], "mu": mu, "r": r, "n": per_unit})
    d = d[d["n"] > 0]
    if d.empty:
        return 0.0, (1.0 if unit_scale is not None else SimParams.rush_sd ** 2)
    gsum = d.groupby("k")[["mu", "r"]].sum()
    gsq = d.assign(mu2=d["mu"] ** 2, r2=d["r"] ** 2, mr=d["mu"] * d["r"]).groupby("k")[["mu2", "r2"]].sum()
    pair_rr = (gsum["r"] ** 2 - gsq["r2"]).sum() / 2
    pair_mm = (gsum["mu"] ** 2 - gsq["mu2"]).sum() / 2
    s2 = max(float(pair_rr / pair_mm), 0.0) if pair_mm > 0 else 0.0
    resid = (d["r"] ** 2 - s2 * d["mu"] ** 2).clip(lower=0)
    if unit_scale is not None:
        scale = unit_scale.reindex(d.index)
        shape = float((d["n"] * scale ** 2).sum() / max(resid.sum(), 1e-9))   # Var = n ypr^2 / k
        return s2, max(shape, 0.05)
    return s2, max(float(resid.sum() / d["n"].sum()), 1.0)                     # Var per carry


def simulate_team(players: pd.DataFrame, team_targets: float, team_carries: float, team_rush_tds: float,
                  team_rec_tds: float, params: SimParams, n: int = 5000,
                  rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
    """Simulate one team-game ``n`` times.

    ``players``: one row per projected player with ``player_id``, ``tgt_share``,
    ``car_share``, ``catch_rate``, ``ypt``, ``ypc``, ``rush_xtd_share``,
    ``rec_xtd_share``, ``is_starting_qb``. Returns arrays of shape (n, players)
    for ``receptions``, ``receiving_yards``, ``rushing_yards``, ``passing_yards``,
    ``tds``, ``passing_tds`` (column order = ``players`` rows).
    """
    rng = rng or np.random.default_rng(0)
    k = len(players)
    ts = players["tgt_share"].to_numpy(float).clip(0, None)
    cs = players["car_share"].to_numpy(float).clip(0, None)
    # "other" slot keeps unprojected players' usage out of the projected players.
    t_other = max(1.0 - ts.sum(), 0.02)
    c_other = max(1.0 - cs.sum(), 0.02)
    tvec = np.append(ts, t_other); tvec = tvec / tvec.sum()
    cvec = np.append(cs, c_other); cvec = cvec / cvec.sum()

    cv2 = params.team_vol_cv ** 2
    gfac = rng.gamma(1 / cv2, cv2, size=(n, 2)) if cv2 > 0 else np.ones((n, 2))
    T = rng.poisson(team_targets * gfac[:, 0])
    C = rng.poisson(team_carries * gfac[:, 1])
    pt = rng.dirichlet(np.maximum(params.target_kappa * tvec, 1e-6), size=n)
    pc = rng.dirichlet(np.maximum(params.carry_kappa * cvec, 1e-6), size=n)
    tgt = rng.multinomial(T, pt)[:, :k]
    car = rng.multinomial(C, pc)[:, :k]

    cr = players["catch_rate"].to_numpy(float).clip(0.01, 0.99)
    rec = rng.binomial(tgt, cr)
    ypr = (players["ypt"].to_numpy(float) / cr).clip(0.5, None)
    e_pass = np.exp(rng.normal(-params.pass_eff_sd ** 2 / 2, params.pass_eff_sd, size=(n, 1)))
    e_rush = np.exp(rng.normal(-params.rush_eff_sd ** 2 / 2, params.rush_eff_sd, size=(n, 1)))
    shape = rec * params.rec_shape
    rec_yds = np.where(rec > 0, rng.gamma(np.maximum(shape, 1e-9), ypr / params.rec_shape) * e_pass, 0.0)
    ypc = players["ypc"].to_numpy(float)
    rush_yds = car * ypc * e_rush + rng.normal(0.0, 1.0, size=(n, k)) * np.sqrt(car) * params.rush_sd

    # "other" receivers' yards also count toward the QB: same team factor, league-like rate.
    other_t = T - tgt.sum(axis=1)
    other_yds = other_t * float(np.average(players["ypt"], weights=np.maximum(ts, 1e-9))) * e_pass[:, 0]
    team_pass = rec_yds.sum(axis=1) + other_yds
    qb = players["is_starting_qb"].to_numpy(bool)
    pass_yds = np.where(qb[None, :], team_pass[:, None], 0.0)

    rs = players["rush_xtd_share"].to_numpy(float).clip(0, None)
    xs = players["rec_xtd_share"].to_numpy(float).clip(0, None)
    rvec = np.append(rs, max(1 - rs.sum(), 0.02)); rvec /= rvec.sum()
    xvec = np.append(xs, max(1 - xs.sum(), 0.02)); xvec /= xvec.sum()
    n_rush_td = rng.poisson(team_rush_tds * e_rush[:, 0])
    n_rec_td = rng.poisson(team_rec_tds * e_pass[:, 0])
    tds = rng.multinomial(n_rush_td, rvec)[:, :k] + rng.multinomial(n_rec_td, xvec)[:, :k]
    pass_tds = np.where(qb[None, :], n_rec_td[:, None], 0)
    return {"receptions": rec, "receiving_yards": rec_yds, "rushing_yards": rush_yds,
            "passing_yards": pass_yds, "tds": tds, "passing_tds": pass_tds}


def summarize(players: pd.DataFrame, sims: dict[str, np.ndarray]) -> pd.DataFrame:
    """Per player: simulated median / p10 / p90 for each stat and P(anytime TD)."""
    out = players[["player_id"]].copy()
    for stat in ("receptions", "receiving_yards", "rushing_yards", "passing_yards"):
        a = sims[stat]
        out[f"sim_{stat}"] = np.median(a, axis=0)
        out[f"sim_p10_{stat}"] = np.quantile(a, 0.1, axis=0)
        out[f"sim_p90_{stat}"] = np.quantile(a, 0.9, axis=0)
    out["sim_p_anytime_td"] = (sims["tds"] >= 1).mean(axis=0)
    return out


def joint_probability(sims: dict[str, np.ndarray], legs: list[tuple[str, int, float]]) -> float:
    """P(every leg) for legs (stat, player column index, threshold): stat >= threshold."""
    ok = np.ones(sims["receptions"].shape[0], dtype=bool)
    for stat, j, thr in legs:
        ok &= sims[stat][:, j] >= thr
    return float(ok.mean())


def qb_stacks(players: pd.DataFrame, sims: dict[str, np.ndarray], receivers: int = 3) -> list[dict]:
    """QB + receiver stacks: P(both legs clear their simulated medians) vs independence.

    ``players`` is indexed 0..n-1 in the column order of ``sims`` (as
    ``live.simulate_game`` returns it). One dict per receiver among the
    ``receivers`` with the most projected targets: the two names, the two
    thresholds (simulated medians), ``joint`` and ``independent`` (the product
    of the separate probabilities). Empty when the team has no starting QB.
    """
    qbs = players.index[players["is_starting_qb"].astype(bool)].tolist()
    if not qbs:
        return []
    q = qbs[0]
    qthr = float(np.median(sims["passing_yards"][:, q]))
    p_qb = float((sims["passing_yards"][:, q] >= qthr).mean())
    out = []
    for w in players.drop(index=q).sort_values("proj_targets", ascending=False).index[:receivers]:
        wthr = float(np.median(sims["receiving_yards"][:, w]))
        out.append({
            "qb": players.at[q, "player_display_name"], "qb_passing_yards": qthr,
            "receiver": players.at[w, "player_display_name"], "receiver_receiving_yards": wthr,
            "joint": joint_probability(sims, [("passing_yards", q, qthr), ("receiving_yards", w, wthr)]),
            "independent": p_qb * float((sims["receiving_yards"][:, w] >= wthr).mean()),
        })
    return out
