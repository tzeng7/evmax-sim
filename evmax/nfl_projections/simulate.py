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

Participation: a pre-game roster holds players who will not all play (about
9% of a live roster's skill players do not). Each simulation first draws who plays from
each player's ``p_active`` (``player_model.participation_probability``; 1 when
absent), and shares are split among the players who play. Without it, every
expected player's share was spread over the whole roster: on the 2025 holdout
the simulated medians ran 9-13% below the per-player model's on live rosters.
Player summaries are conditional on playing (a prop on a player who does not
play is void).

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
    ``rec_xtd_share``, ``is_starting_qb`` and optionally ``p_active`` (P(plays);
    1 when absent). Returns arrays of shape (n, players) for ``receptions``,
    ``receiving_yards``, ``rushing_yards``, ``passing_yards``, ``tds``,
    ``passing_tds`` and ``active`` (whether the player plays in that simulation;
    column order = ``players`` rows), and arrays of shape (n,) for the team
    totals ``team_receptions``, ``team_passing_yards``, ``team_rushing_yards``
    and ``team_tds`` (unprojected players included).
    """
    rng = rng or np.random.default_rng(0)
    k = len(players)
    ts = players["tgt_share"].to_numpy(float).clip(0, None)
    cs = players["car_share"].to_numpy(float).clip(0, None)
    p_active = (np.nan_to_num(players["p_active"].to_numpy(float), nan=1.0).clip(0, 1)
                if "p_active" in players else np.ones(k))
    active = rng.random((n, k)) < p_active
    # "other" slot keeps unprojected players' usage out of the projected players;
    # its size is set by the players EXPECTED to play (sum of p_active x share).
    t_other = max(1.0 - (p_active * ts).sum(), 0.02)
    c_other = max(1.0 - (p_active * cs).sum(), 0.02)
    tvec = np.append(ts, t_other)
    cvec = np.append(cs, c_other)

    cv2 = params.team_vol_cv ** 2
    gfac = rng.gamma(1 / cv2, cv2, size=(n, 2)) if cv2 > 0 else np.ones((n, 2))
    T = rng.poisson(team_targets * gfac[:, 0])
    C = rng.poisson(team_carries * gfac[:, 1])
    mask = np.hstack([active, np.ones((n, 1), dtype=bool)])
    pt = _masked_dirichlet(rng, params.target_kappa, tvec, mask)
    pc = _masked_dirichlet(rng, params.carry_kappa, cvec, mask)
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

    # Team totals, including the unprojected players: their carries at the team's
    # share-weighted yards per carry (the receiving side is already in team_pass).
    other_c = C - car.sum(axis=1)
    other_ypc = float(np.average(ypc, weights=np.maximum(cs, 1e-9)))
    team_rush = rush_yds.sum(axis=1) + other_c * other_ypc * e_rush[:, 0]
    other_cr = float(np.average(cr, weights=np.maximum(ts, 1e-9)))
    team_rec = rec.sum(axis=1) + rng.binomial(np.maximum(other_t, 0), other_cr)

    rs = players["rush_xtd_share"].to_numpy(float).clip(0, None)
    xs = players["rec_xtd_share"].to_numpy(float).clip(0, None)
    rvec = _masked_rows(np.append(rs, max(1 - (p_active * rs).sum(), 0.02)), mask)
    xvec = _masked_rows(np.append(xs, max(1 - (p_active * xs).sum(), 0.02)), mask)
    n_rush_td = rng.poisson(team_rush_tds * e_rush[:, 0])
    n_rec_td = rng.poisson(team_rec_tds * e_pass[:, 0])
    tds = rng.multinomial(n_rush_td, rvec)[:, :k] + rng.multinomial(n_rec_td, xvec)[:, :k]
    pass_tds = np.where(qb[None, :], n_rec_td[:, None], 0)
    return {"receptions": rec, "receiving_yards": rec_yds, "rushing_yards": rush_yds,
            "passing_yards": pass_yds, "tds": tds, "passing_tds": pass_tds, "active": active,
            "team_receptions": team_rec, "team_passing_yards": team_pass, "team_rushing_yards": team_rush,
            "team_tds": n_rush_td + n_rec_td}


def _masked_rows(vec: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``vec`` repeated per simulation with masked-out slots zeroed, rows renormalized."""
    rows = np.where(mask, vec[None, :], 0.0)
    return rows / rows.sum(axis=1, keepdims=True)


def _masked_dirichlet(rng: np.random.Generator, kappa: float, vec: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-simulation share draws: Dirichlet(kappa x shares) over every slot, then the
    players who do not play are dropped and the rest renormalized (by Dirichlet
    aggregation, a Dirichlet over the players who play)."""
    alpha = np.maximum(kappa * vec / vec.sum(), 1e-6)
    draws = np.where(mask, rng.dirichlet(alpha, size=mask.shape[0]), 0.0)
    return draws / draws.sum(axis=1, keepdims=True)


def _active(sims: dict[str, np.ndarray]) -> np.ndarray:
    a = sims.get("active")
    return a if a is not None else np.ones(sims["receptions"].shape, dtype=bool)


def summarize(players: pd.DataFrame, sims: dict[str, np.ndarray]) -> pd.DataFrame:
    """Per player, over the simulations he plays in: median / p10 / p90 for each stat,
    the mean (``sim_mean_*``) and P(anytime TD); ``sim_p_active`` = share of
    simulations he plays in."""
    out = players[["player_id"]].copy()
    act = _active(sims)
    for stat in ("receptions", "receiving_yards", "rushing_yards", "passing_yards"):
        a = np.where(act, sims[stat].astype(float), np.nan)
        out[f"sim_{stat}"] = np.nanmedian(a, axis=0)
        out[f"sim_p10_{stat}"] = np.nanquantile(a, 0.1, axis=0)
        out[f"sim_p90_{stat}"] = np.nanquantile(a, 0.9, axis=0)
        out[f"sim_mean_{stat}"] = np.nanmean(a, axis=0)
    out["sim_p_anytime_td"] = ((sims["tds"] >= 1) & act).sum(axis=0) / np.maximum(act.sum(axis=0), 1)
    out["sim_p_active"] = act.mean(axis=0)
    return out


def summarize_team(sims: dict[str, np.ndarray]) -> dict[str, tuple[float, float, float]]:
    """Team totals: stat -> (median, p10, p90) over the simulations."""
    out = {}
    for stat in ("receptions", "passing_yards", "rushing_yards", "tds"):
        a = sims.get(f"team_{stat}")
        if a is not None:
            out[stat] = (float(np.median(a)), float(np.quantile(a, 0.1)), float(np.quantile(a, 0.9)))
    return out


def joint_probability(sims: dict[str, np.ndarray], legs: list[tuple[str, int, float]]) -> float:
    """P(every leg | every leg's player plays) for legs (stat, player column index,
    threshold): stat >= threshold."""
    act = _active(sims)
    played = np.ones(act.shape[0], dtype=bool)
    ok = np.ones(act.shape[0], dtype=bool)
    for stat, j, thr in legs:
        played &= act[:, j]
        ok &= sims[stat][:, j] >= thr
    return float((ok & played).sum() / max(played.sum(), 1))


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
    act = _active(sims)
    qthr = float(np.median(sims["passing_yards"][:, q]))
    out = []
    for w in players.drop(index=q).sort_values("proj_targets", ascending=False).index[:receivers]:
        on = act[:, w]          # both legs need the receiver to play (a void leg otherwise)
        if not on.any():
            continue
        wthr = float(np.median(sims["receiving_yards"][on, w]))
        out.append({
            "qb": players.at[q, "player_display_name"], "qb_passing_yards": qthr,
            "receiver": players.at[w, "player_display_name"], "receiver_receiving_yards": wthr,
            "joint": joint_probability(sims, [("passing_yards", q, qthr), ("receiving_yards", w, wthr)]),
            "independent": (float((sims["passing_yards"][on, q] >= qthr).mean())
                            * float((sims["receiving_yards"][on, w] >= wthr).mean())),
        })
    return out
