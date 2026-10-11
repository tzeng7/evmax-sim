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
VOLUME_STATS = ("team_targets", "team_carries", "team_attempts", "team_rush_tds", "team_rec_tds")


@dataclass(frozen=True)
class PlayerModelConfig:
    half_life_days: float = 70.0
    offseason_days: float = 180.0
    lookback_days: int = 730
    lam: float = 4.0
    share_prior_games: float = 0.5     # pseudo-games at the position-mean share (share is very stable; 3.0 over-shrank stars)
    k_targets: float = 60.0            # pseudo-targets for catch rate / yards per target
    k_carries: float = 80.0            # pseudo-carries for yards per carry
    k_attempts: float = 150.0          # pseudo-attempts for QB yards per attempt
    k_team_attempts: float = 100.0     # pseudo team-attempts for a starter's attempt share
    td_share_prior_games: float = 2.0  # pseudo-games at the position-mean expected-TD share
    td_actual_weight: float = 0.25     # weight on the actual TD share (TD2: z -5.5, 6/6 seasons vs xTD-only)
    min_start_attempts: float = 10.0   # a game counts as a start at >= this many attempts
    # Fraction of a ruled-out player's expected target/carry share that goes to the
    # teammates expected to play (the rest goes to call-ups and players with no
    # recent games). 0.6 = InjuryReportAgent.compute_prop_injury_boost's default.
    injury_redistribution: float = 0.6
    # Also redistribute the usage of recent players the weekly roster rules out
    # (reserve lists, practice squad, released). REJECTED 2026-10-10: dev_score
    # 0.9246 -> 0.9319 (reserve only 0.9285, practice squad / released only
    # 0.9299), holdout receiving MAE 19.00 -> 19.18 — departing players are mostly
    # replaced by signings with no recent games. Such players are still dropped
    # from the expected roster; only their usage stays unassigned.
    roster_out_redistribution: bool = False


@dataclass
class PlayerState:
    """Point-in-time usage and efficiency for every player seen before the cutoff."""
    usage: pd.DataFrame                 # index player_id: tgt_share, car_share, games, position, team
    fits: dict[str, RatingFit] = field(default_factory=dict)
    # Distribution shape fitted from past games (leak-free): Gamma Var = theta * mean
    # for yardage, NegBin Var = mu + mu^2 / k for receptions.
    theta: dict[str, float] = field(default_factory=dict)
    negbin_k: float = 5.0
    pass_sd: float = 75.0
    # Empirical (game value / player window mean) quantile curves by usage level,
    # for the REPORTED yardage ranges: (stat, q) -> (bin centers, ratios).
    quantile_curves: dict[tuple[str, float], tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)


def team_volume_rows(player_games: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    """Team-game rows with team targets / carries / attempts / rushing and receiving TDs, and the home flag."""
    tv = player_games.groupby(["game_id", "team"]).agg(
        team_targets=("team_targets", "first"), team_carries=("team_carries", "first"),
        team_attempts=("team_attempts", "first"), team_rush_tds=("rushing_tds", "sum"),
        team_rec_tds=("receiving_tds", "sum")).reset_index()
    return tv.merge(team_games[["game_id", "team", "opp", "home", "gameday"]], on=["game_id", "team"])


def fit_player_state(player_games: pd.DataFrame, volume_rows: pd.DataFrame, cutoff: pd.Timestamp,
                     cfg: PlayerModelConfig = PlayerModelConfig(),
                     rz: pd.DataFrame | None = None) -> PlayerState:
    """Usage/efficiency state from games before ``cutoff``. With ``rz`` (``td_model.load_rz_usage``)
    each player also gets his expected-TD shares (``rush_xtd_share``, ``rec_xtd_share``)."""
    lo = cutoff - pd.Timedelta(days=cfg.lookback_days)
    vr = volume_rows[(volume_rows["gameday"] < cutoff) & (volume_rows["gameday"] >= lo)]
    fits = {m: fit_rating(vr, m, cutoff, cfg.half_life_days, cfg.lam, offseason_days=cfg.offseason_days)
            for m in VOLUME_STATS if m in vr}

    pg = player_games[(player_games["gameday"] < cutoff) & (player_games["gameday"] >= lo)]
    xtd = _xtd_shares(pg, rz) if rz is not None else None
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
    if xtd is not None:
        x = xtd.reindex(pd.MultiIndex.from_arrays([pg["game_id"], pg["player_id"]]))
        beta = cfg.td_actual_weight
        rush = ((1 - beta) * x["rush_share"] + beta * x["rush_td_share"]).to_numpy()
        rec = ((1 - beta) * x["rec_share"] + beta * x["rec_td_share"]).to_numpy()
        w_r = np.where(np.isnan(rush), 0.0, w)
        w_c = np.where(np.isnan(rec), 0.0, w)
        xa = pd.DataFrame({"player_id": pg["player_id"].to_numpy(), "position": pg["position"].to_numpy(),
                           "w_r": w_r, "w_c": w_c, "wr": w_r * np.nan_to_num(rush), "wc": w_c * np.nan_to_num(rec)})
        xs = xa.groupby("player_id")[["w_r", "w_c", "wr", "wc"]].sum().reindex(agg.index).fillna(0.0)
        xp = xa.groupby("position")[["w_r", "w_c", "wr", "wc"]].sum()
        kt = cfg.td_share_prior_games
        pos_r = (xp["wr"] / xp["w_r"].clip(lower=1e-9)).reindex(agg["position"]).fillna(0.0).to_numpy()
        pos_c = (xp["wc"] / xp["w_c"].clip(lower=1e-9)).reindex(agg["position"]).fillna(0.0).to_numpy()
        agg["rush_xtd_share"] = (xs["wr"].to_numpy() + kt * pos_r) / np.maximum(xs["w_r"].to_numpy() + kt, 1e-9)
        agg["rec_xtd_share"] = (xs["wc"].to_numpy() + kt * pos_c) / np.maximum(xs["w_c"].to_numpy() + kt, 1e-9)
    return PlayerState(usage=agg, fits=fits, theta=fit_dispersion(pg), negbin_k=fit_negbin_k(pg),
                       pass_sd=fit_pass_sd(pg),
                       quantile_curves={(st, q): fit_quantile_curve(pg, st, q)
                                        for st in ("receiving_yards", "rushing_yards") for q in QUANTILES})


def _xtd_shares(window: pd.DataFrame, rz: pd.DataFrame) -> pd.DataFrame:
    """Per (game_id, player_id): the player's share of his team's rushing / receiving
    expected TDs in that game. Bucket TD rates come from the same window (leak-free)."""
    from evmax.nfl_projections.td_model import REC_TD, RUSH_TD, RZ_COLS, bucket_td_rates, expected_tds

    w = window[["game_id", "team", "player_id"]].merge(rz, on=["game_id", "player_id"], how="left")
    w[RZ_COLS] = w[RZ_COLS].fillna(0.0)
    r_rush, r_tgt = bucket_td_rates(w)
    w["x_rush"], w["x_rec"] = expected_tds(w, r_rush, r_tgt)
    w["a_rush"] = w[RUSH_TD].sum(axis=1)
    w["a_rec"] = w[REC_TD].sum(axis=1)
    tot = w.groupby(["game_id", "team"])[["x_rush", "x_rec", "a_rush", "a_rec"]].transform("sum")
    # A game where the team had no expected TDs of a kind says nothing about how its
    # TDs split, so its share is NaN (left out of the player's average), not 0.
    for share, num, base in (("rush_share", "x_rush", "x_rush"), ("rec_share", "x_rec", "x_rec"),
                             ("rush_td_share", "a_rush", "x_rush"), ("rec_td_share", "a_rec", "x_rec")):
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(tot[num] > 0, w[num] / tot[num].clip(lower=1e-9), 0.0)
        w[share] = np.where(tot[base] > 0, frac, np.nan)
    return w.drop_duplicates(["game_id", "player_id"]).set_index(["game_id", "player_id"])[
        ["rush_share", "rec_share", "rush_td_share", "rec_td_share"]]


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


def fit_pass_sd(pg: pd.DataFrame, min_attempts: float = 10.0) -> float:
    """Pooled within-QB SD of passing yards over starts (games with >= min_attempts)."""
    st = pg[pg["attempts"] >= min_attempts]
    m = st.groupby("player_id")["passing_yards"].agg(["var", "size"])
    m = m[m["size"] >= MIN_GAMES_FOR_DISPERSION]
    if not len(m):
        return 75.0
    return max(float(np.sqrt((m["var"] * m["size"]).sum() / m["size"].sum())), 1.0)  # floor: degenerate windows


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
    return negbin_quantile(mean, k, 0.5)


QUANTILE_CURVE_MIN_GAMES = 4
QUANTILE_CURVE_BINS = 8


def fit_quantile_curve(pg: pd.DataFrame, stat: str, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Quantile ``q`` of (game value / player's window mean), by usage level.

    The Gamma's lower tail is too thin for yardage (players have more near-zero
    games than it allows: 23% of receiving results fell below its 10th
    percentile), so the reported range uses this empirical, usage-dependent
    shape fitted on past games only.
    """
    m = pg.groupby("player_id")[stat].agg(["mean", "size"])
    m = m[(m["size"] >= QUANTILE_CURVE_MIN_GAMES) & (m["mean"] > 1.0)]
    if len(m) < QUANTILE_CURVE_BINS * 5:
        return np.array([1.0]), np.array([np.nan])
    d = pg[pg["player_id"].isin(m.index)][["player_id", stat]].join(m["mean"], on="player_id")
    d["z"] = d[stat] / d["mean"]
    edges = np.unique(np.quantile(m["mean"], np.linspace(0, 1, QUANTILE_CURVE_BINS + 1)))
    d["bin"] = pd.cut(d["mean"], edges, include_lowest=True)
    g = d.groupby("bin", observed=True).agg(center=("mean", "median"), ratio=("z", lambda z: z.quantile(q)))
    return g["center"].to_numpy(dtype=float), g["ratio"].to_numpy(dtype=float)


def curve_quantile(mean: np.ndarray, curve: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    centers, ratios = curve
    mean = np.asarray(mean, dtype=float)
    return np.where(mean > 0, mean * np.interp(mean, centers, ratios), 0.0)


def gamma_quantile(mean: np.ndarray, theta: float, q: float) -> np.ndarray:
    mean = np.asarray(mean, dtype=float)
    out = np.zeros_like(mean)
    pos = mean > 0
    out[pos] = stats.gamma.ppf(q, a=mean[pos] / theta, scale=theta)
    return out


def negbin_quantile(mean: np.ndarray, k: float, q: float) -> np.ndarray:
    mean = np.asarray(mean, dtype=float)
    out = np.zeros_like(mean)
    pos = mean > 0
    out[pos] = stats.nbinom.ppf(q, n=k, p=k / (k + mean[pos]))
    return out


QUANTILES = (0.1, 0.9)  # reported range around the median


# ── pre-game injury report -> usage redistribution ───────────────────────────
# Same rule as the basketball prop boost (InjuryReportAgent.get_out_players /
# compute_prop_injury_boost): Out and Doubtful players do not play, and part of
# their usage goes to the teammates who do. Here the freed usage is the out
# player's own modeled share (not a fixed tier share). It is split across the
# team's expected players in proportion to their shares. Only the PRE-GAME
# report is used, never who actually played (that leaks garbage-time
# participation — the rejected P1 iteration).
OUT_STATUSES = ("Out", "Doubtful")
RECENT_GAMES = 3
REDISTRIBUTED_POSITIONS = ("WR", "TE", "RB")  # a QB's usage moves with the starter, not to teammates


def out_players(injuries: pd.DataFrame, week: int, season: int | None = None) -> set[tuple[str, str]]:
    """(team, gsis id) pairs listed Out/Doubtful on the ``week`` injury report (of ``season``, when given).

    Keyed by team so a player who changed teams never frees usage on his old team.
    """
    if injuries is None or injuries.empty:
        return set()
    i = injuries[(injuries["week"] == week) & injuries["report_status"].isin(OUT_STATUSES)]
    if season is not None:
        i = i[i["season"] == season]
    i = i.dropna(subset=["gsis_id"])
    return set(zip(i["team"], i["gsis_id"]))


# Weekly roster status (data.load_rosters). The injury report misses every player
# who is not on the active roster — injured reserve, PUP, NFI, suspension, the
# practice squad, released — so a player placed on IR after his last game would
# otherwise stay in the expected roster with his full usage. ACT = active roster;
# INA = declared inactive for that week's game (published ~90 minutes before
# kickoff); everything else (RES, DEV, CUT, RET, …) does not play. Validation:
# 99.97% of 2016-26 skill players who played a week are ACT on that week's roster,
# so the status is a pre-game state (no leak in the walk-forward).
ROSTER_ACTIVE = "ACT"
ROSTER_INACTIVE = "INA"


def roster_unavailable(rosters: pd.DataFrame | None, players: pd.DataFrame, season: int, week: int,
                       inactive: bool = True) -> set[tuple[str, str]]:
    """(team, player_id) pairs of ``players`` (columns team, player_id) the weekly roster rules out.

    Each team's roster for ``season`` ``week`` is used, or its latest earlier week when
    this week's is not published yet. A player missing from his team's roster has left
    the team; a team with no roster rows at all is left alone (no information). An INA
    row counts as out only on the week's own roster and only with ``inactive`` (an
    earlier week's INA was another game's inactive list).
    """
    if rosters is None or rosters.empty or players.empty:
        return set()
    r = rosters[(rosters["season"] == season) & (rosters["week"] <= week)].dropna(subset=["gsis_id"])
    if r.empty:
        return set()
    latest = r.groupby("team")["week"].max()
    r = r[r["week"].to_numpy() == latest.reindex(r["team"]).to_numpy()]
    status = r.drop_duplicates(["team", "gsis_id"], keep="last").set_index(["team", "gsis_id"])["status"]
    st = status.reindex(pd.MultiIndex.from_arrays([players["team"], players["player_id"]])).to_numpy()
    roster_week = players["team"].map(latest).to_numpy(dtype=float)
    ok = (st == ROSTER_ACTIVE) | ((st == ROSTER_INACTIVE) & ((roster_week < week) | (not inactive)))
    ruled_out = players["team"].isin(latest.index).to_numpy() & ~ok
    return set(zip(players["team"].to_numpy()[ruled_out], players["player_id"].to_numpy()[ruled_out]))


def recent_team_players(player_games: pd.DataFrame, teams, cutoff: pd.Timestamp,
                        n_games: int = RECENT_GAMES) -> pd.DataFrame:
    """(team, player_id, player_display_name, position, recent_games, played_last) for
    everyone who played for each of ``teams`` in any of its last ``n_games`` games
    before ``cutoff``: ``recent_games`` = how many of those games he played,
    ``played_last`` = whether he played the most recent one."""
    window = player_games[(player_games["gameday"] < cutoff)
                          & (player_games["gameday"] >= cutoff - pd.Timedelta(days=400))]
    past = window[window["team"].isin(set(teams))]
    gids = past[["team", "game_id", "gameday"]].drop_duplicates(["team", "game_id"]).sort_values("gameday")
    last = gids.groupby("team").tail(n_games)[["team", "game_id"]]
    # Each team's own last game: keyed by (team, game id) — a game id is shared by both
    # teams, and the opponent's last game may be an earlier one (a bye in between).
    final = set(gids.groupby("team")["game_id"].last().items())
    rows = past.merge(last, on=["team", "game_id"]).sort_values("gameday")
    # A traded player belongs to the team he played for most recently (over all teams).
    latest = window.sort_values("gameday").drop_duplicates("player_id", keep="last").set_index("player_id")["team"]
    rows = rows[rows["team"].to_numpy() == latest.reindex(rows["player_id"]).to_numpy()]
    rows = rows.assign(last_game=[(t, g) in final for t, g in zip(rows["team"], rows["game_id"])])
    feats = rows.groupby("player_id").agg(recent_games=("game_id", "nunique"), played_last=("last_game", "any"))
    out = rows.drop_duplicates("player_id", keep="last")[["team", "player_id", "player_display_name", "position"]]
    return out.join(feats, on="player_id").reset_index(drop=True)


# Probability that an expected player plays, by pre-game evidence: (games played of
# the team's last 3, played the team's last game, Questionable on the report).
# ``scripts/fit_nfl_participation.py``: fitted on 2019-24 pre-game rosters
# (``pregame_rosters``, skill positions, starting QB excluded; n 36,168); 2025
# holdout mean 0.905 predicted vs 0.902 played, every probability bin within
# 3.5 pp, Brier 0.070 vs 0.089 for a constant. The starting QB always plays.
PARTICIPATION = {
    (1, False, False): 0.503, (1, False, True): 0.538, (1, True, False): 0.830, (1, True, True): 0.620,
    (2, False, False): 0.682, (2, False, True): 0.609, (2, True, False): 0.885, (2, True, True): 0.635,
    (3, True, False): 0.971, (3, True, True): 0.739,
}


def participation_probability(recent_games, played_last, questionable, starting_qb) -> np.ndarray:
    """P(plays) per player from ``PARTICIPATION`` (1.0 for the starting QB and for players
    without recent games, i.e. the added starter)."""
    keys = zip(np.asarray(recent_games, dtype=int).clip(0, RECENT_GAMES), np.asarray(played_last, dtype=bool),
               np.asarray(questionable, dtype=bool))
    p = np.array([PARTICIPATION.get((int(k), bool(lp), bool(q)), 1.0) for k, lp, q in keys], dtype=float)
    return np.where(np.asarray(starting_qb, dtype=bool), 1.0, p)


def injury_share_multipliers(usage: pd.DataFrame, recent: pd.DataFrame, out: set[tuple[str, str]],
                             alpha: float) -> pd.DataFrame:
    """Target/carry share multipliers for the teammates of ruled-out players.

    ``out`` holds (team, player_id) pairs. For each team: freed = sum of the out
    WR/TE/RB shares; every expected player (recent and not out) gets
    ``share * (1 + alpha * freed / sum of expected shares)``.
    Returns one row per expected player, index (player_id, team): ``tgt_mult``, ``car_mult``.
    """
    empty = pd.DataFrame(columns=["tgt_mult", "car_mult"],
                         index=pd.MultiIndex.from_arrays([[], []], names=["player_id", "team"]))
    r = recent[recent["player_id"].isin(usage.index)].copy()
    if r.empty or not out:
        return empty
    u = usage.loc[r["player_id"]]
    r["tgt"] = u["tgt_share"].to_numpy()
    r["car"] = u["car_share"].to_numpy()
    r["out"] = [(t, pid) in out for t, pid in zip(r["team"], r["player_id"])]
    freeing = r["out"] & u["position"].isin(REDISTRIBUTED_POSITIONS).to_numpy()
    active = r[~r["out"]]
    res = pd.DataFrame(index=pd.MultiIndex.from_arrays([active["player_id"], active["team"]],
                                                       names=["player_id", "team"]))
    pairs = [("tgt", "tgt_mult"), ("car", "car_mult")]
    if "rush_xtd_share" in usage and "rec_xtd_share" in usage:
        # Expected-TD shares move by their own freed amount (a goal-line back's TD
        # share is far larger than his carry share).
        r["rxtd"] = u["rush_xtd_share"].to_numpy()
        r["cxtd"] = u["rec_xtd_share"].to_numpy()
        active = r[~r["out"]]
        pairs += [("rxtd", "rush_xtd_mult"), ("cxtd", "rec_xtd_mult")]
    for col, mult in pairs:
        freed = r[freeing].groupby("team")[col].sum()
        kept = active.groupby("team")[col].sum()
        m = 1.0 + alpha * freed.reindex(kept.index).fillna(0.0) / kept.clip(lower=1e-9)
        res[mult] = active["team"].map(m).fillna(1.0).to_numpy()
    return res


def project_players(state: PlayerState, roster: pd.DataFrame,
                    team_volume: pd.DataFrame | None = None,
                    share_mult: pd.DataFrame | None = None) -> pd.DataFrame:
    """Project each (player, team, opp, home, is_starting_qb) row of ``roster``.

    ``share_mult`` (from ``injury_share_multipliers``, index (player_id, team))
    scales target/carry shares for teammates of ruled-out players; a player only
    takes the multiplier of the team he plays for in this game.
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
           for m in VOLUME_STATS if m in f}
    if team_volume is not None:  # script-conditioned team volume (``volume_combiners``)
        idx = pd.MultiIndex.from_arrays([r["game_id"], r["team"]])
        for m in exp:
            if m not in team_volume:
                continue
            v = team_volume[m].reindex(idx).to_numpy()
            exp[m] = np.where(np.isnan(v), exp[m], v)
    uu = u.loc[r["player_id"]]
    tgt_share = uu["tgt_share"].to_numpy()
    car_share = uu["car_share"].to_numpy()
    if share_mult is not None and len(share_mult):
        m = share_mult.reindex(pd.MultiIndex.from_arrays([r["player_id"], r["team"]]))
        tgt_share = tgt_share * m["tgt_mult"].fillna(1.0).to_numpy(dtype=float)
        car_share = car_share * m["car_mult"].fillna(1.0).to_numpy(dtype=float)
    # Inputs kept for the joint simulation (simulate.simulate_team).
    r["tgt_share"], r["car_share"] = tgt_share, car_share
    if "recent_games" in r:  # a pre-game roster (live.active_roster): who might not play
        r["p_active"] = participation_probability(r["recent_games"].fillna(0), r["played_last"].fillna(False),
                                                  r.get("questionable", pd.Series(False, index=r.index)).fillna(False),
                                                  r["is_starting_qb"])
    r["catch_rate"], r["ypt"], r["ypc"] = (uu["catch_rate"].to_numpy(), uu["ypt"].to_numpy(),
                                           uu["ypc"].to_numpy())
    for m, v in exp.items():
        r[f"exp_{m}"] = v
    r["proj_targets"] = exp["team_targets"] * tgt_share
    r["proj_receptions"] = r["proj_targets"] * uu["catch_rate"].to_numpy()
    r["proj_receiving_yards"] = r["proj_targets"] * uu["ypt"].to_numpy()
    r["proj_carries"] = exp["team_carries"] * car_share
    r["proj_rushing_yards"] = r["proj_carries"] * uu["ypc"].to_numpy()
    r["proj_passing_yards"] = np.where(
        r["is_starting_qb"], exp["team_attempts"] * uu["att_share"].to_numpy() * uu["ypa"].to_numpy(), 0.0)
    if "rush_xtd_share" in uu and "team_rush_tds" in exp:
        # Touchdowns: team TD volume x expected-TD share; Poisson counts (td_model).
        from evmax.nfl_projections.td_model import poisson_at_least

        base_t, base_c = uu["tgt_share"].to_numpy(), uu["car_share"].to_numpy()
        tmult = np.where(base_t > 0, tgt_share / np.where(base_t > 0, base_t, 1.0), 1.0)
        cmult = np.where(base_c > 0, car_share / np.where(base_c > 0, base_c, 1.0), 1.0)
        rmult, xmult = cmult, tmult
        if share_mult is not None and len(share_mult) and "rush_xtd_mult" in share_mult:
            m2 = share_mult.reindex(pd.MultiIndex.from_arrays([r["player_id"], r["team"]]))
            rmult = m2["rush_xtd_mult"].fillna(1.0).to_numpy(dtype=float)
            xmult = m2["rec_xtd_mult"].fillna(1.0).to_numpy(dtype=float)
        r["rush_xtd_share"] = uu["rush_xtd_share"].to_numpy() * rmult
        r["rec_xtd_share"] = uu["rec_xtd_share"].to_numpy() * xmult
        lam_rush = exp["team_rush_tds"] * r["rush_xtd_share"].to_numpy()
        lam_rec = exp["team_rec_tds"] * r["rec_xtd_share"].to_numpy()
        r["proj_rush_tds"] = np.clip(lam_rush, 0.0, None)
        r["proj_rec_tds"] = np.clip(lam_rec, 0.0, None)
        r["proj_tds"] = r["proj_rush_tds"] + r["proj_rec_tds"]
        r["p_anytime_td"] = poisson_at_least(r["proj_tds"].to_numpy(), 1)
        r["p_two_plus_td"] = poisson_at_least(r["proj_tds"].to_numpy(), 2)
        r["proj_passing_tds"] = np.where(
            r["is_starting_qb"], np.clip(exp["team_rec_tds"] * uu["att_share"].to_numpy(), 0.0, None), 0.0)
    # Means above; the reported point projection is the MEDIAN (MAE-optimal for
    # skewed stats). Passing yards are near-symmetric for starters: median = mean.
    for stat in ("receptions", "receiving_yards", "rushing_yards"):
        r[f"mean_{stat}"] = r[f"proj_{stat}"]
    theta = {**DEFAULT_THETA, **state.theta}
    r["proj_receiving_yards"] = gamma_median(r["mean_receiving_yards"], theta["receiving_yards"])
    r["proj_rushing_yards"] = gamma_median(r["mean_rushing_yards"], theta["rushing_yards"])
    r["proj_receptions"] = negbin_median(r["mean_receptions"], state.negbin_k)
    # 10th / 90th percentiles from the same fitted distributions (passing: Normal).
    for q in QUANTILES:
        tag = f"p{int(q * 100)}"
        for st in ("receiving_yards", "rushing_yards"):
            curve = state.quantile_curves.get((st, q))
            r[f"{tag}_{st}"] = (curve_quantile(r[f"mean_{st}"], curve)
                                if curve is not None and not np.isnan(curve[1]).any()
                                else gamma_quantile(r[f"mean_{st}"], theta[st], q))
        r[f"{tag}_receptions"] = negbin_quantile(r["mean_receptions"], state.negbin_k, q)
        r[f"{tag}_passing_yards"] = np.where(
            r["is_starting_qb"], np.maximum(stats.norm.ppf(q, r["proj_passing_yards"], state.pass_sd), 0.0), 0.0)
    return r


# ── game script -> team volume ───────────────────────────────────────────────
# Team volume depends on the expected game flow: favorites run more, trailing
# teams throw more, high-total games have more plays. The script comes from OUR
# game model (no market input); a per-stat OLS on top of the opponent-adjusted
# volume rating is trained on earlier seasons only.
SCRIPT_FIRST_SEASON = 2016  # earliest script season (game combiner trains from 2015)


def first_script_season(team_games: pd.DataFrame) -> int:
    """First season the game model can project: one season after the data starts (its
    combiner needs a prior season), and never before SCRIPT_FIRST_SEASON."""
    return max(SCRIPT_FIRST_SEASON, int(team_games["season"].min()) + 1)


def game_script(team_games: pd.DataFrame, games: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    """(game_id, team) -> the game model's projected margin (team - opp) and total."""
    from evmax.nfl_projections.game_model import GameModelConfig
    from evmax.nfl_projections.game_model import walk_forward as game_walk_forward

    gp = game_walk_forward(team_games, games, seasons, GameModelConfig())
    sides = []
    for team, sign in (("home_team", 1.0), ("away_team", -1.0)):
        sides.append(pd.DataFrame({"game_id": gp["game_id"], "team": gp[team],
                                   "proj_margin": sign * gp["proj_margin"], "proj_total": gp["proj_total"]}))
    return pd.concat(sides).set_index(["game_id", "team"])


def volume_feature_table(volume_rows: pd.DataFrame, script: pd.DataFrame, games: pd.DataFrame,
                         seasons: list[int], cfg: PlayerModelConfig = PlayerModelConfig()) -> pd.DataFrame:
    """Team-game rows: point-in-time volume rating expectations + game script + actual volume."""
    sched = games[games["season"].isin(seasons) & games["home_score"].notna()]
    out = []
    for (season, week), wk in sched.groupby(["season", "week"], sort=True):
        cutoff = wk["gameday"].min()
        lo = cutoff - pd.Timedelta(days=cfg.lookback_days)
        vr = volume_rows[(volume_rows["gameday"] < cutoff) & (volume_rows["gameday"] >= lo)]
        fits = {m: fit_rating(vr, m, cutoff, cfg.half_life_days, cfg.lam, offseason_days=cfg.offseason_days)
                for m in VOLUME_STATS if m in vr}
        cur = volume_rows[volume_rows["game_id"].isin(wk["game_id"])].copy()
        for m in fits:
            cur["exp_" + m] = [fits[m].expect(t, o, h) for t, o, h in zip(cur["team"], cur["opp"], cur["home"])]
        cur["season"] = season
        out.append(cur)
    vft = pd.concat(out, ignore_index=True)
    return vft.join(script, on=["game_id", "team"]).dropna(subset=["proj_margin", "proj_total"])


def volume_combiners(vft: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per volume stat: OLS coef for [1, rating expectation, proj margin, proj total]."""
    out = {}
    for m in VOLUME_STATS:
        if m not in vft:
            continue
        X = np.column_stack([np.ones(len(vft)), vft["exp_" + m], vft["proj_margin"], vft["proj_total"]])
        out[m] = np.linalg.lstsq(X, vft[m].to_numpy(dtype=float), rcond=None)[0]
    return out


def predict_volume(vft: pd.DataFrame, coefs: dict[str, np.ndarray]) -> pd.DataFrame:
    res = pd.DataFrame(index=pd.MultiIndex.from_arrays([vft["game_id"], vft["team"]]))
    for m, c in coefs.items():
        res[m] = (c[0] + c[1] * vft["exp_" + m] + c[2] * vft["proj_margin"] + c[3] * vft["proj_total"]).to_numpy()
    return res


def walk_forward(team_games: pd.DataFrame, player_games: pd.DataFrame, games: pd.DataFrame,
                 seasons: list[int], cfg: PlayerModelConfig = PlayerModelConfig(),
                 injuries: pd.DataFrame | None = None, rz: pd.DataFrame | None = None,
                 rosters: pd.DataFrame | None = None, roster: str = "played") -> pd.DataFrame:
    """Project ``seasons``' completed games week by week, leak-free.

    ``roster="played"``: everyone who played those games (the backtest stand-in
    for the pre-game active list). ``roster="live"``: the roster a pre-game run
    builds (``live.active_roster``: recent players minus the injury report's
    Out/Doubtful and the weekly roster's ruled-out players, backup QBs dropped);
    its rows carry the game's actual stats when the player played and
    ``played`` = False when he did not. Either way the starting QB is the
    team's first-dropback passer (pre-game starter identity). Each week uses one
    state fit on games strictly before its first kickoff. With ``injuries``
    (nflverse weekly reports), teammates of players ruled Out/Doubtful on that
    week's PRE-GAME report get their usage (``injury_share_multipliers``).
    ``rosters`` (weekly rosters) drop the players they rule out (reserve lists,
    released) from the live roster; their usage is redistributed only with
    ``cfg.roster_out_redistribution`` (rejected, see PlayerModelConfig).
    Game-day inactives are not used: a pre-game run usually happens before they
    are published.
    """
    if roster not in ("played", "live"):
        raise ValueError(f"roster must be 'played' or 'live', not {roster!r}")
    vol = team_volume_rows(player_games, team_games)
    script_seasons = list(range(first_script_season(team_games), max(seasons) + 1))
    vft = volume_feature_table(vol, game_script(team_games, games, script_seasons), games, script_seasons, cfg)
    home = team_games.set_index(["game_id", "team"])["home"]
    starters = team_games.set_index(["game_id", "team"])["first_qb_id"]
    sched = games[games["season"].isin(seasons) & games["home_score"].notna()]
    results = []
    for season in sorted(sched["season"].unique()):
        train = vft[vft["season"] < season]
        team_volume = (predict_volume(vft[vft["season"] == season], volume_combiners(train))
                       if len(train) else None)
        season_inj = injuries[injuries["season"] == season] if injuries is not None else None
        for week, wk in sched[sched["season"] == season].groupby("week", sort=True):
            cutoff = wk["gameday"].min()
            state = fit_player_state(player_games, vol, cutoff, cfg, rz=rz)
            recent = recent_team_players(player_games, set(wk["home_team"]) | set(wk["away_team"]), cutoff)
            report_out, roster_out = _week_out_sets(season_inj, rosters, recent, season, week)
            freed = report_out | (roster_out if cfg.roster_out_redistribution else set())
            mult = (injury_share_multipliers(state.usage, recent, freed, cfg.injury_redistribution)
                    if injuries is not None or (rosters is not None and cfg.roster_out_redistribution) else None)
            if roster == "played":
                r = player_games[player_games["game_id"].isin(wk["game_id"])].copy()
                idx = pd.MultiIndex.from_arrays([r["game_id"], r["team"]])
                r["home"] = home.reindex(idx).fillna(0).to_numpy()
                r["is_starting_qb"] = starters.reindex(idx).to_numpy() == r["player_id"].to_numpy()
            else:
                r = _live_week_roster(player_games, wk, cutoff, week, season_inj, report_out | roster_out, starters)
            results.append(project_players(state, r, team_volume, mult))
    return pd.concat(results, ignore_index=True)


def _week_out_sets(injuries: pd.DataFrame | None, rosters: pd.DataFrame | None, recent: pd.DataFrame,
                   season: int, week: int) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """(injury report Out/Doubtful, weekly-roster ruled out) for a completed week, pre-game
    information only (no game-day inactive list)."""
    report_out = out_players(injuries, week, season) if injuries is not None else set()
    return report_out, roster_unavailable(rosters, recent, season, week, inactive=False)


def pregame_rosters(team_games: pd.DataFrame, player_games: pd.DataFrame, games: pd.DataFrame,
                    seasons: list[int], injuries: pd.DataFrame | None,
                    rosters: pd.DataFrame | None) -> pd.DataFrame:
    """Every completed week's pre-game roster (``walk_forward(roster="live")``'s, without
    the projection): one row per expected player with the participation evidence and
    ``played``. The population ``PARTICIPATION`` is fitted on
    (``scripts/fit_nfl_participation.py``)."""
    starters = team_games.set_index(["game_id", "team"])["first_qb_id"]
    sched = games[games["season"].isin(seasons) & games["home_score"].notna()]
    out = []
    for season in sorted(sched["season"].unique()):
        season_inj = injuries[injuries["season"] == season] if injuries is not None else None
        for week, wk in sched[sched["season"] == season].groupby("week", sort=True):
            cutoff = wk["gameday"].min()
            recent = recent_team_players(player_games, set(wk["home_team"]) | set(wk["away_team"]), cutoff)
            report_out, roster_out = _week_out_sets(season_inj, rosters, recent, season, week)
            r = _live_week_roster(player_games, wk, cutoff, week, season_inj, report_out | roster_out, starters)
            if not r.empty:
                out.append(r.assign(season=season, week=week))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _live_week_roster(player_games: pd.DataFrame, wk: pd.DataFrame, cutoff: pd.Timestamp, week: int,
                      injuries: pd.DataFrame | None, ruled_out: set[tuple[str, str]],
                      starters: pd.Series) -> pd.DataFrame:
    """A completed week's pre-game roster (``live.active_roster``) with each player's actual
    stats attached (zeros are real games; ``played`` = False rows have no stats)."""
    from evmax.nfl_projections.live import active_roster

    qbs = {}
    for g in wk.itertuples():
        for t in (g.home_team, g.away_team):
            qb = starters.get((g.game_id, t))
            qbs[t] = qb if isinstance(qb, str) else None
    inj = injuries if injuries is not None else pd.DataFrame(columns=["week", "team", "gsis_id", "report_status"])
    r = active_roster(player_games, inj, wk, cutoff, week, extra_out=ruled_out, starters=qbs)
    if r.empty:
        return r
    meta = ["game_id", "season", "week", "season_type", "gameday"]
    stats_cols = [c for c in player_games.columns
                  if c not in meta + ["team", "opp", "player_id", "player_display_name", "position"]]
    actual = player_games[player_games["game_id"].isin(wk["game_id"])][["game_id", "player_id"] + stats_cols]
    r = r.merge(actual, on=["game_id", "player_id"], how="left")
    r["played"] = r["offense_snaps"].notna()
    gm = player_games[player_games["game_id"].isin(wk["game_id"])].drop_duplicates("game_id")[meta]
    return r.merge(gm, on="game_id", how="left")
