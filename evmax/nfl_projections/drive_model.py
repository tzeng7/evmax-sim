"""Drive simulator: an NFL game as a sequence of possessions.

Every drive starts in a field-position zone and ends in one outcome:
touchdown, field goal, a defensive touchdown or safety, or no score (punt,
turnover, turnover on downs, missed field goal, end of half). Probabilities
come from:

* league rates per zone (point-in-time, from drives before the cutoff), and
* team ratings: how many more touchdowns / field goals per drive than its
  zones imply an offense scores (``off``) and a defense allows (``def``) —
  opponent-adjusted ridge fits (``ratings.fit_rating``) on team-game rows.

The next drive's start zone depends on how the previous one ended (after a
score the other team receives a kickoff; after a turnover it often starts in
plus territory), from the league transition table. The number of drives comes
from a pace rating on drives per team-game.

Simulated points feed the game model as the ``drive`` feature (the
``GameModelConfig.features`` toggle) and can be evaluated standalone with
``scripts/backtest_nfl_drive_sim.py``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from evmax.nfl_projections import data
from evmax.nfl_projections.ratings import RatingFit, fit_rating

ZONE_EDGES = (0, 49, 64, 79, 100)          # yardline_100 bins: opp territory, 50-64, 65-79, 80+
N_ZONES = len(ZONE_EDGES) - 1
OUTCOMES = ("td", "fg", "opp_td", "safety", "punt", "turnover", "downs", "missed_fg", "end_half")
SCORING = {"td": 6.95, "fg": 3.0}           # TD + league extra-point rate; defensive scores below
DEF_POINTS = {"opp_td": 6.95, "safety": 2.0}
_RESULT_MAP = {"Touchdown": "td", "Field goal": "fg", "Opp touchdown": "opp_td", "Safety": "safety",
               "Punt": "punt", "Turnover": "turnover", "Turnover on downs": "downs",
               "Missed field goal": "missed_fg", "End of half": "end_half"}
DRIVES_SCHEMA_VERSION = 1
_PBP_COLS = ["game_id", "posteam", "defteam", "home_team", "fixed_drive", "fixed_drive_result", "yardline_100",
             "play_type"]
_SCRIMMAGE = ("run", "pass", "no_play", "qb_kneel", "qb_spike", "field_goal", "punt")


def zone_of(yardline_100: np.ndarray) -> np.ndarray:
    z = np.digitize(np.asarray(yardline_100, dtype=float), ZONE_EDGES[1:-1], right=True)
    return np.clip(z, 0, N_ZONES - 1)


def build_drives(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per drive: game, offense, defense, home flag, start yardline, zone, outcome."""
    p = pbp[pbp["posteam"].notna() & pbp["fixed_drive"].notna() & pbp["play_type"].isin(_SCRIMMAGE)]
    d = p.groupby(["game_id", "fixed_drive"], sort=False).agg(
        team=("posteam", "first"), opp=("defteam", "first"), home_team=("home_team", "first"),
        start=("yardline_100", "first"), result=("fixed_drive_result", "first")).reset_index()
    d["outcome"] = d["result"].map(_RESULT_MAP)
    d = d[d["outcome"].notna() & d["start"].notna()].copy()
    d["zone"] = zone_of(d["start"].to_numpy())
    d["home"] = (d["team"] == d["home_team"]).astype(int)
    d = d.sort_values(["game_id", "fixed_drive"]).reset_index(drop=True)
    nxt = d.groupby("game_id")
    d["next_zone"] = nxt["zone"].shift(-1)
    d["next_team"] = nxt["team"].shift(-1)
    return d[["game_id", "fixed_drive", "team", "opp", "home", "start", "zone", "outcome", "next_zone", "next_team"]]


def drives_file(d: Optional[Path] = None) -> Path:
    return data.data_dir(d) / f"drives_v{DRIVES_SCHEMA_VERSION}.parquet"


def load_drives(seasons: Iterable[int], d: Optional[Path] = None, rebuild: bool = False) -> pd.DataFrame:
    """Cached drive table for ``seasons`` (rebuilt when a play-by-play file is newer)."""
    seasons = sorted(set(seasons))
    out = drives_file(d)
    newest = max((data.pbp_file(s, d).stat().st_mtime for s in seasons if data.pbp_file(s, d).exists()), default=0)
    if not rebuild and out.exists() and out.stat().st_mtime >= newest:
        dr = pd.read_parquet(out)
        have = set(dr["game_id"].str[:4].astype(int).unique())
        if {s for s in seasons if data.pbp_file(s, d).exists()} <= have:
            return dr[dr["game_id"].str[:4].astype(int).isin(seasons)].reset_index(drop=True)
    dr = build_drives(data.load_pbp(seasons, d, columns=_PBP_COLS))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".{os.getpid()}.part")
    dr.to_parquet(tmp, index=False)
    tmp.replace(out)
    return dr


@dataclass
class DriveState:
    base: np.ndarray                      # (zones, outcomes) league outcome probabilities
    transition: dict[str, np.ndarray]     # outcome -> start-zone distribution of the next drive
    kickoff: np.ndarray                   # first-drive zone distribution
    td: RatingFit
    fg: RatingFit
    pace: RatingFit
    drives_sd: float


@dataclass(frozen=True)
class DriveConfig:
    half_life_days: float = 70.0
    lam: float = 4.0
    lookback_days: int = 730
    offseason_days: float = 180.0


def team_drive_rows(drives: pd.DataFrame, base: np.ndarray, gameday: pd.Series) -> pd.DataFrame:
    """Team-game rows: drives, touchdowns / field goals over zone expectation per drive."""
    oi = {o: i for i, o in enumerate(OUTCOMES)}
    x = drives.assign(td=(drives["outcome"] == "td").astype(float), fg=(drives["outcome"] == "fg").astype(float),
                      exp_td=base[drives["zone"].to_numpy(), oi["td"]],
                      exp_fg=base[drives["zone"].to_numpy(), oi["fg"]])
    g = x.groupby(["game_id", "team", "opp", "home"]).agg(
        drives=("td", "size"), td=("td", "sum"), fg=("fg", "sum"),
        exp_td=("exp_td", "sum"), exp_fg=("exp_fg", "sum")).reset_index()
    g["td_oe"] = (g["td"] - g["exp_td"]) / g["drives"]
    g["fg_oe"] = (g["fg"] - g["exp_fg"]) / g["drives"]
    g["gameday"] = g["game_id"].map(gameday)
    return g.dropna(subset=["gameday"])


def fit_drive_state(drives: pd.DataFrame, gameday: pd.Series, cutoff: pd.Timestamp,
                    cfg: DriveConfig = DriveConfig()) -> DriveState:
    """League zone rates, transitions and team ratings from drives of games before ``cutoff``."""
    gd = drives["game_id"].map(gameday)
    w = drives[(gd < cutoff) & (gd >= cutoff - pd.Timedelta(days=cfg.lookback_days))]
    counts = pd.crosstab(w["zone"], w["outcome"]).reindex(index=range(N_ZONES), columns=list(OUTCOMES), fill_value=0)
    base = (counts.to_numpy(float) + 1.0) / (counts.to_numpy(float) + 1.0).sum(axis=1, keepdims=True)
    nxt = w[w["next_team"].notna()]
    trans = {}
    for o in OUTCOMES:
        c = np.bincount(nxt.loc[nxt["outcome"] == o, "next_zone"].astype(int), minlength=N_ZONES).astype(float) + 0.5
        trans[o] = c / c.sum()
    first = w.groupby("game_id").head(1)
    kick = np.bincount(first["zone"].astype(int), minlength=N_ZONES).astype(float) + 0.5
    rows = team_drive_rows(w, base, gameday)
    fits = {m: fit_rating(rows, m, cutoff, cfg.half_life_days, cfg.lam,
                          weight_col="drives" if m != "drives" else None, offseason_days=cfg.offseason_days)
            for m in ("td_oe", "fg_oe", "drives")}
    resid = rows["drives"] - np.array([fits["drives"].expect(t, o, h)
                                       for t, o, h in zip(rows["team"], rows["opp"], rows["home"])])
    return DriveState(base=base, transition=trans, kickoff=kick / kick.sum(), td=fits["td_oe"], fg=fits["fg_oe"],
                      pace=fits["drives"], drives_sd=float(np.nanstd(resid)) if len(resid) else 1.7)


def _outcome_probs(state: DriveState, team: str, opp: str, home: int) -> np.ndarray:
    """(zones, outcomes) probabilities for ``team`` on offense against ``opp``."""
    p = state.base.copy()
    oi = {o: i for i, o in enumerate(OUTCOMES)}
    for o, fit in (("td", state.td), ("fg", state.fg)):
        adj = fit.expect(team, opp, home) - fit.mu if not np.isnan(fit.mu) else 0.0
        p[:, oi[o]] = np.clip(p[:, oi[o]] + adj, 0.005, 0.9)
    score = p[:, [oi["td"], oi["fg"], oi["opp_td"], oi["safety"]]].sum(axis=1, keepdims=True)
    rest = [oi[o] for o in OUTCOMES if o not in ("td", "fg", "opp_td", "safety")]
    nonscore = state.base[:, rest] / state.base[:, rest].sum(axis=1, keepdims=True)
    p[:, rest] = nonscore * np.clip(1.0 - score, 0.0, None)
    return p / p.sum(axis=1, keepdims=True)


def simulate_game(state: DriveState, home_team: str, away_team: str, neutral: bool = False, n: int = 2000,
                  rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Simulated (home points, away points), each of shape (n,)."""
    rng = rng or np.random.default_rng(0)
    h = 0 if neutral else 1
    probs = {0: _outcome_probs(state, home_team, away_team, h), 1: _outcome_probs(state, away_team, home_team, 0)}
    mean_drives = state.pace.expect(home_team, away_team, h) + state.pace.expect(away_team, home_team, 0)
    total = np.clip(np.rint(rng.normal(mean_drives, state.drives_sd * np.sqrt(2), size=n)), 12, 32).astype(int)
    pts = np.zeros((n, 2))
    side = rng.integers(0, 2, size=n)                     # who receives the opening kickoff
    zone = rng.choice(N_ZONES, size=n, p=state.kickoff)
    cum = {s: np.cumsum(probs[s], axis=1) for s in (0, 1)}
    trans_cum = np.cumsum(np.stack([state.transition[o] for o in OUTCOMES]), axis=1)
    pts_off = np.array([SCORING.get(o, 0.0) for o in OUTCOMES])
    pts_def = np.array([DEF_POINTS.get(o, 0.0) for o in OUTCOMES])
    idx = np.arange(n)
    for k in range(int(total.max())):
        live = k < total
        u = rng.random(n)
        c = np.where(side[:, None] == 0, cum[0][zone], cum[1][zone])
        out = (u[:, None] > c).sum(axis=1).clip(0, len(OUTCOMES) - 1)
        pts[idx, side] += np.where(live, pts_off[out], 0.0)
        pts[idx, 1 - side] += np.where(live, pts_def[out], 0.0)
        u2 = rng.random(n)
        zone = (u2[:, None] > trans_cum[out]).sum(axis=1).clip(0, N_ZONES - 1)
        # After a defensive touchdown the scoring team kicks off: the offense has the ball again.
        side = np.where(out == OUTCOMES.index("opp_td"), side, 1 - side)
    return pts[:, 0], pts[:, 1]
