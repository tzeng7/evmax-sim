"""Walk-forward lens for NFL efficiency RATING quality (standalone, no market inputs).

Question: does a different way of turning play-by-play into team ratings predict
game winners better than the shipped one-pass opponent adjustment?

Protocol (leak-free, deterministic):
  * Ratings for week W of season S use only plays with game_date < the earliest
    kickoff date of (S, W), so a team's own game never sits in its rating.
  * Every variant is scored through the SAME 2-parameter logistic map
    P(home) = sigmoid(a*delta + b), fitted ONLY on earlier seasons' games
    (expanding window from FIT_START). That removes the scale mismatch
    (the shipped 64 pts/EPA is ~1.7x too wide) so the metric compares
    rating QUALITY, not a tuned constant.
  * DEV seasons (2023, 2024) are where choices are made. HOLDOUT (2025) is
    scored once at the end (--holdout).
  * Game set = regular-season games where every variant produced a rating for
    both teams, so Brier is comparable across variants.

Usage:
    python scripts/exp_nfl_eff_ridge.py                  # dev only
    python scripts/exp_nfl_eff_ridge.py --variants base,ridge:500
    python scripts/exp_nfl_eff_ridge.py --holdout        # also score 2025
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import nflreadpy as nfl

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.seed_nfl_efficiency import SEASON_DECAY, compute_team_stats, filter_valid_pbp

LOAD_FROM = 2014
FIT_START = 2017  # first season whose games train the logistic map
RANK = (2020, 2021, 2022)  # choose hyperparameters here
DEV = (2023, 2024)  # confirm here
HOLDOUT = (2025,)


# ---------------------------------------------------------------- data

def load():
    seasons = list(range(LOAD_FROM, max(HOLDOUT) + 1))
    pbp = nfl.load_pbp(seasons=seasons)
    pbp = pbp.with_columns(pl.col("game_date").cast(pl.Date))
    games = (
        pbp.filter(pl.col("season_type") == "REG")
        .group_by("game_id")
        .agg(
            pl.col("season").first(),
            pl.col("week").first(),
            pl.col("game_date").first(),
            pl.col("home_team").first(),
            pl.col("away_team").first(),
            pl.col("home_score").max(),
            pl.col("away_score").max(),
        )
        .drop_nulls(["home_score", "away_score"])
        .sort(["season", "week", "game_id"])
    )
    plays = filter_valid_pbp(pbp)
    return plays, games


# ---------------------------------------------------------------- variants
# Each rating fn: (plays_before: pl.DataFrame, cur_season: int, **params)
#   -> dict[team_abbrev] -> net rating (higher = better)

def rate_base(df, cur, **_):
    stats = compute_team_stats(df, cur)
    return {s["abbrev"]: s["off_epa_adj"] - s["def_epa_adj"] for s in stats.values()}


def _decay_weight(df, cur, decay, week_hl=None):
    w = pl.lit(decay) ** (cur - pl.col("season"))
    if week_hl:  # within-season recency: halve weight every `week_hl` weeks of age
        # age in weeks measured from the latest play in the slice
        newest = df["game_date"].max()
        age_w = (pl.lit(newest) - pl.col("game_date")).dt.total_days() / 7.0
        w = w * (0.5 ** (age_w / week_hl))
    return df.with_columns(w.alias("w"))


def _ridge(cells, teams, lam, use_home=True):
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    p = 2 + 2 * n  # mu, h, off[n], def_allowed[n]
    rows = cells.iter_rows()
    X = np.zeros((cells.height, p))
    y = np.zeros(cells.height)
    sw = np.zeros(cells.height)
    for r, (pos, dfn, home, w, wy) in enumerate(rows):
        X[r, 0] = 1.0
        X[r, 1] = float(home) if use_home else 0.0
        X[r, 2 + idx[pos]] = 1.0
        X[r, 2 + n + idx[dfn]] = 1.0
        sw[r] = w
        y[r] = wy / w
    lamv = np.full(p, lam)
    lamv[:2] = 1e-6
    A = (X * sw[:, None]).T @ X + np.diag(lamv)
    b = (X * sw[:, None]).T @ y
    beta = np.linalg.solve(A, b)
    off = beta[2:2 + n]
    dfa = beta[2 + n:]
    return {t: float(off[idx[t]] - dfa[idx[t]]) for t in teams}


def rate_ridge(df, cur, lam=1000.0, decay=SEASON_DECAY, week_hl=None, clip=None, target=0.0, **_):
    """target=0 -> EPA/play; target=1 -> success rate scaled to EPA units; 0<t<1 mixes them."""
    df = _decay_weight(df, cur, decay, week_hl)
    if clip:
        df = df.with_columns(pl.col("epa").clip(-clip, clip))
    if target:
        df = df.with_columns(((1 - target) * pl.col("epa") + target * (pl.col("success") - 0.45) * 1.2).alias("epa"))
    df = df.with_columns((pl.col("posteam") == pl.col("home_team")).alias("home_off"))
    cells = df.group_by(["posteam", "defteam", "home_off"]).agg(
        pl.col("w").sum().alias("sw"), (pl.col("w") * pl.col("epa")).sum().alias("swy")
    )
    teams = sorted(set(cells["posteam"].to_list()) | set(cells["defteam"].to_list()))
    return _ridge(cells, teams, lam)


def rate_shared(df, cur, lam=None, week_hl=None, **_):
    """The production module (evmax.agents.models._nfl_ridge) — must equal ridge decay=1."""
    from evmax.agents.models._nfl_ridge import RIDGE_LAMBDA, WEEK_HALF_LIFE, ridge_ratings

    r = ridge_ratings(df, lam or RIDGE_LAMBDA, week_hl if week_hl is not None else WEEK_HALF_LIFE)
    return {t: v["net"] for t, v in r.items()}


VARIANTS = {"base": rate_base, "ridge": rate_ridge, "shared": rate_shared}


def parse_variant(spec):
    name, _, arg = spec.partition(":")
    kw = {}
    if arg:
        for part in arg.split("/"):
            k, _, v = part.partition("=")
            if not v:  # bare number -> lam
                k, v = "lam", k
            kw[k] = float(v)
    return name, kw


# ---------------------------------------------------------------- scoring

def fit_logistic(d, y, iters=50):
    a, b = 1.0, 0.0
    for _ in range(iters):
        z = a * d + b
        p = 1 / (1 + np.exp(-z))
        g = np.array([np.sum((p - y) * d), np.sum(p - y)])
        wgt = p * (1 - p) + 1e-9
        H = np.array([[np.sum(wgt * d * d), np.sum(wgt * d)], [np.sum(wgt * d), np.sum(wgt)]])
        H += np.eye(2) * 1e-6
        step = np.linalg.solve(H, g)
        a, b = a - step[0], b - step[1]
        if np.max(np.abs(step)) < 1e-8:
            break
    return a, b


def ratings_by_week(plays, games, fn, kw):
    out = {}
    keys = games.group_by(["season", "week"]).agg(pl.col("game_date").min().alias("cut")).sort(["season", "week"])
    for season, week, cut in keys.iter_rows():
        if season < FIT_START:
            continue
        sl = plays.filter(pl.col("game_date") < cut)
        out[(season, week)] = fn(sl, season, **kw)
    return out


def deltas(games, rbw):
    rows = []
    for g in games.iter_rows(named=True):
        r = rbw.get((g["season"], g["week"]))
        if r is None:
            continue
        h, a = r.get(g["home_team"]), r.get(g["away_team"])
        if h is None or a is None:
            continue
        rows.append((g["game_id"], g["season"], h - a, 1.0 if g["home_score"] > g["away_score"] else (0.5 if g["home_score"] == g["away_score"] else 0.0)))
    return rows


def per_game_se(rows, eval_seasons, common_ids):
    d = np.array([r[2] for r in rows if r[0] in common_ids])
    y = np.array([r[3] for r in rows if r[0] in common_ids])
    s = np.array([r[1] for r in rows if r[0] in common_ids])
    ids = [r[0] for r in rows if r[0] in common_ids]
    out = {}
    for ev in eval_seasons:
        a, b = fit_logistic(d[s < ev], y[s < ev])
        te = np.where(s == ev)[0]
        p = 1 / (1 + np.exp(-(a * d[te] + b)))
        for i, pi in zip(te, p):
            out[ids[i]] = (pi - y[i]) ** 2
    return out


def fit_margin(d, y, sigma=13.5):
    """Probit fit P(home) = Phi((k*d + h)/sigma); returns (k, h) in margin points."""
    from math import erf, sqrt
    from scipy.optimize import minimize

    def nll(v):
        z = (v[0] * d + v[1]) / sigma
        p = np.clip(0.5 * (1 + np.vectorize(erf)(z / sqrt(2))), 1e-6, 1 - 1e-6)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    r = minimize(nll, x0=[40.0, 2.0], method="Nelder-Mead")
    return float(r.x[0]), float(r.x[1])


def fit_margin_report(rows, common):
    d = np.array([r[2] for r in rows if r[0] in common])
    y = np.array([r[3] for r in rows if r[0] in common])
    s = np.array([r[1] for r in rows if r[0] in common])
    parts = []
    for ev in (2023, 2024, 2025, 2026):
        tr = s < ev
        if tr.sum() < 500:
            continue
        k, h = fit_margin(d[tr], y[tr])
        parts.append(f"{ev}={k:.1f}:{h:.2f}")
    return " ".join(parts)


def score(rows, eval_seasons, common_ids):
    d = np.array([r[2] for r in rows if r[0] in common_ids])
    y = np.array([r[3] for r in rows if r[0] in common_ids])
    s = np.array([r[1] for r in rows if r[0] in common_ids])
    res = {}
    for ev in eval_seasons:
        tr = s < ev
        te = s == ev
        a, b = fit_logistic(d[tr], y[tr])
        p = 1 / (1 + np.exp(-(a * d[te] + b)))
        res[ev] = (float(np.mean((p - y[te]) ** 2)), int(te.sum()), a, b)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="base,ridge:1000")
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--fit-margin", action="store_true",
                    help="probit-fit margin = k*delta + h (points, sigma fixed 13.5) per eval season on EARLIER seasons")
    args = ap.parse_args()

    plays, games = load()
    specs = [parse_variant(v) for v in args.variants.split(",")]
    per = {}
    for name, kw in specs:
        per[(name, tuple(sorted(kw.items())))] = deltas(games, ratings_by_week(plays, games, VARIANTS[name], kw))
    common = set.intersection(*[{r[0] for r in rows} for rows in per.values()])
    evals = RANK + DEV + (HOLDOUT if args.holdout else ())
    print(f"common games: {len(common)}")
    base_se = None
    for (name, kw), rows in per.items():
        res = score(rows, evals, common)
        def pooled(group):
            xs = [res[e] for e in group]
            return sum(x[0] * x[1] for x in xs) / sum(x[1] for x in xs)
        line = f"{name}{dict(kw)}  rank={pooled(RANK):.4f} confirm={pooled(DEV):.4f}"
        if args.holdout:
            line += f"  holdout={res[HOLDOUT[0]][0]:.4f}"
        if args.fit_margin:
            print("  margin fit (expanding, earlier seasons only): " + fit_margin_report(rows, common))
        se = per_game_se(rows, RANK + DEV, common)
        if base_se is None:
            base_se = se
        else:
            diff = np.array([se[g] - base_se[g] for g in se])
            line += f"  d_vs_first={1000 * diff.mean():+.2f}/1000 z={diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))):+.2f}"
        print(line)


if __name__ == "__main__":
    main()
