"""Fit and validate the alternate-line models (evmax/models_ml/line_pmf.py).

One model per sector and market:

    python scripts/fit_line_pmf.py --sector nfl   --market total
    python scripts/fit_line_pmf.py --sector nba   --market spread
    python scripts/fit_line_pmf.py --sector ncaab --market total --dry-run

Data
    nfl            the nflverse schedule (closing spread, total, final score)
    nba/ncaab/ncaaw  data/backtest/line_history/{league}.csv
                   (scripts/build_line_history.py: ESPN closing lines + scores)

Protocol (the same one as scripts/fit_nfl_margin_pmf.py)
    1. HOLDOUT = the last --holdout-seasons seasons. Everything before is FIT.
    2. The ridge strength on the key-number multipliers is chosen on an INNER
       split of the fit seasons (last fit season = validation), never on the
       holdout.
    3. The chosen configuration is refit on all fit seasons and scored ONCE on
       the holdout: every half-point rung within --rung-range points of the
       closing line, Brier score against the realized result, paired by rung
       and clustered by game. It is compared with
         * the PRODUCTION normal curve (the sigma the scanner uses today), and
         * a normal curve with the sigma re-estimated on the fit seasons,
       so a gain from a better sigma is not mistaken for a gain from shape.
    4. The artifact is then fit on ALL seasons with that frozen configuration
       and written with the holdout numbers inside it.

The closing line is taken at even odds (the push-chart convention): the model
is anchored so the favorite covers / the game goes over half the time,
conditional on no push. Negative Brier differences mean the model is better.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from evmax.models_ml.line_pmf import FitConfig, LinePMF, artifact_path, fit_line_pmf  # noqa: E402

PROD_SIGMA = {
    ("nfl", "spread"): 14.0, ("nba", "spread"): 12.5, ("ncaab", "spread"): 12.5,
    ("ncaaw", "spread"): 11.5,          # no ncaaw entry in production: the 11.5 default
    ("nfl", "total"): 14.0, ("nba", "total"): 22.0, ("ncaab", "total"): 14.0,
    ("ncaaw", "total"): 18.0,           # production default
}
# Candidate ridge strengths for the key multipliers (None = smooth base only).
LAMBDAS = (None, 200.0, 50.0, 15.0)
RUNG_RANGE = {"spread": 14.0, "total": 20.0}


def load_games(sector: str, nfl_games: str, history_dir: str) -> list[dict]:
    """Rows of {season, home_margin, total_points, home_spread, total}."""
    out = []
    if sector == "nfl":
        import pandas as pd

        df = pd.read_parquet(nfl_games)
        df = df[df["home_score"].notna() & df["away_score"].notna()]
        for r in df.itertuples():
            spread = None if r.spread_line != r.spread_line else -float(r.spread_line)
            total = None if r.total_line != r.total_line else float(r.total_line)
            out.append({"season": int(r.season), "game": str(r.game_id),
                        "home_margin": float(r.home_score - r.away_score),
                        "total_points": float(r.home_score + r.away_score),
                        "home_spread": spread, "total": total})
        return out
    path = Path(history_dir) / f"{sector}.csv"
    with open(path) as f:
        for r in csv.DictReader(f):
            hs, as_ = float(r["home_score"]), float(r["away_score"])
            out.append({"season": int(r["season"]), "game": r["event_id"],
                        "home_margin": hs - as_, "total_points": hs + as_,
                        "home_spread": float(r["home_spread"]) if r["home_spread"] else None,
                        "total": float(r["total"]) if r["total"] else None})
    return out


def to_xy(games: list[dict], market: str):
    """(outcome, main line, game total, season, game id) arrays on the model's axis."""
    xs, ls, ts, ss, gs = [], [], [], [], []
    for g in games:
        if market == "total":
            if g["total"] is None:
                continue
            xs.append(g["total_points"])
            ls.append(g["total"])
            ts.append(g["total"])
        else:
            h = g["home_spread"]
            if h is None:
                continue
            # favorite margin; the home team is the "favorite" at a pick'em
            xs.append(g["home_margin"] if h <= 0 else -g["home_margin"])
            ls.append(abs(h))
            ts.append(g["total"] if g["total"] is not None else np.nan)
        ss.append(g["season"])
        gs.append(g["game"])
    return (np.array(xs), np.array(ls), np.array(ts), np.array(ss), np.array(gs))


def base_config(sector: str, market: str, lam: float | None, x: np.ndarray,
                line: np.ndarray) -> FitConfig:
    football = sector in ("nfl", "ncaaf")
    if market == "spread":
        window = None if lam is None else ((-30, 30) if football else (-15, 15))
        return FitConfig(family="t", exclude_zero=not football, beta_window=window,
                         lam=lam or 0.0, use_total_scale=not football)
    lo, hi = np.percentile(x, [1, 99])
    window = None if lam is None else (int(lo), int(hi))
    return FitConfig(family="t", overtime=not football, beta_window=window, lam=lam or 0.0)


def fit(sector, market, x, line, tot, cfg) -> LinePMF:
    totals = None if (not cfg.use_total_scale or np.isnan(tot).any()) else tot
    if cfg.use_total_scale and totals is None:
        cfg = FitConfig(**{**cfg.__dict__, "use_total_scale": False})
    return fit_line_pmf(sector, market, x, line, totals, cfg)


def mean_nll(model: LinePMF, x, line, tot) -> float:
    """Out-of-sample log loss with the closing line (+ fitted offset) as location."""
    use_tot = model.s2 != 0.0 and not np.isnan(tot).any()
    total = 0.0
    for i in range(len(x)):
        p = model.pmf(line[i] + model.c0, line[i], tot[i] if use_tot else None)
        k = int(round(x[i])) - model.k_min
        total -= math.log(max(p[k] if 0 <= k < len(p) else 0.0, 1e-12))
    return total / len(x)


def _ncdf(z: np.ndarray) -> np.ndarray:
    from scipy.special import ndtr

    return ndtr(z)


def evaluate(model: LinePMF, x, line, tot, games, sigma_prod: float, sigma_fit: float,
             rung_range: float) -> dict:
    """Paired Brier on synthetic half-point rungs, clustered by game."""
    offsets = np.arange(-rung_range, rung_range + 0.01, 1.0)
    use_tot = model.s2 != 0.0 and not np.isnan(tot).any()
    per_game = {"model": [], "prod": [], "refit": []}
    bands: dict[str, list] = {"0-3": [], "3-7": [], "7+": []}
    for i in range(len(x)):
        a = line[i]
        mu = model.anchor(a, 0.5, tot[i] if use_tot else None)
        if mu is None:
            continue
        # half-point rungs only: no pushes, so every rung has a clean 0/1 result
        base = math.floor(a) + 0.5
        rungs = base + offsets
        pm = np.array([model.conditional_survival(mu, t, a, tot[i] if use_tot else None)
                       for t in rungs])
        y = (x[i] > rungs).astype(float)
        pp = 1.0 - _ncdf((rungs - a) / sigma_prod)
        pr = 1.0 - _ncdf((rungs - a) / sigma_fit)
        bm, bp, br = (pm - y) ** 2, (pp - y) ** 2, (pr - y) ** 2
        per_game["model"].append(bm.mean())
        per_game["prod"].append(bp.mean())
        per_game["refit"].append(br.mean())
        d = np.abs(rungs - a)
        for name, mask in (("0-3", d <= 3), ("3-7", (d > 3) & (d <= 7)), ("7+", d > 7)):
            if mask.any():
                bands[name].append((bm[mask].mean(), bp[mask].mean(), br[mask].mean()))

    def paired(a_, b_):
        d = (np.array(a_) - np.array(b_)) * 1000.0
        se = d.std(ddof=1) / math.sqrt(len(d))
        return {"delta_per_1000": round(float(d.mean()), 3),
                "z": round(float(d.mean() / se), 2) if se > 0 else None}

    out = {
        "n_games": len(per_game["model"]), "rungs_per_game": len(offsets),
        "brier_model": round(float(np.mean(per_game["model"])), 6),
        "brier_production_normal": round(float(np.mean(per_game["prod"])), 6),
        "brier_refit_normal": round(float(np.mean(per_game["refit"])), 6),
        "vs_production_normal": paired(per_game["model"], per_game["prod"]),
        "vs_refit_normal": paired(per_game["model"], per_game["refit"]),
        "by_distance": {},
    }
    for name, vals in bands.items():
        if vals:
            v = np.array(vals)
            out["by_distance"][name] = {
                "vs_production_normal": paired(v[:, 0], v[:, 1]),
                "vs_refit_normal": paired(v[:, 0], v[:, 2])}
    return out


def top_keys(model: LinePMF, n: int = 8) -> list[tuple[int, float]]:
    items = sorted(model.beta.items(), key=lambda kv: -abs(kv[1]))[:n]
    return [(k, round(math.exp(v), 3)) for k, v in items]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sector", required=True, choices=["nfl", "nba", "ncaab", "ncaaw"])
    ap.add_argument("--market", required=True, choices=["spread", "total"])
    ap.add_argument("--nfl-games", default=str(
        _REPO / "data" / "backtest" / "nfl_projections" / "games.parquet"))
    ap.add_argument("--history-dir", default=str(_REPO / "data" / "backtest" / "line_history"))
    ap.add_argument("--first-season", type=int, default=None)
    ap.add_argument("--holdout-seasons", type=int, default=1)
    ap.add_argument("--rung-range", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true", help="fit and evaluate, write nothing")
    args = ap.parse_args()

    sector, market = args.sector, args.market
    games = load_games(sector, args.nfl_games, args.history_dir)
    x, line, tot, season, gid = to_xy(games, market)
    first = args.first_season or (2015 if sector == "nfl" else int(season.min()))
    keep = season >= first
    x, line, tot, season, gid = x[keep], line[keep], tot[keep], season[keep], gid[keep]
    seasons = sorted(set(season.tolist()))
    if len(seasons) < 3:
        print(f"need at least 3 seasons, have {seasons}")
        return 1
    hold = seasons[-args.holdout_seasons:]
    fit_seasons = [s for s in seasons if s not in hold]
    inner_val = fit_seasons[-1]
    m_fit, m_hold = np.isin(season, fit_seasons), np.isin(season, hold)
    m_in = np.isin(season, fit_seasons[:-1])
    m_val = season == inner_val
    rung_range = args.rung_range or RUNG_RANGE[market]
    print(f"{sector} {market}: {len(x)} games, seasons {seasons[0]}-{seasons[-1]} | "
          f"fit {fit_seasons[0]}-{fit_seasons[-1]} ({m_fit.sum()}) | holdout {hold} "
          f"({m_hold.sum()}) | residual SD around the close {np.std(x - line):.2f}")

    # 1. choose the ridge strength on the inner split
    best = None
    for lam in LAMBDAS:
        cfg = base_config(sector, market, lam, x[m_in], line[m_in])
        model = fit(sector, market, x[m_in], line[m_in], tot[m_in], cfg)
        nll = mean_nll(model, x[m_val], line[m_val], tot[m_val])
        print(f"   inner split (validate {inner_val}): lam={lam}  validation log loss {nll:.5f}")
        if best is None or nll < best[0] - 1e-5:
            best = (nll, lam)
    lam = best[1]
    print(f"   chosen: lam={lam}")

    # 2. refit on the fit seasons, score the holdout once
    cfg = base_config(sector, market, lam, x[m_fit], line[m_fit])
    model = fit(sector, market, x[m_fit], line[m_fit], tot[m_fit], cfg)
    sigma_fit = float(np.std(x[m_fit] - line[m_fit]))
    sigma_prod = PROD_SIGMA[(sector, market)]
    ev = evaluate(model, x[m_hold], line[m_hold], tot[m_hold], gid[m_hold], sigma_prod,
                  sigma_fit, rung_range)
    print(f"\n   fitted on {fit_seasons[0]}-{fit_seasons[-1]}: scale s0={model.s0:.2f} "
          f"s1={model.s1:+.3f}/pt s2={model.s2:+.3f}/total-pt nu={model.nu:.1f} "
          f"offset={model.c0:+.2f} overtime p={model.p_ot:.3f} +{model.d_ot:.1f}pts")
    if model.beta:
        print(f"   largest key multipliers (outcome, x): {top_keys(model)}")
    print(f"\n   HOLDOUT {hold}: {ev['n_games']} games x {ev['rungs_per_game']} rungs")
    print(f"     Brier  model {ev['brier_model']:.5f} | production normal "
          f"(sigma {sigma_prod}) {ev['brier_production_normal']:.5f} | refit normal "
          f"(sigma {sigma_fit:.2f}) {ev['brier_refit_normal']:.5f}")
    for name, key in (("vs production normal", "vs_production_normal"),
                      ("vs refit normal", "vs_refit_normal")):
        v = ev[key]
        print(f"     {name:<22} {v['delta_per_1000']:+.3f} per 1000  z {v['z']}")
    for band, v in ev["by_distance"].items():
        print(f"       rungs {band:>4} pts from the line: vs production "
              f"{v['vs_production_normal']['delta_per_1000']:+.3f} (z "
              f"{v['vs_production_normal']['z']}) | vs refit "
              f"{v['vs_refit_normal']['delta_per_1000']:+.3f} (z {v['vs_refit_normal']['z']})")

    # 3. final artifact on all seasons, frozen configuration
    cfg = base_config(sector, market, lam, x, line)
    final = fit(sector, market, x, line, tot, cfg)
    final.meta.update({
        "created": date.today().isoformat(),
        "source": "nflverse schedule" if sector == "nfl" else "ESPN closing lines + scores",
        "seasons": [int(seasons[0]), int(seasons[-1])],
        "fit_seasons": [int(fit_seasons[0]), int(fit_seasons[-1])],
        "holdout_seasons": [int(h) for h in hold],
        "lam": lam, "sigma_fit": round(sigma_fit, 3), "sigma_production": sigma_prod,
        "validation": ev,
    })
    out = artifact_path(sector, market)
    if args.dry_run:
        print(f"\n   dry run: {out} not written")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(final.to_artifact(), indent=1))
        print(f"\n   wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
