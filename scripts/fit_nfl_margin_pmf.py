"""Fit the NFL key-number margin PMF and write data/models/nfl_margin_pmf.json.

The artifact feeds ``evmax/models_ml/spread_pmf.py::MarginPMF``, which prices
NFL alt-spread rungs in ``SpreadDistributionModel`` in place of the normal CDF
(``_PMF_SECTORS`` in ``spread_distribution.py``).

Model (favorite-margin axis, F = favorite's final margin):

    P(F = k | mu) ∝ exp( -(k - mu)² / (2·s(mu)²) + beta_k + gamma[bucket(mu), k] )

    s(mu)  = s0 + s1·mu                  spread-dependent kernel width
    beta_k = signed key-number log-multiplier for |k| <= K (ridge lam)
    gamma  = per-bucket deviation, buckets mu<=3 / 3<mu<=7 / mu>7 (ridge lam_bucket)

Training uses the closing spread as mu (nflverse ``spread_line``, REG + POST),
so the bucket covariate is the spread bucket. Pricing keeps the bucket of the
main line and anchors mu on Pinnacle's devigged main-line price (see
spread_pmf.py for why the bucket is not taken from the anchored mu).

The hyperparameters in ``FROZEN_CONFIG`` were selected ONCE by the 2026-09-22
research on an inner split (fit 2003-2014, validate 2015-2018) and then
evaluated once on 2019-2025. Do NOT re-tune them on the holdout seasons. This
script only refits the parameters on more seasons. Run it each offseason after
the Super Bowl.

Evaluation printed by the script (the same protocol as the research):
  * holdout: fit [fit_start, holdout_start-1], score [holdout_start, fit_end]
    ONCE on every half-point rung within 14 points of the closing spread.
  * walk-forward: for each season Y in the holdout range, fit on
    [fit_start, Y-1] and score Y.
Both compare the PMF anchored on the closing juice against the production
normal (σ=14, mean inferred from the same juice). Brier differences are paired
and clustered by game. Negative ΔBrier = the PMF is better.

Usage:
    uv run python scripts/fit_nfl_margin_pmf.py                # fit, evaluate, write
    uv run python scripts/fit_nfl_margin_pmf.py --dry-run      # fit, evaluate, write nothing
    uv run python scripts/fit_nfl_margin_pmf.py --schedules cached.parquet   # offline input
    uv run python scripts/fit_nfl_margin_pmf.py --no-eval      # fit + write only (fast)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evmax.models_ml.spread_pmf import SCHEMA_VERSION, MarginPMF, artifact_path  # noqa: E402

# Frozen by the 2026-09-22 research (inner validation 2003-14 → 2015-18).
FROZEN_CONFIG: dict = {
    "K": 35,
    "lam": 5.0,
    "sigma_slope": True,
    "spread_buckets": True,
    "lam_bucket": 10.0,
    "bucket_edges": [3.0, 7.0],
}
FIT_START = 2003          # first season with nflverse closing spread juice
HOLDOUT_START = 2019      # first holdout / walk-forward season
KS = np.arange(-80, 91)   # support of F (integer margins)
EVAL_MAX_OFFSET = 14.0    # rungs scored: |t - a| <= 14 (the production 1σ gate)
PROD_SIGMA = 14.0         # _SECTOR_SIGMA["nfl"] — the normal this replaces
EVAL_CLIP = (0.005, 0.995)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def last_complete_season(today: Optional[date] = None) -> int:
    """Latest NFL season whose Super Bowl has been played (season = start year)."""
    today = today or date.today()
    return today.year - 1 if today.month >= 3 else today.year - 2


def load_schedules(first: int, last: int, parquet: Optional[str] = None) -> pd.DataFrame:
    """nflverse schedules as pandas. ``parquet`` reads a cached copy instead."""
    if parquet:
        df = pd.read_parquet(parquet)
    else:
        import nflreadpy as nfl

        df = nfl.load_schedules(seasons=list(range(first, last + 1)))
        df = df.to_pandas() if hasattr(df, "to_pandas") else df
    return df[(df.season >= first) & (df.season <= last)].copy()


def _american_to_decimal(o: np.ndarray) -> np.ndarray:
    o = o.astype(float)
    return np.where(o > 0, 1.0 + o / 100.0, 1.0 + 100.0 / np.abs(o))


def prepare_games(df: pd.DataFrame, first: int, last: int) -> pd.DataFrame:
    """One row per completed game with a closing spread.

    Columns added: ``a`` = |closing spread| (favorite's line magnitude),
    ``F`` = favorite's final margin, ``p_fav`` = multiplicatively devigged
    P(favorite covers the closing spread) from nflverse spread juice (NaN when
    the juice is missing). nflverse ``spread_line`` > 0 means the HOME team is
    favored and ``result`` = home - away, so F = result · sign(spread_line).
    A pick'em (spread 0) takes the home team as the favorite.
    """
    df = df[(df.season >= first) & (df.season <= last)
            & df.result.notna() & df.spread_line.notna()].copy()
    s = df.spread_line.astype(float).to_numpy()
    sign = np.where(s >= 0, 1.0, -1.0)
    df["a"] = np.abs(s)
    df["F"] = (df.result.astype(float).to_numpy() * sign).astype(int)
    ho = df.home_spread_odds.to_numpy(dtype=float)
    ao = df.away_spread_odds.to_numpy(dtype=float)
    fav_o = np.where(s >= 0, ho, ao)
    dog_o = np.where(s >= 0, ao, ho)
    ok = np.isfinite(fav_o) & np.isfinite(dog_o)
    p = np.full(len(df), np.nan)
    if ok.any():
        fd = _american_to_decimal(fav_o[ok])
        dd = _american_to_decimal(dog_o[ok])
        p[ok] = (1.0 / fd) / ((1.0 / fd) + (1.0 / dd))
    df["p_fav"] = p
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Fitter (ridge-penalized multinomial likelihood, analytic gradient, L-BFGS-B)
# ---------------------------------------------------------------------------

class HybridMarginFitter:
    """Fit s0, s1, beta, gamma by penalized maximum likelihood on (mu=a, F)."""

    def __init__(self, K: int, lam: float, sigma_slope: bool, spread_buckets: bool,
                 lam_bucket: float, bucket_edges: list[float]) -> None:
        self.K, self.lam, self.sig_slope = K, lam, sigma_slope
        self.buckets, self.lam_b = spread_buckets, lam_bucket
        self.edges = (float(bucket_edges[0]), float(bucket_edges[1]))
        self.bk = np.where(np.abs(KS) <= K)[0]          # support indices carrying beta
        self.th: Optional[np.ndarray] = None
        self.fit_ok = False

    def bucket(self, mu: np.ndarray) -> np.ndarray:
        return np.where(mu <= self.edges[0], 0, np.where(mu <= self.edges[1], 1, 2))

    def unpack(self, th: np.ndarray):
        nb = len(self.bk)
        s0 = th[0]
        s1 = th[1] if self.sig_slope else 0.0
        beta = np.zeros(len(KS))
        beta[self.bk] = th[2: 2 + nb]
        gam = np.zeros((3, len(KS)))
        if self.buckets:
            off = 2 + nb
            for b in range(3):
                gam[b, self.bk] = th[off: off + nb]
                off += nb
        return s0, s1, beta, gam

    def _logits(self, mu: np.ndarray, th: np.ndarray):
        s0, s1, beta, gam = self.unpack(th)
        sig = s0 + s1 * mu
        L = -((KS[None, :] - mu[:, None]) ** 2) / (2 * sig[:, None] ** 2) + beta[None, :]
        if self.buckets:
            L = L + gam[self.bucket(mu)]
        return L, sig

    def _nll(self, th: np.ndarray, mu: np.ndarray, F: np.ndarray, w: np.ndarray):
        L, sig = self._logits(mu, th)
        m = L.max(1, keepdims=True)
        E = np.exp(L - m)
        Z = E.sum(1, keepdims=True)
        P = E / Z
        idx = F - KS[0]
        ll = L[np.arange(len(F)), idx] - (m[:, 0] + np.log(Z[:, 0]))
        s0, s1, beta, gam = self.unpack(th)
        nll = -np.sum(w * ll) + self.lam * np.sum(beta ** 2)
        if self.buckets:
            nll += self.lam_b * np.sum(gam ** 2)
        # gradient
        Y = np.zeros_like(P)
        Y[np.arange(len(F)), idx] = 1.0
        G = (Y - P) * w[:, None]                        # d ll / d logit
        g = np.zeros_like(th)
        dL_dsig = ((KS[None, :] - mu[:, None]) ** 2) / sig[:, None] ** 3
        gs = (G * dL_dsig).sum(1)
        g[0] = -gs.sum()
        if self.sig_slope:
            g[1] = -(gs * mu).sum()
        nb = len(self.bk)
        g[2: 2 + nb] = -G[:, self.bk].sum(0) + 2 * self.lam * beta[self.bk]
        if self.buckets:
            b = self.bucket(mu)
            off = 2 + nb
            for bb in range(3):
                g[off: off + nb] = -G[b == bb][:, self.bk].sum(0) + 2 * self.lam_b * gam[bb, self.bk]
                off += nb
        return nll, g

    def fit(self, a: np.ndarray, F: np.ndarray, w: Optional[np.ndarray] = None) -> "HybridMarginFitter":
        a = np.asarray(a, float)
        F = np.asarray(F, int)
        if F.min() < KS[0] or F.max() > KS[-1]:
            raise ValueError(f"margin outside support [{KS[0]}, {KS[-1]}]")
        w = np.ones(len(a)) if w is None else np.asarray(w, float)
        nb = len(self.bk)
        th0 = np.zeros(2 + nb * (4 if self.buckets else 1))
        th0[0] = 13.0
        bounds = [(8, 20), (-0.5, 0.5)] + [(-6, 6)] * (len(th0) - 2)
        r = minimize(self._nll, th0, args=(a, F, w), jac=True, method="L-BFGS-B",
                     bounds=bounds, options={"maxiter": 3000})
        self.th, self.fit_ok = r.x, bool(r.success)
        return self

    def to_margin_pmf(self) -> MarginPMF:
        """The fitted parameters as the runtime MarginPMF (the priced object)."""
        return MarginPMF.from_artifact(self.to_artifact_params(round_to=None))

    def to_artifact_params(self, round_to: Optional[int] = 5) -> dict:
        s0, s1, beta, gam = self.unpack(self.th)
        rnd = (lambda x: round(float(x), round_to)) if round_to is not None else float
        cfg = dict(K=self.K, lam=self.lam, sigma_slope=self.sig_slope,
                   spread_buckets=self.buckets, lam_bucket=self.lam_b,
                   bucket_edges=list(self.edges))
        return {
            "schema_version": SCHEMA_VERSION,
            "sector": "nfl",
            "config": cfg,
            "s0": rnd(s0),
            "s1": rnd(s1),
            "ks": [int(k) for k in KS],
            "beta": [rnd(x) for x in beta],
            "gamma": [[rnd(x) for x in g] for g in gam],
        }


def fit_frozen(games: pd.DataFrame) -> HybridMarginFitter:
    cfg = FROZEN_CONFIG
    return HybridMarginFitter(cfg["K"], cfg["lam"], cfg["sigma_slope"], cfg["spread_buckets"],
                              cfg["lam_bucket"], cfg["bucket_edges"]).fit(
        games.a.to_numpy(), games.F.to_numpy())


# ---------------------------------------------------------------------------
# Evaluation (same protocol as the research)
# ---------------------------------------------------------------------------

def rung_grid(games: pd.DataFrame, max_offset: float = EVAL_MAX_OFFSET):
    """Half-point thresholds t with |t - a| <= max_offset on the F axis.

    Returns t (n,m), mask (n,m), y = 1[F > t] (n,m), d = t - a (n,m).
    """
    a = games.a.to_numpy()
    F = games.F.to_numpy()
    base = np.floor(a - max_offset) + 0.5
    m = int(2 * max_offset + 2)
    t = base[:, None] + np.arange(m)[None, :]
    d = t - a[:, None]
    mask = np.abs(d) <= max_offset + 1e-9
    y = (F[:, None] > t).astype(float)
    return t, mask, y, d


def pmf_survival(pmf: MarginPMF, mu: np.ndarray, a: np.ndarray, t: np.ndarray) -> np.ndarray:
    """P(F > t) for each game row (half-point t, so push-free).

    The gamma bucket is the main line's, ``bucket(a)`` — the runtime pricing
    convention (spread_pmf.MarginPMF.cover_probability).
    """
    out = np.empty_like(t, dtype=float)
    for i, m in enumerate(mu):
        p = pmf.pmf(float(m), pmf.bucket(float(a[i])))
        c = np.cumsum(p[::-1])[::-1]                     # c[j] = P(F >= ks[j])
        idx = np.clip(np.floor(t[i]).astype(int) + 1 - int(pmf.ks[0]), 0, len(pmf.ks) - 1)
        out[i] = c[idx]
    return out


def paired(dl: np.ndarray, mask: np.ndarray) -> dict:
    """Mean per-rung loss difference and its game-clustered z."""
    per = np.where(mask, dl, 0).sum(1)
    cnt = mask.sum(1)
    keep = cnt > 0
    per, cnt = per[keep], cnt[keep]
    n = cnt.sum()
    mean = per.sum() / n
    se = np.sqrt(np.sum((per - cnt * mean) ** 2)) / n
    return {"games": int(keep.sum()), "rungs": int(n), "diff": float(mean),
            "z": float(mean / se) if se > 0 else float("nan")}


def evaluate(pmf: MarginPMF, test: pd.DataFrame) -> dict:
    """ΔBrier (×1000) of the juice-anchored PMF vs the production normal."""
    t, mask, y, d = rung_grid(test)
    a = test.a.to_numpy()
    p = np.where(np.isnan(test.p_fav.to_numpy()), 0.5, test.p_fav.to_numpy())
    mu_pmf = np.array([pmf.anchor(aa, pp) for aa, pp in zip(a, p)], dtype=float)
    if np.isnan(mu_pmf).any():
        raise RuntimeError("anchor failed on a test game")
    s_pmf = pmf_survival(pmf, mu_pmf, a, t)
    mu_n14 = a - norm.ppf(1.0 - p) * PROD_SIGMA            # production: mean from the juice
    s_n14 = np.clip(1.0 - norm.cdf((t - mu_n14[:, None]) / PROD_SIGMA), 0.01, 0.99)
    s_n14_raw = 1.0 - norm.cdf((t - a[:, None]) / PROD_SIGMA)  # normal centred on the spread

    def brier(s):
        return (np.clip(s, *EVAL_CLIP) - y) ** 2

    b_pmf, b_n14, b_raw = brier(s_pmf), brier(s_n14), brier(s_n14_raw)
    cross = np.zeros_like(mask)
    lo, hi = np.minimum(a[:, None], t), np.maximum(a[:, None], t)
    for key in (3, 7, -3, -7):
        cross |= (lo < key) & (hi > key)
    vs_prod = paired(b_pmf - b_n14, mask)
    vs_raw = paired(b_pmf - b_raw, mask)
    vs_cross = paired(b_pmf - b_n14, mask & cross)
    vs_nocross = paired(b_pmf - b_n14, mask & ~cross)
    return {
        "games": vs_prod["games"],
        "rungs": vs_prod["rungs"],
        "brier_pmf": round(float(np.sum(b_pmf * mask) / mask.sum()), 5),
        "brier_normal14": round(float(np.sum(b_n14 * mask) / mask.sum()), 5),
        "dbrier_x1000_vs_normal14_juice": round(1000 * vs_prod["diff"], 3),
        "z_vs_normal14_juice": round(vs_prod["z"], 2),
        "dbrier_x1000_vs_normal14_at_spread": round(1000 * vs_raw["diff"], 3),
        "z_vs_normal14_at_spread": round(vs_raw["z"], 2),
        "dbrier_x1000_crosses_3_or_7": round(1000 * vs_cross["diff"], 3),
        "z_crosses_3_or_7": round(vs_cross["z"], 2),
        "dbrier_x1000_no_key_cross": round(1000 * vs_nocross["diff"], 3),
        "z_no_key_cross": round(vs_nocross["z"], 2),
    }


def walkforward(games: pd.DataFrame, fit_start: int, first_test: int, last_test: int) -> list[dict]:
    rows = []
    for season in range(first_test, last_test + 1):
        train = games[(games.season >= fit_start) & (games.season < season)]
        test = games[games.season == season].reset_index(drop=True)
        if test.empty:
            continue
        fitter = fit_frozen(train)
        r = evaluate(fitter.to_margin_pmf(), test)
        row = {"season": season, "games": r["games"],
               "dbrier_x1000": r["dbrier_x1000_vs_normal14_juice"],
               "z": r["z_vs_normal14_juice"], "fit_ok": fitter.fit_ok}
        rows.append(row)
        print(f"  walk-forward {season}: games {row['games']:4d}  ΔBrier×1000 "
              f"{row['dbrier_x1000']:+.3f}  z {row['z']:+.2f}", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def golden_table(pmf: MarginPMF) -> list[str]:
    """Human-readable spot checks (main -a at devig p → P(favorite covers t))."""
    lines = []
    for a, p, t in [(3, .5, 7.5), (3, .5, 3.5), (3, .5, 16.5), (3.5, .5, 2.5),
                    (7, .5, 2.5), (2.5, .514, 7.5), (10, .5, 2.5)]:
        hyb = pmf.cover_probability(-a, p, -t, yes_is_underdog=False)
        mu = a - norm.ppf(1 - p) * PROD_SIGMA
        n14 = 1 - norm.cdf((t - mu) / PROD_SIGMA)
        lines.append(f"  main -{a:<4} p={p:<5} fav -{t:<5}: PMF {hyb:.3f}  normal(σ=14) {n14:.3f}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fit-start", type=int, default=FIT_START)
    ap.add_argument("--fit-end", type=int, default=None,
                    help="last season to fit (default: last completed season)")
    ap.add_argument("--holdout-start", type=int, default=HOLDOUT_START,
                    help="first holdout / walk-forward season")
    ap.add_argument("--schedules", default=None,
                    help="read a cached nflverse schedules parquet instead of fetching")
    ap.add_argument("--no-eval", action="store_true", help="skip holdout + walk-forward")
    ap.add_argument("--out", default=None, help=f"output path (default {artifact_path('nfl')})")
    ap.add_argument("--dry-run", action="store_true", help="fit + evaluate, write nothing")
    args = ap.parse_args()

    fit_end = args.fit_end if args.fit_end is not None else last_complete_season()
    if not args.fit_start < args.holdout_start <= fit_end:
        raise SystemExit(f"need fit_start < holdout_start <= fit_end, got "
                         f"{args.fit_start} / {args.holdout_start} / {fit_end}")

    raw = load_schedules(args.fit_start, fit_end, args.schedules)
    games = prepare_games(raw, args.fit_start, fit_end)
    per_season = games.groupby("season").size()
    missing = [s for s in range(args.fit_start, fit_end + 1) if per_season.get(s, 0) < 200]
    if missing:
        raise SystemExit(f"seasons with <200 completed games (incomplete data?): {missing}")
    print(f"games {len(games)} over {args.fit_start}-{fit_end} "
          f"(juice available on {int(games.p_fav.notna().sum())})")

    validation: dict = {
        "protocol": (
            "hyperparameters frozen by the 2026-09-22 research (inner split fit 2003-2014 / "
            "validate 2015-2018); holdout = fit [fit_start, holdout_start-1], score "
            "[holdout_start, fit_end] once; walk-forward = fit all prior seasons, score one "
            "season. Rungs: every half-point |t-a|<=14. Baseline: production normal σ=14 with "
            "the mean inferred from the same devigged juice. Brier diff paired, clustered by game."
        ),
        "holdout_seasons": [args.holdout_start, fit_end],
    }
    if not args.no_eval:
        t0 = time.time()
        train = games[games.season < args.holdout_start]
        test = games[games.season >= args.holdout_start].reset_index(drop=True)
        ho_fitter = fit_frozen(train)
        holdout = evaluate(ho_fitter.to_margin_pmf(), test)
        holdout["fit_ok"] = ho_fitter.fit_ok
        validation["holdout"] = holdout
        print(f"holdout {args.holdout_start}-{fit_end} (fit {args.fit_start}-{args.holdout_start - 1}):")
        for k, v in holdout.items():
            print(f"  {k}: {v}")
        wf = walkforward(games, args.fit_start, args.holdout_start, fit_end)
        validation["walkforward"] = wf
        better = sum(1 for r in wf if r["dbrier_x1000"] < 0)
        mean_wf = float(np.mean([r["dbrier_x1000"] for r in wf])) if wf else float("nan")
        validation["walkforward_summary"] = {
            "seasons_better": better, "seasons": len(wf), "mean_dbrier_x1000": round(mean_wf, 3)}
        print(f"walk-forward: PMF better in {better}/{len(wf)} seasons, mean ΔBrier×1000 {mean_wf:+.3f}"
              f"  ({time.time() - t0:.0f}s)")

    fitter = fit_frozen(games)
    if not fitter.fit_ok:
        raise SystemExit("production fit did not converge — refusing to write")
    art = fitter.to_artifact_params(round_to=5)
    art.update({
        "created": date.today().isoformat(),
        "source": "nflverse load_schedules spread_line/result (REG+POST)",
        "fit_seasons": [args.fit_start, fit_end],
        "n_games_fit": int(len(games)),
        "validation": validation,
    })
    pmf = MarginPMF.from_artifact(art)                    # round-trip through the loader
    print(f"production fit {args.fit_start}-{fit_end}: s0 {art['s0']}  s1 {art['s1']}")
    print("key-number multipliers exp(beta):",
          {k: round(float(np.exp(pmf.beta[int(k - pmf.ks[0])])), 2) for k in (3, 7, 10, 14, -3, -7, 0)})
    print("\n".join(golden_table(pmf)))

    if args.dry_run:
        print("--dry-run: nothing written")
        return
    out = Path(args.out) if args.out else artifact_path("nfl")
    out.write_text(json.dumps(art) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
