"""Do schedule features explain the WNBA spread market's residual? (2026-07-06: NO)

Worth-gate experiment for building a new WNBA spread model blend. The sibling
walk-forward (backtest_wnba_spread_walkforward.py) already established that the
possession sim adds ZERO incremental signal over the market anchor
(corr(sim, market residual) = +0.05; optimal sim weight = 0; even an in-sample
DEBIASED sim does not beat market-alone). So a new blend is only worth building
if some feature the market might misprice explains the market's residual.
This script tests the cheapest orthogonal candidate: schedule fatigue.

Leak-free features per team per game, from the ESPN game log (schedule is known
pre-game; only past games feed the counters):

  rest   : days since previous game (capped 7; season opener -> 7)
  b2b    : rest == 1
  dens7  : games in the trailing 7 days
  rr     : consecutive road games including this one (0 at home)

Tests (all on 2026 season, May 2 - Jul 5):

  A. MARGIN space (max power): OLS of (actual home margin - Pinnacle-implied
     margin) on feature diffs (home - away).
  B. COVER space: correlation of features (YES-side perspective) with
     (outcome - Kalshi main-line close prob).
  C. IN-SAMPLE Brier: shift the market prob by the fitted margin effect through
     the sigma=12.5 normal CDF — generous best case for the hypothesis.
  D. OUT-OF-SAMPLE Brier: fit A on games before --split, score C on games after.

RESULT 2026-07-06 (n=154 margin rows / 157 main lines):
  - No feature reaches |t| >= 2 in margin space (best: rest_diff t=0.77,
    univariate z=+1.6). Signs are fatigue-coherent but individually null.
  - In-sample Brier gain (0.2348 -> 0.2297) is unstable across halves.
  - OUT-OF-SAMPLE the full model is WORSE (market 0.2485 -> 0.2519) and the
    b2b coefficient flips sign between halves; rest-only gains a sub-noise
    -1.3/1000 on n=68.
  VERDICT: schedule features do NOT clear the gate; do not build a spread
  blend on them. Re-run at season end (~240 games): if the full-sample
  rest_diff point estimate (+0.6 pts/rest-day) is real it would reach z≈2 by
  then. The exploitable WNBA spread signal remains microstructure/CLV
  (watch-listings + first-anchored-sweep + `cleanup shadow clv --side lay`),
  which needs no model change.

Run:  .venv/bin/python scripts/experiment_wnba_schedule_residual.py
      [--months 202605,202606,202607] [--split 2026-06-10]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
from scipy.stats import norm as N

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

from backtest_wnba_spread_walkforward import (  # noqa: E402
    _lookup_pm1, fetch_game_log, load_archive_spreads, load_pinnacle_spreads,
)
from evmax.matching.normalizer import NameNormalizer  # noqa: E402

SIGMA = 12.5
FEATS = ["rest_diff", "b2b_diff", "dens7_diff", "opp_road_run"]


def schedule_rows(games: list[dict]) -> list[dict]:
    norm = NameNormalizer("wnba")
    last: dict[str, date] = {}
    recent: dict[str, list[date]] = {}
    rr: dict[str, int] = {}
    rows = []
    for g in games:
        d = date.fromisoformat(g["date"])
        home, away = norm.normalize(g["home_name"]), norm.normalize(g["away_name"])
        if not home or not away:
            continue
        f = {}
        for t, is_home in ((home, True), (away, False)):
            lp = last.get(t)
            rest = min((d - lp).days, 7) if lp else 7
            dens = sum(1 for dd in recent.get(t, []) if 0 < (d - dd).days <= 7)
            f[t] = {"rest": rest, "b2b": int(rest == 1), "dens7": dens,
                    "rr": 0 if is_home else rr.get(t, 0) + 1}
        rows.append({"date": g["date"], "home": home, "away": away,
                     "mh": g["home_score"] - g["away_score"],
                     "h": f[home], "a": f[away]})
        for t, is_home in ((home, True), (away, False)):
            last[t] = d
            recent.setdefault(t, []).append(d)
            rr[t] = 0 if is_home else rr.get(t, 0) + 1
    return rows


def build_datasets(rows, arch, pinn):
    """A: (date, x, margin_resid) vs Pinnacle. B: (date, x, kalshi_p, went)."""
    A, B = [], []
    for r in rows:
        fs = frozenset({r["home"], r["away"]})
        xh = [r["h"]["rest"] - r["a"]["rest"], r["h"]["b2b"] - r["a"]["b2b"],
              r["h"]["dens7"] - r["a"]["dens7"], r["a"]["rr"]]
        pin = _lookup_pm1(r["date"], fs, pinn)
        if pin is not None and pin["a"] in fs:
            line_home = pin["line"] if pin["a"] == r["home"] else -pin["line"]
            A.append((r["date"], xh, r["mh"] + line_home))
        best = None  # Kalshi main line = close nearest 0.5, either YES side
        for yc, is_home in ((r["home"], True), (r["away"], False)):
            rungs = _lookup_pm1(r["date"], yc, arch)
            if not rungs:
                continue
            for strike, px in rungs.items():
                c = (abs(px["close"] - 0.5), px["close"], strike, is_home)
                if best is None or c[0] < best[0]:
                    best = c
        if best:
            _, kp, strike, is_home = best
            x = xh if is_home else [-xh[0], -xh[1], -xh[2], -r["a"]["rr"]]
            ym = r["mh"] if is_home else -r["mh"]
            B.append((r["date"], x, kp, float(ym > strike)))
    return A, B


def ols(X, y, names):
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    s2 = resid @ resid / (len(y) - X1.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X1.T @ X1)))
    print(f"    {'feature':<14} {'beta':>8} {'se':>7} {'t':>6}")
    for nm, b, s in zip(["intercept"] + names, beta, se):
        print(f"    {nm:<14} {b:>8.3f} {s:>7.3f} {b / s:>6.2f}")
    return beta


def shifted(p, shift_pts):
    return N.cdf(N.ppf(np.clip(p, 0.01, 0.99)) + shift_pts / SIGMA)


def brier(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", default="202605,202606,202607")
    ap.add_argument("--split", default="2026-06-10",
                    help="OOS boundary: fit margin model before, test cover after")
    args = ap.parse_args()
    months = args.months.split(",")
    cache = REPO / "data" / "backtest" / f"wnba_games_{months[0]}_{months[-1]}.json"

    games = fetch_game_log(months=months, cache=cache)
    rows = schedule_rows(games)
    arch, pinn = load_archive_spreads(), load_pinnacle_spreads()
    A, B = build_datasets(rows, arch, pinn)
    print(f"games={len(rows)}  margin rows={len(A)}  main lines={len(B)}")

    AX = np.array([x for _, x, _ in A], float)
    Ay = np.array([y for _, _, y in A], float)
    print(f"\n=== A. MARGIN residual vs Pinnacle (n={len(Ay)}) "
          f"mean={Ay.mean():+.2f} sd={Ay.std():.1f} ===")
    for i, nm in enumerate(FEATS):
        if AX[:, i].std() > 0:
            c = float(np.corrcoef(AX[:, i], Ay)[0, 1])
            print(f"  corr({nm:<13}, resid) = {c:+.3f}  (z~{c * np.sqrt(len(Ay)):+.1f})")
    beta = ols(AX, Ay, FEATS)

    BX = np.array([x for _, x, _, _ in B], float)
    Bp = np.array([p for _, _, p, _ in B], float)
    By = np.array([y for _, _, _, y in B], float)
    print(f"\n=== B. COVER residual vs Kalshi main line (n={len(By)}) "
          f"mean={(By - Bp).mean():+.3f} ===")
    for i, nm in enumerate(FEATS):
        if BX[:, i].std() > 0:
            c = float(np.corrcoef(BX[:, i], By - Bp)[0, 1])
            print(f"  corr({nm:<13}, resid) = {c:+.3f}  (z~{c * np.sqrt(len(By)):+.1f})")

    padj = shifted(Bp, BX @ beta[1:])
    print(f"\n=== C. IN-SAMPLE Brier ===  market={brier(Bp, By):.4f}  "
          f"+schedule={brier(padj, By):.4f}")

    A_tr = [(x, y) for dt, x, y in A if dt < args.split]
    B_te = [(x, p, y) for dt, x, p, y in B if dt >= args.split]
    if len(A_tr) >= 30 and len(B_te) >= 20:
        Xtr = np.array([x for x, _ in A_tr], float)
        ytr = np.array([y for _, y in A_tr], float)
        bt, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(ytr)), Xtr]),
                                 ytr, rcond=None)
        Xte = np.array([x for x, _, _ in B_te], float)
        pte = np.array([p for _, p, _ in B_te], float)
        yte = np.array([y for _, _, y in B_te], float)
        pte_adj = shifted(pte, Xte @ bt[1:])
        print(f"\n=== D. OUT-OF-SAMPLE (fit<{args.split}, n={len(ytr)}; "
              f"test>=, n={len(yte)}) ===")
        print(f"  beta: {np.round(bt[1:], 3)}")
        print(f"  market={brier(pte, yte):.4f}  +schedule={brier(pte_adj, yte):.4f}")
    else:
        print("\n[D] not enough rows on one side of the split")


if __name__ == "__main__":
    main()
