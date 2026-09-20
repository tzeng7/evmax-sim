"""Run the sizing replay harness — the evidence for turning each sizing layer on.

Reports geometric growth, drawdown, and a week-block bootstrap for a panel of sizing
policies, plus the walk-forward growth of edge shrinkage and a base-fraction sweep.
Read-only; touches nothing but predictions.db.

Usage:
    uv run python scripts/backtest_sizing.py                      # full panel, 180d, live
    uv run python scripts/backtest_sizing.py --days 365 --modes live,shadow
    uv run python scripts/backtest_sizing.py --sector wnba        # one sector
    uv run python scripts/backtest_sizing.py --sweep-base         # base-fraction sweep (Phase 2)
"""

from __future__ import annotations

import argparse

from evmax.backtest.sizing import (
    block_bootstrap_log_growth,
    edge_ratio,
    load_resolved_rows,
    make_kelly_policy,
    simulate,
    walk_forward_months,
)
from evmax.ev.sizing import load_shrinkage_model


def _line(name, res, boot):
    return (f"{name:34s} n={res.n_bets:4d}  logG={res.log_growth:+7.3f}  "
            f"x={res.final_multiple:6.3f}  maxDD={res.max_drawdown:6.1%}  "
            f"boot[5,50,95]=[{boot[5.0]:+.2f},{boot[50.0]:+.2f},{boot[95.0]:+.2f}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--modes", default="live")
    ap.add_argument("--sector", default=None)
    ap.add_argument("--event-cap", type=float, default=0.08)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--sweep-base", action="store_true")
    args = ap.parse_args()

    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    rows = load_resolved_rows(days=args.days, modes=modes, exclude_contaminated=True)
    if args.sector:
        rows = [r for r in rows if r.sector.lower() == args.sector.lower()]
    print(f"loaded {len(rows)} clean rows | days={args.days} modes={modes} "
          f"sector={args.sector or 'ALL'} event_cap={args.event_cap}")
    print(f"overall edge_ratio (realized/predicted, net fee) = {edge_ratio(rows):.2f}\n")

    shrink = load_shrinkage_model()
    cap = args.event_cap

    def run(name, policy):
        res = simulate(rows, policy, event_cap=cap)
        boot = block_bootstrap_log_growth(rows, policy, event_cap=cap, n_boot=args.boot)
        print(_line(name, res, boot))

    print("== fixed policies (in-sample) ==")
    run("flat 1%", lambda r: 0.01)
    run("half-Kelly cap 2%", make_kelly_policy(base_fraction=0.5, max_kelly=0.02))
    run("half-Kelly cap 5% (SHIPPED)", make_kelly_policy(base_fraction=0.5, max_kelly=0.05))
    run("half-Kelly cap 10%", make_kelly_policy(base_fraction=0.5, max_kelly=0.10))
    run("quarter-Kelly cap 5%", make_kelly_policy(base_fraction=0.25, max_kelly=0.05))
    if shrink is not None:
        print("\n== with edge shrinkage (in-sample coeffs) ==")
        run("shrunk half-Kelly cap 5%",
            make_kelly_policy(base_fraction=0.5, max_kelly=0.05, shrinkage_model=shrink))
        run("shrunk half-Kelly cap 10%",
            make_kelly_policy(base_fraction=0.5, max_kelly=0.10, shrinkage_model=shrink))
    else:
        print("\n(no shrinkage state — run scripts/fit_edge_shrinkage.py first)")

    # Walk-forward: fit shrinkage on the past, place the future.
    print("\n== walk-forward (fit shrinkage on months 1..k, place month k+1) ==")
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    from evmax.ev.sizing import ShrinkageCoeffs, ShrinkageModel, _logit, MIN_SECTOR_FIT_N

    def fit_model(train_rows) -> ShrinkageModel:
        def fit(rs):
            if len(rs) < 30:
                return None
            X = np.array([[_logit(r.blended), _logit(r.price)] for r in rs])
            y = np.array([r.outcome for r in rs])
            if y.min() == y.max():
                return None
            clf = LogisticRegression(C=50.0, max_iter=1000).fit(X, y)
            return ShrinkageCoeffs(clf.intercept_[0], clf.coef_[0][0], clf.coef_[0][1], len(rs))
        pooled = fit(train_rows)
        secs = {}
        bysec: dict = {}
        for r in train_rows:
            bysec.setdefault(r.sector.lower(), []).append(r)
        for s, rs in bysec.items():
            c = fit(rs)
            if c:
                secs[s] = c
        return ShrinkageModel(pooled=pooled, sectors=secs)

    def wf_policy(model_kwargs):
        def build(train_rows):
            m = fit_model(train_rows)
            return make_kelly_policy(shrinkage_model=m, **model_kwargs)
        return build

    for name, kw in [("raw half-Kelly cap5", dict(base_fraction=0.5, max_kelly=0.05)),
                     ("raw half-Kelly cap10", dict(base_fraction=0.5, max_kelly=0.10))]:
        # baseline (no shrinkage) walk-forward for the same months
        base_res = walk_forward_months(rows, lambda tr, kw=kw: make_kelly_policy(**kw), event_cap=cap)
        shr_res = walk_forward_months(rows, wf_policy(kw), event_cap=cap)
        print(f"{name:22s} baseline logG={base_res.log_growth:+.3f} (n={base_res.n_bets}) "
              f"|  +shrinkage logG={shr_res.log_growth:+.3f} (n={shr_res.n_bets})")

    if args.sweep_base:
        print("\n== base-fraction sweep (Phase 2; select on 5th-pct bootstrap growth) ==")
        for bf in [0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 1.00]:
            pol = make_kelly_policy(base_fraction=bf, max_kelly=0.05, shrinkage_model=shrink)
            res = simulate(rows, pol, event_cap=cap)
            boot = block_bootstrap_log_growth(rows, pol, event_cap=cap, n_boot=args.boot)
            print(f"base={bf:.2f} cap5%  logG={res.log_growth:+.3f}  maxDD={res.max_drawdown:5.1%}  "
                  f"boot5%={boot[5.0]:+.2f}  boot50%={boot[50.0]:+.2f}")


if __name__ == "__main__":
    main()
