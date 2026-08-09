#!/usr/bin/env python3
"""Backtest the injury/rest/playoff adjustment over-application (double-count).

Motivation
----------
`EVGapAgent._compute_gap` builds `blended_prob = effective_sharp_weight*sharp +
(1-effective_sharp_weight)*model` and then applies injury/rest/playoff layers on
TOP of it. Sharp (the dominant weight, ~0.70 NBA / ~0.85 elsewhere) already
prices known injuries/rest/playoff — that is exactly why those layers are
skipped for spreads. Applying the FULL adjustment to the whole (mostly-sharp)
blend therefore re-counts what sharp already knows, over-correcting the blend by
`effective_sharp_weight * impact` and manufacturing phantom edges.

Signal
------
The premise is that scan-time sharp `S` already prices the injury. So the test
is: does our adjustment shove `(B - S)` actually anticipate where the Pinnacle
line goes by close, `(close - S)`? We regress the realized move on our shove:

    slope ~ 1        -> full adjustment justified
    slope ~ (1-sw)   -> the scaled (1-sharp_weight) fix is right-sized
    slope ~ 0        -> the adjustment is noise (skip on moneyline)

The slope is the empirically-optimal fraction of the adjustment to keep. We also
report a lambda-sweep Brier vs outcome and vs the close, and the practical impact
on logged EV under the scaled policy.

Reads data/predictions.db (read-only). Run from the repo root:

    python scripts/backtest_adjustment_scaling.py
"""
from __future__ import annotations

import math
import sqlite3
import statistics as st
from pathlib import Path

try:
    from evmax.agents.cleanup.db import DB_PATH as DB
except Exception:  # pragma: no cover - fallback when run outside the package
    DB = Path(__file__).resolve().parent.parent / "data" / "predictions.db"


def _ols(x: list[float], y: list[float]) -> dict:
    """Slope + SE + R^2 for y ~ a + b*x."""
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    b = sxy / sxx
    a = my - b * mx
    ss_res = sum((yi - (a + b * xi)) ** 2 for xi, yi in zip(x, y))
    ss_tot = sum((yi - my) ** 2 for yi in y)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    se = math.sqrt((ss_res / (n - 2)) / sxx) if n > 2 else float("nan")
    return {"slope": b, "se": se, "r2": r2, "n": n}


def _fetch(cur, where: str, params: tuple = ()) -> list[dict]:
    cur.execute(
        f"""SELECT p.sector, p.kalshi_yes_price AS kp, p.sharp_true_prob AS S,
                   p.blended_true_prob AS B, p.sharp_weight_used AS sw,
                   p.ev_pct AS ev, o.outcome AS y, o.pinnacle_close_prob AS close
            FROM ev_predictions p JOIN ev_outcomes o ON p.market_id = o.market_id
            WHERE o.outcome IS NOT NULL AND p.market_type = 'moneyline'
              AND p.sharp_true_prob IS NOT NULL AND p.blended_true_prob IS NOT NULL
              {where}""",
        params,
    )
    return [dict(r) for r in cur.fetchall()]


def _brier(rows: list[dict], lam: float, target: str) -> float:
    vals = []
    for r in rows:
        p = min(0.999, max(0.001, r["S"] + lam * (r["B"] - r["S"])))
        vals.append((p - r[target]) ** 2)
    return sum(vals) / len(vals) if vals else float("nan")


def analyze(label: str, rows: list[dict]) -> None:
    rows = [r for r in rows if r["close"] is not None and 0 < r["close"] < 1]
    print(f"\n===== {label} =====")
    if len(rows) < 20:
        print(f"  n={len(rows)} — underpowered, skipping regression")
        return
    shove = [abs(r["B"] - r["S"]) for r in rows]
    print(f"  n={len(rows)}   mean |B-S| = {1e2 * st.mean(shove):.2f}pp   "
          f"max = {1e2 * max(shove):.2f}pp")

    reg = _ols([r["B"] - r["S"] for r in rows], [r["close"] - r["S"] for r in rows])
    lo, hi = reg["slope"] - 1.96 * reg["se"], reg["slope"] + 1.96 * reg["se"]
    mean_sw = st.mean([r["sw"] for r in rows])
    print(f"  REG (close-S) ~ (B-S): slope={reg['slope']:+.3f} "
          f"[95% {lo:+.3f},{hi:+.3f}]  R2={reg['r2']:.3f}")
    print(f"      full=1.00  scaled(1-sw)={1 - mean_sw:.2f}  skip=0.00  "
          f"-> close realizes ~{100 * max(0, reg['slope']):.0f}% of the shove")

    lams = [0.0, 0.15, 0.30, 0.50, 0.75, 1.0]
    bo = [(l, _brier(rows, l, "y")) for l in lams]
    bc = [(l, _brier(rows, l, "close")) for l in lams]
    print("  Brier vs OUTCOME: " + "  ".join(f"λ{l:.2f}={b:.4f}" for l, b in bo)
          + f"  [best λ={min(bo, key=lambda t: t[1])[0]}]")
    print("  Sq-err vs CLOSE:  " + "  ".join(f"λ{l:.2f}={b:.5f}" for l, b in bc)
          + f"  [best λ={min(bc, key=lambda t: t[1])[0]}]")


def impact(rows: list[dict]) -> None:
    """Practical effect of the scaled fix on logged EV (2% threshold).

    Approximation: on injury-fired rows the shove B-S is adjustment-dominated,
    so the scaled blend ~ S + (1-sw)*(B-S). The exact fix keeps the full model
    tilt + (1-sw)*adjustment, so this slightly overstates the phantom fraction.
    """
    rows = [r for r in rows if 0 < r["kp"] < 1]
    def ev(p, price):
        return p / price - 1.0
    def fixed(r):
        return r["S"] + (1 - r["sw"]) * (r["B"] - r["S"])
    cur_shove = st.mean([abs(r["B"] - r["S"]) for r in rows])
    fix_shove = st.mean([abs((1 - r["sw"]) * (r["B"] - r["S"])) for r in rows])
    flagged = [r for r in rows if (r["ev"] or 0) >= 0.02]
    drop = sum(1 for r in flagged if ev(fixed(r), r["kp"]) < 0.02)
    print(f"\n===== IMPACT (injury-fired ML, approx) =====")
    print(f"  mean shove off sharp: {1e2 * cur_shove:.2f}pp -> {1e2 * fix_shove:.2f}pp")
    print(f"  of {len(flagged)} logged edges (EV>=2%), {drop} "
          f"({100 * drop / max(1, len(flagged)):.0f}%) fall below 2% under the fix")


def main() -> None:
    con = sqlite3.connect(str(DB))
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    inj = _fetch(cur, "AND p.model_sources LIKE '%injury%'")
    analyze("INJURY pooled (all sectors) ML", inj)
    for sec in ("wnba", "nba"):
        analyze(f"INJURY · {sec} ML",
                _fetch(cur, "AND p.sector=? AND p.model_sources LIKE '%injury%'", (sec,)))
    analyze("REST · nba ML", _fetch(cur, "AND p.model_sources LIKE '%rest%'"))
    analyze("PLAYOFF · nba ML", _fetch(cur, "AND p.model_sources LIKE '%playoff%'"))
    impact(inj)


if __name__ == "__main__":
    main()
