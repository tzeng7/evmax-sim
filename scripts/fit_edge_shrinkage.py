"""Fit the edge-shrinkage logistic and write data/models/edge_shrinkage_state.json.

The layer this feeds (evmax/ev/sizing.py) replaces the dead confidence discount. It
sizes stake on the OUT-OF-SAMPLE-calibrated win probability rather than the raw blended
model probability, correcting the tail-selection bias: the rows that earn the biggest
stakes are the ones where the model disagrees most with the market, which is exactly
where its estimation error concentrates.

Model, per sector (pooled fit for sectors with < MIN_SECTOR_FIT_N resolved rows):

    P(win) = sigmoid( a + s_b · logit(blended) + s_p · logit(price) )

A perfectly calibrated, selection-free model gives (a, s_b, s_p) = (0, 1, 0). The fit
pulls s_b below 1 and s_p above 0, which shrinks the sized probability toward the market
on disagreement rows. Sizing consumes this; the EV gate is untouched, so enabling the
layer does not change which markets are flagged.

The COEFFICIENTS here are fit on all available history (in-sample is fine — they are
descriptive of the calibration). What JUSTIFIES turning the layer on is the walk-forward
growth check in scripts/backtest_sizing.py, which fits on the past and places the future.

Usage:
    uv run python scripts/fit_edge_shrinkage.py            # fit + write state
    uv run python scripts/fit_edge_shrinkage.py --dry-run  # print coeffs, write nothing
    uv run python scripts/fit_edge_shrinkage.py --days 365 --modes live,shadow
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
from sklearn.linear_model import LogisticRegression

from evmax.backtest.sizing import load_resolved_rows
from evmax.ev.sizing import MIN_SECTOR_FIT_N, SHRINKAGE_STATE_PATH, _logit


def _fit(rows) -> dict | None:
    """Fit the two-input logistic on a list of ResolvedRow. None if too few rows."""
    if len(rows) < 30:
        return None
    X = np.array([[_logit(r.blended), _logit(r.price)] for r in rows], dtype=float)
    y = np.array([r.outcome for r in rows], dtype=int)
    if y.min() == y.max():  # degenerate — all same outcome
        return None
    # Light L2 (C large ⇒ near-MLE); keeps a thin/collinear sector from blowing up.
    clf = LogisticRegression(C=50.0, solver="lbfgs", max_iter=1000)
    clf.fit(X, y)
    s_b, s_p = clf.coef_[0]
    a = clf.intercept_[0]
    return {"a": float(a), "s_b": float(s_b), "s_p": float(s_p), "n": int(len(rows))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--modes", default="live,shadow",
                    help="comma list: which mode rows to fit on (shadow rows are valid evidence)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    rows = load_resolved_rows(days=args.days, modes=modes, exclude_contaminated=True)
    print(f"loaded {len(rows)} clean resolved rows over {args.days}d, modes={modes}")

    pooled = _fit(rows)
    if pooled is None:
        raise SystemExit("not enough rows for a pooled fit — aborting")

    by_sector: dict[str, list] = {}
    for r in rows:
        by_sector.setdefault(r.sector.lower(), []).append(r)

    sectors: dict[str, dict] = {}
    for sec, srows in sorted(by_sector.items()):
        fit = _fit(srows)
        if fit is None:
            continue
        sectors[sec] = fit
        used = "per-sector" if fit["n"] >= MIN_SECTOR_FIT_N else "(pooled at runtime)"
        print(f"  {sec:10s} n={fit['n']:4d}  a={fit['a']:+.3f}  s_b={fit['s_b']:+.3f}  "
              f"s_p={fit['s_p']:+.3f}  {used}")

    print(f"  {'POOLED':10s} n={pooled['n']:4d}  a={pooled['a']:+.3f}  "
          f"s_b={pooled['s_b']:+.3f}  s_p={pooled['s_p']:+.3f}")

    state = {
        "schema_version": 1,
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "days": args.days,
        "modes": list(modes),
        "pooled": pooled,
        "sectors": sectors,
    }
    if args.dry_run:
        print("\n--dry-run: not writing.\n" + json.dumps(state, indent=2)[:1200])
        return
    SHRINKAGE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SHRINKAGE_STATE_PATH.write_text(json.dumps(state, indent=2))
    print(f"\nwrote {SHRINKAGE_STATE_PATH}")


if __name__ == "__main__":
    main()
