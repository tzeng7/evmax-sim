"""Isotonic probability calibration (pool-adjacent-violators).

The walk-forward backtest showed the model is well-ordered but under-confident:
its probabilities are monotone in the truth but sit below the diagonal (a 13%
top-10 pick actually finishes top-10 ~15% of the time; a 55% pick, ~74%). A
monotone recalibration map fixes exactly that shape — it stretches the compressed
probabilities onto the diagonal WITHOUT reordering anyone.

Implemented with the pool-adjacent-violators algorithm in numpy (no sklearn —
it isn't a declared dependency). The map is a non-decreasing step function fit on
(model_prob, outcome) pairs; `transform` interpolates it.

**This does not create edge.** Calibration makes the model's numbers honest; it
cannot make a model that agrees with the market disagree-and-be-right. It must be
fit on a TRAIN split and validated on a HOLDOUT split (temporally separated, so
the map never sees the outcomes it is judged on) — `scripts/golf_calibration.py`
does exactly that (fit on 2025, test on 2026).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _pav(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool-adjacent-violators: fit a non-decreasing step function to (x, y).

    Returns (x_thresholds, y_levels) sorted by x. ``x`` need not be unique;
    equal-x points are pooled first.
    """
    order = np.argsort(x, kind="stable")
    xs, ys, ws = x[order], y[order], w[order]

    # Pool exact-equal x first (a level can't split within one x value).
    ux, inv = np.unique(xs, return_inverse=True)
    wy = np.bincount(inv, weights=ys * ws)
    ww = np.bincount(inv, weights=ws)
    vals = wy / ww  # weighted mean outcome per unique x

    # PAV over the unique-x levels.
    levels = list(vals)
    weights = list(ww)
    xsu = list(ux)
    i = 0
    stack_v: list[float] = []
    stack_w: list[float] = []
    stack_x: list[float] = []
    for xi, vi, wi in zip(xsu, levels, weights):
        cv, cw, cx = vi, wi, xi
        while stack_v and stack_v[-1] > cv:
            pv, pw = stack_v.pop(), stack_w.pop()
            stack_x.pop()
            cv = (pv * pw + cv * cw) / (pw + cw)
            cw = pw + cw
        stack_v.append(cv)
        stack_w.append(cw)
        stack_x.append(cx)
    return np.array(stack_x), np.array(stack_v)


@dataclass
class IsotonicCalibrator:
    x_thresh: np.ndarray
    y_level: np.ndarray

    @classmethod
    def fit(cls, probs, outcomes, weights=None) -> "IsotonicCalibrator":
        x = np.asarray(probs, dtype=float)
        y = np.asarray(outcomes, dtype=float)
        w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)
        if x.size == 0:
            return cls(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        xt, yl = _pav(x, y, w)
        return cls(xt, yl)

    def transform(self, probs) -> np.ndarray:
        """Map raw probs through the calibration curve (clipped to [0, 1])."""
        p = np.asarray(probs, dtype=float)
        if self.x_thresh.size == 0:
            return np.clip(p, 0.0, 1.0)
        # Piecewise-linear interpolation between the isotonic step levels;
        # flat extrapolation at the ends.
        out = np.interp(p, self.x_thresh, self.y_level,
                        left=self.y_level[0], right=self.y_level[-1])
        return np.clip(out, 0.0, 1.0)

    def transform_one(self, prob: float) -> float:
        return float(self.transform([prob])[0])


def brier(probs, outcomes) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if p.size == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))
