"""Anchored discrete distributions for alternate spread and total lines.

A sportsbook posts one MAIN spread and one MAIN total per game. Every other
rung of the ladder is a statement about the SHAPE of the final margin or the
final total around that main line. ``LinePMF`` is that shape, fitted per sport
and per market from closing lines and final scores, then anchored on the main
line's own devigged probability at pricing time.

It generalizes ``spread_pmf.MarginPMF`` (NFL spreads) to totals and to other
sports. It is not wired into live pricing: ``SpreadDistributionModel`` and
``TotalDistributionModel`` are unchanged. Artifacts live in
``data/models/line_pmf/{sector}_{market}.json`` and are written by
``scripts/fit_line_pmf.py``.

Model
-----
``X`` is the integer outcome on the line's own axis:

    spread   X = F, the FAVORITE's final margin (home team at a pick'em)
    total    X = T, the combined final score

    P(X = k | mu) ∝ base(k; mu, s) · exp(beta_k)

``base`` is a location-scale kernel:

    family 't'       Student-t with ``nu`` degrees of freedom (fat tails)
    family 'normal'  Gaussian
    + overtime       totals only: a second component shifted up by ``d_ot`` and
                     weighted ``p_ot`` — an overtime period adds points without
                     adding much to the regulation spread of outcomes, which is
                     what puts the long right tail on basketball totals

The scale depends on the main line: ``s = s0 + s1·(line − line_ref)`` (bigger
favorites and higher totals have wider outcomes), plus ``s2·(total −
total_ref)`` for spreads when the game total is known (a faster game has a
wider margin).

``beta_k`` is a ridge-penalized log-multiplier for individual outcomes: the
key numbers. NFL margins pile up on 3 and 7 and NFL totals on 37 / 41 / 44 /
51; basketball has no ties, so 0 is impossible, and small margins are thinned
by end-game fouling. ``beta`` is what a smooth curve cannot express.

Anchoring
---------
At pricing time ``mu`` is solved so the model reproduces the book's own main
line exactly::

    P(X > a | X != a) == p_main

``a`` is the main line on X's axis (|spread|, or the total) and ``p_main`` the
devigged probability that the favorite covers / the game goes over. The
condition is push-conditional: a whole-number line refunds a push. Everything
the model then says about another rung is a statement about shape only.

Conventions
-----------
Spreads use ``MarginPMF``'s conventions. ``target_line`` is the venue line from
the YES team's side:

    YES = favorite    covers iff F > −target_line
    YES = underdog    covers iff F <  target_line

Totals: over covers iff T > target; under iff T < target. A whole-number
target is priced push-conditionally, so the two sides always sum to 1.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.optimize import brentq

SCHEMA_VERSION = 1
MARKETS = ("spread", "total")
FAMILIES = ("t", "normal")
_REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = _REPO_ROOT / "data" / "models" / "line_pmf"


class LinePMFArtifactError(ValueError):
    """The artifact on disk is not a usable LinePMF."""


def artifact_path(sector: str, market: str) -> Path:
    return ARTIFACT_DIR / f"{sector}_{market}.json"


@dataclass
class LinePMF:
    sector: str
    market: str                      # 'spread' | 'total'
    k_min: int
    k_max: int
    family: str = "t"
    nu: float = 8.0
    s0: float = 12.0
    s1: float = 0.0
    line_ref: float = 0.0
    s2: float = 0.0                  # spreads: scale per point of game total
    total_ref: float = 0.0
    c0: float = 0.0                  # training-time location offset (mu = line + c0)
    p_ot: float = 0.0                # totals: overtime mixture weight
    d_ot: float = 0.0                # totals: points added by overtime
    exclude_zero: bool = False       # spreads in sports without ties
    beta: dict[int, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.market not in MARKETS:
            raise LinePMFArtifactError(f"market must be one of {MARKETS}")
        if self.family not in FAMILIES:
            raise LinePMFArtifactError(f"family must be one of {FAMILIES}")
        if self.k_max <= self.k_min:
            raise LinePMFArtifactError("empty support")
        if not (0.0 <= self.p_ot < 1.0):
            raise LinePMFArtifactError("p_ot must be in [0, 1)")
        self.ks = np.arange(self.k_min, self.k_max + 1, dtype=float)
        b = np.zeros_like(self.ks)
        for k, v in self.beta.items():
            k = int(k)
            if self.k_min <= k <= self.k_max:
                b[k - self.k_min] = float(v)
        if self.exclude_zero and self.k_min <= 0 <= self.k_max:
            b[0 - self.k_min] = -np.inf
        self._beta = b
        self._anchor_cache: dict[tuple, Optional[float]] = {}

    # ------------------------------------------------------------------ shape

    def scale(self, line: float, total: Optional[float] = None) -> float:
        s = self.s0 + self.s1 * (line - self.line_ref)
        if total is not None and self.s2:
            s += self.s2 * (total - self.total_ref)
        return max(s, 1.0)

    def _kernel(self, z: np.ndarray) -> np.ndarray:
        if self.family == "normal":
            return -0.5 * z * z
        return -0.5 * (self.nu + 1.0) * np.log1p(z * z / self.nu)

    def log_weights(self, mu: float, s: float) -> np.ndarray:
        """Unnormalized log P(X = k) over the support."""
        base = self._kernel((self.ks - mu) / s)
        if self.p_ot > 0.0:
            ot = self._kernel((self.ks - mu - self.d_ot) / s)
            base = np.logaddexp(math.log1p(-self.p_ot) + base, math.log(self.p_ot) + ot)
        return base + self._beta

    def pmf(self, mu: float, line: float, total: Optional[float] = None) -> np.ndarray:
        lw = self.log_weights(mu, self.scale(line, total))
        lw = lw - np.max(lw[np.isfinite(lw)])
        w = np.exp(lw)
        return w / w.sum()

    def split(self, mu: float, t: float, line: float,
              total: Optional[float] = None) -> tuple[float, float]:
        """(P(X > t), P(X == t))."""
        p = self.pmf(mu, line, total)
        above = float(p[self.ks > t + 1e-9].sum())
        at = float(p[np.abs(self.ks - t) < 1e-9].sum())
        return above, at

    def conditional_survival(self, mu: float, t: float, line: float,
                             total: Optional[float] = None) -> float:
        """P(X > t | X != t): the price of the side above ``t`` with pushes refunded."""
        above, at = self.split(mu, t, line, total)
        return above / (1.0 - at) if at < 1.0 else 0.5

    # --------------------------------------------------------------- anchoring

    def anchor(self, line: float, p_main: float,
               total: Optional[float] = None) -> Optional[float]:
        """``mu`` such that P(X > line | X != line) == p_main. None if unsolvable."""
        if not (0.0 < p_main < 1.0):
            return None
        key = (round(line, 3), round(p_main, 6), None if total is None else round(total, 2))
        if key not in self._anchor_cache:
            s = self.scale(line, total)
            lo, hi = line - 12.0 * s, line + 12.0 * s

            def f(mu: float) -> float:
                return self.conditional_survival(mu, line, line, total) - p_main

            try:
                self._anchor_cache[key] = (
                    float(brentq(f, lo, hi, xtol=1e-7)) if f(lo) * f(hi) < 0 else None)
            except ValueError:
                self._anchor_cache[key] = None
        return self._anchor_cache[key]

    # ----------------------------------------------------------------- pricing

    def over_probability(self, main_total: float, p_over: float,
                         target: float) -> Optional[float]:
        """P(total > target | no push), anchored on the main total's own price."""
        if self.market != "total":
            raise ValueError("over_probability is for a total model")
        mu = self.anchor(main_total, p_over)
        if mu is None:
            return None
        return self.conditional_survival(mu, target, main_total)

    def cover_probability(self, main_line: float, p_fav_cover: float, target_line: float,
                          yes_is_underdog: bool = False,
                          total: Optional[float] = None) -> Optional[float]:
        """P(YES covers ``target_line`` | no push) for a spread.

        ``main_line`` is the favorite's main handicap (sign ignored) and
        ``p_fav_cover`` its devigged cover probability.
        """
        if self.market != "spread":
            raise ValueError("cover_probability is for a spread model")
        a = abs(main_line)
        mu = self.anchor(a, p_fav_cover, total)
        if mu is None:
            return None
        if yes_is_underdog:
            # underdog covers iff F < target_line
            return 1.0 - self.conditional_survival(mu, target_line, a, total)
        return self.conditional_survival(mu, -target_line, a, total)

    # --------------------------------------------------------------- artifacts

    def to_artifact(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION, "sector": self.sector, "market": self.market,
            "k_min": self.k_min, "k_max": self.k_max, "family": self.family, "nu": self.nu,
            "s0": self.s0, "s1": self.s1, "line_ref": self.line_ref, "s2": self.s2,
            "total_ref": self.total_ref, "c0": self.c0, "p_ot": self.p_ot, "d_ot": self.d_ot,
            "exclude_zero": self.exclude_zero,
            "beta": {str(k): round(float(v), 6) for k, v in sorted(self.beta.items())
                     if abs(v) > 1e-9},
            "meta": self.meta,
        }

    @classmethod
    def from_artifact(cls, art: dict[str, Any]) -> "LinePMF":
        if art.get("schema_version") != SCHEMA_VERSION:
            raise LinePMFArtifactError(f"unsupported schema_version {art.get('schema_version')}")
        try:
            return cls(
                sector=art["sector"], market=art["market"], k_min=int(art["k_min"]),
                k_max=int(art["k_max"]), family=art["family"], nu=float(art["nu"]),
                s0=float(art["s0"]), s1=float(art["s1"]), line_ref=float(art["line_ref"]),
                s2=float(art.get("s2", 0.0)), total_ref=float(art.get("total_ref", 0.0)),
                c0=float(art.get("c0", 0.0)), p_ot=float(art.get("p_ot", 0.0)),
                d_ot=float(art.get("d_ot", 0.0)), exclude_zero=bool(art.get("exclude_zero")),
                beta={int(k): float(v) for k, v in (art.get("beta") or {}).items()},
                meta=dict(art.get("meta") or {}),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise LinePMFArtifactError(f"bad artifact: {e}") from e


_CACHE: dict[tuple[str, str], Optional[LinePMF]] = {}


def load_line_pmf(sector: str, market: str) -> Optional[LinePMF]:
    """The fitted model for a sector and market, or None when there is none.

    A missing or invalid artifact returns None (callers fall back to the
    normal curve); it never raises.
    """
    key = (sector, market)
    if key not in _CACHE:
        path = artifact_path(sector, market)
        try:
            _CACHE[key] = LinePMF.from_artifact(json.loads(path.read_text()))
        except (OSError, ValueError):
            _CACHE[key] = None
    return _CACHE[key]


def clear_line_pmf_cache() -> None:
    _CACHE.clear()


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


@dataclass
class FitConfig:
    family: str = "t"
    nu: float = 8.0
    fit_nu: bool = True
    overtime: bool = False          # totals: fit the overtime mixture
    exclude_zero: bool = False
    beta_window: Optional[tuple[int, int]] = None   # outcomes that may carry a key multiplier
    lam: float = 25.0               # ridge on beta; larger = closer to the smooth base
    use_total_scale: bool = False   # spreads: let the scale depend on the game total


def _nll(model: LinePMF, x_idx: np.ndarray, mu: np.ndarray, s: np.ndarray) -> float:
    """Mean negative log-likelihood of observed outcomes (rows = games)."""
    z = (model.ks[None, :] - mu[:, None]) / s[:, None]
    base = model._kernel(z)
    if model.p_ot > 0.0:
        ot = model._kernel(z - model.d_ot / s[:, None])
        base = np.logaddexp(math.log1p(-model.p_ot) + base, math.log(model.p_ot) + ot)
    lw = base + model._beta[None, :]
    m = np.max(np.where(np.isfinite(lw), lw, -np.inf), axis=1, keepdims=True)
    logz = m[:, 0] + np.log(np.exp(lw - m).sum(axis=1))
    return float(-(lw[np.arange(len(x_idx)), x_idx] - logz).mean())


def _row_pmf(model: LinePMF, mu: np.ndarray, s: np.ndarray) -> np.ndarray:
    z = (model.ks[None, :] - mu[:, None]) / s[:, None]
    base = model._kernel(z)
    if model.p_ot > 0.0:
        ot = model._kernel(z - model.d_ot / s[:, None])
        base = np.logaddexp(math.log1p(-model.p_ot) + base, math.log(model.p_ot) + ot)
    lw = base + model._beta[None, :]
    lw = lw - np.max(np.where(np.isfinite(lw), lw, -np.inf), axis=1, keepdims=True)
    w = np.exp(lw)
    return w / w.sum(axis=1, keepdims=True)


def fit_line_pmf(sector: str, market: str, outcomes: np.ndarray, lines: np.ndarray,
                 totals: Optional[np.ndarray] = None,
                 config: Optional[FitConfig] = None) -> LinePMF:
    """Maximum-likelihood fit of a ``LinePMF``.

    ``outcomes`` are the integer results on the line's axis (favorite margin,
    or combined score); ``lines`` the closing main line on the same axis
    (|spread|, or the total); ``totals`` the closing game total for spreads
    (optional). The closing line is used as the location (plus a fitted
    offset ``c0``), as in ``fit_nfl_margin_pmf.py``.

    Three stages: smooth parameters with no key multipliers; the key
    multipliers ``beta`` with the smooth part fixed (a convex ridge problem
    with an exact gradient); the smooth parameters once more.
    """
    from scipy.optimize import minimize

    cfg = config or FitConfig()
    x = np.asarray(outcomes, dtype=float).round().astype(int)
    line = np.asarray(lines, dtype=float)
    tot = None if totals is None or not cfg.use_total_scale else np.asarray(totals, dtype=float)
    if len(x) < 200:
        raise ValueError("need at least 200 games to fit a line PMF")

    resid = x - line
    pad = int(6 * np.std(resid)) + 10
    k_min, k_max = int(np.floor(x.min() - pad)), int(np.ceil(x.max() + pad))
    if market == "total":
        k_min = max(k_min, 0)
    line_ref = float(np.mean(line))
    total_ref = float(np.mean(tot)) if tot is not None else 0.0
    s_init = float(np.std(resid))

    def build(theta: np.ndarray, beta: dict[int, float]) -> LinePMF:
        s0, s1, c0, log_nu, s2, p_logit, d_ot = theta
        return LinePMF(
            sector=sector, market=market, k_min=k_min, k_max=k_max, family=cfg.family,
            nu=float(np.clip(np.exp(log_nu), 2.5, 200.0)), s0=float(abs(s0)), s1=float(s1),
            line_ref=line_ref, s2=float(s2) if tot is not None else 0.0, total_ref=total_ref,
            c0=float(c0),
            p_ot=float(1.0 / (1.0 + np.exp(-p_logit))) if cfg.overtime else 0.0,
            d_ot=float(abs(d_ot)) if cfg.overtime else 0.0,
            exclude_zero=cfg.exclude_zero, beta=beta)

    def rows(model: LinePMF) -> tuple[np.ndarray, np.ndarray]:
        s = model.s0 + model.s1 * (line - model.line_ref)
        if tot is not None:
            s = s + model.s2 * (tot - model.total_ref)
        return line + model.c0, np.maximum(s, 1.0)

    x_idx = x - k_min
    theta0 = np.array([s_init, 0.0, 0.0, math.log(cfg.nu), 0.0, math.log(0.06 / 0.94),
                       max(s_init, 8.0)])
    free = [0, 1, 2] + ([3] if cfg.family == "t" and cfg.fit_nu else []) \
        + ([4] if tot is not None else []) + ([5, 6] if cfg.overtime else [])

    def fit_theta(theta: np.ndarray, beta: dict[int, float]) -> np.ndarray:
        def obj(v: np.ndarray) -> float:
            th = theta.copy()
            th[free] = v
            m = build(th, beta)
            mu, s = rows(m)
            return _nll(m, x_idx, mu, s)

        res = minimize(obj, theta[free], method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 4000})
        out = theta.copy()
        out[free] = res.x
        return out

    theta = fit_theta(theta0, {})
    beta: dict[int, float] = {}
    if cfg.beta_window is not None:
        lo, hi = max(cfg.beta_window[0], k_min), min(cfg.beta_window[1], k_max)
        keys = [k for k in range(lo, hi + 1) if not (cfg.exclude_zero and k == 0)]
        cols = np.array([k - k_min for k in keys])
        counts = np.bincount(x_idx, minlength=k_max - k_min + 1)[cols].astype(float)
        n = float(len(x))

        def obj_beta(v: np.ndarray) -> tuple[float, np.ndarray]:
            m = build(theta, dict(zip(keys, v)))
            mu, s = rows(m)
            p = _row_pmf(m, mu, s)
            nll = -np.log(p[np.arange(len(x_idx)), x_idx]).mean()
            grad = -(counts - p[:, cols].sum(axis=0)) / n + 2.0 * cfg.lam * v / n
            return float(nll + cfg.lam * float(v @ v) / n), grad

        res = minimize(obj_beta, np.zeros(len(keys)), jac=True, method="L-BFGS-B",
                       options={"maxiter": 500})
        beta = {k: float(v) for k, v in zip(keys, res.x)}
        theta = fit_theta(theta, beta)

    model = build(theta, beta)
    mu, s = rows(model)
    model.meta = {"n_games": int(len(x)), "nll": round(_nll(model, x_idx, mu, s), 6),
                  "resid_sd": round(s_init, 4)}
    return model
