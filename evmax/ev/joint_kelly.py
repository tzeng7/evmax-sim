"""Joint (correlation-aware) Kelly sizing for multiple bets on the same event.

The independent path (``evmax.ev.kelly.kelly_fraction`` + the coordinator's
per-event proportional exposure guard) sizes each leg as if it were the only
bet on the game and then scales the *sum* down to a fixed cap. That ignores the
covariance structure between legs that share an underlying game outcome:

* Two contradictory legs (``A ML`` + ``B ML``; or ``A ML`` + the opposite team's
  spread) have anti-correlated payoffs — they hedge each other, so the portfolio
  variance is *lower* than the independent sum implies and the log-optimal total
  stake can be larger.
* Two same-direction legs (``A ML`` + ``A -2.5``) have strongly positively
  correlated payoffs — they are nearly the same bet and should not both be sized
  as if independent.

This module models each event's legs jointly. Margin-based legs (moneyline,
spread) share a single latent "margin" normal; total legs share a single latent
"total" normal; everything else (e.g. player props) gets a private, independent
latent. We draw correlated win/loss scenarios via a Gaussian copula, then
maximize expected log bankroll growth over the leg stakes simultaneously.

The result is fractional-Kelly shrunk and bounded by:

* a per-leg cap (``max_fraction`` of bankroll), and
* a *variance-scaled* gross cap: the per-event gross exposure may grow from the
  base cap toward ``max_gross_cap`` only to the extent that hedging reduces
  portfolio variance below the naive independent sum.

For a single-leg event the optimizer reduces exactly to fractional Kelly, so
enabling it never changes the common case.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

# Axis labels for the latent-factor model.
AXIS_MARGIN = "margin"
AXIS_TOTAL = "total"
AXIS_INDEPENDENT = "independent"

# Keep the full-Kelly solve bounded and the log argument strictly positive:
# the worst (all-lose) scenario gives ``1 - sum(f)`` which must stay > 0.
_FULL_GROSS_CAP = 0.95
_FULL_LEG_CAP = 0.95


@dataclass
class JointLeg:
    """One bettable leg within an event, oriented onto a latent axis.

    ``win_prob`` is the marginal probability that this leg's YES side wins
    (the model's ``blended_true_prob``). ``decimal_odds`` is ``1 / yes_price``.
    ``axis`` / ``sign`` place the leg on the shared margin or total latent
    variable so the copula can correlate it with the event's other legs.
    """

    win_prob: float
    decimal_odds: float
    axis: str = AXIS_INDEPENDENT
    sign: float = 1.0
    confidence: float = 1.0
    liquidity: float = 1.0
    label: str = ""


@dataclass
class JointKellyConfig:
    """Tuning knobs for the joint optimizer."""

    kelly_multiplier: float = 0.25
    max_fraction: float = 0.05          # per-leg final cap (fraction of bankroll)
    base_gross_cap: float = 0.08        # per-event gross cap with no hedging
    max_gross_cap: float = 0.15         # ceiling reachable only via hedging
    rho_margin_total: float = 0.0       # latent corr between margin & total axes
    n_samples: int = 20000
    seed: int = 12345


@dataclass
class JointKellyResult:
    """Output of :func:`joint_kelly_fractions`.

    ``fractions`` are the final per-leg stakes as a fraction of bankroll, in the
    same order as the input legs. The remaining fields are diagnostics.
    """

    fractions: list[float]
    full_fractions: list[float] = field(default_factory=list)
    variance_ratio: float = 1.0
    allowed_gross: float = 0.0
    gross: float = 0.0


def _is_active(leg: JointLeg) -> bool:
    """A leg can carry stake only if it has valid odds and a real probability."""
    return (
        leg.decimal_odds > 1.0
        and 0.0 < leg.win_prob < 1.0
    )


def _independent_full_kelly(leg: JointLeg) -> float:
    """Single-bet full-Kelly fraction f* = (bp - q) / b, floored at 0."""
    b = leg.decimal_odds - 1.0
    p = leg.win_prob
    q = 1.0 - p
    f = (b * p - q) / b
    return f if f > 0 else 0.0


def _simulate_returns(legs: list[JointLeg], config: JointKellyConfig) -> np.ndarray:
    """Return an ``(n_samples, n_legs)`` matrix of per-unit net returns.

    Each entry is ``decimal_odds - 1`` if the leg wins in that scenario, else
    ``-1``. Wins are drawn from a Gaussian copula: margin legs share one latent
    normal, total legs share another (optionally correlated), and any other leg
    gets its own independent latent.
    """
    rng = np.random.default_rng(config.seed)
    n = len(legs)
    s = config.n_samples

    rho = float(np.clip(config.rho_margin_total, -0.99, 0.99))
    cov = np.array([[1.0, rho], [rho, 1.0]])
    z = rng.multivariate_normal([0.0, 0.0], cov, size=s)
    z_margin = z[:, 0]
    z_total = z[:, 1]

    latent = np.empty((s, n))
    for k, leg in enumerate(legs):
        if leg.axis == AXIS_MARGIN:
            base = z_margin
        elif leg.axis == AXIS_TOTAL:
            base = z_total
        else:
            base = rng.standard_normal(s)
        # sign * N(0,1) is still N(0,1); the sign sets the orientation so that
        # opposite-side legs on the same axis become anti-correlated.
        latent[:, k] = leg.sign * base

    # Threshold so that P(latent > t) == win_prob exactly in the marginal.
    probs = np.array([leg.win_prob for leg in legs])
    thresholds = norm.ppf(1.0 - probs)
    wins = latent > thresholds

    payoffs = np.array([leg.decimal_odds - 1.0 for leg in legs])
    returns = np.where(wins, payoffs, -1.0)
    return returns


def _variance_ratio(returns: np.ndarray, weights: np.ndarray) -> float:
    """Joint variance / independent variance under a reference weighting.

    < 1 means the legs hedge (joint variance below the diagonal-only sum);
    > 1 means they reinforce. Returns 1.0 when the reference variance is ~0.
    """
    if returns.shape[1] == 0 or not np.any(weights):
        return 1.0
    cov = np.cov(returns, rowvar=False)
    cov = np.atleast_2d(cov)
    diag = np.diag(cov)
    var_independent = float(np.sum((weights ** 2) * diag))
    if var_independent <= 1e-12:
        return 1.0
    var_joint = float(weights @ cov @ weights)
    return max(var_joint, 0.0) / var_independent


def _solve_full_kelly(
    returns: np.ndarray, warm_start: np.ndarray
) -> np.ndarray:
    """Maximize mean log(1 + R @ f) over f >= 0 with bounded gross exposure."""
    n = returns.shape[1]

    def neg_growth(f: np.ndarray) -> float:
        wealth = 1.0 + returns @ f
        # Bounds keep wealth >= 0.05; guard anyway for the optimizer's probes.
        wealth = np.clip(wealth, 1e-9, None)
        return -float(np.mean(np.log(wealth)))

    def neg_growth_jac(f: np.ndarray) -> np.ndarray:
        wealth = np.clip(1.0 + returns @ f, 1e-9, None)
        return -np.mean(returns / wealth[:, None], axis=0)

    bounds = [(0.0, _FULL_LEG_CAP)] * n
    constraints = [
        {"type": "ineq", "fun": lambda f: _FULL_GROSS_CAP - float(np.sum(f))}
    ]
    x0 = np.clip(warm_start, 0.0, _FULL_LEG_CAP)
    gross = float(np.sum(x0))
    if gross > _FULL_GROSS_CAP:
        x0 = x0 * (_FULL_GROSS_CAP / gross)

    res = minimize(
        neg_growth,
        x0,
        jac=neg_growth_jac,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-10},
    )
    f = res.x if res.success else x0
    return np.clip(f, 0.0, _FULL_LEG_CAP)


def joint_kelly_fractions(
    legs: list[JointLeg], config: Optional[JointKellyConfig] = None
) -> JointKellyResult:
    """Jointly size every leg of one event, returning fraction-of-bankroll stakes.

    Steps: (1) build a per-leg reference weighting from independent full Kelly,
    (2) Monte-Carlo correlated win/loss scenarios, (3) compute the variance
    ratio and the variance-scaled gross cap, (4) jointly maximize log growth,
    (5) apply the fractional multiplier + per-leg confidence/liquidity discounts,
    then enforce the per-leg and gross caps.
    """
    config = config or JointKellyConfig()
    n = len(legs)
    if n == 0:
        return JointKellyResult(fractions=[])

    active = [_is_active(leg) for leg in legs]
    if not any(active):
        return JointKellyResult(
            fractions=[0.0] * n,
            full_fractions=[0.0] * n,
            allowed_gross=config.base_gross_cap,
        )

    active_legs = [leg for leg, ok in zip(legs, active) if ok]
    returns = _simulate_returns(active_legs, config)

    ref_weights = np.array([_independent_full_kelly(leg) for leg in active_legs])
    vrr = _variance_ratio(returns, ref_weights)

    # Variance-scaled gross cap: expand toward the ceiling only as hedging cuts
    # variance (vrr < 1). Correlated legs (vrr >= 1) stay pinned at the base cap.
    allowed_gross = min(
        config.max_gross_cap, config.base_gross_cap / min(1.0, max(vrr, 1e-9))
    )
    allowed_gross = max(allowed_gross, 0.0)

    full = _solve_full_kelly(returns, ref_weights)

    # Fractional Kelly + per-leg discounts.
    shrunk = np.array(
        [
            f * config.kelly_multiplier * leg.confidence * leg.liquidity
            for f, leg in zip(full, active_legs)
        ]
    )
    # Per-leg hard cap.
    shrunk = np.minimum(shrunk, config.max_fraction)
    # Variance-scaled gross cap.
    gross = float(np.sum(shrunk))
    if gross > allowed_gross and gross > 0:
        shrunk = shrunk * (allowed_gross / gross)
        gross = float(np.sum(shrunk))
    # Snap optimizer dust (dominated legs land at ~1e-19) to a clean zero so
    # downstream sizing/logging/persistence never sees a microscopic stake.
    shrunk = np.where(shrunk < 1e-9, 0.0, shrunk)
    gross = float(np.sum(shrunk))

    # Scatter active results back into the full-length output vectors.
    out_fractions = [0.0] * n
    out_full = [0.0] * n
    ai = 0
    for i, ok in enumerate(active):
        if ok:
            out_fractions[i] = float(shrunk[ai])
            out_full[i] = float(full[ai])
            ai += 1

    return JointKellyResult(
        fractions=out_fractions,
        full_fractions=out_full,
        variance_ratio=float(vrr),
        allowed_gross=float(allowed_gross),
        gross=gross,
    )


def infer_axis_and_sign(
    market_type: str, yes_team: str, reference_team: Optional[str]
) -> tuple[str, float]:
    """Map a market to its latent axis and orientation sign.

    * ``total`` markets ride the total axis; ``over`` is +1, ``under`` is -1.
    * ``moneyline`` / ``spread`` markets ride the margin axis; +1 if the YES side
      backs the event's reference team, -1 if it backs the opponent.
    * Anything else (e.g. ``player_prop``) gets an independent axis.
    """
    mt = (market_type or "").lower()
    yt = (yes_team or "").lower().strip()

    if mt == "total":
        if "under" in yt or yt.startswith("u "):
            return AXIS_TOTAL, -1.0
        return AXIS_TOTAL, 1.0

    if mt in ("moneyline", "spread"):
        ref = (reference_team or "").lower().strip()
        if ref and yt and yt == ref:
            return AXIS_MARGIN, 1.0
        return AXIS_MARGIN, -1.0 if ref else 1.0

    return AXIS_INDEPENDENT, 1.0
