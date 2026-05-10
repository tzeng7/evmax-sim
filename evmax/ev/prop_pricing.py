"""Prop pricing — convert a single Pinnacle anchor into a probability for any Kalshi threshold.

Pinnacle posts ONE half-point line per (player, stat) — e.g. ``cade_cunningham
points 26.5`` over/under. Kalshi posts MANY integer ``X+`` thresholds for the
same (player, stat) — ``20+, 25+, 30+, 35+ points``. Exact-line matching only
catches the rare case where Kalshi happens to land on Pinnacle's line; the rest
go unpriced.

This module fits a parametric distribution to the Pinnacle anchor (line +
prob_over) so we can read off ``P(stat >= K)`` for any Kalshi threshold ``K``.

Distribution choice is per stat-type via :data:`_STAT_DISTRIBUTION`:

  - **NegativeBinomial** — overdispersed count stats. Variance = μ + μ²/k where
    k is the per-stat dispersion calibrated from historical data
    (:data:`_NEGBIN_STAT_K`). NegBin handles tail events that Poisson under-
    weights — the textbook fix when var/μ > 1, well-established in basketball
    analytics literature for player scoring. Used for all NBA count props.
  - **Normal** — high-mean continuous-style stats: NBA PRA, NFL yardage. Needs
    a per-stat σ (one constraint, two parameters → fix σ from
    :data:`_NORMAL_STAT_SIGMA`, solve for μ).
  - **Poisson** — single-event count stats with var/μ ≈ 1: NFL anytime-TD style
    props. One-parameter, self-fitting from a single anchor. Kept as a
    defensive fallback for any low-mean stat where NegBin's k is essentially
    infinite (NegBin → Poisson as k → ∞).

Designed to be sport-agnostic — adding a new NFL stat is just an entry in
:data:`_STAT_DISTRIBUTION` and (for Normal/NegBin stats) the corresponding
parameter table.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from scipy import optimize, stats


# Per-stat distribution choice. Keys are the canonical evmax stat_type strings
# emitted by parse_prop_description in evmax/clients/esports_pinnacle.py.
#
# Distribution choice driven by walk-forward backtest in
# scripts/backtest_prop_pricing.py against 2024-25 NBA player game logs:
#   - High-mean roughly-bell-shaped stats (points, PRA, NFL yards) → Normal
#     beat NegBin at extreme thresholds (Δ+5 from rolling mean) by ~4% relative
#     Brier — Normal handles the symmetric tail well when σ ≪ μ.
#   - Low-mean count stats (rebounds, assists, threes, steals, blocks) → NegBin
#     beat both Normal variants because Normal puts probability mass below
#     zero at low μ, biasing the upper tail. NegBin also beats Poisson
#     uniformly because var/μ > 1 for every NBA count stat empirically.
_STAT_DISTRIBUTION: dict[str, str] = {
    "points":                  "normal",   # backtested: 0.0022 Brier better than NegBin
    "rebounds":                "negbin",
    "assists":                 "negbin",
    "threes":                  "negbin",
    "steals":                  "negbin",
    "blocks":                  "negbin",
    "points_rebounds_assists": "normal",
    "passing_yards":           "normal",
    "rushing_yards":           "normal",
    "receiving_yards":         "normal",
}

# Per-stat σ for Normal-priced stats. Yardage σ values mirror YARDAGE_STAT_SIGMA
# in evmax/clients/nfl_props_cache.py so anchor pricing and the L15 cache share
# the same dispersion assumptions. PRA σ updated 2026-05-10 from the empirical
# 2024-25 shufinskiy parquet (327 high-volume players, ≥30 GP).
_NORMAL_STAT_SIGMA: dict[str, float] = {
    "points":                  5.9,   # 2024-25 high-vol cohort empirical
    "points_rebounds_assists": 7.5,
    "passing_yards":           70.0,
    "rushing_yards":           30.0,
    "receiving_yards":         24.0,
}

# Per-stat NegBin dispersion k. Var = μ + μ²/k; smaller k = more overdispersion.
#
# Derived from 2024-25 NBA player-game logs (≥30 GP, ≥12 MIN/game), filtered to
# the HIGH-VOLUME cohort (top 25% of players by per-stat mean) — this is the
# population Pinnacle actually posts props on. The all-player median k is much
# lower (~5 for points), but it's inflated by bench players whose erratic
# minutes drive game-to-game variance well above what a starter sees. See
# scripts/calibrate_prop_sigma.py --report-k for the per-stat derivation.
_NEGBIN_STAT_K: dict[str, float] = {
    "points":   13.0,
    "rebounds": 19.0,
    "assists":  25.0,
    "threes":   16.0,
    "steals":   8.0,
    "blocks":   8.0,
}


class PropDistribution(ABC):
    """Pricing surface fit to a single Pinnacle anchor.

    Concrete subclasses implement :meth:`prob_at_or_above` for the Kalshi
    ``X+`` semantics: ``P(stat >= threshold)``.
    """

    @abstractmethod
    def prob_at_or_above(self, threshold: float) -> float:
        """Return ``P(stat >= threshold)``."""


@dataclass(frozen=True)
class PoissonProp(PropDistribution):
    lam: float

    @classmethod
    def from_anchor(cls, line: float, prob_over: float) -> Optional["PoissonProp"]:
        """Fit ``λ`` such that ``P(X > line) = prob_over``.

        ``line`` is Pinnacle's half-point line (e.g. 26.5). For an integer-
        valued stat ``X > 26.5`` is the same as ``X >= 27``, so the Poisson
        constraint is ``P(X >= ceil(line)) = prob_over``.
        """
        if not math.isfinite(line) or not math.isfinite(prob_over):
            return None
        if not (0.001 < prob_over < 0.999):
            return None
        cutoff = int(math.ceil(line + 1e-9))  # smallest integer k with k > line
        if cutoff < 1:
            return None
        target_cdf = 1.0 - prob_over  # P(X <= cutoff - 1)
        try:
            lam = optimize.brentq(
                lambda l: stats.poisson.cdf(cutoff - 1, l) - target_cdf,
                a=1e-6, b=500.0, xtol=1e-7,
            )
        except (ValueError, RuntimeError):
            return None
        return cls(lam=float(lam))

    def prob_at_or_above(self, threshold: float) -> float:
        # Kalshi 'X+' is integer-valued. If threshold is non-integer, take ceil
        # because 'score > 24.5' = 'score >= 25' for an integer counting stat.
        cutoff = int(math.ceil(threshold - 1e-9))
        if cutoff <= 0:
            return 1.0
        return float(stats.poisson.sf(cutoff - 1, self.lam))


@dataclass(frozen=True)
class NegBinomialProp(PropDistribution):
    """Negative Binomial — variance = μ + μ²/k. Heavier tails than Poisson.

    Parameterized via scipy as ``nbinom(n=k, p=k/(k+μ))``. The dispersion
    parameter ``k`` is fixed from a per-stat empirical table; ``μ`` is fitted
    from the Pinnacle anchor.
    """
    mu: float
    k: float

    @classmethod
    def from_anchor(
        cls,
        line: float,
        prob_over: float,
        k: float,
    ) -> Optional["NegBinomialProp"]:
        """Fit μ such that ``P(X > line) = prob_over`` under NegBin(μ, k).

        ``line`` is Pinnacle's half-point line (e.g. 26.5). For an integer-
        valued stat ``X > 26.5`` is the same as ``X >= 27``, so the constraint
        is ``P(X >= ceil(line)) = prob_over``. As ``k → ∞`` this degenerates
        to Poisson.
        """
        if not math.isfinite(line) or not math.isfinite(prob_over) or not math.isfinite(k):
            return None
        if k <= 0:
            return None
        if not (0.001 < prob_over < 0.999):
            return None
        cutoff = int(math.ceil(line + 1e-9))
        if cutoff < 1:
            return None
        target_cdf = 1.0 - prob_over

        def _residual(mu: float) -> float:
            p = k / (k + mu)
            return float(stats.nbinom.cdf(cutoff - 1, n=k, p=p)) - target_cdf

        try:
            mu = optimize.brentq(_residual, a=1e-6, b=1000.0, xtol=1e-7)
        except (ValueError, RuntimeError):
            return None
        return cls(mu=float(mu), k=float(k))

    def prob_at_or_above(self, threshold: float) -> float:
        cutoff = int(math.ceil(threshold - 1e-9))
        if cutoff <= 0:
            return 1.0
        p = self.k / (self.k + self.mu)
        return float(stats.nbinom.sf(cutoff - 1, n=self.k, p=p))


@dataclass(frozen=True)
class NormalProp(PropDistribution):
    mu: float
    sigma: float

    @classmethod
    def from_anchor(
        cls,
        line: float,
        prob_over: float,
        sigma: float,
    ) -> Optional["NormalProp"]:
        """Fit μ such that ``P(X > line) = prob_over``, given σ.

        With σ fixed, μ = ``line + σ · Φ⁻¹(prob_over)``. When ``prob_over =
        0.5`` this collapses to ``μ = line`` (the line is at the median).
        """
        if not math.isfinite(line) or not math.isfinite(prob_over) or not math.isfinite(sigma):
            return None
        if sigma <= 0:
            return None
        if not (0.001 < prob_over < 0.999):
            return None
        z = float(stats.norm.ppf(prob_over))
        mu = line + sigma * z
        return cls(mu=float(mu), sigma=float(sigma))

    def prob_at_or_above(self, threshold: float) -> float:
        # Kalshi 'X+' = X >= threshold. Apply continuity correction so the
        # integer ≥-threshold maps to a continuous > (threshold - 0.5) cutoff.
        return float(stats.norm.sf(threshold - 0.5, loc=self.mu, scale=self.sigma))


def fit_distribution(
    stat_type: str,
    line: float,
    prob_over: float,
) -> Optional[PropDistribution]:
    """Dispatch a per-stat distribution fit from a single Pinnacle anchor.

    Returns ``None`` when:
      - ``stat_type`` is not in :data:`_STAT_DISTRIBUTION`,
      - the anchor probability is degenerate (≤ 0.001 or ≥ 0.999),
      - a Normal stat is missing a σ entry in :data:`_NORMAL_STAT_SIGMA`, or
      - a NegBin stat is missing a k entry in :data:`_NEGBIN_STAT_K`.
    """
    dist_name = _STAT_DISTRIBUTION.get(stat_type)
    if dist_name == "negbin":
        k = _NEGBIN_STAT_K.get(stat_type)
        if k is None:
            return None
        return NegBinomialProp.from_anchor(line, prob_over, k=k)
    if dist_name == "normal":
        sigma = _NORMAL_STAT_SIGMA.get(stat_type)
        if sigma is None:
            return None
        return NormalProp.from_anchor(line, prob_over, sigma=sigma)
    if dist_name == "poisson":
        return PoissonProp.from_anchor(line, prob_over)
    return None


def price_kalshi_threshold(
    stat_type: str,
    pinn_line: float,
    pinn_prob_over: float,
    kalshi_threshold: float,
) -> Optional[float]:
    """Top-level convenience: ``P(stat >= kalshi_threshold)`` from a Pinnacle anchor.

    Returns ``None`` when the anchor cannot be fit (see :func:`fit_distribution`).
    """
    dist = fit_distribution(stat_type, pinn_line, pinn_prob_over)
    if dist is None:
        return None
    return dist.prob_at_or_above(kalshi_threshold)


def supported_stats() -> frozenset[str]:
    """Stat types this module can price. Useful for guards in the coordinator."""
    return frozenset(_STAT_DISTRIBUTION.keys())
