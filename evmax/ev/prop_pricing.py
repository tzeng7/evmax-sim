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
  - **Normal** — high-mean roughly symmetric stats: NBA points/PRA, NFL passing
    yards, MLB pitching outs. Needs a per-stat σ (one constraint, two
    parameters → fix σ from :data:`_NORMAL_STAT_SIGMA`, solve for μ).
  - **Gamma (fixed scale)** — right-skewed, zero-floored yardage: NFL receiving
    and rushing yards. Shape = μ/θ with a per-stat scale θ
    (:data:`_GAMMA_STAT_SCALE`), so the variance θ·μ grows with the player's
    median instead of sitting at one σ for every player. Fit offline against
    settled outcomes by ``scripts/fit_nfl_prop_dispersion.py``.
  - **Poisson** — single-event count stats with var/μ ≈ 1: NFL anytime-TD style
    props. One-parameter, self-fitting from a single anchor. Kept as a
    defensive fallback for any low-mean stat where NegBin's k is essentially
    infinite (NegBin → Poisson as k → ∞).

Designed to be sport-agnostic — adding a new NFL stat is just an entry in
:data:`_STAT_DISTRIBUTION` and (for Normal/Gamma/NegBin stats) the corresponding
parameter table.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from scipy import optimize, special, stats


# Per-stat distribution choice. Keys are the canonical evmax stat_type strings
# emitted by parse_prop_description in evmax/clients/esports_pinnacle.py.
#
# Distribution choice driven by walk-forward backtest in
# scripts/backtest_prop_pricing.py against 2024-25 NBA player game logs:
#   - High-mean roughly-bell-shaped stats (points, PRA) → Normal beat NegBin
#     at extreme thresholds (Δ+5 from rolling mean) by ~4% relative Brier —
#     Normal handles the symmetric tail well when σ ≪ μ. (NFL receiving and
#     rushing yards were Normal too until 2026-10-09; their tails are skewed and
#     their spread scales with the median, so they moved to Gamma.)
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
    # NFL receiving/rushing yards → fixed-scale Gamma (2026-10-09). The fixed-σ
    # Normal priced every player with the same SD (24 / 30 yd): too thin in the
    # upper tail for a high-median WR (P(≥ line+60) 0.6% vs 4.7% realized) and
    # mass below zero for a 12-yard receiver, which made deep Kalshi rungs look
    # like free NO edges. See _GAMMA_STAT_SCALE for the fit and the evidence.
    "rushing_yards":           "gamma",
    "receiving_yards":         "gamma",
    # --- NFL count props (added 2026-09-06) ---
    # These have live Pinnacle anchors (Total Receptions / Total Touchdown
    # Passes on sport 15) but were previously absent from this table, so
    # price_kalshi_threshold returned None and ~31% of the NFL prop book
    # (unpriced=630 on the 2026-09-06 scan) was silently discarded one dict
    # entry short of tradeable.
    #   - receptions (WR/TE/RB per game): low mean (~3-6), overdispersed
    #     (target share + game script drive variance) → NegBin.
    #   - passing_tds (QB per game): low-mean, var/μ ≈ 1 → Poisson.
    "receptions":              "negbin",
    "passing_tds":             "poisson",
    # --- MLB player props (added 2026-06-27) ---
    # Per-game count stats. k values are PROVISIONAL priors pending calibration
    # in scripts/backtest_mlb_props.py against historical box scores — see the
    # _NEGBIN_STAT_K note. Distribution choice rationale:
    #   - strikeouts (pitcher/start): mean ~5-6, mild overdispersion → NegBin.
    #   - pitching_outs (per start): bounded, roughly bell-shaped around ~18
    #     outs (6 IP) → Normal, σ from innings dispersion.
    #   - total_bases / hits / hits_runs_rbis (hitter/game): low-mean, many
    #     zeros, heavy upper tail → NegBin.
    #   - home_runs / rbis (hitter/game): rare low-mean events, var/μ ≈ 1 →
    #     Poisson (NegBin k → ∞).
    "strikeouts":              "negbin",
    "pitching_outs":           "normal",
    "total_bases":             "negbin",
    "hits":                    "negbin",
    "hits_runs_rbis":          "negbin",
    "home_runs":               "poisson",
    "rbis":                    "poisson",
}

# Per-stat σ for Normal-priced stats. PRA σ updated 2026-05-10 from the
# empirical 2024-25 shufinskiy parquet (327 high-volume players, ≥30 GP).
# NFL receiving/rushing yards left this table on 2026-10-09 (now Gamma, see
# _GAMMA_STAT_SCALE). passing_yards keeps σ=70: on the same 2026 Weeks 1-5
# check (n=130 QB-games) the Normal sat within 1 SE of realized at every
# offset from line−40 to line+80.
_NORMAL_STAT_SIGMA: dict[str, float] = {
    "points":                  5.9,   # 2024-25 high-vol cohort empirical
    "points_rebounds_assists": 7.5,
    "passing_yards":           70.0,
    # MLB: a starter's outs recorded ≈ 18 ± ~5 (a clean 6 IP is 18 outs; a
    # short hook or a complete-game both happen). Provisional pending calibration.
    "pitching_outs":           5.0,
}

# Per-stat Gamma scale θ (stat units) for Gamma-priced stats. The shape is
# μ/θ, so Var = θ·μ and SD = √(θ·μ): at θ=25.4 a 12.5-yard 50/50 line gives
# μ ≈ 20, SD ≈ 23 yd; an 80.5-yard line gives μ ≈ 89, SD ≈ 47 yd. Skew falls
# as the median rises.
#
# Fitted 2026-10-09 by scripts/fit_nfl_prop_dispersion.py: one closing Pinnacle
# anchor per player-game (2026 Weeks 1-5, recovered from the archive), scored
# as rung Brier on every Kalshi threshold actually listed against the settled
# stat. Variance power was free in the search (Var = φ·μ^p): p fitted between
# 0.9 and 1.4 across folds at no Brier gain over p=1, so the one-parameter
# form ships.
# Out of sample (θ fit on Weeks 1-2 only, scored on Weeks 3-5, Brier/1000 with
# player-game-clustered SE):
#   receiving  Normal σ=24 174.2 → Gamma 168.8  (Δ −5.4 ± 1.7, 366 player-games)
#              Gamma − Kalshi mid on the same rungs: −0.6 ± 0.8
#   rushing    Normal σ=30 163.3 → Gamma 160.4  (Δ −2.9 ± 2.3, 183 player-games)
#              Gamma − Kalshi mid on the same rungs: +1.2 ± 1.0
# The holdout Brier is flat for θ within ±5 of these values. A zero-inflated
# Gamma (extra "dud" mass at 0) gained only −1.0 ± 0.5 / −0.8 ± 1.0, thinned the
# upper tail and its mass parameter swung 0.03-0.18 between folds — not shipped.
# Known residual: high-median receivers have a fatter LOWER tail than this
# family (early exits; rungs ≥30 yd below the line realize 68% vs 90% priced).
# Refit with the script as the season accrues.
_GAMMA_STAT_SCALE: dict[str, float] = {
    "receiving_yards": 25.4,
    "rushing_yards":   15.3,
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
    # NFL receptions: k=5 re-checked 2026-10-09 by
    # scripts/fit_nfl_prop_dispersion.py (same anchor/outcome sample as the
    # yardage Gamma) and KEPT. A refit k is not better out of sample (Weeks 3-5
    # holdout Brier +0.5 ± 1.1 /1000 vs k=5; NB1, Var = μ(1+δ), ties too)
    # because the upper tail is unstable: Weeks 1-2 fit k≈50, Weeks 3-5
    # match k=5. One stable miss: k=5 prices P(≥ line−1) ~3.5pp too low in both
    # halves (fewer duds than NegBin implies).
    "receptions":       5.0,
    # MLB (PROVISIONAL — calibrate via scripts/backtest_mlb_props.py --report-k
    # against historical box scores before trusting tail thresholds). Hitter
    # per-game counts are far more overdispersed than NBA stats (most games are
    # 0, a multi-hit/multi-base game is the tail), so k is much smaller.
    "strikeouts":      12.0,  # pitcher K per start — mild overdispersion
    "total_bases":      3.0,  # hitter — heavy zero mass + 4-base tail
    "hits":             4.0,  # hitter
    "hits_runs_rbis":   4.0,  # hitter composite
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


@dataclass(frozen=True)
class GammaProp(PropDistribution):
    """Gamma with a fixed scale θ and shape μ/θ — variance θ·μ grows with μ.

    Built for NFL yardage, which is right-skewed and floored at zero. With θ
    fixed per stat the SD is √(θ·μ), so a high-median player gets a wider
    distribution than a low-median one, and no player gets mass below zero.
    μ (the mean) is fitted from the Pinnacle anchor; θ comes from
    :data:`_GAMMA_STAT_SCALE`. Continuity convention matches
    :class:`NormalProp`: an integer stat ``X >= K`` is ``X > K - 0.5``.
    """
    mu: float
    scale: float

    @property
    def shape(self) -> float:
        return self.mu / self.scale

    def _sf(self, x: float) -> float:
        if x <= 0:
            return 1.0
        return float(special.gammaincc(self.mu / self.scale, x / self.scale))

    @classmethod
    def from_anchor(
        cls,
        line: float,
        prob_over: float,
        scale: float,
    ) -> Optional["GammaProp"]:
        """Fit μ such that ``P(X > line) = prob_over``, given the scale θ.

        ``P(X > line)`` rises monotonically with μ (shape and mean grow
        together at fixed θ), so a bracketed root find is exact. ``line`` must
        be positive: a zero-floored family cannot place a line at or below 0.
        """
        if not math.isfinite(line) or not math.isfinite(prob_over) or not math.isfinite(scale):
            return None
        if scale <= 0 or line <= 0:
            return None
        if not (0.001 < prob_over < 0.999):
            return None

        def _residual(mu: float) -> float:
            return float(special.gammaincc(mu / scale, line / scale)) - prob_over

        lo = 1e-6
        hi = max(line, scale) * 4.0
        while _residual(hi) <= 0:
            hi *= 2.0
            if hi > 1e7:
                return None
        try:
            mu = optimize.brentq(_residual, a=lo, b=hi, xtol=1e-9)
        except (ValueError, RuntimeError):
            return None
        return cls(mu=float(mu), scale=float(scale))

    def prob_at_or_above(self, threshold: float) -> float:
        return self._sf(threshold - 0.5)


def fit_distribution(
    stat_type: str,
    line: float,
    prob_over: float,
) -> Optional[PropDistribution]:
    """Dispatch a per-stat distribution fit from a single Pinnacle anchor.

    Returns ``None`` when:
      - ``stat_type`` is not in :data:`_STAT_DISTRIBUTION`,
      - the anchor probability is degenerate (≤ 0.001 or ≥ 0.999),
      - a Normal stat is missing a σ entry in :data:`_NORMAL_STAT_SIGMA`,
      - a Gamma stat is missing a θ entry in :data:`_GAMMA_STAT_SCALE`, or
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
    if dist_name == "gamma":
        scale = _GAMMA_STAT_SCALE.get(stat_type)
        if scale is None:
            return None
        return GammaProp.from_anchor(line, prob_over, scale=scale)
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


def distribution_from_mean(
    stat_type: str,
    mean: float,
) -> Optional[PropDistribution]:
    """Build the per-stat distribution from a model-projected MEAN.

    The market-anchor path (:func:`fit_distribution`) solves for the
    distribution parameter that reproduces a Pinnacle line+prob. This is the
    model-side complement: a projection model supplies the expected value
    directly (e.g. expected strikeouts = k_rate × batters_faced), and we wrap
    it in the SAME per-stat distribution family (and dispersion table) so model
    and sharp probabilities are read off comparable surfaces.

    Returns ``None`` for an unknown stat, a non-finite/negative mean, or a
    Normal/Gamma/NegBin stat missing its dispersion-table entry.
    """
    if not math.isfinite(mean) or mean < 0:
        return None
    dist_name = _STAT_DISTRIBUTION.get(stat_type)
    if dist_name == "negbin":
        k = _NEGBIN_STAT_K.get(stat_type)
        if k is None or k <= 0:
            return None
        return NegBinomialProp(mu=float(mean), k=float(k))
    if dist_name == "normal":
        sigma = _NORMAL_STAT_SIGMA.get(stat_type)
        if sigma is None:
            return None
        return NormalProp(mu=float(mean), sigma=float(sigma))
    if dist_name == "gamma":
        scale = _GAMMA_STAT_SCALE.get(stat_type)
        if scale is None:
            return None
        # Shape μ/θ must stay positive; same tiny floor as the Poisson branch.
        return GammaProp(mu=max(float(mean), 1e-6), scale=float(scale))
    if dist_name == "poisson":
        # Poisson is degenerate at λ=0 (all mass on 0); clamp to a tiny floor so
        # prob_at_or_above stays well-defined for rare-event stats (HR/RBI).
        return PoissonProp(lam=max(float(mean), 1e-6))
    return None


def model_prob_at_or_above(
    stat_type: str,
    mean: float,
    threshold: float,
) -> Optional[float]:
    """``P(stat >= threshold)`` from a model-projected mean. ``None`` if unpriceable."""
    dist = distribution_from_mean(stat_type, mean)
    if dist is None:
        return None
    return dist.prob_at_or_above(threshold)


def supported_stats() -> frozenset[str]:
    """Stat types this module can price. Useful for guards in the coordinator."""
    return frozenset(_STAT_DISTRIBUTION.keys())
