"""Key-number-aware margin PMF for alt-spread pricing (NFL).

``SpreadDistributionModel`` prices every alt-spread rung by extrapolating a
continuous normal CDF off Pinnacle's devigged main line. NFL margins are not
continuous: about 15% of games (2003-2025) are decided by exactly 3 points and
about 9% by exactly 7. A normal with σ=14 puts about 5.5% on a margin of ±3.
The normal therefore misprices every rung whose distance from the main line
crosses 3 or 7. This module replaces the normal for the sectors that ship a
fitted artifact.

Model (favorite-margin axis)
----------------------------
``F`` is the favorite's final margin (an integer). The favorite is Pinnacle's
``outcome_a`` (``_parse_spread`` puts the negative handicap on ``outcome_a``).

    P(F = k | mu) ∝ exp( -(k - mu)² / (2·s(mu)²) + beta_k + gamma[bucket, k] )

- ``s(mu) = s0 + s1·mu`` is a spread-dependent kernel width.
- ``beta_k`` is a signed key-number log-multiplier ("win by 3" and "lose by 3"
  are separate parameters). The fit only frees it for ``|k| <= K``.
- ``gamma`` is a per-spread-bucket deviation from ``beta`` (buckets
  ``<= 3``, ``3 < . <= 7``, ``> 7``).

Anchoring
---------
The fit uses the closing spread as ``mu``, so the fitted bucket covariate is
the SPREAD bucket. At pricing time the bucket is taken from the main line
``a = |main line|`` and ``mu`` is anchored so that the model reproduces
Pinnacle's own devigged main-line price::

    P(F > a | F != a) == devig_cover_prob

The condition is push-conditional because a sportsbook refunds a push on a
whole-number line. On a half-point line the push mass is zero, so the
condition reduces to ``P(F > a)``.

Why the bucket comes from the main line and not from the anchored ``mu``: the
research sketch keyed the bucket on ``mu``. That makes the anchor equation
discontinuous at ``mu = 3`` and ``mu = 7``. brentq then stops on the jump, and
the anchored main-line price misses Pinnacle's devig by up to 2pp for common
lines (for example ``-3`` at ``p`` 0.49-0.525). Keying on the main line keeps
the bucket constant during the root search, so the anchor is exact. The two
variants are equal within noise on every measured lens: inner split 2015-18
ΔBrier +0.04/1000 (z +0.8), holdout 2019-25 -0.02/1000 (z -0.6), 0.73pp MAE vs
Pinnacle's alt ladder for both.

Sign conventions (identical to ``SpreadDistributionModel.predict``)
-------------------------------------------------------------------
``target_line`` is the venue line from the YES team's perspective.

- YES = favorite (``yes_is_underdog=False``): YES covers iff ``F > -target_line``.
- YES = underdog (``yes_is_underdog=True``): YES covers iff ``F < target_line``.

Examples: favorite -7.5 covers iff ``F > 7.5``. Underdog +7.5 covers iff
``F < 7.5``. "Underdog wins by over 3.5" is underdog at -3.5 and covers iff
``F < -3.5``. That rung is NOT the complement of favorite -3.5.

A whole-number TARGET rung is also priced push-conditionally
(``P(F > t | F != t)``). This keeps ``P(YES) + P(NO) == 1`` for the scanner's
NO-side derivation. Every Kalshi/Polymarket US NFL spread line logged to date is
a half-point, so this branch does not change a logged row today.

Validation (research 2026-09-22; ``scripts/fit_nfl_margin_pmf.py`` reprints it)
------------------------------------------------------------------------------
Fit on nflverse 2003-2018, evaluated once on 2019-2025 (1,960 games, 55,924
half-point rungs within 14 points of the main line): ΔBrier -1.62/1000 vs the
production normal anchored on the same juice (z -4.95, clustered by game);
-1.76/1000 (z -5.4) vs a normal centred on the spread. The gain is on rungs
that cross 3 or 7 (-2.40/1000, z -5.7); rungs that cross neither are unchanged
(-0.02/1000). Better in 7/7 walk-forward seasons (2019-2025). Against
Pinnacle's own archived alt ladder (17 games, 184 rungs) the mean absolute
error is 0.73pp vs 2.90pp for the normal. Anchored at the 2003-25 pooled
cover rate of -3 favorites (0.474), the model reproduces the pooled empirical
P(win by 8+) 0.295 and P(win by 4+) 0.432.

Pure numpy/scipy. No network access. The only I/O is the artifact read.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
import structlog
from scipy.optimize import brentq

logger = structlog.get_logger(__name__)

SCHEMA_VERSION = 1

# Artifacts live next to the other model state files. The path is resolved from
# this file (not the CWD) so scheduled runs and worktrees read the same file.
MODELS_DIR = Path(__file__).resolve().parents[2] / "data" / "models"

# Half-width (points) of the brentq bracket around the main line for the anchor
# search. This matches the research evaluation. A devigged main-line price
# needs an implausibly large mean shift (about 1.8σ) to fall outside it.
ANCHOR_BRACKET_PTS = 25.0

# Anchor results are cached per (|main line|, devig prob). Every venue rung of
# one game shares one Pinnacle main line, so a scan anchors once per game.
_ANCHOR_CACHE_SIZE = 4096


class MarginPMFArtifactError(ValueError):
    """The artifact is structurally invalid (wrong schema or shapes)."""


def artifact_path(sector: str) -> Path:
    """Path of the fitted artifact for ``sector`` (e.g. ``nfl_margin_pmf.json``)."""
    return MODELS_DIR / f"{sector}_margin_pmf.json"


def favorite_threshold(target_line: float, yes_is_underdog: bool) -> float:
    """Map a YES-perspective venue line to the threshold ``t`` on the F axis.

    YES = favorite at line L covers iff ``F > -L``, so ``t = -L``.
    YES = underdog at line L covers iff ``F < L``, so ``t = L``.
    """
    return float(target_line) if yes_is_underdog else -float(target_line)


class MarginPMF:
    """Discrete favorite-margin distribution with key-number multipliers."""

    def __init__(
        self,
        ks: np.ndarray,
        s0: float,
        s1: float,
        beta: np.ndarray,
        gamma: np.ndarray,
        bucket_edges: tuple[float, float],
        sector: str = "nfl",
        meta: Optional[dict[str, Any]] = None,
    ) -> None:
        self.ks = np.asarray(ks, dtype=float)
        self.s0 = float(s0)
        self.s1 = float(s1)
        self.beta = np.asarray(beta, dtype=float)
        self.gamma = np.asarray(gamma, dtype=float)
        self.bucket_edges = (float(bucket_edges[0]), float(bucket_edges[1]))
        self.sector = sector
        self.meta = dict(meta or {})
        # beta + gamma[bucket] is constant per bucket. Precompute it once so
        # each pmf() call is one exp over the support.
        self._offset = self.beta[None, :] + self.gamma
        self._anchor_cached = lru_cache(maxsize=_ANCHOR_CACHE_SIZE)(self._anchor_uncached)

    # ------------------------------------------------------------------
    # Construction / validation
    # ------------------------------------------------------------------

    @classmethod
    def from_artifact(cls, art: dict[str, Any]) -> "MarginPMF":
        """Build from a parsed artifact dict. Raise MarginPMFArtifactError if invalid."""
        if not isinstance(art, dict):
            raise MarginPMFArtifactError("artifact is not a JSON object")
        if art.get("schema_version") != SCHEMA_VERSION:
            raise MarginPMFArtifactError(
                f"schema_version {art.get('schema_version')!r} != {SCHEMA_VERSION}"
            )
        try:
            ks = np.asarray(art["ks"], dtype=float)
            beta = np.asarray(art["beta"], dtype=float)
            gamma = np.asarray(art["gamma"], dtype=float)
            s0 = float(art["s0"])
            s1 = float(art["s1"])
            edges = art["config"]["bucket_edges"]
        except (KeyError, TypeError, ValueError) as exc:
            raise MarginPMFArtifactError(f"missing or malformed field: {exc}") from exc

        n = len(ks)
        if n < 3 or ks.ndim != 1:
            raise MarginPMFArtifactError("ks must be a 1-D support of length >= 3")
        if not np.all(np.diff(ks) == 1.0) or not np.all(ks == np.round(ks)):
            raise MarginPMFArtifactError("ks must be consecutive integers")
        if beta.shape != (n,):
            raise MarginPMFArtifactError(f"beta shape {beta.shape} != ({n},)")
        if gamma.shape != (3, n):
            raise MarginPMFArtifactError(f"gamma shape {gamma.shape} != (3, {n})")
        if len(edges) != 2 or not float(edges[0]) < float(edges[1]):
            raise MarginPMFArtifactError(f"bucket_edges must be 2 ascending values, got {edges}")
        if not (np.all(np.isfinite(beta)) and np.all(np.isfinite(gamma))
                and math.isfinite(s0) and math.isfinite(s1)):
            raise MarginPMFArtifactError("non-finite parameter")
        # s(mu) must stay positive over every location a real main line can
        # anchor to (|mu| well under 60 points).
        if min(s0 + s1 * -60.0, s0 + s1 * 60.0) <= 1.0:
            raise MarginPMFArtifactError(f"kernel width s0={s0}, s1={s1} is not positive")
        meta = {k: v for k, v in art.items() if k not in ("ks", "beta", "gamma", "s0", "s1")}
        return cls(ks, s0, s1, beta, gamma, (edges[0], edges[1]),
                   sector=str(art.get("sector", "nfl")), meta=meta)

    # ------------------------------------------------------------------
    # Distribution
    # ------------------------------------------------------------------

    def sigma(self, mu: float) -> float:
        """Kernel width s(mu) = s0 + s1·mu."""
        return self.s0 + self.s1 * float(mu)

    def bucket(self, spread: float) -> int:
        """Spread bucket for the gamma interaction: 0 (<=3), 1 (<=7), 2 (>7)."""
        lo, hi = self.bucket_edges
        return 0 if spread <= lo else (1 if spread <= hi else 2)

    def pmf(self, mu: float, bucket: Optional[int] = None) -> np.ndarray:
        """P(F = k | mu) over ``self.ks``. Sums to 1.

        ``bucket`` defaults to ``bucket(mu)``, the training convention where
        the location IS the closing spread. Pricing passes the main line's
        bucket explicitly (see the module docstring).
        """
        b = self.bucket(mu) if bucket is None else bucket
        sig = self.sigma(mu)
        logit = -((self.ks - mu) ** 2) / (2.0 * sig * sig) + self._offset[b]
        e = np.exp(logit - logit.max())
        return e / e.sum()

    def split(self, mu: float, t: float, bucket: Optional[int] = None) -> tuple[float, float]:
        """(P(F > t), P(F < t)) at location ``mu``. P(F == t) is the remainder."""
        p = self.pmf(mu, bucket)
        return float(p[self.ks > t].sum()), float(p[self.ks < t].sum())

    def conditional_survival(self, mu: float, t: float, bucket: Optional[int] = None) -> float:
        """P(F > t | F != t). Equal to P(F > t) for a half-point ``t``."""
        up, dn = self.split(mu, t, bucket)
        return up / (up + dn)

    # ------------------------------------------------------------------
    # Anchoring + pricing
    # ------------------------------------------------------------------

    def anchor(self, main_line: float, devig_cover_prob: float) -> Optional[float]:
        """Location ``mu`` with P(F > a | F != a) == devig_cover_prob, a = |main_line|.

        The bucket is ``bucket(a)`` throughout the search, so the equation is
        continuous in ``mu`` and the root reproduces the devig exactly.
        Returns None when no root exists inside the bracket (an implausible
        main-line price). The caller then falls back to the normal CDF.
        """
        p = float(devig_cover_prob)
        if not (0.0 < p < 1.0) or not math.isfinite(float(main_line)):
            return None
        return self._anchor_cached(abs(float(main_line)), p)

    def _anchor_uncached(self, a: float, p: float) -> Optional[float]:
        b = self.bucket(a)

        def f(mu: float) -> float:
            return self.conditional_survival(mu, a, b) - p

        lo, hi = a - ANCHOR_BRACKET_PTS, a + ANCHOR_BRACKET_PTS
        try:
            f_lo, f_hi = f(lo), f(hi)
            if not (f_lo < 0.0 < f_hi):
                return None
            return float(brentq(f, lo, hi, xtol=1e-10))
        except (ValueError, FloatingPointError, ZeroDivisionError):
            return None

    def cover_probability(
        self,
        main_line: float,
        devig_cover_prob: float,
        target_line: float,
        yes_is_underdog: bool,
    ) -> Optional[float]:
        """P(YES covers ``target_line``), anchored on Pinnacle's main line.

        Args:
            main_line: Pinnacle ``outcome_a`` (favorite) spread line, e.g. -3.0.
            devig_cover_prob: Pinnacle's devigged P(favorite covers main_line).
            target_line: venue line from the YES team's perspective.
            yes_is_underdog: True when YES is ``outcome_b`` (the underdog).

        Returns None when the anchor fails. The value is not clamped.
        """
        mu = self.anchor(main_line, devig_cover_prob)
        if mu is None:
            return None
        t = favorite_threshold(target_line, yes_is_underdog)
        p_fav = self.conditional_survival(mu, t, self.bucket(abs(float(main_line))))
        return 1.0 - p_fav if yes_is_underdog else p_fav


# ----------------------------------------------------------------------
# Artifact loading (one read per sector per process; failures logged once)
# ----------------------------------------------------------------------

_PMF_CACHE: dict[str, Optional[MarginPMF]] = {}


def load_margin_pmf(sector: str) -> Optional[MarginPMF]:
    """Return the fitted MarginPMF for ``sector``, or None if unavailable.

    The result (including None) is cached for the life of the process, so a
    missing or invalid artifact logs one warning, not one per rung. Call
    ``clear_margin_pmf_cache()`` after replacing the artifact in-process.
    """
    key = (sector or "").lower()
    if key in _PMF_CACHE:
        return _PMF_CACHE[key]
    path = artifact_path(key)
    pmf: Optional[MarginPMF] = None
    try:
        art = json.loads(path.read_text())
        pmf = MarginPMF.from_artifact(art)
        if pmf.sector != key:
            raise MarginPMFArtifactError(f"artifact sector {pmf.sector!r} != {key!r}")
    except FileNotFoundError:
        pmf = None
        logger.warning("spread_pmf_artifact_missing", sector=key, path=str(path),
                       fallback="normal_cdf")
    except (OSError, ValueError) as exc:  # JSONDecodeError / MarginPMFArtifactError / bad bytes
        pmf = None
        logger.warning("spread_pmf_artifact_invalid", sector=key, path=str(path),
                       error=str(exc), fallback="normal_cdf")
    _PMF_CACHE[key] = pmf
    return pmf


def clear_margin_pmf_cache() -> None:
    """Forget loaded artifacts (tests, or after refitting in-process)."""
    _PMF_CACHE.clear()
