"""Shared numeric primitives for model agents.

`models_ml` is a dependency leaf (it imports only `evmax.models.*` + scipy),
so model agents can pull from here without risking an import cycle through the
heavier `evmax.agents.models` package.
"""

from __future__ import annotations

import math


def normal_cdf(x: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun).

    This hand-rolled approximation is kept (rather than scipy.stats.norm.cdf)
    so the lightweight model agents don't pull in scipy at import time. The
    coefficients and ±6 tail clamps match the six historical inline copies
    this function replaces — numerics are identical, not merely close.
    """
    if x < -6:
        return 0.0
    if x > 6:
        return 1.0
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = 0.3989422804014327  # 1/sqrt(2π)
    p = d * math.exp(-x * x / 2.0) * t * (
        0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.8212560 + t * 1.3302744)))
    )
    return 1.0 - p if x > 0 else p
