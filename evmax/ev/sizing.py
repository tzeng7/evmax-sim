"""Single entry point for stake sizing — the choke point every caller routes through.

Before this module, stake was sized by calling :func:`evmax.ev.kelly.compute_kelly`
directly at seven sites (five in the EV agent, the pruner, the CLI pick path). They
had already drifted — the pruner passed ``spread_pct=0`` so a play sized with a
liquidity discount at scan time was re-sized without it at prune time. This module
is the one place that composes the four sizing layers so they can never disagree
again:

  1. **Edge shrinkage** (Phase 1) — replaces the dead confidence discount. The
     probability fed to Kelly is not the raw blended model probability; it is the
     out-of-sample-calibrated ``P(win | blended, price)`` from a per-sector two-input
     logistic (:class:`ShrinkageModel`). This corrects the selection bias in the tail:
     the rows that earn the biggest stakes are the ones where the model disagrees most
     with the market, which is exactly where estimation error concentrates (realized /
     predicted edge ≈ 0.61 in the top tercile on live data). Sizing only — the EV gate
     upstream still uses the raw blended probability, so the play list does not change.

  2. **Fractional Kelly** — unchanged; the ``base_fraction`` multiplier (half Kelly by
     CLI default) is the standard response to parameter uncertainty.

  3. **Liquidity discount** (Phase 3) — when top-of-book dollar depth is available, the
     discount is ``min(1, alpha · depth_usd / stake_usd)`` (fillability), not the
     spread-over-mid proxy. The proxy penalizes cheap contracts for the same absolute
     spread and ignores depth entirely; the depth form measures the thing the discount
     claims to measure. Falls back to the spread proxy when depth is absent.

  4. **Hard cap** — unchanged ``min(k, max_kelly)``. With shrinkage doing the tail
     work, the cap is a backstop against a wholly wrong probability (a mis-aligned YES
     side, a stale seed), not a growth-tuning device.

Every layer is **identity by default**. With a default :class:`SizingConfig` (all flags
off, no shrinkage model), :func:`size_position` is byte-identical to the old
``compute_kelly`` call. Each phase's behaviour is gated behind a settings flag and only
turns on after the sizing replay harness (:mod:`evmax.backtest.sizing`) validates it
out-of-sample.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from evmax.ev.kelly import KellyResult, compute_kelly

# Shipped shrinkage coefficients live here (fit by scripts/fit_edge_shrinkage.py,
# validated walk-forward by scripts/backtest_sizing.py). Absent file → identity.
SHRINKAGE_STATE_PATH = Path(__file__).resolve().parents[2] / "data" / "models" / "edge_shrinkage_state.json"

# Clip probabilities away from 0/1 before the logit so the transform stays finite.
_P_EPS = 1e-4
# Minimum sectorwise sample size to trust a per-sector fit; below this the pooled
# fit is used. Mirrors the 300-row threshold from the sizing scope analysis.
MIN_SECTOR_FIT_N = 300


def _logit(p: float) -> float:
    p = min(max(p, _P_EPS), 1.0 - _P_EPS)
    return math.log(p / (1.0 - p))


def _as_float(v, default: float) -> float:
    """Coerce a settings value to float, falling back on a non-numeric (e.g. a mock)."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


@dataclass(frozen=True)
class ShrinkageCoeffs:
    """Two-input logistic ``P(win) = sigmoid(a + s_b·logit(blended) + s_p·logit(price))``.

    ``s_b`` is the weight the fit places on the model's own probability; ``s_p`` the
    weight on the market price. A calibrated, selection-free model would give
    ``s_b≈1, s_p≈0, a≈0`` (identity). In practice the fit pulls ``s_b`` below 1 and
    ``s_p`` above 0, shrinking the sized probability toward the price on exactly the
    disagreement rows that carry the most estimation error.
    """

    a: float
    s_b: float
    s_p: float
    n: int = 0

    def predict(self, blended: float, price: float) -> float:
        return _sigmoid(self.a + self.s_b * _logit(blended) + self.s_p * _logit(price))


@dataclass
class ShrinkageModel:
    """Per-sector + pooled shrinkage coefficients loaded from the state file."""

    pooled: Optional[ShrinkageCoeffs] = None
    sectors: dict[str, ShrinkageCoeffs] = field(default_factory=dict)

    def coeffs_for(self, sector: Optional[str]) -> Optional[ShrinkageCoeffs]:
        """Per-sector fit when it exists and is well-powered, else the pooled fit."""
        key = (sector or "").lower()
        c = self.sectors.get(key)
        if c is not None and c.n >= MIN_SECTOR_FIT_N:
            return c
        return self.pooled

    def sizing_probability(self, blended: float, price: float, sector: Optional[str]) -> float:
        """Calibrated ``P(win)`` to size on; identity when no fit covers the sector."""
        c = self.coeffs_for(sector)
        if c is None:
            return blended
        return c.predict(blended, price)


def load_shrinkage_model(path: Path | str | None = None) -> Optional[ShrinkageModel]:
    """Load the shipped shrinkage coefficients. Returns None when the file is absent."""
    p = Path(path) if path is not None else SHRINKAGE_STATE_PATH
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    def _mk(d: dict) -> Optional[ShrinkageCoeffs]:
        try:
            return ShrinkageCoeffs(
                a=float(d["a"]), s_b=float(d["s_b"]), s_p=float(d["s_p"]), n=int(d.get("n", 0))
            )
        except (KeyError, TypeError, ValueError):
            return None

    pooled = _mk(raw["pooled"]) if isinstance(raw.get("pooled"), dict) else None
    sectors: dict[str, ShrinkageCoeffs] = {}
    for sec, d in (raw.get("sectors") or {}).items():
        c = _mk(d) if isinstance(d, dict) else None
        if c is not None:
            sectors[sec.lower()] = c
    if pooled is None and not sectors:
        return None
    return ShrinkageModel(pooled=pooled, sectors=sectors)


@dataclass
class SizingConfig:
    """Which sizing layers are active. Default = every layer off (identity)."""

    edge_shrinkage_enabled: bool = False
    liquidity_depth_enabled: bool = False
    depth_alpha: float = 1.0
    depth_floor: float = 0.0
    shrinkage_model: Optional[ShrinkageModel] = None

    @classmethod
    def from_settings(cls, settings, *, model: Optional[ShrinkageModel] = None) -> "SizingConfig":
        """Build from a Settings object, loading the shrinkage model when enabled.

        Defensive by design: a flag is only ON when the attribute is literally
        ``True`` (so a test's bare ``MagicMock`` settings reads as every layer off,
        not accidentally on), and float knobs coerce with a safe fallback.
        """
        shrink_on = getattr(settings, "edge_shrinkage_enabled", False) is True
        depth_on = getattr(settings, "liquidity_depth_enabled", False) is True
        sm = model
        if shrink_on and sm is None:
            sm = load_shrinkage_model()
        return cls(
            edge_shrinkage_enabled=shrink_on,
            liquidity_depth_enabled=depth_on,
            depth_alpha=_as_float(getattr(settings, "liquidity_depth_alpha", 1.0), 1.0),
            depth_floor=_as_float(getattr(settings, "liquidity_depth_floor", 0.0), 0.0),
            shrinkage_model=sm,
        )


def depth_liquidity_discount(
    depth_usd: Optional[float],
    stake_usd: Optional[float],
    *,
    alpha: float = 1.0,
    floor: float = 0.0,
) -> Optional[float]:
    """Fillability discount ``min(1, alpha · depth / stake)``, floored.

    Returns None when depth or a positive stake is unavailable, so the caller
    falls back to the spread-over-mid proxy. A tiny/zero stake needs no discount
    (returns 1.0).
    """
    if depth_usd is None or stake_usd is None or stake_usd <= 0:
        return None
    if depth_usd <= 0:
        return floor
    return max(floor, min(1.0, alpha * depth_usd / stake_usd))


def edge_is_quarantined(
    blended: float,
    sharp: Optional[float],
    price: Optional[float],
    pp: float,
) -> bool:
    """Phase 2 safety gate: True when the blended prob grossly disagrees with BOTH anchors.

    A legitimate edge disagrees with the *price* (that is the edge) but stays close to the
    *sharp* devigged probability (the model and the sharp book roughly agree, the soft
    venue is stale). A row that is more than ``pp`` probability points from BOTH the sharp
    prob and the price is not an edge — it is a wrong probability (mis-aligned YES side,
    stale seed, parse error), the failure mode the hard cap contains bluntly. Quarantining
    it demotes it to shadow before it can be bankroll-sized.

    ``pp <= 0`` disables the gate (returns False). A missing sharp anchor also returns
    False — with only one anchor there is nothing to cross-check against, and the EV gate
    plus the hard cap remain in force.
    """
    pp = _as_float(pp, 0.0)
    if pp <= 0 or sharp is None or price is None:
        return False
    return abs(blended - sharp) > pp and abs(blended - price) > pp


def size_position(
    *,
    true_prob: float,
    payout_decimal: float,
    edge_pct: float,
    sector: Optional[str] = None,
    price: Optional[float] = None,
    base_fraction: float = 0.25,
    max_kelly: float = 0.05,
    min_kelly: float = 0.0,
    spread_pct: float = 0.0,
    depth_usd: Optional[float] = None,
    bankroll: Optional[float] = None,
    config: Optional[SizingConfig] = None,
) -> KellyResult:
    """Compose the four sizing layers into one ``KellyResult``.

    With ``config=None`` (or a default :class:`SizingConfig`) this reduces exactly to
    ``compute_kelly(true_prob, payout_decimal, edge_pct, spread_pct, base_fraction,
    max_kelly, min_kelly)`` — the pre-Phase-1 behaviour, byte for byte.

    ``true_prob`` is the raw blended model probability (the EV gate already used it).
    ``price`` is the venue's raw YES ask for the side being sized, needed by the
    shrinkage logistic; when it is None, shrinkage is skipped (identity) even if enabled.
    ``depth_usd`` is top-of-book fillable dollars; ``bankroll`` lets the depth discount
    convert a Kelly fraction to a dollar stake.
    """
    cfg = config or SizingConfig()

    # Layer 1 — edge shrinkage. Size on the calibrated P(win|blended,price).
    size_prob = true_prob
    if (
        cfg.edge_shrinkage_enabled
        and cfg.shrinkage_model is not None
        and price is not None
        and 0.0 < price < 1.0
    ):
        size_prob = cfg.shrinkage_model.sizing_probability(true_prob, price, sector)

    # Layers 2 + 4 — fractional Kelly + hard cap. When the depth-liquidity layer is
    # active we compute the pre-liquidity fraction with the spread proxy disabled
    # (spread_pct=0), then apply the depth discount below; otherwise the spread proxy
    # runs inside compute_kelly exactly as before.
    depth_active = cfg.liquidity_depth_enabled and depth_usd is not None and bankroll
    kelly = compute_kelly(
        true_prob=size_prob,
        payout_decimal=payout_decimal,
        edge_pct=edge_pct,
        spread_pct=0.0 if depth_active else spread_pct,
        base_fraction=base_fraction,
        max_kelly=max_kelly,
        min_kelly=min_kelly,
    )

    # Layer 3 — depth-keyed liquidity discount (fillability). Uses the capped
    # pre-discount fraction to estimate the stake, one fixed-point step.
    if depth_active and kelly.kelly_fraction > 0:
        provisional_stake = bankroll * kelly.kelly_fraction
        disc = depth_liquidity_discount(
            depth_usd, provisional_stake, alpha=cfg.depth_alpha, floor=cfg.depth_floor
        )
        if disc is not None and disc < 1.0:
            new_frac = max(min_kelly, kelly.kelly_fraction * disc)
            kelly = KellyResult(
                kelly_full=kelly.kelly_full,
                kelly_fraction=new_frac,
                confidence_discount=kelly.confidence_discount,
                liquidity_discount=disc,
                suggested_units=new_frac,
            )

    return kelly
