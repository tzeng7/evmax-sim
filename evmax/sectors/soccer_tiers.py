"""Soccer league tier lookup — maps Kalshi ticker → sharp_weight override.

The walk-forward backtest showed that in top-5 European leagues Pinnacle's
closing Brier is the informational ceiling. Mixing our stat models at the
default sharp_weight=0.40 drags the blend 0.003–0.007 Brier worse than pure
sharp. This module reads data/soccer_league_tiers.yaml and returns the
per-league sharp_weight the coordinator should apply.

Tickers are matched by their Kalshi series prefix (the portion before the
first `-`), which identifies the league unambiguously.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "data" / "soccer_league_tiers.yaml"


@lru_cache(maxsize=1)
def _load_tiers() -> dict:
    """Read and cache the tier config. Raises if the file is missing or
    malformed — there's no sensible silent default for a pricing decision."""
    if not _CONFIG_PATH.exists():
        return {"default_sharp_weight": 0.40, "tiers": {}}
    with _CONFIG_PATH.open("r") as f:
        data = yaml.safe_load(f) or {}
    return data


@lru_cache(maxsize=1)
def _series_to_weight() -> dict[str, float]:
    """Flatten tiers into series_prefix → sharp_weight for fast lookup."""
    cfg = _load_tiers()
    mapping: dict[str, float] = {}
    for tier_name, tier in (cfg.get("tiers") or {}).items():
        weight = float(tier.get("sharp_weight", cfg.get("default_sharp_weight", 0.40)))
        for series in tier.get("kalshi_series") or []:
            mapping[series.upper()] = weight
    return mapping


def default_sharp_weight() -> float:
    """Fallback weight used when a ticker doesn't map to any tier."""
    return float(_load_tiers().get("default_sharp_weight", 0.40))


def sharp_weight_for_ticker(ticker: Optional[str]) -> float:
    """Return the sharp_weight for a Kalshi ticker's series prefix.

    Tickers look like `KXEPLGAME-26APR24MANCHE-CHE`. The series prefix is
    everything before the first `-`. Returns `default_sharp_weight()` when
    the ticker is empty or the series isn't configured.
    """
    if not ticker:
        return default_sharp_weight()
    # Strip any leading "kalshi:" source prefix defensively.
    t = ticker.split(":", 1)[-1] if ":" in ticker else ticker
    series = t.split("-", 1)[0].upper()
    return _series_to_weight().get(series, default_sharp_weight())


def reset_cache() -> None:
    """Clear caches. Useful in tests when the YAML is monkey-patched."""
    _load_tiers.cache_clear()
    _series_to_weight.cache_clear()
