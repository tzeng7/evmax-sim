"""Category mode lookup — live / shadow / disabled.

Reads the base mode from ``evmax.categories`` (the YAML registry) and
layers two override sources on top:

  1. Environment variable ``EVMAX_CATEGORY_MODES`` — a JSON mapping
     ``{"nfl_props": "shadow", "nhl": "disabled"}``. Useful for CI,
     test runs, and wrapper scripts that want to flip a category
     without touching the YAML.

  2. Runtime overrides passed explicitly by the CLI via
     ``set_runtime_overrides({"nba": "disabled"})``. These are used
     when the user passes ``--shadow X --live Y --disabled Z`` on
     one scan command.

Override precedence (highest wins):  runtime  >  env var  >  YAML base

Callers that need "the mode for this category right now" should always
call ``get_mode(key)`` — never read ``CategorySpec.mode`` directly, or
they'll miss the overrides.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Literal, Mapping, Optional

from evmax.categories import Mode, get_category

_ENV_VAR = "EVMAX_CATEGORY_MODES"

_runtime_overrides: dict[str, Mode] = {}
_env_cache: Optional[dict[str, Mode]] = None
_LOCK = threading.Lock()

_LEGAL_MODES: set[str] = {"live", "shadow", "disabled"}


def _load_env_overrides() -> dict[str, Mode]:
    """Parse EVMAX_CATEGORY_MODES once and cache. Explicit reset via
    ``reset_overrides_cache()`` for tests."""
    global _env_cache
    if _env_cache is not None:
        return _env_cache
    raw = os.environ.get(_ENV_VAR, "").strip()
    if not raw:
        _env_cache = {}
        return _env_cache
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"{_ENV_VAR} is not valid JSON: {raw!r} ({e})"
        ) from e
    if not isinstance(parsed, dict):
        raise ValueError(f"{_ENV_VAR} must be a JSON object, got {type(parsed).__name__}")
    out: dict[str, Mode] = {}
    for k, v in parsed.items():
        if v not in _LEGAL_MODES:
            raise ValueError(
                f"{_ENV_VAR}: category {k!r} has illegal mode {v!r}. "
                f"Legal: {sorted(_LEGAL_MODES)}"
            )
        out[k] = v  # type: ignore[assignment]
    _env_cache = out
    return out


def reset_overrides_cache() -> None:
    """Forget the env-var cache. Used by tests that monkeypatch environ."""
    global _env_cache
    with _LOCK:
        _env_cache = None


def set_runtime_overrides(overrides: Mapping[str, str]) -> None:
    """Install CLI-level overrides that take precedence over env + YAML.

    Validates each mode string. Call with an empty dict to clear.
    """
    global _runtime_overrides
    validated: dict[str, Mode] = {}
    for k, v in overrides.items():
        if v not in _LEGAL_MODES:
            raise ValueError(
                f"runtime override: category {k!r} has illegal mode {v!r}. "
                f"Legal: {sorted(_LEGAL_MODES)}"
            )
        validated[k] = v  # type: ignore[assignment]
    with _LOCK:
        _runtime_overrides = validated


def clear_runtime_overrides() -> None:
    set_runtime_overrides({})


def get_mode(category: str, market_type: Optional[str] = None) -> Mode:
    """Return the effective mode for `category` — runtime > env > YAML.

    If `market_type` is provided and the category's YAML spec lists it in
    ``shadow_market_types``, the result is downgraded to ``shadow`` even when
    the sector-level mode is ``live``. This lets us keep one or two market
    types under validation while the rest of the category goes live.

    A `market_type` listed in ``disabled_market_types`` resolves to
    ``disabled`` regardless of the sector-level mode (live OR shadow) —
    used to stop persisting a measured-negative market type (e.g. baseball
    totals) without disabling the whole category.

    Runtime + env overrides take precedence — they apply uniformly to all
    market types within a category (no per-market-type override there).
    """
    if category in _runtime_overrides:
        return _runtime_overrides[category]
    env = _load_env_overrides()
    if category in env:
        return env[category]
    spec = get_category(category)
    if market_type and market_type in spec.disabled_market_types:
        return "disabled"
    if (
        spec.mode == "live"
        and market_type
        and market_type in spec.shadow_market_types
    ):
        return "shadow"
    return spec.mode


def is_live(category: str) -> bool:
    return get_mode(category) == "live"


def is_shadow(category: str) -> bool:
    return get_mode(category) == "shadow"


def is_disabled(category: str) -> bool:
    return get_mode(category) == "disabled"


def effective_modes() -> dict[str, Mode]:
    """Return {category: mode} for every registered category, with all
    overrides applied. Used by the CLI ``evmax categories list`` and by
    debugging output.
    """
    from evmax.categories import all_categories

    out: dict[str, Mode] = {}
    for spec in all_categories():
        out[spec.key] = get_mode(spec.key)
    return out
