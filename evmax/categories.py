"""Betting-category registry — the single source of truth.

Loads ``data/categories.yaml`` into typed ``CategorySpec`` instances and
exposes a small API used by the coordinator, CLI, and backtest code:

    from evmax.categories import (
        get_category, all_categories, categories_in_mode, validate_registry,
    )

    nba = get_category("nba")
    if nba.mode == "live":
        ...

The registry is loaded once at import time from ``data/categories.yaml``.
``validate_registry()`` runs immediately and raises on any integrity
problem — missing category for a key in ``SECTOR_SERIES_MAP``, unknown
model / resolver / mode / market_type, or prop-vs-game shape mismatches.
Fail-fast at import beats mysterious runtime behavior when someone adds
a new sector and forgets to register it.

The YAML's ``mode`` field is the base mode; ``evmax.modes.get_mode()``
layers environment variable and CLI overrides on top of that base.
Callers that care about "the live mode right now" should go through
``evmax.modes``, not read ``CategorySpec.mode`` directly.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import yaml

from evmax.models.market import MarketType

Mode = Literal["live", "shadow", "disabled"]
Status = Literal["shipped", "wip", "unresolved", "blocked"]

# Default YAML location relative to repo root.
_DEFAULT_YAML_PATH = Path(__file__).resolve().parents[1] / "data" / "categories.yaml"

# ---------------------------------------------------------------------------
# Known-values tables — edit these when adding a new model or resolver.
# ---------------------------------------------------------------------------

KNOWN_MODELS: set[str] = {
    # Shared game-level statistical agents (per evmax/agents/models/)
    "elo",
    "form",
    "poisson",
    "sharp",
    "pitcher",
    # NBA-specific advanced agents
    "efficiency",
    "shot_quality",
    "matchup",
    "possession_sim",
    # NFL-specific advanced agents
    "nfl_efficiency",
    "nfl_qb_elo",
    # NHL-specific advanced agents
    "nhl_xg",
    # WNBA-specific advanced agents (kept parallel to the NBA set so the
    # two leagues can tune independently without risk of cross-contamination)
    "wnba_efficiency",
    "wnba_possession_sim",
    # Tennis-specific agents
    "tennis_surface_elo",
    "tennis_serve_return",
    "tennis_h2h",
    "tennis_ranking_trend",
    "tennis_form",
    "tennis_advanced",
    # Prop caches
    "nba_props_cache",
    "nfl_props_cache_v1_qb_only",
}

KNOWN_RESOLVERS: set[str] = {
    "espn_scoreboard",
    "espn_boxscore",
    "bo3gg",
    "kalshi_settlement",
    "none",
}

_LEGAL_MODES: set[str] = {"live", "shadow", "disabled"}
_LEGAL_STATUSES: set[str] = {"shipped", "wip", "unresolved", "blocked"}


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CategorySpec:
    """One betting category — game sector or prop sector."""

    key: str
    display_name: str
    market_types: tuple[MarketType, ...]
    models: tuple[str, ...]
    mode: Mode
    resolver: str
    status: Status
    prop_stat_types: tuple[str, ...] = field(default_factory=tuple)
    notes: Optional[str] = None
    # Market types within this category that should be forced to 'shadow' even
    # when the sector mode is 'live'. Lets us promote a category for some bet
    # types (ML, spread) while keeping others under validation (totals). Empty
    # = sector mode applies to every market type. Validated against the
    # category's market_types list at parse time.
    shadow_market_types: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_prop(self) -> bool:
        return self.market_types == (MarketType.player_prop,)


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


def _coerce_market_types(values: list[str], key: str) -> tuple[MarketType, ...]:
    out: list[MarketType] = []
    for v in values:
        try:
            out.append(MarketType(v))
        except ValueError as e:
            raise ValueError(
                f"category {key!r} references unknown MarketType {v!r}. "
                f"Legal values: {[m.value for m in MarketType]}"
            ) from e
    return tuple(out)


def _parse_entry(key: str, raw: dict) -> CategorySpec:
    required = {"display_name", "market_types", "models", "mode", "resolver", "status"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"category {key!r} missing required fields: {sorted(missing)}")

    display_name = raw["display_name"]
    if not isinstance(display_name, str) or not display_name:
        raise ValueError(f"category {key!r}: display_name must be a non-empty string")

    market_types = _coerce_market_types(list(raw["market_types"]), key)
    if not market_types:
        raise ValueError(f"category {key!r}: market_types must be non-empty")

    models = tuple(raw["models"])
    if not all(isinstance(m, str) and m for m in models):
        raise ValueError(f"category {key!r}: models must be a list of non-empty strings")

    mode = raw["mode"]
    if mode not in _LEGAL_MODES:
        raise ValueError(
            f"category {key!r}: illegal mode {mode!r}. Legal: {sorted(_LEGAL_MODES)}"
        )

    resolver = raw["resolver"]
    if not isinstance(resolver, str) or not resolver:
        raise ValueError(f"category {key!r}: resolver must be a non-empty string")

    status = raw["status"]
    if status not in _LEGAL_STATUSES:
        raise ValueError(
            f"category {key!r}: illegal status {status!r}. Legal: {sorted(_LEGAL_STATUSES)}"
        )

    prop_stat_types = tuple(raw.get("prop_stat_types") or ())
    notes = raw.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise ValueError(f"category {key!r}: notes must be null or a string")

    raw_shadow_mts = raw.get("shadow_market_types") or ()
    shadow_market_types = tuple(raw_shadow_mts)
    if shadow_market_types:
        legal = {mt.value for mt in market_types}
        for mt in shadow_market_types:
            if mt not in legal:
                raise ValueError(
                    f"category {key!r}: shadow_market_types entry {mt!r} is "
                    f"not in this category's market_types {sorted(legal)}"
                )

    return CategorySpec(
        key=key,
        display_name=display_name,
        market_types=market_types,
        models=models,
        mode=mode,  # type: ignore[arg-type]
        resolver=resolver,
        status=status,  # type: ignore[arg-type]
        prop_stat_types=prop_stat_types,
        notes=notes,
        shadow_market_types=shadow_market_types,
    )


def _load_yaml(path: Path) -> dict[str, CategorySpec]:
    if not path.exists():
        raise FileNotFoundError(f"category registry not found at {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must be a YAML mapping of category → fields")
    return {key: _parse_entry(key, value) for key, value in raw.items()}


# ---------------------------------------------------------------------------
# Module-level registry (lazy, thread-safe)
# ---------------------------------------------------------------------------

_REGISTRY: Optional[dict[str, CategorySpec]] = None
_LOCK = threading.Lock()


def _registry(reload: bool = False) -> dict[str, CategorySpec]:
    global _REGISTRY
    if _REGISTRY is None or reload:
        with _LOCK:
            if _REGISTRY is None or reload:
                _REGISTRY = _load_yaml(_DEFAULT_YAML_PATH)
    return _REGISTRY


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def reload_registry(path: Optional[Path] = None) -> None:
    """Re-read the YAML from disk. Used by tests and the CLI validate command."""
    global _REGISTRY
    with _LOCK:
        _REGISTRY = _load_yaml(path or _DEFAULT_YAML_PATH)


def get_category(key: str) -> CategorySpec:
    """Return the CategorySpec for `key`, raising KeyError if unknown."""
    reg = _registry()
    if key not in reg:
        raise KeyError(
            f"unknown category {key!r}. Known: {sorted(reg.keys())}"
        )
    return reg[key]


def all_categories() -> list[CategorySpec]:
    """Return every registered category, sorted by key."""
    return [v for _, v in sorted(_registry().items())]


def categories_in_mode(mode: Mode) -> list[str]:
    """Return the list of category keys currently in the given mode.

    Note: this reads only the YAML's base mode. Runtime overrides from
    env vars / CLI flags do not apply here — use ``evmax.modes`` for the
    effective mode at scan time.
    """
    if mode not in _LEGAL_MODES:
        raise ValueError(f"illegal mode {mode!r}. Legal: {sorted(_LEGAL_MODES)}")
    return sorted(k for k, spec in _registry().items() if spec.mode == mode)


def validate_registry() -> None:
    """Cross-check the registry against the rest of the codebase.

    Raises ValueError on any failure. Checks:
      1. Every key in SECTOR_SERIES_MAP has a registry entry.
      2. Every registry key is in SECTOR_SERIES_MAP (no orphan categories).
      3. Every referenced model name is in KNOWN_MODELS.
      4. Every resolver value is in KNOWN_RESOLVERS.
      5. Prop categories have exactly [MarketType.player_prop] and
         non-empty prop_stat_types; game categories have no prop_stat_types.

    Called at import time via ``_eager_validate`` below so the whole app
    fails fast if the catalog is broken.
    """
    from evmax.clients.kalshi import SECTOR_SERIES_MAP

    reg = _registry()
    series_keys = set(SECTOR_SERIES_MAP.keys())
    reg_keys = set(reg.keys())

    missing_from_registry = series_keys - reg_keys
    if missing_from_registry:
        raise ValueError(
            f"categories registered in SECTOR_SERIES_MAP but missing from "
            f"data/categories.yaml: {sorted(missing_from_registry)}"
        )
    orphan = reg_keys - series_keys
    if orphan:
        raise ValueError(
            f"categories in data/categories.yaml but missing from "
            f"SECTOR_SERIES_MAP: {sorted(orphan)}"
        )

    for key, spec in reg.items():
        unknown_models = set(spec.models) - KNOWN_MODELS
        if unknown_models:
            raise ValueError(
                f"category {key!r} references unknown models: "
                f"{sorted(unknown_models)}. Known: {sorted(KNOWN_MODELS)}"
            )
        if spec.resolver not in KNOWN_RESOLVERS:
            raise ValueError(
                f"category {key!r} references unknown resolver "
                f"{spec.resolver!r}. Known: {sorted(KNOWN_RESOLVERS)}"
            )
        if spec.is_prop:
            if not spec.prop_stat_types:
                raise ValueError(
                    f"category {key!r} is a prop category (market_types == "
                    f"[player_prop]) but has empty prop_stat_types"
                )
        else:
            if spec.prop_stat_types:
                raise ValueError(
                    f"category {key!r} is a game category but has "
                    f"prop_stat_types set — only prop categories should populate this"
                )
            if MarketType.player_prop in spec.market_types:
                raise ValueError(
                    f"category {key!r} has player_prop mixed with game market "
                    f"types; split into a separate prop category"
                )


# Fail-fast on import: the whole app should refuse to start if the catalog
# is inconsistent. Tests can patch _DEFAULT_YAML_PATH before import if needed.
validate_registry()
