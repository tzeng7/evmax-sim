"""The Projections-tab sector catalog: ``data/projections.yaml`` bound to engines.

The YAML decides which sectors the tab shows, in which order, and which
engine runs each one; per-sector ``defaults`` override an engine's option
defaults. ``ENGINES`` maps an engine key to its implementation. Engines are
singletons because the NFL engine caches its last week projection between a
slate run and the per-game runs that follow it.

``load_catalog`` validates eagerly and raises ``CatalogError`` on an unknown
engine, a missing engine on an available sector, or a default for an option
the engine does not declare (or that fails the option's own validation).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Optional

import yaml

from evmax.projections.base import OptionSpec, ProjectionEngine, ProjectionError, resolve_options

Status = Literal["available", "planned"]
STATUSES: frozenset[str] = frozenset({"available", "planned"})

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "projections.yaml"


def _nfl_engine() -> ProjectionEngine:
    from evmax.projections.nfl import NflProjectionEngine

    return NflProjectionEngine()


def _point_engine() -> ProjectionEngine:
    from evmax.projections.point import PointProjectionEngine

    return PointProjectionEngine()


# Engine key (the YAML ``engine`` field) -> factory. Add an entry to wire a new engine.
ENGINES: dict[str, Callable[[], ProjectionEngine]] = {
    "nfl_projections": _nfl_engine,
    "point_projection": _point_engine,
}

_instances: dict[str, ProjectionEngine] = {}
_instances_lock = threading.Lock()


def engine_instance(name: str) -> ProjectionEngine:
    """The process-wide instance of engine ``name``."""
    with _instances_lock:
        if name not in _instances:
            _instances[name] = ENGINES[name]()
        return _instances[name]


class CatalogError(ValueError):
    """``data/projections.yaml`` is malformed or disagrees with the engines."""


class UnknownSector(KeyError):
    """The sector is not in the catalog."""


class SectorUnavailable(ProjectionError):
    """The sector is in the catalog but cannot run (planned, or lacks the capability)."""


@dataclass(frozen=True)
class ProjectionSector:
    key: str
    label: str
    status: Status
    engine: Optional[str] = None
    description: str = ""
    note: str = ""
    defaults: dict = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.status == "available"

    def engine_obj(self) -> ProjectionEngine:
        if not self.available or self.engine is None:
            raise SectorUnavailable(f"{self.label} projections are not available yet"
                                    + (f": {self.note}" if self.note else ""))
        return engine_instance(self.engine)

    def slate_options(self) -> list[OptionSpec]:
        return self._with_defaults(self.engine_obj().slate_options(self.key))

    def game_options(self) -> list[OptionSpec]:
        return self._with_defaults(self.engine_obj().game_options(self.key))

    def _with_defaults(self, specs: list[OptionSpec]) -> list[OptionSpec]:
        return [s.with_default(self.defaults[s.key]) if s.key in self.defaults else s for s in specs]

    def to_dict(self) -> dict:
        d = {"key": self.key, "label": self.label, "status": self.status, "engine": self.engine,
             "description": self.description, "note": self.note}
        if self.available:
            eng = self.engine_obj()
            d.update({
                "slate_options": [s.to_dict() for s in self.slate_options()],
                "game_options": [s.to_dict() for s in self.game_options()],
                "capabilities": {"stored": eng.supports_stored, "game_run": eng.supports_game_run},
                "game_run_label": eng.game_run_label,
            })
        return d


@dataclass(frozen=True)
class Catalog:
    sectors: tuple[ProjectionSector, ...]

    def get(self, key: str) -> ProjectionSector:
        for s in self.sectors:
            if s.key == key:
                return s
        raise UnknownSector(key)

    def to_dicts(self) -> list[dict]:
        return [s.to_dict() for s in self.sectors]


def _parse_sector(key: str, raw: dict) -> ProjectionSector:
    if not isinstance(raw, dict):
        raise CatalogError(f"{key}: entry must be a mapping")
    unknown = set(raw) - {"label", "status", "engine", "description", "note", "defaults"}
    if unknown:
        raise CatalogError(f"{key}: unknown field(s) {sorted(unknown)}")
    status = raw.get("status", "available")
    if status not in STATUSES:
        raise CatalogError(f"{key}: status must be one of {sorted(STATUSES)}, got {status!r}")
    engine = raw.get("engine")
    if engine is not None and engine not in ENGINES:
        raise CatalogError(f"{key}: unknown engine {engine!r} (known: {sorted(ENGINES)})")
    if status == "available" and engine is None:
        raise CatalogError(f"{key}: an available sector needs an engine")
    defaults = raw.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise CatalogError(f"{key}: defaults must be a mapping")
    if defaults and engine is None:
        raise CatalogError(f"{key}: defaults need an engine")
    sector = ProjectionSector(
        key=key, label=str(raw.get("label") or key.upper()), status=status, engine=engine,
        description=str(raw.get("description") or "").strip(), note=str(raw.get("note") or "").strip(),
        defaults=dict(defaults),
    )
    if defaults:
        eng = engine_instance(engine)
        specs = {s.key: s for s in eng.slate_options(key) + eng.game_options(key)}
        for opt, value in defaults.items():
            if opt not in specs:
                raise CatalogError(f"{key}: default for unknown option {opt!r} (engine {engine} "
                                   f"declares {sorted(specs)})")
            try:
                specs[opt].coerce(value)
            except ProjectionError as e:
                raise CatalogError(f"{key}: bad default for {opt!r}: {e}") from None
    return sector


def load_catalog(path: Optional[Path] = None) -> Catalog:
    """Parse and validate the catalog YAML."""
    p = Path(path) if path is not None else DEFAULT_PATH
    raw = yaml.safe_load(p.read_text()) or {}
    sectors = raw.get("sectors")
    if not isinstance(sectors, dict) or not sectors:
        raise CatalogError(f"{p}: needs a non-empty 'sectors' mapping")
    return Catalog(tuple(_parse_sector(str(k), v) for k, v in sectors.items()))


_catalog: Optional[Catalog] = None


def get_catalog() -> Catalog:
    """The catalog from the default path, loaded once per process."""
    global _catalog
    if _catalog is None:
        _catalog = load_catalog()
    return _catalog


def resolve_run_options(sector: ProjectionSector, kind: Literal["slate", "game"], raw: Optional[dict]) -> dict:
    """Validated options for a run, with the sector's YAML defaults applied."""
    specs = sector.slate_options() if kind == "slate" else sector.game_options()
    return resolve_options(specs, raw)
