"""Engine contract for the dashboard Projections tab.

An engine adapts one projection model to JSON that the generic UI renders
without knowing the sector:

* Options. ``slate_options`` / ``game_options`` return ``OptionSpec`` lists.
  The UI renders one input per spec and posts the values back;
  ``resolve_options`` coerces and validates them.
* Slate. ``run_slate`` (and ``stored``, for engines that persist runs)
  returns one row per game plus optional player rows. Player columns are
  described by ``Column`` specs, so a new sector needs no frontend change.
* Game. ``run_game`` runs the deeper model for one game and returns
  render-ready sections (``kv``, ``players``, ``table``).

Line convention: every ``*_margin`` field is the expected HOME margin
(home − away), positive when the home team is favored.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

OptionType = Literal["int", "bool"]
ColumnKind = Literal["range", "prob"]


class ProjectionError(ValueError):
    """A run cannot proceed for a reason the user can fix (bad option, no games that week)."""


@dataclass(frozen=True)
class OptionSpec:
    """One configurable input of a run. ``nullable`` ints accept None ("auto")."""

    key: str
    label: str
    type: OptionType
    default: Any = None
    help: str = ""
    min: Optional[int] = None
    max: Optional[int] = None
    step: Optional[int] = None
    nullable: bool = False
    placeholder: str = ""

    def with_default(self, value: Any) -> "OptionSpec":
        return replace(self, default=self.coerce(value))

    def coerce(self, raw: Any) -> Any:
        """The validated value for ``raw``; None/"" means the default (or None when nullable)."""
        if raw is None or raw == "":
            if self.nullable:
                return None
            if self.default is None:
                raise ProjectionError(f"{self.label} is required")
            return self.default
        if self.type == "bool":
            if not isinstance(raw, bool):
                raise ProjectionError(f"{self.label} must be true or false")
            return raw
        # int: reject bools (a JSON true is not 1) and non-integral numbers.
        if isinstance(raw, bool):
            raise ProjectionError(f"{self.label} must be a whole number")
        try:
            num = float(raw)
        except (TypeError, ValueError):
            raise ProjectionError(f"{self.label} must be a whole number") from None
        if not num.is_integer():
            raise ProjectionError(f"{self.label} must be a whole number")
        val = int(num)
        if self.min is not None and val < self.min:
            raise ProjectionError(f"{self.label} must be at least {self.min}")
        if self.max is not None and val > self.max:
            raise ProjectionError(f"{self.label} must be at most {self.max}")
        return val

    def to_dict(self) -> dict:
        """JSON for the UI: ``default`` is always present (null = auto); unset bounds are omitted."""
        return {k: v for k, v in asdict(self).items() if k == "default" or (v is not None and v != "")}


def resolve_options(specs: list[OptionSpec], raw: Optional[dict]) -> dict:
    """Coerce ``raw`` against ``specs``. Unknown keys are an error; missing keys take the default."""
    if raw is not None and not isinstance(raw, dict):
        raise ProjectionError("options must be an object")
    raw = dict(raw or {})
    known = {s.key for s in specs}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ProjectionError(f"unknown option(s): {', '.join(unknown)}")
    return {s.key: s.coerce(raw.get(s.key)) for s in specs}


@dataclass(frozen=True)
class Column:
    """A player-table column. ``range`` cells show value (lo–hi); ``prob`` cells show a percentage."""

    key: str
    label: str
    kind: ColumnKind = "range"
    title: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def range_cell(value: Any, lo: Any = None, hi: Any = None, actual: Any = None) -> Optional[dict]:
    """A ``range`` cell. ``result`` / ``hit`` report the actual against the lo–hi range once known."""
    value = jsonable(value)
    if value is None:
        return None
    cell = {"value": value, "lo": jsonable(lo), "hi": jsonable(hi)}
    actual = jsonable(actual)
    if actual is not None:
        cell["result"] = f"actual {actual:.0f}"
        if cell["lo"] is not None and cell["hi"] is not None:
            cell["hit"] = cell["lo"] <= actual <= cell["hi"]
    return cell


def jsonable(v: Any) -> Any:
    """JSON-safe scalar: numpy → Python, NaN/NaT → None, dates → ISO strings."""
    if v is None or v is pd.NaT:  # NaT subclasses datetime, so test it first
        return None
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if math.isnan(f) else f
    if isinstance(v, (datetime, date)):  # includes pd.Timestamp
        return v.isoformat()
    return v


def favorite_line(home: str, away: str, home_margin: Optional[float]) -> Optional[str]:
    """'BOS -4.5', 'PK', or None when there is no margin. Same convention as the NFL store."""
    from evmax.nfl_projections.store import favorite_line as _fav

    if jsonable(home_margin) is None:
        return None
    return _fav(home, away, home_margin)


def game_row(*, game_id: str, home: str, away: str, home_name: str, away_name: str,
             proj_home: float, proj_away: float, p_home_win: float,
             kickoff: Optional[str] = None, game_date: Optional[str] = None, neutral: bool = False,
             market_home_margin: Optional[float] = None, market_total: Optional[float] = None,
             actual_home: Optional[float] = None, actual_away: Optional[float] = None,
             subtitle: Optional[str] = None, flags: Optional[list[str]] = None,
             context: Optional[dict] = None, pick: Optional[dict] = None) -> dict:
    """One game in the shared row shape (``home`` / ``away`` are short labels for score cells).

    ``pick`` is the engine's model pick against the market line, for engines that make
    one (shape: ``pick_payload``); the UI shows it as the row's Outcome.
    """
    proj_home, proj_away = float(proj_home), float(proj_away)
    margin = proj_home - proj_away
    market_home_margin = jsonable(market_home_margin)
    return {
        "game_id": game_id, "kickoff": kickoff, "date": game_date, "neutral": bool(neutral),
        "home": home, "away": away, "home_name": home_name, "away_name": away_name,
        "proj_home": proj_home, "proj_away": proj_away, "proj_margin": margin,
        "proj_total": proj_home + proj_away, "p_home_win": float(p_home_win),
        "model_line": favorite_line(home, away, margin),
        "market_home_margin": market_home_margin,
        "market_line": favorite_line(home, away, market_home_margin),
        "market_total": jsonable(market_total),
        "actual_home": jsonable(actual_home), "actual_away": jsonable(actual_away),
        "subtitle": subtitle, "flags": flags or [], "context": context or {}, "pick": pick,
    }


def pick_payload(*, spread: Optional[str], spread_edge: Any, total: Optional[str], total_edge: Any,
                 model_total: Any, line: Optional[str], recorded: bool,
                 spread_result: Optional[str] = None, spread_result_close: Optional[str] = None,
                 total_result: Optional[str] = None, total_result_close: Optional[str] = None) -> Optional[dict]:
    """A game row's model pick: the spread side and O/U the model prefers vs ``line`` (the market line
    the pick was made against), edges in points, and W/L/P once graded — at that line and at the close.
    ``recorded`` marks the frozen pick a tracked record grades. None when there is no pick at all."""
    if spread is None and total is None:
        return None
    return {
        "spread": spread, "spread_edge": jsonable(spread_edge), "total": total, "total_edge": jsonable(total_edge),
        "model_total": jsonable(model_total), "line": line, "recorded": bool(recorded),
        "spread_result": spread_result, "spread_result_close": spread_result_close,
        "total_result": total_result, "total_result_close": total_result_close,
    }


def slate_result(*, title: str, source: Literal["run", "stored"], games: list[dict],
                 players: Optional[list[dict]] = None, player_columns: Optional[list[Column]] = None,
                 player_sort: Optional[str] = None, periods: Optional[list[dict]] = None,
                 period: Optional[str] = None, summary: Optional[list[str]] = None,
                 notes: Optional[list[str]] = None, footnote: Optional[str] = None) -> dict:
    """The shared slate payload. ``periods`` / ``period`` drive the stored-view picker."""
    return {
        "title": title, "source": source, "games": games,
        "players": players, "player_columns": [c.to_dict() for c in (player_columns or [])],
        "player_sort": player_sort, "periods": periods or [], "period": period,
        "summary": summary or [], "notes": notes or [], "footnote": footnote,
    }


class ProjectionEngine(ABC):
    """A projection model behind the Projections tab. Methods are synchronous;
    the web layer runs them in a worker thread."""

    name: str = ""
    supports_stored: bool = False
    supports_game_run: bool = False
    game_run_label: str = "Run"

    @abstractmethod
    def slate_options(self, sector: str) -> list[OptionSpec]:
        """Inputs of a slate run for ``sector``."""

    def game_options(self, sector: str) -> list[OptionSpec]:
        """Inputs of a one-game run for ``sector``."""
        return []

    @abstractmethod
    def run_slate(self, sector: str, options: dict) -> dict:
        """Project every game of the sector's next slate; returns ``slate_result``."""

    def run_game(self, sector: str, game: dict, options: dict) -> dict:
        """Run the deeper model for one game row; returns {title, sections, notes}."""
        raise ProjectionError(f"{self.name} has no per-game run")

    def stored(self, sector: str, params: dict) -> dict:
        """Read stored projections (``params`` from the period picker); returns ``slate_result``."""
        raise ProjectionError(f"{self.name} stores no projections")
