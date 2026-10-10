"""evmax.projections: option specs, row helpers, and the data/projections.yaml catalog."""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pytest

from evmax.projections import catalog
from evmax.projections.base import (
    OptionSpec, ProjectionEngine, ProjectionError, game_row, jsonable, range_cell, resolve_options,
)

# ── options ──────────────────────────────────────────────────────────────────

WEEK = OptionSpec("week", "Week", "int", None, min=1, max=22, nullable=True)
SIMS = OptionSpec("sims", "Simulations", "int", 10000, min=1000, max=50000)
STORE = OptionSpec("store", "Store", "bool", False)


def test_int_option_coercion_and_bounds():
    assert SIMS.coerce(5000) == 5000 and SIMS.coerce("5000") == 5000 and SIMS.coerce(5000.0) == 5000
    assert SIMS.coerce(None) == 10000 and SIMS.coerce("") == 10000          # missing -> default
    assert WEEK.coerce(None) is None                                       # nullable -> auto
    for bad, msg in ((999, "at least 1000"), (60000, "at most 50000"), (5000.5, "whole number"),
                     ("abc", "whole number"), (True, "whole number")):
        with pytest.raises(ProjectionError, match=msg):
            SIMS.coerce(bad)


def test_bool_option_is_strict():
    assert STORE.coerce(True) is True and STORE.coerce(None) is False
    with pytest.raises(ProjectionError, match="true or false"):
        STORE.coerce("yes")
    with pytest.raises(ProjectionError, match="true or false"):
        STORE.coerce(1)


def test_required_option_without_default():
    with pytest.raises(ProjectionError, match="required"):
        OptionSpec("n", "N", "int").coerce(None)


def test_resolve_options_fills_defaults_and_rejects_unknown_keys():
    assert resolve_options([WEEK, SIMS, STORE], {"week": 5}) == {"week": 5, "sims": 10000, "store": False}
    with pytest.raises(ProjectionError, match="unknown option"):
        resolve_options([WEEK], {"week": 5, "season": 2026})


def test_option_to_dict_always_carries_the_default():
    d = WEEK.to_dict()
    assert d["default"] is None and d["nullable"] is True and "step" not in d and "help" not in d
    assert SIMS.with_default(20000).to_dict()["default"] == 20000
    with pytest.raises(ProjectionError):
        SIMS.with_default(5)                                               # a default must pass validation


# ── row helpers ──────────────────────────────────────────────────────────────

def test_range_cell_reports_result_against_the_range():
    assert range_cell(None) is None and range_cell(float("nan")) is None
    assert range_cell(50.0, 20.0, 90.0) == {"value": 50.0, "lo": 20.0, "hi": 90.0}
    inside = range_cell(50.0, 20.0, 90.0, 75.0)
    assert inside["result"] == "actual 75" and inside["hit"] is True
    assert range_cell(50.0, 20.0, 90.0, 110.0)["hit"] is False
    assert "hit" not in range_cell(50.0, None, None, 10.0)                  # no range, no verdict


def test_jsonable_scalars():
    assert jsonable(np.float64(1.5)) == 1.5 and isinstance(jsonable(np.int64(3)), int)
    assert jsonable(float("nan")) is None and jsonable(np.bool_(True)) is True
    assert jsonable(date(2026, 10, 11)) == "2026-10-11"
    import pandas as pd
    assert jsonable(pd.NaT) is None and jsonable(pd.Timestamp("2026-10-11")).startswith("2026-10-11")


def test_game_row_uses_the_home_margin_convention():
    g = game_row(game_id="g", home="PHI", away="DAL", home_name="Philadelphia Eagles", away_name="Dallas Cowboys",
                 proj_home=24.0, proj_away=20.0, p_home_win=0.62, market_home_margin=-1.5, market_total=44.5)
    assert g["proj_margin"] == 4.0 and g["proj_total"] == 44.0
    assert g["model_line"] == "PHI -4.0" and g["market_line"] == "DAL -1.5"   # negative home margin = away favored
    no_market = game_row(game_id="g", home="A", away="B", home_name="A", away_name="B",
                         proj_home=1, proj_away=1, p_home_win=0.5)
    assert no_market["market_line"] is None and no_market["model_line"] == "PK"
    json.dumps(g)


# ── catalog ──────────────────────────────────────────────────────────────────

def test_shipped_catalog_is_valid_and_serializable():
    cat = catalog.load_catalog()
    keys = [s.key for s in cat.sectors]
    assert keys[0] == "nfl" and {"nfl", "nba", "wnba"} <= set(keys)
    dicts = cat.to_dicts()
    json.dumps(dicts)
    by = {d["key"]: d for d in dicts}
    nfl = by["nfl"]
    assert nfl["status"] == "available" and nfl["capabilities"] == {"stored": True, "game_run": True}
    assert {o["key"] for o in nfl["slate_options"]} >= {"season", "week", "store"}
    assert [o["key"] for o in nfl["game_options"]] == ["sims"]
    assert by["nba"]["capabilities"]["stored"] is False and by["nba"]["slate_options"][0]["key"] == "injuries"
    assert by["ncaab"]["slate_options"] == []                               # no injury model outside NBA
    assert by["wnba"]["status"] == "planned" and "slate_options" not in by["wnba"] and by["wnba"]["note"]
    for d in dicts:                                                         # every available sector is runnable
        if d["status"] == "available":
            assert d["engine"] in catalog.ENGINES


def _write(tmp_path, body: str):
    p = tmp_path / "projections.yaml"
    p.write_text(body)
    return p


@pytest.mark.parametrize("body, msg", [
    ("sectors: {}", "non-empty"),
    ("sectors:\n  x: {status: available}", "needs an engine"),
    ("sectors:\n  x: {status: available, engine: nope}", "unknown engine"),
    ("sectors:\n  x: {status: maybe, engine: nfl_projections}", "status must be"),
    ("sectors:\n  x: {engine: nfl_projections, colour: red}", "unknown field"),
    ("sectors:\n  x: {engine: nfl_projections, defaults: {bogus: 1}}", "unknown option 'bogus'"),
    ("sectors:\n  x: {engine: nfl_projections, defaults: {sims: 5}}", "bad default for 'sims'"),
    ("sectors:\n  x: {status: planned, defaults: {sims: 5000}}", "defaults need an engine"),
])
def test_catalog_validation_errors(tmp_path, body, msg):
    with pytest.raises(catalog.CatalogError, match=msg):
        catalog.load_catalog(_write(tmp_path, body))


def test_catalog_defaults_override_engine_defaults(tmp_path):
    cat = catalog.load_catalog(_write(tmp_path, """
sectors:
  nfl: {label: NFL, engine: nfl_projections, defaults: {sims: 20000, store: true}}
  wnba: {label: WNBA, status: planned, note: soon}
"""))
    nfl = cat.get("nfl")
    assert {o.key: o.default for o in nfl.game_options()}["sims"] == 20000
    assert catalog.resolve_run_options(nfl, "slate", {})["store"] is True
    assert catalog.resolve_run_options(nfl, "game", {"sims": 3000})["sims"] == 3000
    with pytest.raises(catalog.SectorUnavailable, match="not available yet: soon"):
        cat.get("wnba").engine_obj()
    with pytest.raises(catalog.UnknownSector):
        cat.get("mlb")


def test_engine_instances_are_shared(monkeypatch):
    class Dummy(ProjectionEngine):
        name = "dummy"

        def slate_options(self, sector):
            return []

        def run_slate(self, sector, options):
            return {}

    monkeypatch.setitem(catalog.ENGINES, "dummy", Dummy)
    monkeypatch.setattr(catalog, "_instances", {})
    assert catalog.engine_instance("dummy") is catalog.engine_instance("dummy")
    with pytest.raises(ProjectionError, match="no per-game run"):
        Dummy().run_game("x", {}, {})
    with pytest.raises(ProjectionError, match="stores no projections"):
        Dummy().stored("x", {})
