"""Dashboard Projections endpoints: /api/projections/* (catalog, run, game, stored) and their error codes."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from evmax.projections import catalog
from evmax.projections.base import OptionSpec, ProjectionEngine, ProjectionError, game_row, slate_result
from evmax.web.app import app


class FakeEngine(ProjectionEngine):
    name = "fake"
    supports_game_run = True
    game_run_label = "Simulate"
    calls: list = []

    def slate_options(self, sector):
        return [OptionSpec("week", "Week", "int", None, min=1, max=22, nullable=True),
                OptionSpec("players", "Players", "bool", True)]

    def game_options(self, sector):
        return [OptionSpec("sims", "Simulations", "int", 10000, min=1000, max=50000)]

    def run_slate(self, sector, options):
        FakeEngine.calls.append(("slate", sector, options))
        if options["week"] == 13:
            raise ProjectionError("No games in week 13")
        if options["week"] == 14:
            raise RuntimeError("model blew up")
        g = game_row(game_id="g1", home="PHI", away="DAL", home_name="Philadelphia", away_name="Dallas",
                     proj_home=24, proj_away=20, p_home_win=0.6, context={"week": options["week"]})
        return slate_result(title=f"Fake week {options['week']}", source="run", games=[g])

    def run_game(self, sector, game, options):
        FakeEngine.calls.append(("game", game["game_id"], options))
        return {"title": game["game_id"], "sections": [{"kind": "kv", "title": "x", "items": []}], "notes": []}


class NoExtrasEngine(FakeEngine):
    name = "no_extras"
    supports_game_run = False


@pytest.fixture
def client(monkeypatch, tmp_path):
    path = tmp_path / "projections.yaml"
    path.write_text("""
sectors:
  fake: {label: Fake, engine: fake, description: d, defaults: {sims: 2000}}
  plain: {label: Plain, engine: no_extras}
  soon: {label: Soon, status: planned, note: not yet}
""")
    monkeypatch.setitem(catalog.ENGINES, "fake", FakeEngine)
    monkeypatch.setitem(catalog.ENGINES, "no_extras", NoExtrasEngine)
    monkeypatch.setattr(catalog, "_instances", {})
    monkeypatch.setattr(catalog, "_catalog", catalog.load_catalog(path))
    FakeEngine.calls = []
    with TestClient(app) as c:
        yield c


def test_sectors_lists_the_catalog(client):
    sectors = {s["key"]: s for s in client.get("/api/projections/sectors").json()["sectors"]}
    assert list(sectors) == ["fake", "plain", "soon"]
    fake = sectors["fake"]
    assert fake["capabilities"] == {"stored": False, "game_run": True} and fake["game_run_label"] == "Simulate"
    assert fake["game_options"][0]["default"] == 2000                    # YAML default applied
    assert fake["slate_options"][0] == {"key": "week", "label": "Week", "type": "int", "default": None,
                                        "min": 1, "max": 22, "nullable": True}
    assert sectors["soon"]["status"] == "planned" and "slate_options" not in sectors["soon"]


def test_run_resolves_options_and_times_the_call(client):
    r = client.post("/api/projections/FAKE/run", json={"options": {"week": 5}})
    assert r.status_code == 200
    j = r.json()
    assert j["sector"] == "fake" and j["options"] == {"week": 5, "players": True} and j["elapsed_s"] >= 0
    assert j["games"][0]["model_line"] == "PHI -4.0"
    assert FakeEngine.calls == [("slate", "fake", {"week": 5, "players": True})]
    assert client.post("/api/projections/fake/run", json={}).json()["options"] == {"week": None, "players": True}


@pytest.mark.parametrize("sector, body, code, msg", [
    ("nope", {}, 404, "unknown projection sector 'nope'"),
    ("soon", {}, 409, "not available yet: not yet"),
    ("fake", {"options": {"week": 40}}, 400, "Week must be at most 22"),
    ("fake", {"options": {"season": 2026}}, 400, "unknown option(s): season"),
    ("fake", {"options": {"players": "yes"}}, 400, "Players must be true or false"),
    ("fake", {"options": {"week": 13}}, 400, "No games in week 13"),
    ("fake", {"options": {"week": 14}}, 500, "RuntimeError: model blew up"),
])
def test_run_errors(client, sector, body, code, msg):
    r = client.post(f"/api/projections/{sector}/run", json=body)
    assert r.status_code == code and msg in r.json()["error"]


def test_game_run(client):
    game = client.post("/api/projections/fake/run", json={"options": {"week": 5}}).json()["games"][0]
    r = client.post("/api/projections/fake/game", json={"game": game, "options": {"sims": 5000}})
    assert r.status_code == 200
    j = r.json()
    assert j["game_id"] == "g1" and j["options"] == {"sims": 5000} and j["sections"][0]["kind"] == "kv"
    assert FakeEngine.calls[-1] == ("game", "g1", {"sims": 5000})
    assert client.post("/api/projections/fake/game", json={"game": game}).json()["options"] == {"sims": 2000}


@pytest.mark.parametrize("sector, body, code, msg", [
    ("fake", {}, 400, "needs the game row"),
    ("fake", {"game": {"home": "PHI"}}, 400, "needs the game row"),
    ("fake", {"game": {"game_id": "g1"}, "options": {"sims": 10}}, 400, "at least 1000"),
    ("plain", {"game": {"game_id": "g1"}}, 409, "has no per-game run"),
    ("soon", {"game": {"game_id": "g1"}}, 409, "not available yet"),
])
def test_game_errors(client, sector, body, code, msg):
    r = client.post(f"/api/projections/{sector}/game", json=body)
    assert r.status_code == code and msg in r.json()["error"]


def test_stored_needs_the_capability(client):
    r = client.get("/api/projections/fake/stored")
    assert r.status_code == 409 and "not stored" in r.json()["error"]


def test_stored_passes_query_params(client, monkeypatch):
    seen = {}

    def stored(self, sector, params):
        seen.update(params)
        return slate_result(title="t", source="stored", games=[])

    monkeypatch.setattr(FakeEngine, "supports_stored", True)
    monkeypatch.setattr(FakeEngine, "stored", stored)
    r = client.get("/api/projections/fake/stored?season=2026&week=5")
    assert r.status_code == 200 and r.json()["source"] == "stored" and seen == {"season": "2026", "week": "5"}


@pytest.mark.parametrize("path", ["/api/projections/fake/run", "/api/projections/fake/game"])
@pytest.mark.parametrize("raw", [b"[1, 2]", b"not json", b'"text"'])
def test_bodies_must_be_json_objects(client, path, raw):
    r = client.post(path, content=raw, headers={"content-type": "application/json"})
    assert r.status_code == 400 and r.json() == {"error": "request body must be a JSON object"}


def test_options_must_be_an_object(client):
    r = client.post("/api/projections/fake/run", json={"options": [5]})
    assert r.status_code == 400 and r.json()["error"] == "options must be an object"


def test_unserializable_output_maps_to_an_error_body(client, monkeypatch):
    def nan_slate(self, sector, options):
        return {"title": "t", "value": float("nan")}

    monkeypatch.setattr(FakeEngine, "run_slate", nan_slate)
    r = client.post("/api/projections/fake/run", json={})
    assert r.status_code == 500 and "ValueError" in r.json()["error"]
