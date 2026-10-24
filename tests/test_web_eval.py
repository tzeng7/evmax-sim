"""Dashboard /api/eval — a read-only passthrough of `evmax cleanup eval`."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import evmax.agents.cleanup.db as cleanup_db
from evmax.agents.cleanup import evaluation as ev
from evmax.web.app import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(cleanup_db, "DB_PATH", tmp_path / "predictions.db")
    with cleanup_db.get_connection() as conn:
        for i, (sector, outcome) in enumerate([("wnba", 1), ("wnba", 0), ("nba", 1)]):
            conn.execute(
                """INSERT INTO ev_predictions
                   (scan_date, market_id, event_id, sector, yes_team, market_type,
                    event_date, ev_pct, kelly_fraction, kalshi_yes_price, sharp_true_prob, blended_true_prob,
                    model_sources, mode, venue)
                   VALUES ('2026-10-01', ?, ?, ?, 't', 'moneyline', '2026-10-01', 0.1, 0.01,
                           0.5, 0.55, 0.6, 'elo+sharp', 'live', 'kalshi')""",
                (f"m{i}", f"{sector}-g{i}", sector),
            )
            conn.execute(
                "INSERT INTO ev_outcomes (market_id, event_id, event_date, sector, yes_team, outcome, resolved_at)"
                " VALUES (?, ?, '2026-10-01', ?, 't', ?, '2026-10-02')",
                (f"m{i}", f"{sector}-g{i}", sector, outcome),
            )
    return TestClient(app)


def test_eval_is_byte_identical_to_run_eval(client):
    r = client.get("/api/eval", params={"sector": "wnba", "market": "moneyline",
                                        "population": "raw", "max_staleness_h": 0})
    assert r.status_code == 200
    spec = ev.EvalSpec(sector="wnba", market_type="moneyline", population="raw",
                       max_staleness_h=0.0)
    assert r.text == ev.to_json(ev.run_eval(spec))


def test_eval_returns_required_blocks(client):
    body = client.get("/api/eval", params={"sector": "wnba", "population": "raw"}).json()
    assert body["spec"]["sector"] == "wnba"
    assert {"predictions", "resolved"} <= set(body["db"])
    g = body["groups"][0]
    assert {"funnel", "roi_population", "clv_population", "realized",
            "roi_empty_reason"} <= set(g)
    assert g["funnel"][0]["stage"] == "scope" and g["funnel"][0]["n_rows"] == 2
    assert {"n_rows", "n_games", "row_digest"} <= set(g["roi_population"])
    assert "uncapped" in g["clv_population"]
    assert {"pnl_raw_usd", "pnl_fee_inclusive_usd"} <= set(g["realized"])


def test_eval_empty_group_explains_itself(client):
    g = client.get("/api/eval", params={"sector": "nhl"}).json()["groups"][0]
    assert g["roi_population"]["n_rows"] == 0
    assert g["roi_empty_reason"]


@pytest.mark.parametrize("params", [{"population": "bogus"}, {"venue": "x"},
                                    {"mode": "x"}, {"side": "x"}])
def test_eval_bad_spec_is_400(client, params):
    r = client.get("/api/eval", params=params)
    assert r.status_code == 400 and "error" in r.json()


def test_eval_does_not_write(client):
    before = client.get("/api/eval").json()["db"]
    client.get("/api/eval", params={"sector": "wnba"})
    assert client.get("/api/eval").json()["db"] == before
    assert json.loads(client.get("/api/eval").text)["db"]["predictions"] == 3
