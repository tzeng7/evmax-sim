"""Dashboard /api/integrity — read-only wrapper over the integrity sweep."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import evmax.agents.cleanup.integrity as integrity
import evmax.web.app as web
from evmax.web.app import app


@pytest.fixture(autouse=True)
def _clear_cache():
    web._integrity_cache.clear()
    yield
    web._integrity_cache.clear()


def _fake(calls, issues, ran=("cadence", "inplay"), failed=()):
    def run(**kw):
        calls.append(kw)
        return {"ok": not issues, "issues": list(issues), "notified": False,
                "ran": list(ran), "failed": list(failed)}
    return run


def test_rollup_status_and_never_notifies(monkeypatch):
    calls = []
    issues = [
        {"check": "pinnacle", "severity": "critical", "detail": "Pinnacle down"},
        {"check": "inplay", "severity": "warning", "detail": "x"},
    ]
    monkeypatch.setattr(integrity, "run_integrity",
                        _fake(calls, issues, ran=("cadence", "inplay", "pinnacle")))
    r = TestClient(app).get("/api/integrity?check_pinnacle=true&only=cadence, pinnacle")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "critical"
    by = {c["check"]: c for c in body["checks"]}
    assert by["cadence"]["severity"] == "ok"
    assert by["pinnacle"]["severity"] == "critical"
    assert "generated_at" in body
    assert calls[0]["notify"] is False
    assert calls[0]["check_pinnacle"] is True
    assert calls[0]["only"] == {"cadence", "pinnacle"}


def test_clean_is_ok_and_info_only_is_ok(monkeypatch):
    calls = []
    monkeypatch.setattr(integrity, "run_integrity",
                        _fake(calls, [{"check": "gates", "severity": "info", "detail": "cleared"}]))
    assert TestClient(app).get("/api/integrity").json()["status"] == "ok"


def test_crashed_check_is_warning(monkeypatch):
    monkeypatch.setattr(integrity, "run_integrity", _fake([], [], failed=("board",)))
    assert TestClient(app).get("/api/integrity").json()["status"] == "warning"


def test_cached_for_ttl(monkeypatch):
    calls = []
    monkeypatch.setattr(integrity, "run_integrity", _fake(calls, []))
    c = TestClient(app)
    c.get("/api/integrity")
    c.get("/api/integrity")
    assert len(calls) == 1
    c.get("/api/integrity?check_pinnacle=true")
    assert len(calls) == 2


def test_error_returns_500_json(monkeypatch):
    def boom(**kw):
        raise RuntimeError("db locked")
    monkeypatch.setattr(integrity, "run_integrity", boom)
    r = TestClient(app).get("/api/integrity")
    assert r.status_code == 500
    assert "db locked" in r.json()["error"]
