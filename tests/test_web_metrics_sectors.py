"""The dashboard /api/metrics route returns a per-sector performance
breakdown (CLV, EV%, ROI) alongside the model-calibration figures. The
"Metrics" button surfaces this — see evmax.web.app.api_metrics.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

import evmax.agents.cleanup.db as dbmod
from evmax.web.app import app


def _seed(conn, *, market_id, sector, market_type, kalshi_price, blended, sharp,
          ev_pct, clv, kelly, bankroll, outcome, event_date):
    conn.execute(
        """
        INSERT INTO ev_predictions
            (scan_date, market_id, event_id, sector, yes_team, market_type,
             event_title, event_date, kalshi_yes_price, sharp_true_prob,
             blended_true_prob, ev_pct, kelly_fraction, bankroll_used,
             kalshi_clv_pct, mode, voided)
        VALUES (?, ?, ?, ?, 'A', ?, 'A vs B', ?, ?, ?, ?, ?, ?, ?, ?, 'live', 0)
        """,
        (event_date, market_id, market_id + "_evt", sector, market_type,
         event_date, kalshi_price, sharp, blended, ev_pct, kelly, bankroll, clv),
    )
    conn.execute(
        """
        INSERT INTO ev_outcomes
            (market_id, event_id, event_date, sector, yes_team, outcome)
        VALUES (?, ?, ?, ?, 'A', ?)
        """,
        (market_id, market_id + "_evt", event_date, sector, outcome),
    )


@pytest.fixture
def metrics_db(tmp_path, monkeypatch):
    db_path = tmp_path / "predictions.db"
    monkeypatch.setattr(dbmod, "DB_PATH", db_path)
    conn = dbmod.get_connection()  # creates schema + migrations (kalshi_clv_pct)
    today = date.today().isoformat()
    # nba: two bets, one win one loss, both with CLV captured
    _seed(conn, market_id="nba1", sector="nba", market_type="spread",
          kalshi_price=0.50, blended=0.60, sharp=0.58, ev_pct=0.20, clv=2.0,
          kelly=0.04, bankroll=250.0, outcome=1, event_date=today)
    _seed(conn, market_id="nba2", sector="nba", market_type="spread",
          kalshi_price=0.50, blended=0.55, sharp=0.54, ev_pct=0.10, clv=-1.0,
          kelly=0.04, bankroll=250.0, outcome=0, event_date=today)
    # soccer: one bet, missing CLV → avg_clv_pct should be None
    _seed(conn, market_id="soc1", sector="soccer", market_type="moneyline",
          kalshi_price=0.40, blended=0.45, sharp=0.44, ev_pct=0.05, clv=None,
          kelly=0.02, bankroll=250.0, outcome=1, event_date=today)
    conn.commit()
    conn.close()
    return db_path


def _post_metrics():
    with TestClient(app) as client:
        resp = client.post("/api/metrics", json={"weeks": 4})
    assert resp.status_code == 200
    return resp.json()


def test_metrics_includes_sector_breakdown(metrics_db):
    body = _post_metrics()
    assert "sectors" in body
    by = {s["sector"]: s for s in body["sectors"]}
    assert set(by) == {"nba", "soccer"}

    nba = by["nba"]
    assert nba["bets"] == 2
    assert nba["win_rate"] == 50.0
    # avg EV% — fraction → percent: mean(0.20, 0.10) * 100 = 15.0
    assert nba["avg_ev_pct"] == 15.0
    # CLV already in pp: mean(2.0, -1.0) = 0.5
    assert nba["avg_clv_pct"] == 0.5
    assert nba["clv_n"] == 2
    # ROI: stake = 2 * (250 * 0.04) = 20; win pays 0.04*250*(1/0.5-1)=10,
    # loss = -10 → pnl 0 → roi 0%
    assert nba["pnl"] == 0.0
    assert nba["roi_pct"] == 0.0


def test_metrics_sector_handles_missing_clv(metrics_db):
    body = _post_metrics()
    soccer = next(s for s in body["sectors"] if s["sector"] == "soccer")
    assert soccer["avg_clv_pct"] is None  # no CLV captured
    assert soccer["clv_n"] == 0
    assert soccer["avg_ev_pct"] == 5.0  # 0.05 * 100


def test_metrics_sectors_sorted_by_pnl_desc(metrics_db):
    body = _post_metrics()
    pnls = [s["pnl"] for s in body["sectors"]]
    assert pnls == sorted(pnls, reverse=True)


def test_metrics_empty_range_reports_error(tmp_path, monkeypatch):
    monkeypatch.setattr(dbmod, "DB_PATH", tmp_path / "empty.db")
    dbmod.get_connection().close()  # schema only, no rows
    body = _post_metrics()
    assert body.get("error")
    assert "sectors" not in body
