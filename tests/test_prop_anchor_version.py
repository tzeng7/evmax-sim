"""Prop rows carry the pricing version that produced them (model_version).

2026-10-09: NFL receiving/rushing yards moved from a fixed-σ Normal to a
fixed-scale Gamma, so new prop rows are tagged 'pinnacle-anchor-v2'. Readers
that want every anchor-priced row (dashboard settled props, portfolio
backfill) must keep seeing the v1 history too; legacy L15 rows stay excluded.
"""
from __future__ import annotations

import pytest

from evmax.portfolios import ANCHOR_MODEL_VERSION, ANCHOR_MODEL_VERSIONS, anchor_version_clause


def test_current_tag_is_v2_and_readers_accept_v1_history():
    assert ANCHOR_MODEL_VERSION == "pinnacle-anchor-v2"
    assert ANCHOR_MODEL_VERSIONS == ("pinnacle-anchor-v1", "pinnacle-anchor-v2")
    sql, params = anchor_version_clause()
    assert sql == "model_version IN (?,?)"
    assert params == ANCHOR_MODEL_VERSIONS


@pytest.fixture
def pred_db(tmp_path, monkeypatch):
    from evmax.agents.cleanup import db as cleanup_db

    monkeypatch.setattr(cleanup_db, "DB_PATH", tmp_path / "predictions.db")
    conn = cleanup_db.get_connection()
    for i, version in enumerate(
        ["pinnacle-anchor-v1", "pinnacle-anchor-v2", None, "pinnacle-v1"]
    ):
        conn.execute(
            "INSERT INTO prop_observations (scan_date, event_date, sector, player_name, "
            "stat_type, line, kalshi_price, sharp_prob, ev_pct, market_id, event_id, "
            "outcome, model_version) VALUES ('2026-10-04', '2026-10-04', 'nfl', ?, "
            "'receiving_yards', 90, 0.40, 0.45, 0.05, ?, 'e', 1, ?)",
            (f"p{i}", f"kalshi:T{i}", version),
        )
    conn.commit()
    conn.close()
    return tmp_path / "predictions.db"


def test_dashboard_settled_props_read_both_anchor_versions(pred_db):
    from evmax.web.app import _settled_prop_bets

    versions = sorted(r["model_sources"] for r in _settled_prop_bets())
    assert versions == ["pinnacle-anchor-v1", "pinnacle-anchor-v2"]
