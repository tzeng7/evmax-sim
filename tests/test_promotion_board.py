"""Tests for the promotion scoreboard (WS4, 2026-07-19).

compute_promotion_board groups ev_predictions by (sector, market_type,
venue) and combines sample counts, Brier blend-vs-sharp, CLV gate status,
blend divergence (the sharp-passthrough detector), and why-not blocker
aggregation. Tests run with staleness_h=None so archive.db is not touched.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pytest

from evmax.agents.cleanup.promotion_board import (
    SHARP_PASSTHROUGH_PP,
    _top_blockers,
    compute_promotion_board,
)

_TODAY = date.today()
_GAME_DATE = (_TODAY - timedelta(days=3)).isoformat()


def _make_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE ev_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_date TEXT NOT NULL,
            market_id TEXT NOT NULL,
            event_id TEXT NOT NULL,
            sector TEXT, yes_team TEXT, market_type TEXT,
            event_title TEXT, event_date TEXT,
            kalshi_yes_price REAL, sharp_true_prob REAL, blended_true_prob REAL,
            ev_pct REAL, kelly_fraction REAL, volume_usd REAL,
            model_sources TEXT, line REAL,
            voided INTEGER NOT NULL DEFAULT 0,
            placed INTEGER NOT NULL DEFAULT 0,
            placed_at TEXT,
            mode TEXT NOT NULL DEFAULT 'live',
            venue TEXT NOT NULL DEFAULT 'kalshi',
            kalshi_clv_pct REAL,
            model_diagnostics TEXT,
            UNIQUE(market_id)
        );
        CREATE TABLE ev_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_id TEXT UNIQUE, outcome INTEGER
        );
        """
    )
    return conn


def _insert(
    conn,
    market_id: str,
    sector: str = "wnba",
    market_type: str = "moneyline",
    venue: str = "kalshi",
    mode: str = "live",
    blended: float = 0.60,
    sharp: float = 0.55,
    sources: str = "elo+wnba_efficiency+sharp",
    outcome: int | None = 1,
    clv: float | None = 1.0,
    line: float | None = None,
    diagnostics: str | None = None,
) -> None:
    conn.execute(
        """INSERT INTO ev_predictions
           (scan_date, market_id, event_id, sector, yes_team, market_type,
            event_date, kalshi_yes_price, sharp_true_prob, blended_true_prob,
            ev_pct, kelly_fraction, model_sources, line, mode, venue,
            kalshi_clv_pct, model_diagnostics)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            _GAME_DATE, market_id, f"evt:{market_id}", sector, "team", market_type,
            _GAME_DATE, 0.50, sharp, blended, 0.05, 0.02, sources, line,
            mode, venue, clv, diagnostics,
        ),
    )
    if outcome is not None:
        conn.execute(
            "INSERT INTO ev_outcomes (market_id, outcome) VALUES (?, ?)",
            (market_id, outcome),
        )


@pytest.fixture
def patched(tmp_path, monkeypatch):
    db_path = tmp_path / "predictions.db"
    conn = _make_db(db_path)
    from evmax.agents.cleanup import db as db_module

    monkeypatch.setattr(db_module, "DB_PATH", db_path)
    return conn


def _board(**kw):
    kw.setdefault("staleness_h", None)
    return compute_promotion_board(**kw)


class TestDivergenceAndPassthrough:
    def test_divergence_math(self, patched):
        _insert(patched, "m1", blended=0.60, sharp=0.55)   # 5pp
        _insert(patched, "m2", blended=0.50, sharp=0.53)   # 3pp
        patched.commit()
        row = _board(sector="wnba")[0]
        assert row["blend_divergence_pp"] == pytest.approx(4.0)
        assert row["sharp_passthrough"] is False

    def test_moneyline_passthrough_flag(self, patched):
        for i in range(3):
            _insert(patched, f"m{i}", sector="baseball", blended=0.551,
                    sharp=0.550, sources="elo+pitcher_v2+sharp")
        patched.commit()
        row = _board(sector="baseball")[0]
        assert row["blend_divergence_pp"] < SHARP_PASSTHROUGH_PP
        assert row["sharp_passthrough"] is True
        assert row["verdict"] == "SHARP-PASSTHROUGH"

    def test_spread_never_flagged_passthrough(self, patched):
        _insert(patched, "s1", sector="wnba", market_type="spread",
                blended=0.55, sharp=0.55, sources="sharp+spread_dist", line=-5.5)
        patched.commit()
        row = _board(sector="wnba")[0]
        assert row["sharp_passthrough"] is False
        assert row["verdict"].startswith("COLLECTING")


class TestGatesAndVerdicts:
    def _fill(self, conn, n_pos: int, n_neg: int, *, sector="baseball"):
        """Insert n resolved rows with a controlled fraction of +CLV.

        Uses baseball ML — registry mode 'shadow', so the shadow-side verdict
        paths (COLLECTING / PROMOTE-READY / FAILING-CLV) apply. Sources carry
        'pitcher' so the contamination filter keeps them, and the divergence
        (7pp) clears the passthrough flag.
        """
        total = n_pos + n_neg
        for i in range(total):
            _insert(
                conn, f"g{i}", sector=sector, mode="shadow",
                blended=0.62, sharp=0.55, sources="elo+form+pitcher_v2+sharp",
                clv=1.5 if i < n_pos else -1.0,
                outcome=1 if i % 2 == 0 else 0,
            )
        conn.commit()

    def test_collecting_below_30(self, patched):
        self._fill(patched, 15, 14)  # 29 clean resolved
        row = _board(sector="baseball")[0]
        assert row["verdict"] == "COLLECTING 29/30"
        assert row["gates"]["clean_n"]["ok"] is False

    def test_promote_ready_at_boundary(self, patched):
        # 30 rows, 17 positive CLV = 56.7% ≥ 55%, mean > 0 → all gates clear.
        self._fill(patched, 17, 13)
        row = _board(sector="baseball")[0]
        assert row["gates"]["clean_n"]["ok"] is True
        assert row["gates"]["clv_frac_pos"]["ok"] is True
        assert row["verdict"] == "PROMOTE-READY"

    def test_failing_clv_below_frac_threshold(self, patched):
        # 30 rows, 16 positive = 53.3% < 55% → frac gate fails.
        self._fill(patched, 16, 14)
        row = _board(sector="baseball")[0]
        assert row["gates"]["clv_frac_pos"]["ok"] is False
        assert row["verdict"] == "FAILING-CLV"

    def test_live_degrading_on_negative_mean(self, patched):
        # Live mode, n ≥ 30, mean CLV negative.
        for i in range(32):
            _insert(patched, f"d{i}", sector="wnba", mode="live",
                    blended=0.62, sharp=0.55, clv=-1.2,
                    outcome=1 if i % 2 == 0 else 0)
        patched.commit()
        row = _board(sector="wnba")[0]
        assert row["verdict"] == "LIVE-DEGRADING"


class TestGroupingAndFilters:
    def test_venue_split(self, patched):
        _insert(patched, "k1", venue="kalshi")
        _insert(patched, "p1", venue="polymarket_us")
        patched.commit()
        rows = _board(sector="wnba")
        venues = {r["venue"] for r in rows}
        assert venues == {"kalshi", "polymarket_us"}

    def test_contaminated_rows_excluded_from_clean(self, patched):
        # Soccer sharp-only ML = contaminated (pre-guard); with-model = clean.
        _insert(patched, "c1", sector="soccer", sources="sharp", outcome=1)
        _insert(patched, "c2", sector="soccer", sources="elo+sharp", outcome=0)
        patched.commit()
        row = _board(sector="soccer")[0]
        assert row["n_resolved"] == 2
        assert row["n_clean_resolved"] == 1

    def test_voided_and_prop_rows_excluded(self, patched):
        _insert(patched, "v1")
        patched.execute("UPDATE ev_predictions SET voided=1 WHERE market_id='v1'")
        patched.execute(
            """INSERT INTO ev_predictions
               (scan_date, market_id, event_id, sector, yes_team, market_type,
                event_date, kalshi_yes_price, sharp_true_prob, blended_true_prob,
                ev_pct, kelly_fraction, mode, venue)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (_GAME_DATE, "pr1", "wnba::x::prop::player::points::20.5", "wnba",
             "p", "player_prop", _GAME_DATE, 0.5, 0.5, 0.5, 0.02, 0.01,
             "shadow", "kalshi"),
        )
        patched.commit()
        assert _board(sector="wnba") == []


class TestTopBlockers:
    def test_aggregates_missing_and_gated(self):
        diags = [
            json.dumps({"missing": ["tennis_surface"], "gated": {}}),
            json.dumps({"missing": ["tennis_surface"],
                        "gated": {"tennis_advanced": {"conf": 0.3}}}),
            None,
            "not-json",
        ]
        out = _top_blockers(diags)
        assert out[0] == "tennis_surface:missing×2"
        assert "tennis_advanced:gated×1" in out

    def test_blockers_surface_in_board(self, patched):
        diag = json.dumps({"missing": ["poisson"], "gated": {}})
        for i in range(3):
            _insert(patched, f"b{i}", sector="soccer",
                    sources="elo+sharp", diagnostics=diag)
        patched.commit()
        row = _board(sector="soccer")[0]
        assert row["top_blockers"] == ["poisson:missing×3"]


class TestEndpointShape:
    def test_api_promotion_board(self, patched, monkeypatch):
        from fastapi.testclient import TestClient
        from evmax.web.app import app

        _insert(patched, "e1", sector="wnba", blended=0.62, sharp=0.55)
        patched.commit()

        client = TestClient(app)
        resp = client.get("/api/promotion-board?days=30&staleness_h=0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["days"] == 30
        assert isinstance(data["rows"], list)
        wnba = [r for r in data["rows"] if r["sector"] == "wnba"]
        assert wnba, "seeded wnba row missing from board"
        row = wnba[0]
        for key in ("verdict", "gates", "clv", "blend_divergence_pp",
                    "sharp_passthrough", "top_blockers", "mode"):
            assert key in row
