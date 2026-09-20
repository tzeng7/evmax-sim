"""Tests for the entry-price bucket readout (favorite–longshot lens).

Covers evmax/agents/cleanup/price_buckets.py, the ``price_bucket`` filter on
``clv_stats`` / ``_fetch_clv_rows`` / ``compute_promotion_board``, the
``clv-prices`` CLI command, and the replay-harness ``require_sized`` knob.
"""
from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from evmax.agents.cleanup.price_buckets import (
    BUCKET_DESC,
    BUCKET_ORDER,
    bucket_for_row,
    calibration_summary,
    entry_price_for_row,
    price_bucket,
    validate_bucket,
)
from evmax.cli.commands.shadow import app, clv_stats

runner = CliRunner()


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


class TestPriceBucket:
    @pytest.mark.parametrize(
        "price,label",
        [
            (0.01, "0-10"), (0.099, "0-10"),
            (0.10, "10-20"), (0.199, "10-20"),
            (0.20, "20-35"), (0.349, "20-35"),
            (0.35, "35-50"), (0.499, "35-50"),
            (0.50, "50-65"), (0.649, "50-65"),
            (0.65, "65-80"), (0.799, "65-80"),
            (0.80, "80-90"), (0.899, "80-90"),
            (0.90, "90+"), (0.99, "90+"),
        ],
    )
    def test_edges_are_left_closed_right_open(self, price, label):
        assert price_bucket(price) == label

    @pytest.mark.parametrize("bad", [None, 0.0, 1.0, -0.2, 1.7, "abc"])
    def test_unusable_prices_bucket_unknown(self, bad):
        assert price_bucket(bad) == "unknown"

    def test_every_bucket_has_a_description_and_order_is_unique(self):
        assert set(BUCKET_ORDER) == set(BUCKET_DESC)
        assert len(BUCKET_ORDER) == len(set(BUCKET_ORDER))
        assert BUCKET_ORDER[-1] == "unknown"

    def test_validate_bucket_normalizes_and_rejects(self):
        assert validate_bucket(None) is None
        assert validate_bucket(" 35-50 ") == "35-50"
        assert validate_bucket("90+") == "90+"
        with pytest.raises(ValueError):
            validate_bucket("30-40")


class TestEntryPriceForRow:
    def test_scan_ask_when_not_placed(self):
        row = {"placed": 0, "placed_price": None, "kalshi_yes_price": 0.12}
        assert entry_price_for_row(row) == 0.12
        assert bucket_for_row(row) == "10-20"

    def test_placed_fill_takes_precedence(self):
        # A bet scanned at 12c but filled at 22c is bucketed by the FILL — the
        # price its CLV was measured against — not the scan ask.
        row = {"placed": 1, "placed_price": 0.22, "kalshi_yes_price": 0.12}
        assert entry_price_for_row(row) == 0.22
        assert bucket_for_row(row) == "20-35"

    def test_bogus_fill_falls_back_to_scan_ask(self):
        row = {"placed": 1, "placed_price": 0.0, "kalshi_yes_price": 0.12}
        assert bucket_for_row(row) == "10-20"

    def test_no_side_row_is_not_flipped(self):
        # NO-side rows (":no" market ids) already store OUR side's ask in
        # kalshi_yes_price (ev_gap_agent sets kalshi_yes_price=no_ask), so the
        # bucket reads it as-is — no 1-price flip.
        row = {"market_id": "kalshi:T:no", "placed": 0, "placed_price": None,
               "kalshi_yes_price": 0.15}
        assert bucket_for_row(row) == "10-20"

    def test_sqlite_row_and_missing_columns(self):
        con = sqlite3.connect(":memory:")
        con.row_factory = sqlite3.Row
        r = con.execute("SELECT 0 AS placed, NULL AS placed_price, 0.7 AS kalshi_yes_price").fetchone()
        assert bucket_for_row(r) == "65-80"
        r2 = con.execute("SELECT 1 AS x").fetchone()
        assert bucket_for_row(r2) == "unknown"


class TestCalibrationSummary:
    def test_empty(self):
        s = calibration_summary([])
        assert s["n"] == 0 and s["win_rate"] is None

    def test_blend_over_states_bucket(self):
        # Blend says 20% on every row, 10% actually win → blend−real = +10pp.
        rows = [
            {"outcome": 1 if i < 1 else 0, "blended_true_prob": 0.20, "sharp_true_prob": 0.15}
            for i in range(10)
        ]
        s = calibration_summary(rows)
        assert s["n"] == 10
        assert s["win_rate"] == pytest.approx(0.10)
        assert s["mean_blended"] == pytest.approx(0.20)
        assert s["mean_sharp"] == pytest.approx(0.15)
        assert s["blend_minus_realized_pp"] == pytest.approx(10.0)
        assert s["sharp_minus_realized_pp"] == pytest.approx(5.0)

    def test_rows_without_blend_or_outcome_are_skipped_and_sharp_optional(self):
        rows = [
            {"outcome": 1, "blended_true_prob": 0.6},
            {"outcome": None, "blended_true_prob": 0.6},
            {"outcome": 0, "blended_true_prob": None},
        ]
        s = calibration_summary(rows)
        assert s["n"] == 1 and s["mean_sharp"] is None
        assert s["sharp_minus_realized_pp"] is None


# ---------------------------------------------------------------------------
# DB-backed: clv_stats(price_bucket=), clv-prices CLI, promotion board filter
# ---------------------------------------------------------------------------


def _make_db(tmp_path: Path, rows: list[dict]) -> Path:
    """rows: dicts with market_id, price, clv, outcome (+ optional blended,
    sharp, placed, placed_price, sector, event_id). Minimal schema = the
    superset _fetch_clv_rows and compute_promotion_board select; deliberately
    no UNIQUE(market_id, scan_date) so get_connection()'s rebuild is skipped.
    """
    db_path = tmp_path / "predictions.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE ev_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_date TEXT, market_id TEXT, event_id TEXT, event_title TEXT,
            event_date TEXT, sector TEXT, market_type TEXT, line REAL,
            model_sources TEXT, model_diagnostics TEXT, mode TEXT,
            kalshi_clv_pct REAL, venue TEXT NOT NULL DEFAULT 'kalshi',
            voided INTEGER NOT NULL DEFAULT 0,
            placed INTEGER DEFAULT 0, placed_at TEXT, placed_price REAL,
            kalshi_yes_price REAL, blended_true_prob REAL, sharp_true_prob REAL,
            league TEXT
        );
        CREATE TABLE ev_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_id TEXT UNIQUE, outcome INTEGER
        );
        """
    )
    game_date = (date.today() - timedelta(days=2)).isoformat()
    for r in rows:
        conn.execute(
            """INSERT INTO ev_predictions
               (scan_date, market_id, event_id, event_title, event_date, sector,
                market_type, model_sources, mode, kalshi_clv_pct, placed,
                placed_price, kalshi_yes_price, blended_true_prob, sharp_true_prob)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                game_date, r["market_id"], r.get("event_id", r["market_id"] + "-evt"),
                "A vs B", game_date, r.get("sector", "nfl"), "moneyline",
                "nfl_efficiency,elo,sharp", r.get("mode", "shadow"), r["clv"],
                r.get("placed", 0), r.get("placed_price"), r["price"],
                r.get("blended", r["price"] + 0.03), r.get("sharp", r["price"] + 0.01),
            ),
        )
        if r.get("outcome") is not None:
            conn.execute(
                "INSERT INTO ev_outcomes (market_id, outcome) VALUES (?, ?)",
                (r["market_id"], r["outcome"]),
            )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def _patch_db(monkeypatch):
    def _apply(db_path):
        from evmax.agents.cleanup import db as db_module

        monkeypatch.setattr(db_module, "DB_PATH", db_path)

    return _apply


def _two_bucket_rows() -> list[dict]:
    # 10-20c longshots: +3.00pp CLV, blend 0.20, none win (blend over-states).
    dogs = [
        {"market_id": f"dog{i}", "price": 0.15, "clv": 3.0, "outcome": 0,
         "blended": 0.20, "sharp": 0.16}
        for i in range(4)
    ]
    # 65-80c favorites: -1.00pp CLV, blend 0.70, all win.
    favs = [
        {"market_id": f"fav{i}", "price": 0.70, "clv": -1.0, "outcome": 1,
         "blended": 0.70, "sharp": 0.71}
        for i in range(3)
    ]
    return dogs + favs


class TestClvStatsPriceBucket:
    def test_filter_isolates_one_bucket(self, tmp_path, _patch_db):
        _patch_db(_make_db(tmp_path, _two_bucket_rows()))
        pooled = clv_stats("nfl")
        dogs = clv_stats("nfl", price_bucket="10-20")
        favs = clv_stats("nfl", price_bucket="65-80")
        empty = clv_stats("nfl", price_bucket="35-50")
        assert pooled["n"] == 7
        assert dogs["n"] == 4 and dogs["mean_clv_pp"] == pytest.approx(3.0)
        assert favs["n"] == 3 and favs["mean_clv_pp"] == pytest.approx(-1.0)
        assert empty["n"] == 0

    def test_placed_fill_moves_row_between_buckets(self, tmp_path, _patch_db):
        rows = [{"market_id": "m", "price": 0.15, "clv": 1.0, "outcome": 1,
                 "placed": 1, "placed_price": 0.40}]
        _patch_db(_make_db(tmp_path, rows))
        assert clv_stats("nfl", price_bucket="10-20")["n"] == 0
        assert clv_stats("nfl", price_bucket="35-50")["n"] == 1

    def test_invalid_bucket_raises(self, tmp_path, _patch_db):
        _patch_db(_make_db(tmp_path, []))
        with pytest.raises(ValueError):
            clv_stats("nfl", price_bucket="lol")


class TestClvPricesCommand:
    def test_empty_db_prints_friendly_message(self, tmp_path, _patch_db):
        _patch_db(_make_db(tmp_path, []))
        result = runner.invoke(app, ["clv-prices", "nfl"])
        assert result.exit_code == 0
        assert "No current-code resolved CLV rows" in result.stdout

    def test_reports_each_bucket_separately_with_calibration(self, tmp_path, _patch_db):
        _patch_db(_make_db(tmp_path, _two_bucket_rows()))
        result = runner.invoke(app, ["clv-prices", "nfl"])
        assert result.exit_code == 0, result.stdout
        out = result.stdout
        lines = {ln.split()[0]: ln for ln in out.splitlines() if ln.strip()}
        # Per-bucket CLV means on their own rows, not a pooled number.
        assert "+3.00" in lines["10-20c"] and "-1.00" in lines["65-80c"]
        # Calibration: longshot bucket blend 20.0 vs 0.0% wins → +20.0pp over-statement;
        # favorite bucket blend 70.0 vs 100% wins → -30.0pp.
        assert "+20.0" in lines["10-20c"] and "0.0" in lines["10-20c"]
        assert "-30.0" in lines["65-80c"] and "100.0" in lines["65-80c"]
        # Bucket with no members still lists (zeros) — except 'unknown', hidden when empty.
        assert "35-50c" in lines and " 0 " in lines["35-50c"]
        assert not any(k.startswith("unknown") for k in lines)

    def test_pass_through_filters_reach_the_fetcher(self, tmp_path, _patch_db):
        # --venue must narrow the bucketed rows exactly as `clv --venue` does.
        rows = _two_bucket_rows()
        db_path = _make_db(tmp_path, rows)
        con = sqlite3.connect(str(db_path))
        con.execute("UPDATE ev_predictions SET venue='polymarket_us' WHERE market_id LIKE 'fav%'")
        con.commit(); con.close()
        _patch_db(db_path)
        kalshi_only = runner.invoke(app, ["clv-prices", "nfl", "--venue", "kalshi"])
        assert kalshi_only.exit_code == 0, kalshi_only.stdout
        lines = {ln.split()[0]: ln for ln in kalshi_only.stdout.splitlines() if ln.strip()}
        assert "+3.00" in lines["10-20c"]
        assert " 0 " in lines["65-80c"]  # the PolyUS favorites are filtered out
        assert "venue=kalshi" in kalshi_only.stdout

    def test_clv_command_accepts_price_bucket(self, tmp_path, _patch_db):
        _patch_db(_make_db(tmp_path, _two_bucket_rows()))
        result = runner.invoke(app, ["clv", "nfl", "--price-bucket", "10-20"])
        assert result.exit_code == 0, result.stdout
        assert "n=4" in result.stdout and "+3.00pp" in result.stdout
        assert "price=10-20" in result.stdout


class TestPromotionBoardPriceBucket:
    def test_board_filters_every_column_to_the_bucket(self, tmp_path, _patch_db):
        from evmax.agents.cleanup.promotion_board import compute_promotion_board

        _patch_db(_make_db(tmp_path, _two_bucket_rows()))
        all_rows = compute_promotion_board(sector="nfl", staleness_h=None)
        dog_rows = compute_promotion_board(sector="nfl", staleness_h=None, price_bucket="10-20")
        assert len(all_rows) == 1 and len(dog_rows) == 1
        # CLV gate values come from the bucket only.
        assert dog_rows[0]["gates"]["clv_n"]["value"] == 4
        assert dog_rows[0]["gates"]["clv_mean"]["value"] == pytest.approx(3.0)
        assert all_rows[0]["gates"]["clv_n"]["value"] == 7

    def test_board_rejects_bad_bucket(self, tmp_path, _patch_db):
        from evmax.agents.cleanup.promotion_board import compute_promotion_board

        _patch_db(_make_db(tmp_path, []))
        with pytest.raises(ValueError):
            compute_promotion_board(sector="nfl", staleness_h=None, price_bucket="x")

    def test_web_endpoint_400s_on_bad_bucket(self, tmp_path, _patch_db):
        from fastapi.testclient import TestClient

        from evmax.web.app import app as web_app

        _patch_db(_make_db(tmp_path, []))
        client = TestClient(web_app)
        bad = client.get("/api/promotion-board?price_bucket=nope")
        assert bad.status_code == 400
        ok = client.get("/api/promotion-board?price_bucket=10-20&staleness_h=0")
        assert ok.status_code == 200 and "rows" in ok.json()


# ---------------------------------------------------------------------------
# replay harness: require_sized
# ---------------------------------------------------------------------------


def _make_harness_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "harness.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE ev_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_date TEXT, market_id TEXT, event_id TEXT, event_date TEXT,
            sector TEXT, market_type TEXT, line REAL, model_sources TEXT,
            mode TEXT, venue TEXT DEFAULT 'kalshi', voided INTEGER DEFAULT 0,
            kelly_fraction REAL DEFAULT 0, kalshi_yes_price REAL,
            blended_true_prob REAL, ev_pct REAL
        );
        CREATE TABLE ev_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT, market_id TEXT UNIQUE, outcome INTEGER
        );
        """
    )
    today = date.today().isoformat()
    for mid, mode, kelly in (("live1", "live", 0.02), ("sh1", "shadow", 0.0), ("sh2", "shadow", 0.0)):
        conn.execute(
            """INSERT INTO ev_predictions
               (scan_date, market_id, event_id, event_date, sector, market_type,
                model_sources, mode, kelly_fraction, kalshi_yes_price, blended_true_prob, ev_pct)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (today, mid, f"nfl::{today}::{mid}", today, "nfl", "moneyline",
             "nfl_efficiency,elo,sharp", mode, kelly, 0.40, 0.45, 0.05),
        )
        conn.execute("INSERT INTO ev_outcomes (market_id, outcome) VALUES (?, 1)", (mid,))
    conn.commit()
    conn.close()
    return db_path


class TestLoadResolvedRowsRequireSized:
    def test_default_drops_unsized_shadow_rows(self, tmp_path):
        from evmax.backtest.sizing import load_resolved_rows

        db = _make_harness_db(tmp_path)
        live = load_resolved_rows(db, modes=("live",), exclude_contaminated=False)
        shadow_default = load_resolved_rows(db, modes=("shadow",), exclude_contaminated=False)
        shadow_all = load_resolved_rows(
            db, modes=("shadow",), exclude_contaminated=False, require_sized=False
        )
        assert [r.market_id for r in live] == ["live1"]
        # The trap this knob exists for: shadow rows carry Kelly 0 → default loads none.
        assert shadow_default == []
        assert sorted(r.market_id for r in shadow_all) == ["sh1", "sh2"]
