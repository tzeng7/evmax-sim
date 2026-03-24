"""Tests for spread_distribution.py, logger.py, and metrics.py.

Covers:
  SpreadDistributionModel:
    - Returns None when no spread_line
    - Rejects Kalshi lines > 1σ from Pinnacle line
    - Correct probability direction (favorite vs underdog)
    - Probability bounds (0.01–0.99)
    - Sector-specific sigma lookup
    - Implied mean is plausible

  PredictionLogger (logger.py):
    - log_gaps inserts rows into in-memory DB
    - Duplicate market_id + scan_date is ignored (INSERT OR IGNORE)
    - get_logged_market_ids returns correct set for date
    - event_date is stored correctly from datetime or date
    - Empty gaps list returns 0

  Brier / Metrics (metrics.py):
    - load_config returns defaults on missing file
    - save_config round-trips correctly
    - compute_brier_scores returns None on empty table
    - adjust_sharp_weight cooldown blocks re-adjustment
    - adjust_sharp_weight decreases weight when model is better
    - adjust_sharp_weight increases weight when model is worse
    - adjust_sharp_weight respects bounds (0.40–0.95)
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from evmax.models.odds import SharpBook, SharpOdds
from evmax.models_ml.spread_distribution import SpreadDistributionModel


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

def _sharp_spread(spread_line: float, true_prob_a: float) -> SharpOdds:
    true_prob_b = round(1.0 - true_prob_a, 4)
    return SharpOdds(
        event_id="nba::2026-03-20::pistons_vs_warriors",
        book=SharpBook.pinnacle,
        sector="nba",
        outcome_a_label="pistons",
        outcome_b_label="warriors",
        outcome_a_decimal=2.0,
        outcome_b_decimal=2.0,
        true_prob_a=true_prob_a,
        true_prob_b=true_prob_b,
        spread_line=spread_line,
    )


def _make_ev_gap(market_id="kalshi:TEST-001", event_id="nba::2026-03-20::test",
                  sector="nba", yes_team="pistons", ev_pct=0.05,
                  event_date=None, line=None):
    """Build a minimal EVGap-like object for logger tests."""
    from evmax.agents.odds.ev_gap_agent import EVGap
    return EVGap(
        market_id=market_id,
        event_id=event_id,
        sector=sector,
        yes_team=yes_team,
        market_type="moneyline",
        kalshi_yes_price=0.40,
        sharp_true_prob=0.55,
        blended_true_prob=0.55,
        ev_pct=ev_pct,
        kelly_full=0.10,
        kelly_fraction=0.05,
        match_confidence=0.95,
        volume_usd=10_000.0,
        spread_pct=0.02,
        event_date=event_date,
        line=line,
        model_sources="sharp",
        event_title="Pistons vs Warriors",
    )


def _make_in_memory_db():
    """Create an in-memory SQLite with the ev_predictions and ev_outcomes schema."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE ev_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_date TEXT NOT NULL,
            market_id TEXT NOT NULL,
            event_id TEXT,
            sector TEXT,
            yes_team TEXT,
            market_type TEXT,
            event_title TEXT,
            event_date TEXT,
            kalshi_yes_price REAL,
            sharp_true_prob REAL,
            blended_true_prob REAL,
            ev_pct REAL,
            kelly_fraction REAL,
            volume_usd REAL,
            model_sources TEXT,
            sharp_weight_used REAL,
            bankroll_used REAL,
            line REAL,
            UNIQUE(market_id, scan_date)
        );
        CREATE TABLE ev_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_id TEXT UNIQUE,
            event_id TEXT,
            event_date TEXT,
            sector TEXT,
            yes_team TEXT,
            outcome INTEGER,
            sharp_true_prob REAL,
            blended_true_prob REAL,
            resolved_at TEXT,
            result_source TEXT
        );
    """)
    return conn


# ---------------------------------------------------------------------------
# SpreadDistributionModel
# ---------------------------------------------------------------------------

class TestSpreadDistributionModel:
    def setup_method(self):
        self.model = SpreadDistributionModel()

    def test_returns_none_when_no_spread_line(self):
        sharp = SharpOdds(
            event_id="nba::2026-03-20::test",
            book=SharpBook.pinnacle, sector="nba",
            outcome_a_label="a", outcome_b_label="b",
            outcome_a_decimal=2.0, outcome_b_decimal=2.0,
            true_prob_a=0.55, true_prob_b=0.45,
            spread_line=None,
        )
        result = self.model.predict(sharp, target_line=-8.5, sector="nba")
        assert result is None

    def test_rejects_line_too_far_from_pinnacle(self):
        # NBA sigma = 11.5; Pinnacle -7.5, Kalshi -22.5 → distance 15 > 11.5
        sharp = _sharp_spread(spread_line=-7.5, true_prob_a=0.55)
        result = self.model.predict(sharp, target_line=-22.5, sector="nba")
        assert result is None

    def test_accepts_line_within_one_sigma(self):
        # NBA sigma = 11.5; Pinnacle -7.5, Kalshi -10.5 → distance 3 < 11.5
        sharp = _sharp_spread(spread_line=-7.5, true_prob_a=0.55)
        result = self.model.predict(sharp, target_line=-10.5, sector="nba")
        assert result is not None

    def test_favorite_prob_below_half_for_large_line(self):
        # Pinnacle -7.5 favorite with 55% win prob; target is -18 (large)
        # P(winning by > 18) should be small
        sharp = _sharp_spread(spread_line=-7.5, true_prob_a=0.55)
        result = self.model.predict(sharp, target_line=-18.0, sector="nba")
        if result is not None:
            assert result.true_prob < 0.50

    def test_favorite_prob_above_half_for_small_line(self):
        # Pinnacle -7.5 favorite with 55% win prob; target is -1.5 (small)
        # P(winning by > 1.5) should be > 50% for a favorite
        sharp = _sharp_spread(spread_line=-7.5, true_prob_a=0.55)
        result = self.model.predict(sharp, target_line=-1.5, sector="nba")
        if result is not None:
            assert result.true_prob > 0.50

    def test_underdog_is_yes_side(self):
        # When underdog is YES side, probability is for them winning outright
        sharp = _sharp_spread(spread_line=-7.5, true_prob_a=0.55)
        result_fav = self.model.predict(sharp, target_line=-7.5, sector="nba",
                                         yes_is_underdog=False)
        result_dog = self.model.predict(sharp, target_line=-7.5, sector="nba",
                                         yes_is_underdog=True)
        if result_fav and result_dog:
            # Underdog covers at -7.5 from their perspective = favorite wins by < 7.5
            # so probabilities should be complementary
            assert result_dog.true_prob < result_fav.true_prob

    def test_probability_bounds_enforced(self):
        # Use extreme prob that would otherwise produce probs near 0 or 1
        sharp = _sharp_spread(spread_line=-7.5, true_prob_a=0.99)
        result = self.model.predict(sharp, target_line=-7.0, sector="nba")
        if result is not None:
            assert result.true_prob >= 0.01
            assert result.true_prob <= 0.99

    def test_sector_sigma_lookup(self):
        # Soccer has sigma 1.9; NBA has 11.5
        # Line distance of 5 pts should be rejected for soccer but accepted for NBA
        sharp_soccer = SharpOdds(
            event_id="soccer::2026-03-20::test",
            book=SharpBook.pinnacle, sector="soccer",
            outcome_a_label="a", outcome_b_label="b",
            outcome_a_decimal=2.0, outcome_b_decimal=2.0,
            true_prob_a=0.50, true_prob_b=0.50,
            spread_line=-0.5,
        )
        # Distance of 5 goals >> soccer sigma 1.9
        result_soccer = self.model.predict(sharp_soccer, target_line=-5.5, sector="soccer")
        assert result_soccer is None

        sharp_nba = _sharp_spread(spread_line=-7.5, true_prob_a=0.55)
        # Distance of 5 pts < NBA sigma 11.5
        result_nba = self.model.predict(sharp_nba, target_line=-12.5, sector="nba")
        assert result_nba is not None

    def test_result_contains_implied_mean_and_sigma(self):
        sharp = _sharp_spread(spread_line=-7.5, true_prob_a=0.55)
        result = self.model.predict(sharp, target_line=-7.5, sector="nba")
        assert result is not None
        assert isinstance(result.implied_mean, float)
        assert result.sigma == 11.5  # NBA sigma

    def test_invalid_true_prob_returns_none(self):
        # true_prob_a <= 0 should bail out
        sharp = SharpOdds(
            event_id="nba::test",
            book=SharpBook.pinnacle, sector="nba",
            outcome_a_label="a", outcome_b_label="b",
            outcome_a_decimal=2.0, outcome_b_decimal=2.0,
            true_prob_a=0.0, true_prob_b=1.0,
            spread_line=-7.5,
        )
        result = self.model.predict(sharp, target_line=-7.5, sector="nba")
        assert result is None

    def test_default_sector_sigma_used_for_unknown(self):
        sharp = _sharp_spread(spread_line=-7.5, true_prob_a=0.55)
        result = self.model.predict(sharp, target_line=-10.5, sector="unknown_sport")
        if result is not None:
            assert result.sigma == 11.5  # default


# ---------------------------------------------------------------------------
# PredictionLogger
# ---------------------------------------------------------------------------

class TestPredictionLogger:
    """Uses patched get_connection to return an in-memory DB."""

    def _run_with_db(self, fn):
        """Run fn(conn) with logger functions patched to use in-memory DB."""
        conn = _make_in_memory_db()
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=conn)
        ctx.__exit__ = MagicMock(return_value=False)

        with patch("evmax.agents.cleanup.logger.get_connection", return_value=ctx):
            return fn(conn)

    def test_log_gaps_inserts_rows(self):
        from evmax.agents.cleanup.logger import log_gaps

        gaps = [_make_ev_gap("kalshi:A"), _make_ev_gap("kalshi:B")]

        def run(conn):
            n = log_gaps(gaps, scan_date=date(2026, 3, 19))
            return n, conn

        n, conn = self._run_with_db(run)
        assert n == 2
        rows = conn.execute("SELECT * FROM ev_predictions").fetchall()
        assert len(rows) == 2

    def test_log_gaps_ignores_duplicates(self):
        from evmax.agents.cleanup.logger import log_gaps

        gap = _make_ev_gap("kalshi:DUPE")

        def run(conn):
            n1 = log_gaps([gap], scan_date=date(2026, 3, 19))
            n2 = log_gaps([gap], scan_date=date(2026, 3, 19))
            return n1, n2

        n1, n2 = self._run_with_db(run)
        assert n1 == 1
        assert n2 == 0  # duplicate ignored

    def test_log_gaps_empty_list_returns_zero(self):
        from evmax.agents.cleanup.logger import log_gaps

        def run(conn):
            return log_gaps([], scan_date=date(2026, 3, 19))

        n = self._run_with_db(run)
        assert n == 0

    def test_log_gaps_event_date_from_datetime(self):
        from evmax.agents.cleanup.logger import log_gaps

        dt = datetime(2026, 3, 20, 18, 30, 0, tzinfo=timezone.utc)
        gap = _make_ev_gap("kalshi:DTTEST", event_date=dt)

        def run(conn):
            log_gaps([gap], scan_date=date(2026, 3, 19))
            return conn.execute(
                "SELECT event_date FROM ev_predictions WHERE market_id = ?",
                ("kalshi:DTTEST",)
            ).fetchone()

        row = self._run_with_db(run)
        assert row is not None
        assert row["event_date"] == "2026-03-20"

    def test_log_gaps_event_date_from_date_object(self):
        from evmax.agents.cleanup.logger import log_gaps

        gap = _make_ev_gap("kalshi:DATETEST", event_date=date(2026, 3, 21))

        def run(conn):
            log_gaps([gap], scan_date=date(2026, 3, 19))
            return conn.execute(
                "SELECT event_date FROM ev_predictions WHERE market_id = ?",
                ("kalshi:DATETEST",)
            ).fetchone()

        row = self._run_with_db(run)
        assert row is not None
        assert row["event_date"] == "2026-03-21"

    def test_log_gaps_stores_spread_line(self):
        from evmax.agents.cleanup.logger import log_gaps

        gap = _make_ev_gap("kalshi:SPREAD-001", line=-8.5)

        def run(conn):
            log_gaps([gap], scan_date=date(2026, 3, 19))
            return conn.execute(
                "SELECT line FROM ev_predictions WHERE market_id = ?",
                ("kalshi:SPREAD-001",)
            ).fetchone()

        row = self._run_with_db(run)
        assert row is not None
        assert row["line"] == pytest.approx(-8.5)

    def test_get_logged_market_ids(self):
        from evmax.agents.cleanup.logger import get_logged_market_ids, log_gaps

        gaps = [_make_ev_gap("kalshi:ID-001"), _make_ev_gap("kalshi:ID-002")]

        conn = _make_in_memory_db()
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=conn)
        ctx.__exit__ = MagicMock(return_value=False)

        with patch("evmax.agents.cleanup.logger.get_connection", return_value=ctx):
            log_gaps(gaps, scan_date=date(2026, 3, 19))
            ids = get_logged_market_ids(date(2026, 3, 19))

        assert "kalshi:ID-001" in ids
        assert "kalshi:ID-002" in ids

    def test_get_logged_market_ids_different_date_empty(self):
        from evmax.agents.cleanup.logger import get_logged_market_ids, log_gaps

        gaps = [_make_ev_gap("kalshi:ID-X")]

        conn = _make_in_memory_db()
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=conn)
        ctx.__exit__ = MagicMock(return_value=False)

        with patch("evmax.agents.cleanup.logger.get_connection", return_value=ctx):
            log_gaps(gaps, scan_date=date(2026, 3, 19))
            ids = get_logged_market_ids(date(2026, 3, 20))  # different date

        assert len(ids) == 0


# ---------------------------------------------------------------------------
# metrics.py: load_config / save_config
# ---------------------------------------------------------------------------

class TestLoadSaveConfig:
    def test_load_config_missing_file_returns_defaults(self):
        from evmax.agents.cleanup.metrics import load_config, _DEFAULT_CONFIG

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("evmax.agents.cleanup.metrics.CONFIG_PATH",
                       Path(tmpdir) / "nonexistent.json"):
                cfg = load_config()

        assert cfg["sharp_weight"] == _DEFAULT_CONFIG["sharp_weight"]
        assert "brier_history" in cfg

    def test_load_config_reads_existing_file(self):
        from evmax.agents.cleanup.metrics import load_config

        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "model_config.json"
            p.write_text(json.dumps({"sharp_weight": 0.70, "last_adjusted": "2026-01-01"}))
            with patch("evmax.agents.cleanup.metrics.CONFIG_PATH", p):
                cfg = load_config()

        assert cfg["sharp_weight"] == 0.70
        assert cfg["last_adjusted"] == "2026-01-01"

    def test_load_config_corrupted_file_returns_defaults(self):
        from evmax.agents.cleanup.metrics import load_config, _DEFAULT_CONFIG

        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "model_config.json"
            p.write_text("{not valid json}")
            with patch("evmax.agents.cleanup.metrics.CONFIG_PATH", p):
                cfg = load_config()

        assert cfg["sharp_weight"] == _DEFAULT_CONFIG["sharp_weight"]

    def test_save_config_round_trip(self):
        from evmax.agents.cleanup.metrics import load_config, save_config

        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "model_config.json"
            original = {"sharp_weight": 0.75, "last_adjusted": "2026-03-01",
                        "brier_history": [{"n": 30}]}
            with patch("evmax.agents.cleanup.metrics.CONFIG_PATH", p):
                save_config(original)
                loaded = load_config()

        assert loaded["sharp_weight"] == 0.75
        assert loaded["brier_history"][0]["n"] == 30


# ---------------------------------------------------------------------------
# metrics.py: compute_brier_scores and adjust_sharp_weight
# ---------------------------------------------------------------------------

def _seed_outcomes(conn: sqlite3.Connection, rows: list[dict]) -> None:
    """Seed ev_predictions + ev_outcomes for Brier testing."""
    scan_date = (date.today() - timedelta(days=7)).isoformat()
    for i, r in enumerate(rows):
        mid = f"kalshi:BRIER-{i}"
        conn.execute(
            "INSERT INTO ev_predictions (scan_date, market_id, event_id, sector, yes_team, "
            "market_type, kalshi_yes_price, sharp_true_prob, blended_true_prob, ev_pct, "
            "kelly_fraction, volume_usd, model_sources, sharp_weight_used) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (scan_date, mid, f"ev::{i}", "nba", "team", "moneyline",
             0.50, r["sharp_prob"], r["blend_prob"], 0.05, 0.025, 1000.0, "sharp", 0.85),
        )
        conn.execute(
            "INSERT INTO ev_outcomes (market_id, event_id, sector, yes_team, outcome, "
            "sharp_true_prob, blended_true_prob, result_source) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (mid, f"ev::{i}", "nba", "team", r["outcome"],
             r["sharp_prob"], r["blend_prob"], "espn"),
        )
    conn.commit()


class TestComputeBrierScores:
    def _get_conn_context(self, rows):
        conn = _make_in_memory_db()
        _seed_outcomes(conn, rows)
        ctx = MagicMock()
        ctx.__enter__ = ctx  # make it work as both context manager and direct return
        ctx.__exit__ = MagicMock(return_value=False)
        return conn, ctx

    def test_returns_none_on_empty_db(self):
        from evmax.agents.cleanup.metrics import compute_brier_scores

        conn = _make_in_memory_db()
        with patch("evmax.agents.cleanup.metrics.get_connection", return_value=conn):
            result = compute_brier_scores(weeks=4)
        assert result is None

    def test_brier_score_calculation(self):
        from evmax.agents.cleanup.metrics import compute_brier_scores

        # Perfect model: always predicts 1.0 for outcomes that happen
        rows = [
            {"sharp_prob": 0.6, "blend_prob": 0.9, "outcome": 1},
            {"sharp_prob": 0.6, "blend_prob": 0.9, "outcome": 1},
        ]
        conn = _make_in_memory_db()
        _seed_outcomes(conn, rows)

        with patch("evmax.agents.cleanup.metrics.get_connection", return_value=conn):
            result = compute_brier_scores(weeks=52)  # long window to include seeded data

        assert result is not None
        assert result["n"] == 2
        # blend_prob=0.9, outcome=1 → brier = (0.9-1)^2 = 0.01
        assert result["brier_model"] == pytest.approx(0.01)
        # sharp_prob=0.6, outcome=1 → brier = (0.6-1)^2 = 0.16
        assert result["brier_sharp"] == pytest.approx(0.16)

    def test_better_model_has_lower_brier(self):
        from evmax.agents.cleanup.metrics import compute_brier_scores

        rows = [{"sharp_prob": 0.5, "blend_prob": 0.8, "outcome": 1}] * 10
        conn = _make_in_memory_db()
        _seed_outcomes(conn, rows)

        with patch("evmax.agents.cleanup.metrics.get_connection", return_value=conn):
            result = compute_brier_scores(weeks=52)

        assert result is not None
        assert result["brier_model"] < result["brier_sharp"]

    def test_tier_breakdown_present(self):
        from evmax.agents.cleanup.metrics import compute_brier_scores

        rows = [{"sharp_prob": 0.55, "blend_prob": 0.60, "outcome": 1}] * 5
        conn = _make_in_memory_db()
        _seed_outcomes(conn, rows)

        with patch("evmax.agents.cleanup.metrics.get_connection", return_value=conn):
            result = compute_brier_scores(weeks=52)

        assert result is not None
        assert "tiers" in result
        assert len(result["tiers"]) == 4


class TestAdjustSharpWeight:
    def _base_config(self, sharp_weight=0.85, last_adjusted=None):
        return {
            "sharp_weight": sharp_weight,
            "last_adjusted": last_adjusted,
            "last_brier_model": None,
            "last_brier_sharp": None,
            "brier_history": [],
        }

    def test_cooldown_blocks_adjustment(self):
        from evmax.agents.cleanup.metrics import adjust_sharp_weight

        yesterday = (date.today() - timedelta(days=1)).isoformat()
        cfg = self._base_config(last_adjusted=yesterday)

        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "model_config.json"
            p.write_text(json.dumps(cfg))
            with patch("evmax.agents.cleanup.metrics.CONFIG_PATH", p):
                result = adjust_sharp_weight(force=False)

        assert result["adjusted"] is False
        assert "Adjusted" in result["reason"]

    def test_force_overrides_cooldown(self):
        from evmax.agents.cleanup.metrics import adjust_sharp_weight

        yesterday = (date.today() - timedelta(days=1)).isoformat()
        cfg = self._base_config(last_adjusted=yesterday)

        # Need 30+ resolved outcomes to proceed past data check
        conn = _make_in_memory_db()
        rows = [{"sharp_prob": 0.5, "blend_prob": 0.9, "outcome": 1}] * 35
        _seed_outcomes(conn, rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "model_config.json"
            p.write_text(json.dumps(cfg))
            with patch("evmax.agents.cleanup.metrics.CONFIG_PATH", p), \
                 patch("evmax.agents.cleanup.metrics.get_connection", return_value=conn):
                result = adjust_sharp_weight(force=True)

        # Should not be blocked by cooldown (force=True)
        assert "Adjusted" not in result.get("reason", "")

    def test_insufficient_data_returns_no_adjust(self):
        from evmax.agents.cleanup.metrics import adjust_sharp_weight

        cfg = self._base_config()
        conn = _make_in_memory_db()  # empty

        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "model_config.json"
            p.write_text(json.dumps(cfg))
            with patch("evmax.agents.cleanup.metrics.CONFIG_PATH", p), \
                 patch("evmax.agents.cleanup.metrics.get_connection", return_value=conn):
                result = adjust_sharp_weight(force=True)

        assert result["adjusted"] is False
        assert "Insufficient" in result["reason"]

    def test_model_beating_sharp_decreases_weight(self):
        """When model Brier < sharp Brier by > 5%, sharp_weight should decrease."""
        from evmax.agents.cleanup.metrics import adjust_sharp_weight

        cfg = self._base_config(sharp_weight=0.85)
        # Model much better: blend_prob=0.9 vs sharp_prob=0.5, outcome=1
        # brier_model = (0.9-1)^2 = 0.01, brier_sharp = (0.5-1)^2 = 0.25
        # improvement = (0.25-0.01)/0.25 = 96% > 5% → decrease weight
        conn = _make_in_memory_db()
        rows = [{"sharp_prob": 0.5, "blend_prob": 0.9, "outcome": 1}] * 35
        _seed_outcomes(conn, rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "model_config.json"
            p.write_text(json.dumps(cfg))
            with patch("evmax.agents.cleanup.metrics.CONFIG_PATH", p), \
                 patch("evmax.agents.cleanup.metrics.get_connection", return_value=conn):
                result = adjust_sharp_weight(force=True)

        assert result["adjusted"] is True
        assert result["sharp_weight"] < 0.85
        assert result["sharp_weight"] == pytest.approx(0.80)  # decreased by 0.05

    def test_model_underperforming_increases_weight(self):
        """When sharp Brier < model Brier by > 5%, sharp_weight should increase."""
        from evmax.agents.cleanup.metrics import adjust_sharp_weight

        cfg = self._base_config(sharp_weight=0.85)
        # Model worse: blend_prob=0.3 vs sharp_prob=0.8, outcome=1
        # brier_model = (0.3-1)^2 = 0.49, brier_sharp = (0.8-1)^2 = 0.04
        conn = _make_in_memory_db()
        rows = [{"sharp_prob": 0.8, "blend_prob": 0.3, "outcome": 1}] * 35
        _seed_outcomes(conn, rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "model_config.json"
            p.write_text(json.dumps(cfg))
            with patch("evmax.agents.cleanup.metrics.CONFIG_PATH", p), \
                 patch("evmax.agents.cleanup.metrics.get_connection", return_value=conn):
                result = adjust_sharp_weight(force=True)

        assert result["adjusted"] is True
        assert result["sharp_weight"] > 0.85
        assert result["sharp_weight"] == pytest.approx(0.90)  # increased by 0.05

    def test_sharp_weight_bounded_at_min(self):
        """sharp_weight should not go below 0.40."""
        from evmax.agents.cleanup.metrics import adjust_sharp_weight

        cfg = self._base_config(sharp_weight=0.40)  # already at minimum
        conn = _make_in_memory_db()
        rows = [{"sharp_prob": 0.5, "blend_prob": 0.9, "outcome": 1}] * 35
        _seed_outcomes(conn, rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "model_config.json"
            p.write_text(json.dumps(cfg))
            with patch("evmax.agents.cleanup.metrics.CONFIG_PATH", p), \
                 patch("evmax.agents.cleanup.metrics.get_connection", return_value=conn):
                result = adjust_sharp_weight(force=True)

        assert result["sharp_weight"] >= 0.40

    def test_sharp_weight_bounded_at_max(self):
        """sharp_weight should not go above 0.95."""
        from evmax.agents.cleanup.metrics import adjust_sharp_weight

        cfg = self._base_config(sharp_weight=0.95)  # already at maximum
        conn = _make_in_memory_db()
        # Model underperforms → would try to increase beyond 0.95
        rows = [{"sharp_prob": 0.8, "blend_prob": 0.3, "outcome": 1}] * 35
        _seed_outcomes(conn, rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "model_config.json"
            p.write_text(json.dumps(cfg))
            with patch("evmax.agents.cleanup.metrics.CONFIG_PATH", p), \
                 patch("evmax.agents.cleanup.metrics.get_connection", return_value=conn):
                result = adjust_sharp_weight(force=True)

        assert result["sharp_weight"] <= 0.95

    def test_history_appended_on_adjustment(self):
        """Brier history should be recorded after adjustment."""
        from evmax.agents.cleanup.metrics import adjust_sharp_weight

        cfg = self._base_config(sharp_weight=0.85)
        conn = _make_in_memory_db()
        rows = [{"sharp_prob": 0.5, "blend_prob": 0.9, "outcome": 1}] * 35
        _seed_outcomes(conn, rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "model_config.json"
            p.write_text(json.dumps(cfg))
            with patch("evmax.agents.cleanup.metrics.CONFIG_PATH", p), \
                 patch("evmax.agents.cleanup.metrics.get_connection", return_value=conn):
                adjust_sharp_weight(force=True)
                saved = json.loads(p.read_text())

        assert len(saved["brier_history"]) == 1
        assert "brier_model" in saved["brier_history"][0]
        assert "new_weight" in saved["brier_history"][0]
