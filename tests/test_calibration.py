"""Tests for ModelCalibrator (evmax/agents/models/calibration.py).

Focus: the R1 fix — retrain_all_from_db must train the SAME per-sector key the
ensemble applies (``{sector}_ensemble``), dedup markets to their latest scan,
respect the per-sector minimum, and honor dry-run. The previous implementation
trained a single global ``"ensemble"`` key the ensemble never reads, so a refit
was a silent no-op.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from evmax.agents.models import calibration as cal_mod
from evmax.agents.models.calibration import ModelCalibrator

pytest.importorskip("sklearn")


def _make_db(path: Path, rows: list[dict]) -> None:
    """Build a minimal predictions.db with the columns retrain reads."""
    db = sqlite3.connect(str(path))
    db.execute(
        """CREATE TABLE ev_predictions (
               market_id TEXT, scan_date TEXT, sector TEXT,
               blended_true_prob REAL, mode TEXT, voided INTEGER)"""
    )
    db.execute(
        "CREATE TABLE ev_outcomes (market_id TEXT, outcome INTEGER)"
    )
    for r in rows:
        db.execute(
            "INSERT INTO ev_predictions VALUES (?,?,?,?,?,?)",
            (r["market_id"], r["scan_date"], r["sector"], r["prob"],
             r.get("mode", "live"), r.get("voided", 0)),
        )
    # one outcome per distinct market_id
    seen = set()
    for r in rows:
        if r["market_id"] in seen:
            continue
        seen.add(r["market_id"])
        db.execute("INSERT INTO ev_outcomes VALUES (?,?)", (r["market_id"], r["outcome"]))
    db.commit()
    db.close()


@pytest.fixture
def calibrator(tmp_path, monkeypatch):
    """A calibrator whose calibration.json is redirected into tmp_path."""
    cal_path = tmp_path / "calibration.json"
    monkeypatch.setattr(cal_mod, "CALIBRATION_PATH", cal_path)
    return ModelCalibrator()


def _sector_rows(sector: str, n: int, prob: float, outcome: int, start_mkt=0):
    return [
        {"market_id": f"{sector}-{i}", "scan_date": "2026-08-01",
         "sector": sector, "prob": prob, "outcome": outcome}
        for i in range(start_mkt, start_mkt + n)
    ]


class TestFit:
    def test_fit_none_below_min_samples(self, calibrator):
        assert calibrator._fit([0.5] * (cal_mod.MIN_SAMPLES - 1), [1] * (cal_mod.MIN_SAMPLES - 1)) is None

    def test_fit_returns_curve(self, calibrator):
        probs = [i / 100 for i in range(100)]
        outs = [1 if p > 0.5 else 0 for p in probs]
        fit = calibrator._fit(probs, outs)
        assert fit is not None
        assert fit["n_samples"] == 100
        assert len(fit["x"]) == len(fit["y"]) >= 2
        assert fit["brier_after"] <= fit["brier_before"]  # isotonic never worsens in-sample


class TestRetrainKey:
    def test_retrain_stores_under_given_key(self, calibrator):
        ok = calibrator.retrain("nba_ensemble", [0.5] * 40, [1, 0] * 20)
        assert ok is True
        # the applied-side key is present, ready for _apply_sector_calibration
        assert "nba_ensemble" in calibrator._calibrations


class TestRetrainAllFromDb:
    def test_trains_per_sector_ensemble_key(self, tmp_path, calibrator):
        db = tmp_path / "predictions.db"
        # 60 nba rows split so the isotonic has signal, 60 soccer rows
        rows = (_sector_rows("nba", 30, 0.8, 1) + _sector_rows("nba", 30, 0.2, 0, start_mkt=30)
                + _sector_rows("soccer", 30, 0.7, 1, start_mkt=100)
                + _sector_rows("soccer", 30, 0.3, 0, start_mkt=130))
        _make_db(db, rows)

        report = calibrator.retrain_all_from_db(min_per_sector=50, db_path=db)

        # keys are the ones the ensemble reads — NOT a global "ensemble"
        assert "nba_ensemble" in report
        assert "soccer_ensemble" in report
        assert "ensemble" not in report
        assert report["nba_ensemble"]["updated"] is True
        assert calibrator._calibrations["nba_ensemble"]["n_samples"] == 60

    def test_respects_min_per_sector(self, tmp_path, calibrator):
        db = tmp_path / "predictions.db"
        _make_db(db, _sector_rows("nhl", 40, 0.6, 1))  # only 40 rows
        report = calibrator.retrain_all_from_db(min_per_sector=50, db_path=db)
        assert report["nhl_ensemble"]["updated"] is False
        assert "nhl_ensemble" not in calibrator._calibrations

    def test_dry_run_does_not_persist(self, tmp_path, calibrator):
        db = tmp_path / "predictions.db"
        rows = _sector_rows("nba", 30, 0.8, 1) + _sector_rows("nba", 30, 0.2, 0, start_mkt=30)
        _make_db(db, rows)
        report = calibrator.retrain_all_from_db(min_per_sector=50, db_path=db, dry_run=True)
        assert report["nba_ensemble"]["updated"] is False
        assert report["nba_ensemble"]["brier_before"] is not None  # still reported
        assert "nba_ensemble" not in calibrator._calibrations       # but not written

    def test_dedups_to_latest_scan(self, tmp_path, calibrator):
        db = tmp_path / "predictions.db"
        # same 60 markets scanned on two dates -> must count 60, not 120
        rows = _sector_rows("nba", 30, 0.8, 1) + _sector_rows("nba", 30, 0.2, 0, start_mkt=30)
        dup = [dict(r, scan_date="2026-07-31") for r in rows]  # earlier duplicate scan
        _make_db(db, rows + dup)
        report = calibrator.retrain_all_from_db(min_per_sector=50, db_path=db)
        assert report["nba_ensemble"]["n"] == 60

    def test_excludes_shadow_and_void(self, tmp_path, calibrator):
        db = tmp_path / "predictions.db"
        live = _sector_rows("nba", 30, 0.8, 1) + _sector_rows("nba", 30, 0.2, 0, start_mkt=30)
        shadow = [dict(r, market_id=f"sh-{r['market_id']}", mode="shadow")
                  for r in _sector_rows("nba", 40, 0.5, 1, start_mkt=200)]
        _make_db(db, live + shadow)
        report = calibrator.retrain_all_from_db(min_per_sector=50, db_path=db)
        assert report["nba_ensemble"]["n"] == 60  # shadow rows excluded

    def test_sector_filter(self, tmp_path, calibrator):
        db = tmp_path / "predictions.db"
        rows = (_sector_rows("nba", 60, 0.8, 1)
                + _sector_rows("soccer", 60, 0.7, 1, start_mkt=100))
        _make_db(db, rows)
        report = calibrator.retrain_all_from_db(min_per_sector=50, db_path=db, sector="nba")
        assert set(report.keys()) == {"nba_ensemble"}

    def test_missing_db_returns_empty(self, tmp_path, calibrator):
        assert calibrator.retrain_all_from_db(db_path=tmp_path / "nope.db") == {}
