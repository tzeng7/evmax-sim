"""Tests for the PIT tennis-calibration harness (scripts/fit_tennis_calibration.py).

Covers:
  - model_consensus replicates EnsembleModelAgent._blend step 1 exactly
    (confidence gate boundary, eff-weight math, empty case) — verified against
    the real _blend, not a re-derivation
  - fit/apply isotonic round-trips through ModelCalibrator.calibrate semantics
  - chronological split, both-sides doubling, deterministic bootstrap
  - run() gate behavior: no write on gate fail; no write under --report-only
    even on a gate pass
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import fit_tennis_calibration as ftc  # noqa: E402

from evmax.agents.models.base import ModelAgentPrediction  # noqa: E402
from evmax.agents.models.ensemble_agent import EnsembleModelAgent  # noqa: E402


# ---------------------------------------------------------------------------
# model_consensus fidelity vs the real ensemble
# ---------------------------------------------------------------------------

class _IdentityCalibrator:
    def calibrate(self, name: str, prob: float) -> float:
        return prob


def _pred(name: str, prob_a: float, conf: float) -> ModelAgentPrediction:
    return ModelAgentPrediction(
        event_id="e1", model_name=name,
        true_prob_a=prob_a, true_prob_b=1.0 - prob_a,
        confidence=conf, weight=0.99,  # class weight ≠ override to catch fallback misuse
    )


def _ensemble_model_only(preds: dict[str, ModelAgentPrediction]) -> float | None:
    ens = EnsembleModelAgent(models=[], sharp_weight=0.0)
    ens._calibrator = _IdentityCalibrator()
    blended = ens._blend("e1", preds, sharp=None, sharp_weight=0.0, sector="tennis")
    return None if blended is None else blended.true_prob_a


class TestModelConsensusFidelity:
    WEIGHTS = ftc.tennis_weights()

    def test_matches_real_blend_all_models_firing(self):
        mp = {
            "tennis_surface": [0.62, 0.61],
            "tennis_serve_return": [0.55, 0.78],
            "tennis_form": [0.58, 0.62],
            "tennis_advanced": [0.51, 0.56],
            "tennis_h2h": [0.70, 0.62],
        }
        preds = {k: _pred(k, p, c) for k, (p, c) in mp.items()}
        cons, names = ftc.model_consensus(mp, self.WEIGHTS)
        # BlendedPrediction rounds to 5dp — compare at that precision
        assert cons == pytest.approx(_ensemble_model_only(preds), abs=1e-5)
        assert names == sorted(mp)

    def test_matches_real_blend_with_gated_model(self):
        """A model at conf 0.30 (pre-fix surface artifact) must drop out of both."""
        mp = {
            "tennis_surface": [0.62, 0.30],
            "tennis_form": [0.58, 0.62],
        }
        preds = {k: _pred(k, p, c) for k, (p, c) in mp.items()}
        cons, names = ftc.model_consensus(mp, self.WEIGHTS)
        assert names == ["tennis_form"]
        assert cons == pytest.approx(_ensemble_model_only(preds))
        assert cons == pytest.approx(0.58)

    def test_gate_boundary_is_strictly_greater(self):
        """Production: `if pred.confidence <= 0.45: continue` — 0.45 exactly is OUT."""
        cons, names = ftc.model_consensus({"tennis_form": [0.60, 0.45]}, self.WEIGHTS)
        assert cons is None and names == []
        cons, names = ftc.model_consensus({"tennis_form": [0.60, 0.46]}, self.WEIGHTS)
        assert cons == pytest.approx(0.60) and names == ["tennis_form"]

    def test_eff_weight_math(self):
        """Consensus = Σ(w·c·p) / Σ(w·c) with the shipped sector weights."""
        mp = {"tennis_surface": [0.70, 0.60], "tennis_form": [0.50, 0.50]}
        w = self.WEIGHTS
        e_s, e_f = w["tennis_surface"] * 0.60, w["tennis_form"] * 0.50
        expected = (e_s * 0.70 + e_f * 0.50) / (e_s + e_f)
        cons, _ = ftc.model_consensus(mp, w)
        assert cons == pytest.approx(expected)

    def test_unknown_model_ignored(self):
        """A name absent from the weights dict contributes nothing (no crash,
        no class-weight fallback — the PIT weights dict is exhaustive)."""
        cons, names = ftc.model_consensus({"not_a_model": [0.9, 0.9]}, self.WEIGHTS)
        assert cons is None and names == []
        cons, names = ftc.model_consensus(
            {"not_a_model": [0.9, 0.9], "tennis_form": [0.60, 0.62]}, self.WEIGHTS
        )
        assert cons == pytest.approx(0.60) and names == ["tennis_form"]


# ---------------------------------------------------------------------------
# Isotonic fit/apply
# ---------------------------------------------------------------------------

class TestIsotonic:
    def test_apply_matches_model_calibrator_semantics(self):
        pytest.importorskip("sklearn")
        from evmax.agents.models.calibration import ModelCalibrator

        probs = [0.3] * 50 + [0.6] * 50
        labels = [0] * 40 + [1] * 10 + [0] * 20 + [1] * 30
        x, y = ftc.fit_isotonic(probs, labels)

        cal = ModelCalibrator.__new__(ModelCalibrator)  # skip disk load
        cal._calibrations = {"k": {"x": x, "y": y}}
        for p in (0.0, 0.25, 0.3, 0.45, 0.6, 0.9, 1.0):
            assert ftc.apply_isotonic(x, y, p) == pytest.approx(cal.calibrate("k", p))

    def test_learns_the_direction_of_miscalibration(self):
        pytest.importorskip("sklearn")
        # Model says 0.6 but actual freq is 0.8 → calibrated 0.6 must move up
        probs = [0.6] * 100 + [0.4] * 100
        labels = [1] * 80 + [0] * 20 + [1] * 20 + [0] * 80
        x, y = ftc.fit_isotonic(probs, labels)
        assert ftc.apply_isotonic(x, y, 0.6) > 0.7
        assert ftc.apply_isotonic(x, y, 0.4) < 0.3

    def test_short_breakpoints_identity(self):
        assert ftc.apply_isotonic([], [], 0.42) == 0.42
        assert ftc.apply_isotonic([0.5], [0.6], 0.42) == 0.42


# ---------------------------------------------------------------------------
# Split / doubling / bootstrap helpers
# ---------------------------------------------------------------------------

def _row(out: float, date: str, p_mkt: float, prob: float, conf: float = 0.61) -> list:
    return [out, date, p_mkt, {"tennis_surface": [prob, conf]}]


class TestHelpers:
    def test_split_is_chronological_even_if_cache_unsorted(self):
        rows = [_row(1, "2025-03-01", 0.5, 0.5), _row(0, "2025-01-01", 0.5, 0.5),
                _row(1, "2025-02-01", 0.5, 0.5), _row(0, "2025-04-01", 0.5, 0.5)]
        train, hold = ftc.split_chronological(rows, 0.5)
        assert [r[1] for r in train] == ["2025-01-01", "2025-02-01"]
        assert [r[1] for r in hold] == ["2025-03-01", "2025-04-01"]

    def test_both_sides_doubles_and_balances(self):
        probs, labels = ftc.both_sides([(0.7, 1.0), (0.4, 0.0)])
        assert probs == [0.7, pytest.approx(0.3), 0.4, pytest.approx(0.6)]
        assert labels == [1, 0, 0, 1]
        assert sum(labels) * 2 == len(labels)  # balanced by construction

    def test_bootstrap_deterministic_and_signs(self):
        deltas = [0.01] * 200  # uniformly positive improvement
        lo1, hi1, fp1 = ftc.paired_bootstrap_delta(deltas, b=500)
        lo2, hi2, fp2 = ftc.paired_bootstrap_delta(deltas, b=500)
        assert (lo1, hi1, fp1) == (lo2, hi2, fp2)  # seeded
        assert lo1 == pytest.approx(0.01) and fp1 == 1.0
        mixed = [0.01, -0.01] * 100  # mean 0 → CI must straddle 0
        lo, hi, _ = ftc.paired_bootstrap_delta(mixed, b=500)
        assert lo < 0 < hi


# ---------------------------------------------------------------------------
# run() gate behavior — never write unless the gate passes AND not report-only
# ---------------------------------------------------------------------------

def _synthetic_cache(tmp_path: Path, miscalibrated: bool) -> Path:
    """600 rows over 6 months, empirical win rates exactly 0.9 / 0.1.

    miscalibrated=True: the model states 0.6 / 0.4 for those groups, so
    isotonic has a large, real gain to find on the holdout. False: the model
    states the true frequencies — nothing to learn, gate must not fire."""
    rows = []
    months = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06"]
    for month in months:
        for i in range(100):
            actual_p = 0.9 if i % 2 == 0 else 0.1
            stated = (0.6 if i % 2 == 0 else 0.4) if miscalibrated else actual_p
            out = 1.0 if (i % 20) < actual_p * 20 else 0.0  # 9/10 even, 1/10 odd — exact
            rows.append(_row(out, f"{month}-{i % 28 + 1:02d}", stated, stated))
    p = tmp_path / "pit_rows.json"
    p.write_text(json.dumps(rows))
    return p


@pytest.fixture
def _isolated_calibration(tmp_path, monkeypatch):
    """Point the calibration store at tmp so no test can touch the real file."""
    cal_path = tmp_path / "calibration.json"
    monkeypatch.setattr("evmax.agents.models.calibration.CALIBRATION_PATH", cal_path)
    return cal_path


class TestRunGate:
    @pytest.fixture(autouse=True)
    def _sklearn(self):
        pytest.importorskip("sklearn")

    def test_gate_fail_writes_nothing(self, tmp_path, monkeypatch, capsys, _isolated_calibration):
        cache = _synthetic_cache(tmp_path, miscalibrated=False)
        monkeypatch.setattr(ftc, "CACHE", cache)
        ftc.run(train_frac=0.7, sharp_weight=0.85, report_only=False)
        out = capsys.readouterr().out
        assert "NO PROMOTE" in out
        assert not _isolated_calibration.exists()

    def test_gate_pass_report_only_writes_nothing(self, tmp_path, monkeypatch, capsys,
                                                  _isolated_calibration):
        cache = _synthetic_cache(tmp_path, miscalibrated=True)
        monkeypatch.setattr(ftc, "CACHE", cache)
        ftc.run(train_frac=0.7, sharp_weight=0.85, report_only=True)
        out = capsys.readouterr().out
        assert "PASS" in out
        assert "report-only" in out
        assert not _isolated_calibration.exists()

    def test_gate_pass_writes_tennis_ensemble(self, tmp_path, monkeypatch, capsys,
                                              _isolated_calibration):
        cache = _synthetic_cache(tmp_path, miscalibrated=True)
        monkeypatch.setattr(ftc, "CACHE", cache)
        ftc.run(train_frac=0.7, sharp_weight=0.85, report_only=False)
        out = capsys.readouterr().out
        assert "PROMOTED" in out
        state = json.loads(_isolated_calibration.read_text())
        assert "tennis_ensemble" in state
        assert len(state["tennis_ensemble"]["x"]) >= 2

    def test_stale_prefix_cache_aborts(self, tmp_path, monkeypatch, _isolated_calibration):
        """A cache where surface never clears the gate = pre-fix artifact → abort."""
        rows = [_row(1.0, f"2025-01-{i + 1:02d}", 0.5, 0.55, conf=0.30) for i in range(28)]
        cache = tmp_path / "stale.json"
        cache.write_text(json.dumps(rows))
        monkeypatch.setattr(ftc, "CACHE", cache)
        with pytest.raises(SystemExit):
            ftc.run(train_frac=0.7, sharp_weight=0.85, report_only=True)
