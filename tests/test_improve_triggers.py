"""Unit tests for evmax/agents/cleanup/improve_triggers.py (pure trigger functions)."""

from __future__ import annotations

import json

import pytest

from evmax.agents.cleanup import improve_triggers as it


def _board(**over) -> dict:
    r = {"sector": "ncaab", "market_type": "moneyline", "venue": "polymarket_us", "mode": "live",
         "blend_divergence_pp": 0.16, "n_logged": 23, "n_clean_resolved": 19,
         "top_blockers": ["ncaab_efficiency:missing×23"], "clv": {"n": 10}}
    r.update(over)
    return r


class TestPassthrough:
    def test_live_low_divergence_fires(self):
        [f] = it.passthrough_findings([_board()])
        assert f["trigger"] == "passthrough" and f["fires"]
        assert f["judge_metric"] == "rows_priced_frac"
        assert f["venue"] == "polymarket_us" and f["value"] == 0.16
        assert f["details"]["top_blockers"] == ["ncaab_efficiency:missing×23"]

    @pytest.mark.parametrize("over", [
        {"mode": "shadow"},                     # shadow lanes stake nothing
        {"blend_divergence_pp": 0.5},           # boundary: must be strictly below 0.5
        {"blend_divergence_pp": None},
        {"market_type": "spread"},
        {"n_logged": it.PASSTHROUGH_MIN_N - 1},
    ])
    def test_non_firing(self, over):
        assert it.passthrough_findings([_board(**over)]) == []

    def test_lower_divergence_has_higher_priority(self):
        a, b = it.passthrough_findings([_board(blend_divergence_pp=0.4), _board(blend_divergence_pp=0.1)])
        assert b["priority"] > a["priority"]


def _diag(missing=(), gated=None, fired=None) -> dict:
    return {"fired": fired or {}, "gated": gated or {}, "missing": list(missing)}


class TestCoverage:
    def _rows(self, sector, diags):
        return [{"sector": sector, "model_diagnostics": json.dumps(d)} for d in diags]

    def test_missing_majority_fires(self):
        rows = self._rows("ncaab", [_diag(["ncaab_efficiency"])] * 15 + [_diag()] * 5)
        [f] = it.coverage_findings(rows)
        assert (f["sector"], f["model"], f["value"], f["n"]) == ("ncaab", "ncaab_efficiency", 0.75, 20)
        assert f["details"]["missing"] == 15

    def test_gated_counts_and_notes_surface(self):
        g = {"elo": {"conf": 0.3, "note": "n=0"}}
        rows = self._rows("ncaab", [_diag(gated=g)] * 12 + [_diag()] * 8)
        [f] = it.coverage_findings(rows)
        assert f["details"]["gated"] == 12 and f["details"]["top_gate_notes"] == ["n=0 ×12"]

    def test_below_share_or_n_does_not_fire(self):
        assert it.coverage_findings(self._rows("nba", [_diag(["matchup"])] * 9 + [_diag()] * 11)) == []
        assert it.coverage_findings(self._rows("nba", [_diag(["matchup"])] * (it.COVERAGE_MIN_N - 1))) == []

    def test_structurally_sparse_models_skipped(self):
        rows = self._rows("tennis", [_diag(["tennis_h2h", "tennis_ranking_trend", "tennis_form"])] * 30)
        assert [f["model"] for f in it.coverage_findings(rows)] == ["tennis_form"]

    def test_configured_filter_drops_unconfigured_and_zero_weight(self):
        # generic elo shows up "gated" on tennis rows but is not a tennis model
        rows = self._rows("tennis", [_diag(["tennis_form"], gated={"elo": {"conf": 0.3}})] * 30)
        got = it.coverage_findings(rows, {"tennis": {"tennis_form", "tennis_surface"}})
        assert [f["model"] for f in got] == ["tennis_form"]

    def test_rows_without_diagnostics_are_ignored(self):
        rows = [{"sector": "nba", "model_diagnostics": None}] * 50 + [{"sector": "nba", "model_diagnostics": "{bad"}]
        assert it.coverage_findings(rows) == []


class TestCloseCapture:
    def _rows(self, n, nulls):
        return [{"sector": "ncaab", "market_type": "total" if i < nulls else "moneyline",
                 "market_id": f"K{i}", "kalshi_clv_pct": None if i < nulls else 0.5} for i in range(n)]

    def test_fires_above_threshold(self):
        [f] = it.close_capture_findings(self._rows(15, 5))
        assert f["value"] == 0.333 and f["venue"] == "kalshi"
        assert f["details"]["null_by_market_type"] == {"total": 5}
        assert f["details"]["example_market_ids"] == ["K0", "K1", "K2", "K3", "K4"]

    def test_boundary_and_small_n(self):
        assert it.close_capture_findings(self._rows(10, 3)) == []   # exactly 30% does not fire
        assert it.close_capture_findings(self._rows(9, 9)) == []    # n < 10


def _cal_rows(n, blend, sharp, wins, price=0.15, sector="nba", mt="moneyline"):
    return [{"sector": sector, "market_type": mt, "outcome": 1 if i < wins else 0,
             "blended_true_prob": blend, "sharp_true_prob": sharp,
             "placed": 0, "placed_price": None, "kalshi_yes_price": price} for i in range(n)]


class TestCalibrationBucket:
    def test_blend_worse_than_sharp_fires(self):
        # 200 rows at 15c: realized 10%, blend 18% (+8pp), sharp 12% (+2pp)
        [f] = it.calibration_bucket_findings(_cal_rows(200, 0.18, 0.12, 20))
        assert f["bucket"] == "10-20" and f["value"] == pytest.approx(8.0)
        assert f["z"] > it.CALIB_BUCKET_MIN_Z
        assert f["details"]["excess_pp"] == pytest.approx(6.0)

    def test_shared_selection_bias_does_not_fire(self):
        # blend and sharp equally over-state the bucket → EV-selection, not the model
        assert it.calibration_bucket_findings(_cal_rows(200, 0.18, 0.18, 20)) == []

    def test_small_or_insignificant_buckets_do_not_fire(self):
        assert it.calibration_bucket_findings(_cal_rows(it.CALIB_BUCKET_MIN_N - 1, 0.30, 0.10, 5)) == []
        assert it.calibration_bucket_findings(_cal_rows(60, 0.13, 0.10, 6)) == []   # |err| 3pp but z small

    def test_rows_missing_probs_or_price_skipped(self):
        rows = _cal_rows(200, 0.18, 0.12, 20)
        for r in rows:
            r["sharp_true_prob"] = None
        assert it.calibration_bucket_findings(rows) == []
        rows = _cal_rows(200, 0.18, 0.12, 20, price=None)
        assert it.calibration_bucket_findings(rows) == []


class TestCompute:
    def test_unknown_trigger_raises(self):
        with pytest.raises(ValueError):
            it.compute_improve_triggers(["nope"])

    def test_failing_trigger_does_not_mask_others(self, monkeypatch):
        monkeypatch.setattr(it, "_fetch_close_rows", lambda days: [
            {"sector": "ncaaf", "market_type": "moneyline", "market_id": f"K{i}", "kalshi_clv_pct": None}
            for i in range(12)
        ])

        def boom(days):
            raise RuntimeError("db gone")

        monkeypatch.setattr(it, "_fetch_coverage_rows", boom)
        out = it.compute_improve_triggers(["model_coverage", "close_capture"])
        assert out["ran"] == ["close_capture"] and "db gone" in out["failed"]["model_coverage"]
        assert [f["sector"] for f in out["findings"]] == ["ncaaf"]

    def test_configured_models_excludes_sharp_and_zero_weight(self):
        cfg = it.configured_models()
        assert "sharp" not in cfg["nba"]
        assert "poisson" not in cfg.get("nfl", set())  # nfl override weights poisson 0.0
