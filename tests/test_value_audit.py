"""Tests for the per-sector model-blend value audit (evmax/agents/cleanup/value_audit.py).

The observer's whole job is to tell a REAL gap apart from noise, so the tests focus on
the statistics (paired diff sign + significance), the calibration-bias direction logic,
and the verdict classification — not just happy-path plumbing.
"""

from __future__ import annotations

from evmax.agents.cleanup import value_audit as va


def _row(out, sharp, blended, close=None, clv=None, mt="moneyline"):
    return {
        "outcome": out,
        "sharp_true_prob": sharp,
        "blended_true_prob": blended,
        "pinnacle_close_prob": close,
        "kalshi_clv_pct": clv,
        "market_type": mt,
        "sector": "test",
    }


class TestBrier:
    def test_brier_basic(self):
        rows = [_row(1, 0.5, 0.6), _row(0, 0.5, 0.4)]
        # model: (0.6-1)^2=0.16, (0.4-0)^2=0.16 -> 0.16
        assert abs(va._brier(rows, "blended_true_prob") - 0.16) < 1e-9

    def test_brier_skips_none(self):
        rows = [_row(1, 0.5, 0.6, close=0.55), _row(0, 0.5, 0.4, close=None)]
        # only first row has close
        assert abs(va._brier(rows, "pinnacle_close_prob") - (0.55 - 1) ** 2) < 1e-9

    def test_brier_all_none_returns_none(self):
        rows = [_row(1, 0.5, 0.6, close=None)]
        assert va._brier(rows, "pinnacle_close_prob") is None


class TestPairedDiff:
    def test_model_better_positive_mean(self):
        # model closer to outcome than sharp on every row -> mean diff > 0, varied -> z > 0
        rows = [_row(1, 0.5, 0.8), _row(0, 0.5, 0.3),
                _row(1, 0.6, 0.9), _row(0, 0.4, 0.2)] * 10
        st = va._paired_diff_stats(rows, "sharp_true_prob", "blended_true_prob")
        assert st["mean"] > 0
        assert st["z"] > 0

    def test_model_worse_negative_mean(self):
        # model farther from outcome than sharp on every row -> mean diff < 0, z < 0
        rows = [_row(1, 0.8, 0.5), _row(0, 0.3, 0.5),
                _row(1, 0.9, 0.6), _row(0, 0.2, 0.4)] * 10
        st = va._paired_diff_stats(rows, "sharp_true_prob", "blended_true_prob")
        assert st["mean"] < 0
        assert st["z"] < 0

    def test_identical_columns_zero_mean(self):
        rows = [_row(1, 0.6, 0.6), _row(0, 0.4, 0.4)] * 10
        st = va._paired_diff_stats(rows, "sharp_true_prob", "blended_true_prob")
        assert abs(st["mean"]) < 1e-12
        assert st["se"] == 0.0

    def test_needs_two_paired_rows(self):
        assert va._paired_diff_stats([_row(1, 0.5, 0.6)], "sharp_true_prob", "blended_true_prob") is None

    def test_only_uses_rows_with_both_present(self):
        rows = [_row(1, 0.5, 0.6, close=None)] * 5  # close missing -> can't pair vs close
        assert va._paired_diff_stats(rows, "pinnacle_close_prob", "blended_true_prob") is None


class TestClv:
    def test_clv_mean_and_fraction(self):
        rows = [_row(1, 0.5, 0.5, clv=2.0), _row(0, 0.5, 0.5, clv=-1.0),
                _row(1, 0.5, 0.5, clv=3.0), _row(0, 0.5, 0.5, clv=None)]
        st = va._clv_stats(rows)
        assert st["n"] == 3
        assert abs(st["mean_pp"] - (2.0 - 1.0 + 3.0) / 3) < 1e-9
        assert abs(st["frac_positive"] - 2 / 3) < 1e-9

    def test_clv_none_when_empty(self):
        assert va._clv_stats([_row(1, 0.5, 0.5, clv=None)]) is None


class TestCalibration:
    def test_overconfident_positive_bias(self):
        # predict 0.9 but only win half -> over-confident -> positive signed bias
        rows = [_row(1, 0.5, 0.9), _row(0, 0.5, 0.9)] * 25
        cal = va._calibration(rows)
        assert cal["signed_bias_pp"] > 0
        assert cal["consistent_direction"] is True

    def test_underconfident_negative_bias(self):
        rows = [_row(1, 0.5, 0.1), _row(1, 0.5, 0.1)] * 25  # predict 0.1, always win
        cal = va._calibration(rows)
        assert cal["signed_bias_pp"] < 0
        assert cal["consistent_direction"] is True

    def test_mixed_direction_not_consistent(self):
        # one bucket over-confident, another under-confident
        rows = ([_row(0, 0.5, 0.9)] * 10  # 0.9 bucket: predicted high, lost -> over
                + [_row(1, 0.5, 0.1)] * 10)  # 0.1 bucket: predicted low, won -> under
        cal = va._calibration(rows)
        assert cal["consistent_direction"] is False

    def test_ece_is_nonnegative_and_present(self):
        rows = [_row(1, 0.5, 0.9), _row(0, 0.5, 0.9)] * 25
        cal = va._calibration(rows)
        assert "ece_pp" in cal
        assert cal["ece_pp"] >= 0.0

    def test_ece_zero_when_perfectly_calibrated(self):
        # 0.5 bucket that hits exactly 50% -> zero calibration error
        rows = [_row(1, 0.5, 0.5), _row(0, 0.5, 0.5)] * 25
        cal = va._calibration(rows)
        assert cal["ece_pp"] < 1e-9

    def test_ece_does_not_cancel_when_signed_bias_does(self):
        # High-on-favorites + low-on-dogs cancels in signed_bias_pp but NOT in ECE.
        # 0.9 bucket predicted 0.9, hits 0.5 (over by 40pp); 0.1 bucket predicted 0.1,
        # hits 0.5 (under by 40pp). Equal populations -> signed ~0, ECE ~40pp.
        rows = ([_row(1, 0.5, 0.9), _row(0, 0.5, 0.9)] * 25
                + [_row(1, 0.5, 0.1), _row(0, 0.5, 0.1)] * 25)
        cal = va._calibration(rows)
        assert abs(cal["signed_bias_pp"]) < 1.0     # cancels
        assert cal["ece_pp"] > 30.0                 # does not cancel


class TestVerdict:
    def test_insufficient_below_min_n(self):
        v = va._verdict(None, {"signed_bias_pp": 0, "consistent_direction": False}, n=va.MIN_N - 1)
        assert v["tag"] == "insufficient"
        assert v["actionable"] is False

    def test_model_subtracts_flagged_actionable(self):
        edge = {"mean": -0.01, "se": 0.001, "z": -3.0, "ci_lo": -0.012, "ci_hi": -0.008, "n": 100}
        v = va._verdict(edge, {"signed_bias_pp": 0, "consistent_direction": False}, n=100)
        assert v["tag"] == "model_subtracts"
        assert v["actionable"] is True

    def test_calibration_bias_flagged(self):
        edge = {"mean": 0.0, "se": 0.001, "z": 0.1, "ci_lo": 0, "ci_hi": 0, "n": 100}
        cal = {"signed_bias_pp": va.CALIB_BIAS_PP + 1, "consistent_direction": True}
        v = va._verdict(edge, cal, n=100)
        assert v["tag"] == "calibration_bias"
        assert v["actionable"] is True

    def test_adds_value_not_actionable(self):
        edge = {"mean": 0.01, "se": 0.001, "z": 3.0, "ci_lo": 0.008, "ci_hi": 0.012, "n": 100}
        v = va._verdict(edge, {"signed_bias_pp": 0, "consistent_direction": False}, n=100)
        assert v["tag"] == "adds_value"
        assert v["actionable"] is False

    def test_neutral_within_noise(self):
        edge = {"mean": 0.001, "se": 0.002, "z": 0.5, "ci_lo": -0.003, "ci_hi": 0.005, "n": 100}
        v = va._verdict(edge, {"signed_bias_pp": 1.0, "consistent_direction": False}, n=100)
        assert v["tag"] == "neutral"
        assert v["actionable"] is False

    def test_inconsistent_large_bias_not_flagged(self):
        # large signed bias but NOT consistent direction -> not actionable (avoids noise chasing)
        edge = {"mean": 0.0, "se": 0.001, "z": 0.0, "ci_lo": 0, "ci_hi": 0, "n": 100}
        cal = {"signed_bias_pp": 9.0, "consistent_direction": False}
        v = va._verdict(edge, cal, n=100)
        assert v["actionable"] is False
