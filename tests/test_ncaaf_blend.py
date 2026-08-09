"""Tests for scripts/backtest_ncaaf_blend.py pure helpers (NCAAF blend sweep).

The blend sweep re-implements the live models' math as direct replays, so these
tests pin that math to the agents it mirrors:
- _log5   — Bill James head-to-head, matches FormModelAgent._log5.
- _form_rate — exponentially-weighted, shrunk recent win rate.
- _elo_conf  — the >=5-game gate (None = dark), matches EloModelAgent tiers.
- blend_brier — confidence-weighted ensemble blend that drops dark / zero-weight
  models, matches EnsembleModelAgent (eff_w = base_weight * confidence).

No network — synthetic per-game tables only.
"""

from __future__ import annotations

import pytest

from scripts.backtest_ncaaf_blend import _elo_conf, _form_rate, _log5, blend_brier


class TestLog5:
    def test_even_teams(self):
        assert _log5(0.5, 0.5) == pytest.approx(0.5)

    def test_stronger_a_favored(self):
        # (0.6 - 0.24) / (0.6 + 0.4 - 0.48) = 0.36 / 0.52
        assert _log5(0.6, 0.4) == pytest.approx(0.36 / 0.52)

    def test_degenerate_denominator(self):
        # p_a = p_b = 1.0 → denom ~0 → safe 0.5
        assert _log5(1.0, 1.0) == pytest.approx(0.5)


class TestFormRate:
    def test_empty_is_neutral(self):
        assert _form_rate([]) == pytest.approx(0.5)

    def test_all_wins_shrinks_toward_half(self):
        recs = [{"date": f"2025-09-0{i}", "won": True} for i in range(1, 4)]  # n=3
        # raw = 1.0; shrinkage = 3/(3+6) = 1/3; 1.0*1/3 + 0.5*2/3 = 0.6667
        assert _form_rate(recs) == pytest.approx(1.0 / 3 + 0.5 * 2 / 3)

    def test_all_losses_symmetric(self):
        recs = [{"date": f"2025-09-0{i}", "won": False} for i in range(1, 4)]
        # raw = 0.0; 0.0*1/3 + 0.5*2/3 = 0.3333
        assert _form_rate(recs) == pytest.approx(0.5 * 2 / 3)

    def test_recency_weighting_favors_latest(self):
        # SAME win count (1W/1L), only WHICH game is the win differs. Exponential
        # decay must rank the recent-win record above the recent-loss record.
        recent_win = [
            {"date": "2025-09-20", "won": True},
            {"date": "2025-09-13", "won": False},
        ]
        recent_loss = [
            {"date": "2025-09-20", "won": False},
            {"date": "2025-09-13", "won": True},
        ]
        assert _form_rate(recent_win) > _form_rate(recent_loss)


class TestEloConfGate:
    def test_below_five_is_dark(self):
        assert _elo_conf(0) is None
        assert _elo_conf(4) is None

    def test_five_to_fourteen(self):
        assert _elo_conf(5) == pytest.approx(0.60)
        assert _elo_conf(14) == pytest.approx(0.60)

    def test_fifteen_plus(self):
        assert _elo_conf(15) == pytest.approx(0.80)


class TestBlendBrier:
    def _row(self, y, eff=None, elo=None, form=None, bucket="9+"):
        return {"y": y, "bucket": bucket, "eff": eff, "elo": elo, "form": form}

    def test_confidence_weighted_average_with_dark_model(self):
        # eff fires (0.8 @ conf 0.85), elo DARK (None), form fires (0.6 @ 0.5).
        row = self._row(1, eff=(0.8, 0.85), elo=None, form=(0.6, 0.5))
        w = {"eff": 0.55, "elo": 0.25, "form": 0.20}
        # eff_w = 0.55*0.85 = 0.4675 ; form_w = 0.20*0.5 = 0.10
        p = (0.4675 * 0.8 + 0.10 * 0.6) / (0.4675 + 0.10)
        b, acc, n = blend_brier([row], w)
        assert n == 1
        assert b == pytest.approx((p - 1) ** 2)
        assert acc == pytest.approx(1.0)  # p > 0.5, y = 1

    def test_zero_weight_model_excluded(self):
        row = self._row(1, eff=(0.9, 0.8), elo=(0.1, 0.8), form=None)
        # elo weight 0 → excluded entirely; blend = eff prob 0.9
        b, _, n = blend_brier([row], {"eff": 1.0, "elo": 0.0, "form": 0.0})
        assert n == 1
        assert b == pytest.approx((0.9 - 1) ** 2)

    def test_all_models_dark_row_skipped(self):
        row = self._row(1, eff=None, elo=None, form=None)
        b, acc, n = blend_brier([row], {"eff": 0.55, "elo": 0.25, "form": 0.20})
        assert n == 0
        assert b != b  # NaN

    def test_bucket_filter(self):
        rows = [
            self._row(1, eff=(0.8, 0.8), bucket="0-3"),
            self._row(0, eff=(0.8, 0.8), bucket="9+"),
        ]
        _, _, n = blend_brier(rows, {"eff": 1.0, "elo": 0.0, "form": 0.0}, bucket="0-3")
        assert n == 1
