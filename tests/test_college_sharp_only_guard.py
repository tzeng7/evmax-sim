"""College basketball sharp-only guard (MIN_NONSHARP_MODELS, 2026-11-11).

NCAAB and NCAAW are live sectors. A moneyline row whose blend holds no genuine
model is venue divergence, not model edge, and must not be a live play.
"""

from __future__ import annotations

import pytest

from evmax.agents.odds.ev_gap_agent import MIN_NONSHARP_MODELS, has_full_blend


@pytest.mark.parametrize("sector", ["ncaab", "ncaaw"])
class TestCollegeSharpOnlyGuard:
    def test_registered_for_moneyline_only(self, sector):
        assert MIN_NONSHARP_MODELS[sector]["min_count"] == 1
        assert MIN_NONSHARP_MODELS[sector]["market_types"] == frozenset({"moneyline"})

    def test_sharp_only_moneyline_is_not_live_eligible(self, sector):
        assert has_full_blend(sector, "sharp", "moneyline") is False

    def test_adjustment_tokens_do_not_count_as_models(self, sector):
        assert has_full_blend(sector, "sharp+injury+late_news+rest", "moneyline") is False

    def test_one_genuine_model_satisfies_the_floor(self, sector):
        assert has_full_blend(sector, "elo+sharp", "moneyline") is True
        assert has_full_blend(sector, f"{sector}_efficiency+form+sharp", "moneyline") is True

    def test_spread_and_total_are_unaffected(self, sector):
        # No model prices these; a floor would demote every row.
        assert has_full_blend(sector, "sharp+spread_dist", "spread") is True
        assert has_full_blend(sector, "sharp+total_dist", "total") is True
