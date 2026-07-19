"""Tests for the shared bullpen model (evmax/models_ml/bullpen.py, WS2a).

Extracted from the walk-forward so the live pitcher agent and the backtest
run one implementation; these fixtures lock in the fatigue heuristics and
the 60/40 blend the walk-forward validated.
"""

from __future__ import annotations

from datetime import date

import pytest

from evmax.models_ml.bullpen import (
    FATIGUE_PC_THRESHOLD,
    PEN_BLEND_RELIEVER_SHARE,
    PEN_BLEND_STARTER_SHARE,
    TOP_N_AVAILABLE_PEN,
    is_reliever_available,
    league_pen_cfip,
    team_pen_quality,
    team_rate_with_pen,
)

TODAY = date(2026, 7, 19)


class TestIsRelieverAvailable:
    def test_no_appearances_available(self):
        assert is_reliever_available([], TODAY) is True

    def test_heavy_yesterday_unavailable(self):
        apps = [(date(2026, 7, 18), 1.0, FATIGUE_PC_THRESHOLD)]
        assert is_reliever_available(apps, TODAY) is False

    def test_light_yesterday_available(self):
        apps = [(date(2026, 7, 18), 0.1, FATIGUE_PC_THRESHOLD - 1)]
        assert is_reliever_available(apps, TODAY) is True

    def test_two_of_last_three_unavailable(self):
        apps = [
            (date(2026, 7, 16), 1.0, 12),
            (date(2026, 7, 18), 0.2, 8),
        ]
        assert is_reliever_available(apps, TODAY) is False

    def test_heavy_three_days_ago_available(self):
        apps = [(date(2026, 7, 16), 2.0, 40)]
        assert is_reliever_available(apps, TODAY) is True

    def test_today_appearance_ignored(self):
        # A same-day line (in-progress double header etc.) is not fatigue
        # history for TODAY's availability call.
        apps = [(TODAY, 1.0, 30)]
        assert is_reliever_available(apps, TODAY) is True


def _pen_state(n: int = 4, team: str = "yankees", fip_spread: bool = True):
    running = {}
    team_map = {}
    for i in range(n):
        name = f"reliever {i}"
        # so grows with i → FIP falls with i → 'reliever n-1' is the best arm.
        running[name] = {"ip": 20.0, "er": 8, "bb": 6, "so": 15 + (5 * i if fip_spread else 0), "hr": 2}
        team_map[name] = team
    return running, team_map


class TestTeamPenQuality:
    def test_averages_top_n_by_fip(self):
        running, team_map = _pen_state(5)
        q = team_pen_quality("yankees", TODAY, running, {}, team_map, 3.10)
        # Best three arms are the highest-so relievers (indices 4, 3, 2).
        expected = sum(
            (13 * 2 + 3 * 6 - 2 * (15 + 5 * i)) / 20.0 + 3.10 for i in (4, 3, 2)
        ) / 3
        assert q == pytest.approx(expected)

    def test_fatigued_arm_excluded(self):
        running, team_map = _pen_state(4)
        # Fatigue the best arm (index 3) — quality must be computed without it.
        apps = {"reliever 3": [(date(2026, 7, 18), 1.0, 30)]}
        q_with = team_pen_quality("yankees", TODAY, running, {}, team_map, 3.10)
        q_without = team_pen_quality("yankees", TODAY, running, apps, team_map, 3.10)
        assert q_without > q_with  # lost the best arm → worse (higher) FIP

    def test_thin_pool_returns_none(self):
        running, team_map = _pen_state(TOP_N_AVAILABLE_PEN - 1)
        assert team_pen_quality("yankees", TODAY, running, {}, team_map, 3.10) is None

    def test_low_ip_reliever_excluded(self):
        running, team_map = _pen_state(3)
        running["reliever 0"]["ip"] = 4.9  # below RELIEVER_MIN_IP
        assert team_pen_quality("yankees", TODAY, running, {}, team_map, 3.10) is None

    def test_other_teams_relievers_ignored(self):
        running, team_map = _pen_state(3, team="yankees")
        r2, t2 = _pen_state(3, team="red sox")
        running.update({f"b{k}": v for k, v in r2.items()})
        team_map.update({f"b{k}": v for k, v in t2.items()})
        q = team_pen_quality("yankees", TODAY, running, {}, team_map, 3.10)
        assert q is not None


class TestLeaguePenCfip:
    def test_textbook_fallback_when_thin(self):
        assert league_pen_cfip({"ip": 50.0, "er": 20, "bb": 15, "so": 40, "hr": 5}) == 3.10

    def test_computed_from_totals(self):
        totals = {"ip": 1000.0, "er": 450, "bb": 350, "so": 950, "hr": 110}
        era = 450 * 9 / 1000
        fip_raw = (13 * 110 + 3 * 350 - 2 * 950) / 1000
        assert league_pen_cfip(totals) == pytest.approx(era - fip_raw)


class TestTeamRateWithPen:
    def test_blend_shares(self):
        rate = team_rate_with_pen(4.00, 3.50)
        assert rate == pytest.approx(
            PEN_BLEND_STARTER_SHARE * 4.00 + PEN_BLEND_RELIEVER_SHARE * 3.50
        )

    def test_starter_only_fallback(self):
        assert team_rate_with_pen(4.00, None) == 4.00
