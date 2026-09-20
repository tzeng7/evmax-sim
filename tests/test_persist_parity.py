"""CLI ↔ web persistence parity: both scan surfaces log the SAME row set.

Pins ``coordinator.gap_in_persist_window`` / ``CycleResult.persistable_gaps``
and that the dashboard's ``_gap_in_scan_window`` delegates to the same
predicate when both bounds are known.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

from evmax.agents.coordinator import CycleResult, gap_in_persist_window
from evmax.agents.odds.ev_gap_agent import EVGap


def _gap(mid: str, sector: str, when: datetime | None, event_id: str | None = None,
         full_blend: bool = True, ev: float = 0.05) -> EVGap:
    return EVGap(
        market_id=mid, event_id=event_id or f"{sector}::x::{mid}", sector=sector,
        yes_team="a", market_type="moneyline", kalshi_yes_price=0.5,
        sharp_true_prob=0.55, blended_true_prob=0.55, ev_pct=ev,
        kelly_full=0.1, kelly_fraction=0.02, match_confidence=1.0, volume_usd=1.0,
        spread_pct=0.02, event_date=when, full_blend=full_blend,
    )


_D0 = date(2026, 9, 16)  # a Wednesday
_UTC = timezone.utc


class TestGapInPersistWindow:
    def test_daily_sector_exact_window(self):
        inside = datetime(2026, 9, 16, 23, 0, tzinfo=_UTC)   # ET game-day 09-16
        outside = datetime(2026, 9, 18, 23, 0, tzinfo=_UTC)
        assert gap_in_persist_window(inside, "nba", _D0, _D0) is True
        assert gap_in_persist_window(outside, "nba", _D0, _D0) is False

    def test_weekly_sector_horizon_widens_end_only(self):
        # NFL declares scan_horizon_days=7: a Sunday game survives a Wednesday
        # --date TODAY scan; a game BEFORE the start does not.
        sunday = datetime(2026, 9, 20, 20, 0, tzinfo=_UTC)
        last_week = datetime(2026, 9, 13, 20, 0, tzinfo=_UTC)
        assert gap_in_persist_window(sunday, "nfl", _D0, _D0) is True
        assert gap_in_persist_window(sunday, "nba", _D0, _D0) is False
        assert gap_in_persist_window(last_week, "nfl", _D0, _D0) is False

    def test_dateless_gap_is_kept(self):
        assert gap_in_persist_window(None, "nba", _D0, _D0) is True


class TestPersistableGaps:
    def test_props_excluded_partial_blend_included_window_applied(self):
        inside = datetime(2026, 9, 16, 23, 0, tzinfo=_UTC)
        outside = datetime(2026, 9, 25, 23, 0, tzinfo=_UTC)
        gaps = [
            _gap("g1", "nba", inside),
            _gap("g2", "nba", inside, full_blend=False),          # shadow-bound, still persisted
            _gap("p1", "nba", inside, event_id="nba::x::prop::joe::points::20.5"),
            _gap("g3", "nba", outside),
            _gap("g4", "nba", inside, ev=0.001),                   # below any display floor — still persisted
        ]
        cycle = CycleResult(ev_gaps=gaps)
        ids = sorted(g.market_id for g in cycle.persistable_gaps(_D0, _D0))
        assert ids == ["g1", "g2", "g4"]

    def test_no_display_floor_leaks_into_persistence(self):
        # A row with blended prob far below the CLI's 15% display floor is
        # persisted — persistence is the agent floor + window, nothing else.
        when = datetime(2026, 9, 16, 23, 0, tzinfo=_UTC)
        g = _gap("lo", "nba", when)
        g.blended_true_prob = 0.05
        cycle = CycleResult(ev_gaps=[g])
        assert [x.market_id for x in cycle.persistable_gaps(_D0, _D0)] == ["lo"]


class TestWebDelegates:
    def test_web_window_matches_predicate_when_both_bounds_known(self):
        from evmax.web.app import _gap_in_scan_window

        sunday = datetime(2026, 9, 20, 20, 0, tzinfo=_UTC)
        for sector in ("nfl", "nba"):
            assert _gap_in_scan_window(sunday, sector, "2026-09-16", "2026-09-16") == \
                gap_in_persist_window(sunday, sector, _D0, _D0)
