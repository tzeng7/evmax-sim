"""Partial-blend gaps (full_blend=False, e.g. tennis without the full model
blend) are demoted to mode='shadow' with Kelly zeroed and are NOT actionable
plays. The dashboard scan must not surface them — see
evmax.web.app._run_unified_scan."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta

import evmax.agents.cleanup.logger as logger_module
import evmax.agents.coordinator as coordinator_module
from evmax.agents.coordinator import CycleResult
from evmax.agents.odds.ev_gap_agent import EVGap
from evmax.web.app import _run_unified_scan


def _gap(
    market_id: str,
    *,
    full_blend: bool,
    model_sources: str,
    event_date: datetime | None = None,
) -> EVGap:
    return EVGap(
        market_id=market_id,
        event_id=f"evt:{market_id}",
        sector="tennis",
        yes_team="Tiafoe",
        market_type="moneyline",
        kalshi_yes_price=0.38,
        sharp_true_prob=0.39,
        blended_true_prob=0.39,
        ev_pct=0.03,
        kelly_full=0.0 if not full_blend else 0.02,
        kelly_fraction=0.0 if not full_blend else 0.02,
        match_confidence=0.95,
        volume_usd=2_000.0,
        spread_pct=0.02,
        event_date=event_date or datetime.combine(date.today(), datetime.min.time()),
        model_sources=model_sources,
        line=None,
        event_title="Frances Tiafoe vs Felix Auger-Aliassime",
        full_blend=full_blend,
    )


def test_partial_blend_gap_excluded_from_scan(monkeypatch):
    full = _gap("kalshi:full", full_blend=True,
                model_sources="tennis_surface+tennis_serve_return+tennis_form+tennis_advanced+sharp")
    partial = _gap("kalshi:partial", full_blend=False,
                   model_sources="tennis_surface+sharp")

    class _FakeCoord:
        def __init__(self, *a, **k):
            pass

        async def run_cycle(self):
            return CycleResult(ev_gaps=[full, partial], bankroll=500.0, kelly_fraction=0.5)

    monkeypatch.setattr(coordinator_module, "AgentCoordinator", _FakeCoord)
    # Keep the partial gap's shadow logging path intact but inert (no DB write).
    logged: list = []
    monkeypatch.setattr(logger_module, "log_gaps",
                        lambda gaps, **kw: logged.extend(gaps) or len(gaps))

    cycle, gap_dicts, _ = asyncio.run(
        _run_unified_scan(sectors=["tennis"], bankroll=500.0, kelly=0.5,
                          fan_out_portfolio_ids=None)
    )

    mids = {g["market_id"] for g in gap_dicts}
    # The full-blend play is surfaced; the partial-blend, $0.00-stake gap is not.
    assert "kalshi:full" in mids
    assert "kalshi:partial" not in mids
    # ...but the partial gap is still handed to log_gaps for shadow logging.
    assert any(g.market_id == "kalshi:partial" for g in logged)


def test_persistence_respects_scan_date_window(monkeypatch):
    """A ranged dashboard scan must persist ONLY in-range gaps to
    ev_predictions. The coordinator cycle surfaces games days out, but those
    out-of-range rows must not leak into the DB (and thence Open Positions).
    Regression for the drift where the date filter applied only to the
    displayed list, not to persistence."""
    today = date.today()
    in_range = _gap(
        "kalshi:in", full_blend=True,
        model_sources="tennis_surface+tennis_serve_return+tennis_form+tennis_advanced+sharp",
        event_date=datetime.combine(today, datetime.min.time()),
    )
    out_of_range = _gap(
        "kalshi:out", full_blend=True,
        model_sources="tennis_surface+tennis_serve_return+tennis_form+tennis_advanced+sharp",
        event_date=datetime.combine(today + timedelta(days=10), datetime.min.time()),
    )

    class _FakeCoord:
        def __init__(self, *a, **k):
            pass

        async def run_cycle(self):
            return CycleResult(ev_gaps=[in_range, out_of_range],
                               bankroll=500.0, kelly_fraction=0.5)

    monkeypatch.setattr(coordinator_module, "AgentCoordinator", _FakeCoord)
    logged: list = []
    monkeypatch.setattr(logger_module, "log_gaps",
                        lambda gaps, **kw: logged.extend(gaps) or len(gaps))

    iso = today.isoformat()
    asyncio.run(
        _run_unified_scan(sectors=["tennis"], bankroll=500.0, kelly=0.5,
                          fan_out_portfolio_ids=None, date_from=iso, date_to=iso)
    )

    logged_mids = {g.market_id for g in logged}
    assert "kalshi:in" in logged_mids
    assert "kalshi:out" not in logged_mids


def test_persistence_default_window_is_today_tomorrow(monkeypatch):
    """With no explicit range, persistence falls back to today+tomorrow —
    matching the scan display default — so e.g. the portfolio "Scan All"
    path (which passes no range) can't dump days-out games into Open
    Positions either."""
    today = date.today()
    tomorrow_gap = _gap(
        "kalshi:tomorrow", full_blend=True,
        model_sources="tennis_surface+tennis_serve_return+tennis_form+tennis_advanced+sharp",
        event_date=datetime.combine(today + timedelta(days=1), datetime.min.time()),
    )
    far_gap = _gap(
        "kalshi:far", full_blend=True,
        model_sources="tennis_surface+tennis_serve_return+tennis_form+tennis_advanced+sharp",
        event_date=datetime.combine(today + timedelta(days=5), datetime.min.time()),
    )

    class _FakeCoord:
        def __init__(self, *a, **k):
            pass

        async def run_cycle(self):
            return CycleResult(ev_gaps=[tomorrow_gap, far_gap],
                               bankroll=500.0, kelly_fraction=0.5)

    monkeypatch.setattr(coordinator_module, "AgentCoordinator", _FakeCoord)
    logged: list = []
    monkeypatch.setattr(logger_module, "log_gaps",
                        lambda gaps, **kw: logged.extend(gaps) or len(gaps))

    asyncio.run(
        _run_unified_scan(sectors=["tennis"], bankroll=500.0, kelly=0.5,
                          fan_out_portfolio_ids=None)
    )

    logged_mids = {g.market_id for g in logged}
    assert "kalshi:tomorrow" in logged_mids
    assert "kalshi:far" not in logged_mids
