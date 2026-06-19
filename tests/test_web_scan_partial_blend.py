"""Partial-blend gaps (full_blend=False, e.g. tennis without the full model
blend) are demoted to mode='shadow' with Kelly zeroed and are NOT actionable
plays. The dashboard scan must not surface them — see
evmax.web.app._run_unified_scan."""

from __future__ import annotations

import asyncio
from datetime import date, datetime

import evmax.agents.cleanup.logger as logger_module
import evmax.agents.coordinator as coordinator_module
from evmax.agents.odds.ev_gap_agent import EVGap
from evmax.web.app import _run_unified_scan


def _gap(market_id: str, *, full_blend: bool, model_sources: str) -> EVGap:
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
        event_date=datetime.combine(date.today(), datetime.min.time()),
        model_sources=model_sources,
        line=None,
        event_title="Frances Tiafoe vs Felix Auger-Aliassime",
        full_blend=full_blend,
    )


class _FakeCycle:
    def __init__(self, gaps):
        self.top_gaps = gaps
        self.markets_fetched = len(gaps)
        self.markets_matched = len(gaps)


def test_partial_blend_gap_excluded_from_scan(monkeypatch):
    full = _gap("kalshi:full", full_blend=True,
                model_sources="tennis_surface+tennis_serve_return+tennis_form+tennis_advanced+sharp")
    partial = _gap("kalshi:partial", full_blend=False,
                   model_sources="tennis_surface+sharp")

    class _FakeCoord:
        def __init__(self, *a, **k):
            pass

        async def run_cycle(self):
            return _FakeCycle([full, partial])

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
