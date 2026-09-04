"""evmax.web.playlist — the one definition of the dashboard's play list,
shared by /api/scan and the Discord scan feed. Locks in the selection rules
that used to live inline in the FastAPI handlers so neither surface can drift.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta

from evmax.agents.coordinator import CycleResult
from evmax.agents.odds.ev_gap_agent import EVGap
from evmax.web import playlist


def _noon(days: int = 0) -> datetime:
    return (datetime.now().astimezone() + timedelta(days=days)).replace(
        hour=12, minute=0, second=0, microsecond=0,
    )


def _gap(market_id: str, *, ev: float = 0.05, venue: str = "kalshi", kelly: float = 0.02,
         market_type: str = "moneyline", full_blend: bool = True, days: int = 0,
         yes_team: str = "lakers", event_id: str | None = None) -> EVGap:
    return EVGap(
        market_id=market_id,
        # Distinct event per market id unless a test wants the SAME bet on two
        # venues (best-execution collapse folds those into one row).
        event_id=event_id or f"nba::2026-07-08::lakers_vs_warriors_{market_id}",
        sector="nba",
        yes_team=yes_team,
        market_type=market_type,
        kalshi_yes_price=0.45,
        sharp_true_prob=0.55,
        blended_true_prob=0.55,
        ev_pct=ev,
        kelly_full=0.10,
        kelly_fraction=kelly,
        match_confidence=0.95,
        volume_usd=1000.0,
        spread_pct=0.02,
        event_title="Lakers vs Warriors",
        event_date=_noon(days),
        full_blend=full_blend,
        venue=venue,
    )


def _cycle(gaps, bankroll: float = 500.0) -> CycleResult:
    r = CycleResult(bankroll=bankroll, kelly_fraction=0.5)
    r.ev_gaps = list(gaps)
    return r


class TestDashboardPlayDicts:
    def test_full_blend_only_ev_desc_and_dict_shape(self):
        rows = playlist.dashboard_play_dicts(
            _cycle([_gap("A", ev=0.03), _gap("B", ev=0.09), _gap("C", ev=0.20, full_blend=False)]), 500.0,
        )
        assert [r["market_id"] for r in rows] == ["B", "A"]
        r = rows[0]
        assert r["display_label"] == "Lakers ML"
        assert r["kalshi_price"] == 0.45 and r["true_prob"] == 0.55
        assert r["ev_pct"] == 9.0 and r["ev_pct_raw"] == 0.09
        assert r["stake"] == 10.0 and r["kelly_pct"] == 2.0
        assert r["event_date"] == date.today().isoformat()
        assert r["venue"] == "kalshi" and r["mode"] == "live"

    def test_same_bet_on_two_venues_collapses_to_best_execution_row(self, monkeypatch):
        from evmax.settings import get_settings
        monkeypatch.setattr(get_settings(), "polymarket_us_live", True)
        rows = playlist.dashboard_play_dicts(
            _cycle([
                _gap("K", ev=0.04, venue="kalshi", event_id="nba::2026-07-08::lakers_vs_warriors"),
                _gap("P", ev=0.06, venue="polymarket_us", event_id="nba::2026-07-08::lakers_vs_warriors"),
            ]), 500.0,
        )
        assert len(rows) == 1
        assert rows[0]["market_id"] == "P"
        assert rows[0]["alt_venue"] == "kalshi" and rows[0]["alt_venue_price"] == 0.45
        assert [o["venue"] for o in rows[0]["venue_options"]] == ["polymarket_us", "kalshi"]
        assert rows[0]["venue_options"][0]["venue_options"] is None  # never recurses

    def test_cash_cap_applied_when_known(self):
        rows = playlist.dashboard_play_dicts(
            _cycle([_gap("A", kelly=0.05), _gap("B", kelly=0.05, yes_team="warriors")]), 1000.0,
            cash_by_venue={"kalshi": 50.0},
        )
        assert sum(r["stake"] for r in rows) <= 50.0 + 1e-6


class TestFilterScanView:
    def _rows(self):
        today = date.today()
        return [
            {"market_id": "A", "event_date": today.isoformat(), "market_type": "moneyline"},
            {"market_id": "B", "event_date": (today + timedelta(days=1)).isoformat(), "market_type": "spread"},
            {"market_id": "C", "event_date": (today + timedelta(days=2)).isoformat(), "market_type": "moneyline"},
            {"market_id": "D", "event_date": today.isoformat(), "market_type": "map_handicap"},
            {"market_id": "E", "event_date": today.isoformat(), "market_type": "player_prop"},
            {"market_id": "F", "event_date": (today - timedelta(days=1)).isoformat(), "market_type": "total"},
        ]

    def test_default_window_is_today_and_tomorrow(self):
        out = playlist.filter_scan_view(self._rows(), placed_mids=set())
        assert [r["market_id"] for r in out] == ["A", "B"]

    def test_explicit_window_and_one_sided_bounds(self):
        rows = self._rows()
        today = date.today()
        d2 = (today + timedelta(days=2)).isoformat()
        assert [r["market_id"] for r in playlist.filter_scan_view(rows, d2, d2, set())] == ["C"]
        assert [r["market_id"] for r in playlist.filter_scan_view(rows, date_from=d2, placed_mids=set())] == ["C"]
        y = (today - timedelta(days=1)).isoformat()
        assert [r["market_id"] for r in playlist.filter_scan_view(rows, date_to=y, placed_mids=set())] == ["F"]

    def test_placed_markets_dropped_and_order_kept(self):
        out = playlist.filter_scan_view(self._rows(), placed_mids={"A"})
        assert [r["market_id"] for r in out] == ["B"]

    def test_placed_lookup_defaults_to_db(self, monkeypatch):
        monkeypatch.setattr(playlist, "placed_market_ids", lambda: {"B"})
        out = playlist.filter_scan_view(self._rows())
        assert [r["market_id"] for r in out] == ["A"]

    def test_default_scan_window(self):
        a, b = playlist.default_scan_window()
        assert a == date.today().isoformat()
        assert b == (date.today() + timedelta(days=1)).isoformat()


class TestAppUsesPlaylist:
    def test_gap_to_dict_alias_is_the_shared_function(self):
        from evmax.web import app as web_app
        assert web_app._gap_to_dict is playlist.gap_to_dict

    def test_run_dashboard_scan_returns_filtered_gaps(self, monkeypatch):
        from evmax.web import app as web_app

        seen = {}

        async def fake_unified(**kw):
            seen.update(kw)
            cycle = _cycle([_gap("A"), _gap("B", days=3)])
            return cycle, playlist.dashboard_play_dicts(cycle, kw["bankroll"]), []

        class _Plan:
            bankroll = 321.0
            source = "manual"
            selected_venues = None
            cash_by_venue = {}

        async def fake_plan(bankroll, venue):
            return _Plan()

        monkeypatch.setattr(web_app, "_run_unified_scan", fake_unified)
        monkeypatch.setattr("evmax.clients.balances.resolve_bankroll_plan", fake_plan)
        monkeypatch.setattr(playlist, "placed_market_ids", lambda: set())

        payload = asyncio.run(web_app.run_dashboard_scan(sectors_str="nba,wnba", bankroll=100, kelly=0.25))
        assert seen["sectors"] == ["nba", "wnba"]
        assert seen["bankroll"] == 321.0 and seen["kelly"] == 0.25
        assert seen["fan_out_portfolio_ids"] == []
        assert [g["market_id"] for g in payload["gaps"]] == ["A"]  # B is outside today+tomorrow
        assert payload["bankroll"] == 321.0 and payload["bankroll_source"] == "manual"
        assert payload["sectors"] == ["nba", "wnba"]

        payload = asyncio.run(web_app.run_dashboard_scan(fan_out_portfolios=False))
        assert seen["fan_out_portfolio_ids"] is None
