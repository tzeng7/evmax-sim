"""Tests for portfolio management."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from evmax.portfolios import (
    Portfolio,
    SCENARIOS,
    SECTOR_GROUPS,
    create_default_portfolios,
    create_portfolio,
    delete_portfolio,
    drop_voided_portfolio_bets,
    get_portfolio,
    get_portfolio_bets,
    get_portfolio_stats,
    is_excluded_from_portfolio,
    list_portfolios,
    log_portfolio_bet,
    resolve_portfolio_bet,
    sync_portfolio_outcomes,
    _get_conn,
)


@pytest.fixture(autouse=True)
def _patch_db(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "test_predictions.db"
    monkeypatch.setattr("evmax.portfolios.DB_PATH", db_path)
    monkeypatch.setattr("evmax.agents.cleanup.db.DB_PATH", db_path)


class TestCreatePortfolio:
    def test_create_and_retrieve(self):
        p = create_portfolio("test_1", "Test One", ["nba"], 100, 0.25, "conservative")
        assert p.id == "test_1"
        assert p.sectors == ["nba"]
        assert p.current_bankroll == 100

        fetched = get_portfolio("test_1")
        assert fetched is not None
        assert fetched.name == "Test One"

    def test_list_active_only(self):
        create_portfolio("a1", "A1", ["nba"], 100, 0.25)
        create_portfolio("a2", "A2", ["soccer"], 200, 0.50)
        delete_portfolio("a2")

        active = list_portfolios(active_only=True)
        assert len(active) == 1
        assert active[0].id == "a1"

        all_p = list_portfolios(active_only=False)
        assert len(all_p) == 2


class TestDefaultPortfolios:
    def test_creates_all_combinations(self):
        created = create_default_portfolios()
        expected = len(SECTOR_GROUPS) * len(SCENARIOS)
        assert len(created) == expected

        for p in created:
            assert p.scenario in SCENARIOS
            assert p.active is True

    def test_idempotent(self):
        create_default_portfolios()
        create_default_portfolios()
        portfolios = list_portfolios(active_only=False)
        expected = len(SECTOR_GROUPS) * len(SCENARIOS)
        assert len(portfolios) == expected


class TestLogBet:
    def test_log_and_retrieve(self):
        create_portfolio("nba_mod", "NBA Mod", ["nba"], 250, 0.50)
        gap = {
            "market_id": "KXNBA-123",
            "event_id": "nba::2026-04-18::test",
            "event_title": "Celtics vs Lakers",
            "yes_team": "Celtics",
            "market_type": "moneyline",
            "display_label": "Celtics ML",
            "sector": "nba",
            "kalshi_yes_price": 0.45,
            "true_prob": 0.55,
            "ev_pct": 0.08,
            "kelly_fraction": 0.04,
            "event_date": "2026-04-18",
            "volume": 5000,
            "model_sources": "elo,form",
            "scan_date": "2026-04-18",
        }
        log_portfolio_bet("nba_mod", gap, 250, 0.50)

        bets = get_portfolio_bets("nba_mod")
        assert len(bets) == 1
        assert bets[0]["market_id"] == "KXNBA-123"
        assert bets[0]["stake"] == round(250 * 0.50 * 0.04, 2)

    def test_dedup(self):
        create_portfolio("p1", "P1", ["nba"], 100, 0.25)
        gap = {"market_id": "MKT-1", "sector": "nba", "ev_pct": 0.05, "kelly_fraction": 0.02}
        log_portfolio_bet("p1", gap, 100, 0.25)
        log_portfolio_bet("p1", gap, 100, 0.25)
        assert len(get_portfolio_bets("p1")) == 1


class TestResolution:
    def test_resolve_updates_bankroll(self):
        create_portfolio("p_res", "PRes", ["nba"], 100, 0.50)
        gap = {
            "market_id": "MKT-WIN",
            "sector": "nba",
            "kalshi_yes_price": 0.50,
            "ev_pct": 0.10,
            "kelly_fraction": 0.04,
        }
        log_portfolio_bet("p_res", gap, 100, 0.50)

        resolve_portfolio_bet("p_res", "MKT-WIN", 1)

        bets = get_portfolio_bets("p_res")
        assert bets[0]["outcome"] == 1
        assert bets[0]["pnl"] > 0

        p = get_portfolio("p_res")
        assert p.current_bankroll > 100

    def test_resolve_loss(self):
        create_portfolio("p_loss", "PLoss", ["soccer"], 200, 0.25)
        gap = {
            "market_id": "MKT-LOSS",
            "sector": "soccer",
            "kalshi_yes_price": 0.40,
            "ev_pct": 0.05,
            "kelly_fraction": 0.03,
        }
        log_portfolio_bet("p_loss", gap, 200, 0.25)
        resolve_portfolio_bet("p_loss", "MKT-LOSS", 0)

        bets = get_portfolio_bets("p_loss")
        assert bets[0]["outcome"] == 0
        assert bets[0]["pnl"] < 0

        p = get_portfolio("p_loss")
        assert p.current_bankroll < 200


class TestStats:
    def test_empty_portfolio(self):
        create_portfolio("empty", "Empty", ["tennis"], 500, 1.0)
        stats = get_portfolio_stats("empty")
        assert stats["total_bets"] == 0
        assert stats["win_rate"] == 0

    def test_with_bets(self):
        create_portfolio("stats_test", "Stats", ["nba"], 100, 0.50)
        for i in range(3):
            gap = {"market_id": f"ST-{i}", "sector": "nba", "kalshi_yes_price": 0.50, "ev_pct": 0.05, "kelly_fraction": 0.02}
            log_portfolio_bet("stats_test", gap, 100, 0.50)

        resolve_portfolio_bet("stats_test", "ST-0", 1)
        resolve_portfolio_bet("stats_test", "ST-1", 0)

        stats = get_portfolio_stats("stats_test")
        assert stats["total_bets"] == 3
        assert stats["open_bets"] == 1
        assert stats["settled_bets"] == 2
        assert stats["wins"] == 1
        assert stats["losses"] == 1


class TestPortfolioModel:
    def test_to_dict(self):
        p = Portfolio(
            id="test", name="Test", sectors=["nba"],
            initial_bankroll=100, current_bankroll=105.50,
            kelly_fraction=0.25, scenario="conservative",
        )
        d = p.to_dict()
        assert d["id"] == "test"
        assert d["current_bankroll"] == 105.50
        assert d["sectors"] == ["nba"]


class TestPortfolioExclusions:
    """Baseball totals are kept out of portfolio simulation (−CLV from a
    night-before-only scan workflow; see categories.yaml baseball notes)."""

    def test_baseball_total_excluded(self):
        assert is_excluded_from_portfolio("baseball", "total") is True
        assert is_excluded_from_portfolio("BASEBALL", "Total") is True  # case-insensitive

    def test_baseball_other_markets_not_excluded(self):
        assert is_excluded_from_portfolio("baseball", "moneyline") is False
        assert is_excluded_from_portfolio("baseball", "spread") is False

    def test_other_sectors_totals_not_excluded(self):
        # The exclusion is baseball-specific, not a blanket totals ban.
        assert is_excluded_from_portfolio("nba", "total") is False
        assert is_excluded_from_portfolio("soccer", "total") is False

    def test_none_inputs_safe(self):
        assert is_excluded_from_portfolio(None, None) is False


class TestVoidedBetDrop:
    """A cancelled match settles on Kalshi as a scalar fair-price refund:
    the cleanup resolver marks ev_predictions.voided=1 and writes NO
    ev_outcomes row. Such portfolio bets can never resolve through the
    ev_outcomes JOIN, so they must be dropped from the open ledger.
    Regression for the Paul vs Mpetshi Perricard (2026-06-08) stuck-open bug.
    """

    def _seed_prediction(self, market_id, *, voided, with_outcome):
        """Create the predictions schema and a matching ev_predictions row."""
        from evmax.agents.cleanup.db import get_connection

        conn = get_connection()
        conn.execute(
            """INSERT OR REPLACE INTO ev_predictions
               (scan_date, market_id, event_id, sector, yes_team, market_type,
                kalshi_yes_price, sharp_true_prob, blended_true_prob, ev_pct,
                kelly_fraction, voided)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("2026-06-08", market_id, "tennis::x", "tennis", "perricard",
             "moneyline", 0.4, 0.5, 0.5, 0.05, 0.02, 1 if voided else 0),
        )
        if with_outcome:
            conn.execute(
                """INSERT OR REPLACE INTO ev_outcomes
                   (market_id, event_id, event_date, sector, yes_team, outcome)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (market_id, "tennis::x", "2026-06-08", "tennis", "perricard", 1),
            )
        conn.commit()
        conn.close()

    def _open_bet(self, market_id):
        log_portfolio_bet(
            "tennis_aggressive",
            {
                "market_id": market_id,
                "event_id": "tennis::x",
                "sector": "tennis",
                "yes_team": "perricard",
                "market_type": "moneyline",
                "event_title": "Tommy Paul vs Giovanni Mpetshi Perricard",
                "event_date": "2026-06-08",
                "kalshi_yes_price": 0.4,
                "kelly_fraction": 0.02,
            },
            bankroll=500,
            kelly=1.0,
        )

    def test_drops_voided_open_bet(self):
        create_portfolio("tennis_aggressive", "Tennis Agg", ["tennis"], 500, 1.0)
        self._seed_prediction("kalshi:VOID", voided=True, with_outcome=False)
        self._open_bet("kalshi:VOID")

        assert len(get_portfolio_bets("tennis_aggressive", "open")) == 1
        dropped = drop_voided_portfolio_bets()
        assert dropped == 1
        assert get_portfolio_bets("tennis_aggressive", "open") == []

    def test_keeps_live_open_bet(self):
        """A non-voided market still pending must NOT be dropped."""
        create_portfolio("tennis_aggressive", "Tennis Agg", ["tennis"], 500, 1.0)
        self._seed_prediction("kalshi:LIVE", voided=False, with_outcome=False)
        self._open_bet("kalshi:LIVE")

        assert drop_voided_portfolio_bets() == 0
        assert len(get_portfolio_bets("tennis_aggressive", "open")) == 1

    def test_keeps_voided_bet_that_has_outcome(self):
        """If an ev_outcomes row exists, normal resolution owns it — not drop."""
        create_portfolio("tennis_aggressive", "Tennis Agg", ["tennis"], 500, 1.0)
        self._seed_prediction("kalshi:BOTH", voided=True, with_outcome=True)
        self._open_bet("kalshi:BOTH")

        assert drop_voided_portfolio_bets() == 0
        assert len(get_portfolio_bets("tennis_aggressive", "open")) == 1

    def test_no_predictions_table_is_safe(self):
        """Fresh portfolio-only DB (no predictions schema) returns 0, no error."""
        create_portfolio("tennis_aggressive", "Tennis Agg", ["tennis"], 500, 1.0)
        assert drop_voided_portfolio_bets() == 0
