"""Tests for portfolio management."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from evmax.portfolios import (
    Portfolio,
    PROP_SECTOR_GROUPS,
    SCENARIOS,
    SECTOR_GROUPS,
    create_default_portfolios,
    create_portfolio,
    create_prop_portfolios,
    delete_portfolio,
    drop_voided_portfolio_bets,
    get_portfolio,
    get_portfolio_bets,
    get_portfolio_stats,
    is_excluded_from_portfolio,
    list_portfolios,
    log_portfolio_bet,
    portfolio_outcome_key,
    resolve_portfolio_bet,
    select_best_execution_plays,
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


class TestPropPortfolios:
    """Player-prop portfolios mirror SECTOR_GROUPS/create_default_portfolios
    but over PROP_SECTOR_GROUPS. baseball_props was added alongside the
    coordinator._PROP_SECTORS fix that lets MLB props actually log to
    prop_observations — this locks in that it gets its own portfolio
    section the same way nba_props/nfl_props already do."""

    def test_creates_all_prop_groups(self):
        created = create_prop_portfolios(backfill=False)
        expected = len(PROP_SECTOR_GROUPS) * len(SCENARIOS)
        assert len(created) == expected
        for p in created:
            assert p.scenario in SCENARIOS
            assert p.active is True

    def test_baseball_props_group_present(self):
        assert "baseball_props" in PROP_SECTOR_GROUPS
        assert PROP_SECTOR_GROUPS["baseball_props"]["sectors"] == ["baseball_props"]

        create_prop_portfolios(backfill=False)
        ids = {p.id for p in list_portfolios(active_only=False)}
        for scenario in SCENARIOS:
            assert f"baseball_props_{scenario}" in ids

    def test_idempotent(self):
        create_prop_portfolios(backfill=False)
        create_prop_portfolios(backfill=False)
        portfolios = list_portfolios(active_only=False)
        expected = len(PROP_SECTOR_GROUPS) * len(SCENARIOS)
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
        assert log_portfolio_bet("p1", gap, 100, 0.25) is True
        assert log_portfolio_bet("p1", gap, 100, 0.25) is False
        assert len(get_portfolio_bets("p1")) == 1


class TestOutcomeKey:
    def test_game_event_suffix_stripped(self):
        # ML, spread and total share the game base but differ on market_type/line
        k_ml = portfolio_outcome_key("wnba::2026-07-08::sparks_vs_fever", "moneyline", None)
        k_sp = portfolio_outcome_key("wnba::2026-07-08::sparks_vs_fever::spread", "spread", -4.5)
        assert k_ml == ("wnba::2026-07-08::sparks_vs_fever", "moneyline", 0.0)
        assert k_sp == ("wnba::2026-07-08::sparks_vs_fever", "spread", 4.5)
        assert k_ml != k_sp

    def test_opposite_sides_share_key(self):
        # YES on each team of the same moneyline → same market
        a = portfolio_outcome_key("wnba::2026-07-08::sparks_vs_fever", "moneyline", None)
        b = portfolio_outcome_key("wnba::2026-07-08::sparks_vs_fever", "moneyline", None)
        assert a == b
        # opposite spread sides: mirrored line signs collapse via abs()
        s1 = portfolio_outcome_key("nba::2026-04-18::hawks_vs_knicks::spread", "spread", -4.5)
        s2 = portfolio_outcome_key("nba::2026-04-18::hawks_vs_knicks::spread", "spread", 4.5)
        assert s1 == s2

    def test_alt_lines_stay_distinct(self):
        s1 = portfolio_outcome_key("nba::2026-04-18::hawks_vs_knicks::spread", "spread", -4.5)
        s2 = portfolio_outcome_key("nba::2026-04-18::hawks_vs_knicks::spread", "spread", -5.5)
        assert s1 != s2

    def test_prop_thresholds_stay_distinct(self):
        p1 = portfolio_outcome_key("nba::2026-04-18::prop::j_tatum::points::24.5", "player_prop", 24.5)
        p2 = portfolio_outcome_key("nba::2026-04-18::prop::j_tatum::points::29.5", "player_prop", 29.5)
        assert p1 != p2

    def test_none_inputs_safe(self):
        assert portfolio_outcome_key(None, None, None) == ("", "", 0.0)


class TestBestExecutionSelection:
    def _gap(self, market_id, venue, kelly, ev, event_id="wnba::2026-07-08::sparks_vs_fever",
             market_type="moneyline", line=None):
        return {
            "market_id": market_id, "venue": venue, "kelly_fraction": kelly,
            "ev_pct": ev, "event_id": event_id, "market_type": market_type, "line": line,
        }

    def test_live_leg_beats_shadow_leg_even_at_lower_ev(self):
        kalshi = self._gap("KXWNBAGAME-LAF", "kalshi", 0.03, 0.04)
        polyus = self._gap("polymarket_us:aec-wnba-ind-la:ind", "polymarket_us", 0.0, 0.09)
        kept = select_best_execution_plays([polyus, kalshi])
        assert kept == [kalshi]

    def test_both_shadow_keeps_higher_ev(self):
        a = self._gap("polymarket_us:aec-wnba-ind-la:ind", "polymarket_us", 0.0, 0.09)
        b = self._gap("KXWNBAGAME-LAF", "kalshi", 0.0, 0.04)
        kept = select_best_execution_plays([a, b])
        assert kept == [a]

    def test_both_live_keeps_higher_ev(self):
        a = self._gap("KXWNBAGAME-LAF", "kalshi", 0.03, 0.04)
        b = self._gap("polymarket_us:aec-wnba-ind-la:ind", "polymarket_us", 0.02, 0.06)
        kept = select_best_execution_plays([a, b])
        assert kept == [b]

    def test_distinct_markets_all_survive(self):
        ml = self._gap("KX-ML", "kalshi", 0.03, 0.04)
        total = self._gap("KX-TOT", "kalshi", 0.02, 0.03, market_type="total", line=164.5)
        alt1 = self._gap("KX-SP45", "kalshi", 0.02, 0.03, market_type="spread", line=-4.5)
        alt2 = self._gap("KX-SP55", "kalshi", 0.02, 0.05, market_type="spread", line=-5.5)
        other_game = self._gap("KX-OTHER", "kalshi", 0.03, 0.04,
                               event_id="wnba::2026-07-08::sun_vs_lynx")
        kept = select_best_execution_plays([ml, total, alt1, alt2, other_game])
        assert kept == [ml, total, alt1, alt2, other_game]

    def test_accepts_evgap_like_objects(self):
        from types import SimpleNamespace

        kalshi = SimpleNamespace(
            market_id="KXWNBAGAME-LAF", venue="kalshi", kelly_fraction=0.03,
            ev_pct=0.04, event_id="wnba::2026-07-08::sparks_vs_fever",
            market_type="moneyline", line=None,
        )
        polyus = SimpleNamespace(
            market_id="polymarket_us:aec-wnba-ind-la:ind", venue="polymarket_us",
            kelly_fraction=0.0, ev_pct=0.09, event_id="wnba::2026-07-08::sparks_vs_fever",
            market_type="moneyline", line=None,
        )
        kept = select_best_execution_plays([polyus, kalshi])
        assert kept == [kalshi]


class TestLedgerGuards:
    GAME = "wnba::2026-07-08::sparks_vs_fever"

    def _gap(self, market_id, **over):
        gap = {
            "market_id": market_id,
            "event_id": self.GAME,
            "sector": "wnba",
            "yes_team": "Fever",
            "market_type": "moneyline",
            "ev_pct": 0.05,
            "kelly_fraction": 0.03,
            "scan_date": "2026-07-08",
        }
        gap.update(over)
        return gap

    def test_zero_stake_leg_never_logged(self):
        create_portfolio("wnba_m", "WNBA Mod", ["wnba"], 250, 0.50)
        shadow = self._gap("polymarket_us:aec-wnba-ind-la:ind", kelly_fraction=0.0)
        assert log_portfolio_bet("wnba_m", shadow, 250, 0.50) is False
        assert get_portfolio_bets("wnba_m") == []

    def test_cross_venue_twin_blocked(self):
        create_portfolio("wnba_m", "WNBA Mod", ["wnba"], 250, 0.50)
        assert log_portfolio_bet("wnba_m", self._gap("KXWNBAGAME-LAF-IND"), 250, 0.50) is True
        twin = self._gap("polymarket_us:aec-wnba-ind-la:ind")
        assert log_portfolio_bet("wnba_m", twin, 250, 0.50) is False
        bets = get_portfolio_bets("wnba_m")
        assert len(bets) == 1
        assert bets[0]["market_id"] == "KXWNBAGAME-LAF-IND"

    def test_opposite_side_later_scan_blocked(self):
        # the ind/la case: both sides of the same ML accreting across scan days
        create_portfolio("wnba_m", "WNBA Mod", ["wnba"], 250, 0.50)
        assert log_portfolio_bet("wnba_m", self._gap("KXWNBAGAME-LAF-IND"), 250, 0.50) is True
        other_side = self._gap(
            "KXWNBAGAME-LAF-LA", yes_team="Los Angeles", scan_date="2026-07-09"
        )
        assert log_portfolio_bet("wnba_m", other_side, 250, 0.50) is False
        assert len(get_portfolio_bets("wnba_m")) == 1

    def test_distinct_markets_on_same_game_allowed(self):
        create_portfolio("wnba_m", "WNBA Mod", ["wnba"], 250, 0.50)
        assert log_portfolio_bet("wnba_m", self._gap("KX-ML"), 250, 0.50) is True
        spread = self._gap(
            "KX-SP", event_id=f"{self.GAME}::spread", market_type="spread", line=-4.5
        )
        assert log_portfolio_bet("wnba_m", spread, 250, 0.50) is True
        alt = self._gap(
            "KX-SP-ALT", event_id=f"{self.GAME}::spread", market_type="spread", line=-5.5
        )
        assert log_portfolio_bet("wnba_m", alt, 250, 0.50) is True
        assert len(get_portfolio_bets("wnba_m")) == 3

    def test_guard_scoped_per_portfolio(self):
        create_portfolio("wnba_a", "WNBA Agg", ["wnba"], 500, 1.0)
        create_portfolio("wnba_c", "WNBA Con", ["wnba"], 100, 0.25)
        gap = self._gap("KXWNBAGAME-LAF-IND")
        assert log_portfolio_bet("wnba_a", gap, 500, 1.0) is True
        assert log_portfolio_bet("wnba_c", gap, 100, 0.25) is True


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
