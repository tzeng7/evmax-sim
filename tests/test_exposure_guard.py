"""Tests for the per-game exposure guard.

Two layers:
  1. _apply_exposure_guard — pure function over an EVGap list + prior dict
  2. _load_placed_exposure — reads ev_predictions for placed un-resolved bets

The cumulative-cap behavior matters most when the user scans the same
game multiple times in a day: new bets must respect what was already
placed earlier without auto-coordinating across scans.
"""

from __future__ import annotations

import dataclasses
import sqlite3
from datetime import datetime, timezone
from typing import Any

import pytest

from evmax.agents.coordinator import (
    _apply_exposure_guard,
    _apply_joint_kelly,
    _base_event,
    _load_placed_exposure,
)
from evmax.agents.odds.ev_gap_agent import EVGap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gap(
    market_id: str,
    event_id: str,
    kelly: float = 0.025,
    ev: float = 0.05,
    yes_team: str = "lakers",
) -> EVGap:
    return EVGap(
        market_id=market_id,
        event_id=event_id,
        sector="nba",
        yes_team=yes_team,
        market_type="moneyline",
        kalshi_yes_price=0.45,
        sharp_true_prob=0.50,
        blended_true_prob=0.50,
        ev_pct=ev,
        kelly_full=kelly * 2,
        kelly_fraction=kelly,
        match_confidence=95.0,
        volume_usd=5_000.0,
        spread_pct=0.02,
        event_date=datetime(2026, 4, 30, 19, 0, tzinfo=timezone.utc),
        model_sources="sharp",
        line=None,
        event_title="Lakers vs Warriors",
    )


# ---------------------------------------------------------------------------
# _base_event
# ---------------------------------------------------------------------------


class TestBaseEvent:
    def test_strips_spread_suffix(self):
        assert _base_event("nba::2026-04-30::lakers_vs_warriors::spread") == \
               "nba::2026-04-30::lakers_vs_warriors"

    def test_strips_total_suffix(self):
        assert _base_event("nba::2026-04-30::lakers_vs_warriors::total") == \
               "nba::2026-04-30::lakers_vs_warriors"

    def test_moneyline_unchanged(self):
        assert _base_event("nba::2026-04-30::lakers_vs_warriors") == \
               "nba::2026-04-30::lakers_vs_warriors"

    def test_prop_groups_by_player(self):
        # Different stat/line on the SAME player → same base
        a = _base_event("nba::2026-04-30::prop::lebron::points::24.5")
        b = _base_event("nba::2026-04-30::prop::lebron::rebounds::8.5")
        assert a == b == "nba::2026-04-30::prop::lebron"


# ---------------------------------------------------------------------------
# _apply_exposure_guard
# ---------------------------------------------------------------------------


class TestApplyExposureGuard:
    def test_no_prior_unchanged_behavior(self):
        """Empty prior_exposure → identical to original behavior."""
        gaps = [
            _gap("kalshi:A", "nba::2026-04-30::lakers_vs_warriors", kelly=0.04),
            _gap("kalshi:B", "nba::2026-04-30::celtics_vs_heat", kelly=0.05),
        ]
        out = _apply_exposure_guard(gaps, prior_exposure={})
        assert len(out) == 2
        assert all(g.kelly_fraction in (0.04, 0.05) for g in out)

    def test_prior_consumes_budget_scales_new_bet(self):
        """Prior 4% on a game leaves 4% for new bets; 5% gap scales to 4%."""
        gap = _gap("kalshi:NEW", "nba::2026-04-30::lakers_vs_warriors", kelly=0.05)
        prior = {"nba::2026-04-30::lakers_vs_warriors": 0.04}
        out = _apply_exposure_guard([gap], prior_exposure=prior)
        assert len(out) == 1
        assert out[0].kelly_fraction == pytest.approx(0.04, abs=1e-4)

    def test_prior_at_cap_drops_new_bet(self):
        """Prior 8% (full cap) → new bet on same game is dropped."""
        gap = _gap("kalshi:NEW", "nba::2026-04-30::lakers_vs_warriors", kelly=0.03)
        prior = {"nba::2026-04-30::lakers_vs_warriors": 0.08}
        out = _apply_exposure_guard([gap], prior_exposure=prior)
        assert out == []

    def test_prior_on_different_game_no_impact(self):
        """Prior on game X doesn't restrict new bets on game Y."""
        gap = _gap("kalshi:NEW", "nba::2026-04-30::lakers_vs_warriors", kelly=0.05)
        prior = {"nba::2026-04-30::celtics_vs_heat": 0.06}
        out = _apply_exposure_guard([gap], prior_exposure=prior)
        assert len(out) == 1
        assert out[0].kelly_fraction == pytest.approx(0.05, abs=1e-4)

    def test_prior_on_ml_caps_new_spread_same_game(self):
        """Prior placement on the ML market should cap new spread bets on
        the same matchup — they share the base_event budget."""
        # Prior was on ML (no suffix); new is a spread (::spread suffix)
        spread_gap = _gap(
            "kalshi:SPREAD",
            "nba::2026-04-30::lakers_vs_warriors::spread",
            kelly=0.05,
        )
        prior = {"nba::2026-04-30::lakers_vs_warriors": 0.05}
        out = _apply_exposure_guard([spread_gap], prior_exposure=prior)
        assert len(out) == 1
        assert out[0].kelly_fraction == pytest.approx(0.03, abs=1e-4)

    def test_multiple_new_bets_consume_remaining_in_ev_order(self):
        """When several new bets fit in remaining budget, highest-EV first."""
        prior = {"nba::2026-04-30::lakers_vs_warriors": 0.05}  # 3% remaining
        gaps = [
            _gap("kalshi:LO_EV", "nba::2026-04-30::lakers_vs_warriors", kelly=0.02, ev=0.03,
                 yes_team="warriors"),
            _gap("kalshi:HI_EV", "nba::2026-04-30::lakers_vs_warriors", kelly=0.02, ev=0.10,
                 yes_team="lakers"),
        ]
        out = _apply_exposure_guard(gaps, prior_exposure=prior)
        # HI_EV should fit (0.02 ≤ 0.03 remaining), LO_EV scales to 0.01
        markets = {g.market_id: g.kelly_fraction for g in out}
        assert markets["kalshi:HI_EV"] == pytest.approx(0.02, abs=1e-4)
        assert markets["kalshi:LO_EV"] == pytest.approx(0.01, abs=1e-4)


# ---------------------------------------------------------------------------
# Same-side correlation discount
# ---------------------------------------------------------------------------


class TestSameSideDiscount:
    def test_same_side_second_gap_halved(self):
        """ML + same-team spread → second-best gets 0.5x Kelly."""
        gaps = [
            _gap("kalshi:ML", "nba::2026-04-30::lakers_vs_warriors",
                 kelly=0.04, ev=0.10, yes_team="lakers"),
            _gap("kalshi:SPREAD", "nba::2026-04-30::lakers_vs_warriors::spread",
                 kelly=0.03, ev=0.06, yes_team="lakers"),
        ]
        out = _apply_exposure_guard(gaps, prior_exposure={})
        sized = {g.market_id: g.kelly_fraction for g in out}
        assert sized["kalshi:ML"] == pytest.approx(0.04, abs=1e-4)
        assert sized["kalshi:SPREAD"] == pytest.approx(0.015, abs=1e-4)

    def test_opposite_side_no_discount(self):
        """ML on team A + spread on team B → no discount (hedging pair)."""
        gaps = [
            _gap("kalshi:ML_FAV", "nba::2026-04-30::lakers_vs_warriors",
                 kelly=0.04, ev=0.10, yes_team="lakers"),
            _gap("kalshi:SPREAD_DOG", "nba::2026-04-30::lakers_vs_warriors::spread",
                 kelly=0.03, ev=0.06, yes_team="warriors"),
        ]
        out = _apply_exposure_guard(gaps, prior_exposure={})
        sized = {g.market_id: g.kelly_fraction for g in out}
        assert sized["kalshi:ML_FAV"] == pytest.approx(0.04, abs=1e-4)
        assert sized["kalshi:SPREAD_DOG"] == pytest.approx(0.03, abs=1e-4)

    def test_three_same_side_gaps_each_discounted_off_original(self):
        """Three same-side bets: leader keeps full, others each get 0.5x of own (not compound)."""
        gaps = [
            _gap("kalshi:A", "nba::2026-04-30::lakers_vs_warriors",
                 kelly=0.03, ev=0.10, yes_team="lakers"),
            _gap("kalshi:B", "nba::2026-04-30::lakers_vs_warriors::spread",
                 kelly=0.02, ev=0.07, yes_team="lakers"),
            _gap("kalshi:C", "nba::2026-04-30::lakers_vs_warriors::alt_spread",
                 kelly=0.02, ev=0.05, yes_team="lakers"),
        ]
        out = _apply_exposure_guard(gaps, prior_exposure={})
        sized = {g.market_id: g.kelly_fraction for g in out}
        assert sized["kalshi:A"] == pytest.approx(0.03, abs=1e-4)
        assert sized["kalshi:B"] == pytest.approx(0.01, abs=1e-4)
        assert sized["kalshi:C"] == pytest.approx(0.01, abs=1e-4)

    def test_discount_applies_before_cap(self):
        """A 6% same-side bet behind a 4% leader becomes 3% (fits) — not 4% (capped)."""
        gaps = [
            _gap("kalshi:LEAD", "nba::2026-04-30::lakers_vs_warriors",
                 kelly=0.04, ev=0.10, yes_team="lakers"),
            _gap("kalshi:NEXT", "nba::2026-04-30::lakers_vs_warriors::spread",
                 kelly=0.06, ev=0.06, yes_team="lakers"),
        ]
        out = _apply_exposure_guard(gaps, prior_exposure={})
        sized = {g.market_id: g.kelly_fraction for g in out}
        assert sized["kalshi:LEAD"] == pytest.approx(0.04, abs=1e-4)
        # 0.06 × 0.5 = 0.03 → fits in remaining 0.04, doesn't get capped to 0.04
        assert sized["kalshi:NEXT"] == pytest.approx(0.03, abs=1e-4)

    def test_discount_disabled_at_one_keeps_original(self):
        """same_side_kelly_discount=1.0 → no discount applied."""
        gaps = [
            _gap("kalshi:A", "nba::2026-04-30::lakers_vs_warriors",
                 kelly=0.03, ev=0.10, yes_team="lakers"),
            _gap("kalshi:B", "nba::2026-04-30::lakers_vs_warriors::spread",
                 kelly=0.02, ev=0.06, yes_team="lakers"),
        ]
        out = _apply_exposure_guard(
            gaps, prior_exposure={}, same_side_kelly_discount=1.0,
        )
        sized = {g.market_id: g.kelly_fraction for g in out}
        assert sized["kalshi:A"] == pytest.approx(0.03, abs=1e-4)
        assert sized["kalshi:B"] == pytest.approx(0.02, abs=1e-4)


# ---------------------------------------------------------------------------
# _load_placed_exposure
# ---------------------------------------------------------------------------


def _setup_test_db() -> sqlite3.Connection:
    """Build an in-memory ev_predictions + ev_outcomes DB."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE ev_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_date TEXT NOT NULL,
            market_id TEXT NOT NULL,
            event_id TEXT,
            kelly_fraction REAL,
            placed_stake REAL,
            bankroll_used REAL,
            placed INTEGER NOT NULL DEFAULT 0,
            voided INTEGER NOT NULL DEFAULT 0,
            mode TEXT NOT NULL DEFAULT 'live'
        );
        CREATE TABLE ev_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_id TEXT NOT NULL,
            outcome INTEGER
        );
        """
    )
    return conn


def _insert(conn, **kw):
    cols = ",".join(kw.keys())
    qs = ",".join("?" * len(kw))
    conn.execute(f"INSERT INTO ev_predictions ({cols}) VALUES ({qs})", tuple(kw.values()))
    conn.commit()


@pytest.fixture
def patched_conn(monkeypatch):
    conn = _setup_test_db()

    class _Ctx:
        def __init__(self, inner):
            self._inner = inner
        def __enter__(self):
            return self._inner
        def __exit__(self, *a):
            return False
        def __getattr__(self, name):
            return getattr(self._inner, name)

    ctx = _Ctx(conn)
    monkeypatch.setattr(
        "evmax.agents.cleanup.db.get_connection", lambda: ctx,
    )
    yield conn
    conn.close()


class TestLoadPlacedExposure:
    def test_empty_db_returns_empty_dict(self, patched_conn):
        assert _load_placed_exposure() == {}

    def test_sums_placed_unresolved_live_bets(self, patched_conn):
        # dollars = kelly * bankroll_used when no placed_stake: 0.03*1000 + 0.02*1000
        _insert(patched_conn,
                scan_date="2026-04-30",
                market_id="kalshi:A",
                event_id="nba::2026-04-30::lakers_vs_warriors",
                kelly_fraction=0.03, bankroll_used=1000.0,
                placed=1, voided=0, mode="live")
        _insert(patched_conn,
                scan_date="2026-04-30",
                market_id="kalshi:B",
                event_id="nba::2026-04-30::lakers_vs_warriors::spread",
                kelly_fraction=0.02, bankroll_used=1000.0,
                placed=1, voided=0, mode="live")
        out = _load_placed_exposure()
        # Both bets share the same base_event; total = $50 (not "0.05")
        assert out == {"nba::2026-04-30::lakers_vs_warriors": pytest.approx(50.0)}

    def test_mixed_bankroll_bases_sum_in_dollars(self, patched_conn):
        """GAP 1 regression: a 5% bet on a $200 base and a 5% bet on an $800
        base are $10 and $40 — the guard must see $50, never a raw fraction
        sum of 0.10. This is the bug that broke the cap across venues with
        unequal balances."""
        _insert(patched_conn,
                scan_date="2026-04-30",
                market_id="polymarket_us:A",
                event_id="wnba::2026-04-30::aces_vs_liberty",
                kelly_fraction=0.05, bankroll_used=200.0,
                placed=1, voided=0, mode="live")
        _insert(patched_conn,
                scan_date="2026-04-30",
                market_id="kalshi:A",
                event_id="wnba::2026-04-30::aces_vs_liberty",
                kelly_fraction=0.05, bankroll_used=800.0,
                placed=1, voided=0, mode="live")
        out = _load_placed_exposure()
        assert out == {"wnba::2026-04-30::aces_vs_liberty": pytest.approx(50.0)}

    def test_prefers_placed_stake_over_kelly_times_bankroll(self, patched_conn):
        """placed_stake is the REAL amount staked (may differ from the sized
        kelly*bankroll if the user placed a custom amount) — it wins."""
        _insert(patched_conn,
                scan_date="2026-04-30",
                market_id="kalshi:A",
                event_id="nba::2026-04-30::lakers_vs_warriors",
                kelly_fraction=0.05, bankroll_used=1000.0, placed_stake=42.0,
                placed=1, voided=0, mode="live")
        out = _load_placed_exposure()
        # $42 (placed_stake), NOT $50 (kelly*bankroll)
        assert out == {"nba::2026-04-30::lakers_vs_warriors": pytest.approx(42.0)}

    def test_skips_row_without_stake_or_bankroll(self, patched_conn):
        """A placed row with neither placed_stake nor bankroll_used can't be
        valued in dollars, so it's skipped rather than counted as a bare
        fraction (normal rows always carry bankroll_used)."""
        _insert(patched_conn,
                scan_date="2026-04-30",
                market_id="kalshi:A",
                event_id="nba::2026-04-30::lakers_vs_warriors",
                kelly_fraction=0.05,
                placed=1, voided=0, mode="live")
        assert _load_placed_exposure() == {}

    def test_excludes_unplaced_bets(self, patched_conn):
        _insert(patched_conn,
                scan_date="2026-04-30",
                market_id="kalshi:UNPLACED",
                event_id="nba::2026-04-30::lakers_vs_warriors",
                kelly_fraction=0.05, placed=0, voided=0, mode="live")
        assert _load_placed_exposure() == {}

    def test_excludes_voided_bets(self, patched_conn):
        _insert(patched_conn,
                scan_date="2026-04-30",
                market_id="kalshi:VOID",
                event_id="nba::2026-04-30::lakers_vs_warriors",
                kelly_fraction=0.05, placed=1, voided=1, mode="live")
        assert _load_placed_exposure() == {}

    def test_excludes_shadow_bets(self, patched_conn):
        _insert(patched_conn,
                scan_date="2026-04-30",
                market_id="kalshi:SHADOW",
                event_id="nba::2026-04-30::lakers_vs_warriors",
                kelly_fraction=0.05, placed=1, voided=0, mode="shadow")
        assert _load_placed_exposure() == {}

    def test_excludes_resolved_bets(self, patched_conn):
        _insert(patched_conn,
                scan_date="2026-04-30",
                market_id="kalshi:DONE",
                event_id="nba::2026-04-30::lakers_vs_warriors",
                kelly_fraction=0.05, placed=1, voided=0, mode="live")
        # Resolution row → outcome is set, so should be excluded
        patched_conn.execute(
            "INSERT INTO ev_outcomes (market_id, outcome) VALUES (?, ?)",
            ("kalshi:DONE", 1),
        )
        patched_conn.commit()
        assert _load_placed_exposure() == {}

    def test_groups_ml_and_spread_same_game(self, patched_conn):
        """ML + spread bets on the same matchup share the base_event key."""
        _insert(patched_conn,
                scan_date="2026-04-30",
                market_id="kalshi:ML",
                event_id="nba::2026-04-30::knicks_vs_hawks",
                kelly_fraction=0.025, bankroll_used=1000.0,
                placed=1, voided=0, mode="live")
        _insert(patched_conn,
                scan_date="2026-04-30",
                market_id="kalshi:SPREAD",
                event_id="nba::2026-04-30::knicks_vs_hawks::spread",
                kelly_fraction=0.030, bankroll_used=1000.0,
                placed=1, voided=0, mode="live")
        out = _load_placed_exposure()
        # $25 + $30 = $55 on the shared base_event
        assert out == {"nba::2026-04-30::knicks_vs_hawks": pytest.approx(55.0)}


# ---------------------------------------------------------------------------
# _apply_joint_kelly grouping key (GAP 5)
# ---------------------------------------------------------------------------


class TestJointKellyGrouping:
    def test_ml_and_spread_same_game_are_one_joint_event(self, monkeypatch):
        """GAP 5: ML + spread of ONE game must size as a single joint event
        (shared gross cap), not two independent events. Assert
        joint_kelly_fractions is called ONCE with both legs — not twice with
        one leg each (the old split(PROP_MARKER)[0] key left ::spread on, so
        the two never grouped)."""
        import evmax.ev.joint_kelly as jk

        calls: list[int] = []

        class _R:
            def __init__(self, n: int):
                self.fractions = [0.03] * n

        def _fake(legs, config):
            calls.append(len(legs))
            return _R(len(legs))

        monkeypatch.setattr(jk, "joint_kelly_fractions", _fake)

        ml = _gap("kalshi:ML", "nba::2026-04-30::lakers_vs_warriors", yes_team="lakers")
        spread = dataclasses.replace(
            _gap("kalshi:SPREAD", "nba::2026-04-30::lakers_vs_warriors::spread",
                 yes_team="lakers"),
            market_type="spread", line=-4.5,
        )
        out = _apply_joint_kelly([ml, spread])
        assert calls == [2]           # one event, both legs — not [1, 1]
        assert len(out) == 2

    def test_different_games_are_separate_joint_events(self, monkeypatch):
        """Two distinct games must remain separate joint events (one call
        each), so grouping doesn't over-merge unrelated bets."""
        import evmax.ev.joint_kelly as jk

        calls: list[int] = []

        class _R:
            def __init__(self, n: int):
                self.fractions = [0.03] * n

        def _fake(legs, config):
            calls.append(len(legs))
            return _R(len(legs))

        monkeypatch.setattr(jk, "joint_kelly_fractions", _fake)

        g1 = _gap("kalshi:G1", "nba::2026-04-30::lakers_vs_warriors", yes_team="lakers")
        g2 = _gap("kalshi:G2", "nba::2026-04-30::knicks_vs_hawks", yes_team="knicks")
        _apply_joint_kelly([g1, g2])
        assert sorted(calls) == [1, 1]
