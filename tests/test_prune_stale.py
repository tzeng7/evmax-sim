"""Tests for the stale-position pruner (`evmax cleanup prune-stale`).

Covers the candidate query, the pure void/un-void decision (with the fail-safe
guards + hysteresis dead-band), the DB write, the void_reason migration, the
shared reprice_rows engine's new keys + moneyline reblend guard, and command
registration.
"""

from __future__ import annotations

import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from evmax.agents.cleanup.stale_positions import (
    STALE_VOID_REASON,
    apply_actions,
    decide_actions,
    evaluate_candidates,
    prune_candidates,
)
from evmax.ev.calculator import tiered_min_ev

_NOW = datetime(2026, 8, 8, 18, 0, tzinfo=timezone.utc)
_SOON = _NOW + timedelta(hours=2)      # upcoming, inside a 12h lookahead
_STARTED = _NOW - timedelta(hours=1)   # already tipped off
_FAR = _NOW + timedelta(hours=20)      # beyond a 12h lookahead


# --------------------------------------------------------------------------- #
# prune_candidates — the candidate query                                      #
# --------------------------------------------------------------------------- #

_COLS = (
    "id, market_id, event_id, event_title, event_date, yes_team, sector, "
    "market_type, line, venue, kalshi_yes_price, sharp_true_prob, "
    "blended_true_prob, ev_pct, sharp_weight_used, voided, void_reason, "
    "placed, mode, scan_date"
)


def _make_pred_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE ev_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_id TEXT, event_id TEXT, event_title TEXT, event_date TEXT,
            yes_team TEXT, sector TEXT, market_type TEXT, line REAL, venue TEXT,
            kalshi_yes_price REAL, sharp_true_prob REAL, blended_true_prob REAL,
            ev_pct REAL, sharp_weight_used REAL,
            voided INTEGER NOT NULL DEFAULT 0, void_reason TEXT,
            placed INTEGER NOT NULL DEFAULT 0,
            mode TEXT NOT NULL DEFAULT 'live', scan_date TEXT
        )
        """
    )
    conn.execute("CREATE TABLE ev_outcomes (id INTEGER PRIMARY KEY, market_id TEXT, outcome INTEGER)")
    return conn


def _insert_pred(conn, market_id, *, placed=0, mode="live", voided=0,
                 void_reason=None, event_date="2026-08-08", market_type="moneyline"):
    conn.execute(
        f"INSERT INTO ev_predictions ({_COLS}) VALUES "
        "(NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            market_id, f"evt:{market_id}", "A vs B", event_date, "A", "nba",
            market_type, None, "kalshi", 0.50, 0.60, 0.60, 0.08, 0.85,
            voided, void_reason, placed, mode, event_date,
        ),
    )
    conn.commit()


def test_prune_candidates_scope():
    conn = _make_pred_db()
    _insert_pred(conn, "open_live")                                    # included
    _insert_pred(conn, "placed", placed=1)                            # excluded (picked)
    _insert_pred(conn, "shadow", mode="shadow")                       # excluded (shadow)
    _insert_pred(conn, "resolved")                                    # excluded (resolved)
    conn.execute("INSERT INTO ev_outcomes (market_id, outcome) VALUES ('resolved', 1)")
    _insert_pred(conn, "manual_void", voided=1, void_reason=None)     # excluded (cancel void)
    _insert_pred(conn, "pruner_void", voided=1, void_reason=STALE_VOID_REASON)  # included (for un-void)
    _insert_pred(conn, "past", event_date="2026-08-01")              # excluded (before today)
    conn.commit()

    ids = {r["market_id"] for r in prune_candidates(conn, today="2026-08-08")}
    assert ids == {"open_live", "pruner_void"}


def test_prune_candidates_date_hi_bounds_future():
    conn = _make_pred_db()
    _insert_pred(conn, "today", event_date="2026-08-08")
    _insert_pred(conn, "tomorrow", event_date="2026-08-09")
    _insert_pred(conn, "next_week", event_date="2026-08-15")
    conn.commit()

    ids = {r["market_id"] for r in prune_candidates(conn, today="2026-08-08", date_hi="2026-08-09")}
    assert ids == {"today", "tomorrow"}


# --------------------------------------------------------------------------- #
# decide_actions — the pure void/un-void decision                             #
# --------------------------------------------------------------------------- #

def _bet(**overrides) -> dict:
    """A reprice_rows-shaped bet dict with sensible upcoming/live defaults."""
    base = dict(
        market_id="kalshi:m", event_title="A vs B", sector="nba", yes_team="A",
        market_type="moneyline", line=None, venue="kalshi",
        scan_ask=0.50, live_ask=0.58, ev_pct=0.08,
        blended_true_prob=0.60, pin_is_live=True,
        price_is_live=True, live_ev_real=0.05, is_live=True,
        event_start=_SOON, voided=0, void_reason=None,
    )
    base.update(overrides)
    return base


def _act(bet):
    return decide_actions(
        [bet], min_ev=0.02, min_prob=0.15, hysteresis=0.01,
        now=_NOW, lookahead_hours=12,
    )[0]["action"]


def test_void_when_edge_reverted_at_live_price():
    # Open, upcoming, real live quote, edge below threshold (is_live False).
    assert _act(_bet(is_live=False, live_ev_real=0.005)) == "void"


def test_noop_when_still_live():
    assert _act(_bet(is_live=True, live_ev_real=0.05)) == "noop"


def test_failsafe_no_quote_never_voids():
    # Fetch failed → price not live → never void even if is_live is False.
    assert _act(_bet(is_live=False, price_is_live=False, live_ev_real=None)) == "skip_no_quote"


def test_failsafe_empty_book_never_voids():
    # ≥0.99 no-book price makes recompute return ev=None → live_ev_real is None.
    # This is the critical guard: a liquidity gap is NOT a reverted edge.
    assert _act(_bet(is_live=False, price_is_live=True, live_ev_real=None)) == "skip_no_quote"


def test_skip_inplay_game_already_started():
    assert _act(_bet(is_live=False, live_ev_real=0.0, event_start=_STARTED)) == "skip_inplay"


def test_skip_when_no_fresh_start_time():
    # Live ask but the Pinnacle line was pulled → can't confirm upcoming → skip.
    assert _act(_bet(is_live=False, live_ev_real=0.0, event_start=None)) == "skip_no_start"


def test_skip_future_beyond_lookahead():
    assert _act(_bet(is_live=False, live_ev_real=0.0, event_start=_FAR)) == "skip_future"


def test_unvoid_when_edge_recovers_past_hysteresis():
    threshold = tiered_min_ev(0.60, min_ev=0.02, min_prob=0.15)
    recovered = _bet(
        voided=1, void_reason=STALE_VOID_REASON,
        is_live=True, live_ev_real=threshold + 0.01 + 0.005,  # clears threshold+hysteresis
    )
    assert _act(recovered) == "unvoid"


def test_hysteresis_dead_band_holds_voided():
    # Edge back above the bare threshold (is_live True) but inside the dead-band
    # → stays voided (prevents flap across sweeps).
    threshold = tiered_min_ev(0.60, min_ev=0.02, min_prob=0.15)
    marginal = _bet(
        voided=1, void_reason=STALE_VOID_REASON,
        is_live=True, live_ev_real=threshold + 0.005,  # < threshold + hysteresis(0.01)
    )
    assert _act(marginal) == "noop"


def test_manual_void_is_never_unvoided():
    # A voided row the pruner did NOT create (void_reason None) is left alone
    # even with a fully recovered edge.
    assert _act(_bet(voided=1, void_reason=None, is_live=True, live_ev_real=0.20)) == "noop"


# --------------------------------------------------------------------------- #
# apply_actions — the DB write                                                 #
# --------------------------------------------------------------------------- #

def test_apply_actions_voids_and_unvoids():
    conn = _make_pred_db()
    _insert_pred(conn, "to_void", voided=0)
    _insert_pred(conn, "to_unvoid", voided=1, void_reason=STALE_VOID_REASON)
    actions = [
        {"action": "void", "market_id": "to_void"},
        {"action": "unvoid", "market_id": "to_unvoid"},
    ]
    counts = apply_actions(conn, actions, dry_run=False)
    assert counts == {"void": 1, "unvoid": 1}

    voided_row = conn.execute("SELECT voided, void_reason FROM ev_predictions WHERE market_id='to_void'").fetchone()
    assert (voided_row["voided"], voided_row["void_reason"]) == (1, STALE_VOID_REASON)
    unvoid_row = conn.execute("SELECT voided, void_reason FROM ev_predictions WHERE market_id='to_unvoid'").fetchone()
    assert (unvoid_row["voided"], unvoid_row["void_reason"]) == (0, None)


def test_apply_actions_dry_run_writes_nothing():
    conn = _make_pred_db()
    _insert_pred(conn, "to_void", voided=0)
    counts = apply_actions(conn, [{"action": "void", "market_id": "to_void"}], dry_run=True)
    assert counts == {"void": 1}
    row = conn.execute("SELECT voided FROM ev_predictions WHERE market_id='to_void'").fetchone()
    assert row["voided"] == 0  # unchanged


def test_apply_actions_unvoid_only_touches_stale_voids():
    # Defensive WHERE: an unvoid must not reopen a manual/cancel void even if one
    # is (wrongly) handed in.
    conn = _make_pred_db()
    _insert_pred(conn, "manual", voided=1, void_reason=None)
    apply_actions(conn, [{"action": "unvoid", "market_id": "manual"}], dry_run=False)
    row = conn.execute("SELECT voided FROM ev_predictions WHERE market_id='manual'").fetchone()
    assert row["voided"] == 1  # untouched — void_reason wasn't 'stale_reverted'


# --------------------------------------------------------------------------- #
# evaluate_candidates — end-to-end wiring with a stubbed reprice_rows          #
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_evaluate_candidates_wires_query_reprice_decision(monkeypatch):
    conn = _make_pred_db()
    _insert_pred(conn, "kalshi:reverted")
    _insert_pred(conn, "kalshi:still_good")

    async def _fake_reprice(rows, **kwargs):
        out = []
        for r in rows:
            reverted = r["market_id"].endswith("reverted")
            out.append({
                **r,
                "scan_ask": 0.50, "live_ask": 0.60 if reverted else 0.50,
                "price_is_live": True,
                "live_ev_real": -0.01 if reverted else 0.08,
                "is_live": not reverted,
                "event_start": _SOON, "pin_is_live": True,
            })
        return out

    monkeypatch.setattr("evmax.agents.cleanup.stale_positions.reprice_rows", _fake_reprice)

    _bets, actions = await evaluate_candidates(
        conn, min_ev=0.02, min_prob=0.15, hysteresis=0.01,
        kelly=0.5, bankroll=250.0, fees_in_pricing=False, max_kelly=0.05,
        lookahead_hours=12, now=_NOW, today="2026-08-08",
    )
    by_id = {a["market_id"]: a["action"] for a in actions}
    assert by_id == {"kalshi:reverted": "void", "kalshi:still_good": "noop"}


# --------------------------------------------------------------------------- #
# void_reason migration                                                        #
# --------------------------------------------------------------------------- #

def test_void_reason_migration_present_and_idempotent(monkeypatch):
    import evmax.agents.cleanup.db as db

    tmp = Path(tempfile.mkdtemp()) / "predictions.db"
    monkeypatch.setattr(db, "DB_PATH", tmp)

    conn = db.get_connection()
    cols = {r[1] for r in conn.execute("PRAGMA table_info(ev_predictions)").fetchall()}
    conn.close()
    assert "void_reason" in cols

    conn2 = db.get_connection()  # reopen — must not raise
    cols2 = {r[1] for r in conn2.execute("PRAGMA table_info(ev_predictions)").fetchall()}
    conn2.close()
    assert "void_reason" in cols2


# --------------------------------------------------------------------------- #
# reprice_rows — the shared engine's new keys + moneyline reblend guard        #
# --------------------------------------------------------------------------- #

def _pred_row(market_id, market_type, *, blended, sharp):
    return {
        "market_id": market_id, "event_id": "evt1", "sector": "nba",
        "yes_team": "Lakers", "market_type": market_type, "line": None,
        "venue": "kalshi", "kalshi_yes_price": 0.50, "sharp_true_prob": sharp,
        "blended_true_prob": blended, "ev_pct": 0.08, "sharp_weight_used": 1.0,
    }


@pytest.mark.asyncio
async def test_reprice_rows_offline_adds_new_keys():
    from evmax.agents.cleanup.reprice import reprice_rows

    rows = [_pred_row("kalshi:m1", "moneyline", blended=0.60, sharp=0.60)]
    bets = await reprice_rows(
        rows, live=False, kelly=0.5, bankroll=1000.0, min_ev=0.02,
        min_prob=0.15, fees_in_pricing=False, max_kelly=0.05,
    )
    b = bets[0]
    # No live fetch → gates at the scan ask, flags not-live-priced, no fresh start.
    assert b["live_effective"] is False
    assert b["price_is_live"] is False
    assert b["event_start"] is None
    assert b["live_ask"] == 0.50
    assert b["live_ev_real"] is not None  # 0.50 ask is a real (non-degenerate) price


@pytest.mark.asyncio
async def test_reprice_rows_reblend_is_moneyline_only(monkeypatch):
    """A fresh Pinnacle line re-blends the moneyline row but NOT the spread row,
    even though both share the event and get a fresh quote (event_start set)."""
    from evmax.agents.cleanup import reprice as reprice_mod

    class _FakeKalshi:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get_market_asks_batch(self, tickers):
            return {t: 0.50 for t in tickers}

    fresh = SimpleNamespace(
        event_id="evt1", event_date=_SOON,
        outcome_a_label="Lakers", outcome_b_label="Celtics",
        true_prob_a=0.70, true_prob_b=0.30,
    )

    class _FakePinnacle:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get_odds(self, sector):
            return [fresh]

    monkeypatch.setattr("evmax.clients.kalshi.KalshiClient", _FakeKalshi)
    monkeypatch.setattr("evmax.clients.esports_pinnacle.PinnacleGuestClient", _FakePinnacle)

    rows = [
        _pred_row("kalshi:ml", "moneyline", blended=0.60, sharp=0.60),
        _pred_row("kalshi:spread", "spread", blended=0.55, sharp=0.55),
    ]
    bets = await reprice_mod.reprice_rows(
        rows, live=True, kelly=0.5, bankroll=1000.0, min_ev=0.02,
        min_prob=0.15, fees_in_pricing=False, max_kelly=0.05,
    )
    ml = next(b for b in bets if b["market_id"] == "kalshi:ml")
    spread = next(b for b in bets if b["market_id"] == "kalshi:spread")

    # Moneyline row re-blends: 0.60 + 1.0*(0.70 - 0.60) = 0.70.
    assert ml["pin_is_live"] is True
    assert ml["blended_true_prob"] == pytest.approx(0.70)
    # Spread row must NOT re-blend (would mix market types), but still gets the
    # event start time so the pruner can gate it on kickoff.
    assert spread["pin_is_live"] is False
    assert spread["blended_true_prob"] == pytest.approx(0.55)
    assert spread["event_start"] == _SOON


# --------------------------------------------------------------------------- #
# command registration                                                         #
# --------------------------------------------------------------------------- #

def test_prune_stale_command_registered():
    import re

    from typer.testing import CliRunner

    from evmax.cli.commands.cleanup import app

    result = CliRunner().invoke(app, ["prune-stale", "--help"])
    assert result.exit_code == 0
    output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    for flag in ("--dry-run", "--interval", "--once", "--lookahead", "--hysteresis", "--min-ev"):
        assert flag in output


# ---------------------------------------------------------------------------
# apply_cash_cap_to_bets — per-venue cash cap at placement (GAP 2)
# ---------------------------------------------------------------------------

from evmax.agents.cleanup.reprice import apply_cash_cap_to_bets  # noqa: E402


def _cbet(market_id, venue, kf, stake, is_live=True):
    return {
        "market_id": market_id, "venue": venue,
        "kelly_fraction_used": kf, "stake": stake, "is_live": is_live,
    }


def test_cash_cap_scales_venue_over_cash():
    bets = [_cbet("k:a", "kalshi", 0.05, 50.0), _cbet("k:b", "kalshi", 0.03, 30.0)]
    apply_cash_cap_to_bets(bets, {"kalshi": 40.0})  # $80 requested, $40 cash → ×0.5
    assert bets[0]["stake"] == pytest.approx(25.0)
    assert bets[1]["stake"] == pytest.approx(15.0)
    assert bets[0]["kelly_fraction_used"] == pytest.approx(0.025)
    assert bets[1]["kelly_fraction_used"] == pytest.approx(0.015)


def test_cash_cap_under_cash_unchanged():
    bets = [_cbet("k:a", "kalshi", 0.03, 30.0)]
    apply_cash_cap_to_bets(bets, {"kalshi": 100.0})
    assert bets[0]["stake"] == pytest.approx(30.0)
    assert bets[0]["kelly_fraction_used"] == pytest.approx(0.03)


def test_cash_cap_missing_venue_uncapped():
    bets = [_cbet("k:a", "kalshi", 0.05, 50.0)]
    apply_cash_cap_to_bets(bets, {"polymarket_us": 5.0})  # no kalshi entry
    assert bets[0]["stake"] == pytest.approx(50.0)


def test_cash_cap_ignores_non_live_bets():
    # a non-live bet's stake must not count toward the venue's requested sum
    bets = [
        _cbet("k:a", "kalshi", 0.05, 50.0),
        _cbet("k:dead", "kalshi", 0.0, 0.0, is_live=False),
    ]
    apply_cash_cap_to_bets(bets, {"kalshi": 50.0})  # $50 == $50 → no scale
    assert bets[0]["stake"] == pytest.approx(50.0)


def test_cash_cap_per_venue_independent():
    bets = [
        _cbet("k:a", "kalshi", 0.05, 50.0),          # $50 ≤ $50 → ok
        _cbet("p:a", "polymarket_us", 0.05, 50.0),   # $50 > $25 → half
    ]
    apply_cash_cap_to_bets(bets, {"kalshi": 50.0, "polymarket_us": 25.0})
    assert bets[0]["stake"] == pytest.approx(50.0)
    assert bets[1]["stake"] == pytest.approx(25.0)


def test_cash_cap_empty_map_is_noop():
    bets = [_cbet("k:a", "kalshi", 0.05, 50.0)]
    apply_cash_cap_to_bets(bets, None)
    assert bets[0]["stake"] == pytest.approx(50.0)
    apply_cash_cap_to_bets(bets, {})
    assert bets[0]["stake"] == pytest.approx(50.0)
