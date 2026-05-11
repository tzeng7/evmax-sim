"""Tests for ARCH-11 mode-aware logging in evmax.agents.cleanup.logger.

Covers:
  1. log_gaps honors a custom mode_resolver (live / shadow / disabled)
  2. Disabled categories are dropped — row count decreases, no DB row
  3. The `mode`, `captured_yes_price`, and `model_version` columns are
     populated correctly per row
  4. Broken resolver falls back to 'live' and logs the drift
  5. _gap_category_key maps game vs prop gaps to the right registry key
  6. log_prop_observations honors the same resolver contract
"""

from __future__ import annotations

import sqlite3
from datetime import date
from unittest.mock import patch

import pytest

from evmax.agents.cleanup import logger as logger_module
from evmax.agents.cleanup.logger import (
    _gap_category_key,
    log_gaps,
    log_prop_observations,
)
from evmax.agents.odds.ev_gap_agent import EVGap


# -------------------------------------------------------------------------
# In-memory DB fixture — mirrors production schema, ARCH-11 columns present
# -------------------------------------------------------------------------


def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE ev_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            logged_at TEXT DEFAULT (datetime('now')),
            scan_date TEXT NOT NULL,
            market_id TEXT NOT NULL,
            event_id TEXT,
            sector TEXT,
            yes_team TEXT,
            market_type TEXT,
            event_title TEXT,
            event_date TEXT,
            kalshi_yes_price REAL,
            sharp_true_prob REAL,
            blended_true_prob REAL,
            ev_pct REAL,
            kelly_fraction REAL,
            volume_usd REAL,
            model_sources TEXT,
            sharp_weight_used REAL,
            bankroll_used REAL,
            line REAL,
            voided INTEGER NOT NULL DEFAULT 0,
            placed INTEGER NOT NULL DEFAULT 0,
            mode TEXT NOT NULL DEFAULT 'live',
            captured_yes_price REAL,
            model_version TEXT,
            minutes_to_tipoff INTEGER,
            UNIQUE(market_id, scan_date)
        );
        CREATE TABLE prop_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_date TEXT NOT NULL,
            event_date TEXT,
            sector TEXT NOT NULL,
            player_name TEXT NOT NULL,
            stat_type TEXT NOT NULL,
            line REAL NOT NULL,
            kalshi_price REAL,
            sharp_prob REAL,
            ev_pct REAL,
            l15_games INTEGER,
            market_id TEXT,
            event_id TEXT,
            event_title TEXT,
            actual_value REAL,
            outcome INTEGER,
            resolved_at TEXT,
            mode TEXT NOT NULL DEFAULT 'live',
            captured_yes_price REAL,
            model_version TEXT,
            UNIQUE(market_id, scan_date)
        );
        """
    )
    return conn


@pytest.fixture
def patched_db(monkeypatch):
    """Patch get_connection so both calls return the same in-memory conn."""
    conn = _make_db()

    class _CtxConn:
        def __init__(self, inner):
            self._inner = inner

        def __enter__(self):
            return self._inner

        def __exit__(self, *a):
            return False

        def __getattr__(self, name):
            return getattr(self._inner, name)

    ctx = _CtxConn(conn)
    monkeypatch.setattr(logger_module, "get_connection", lambda: ctx)
    yield conn
    conn.close()


def _gap(market_id: str, sector: str = "nba", event_id: str | None = None) -> EVGap:
    from datetime import datetime

    return EVGap(
        market_id=market_id,
        event_id=event_id or f"evt:{market_id}",
        sector=sector,
        yes_team="LAL",
        market_type="moneyline",
        kalshi_yes_price=0.45,
        sharp_true_prob=0.55,
        blended_true_prob=0.55,
        ev_pct=0.07,
        kelly_full=0.04,
        kelly_fraction=0.02,
        match_confidence=0.95,
        volume_usd=5_000.0,
        spread_pct=0.02,
        event_date=datetime.combine(date.today(), datetime.min.time()),
        model_sources="elo+sharp",
        line=None,
        event_title="Lakers vs Warriors",
    )


def _prop_gap(player: str = "LeBron James", sector: str = "nba") -> EVGap:
    from datetime import datetime

    return EVGap(
        market_id=f"kalshi:{player}-points",
        event_id=f"{sector}::2026-04-15::prop::{player}::points::24.5",
        sector=sector,
        yes_team=player,
        market_type="player_prop",
        kalshi_yes_price=0.48,
        sharp_true_prob=0.52,
        blended_true_prob=0.52,
        ev_pct=0.04,
        kelly_full=0.02,
        kelly_fraction=0.01,
        match_confidence=0.92,
        volume_usd=1_500.0,
        spread_pct=0.015,
        event_date=datetime.combine(date.today(), datetime.min.time()),
        model_sources="nba_props_cache",
        line=24.5,
        event_title=f"{player} — LAL vs GSW",
        prop_player_name=player,
        prop_stat_type="points",
        prop_threshold=24.5,
        prop_l15_games=15,
    )


# -------------------------------------------------------------------------
# _gap_category_key
# -------------------------------------------------------------------------


def test_gap_category_key_game():
    assert _gap_category_key(_gap("m1", sector="nba")) == "nba"
    assert _gap_category_key(_gap("m2", sector="nfl")) == "nfl"


def test_gap_category_key_prop():
    assert _gap_category_key(_prop_gap(sector="nba")) == "nba_props"
    assert _gap_category_key(_prop_gap(sector="nfl")) == "nfl_props"


# -------------------------------------------------------------------------
# log_gaps — mode routing
# -------------------------------------------------------------------------


def test_log_gaps_default_resolver_uses_live(patched_db):
    """With the resolver not overridden, every gap should route to the
    shipped YAML default. nba is live in data/categories.yaml so we
    expect both rows to land with mode='live'."""
    gaps = [_gap("g1"), _gap("g2")]
    inserted = log_gaps(gaps, sharp_weight_used=0.85, bankroll_used=500.0)
    assert inserted == 2
    rows = patched_db.execute(
        "SELECT market_id, mode, captured_yes_price FROM ev_predictions"
    ).fetchall()
    assert {r["market_id"] for r in rows} == {"g1", "g2"}
    assert all(r["mode"] == "live" for r in rows)
    assert all(r["captured_yes_price"] == pytest.approx(0.45) for r in rows)


def test_log_gaps_custom_resolver_routes_to_shadow(patched_db):
    gaps = [_gap("g1"), _gap("g2")]
    inserted = log_gaps(
        gaps,
        mode_resolver=lambda c: "shadow",
        model_version="test_v1",
    )
    assert inserted == 2
    rows = patched_db.execute(
        "SELECT market_id, mode, model_version FROM ev_predictions"
    ).fetchall()
    assert all(r["mode"] == "shadow" for r in rows)
    assert all(r["model_version"] == "test_v1" for r in rows)


def test_log_gaps_disabled_category_dropped(patched_db):
    """A disabled category must not produce any DB row."""
    gaps = [_gap("g1"), _gap("g2")]
    inserted = log_gaps(gaps, mode_resolver=lambda c: "disabled")
    assert inserted == 0
    rows = patched_db.execute("SELECT COUNT(*) AS n FROM ev_predictions").fetchone()
    assert rows["n"] == 0


def test_log_gaps_mixed_modes_partition_correctly(patched_db):
    """Three gaps, three modes — exactly one row lands per mode, disabled dropped."""
    gaps = [_gap("live1", sector="nba"), _gap("shadow1", sector="nfl_props"), _gap("drop1", sector="nhl")]

    def resolver(category: str) -> str:
        return {
            "nba": "live",
            "nfl_props": "shadow",
            "nhl": "disabled",
        }[category]

    inserted = log_gaps(gaps, mode_resolver=resolver)
    assert inserted == 2  # disabled dropped

    live_rows = patched_db.execute(
        "SELECT market_id FROM ev_predictions WHERE mode = 'live'"
    ).fetchall()
    shadow_rows = patched_db.execute(
        "SELECT market_id FROM ev_predictions WHERE mode = 'shadow'"
    ).fetchall()
    assert {r["market_id"] for r in live_rows} == {"live1"}
    assert {r["market_id"] for r in shadow_rows} == {"shadow1"}


def test_log_gaps_broken_resolver_defaults_to_live(patched_db, caplog):
    """If the resolver raises, we fall back to 'live' and log the
    failure — preserving pre-ARCH-11 behavior rather than silently
    dropping bets."""
    def broken(category: str) -> str:
        raise KeyError(f"mystery category: {category}")

    gaps = [_gap("g1")]
    inserted = log_gaps(gaps, mode_resolver=broken)
    assert inserted == 1
    row = patched_db.execute("SELECT mode FROM ev_predictions").fetchone()
    assert row["mode"] == "live"


def test_log_gaps_unsticks_voided_on_rescan(patched_db):
    """If a market was marked voided by cleanup but Kalshi is quoting it
    again, the next log_gaps call must reset voided=0 so /api/pick can
    find the row. Snapshot columns stay frozen (freeze-on-first-insert).
    Regression for: silent 'Placed 0 bet(s)' toast when a stale voided
    row blocks a live scan result."""
    gap = _gap("stuck-mid", sector="nba")
    assert log_gaps([gap], sharp_weight_used=0.85, bankroll_used=500.0) == 1

    # Simulate cleanup voiding the row + the user not yet picking it.
    patched_db.execute(
        "UPDATE ev_predictions SET voided = 1 WHERE market_id = ?",
        ("stuck-mid",),
    )
    patched_db.commit()

    # Second scan: INSERT OR IGNORE is a no-op (UNIQUE conflict), but
    # the new branch should flip voided back to 0.
    inserted = log_gaps([gap], sharp_weight_used=0.85, bankroll_used=500.0)
    assert inserted == 0  # snapshot stayed frozen
    row = patched_db.execute(
        "SELECT voided, ev_pct FROM ev_predictions WHERE market_id = ?",
        ("stuck-mid",),
    ).fetchone()
    assert row["voided"] == 0
    # Snapshot preserved — original ev_pct still there.
    assert row["ev_pct"] == pytest.approx(0.07)


def test_log_gaps_does_not_unvoid_placed_rows(patched_db):
    """A row with placed=1 is a deliberate user action — never reset its
    voided flag, even on rescan. Protects against accidental re-activation
    of a manually-voided placed bet."""
    gap = _gap("placed-mid", sector="nba")
    log_gaps([gap])
    patched_db.execute(
        "UPDATE ev_predictions SET voided = 1, placed = 1 WHERE market_id = ?",
        ("placed-mid",),
    )
    patched_db.commit()

    log_gaps([gap])
    row = patched_db.execute(
        "SELECT voided, placed FROM ev_predictions WHERE market_id = ?",
        ("placed-mid",),
    ).fetchone()
    assert row["voided"] == 1  # stays voided
    assert row["placed"] == 1


def test_log_gaps_empty_list_is_noop(patched_db):
    assert log_gaps([]) == 0
    rows = patched_db.execute("SELECT COUNT(*) AS n FROM ev_predictions").fetchone()
    assert rows["n"] == 0


# -------------------------------------------------------------------------
# log_prop_observations — mode routing + prop filter
# -------------------------------------------------------------------------


def test_log_prop_observations_filters_non_props(patched_db):
    """Game gaps mixed in with prop gaps should be ignored by the
    prop-observations writer."""
    gaps = [_gap("game1"), _prop_gap("LeBron James")]
    inserted = log_prop_observations(gaps, mode_resolver=lambda c: "live")
    assert inserted == 1
    rows = patched_db.execute(
        "SELECT player_name FROM prop_observations"
    ).fetchall()
    assert {r["player_name"] for r in rows} == {"LeBron James"}


def test_log_prop_observations_shadow_mode(patched_db):
    gaps = [_prop_gap("Player A"), _prop_gap("Player B")]
    inserted = log_prop_observations(
        gaps,
        mode_resolver=lambda c: "shadow",
        model_version="nfl_qb_v1",
    )
    assert inserted == 2
    rows = patched_db.execute(
        "SELECT player_name, mode, model_version FROM prop_observations"
    ).fetchall()
    assert all(r["mode"] == "shadow" for r in rows)
    assert all(r["model_version"] == "nfl_qb_v1" for r in rows)


def test_log_prop_observations_disabled_category_dropped(patched_db):
    gaps = [_prop_gap("Player A")]
    inserted = log_prop_observations(gaps, mode_resolver=lambda c: "disabled")
    assert inserted == 0
    rows = patched_db.execute(
        "SELECT COUNT(*) AS n FROM prop_observations"
    ).fetchone()
    assert rows["n"] == 0
