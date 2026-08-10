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
            venue TEXT NOT NULL DEFAULT 'kalshi',
            minutes_to_tipoff INTEGER,
            model_diagnostics TEXT,
            maker_ev_pct REAL,
            maker_fill INTEGER NOT NULL DEFAULT 0,
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
            venue TEXT NOT NULL DEFAULT 'kalshi',
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
# model_version — git code provenance default
# -------------------------------------------------------------------------


def test_log_gaps_defaults_model_version_to_code_provenance(patched_db, monkeypatch):
    """When the caller omits model_version, every row is stamped with the
    git code provenance of the running checkout — so cron scans from a
    feature branch or stale main are identifiable after the fact."""
    monkeypatch.setattr(
        logger_module, "code_version", lambda: "abc1234-dirty@fix/branch"
    )
    inserted = log_gaps([_gap("g1"), _gap("g2")], mode_resolver=lambda c: "live")
    assert inserted == 2
    rows = patched_db.execute("SELECT model_version FROM ev_predictions").fetchall()
    assert all(r["model_version"] == "abc1234-dirty@fix/branch" for r in rows)


def test_log_gaps_explicit_model_version_overrides_provenance(patched_db, monkeypatch):
    monkeypatch.setattr(logger_module, "code_version", lambda: "abc1234")
    log_gaps([_gap("g1")], mode_resolver=lambda c: "live", model_version="custom_v2")
    row = patched_db.execute("SELECT model_version FROM ev_predictions").fetchone()
    assert row["model_version"] == "custom_v2"


def test_log_gaps_model_version_null_when_git_unavailable(patched_db, monkeypatch):
    """Provenance must never break a scan — no git means a NULL column."""
    monkeypatch.setattr(logger_module, "code_version", lambda: None)
    inserted = log_gaps([_gap("g1")], mode_resolver=lambda c: "live")
    assert inserted == 1
    row = patched_db.execute("SELECT model_version FROM ev_predictions").fetchone()
    assert row["model_version"] is None


def test_log_prop_observations_defaults_model_version_to_code_provenance(
    patched_db, monkeypatch
):
    monkeypatch.setattr(logger_module, "code_version", lambda: "abc1234")
    inserted = log_prop_observations(
        [_prop_gap("LeBron James")], mode_resolver=lambda c: "live"
    )
    assert inserted == 1
    row = patched_db.execute("SELECT model_version FROM prop_observations").fetchone()
    assert row["model_version"] == "abc1234"


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


# -------------------------------------------------------------------------
# Partial-blend demotion (REQUIRED_BLEND_MODELS gate) — full_blend flag
# -------------------------------------------------------------------------

from dataclasses import replace

from evmax.agents.odds.ev_gap_agent import has_full_blend

_FULL_TENNIS_SOURCES = (
    "tennis_advanced+tennis_form+tennis_serve_return+tennis_surface+sharp"
)


class TestHasFullBlend:
    def test_tennis_full_coverage_passes(self):
        assert has_full_blend("tennis", _FULL_TENNIS_SOURCES) is True

    def test_tennis_extra_models_still_pass(self):
        assert has_full_blend(
            "tennis", _FULL_TENNIS_SOURCES + "+tennis_h2h+tennis_ranking_trend"
        ) is True

    def test_tennis_missing_core_model_fails(self):
        # No tennis_surface
        assert has_full_blend(
            "tennis", "tennis_advanced+tennis_form+tennis_serve_return+sharp"
        ) is False

    def test_tennis_sharp_only_fails(self):
        assert has_full_blend("tennis", "sharp") is False

    def test_unlisted_sector_always_passes(self):
        assert has_full_blend("nba", "sharp") is True
        assert has_full_blend(None, None) is True


class TestMinNonsharpFloor:
    """MIN_NONSHARP_MODELS — the any-N-of floor (soccer/worldcup)."""

    def test_soccer_sharp_only_ml_fails(self):
        assert has_full_blend("soccer", "sharp", "moneyline") is False

    def test_soccer_sharp_capped_ml_fails(self):
        # A capped row's final price IS the sharp price — passthrough.
        assert has_full_blend("soccer", "sharp(capped)", "moneyline") is False

    def test_soccer_adjustment_layers_dont_count(self):
        # injury/late_news ride on top of the blend; they aren't model signal.
        assert has_full_blend("soccer", "sharp+injury+late_news", "moneyline") is False

    def test_soccer_any_one_model_passes(self):
        assert has_full_blend("soccer", "sharp+poisson", "moneyline") is True
        assert has_full_blend("soccer", "elo+sharp", "moneyline") is True
        assert has_full_blend("soccer", "xg+sharp+injury", "moneyline") is True

    def test_soccer_total_out_of_scope(self):
        # No model prices soccer totals — the floor deliberately excludes them.
        assert has_full_blend("soccer", "sharp+total_dist", "total") is True

    def test_soccer_no_market_type_fails_closed(self):
        # market_type=None applies the floor unscoped (fail-closed).
        assert has_full_blend("soccer", "sharp") is False

    def test_worldcup_advance_in_scope(self):
        assert has_full_blend("worldcup", "sharp+advance_derived", "advance") is False
        assert has_full_blend("worldcup", "elo+sharp+advance_derived", "advance") is True

    def test_ufc_unaffected(self):
        # Sharp-dominance is by design for sectors without an entry.
        assert has_full_blend("ufc", "sharp", "moneyline") is True
        assert has_full_blend("ufc", "sharp+ufc_rating", "moneyline") is True

    def test_tennis_all_of_check_unchanged(self):
        # Tennis keeps the all-of gate; the floor doesn't apply to it.
        assert has_full_blend("tennis", _FULL_TENNIS_SOURCES, "moneyline") is True
        assert has_full_blend("tennis", "sharp", "moneyline") is False


class TestModelDiagnosticsPersistence:
    """model_diagnostics JSON rides EVGap → log_gaps → ev_predictions."""

    def test_diagnostics_persisted(self, patched_db):
        import dataclasses
        import json as _json

        diag = {"fired": {"elo": {"conf": 0.6, "n": 20}},
                "gated": {"form": {"conf": 0.3, "note": "thin"}},
                "missing": ["poisson"]}
        g = dataclasses.replace(
            _gap("kalshi:DIAG-1"), model_diagnostics=_json.dumps(diag)
        )
        log_gaps([g], scan_date=date.today())
        row = patched_db.execute(
            "SELECT model_diagnostics FROM ev_predictions WHERE market_id='kalshi:DIAG-1'"
        ).fetchone()
        assert row is not None
        assert _json.loads(row["model_diagnostics"]) == diag

    def test_legacy_gap_without_diagnostics_is_null_safe(self, patched_db):
        g = _gap("kalshi:DIAG-2")
        log_gaps([g], scan_date=date.today())
        row = patched_db.execute(
            "SELECT model_diagnostics FROM ev_predictions WHERE market_id='kalshi:DIAG-2'"
        ).fetchone()
        assert row is not None
        assert row["model_diagnostics"] is None


class TestPartialBlendDemotion:
    def test_partial_blend_live_gap_demoted_to_shadow(self, patched_db):
        gap = replace(
            _gap("t-partial", sector="tennis"),
            model_sources="tennis_ranking_trend+sharp",
            full_blend=False,
        )
        inserted = log_gaps([gap], mode_resolver=lambda c: "live")
        assert inserted == 1
        row = patched_db.execute(
            "SELECT mode FROM ev_predictions WHERE market_id = 't-partial'"
        ).fetchone()
        assert row["mode"] == "shadow"

    def test_full_blend_live_gap_stays_live(self, patched_db):
        gap = replace(
            _gap("t-full", sector="tennis"),
            model_sources=_FULL_TENNIS_SOURCES,
            full_blend=True,
        )
        log_gaps([gap], mode_resolver=lambda c: "live")
        row = patched_db.execute(
            "SELECT mode FROM ev_predictions WHERE market_id = 't-full'"
        ).fetchone()
        assert row["mode"] == "live"

    def test_demotion_does_not_promote_shadow_categories(self, patched_db):
        # A shadow-mode category stays shadow regardless of the flag.
        gap = replace(_gap("t-shadowcat", sector="tennis"), full_blend=True)
        log_gaps([gap], mode_resolver=lambda c: "shadow")
        row = patched_db.execute(
            "SELECT mode FROM ev_predictions WHERE market_id = 't-shadowcat'"
        ).fetchone()
        assert row["mode"] == "shadow"

    def test_default_full_blend_true_keeps_other_sectors_live(self, patched_db):
        gap = _gap("nba-default", sector="nba")  # full_blend defaults True
        log_gaps([gap], mode_resolver=lambda c: "live")
        row = patched_db.execute(
            "SELECT mode FROM ev_predictions WHERE market_id = 'nba-default'"
        ).fetchone()
        assert row["mode"] == "live"


class TestMakerOnlyDemotion:
    """maker_only gaps clear the floor only as a resting limit order — they are
    fill-contingent, so log_gaps demotes a live one to shadow (never bankroll)
    and persists maker_ev_pct. Promote to a real position with `agents fill`."""

    def test_maker_only_live_gap_demoted_to_shadow(self, patched_db):
        gap = replace(
            _gap("m-only", sector="nba"),
            ev_pct=0.014,          # taker below the floor
            maker_ev_pct=0.041,    # maker clears it
            maker_only=True,
            kelly_fraction=0.0,    # no live taker stake
        )
        inserted = log_gaps([gap], mode_resolver=lambda c: "live")
        assert inserted == 1
        row = patched_db.execute(
            "SELECT mode, maker_ev_pct FROM ev_predictions WHERE market_id = 'm-only'"
        ).fetchone()
        assert row["mode"] == "shadow"
        assert row["maker_ev_pct"] == pytest.approx(0.041)

    def test_taker_clearing_gap_stays_live_with_maker_ev(self, patched_db):
        gap = replace(
            _gap("m-taker", sector="nba"),
            ev_pct=0.05,
            maker_ev_pct=0.08,
            maker_only=False,
        )
        log_gaps([gap], mode_resolver=lambda c: "live")
        row = patched_db.execute(
            "SELECT mode, maker_ev_pct FROM ev_predictions WHERE market_id = 'm-taker'"
        ).fetchone()
        assert row["mode"] == "live"
        assert row["maker_ev_pct"] == pytest.approx(0.08)

    def test_default_gap_has_null_maker_ev(self, patched_db):
        # A gap constructed without maker fields (legacy path) is null-safe.
        log_gaps([_gap("m-legacy", sector="nba")], mode_resolver=lambda c: "live")
        row = patched_db.execute(
            "SELECT mode, maker_ev_pct FROM ev_predictions WHERE market_id = 'm-legacy'"
        ).fetchone()
        assert row["mode"] == "live"
        assert row["maker_ev_pct"] is None


class TestShadowVenueDemotion:
    """Venue firewall: polymarket_us gaps stay shadow until the venue is
    promoted via settings.polymarket_us_live (MODEL-9 pattern)."""

    def test_polymarket_us_live_gap_demoted_to_shadow(self, patched_db, monkeypatch):
        from evmax.settings import get_settings
        monkeypatch.setattr(get_settings(), "polymarket_us_live", False)
        gap = replace(_gap("pm-demote", sector="nba"), venue="polymarket_us")
        inserted = log_gaps([gap], mode_resolver=lambda c: "live")
        assert inserted == 1
        row = patched_db.execute(
            "SELECT mode, venue FROM ev_predictions WHERE market_id = 'pm-demote'"
        ).fetchone()
        assert row["mode"] == "shadow"
        assert row["venue"] == "polymarket_us"

    def test_polymarket_us_stays_live_after_promotion(self, patched_db, monkeypatch):
        from evmax.settings import get_settings
        monkeypatch.setattr(get_settings(), "polymarket_us_live", True)
        gap = replace(_gap("pm-promoted", sector="nba"), venue="polymarket_us")
        log_gaps([gap], mode_resolver=lambda c: "live")
        row = patched_db.execute(
            "SELECT mode FROM ev_predictions WHERE market_id = 'pm-promoted'"
        ).fetchone()
        assert row["mode"] == "live"

    def test_kalshi_gap_unaffected_by_venue_firewall(self, patched_db):
        gap = _gap("k-live", sector="nba")  # venue defaults to kalshi
        log_gaps([gap], mode_resolver=lambda c: "live")
        row = patched_db.execute(
            "SELECT mode FROM ev_predictions WHERE market_id = 'k-live'"
        ).fetchone()
        assert row["mode"] == "live"


class TestShadowToLiveUpgrade:
    """A market frozen as shadow (e.g. a tennis partial-blend demotion before
    form/advanced had data) must upgrade to live once a later scan produces a
    full-blend live gap. Regression for: a stale shadow row permanently blocks
    /api/pick ('mode=shadow — only live is pickable') even though the model now
    qualifies the bet as live."""

    def test_partial_then_full_blend_upgrades_to_live(self, patched_db):
        partial = replace(
            _gap("t-up", sector="tennis"),
            model_sources="tennis_ranking_trend+tennis_serve_return+tennis_surface+sharp",
            full_blend=False,
            ev_pct=0.03,
            kalshi_yes_price=0.43,
        )
        assert log_gaps([partial], mode_resolver=lambda c: "live") == 1
        row = patched_db.execute(
            "SELECT mode FROM ev_predictions WHERE market_id = 't-up'"
        ).fetchone()
        assert row["mode"] == "shadow"  # demoted on first insert

        # Later scan: full blend fires, gap resolves to live. INSERT OR IGNORE
        # is a no-op (UNIQUE conflict) but the row must upgrade + refresh.
        full = replace(
            _gap("t-up", sector="tennis"),
            model_sources=_FULL_TENNIS_SOURCES,
            full_blend=True,
            ev_pct=0.05,
            kalshi_yes_price=0.44,
        )
        inserted = log_gaps([full], mode_resolver=lambda c: "live")
        assert inserted == 0  # not a new insert — an in-place upgrade
        row = patched_db.execute(
            "SELECT mode, ev_pct, kalshi_yes_price, captured_yes_price, "
            "model_sources, voided FROM ev_predictions WHERE market_id = 't-up'"
        ).fetchone()
        assert row["mode"] == "live"
        # Snapshot refreshed to the first *bettable* (live) call.
        assert row["ev_pct"] == pytest.approx(0.05)
        assert row["kalshi_yes_price"] == pytest.approx(0.44)
        assert row["captured_yes_price"] == pytest.approx(0.44)
        assert row["model_sources"] == _FULL_TENNIS_SOURCES
        assert row["voided"] == 0

    def test_upgrade_never_touches_placed_shadow_row(self, patched_db):
        """A placed shadow row is a deliberate user action — never auto-upgrade
        it to live, even if a later full-blend scan would qualify it."""
        partial = replace(
            _gap("t-placed", sector="tennis"),
            model_sources="tennis_serve_return+tennis_surface+sharp",
            full_blend=False,
        )
        log_gaps([partial], mode_resolver=lambda c: "live")
        patched_db.execute(
            "UPDATE ev_predictions SET placed = 1 WHERE market_id = 't-placed'"
        )
        patched_db.commit()

        full = replace(
            _gap("t-placed", sector="tennis"),
            model_sources=_FULL_TENNIS_SOURCES,
            full_blend=True,
        )
        log_gaps([full], mode_resolver=lambda c: "live")
        row = patched_db.execute(
            "SELECT mode, placed FROM ev_predictions WHERE market_id = 't-placed'"
        ).fetchone()
        assert row["mode"] == "shadow"  # untouched
        assert row["placed"] == 1

    def test_shadow_category_not_upgraded_when_rescan_still_shadow(self, patched_db):
        """If the new scan also resolves to shadow (sector still in shadow mode),
        there is no live gap to upgrade from — the row stays shadow."""
        gap = replace(_gap("t-stayshadow", sector="tennis"), full_blend=True)
        log_gaps([gap], mode_resolver=lambda c: "shadow")
        log_gaps([gap], mode_resolver=lambda c: "shadow")
        row = patched_db.execute(
            "SELECT mode FROM ev_predictions WHERE market_id = 't-stayshadow'"
        ).fetchone()
        assert row["mode"] == "shadow"


# -------------------------------------------------------------------------
# WNBA anchored-entry ownership (2026-07-19): the scanner's wnba spread/total
# gaps route through the default resolver and get DROPPED (disabled_market_types
# in categories.yaml), while the watch-listings anchored-entry trigger bypasses
# via an explicit shadow resolver — so the trigger owns those market_ids under
# freeze-on-first-insert.
# -------------------------------------------------------------------------


def _wnba_spread_gap(market_id: str) -> EVGap:
    g = _gap(market_id, sector="wnba")
    return EVGap(**{**g.__dict__, "market_type": "spread", "line": -6.5,
                    "model_sources": "sharp+spread_dist"})


def test_scanner_wnba_spread_gap_dropped_as_disabled(patched_db):
    inserted = log_gaps([_wnba_spread_gap("kalshi:KXWNBASPREAD-X-Y7")])
    assert inserted == 0
    rows = patched_db.execute("SELECT * FROM ev_predictions").fetchall()
    assert rows == []


def test_anchored_trigger_wnba_spread_gap_lands_shadow(patched_db):
    inserted = log_gaps(
        [_wnba_spread_gap("kalshi:KXWNBASPREAD-X-Y7")],
        mode_resolver=lambda c: "shadow",
    )
    assert inserted == 1
    row = patched_db.execute(
        "SELECT mode, market_type, captured_yes_price FROM ev_predictions"
    ).fetchone()
    assert row["mode"] == "shadow"
    assert row["market_type"] == "spread"
    assert row["captured_yes_price"] == pytest.approx(0.45)


def test_scanner_wnba_moneyline_still_live(patched_db):
    g = _gap("kalshi:KXWNBAGAME-X-Y", sector="wnba")
    inserted = log_gaps([g])
    assert inserted == 1
    row = patched_db.execute("SELECT mode FROM ev_predictions").fetchone()
    assert row["mode"] == "live"
