"""Tests for the shadow-validation contamination filter.

Covers:
  - is_contaminated() pure predicate (the single source of truth)
  - `evmax cleanup shadow metrics` excludes contaminated rows by default
    and includes them under --include-contaminated
  - `evmax cleanup shadow promote` refuses on a thin clean sample and
    allows it through with --force
"""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from evmax.agents.cleanup.contamination import is_contaminated
from evmax.cli.commands.shadow import app

runner = CliRunner()


# -------------------------------------------------------------------------
# is_contaminated — pure predicate
# -------------------------------------------------------------------------


def test_baseball_poisson_is_contaminated():
    # Poisson was removed from baseball 2026-06-01 — its presence dates the row.
    assert is_contaminated("baseball", "moneyline", "elo+pitcher+poisson+sharp")
    assert is_contaminated("baseball", "total", "elo+poisson+sharp")


def test_baseball_ml_without_pitcher_is_contaminated():
    # Pitcher-required guard shipped 2026-06-06; ML without pitcher is pre-guard.
    assert is_contaminated("baseball", "moneyline", "elo+sharp")
    assert is_contaminated("baseball", "moneyline", None)
    assert is_contaminated("baseball", "moneyline", "")


def test_baseball_clean_ml_row_is_not_contaminated():
    # Current code: pitcher present, no poisson.
    assert not is_contaminated("baseball", "moneyline", "elo+pitcher+sharp")


def test_baseball_non_ml_without_pitcher_is_clean():
    # The pitcher rule is scoped to moneyline — spread/total don't use the
    # pitcher agent, so a pitcher-less spread/total row is not contaminated
    # (as long as it has no poisson).
    assert not is_contaminated("baseball", "spread", "elo+sharp")
    assert not is_contaminated("baseball", "total", "sharp")


def test_baseball_alt_runline_spread_is_contaminated():
    # Current code rejects baseball spreads beyond ±1.5 (alt-run-line gate),
    # so any resolved −2.5/−4.5 row is pre-gate.
    assert is_contaminated("baseball", "spread", "sharp", -4.5)
    assert is_contaminated("baseball", "spread", "sharp", 2.5)
    assert is_contaminated("baseball", "spread", "sharp", -2.5)


def test_baseball_standard_runline_spread_is_clean():
    # Standard ±1.5 run line is exactly what current code prices.
    assert not is_contaminated("baseball", "spread", "sharp", -1.5)
    assert not is_contaminated("baseball", "spread", "sharp", 1.5)
    # A missing line can't be judged by the line rule → not flagged by it.
    assert not is_contaminated("baseball", "spread", "sharp", None)


def test_ruleless_sectors_are_never_flagged():
    # No rules for these sectors → always clean, even with "poisson" present.
    assert not is_contaminated("nba", "moneyline", "elo+poisson+sharp")
    assert not is_contaminated("nfl", "moneyline", "elo+sharp")
    assert not is_contaminated(None, "moneyline", None)


def test_sector_case_insensitive():
    assert is_contaminated("BASEBALL", "moneyline", "elo+poisson+sharp")


# -------------------------------------------------------------------------
# WNBA spread — possession_sim in the blend dates the row to the retired
# sim_weight=0.35 pricing (SPREAD_SIM_WEIGHT_OVERRIDES, 2026-06-19).
# -------------------------------------------------------------------------


def test_wnba_spread_with_possession_sim_is_contaminated():
    assert is_contaminated("wnba", "spread", "sharp+spread_dist+possession_sim", -5.5)


def test_wnba_spread_market_anchored_is_clean():
    # Current code: pure sharp+spread_dist, no possession_sim.
    assert not is_contaminated("wnba", "spread", "sharp+spread_dist", -5.5)


def test_wnba_moneyline_with_possession_sim_is_clean():
    # The rule is spread-only — WNBA ML legitimately blends possession_sim.
    assert not is_contaminated(
        "wnba", "moneyline", "wnba_possession_sim+wnba_efficiency+sharp"
    )


def test_wnba_total_with_possession_sim_is_clean():
    # Totals are a separate (shadow) market; the spread-pricing rule doesn't touch them.
    assert not is_contaminated("wnba", "total", "sharp+total_dist+possession_sim", 165.5)


# -------------------------------------------------------------------------
# Soccer — sharp-only ML rows predate the MIN_NONSHARP_MODELS guard
# (2026-07-19): the unseeded-MLS sharp-passthrough failure.
# -------------------------------------------------------------------------


def test_soccer_sharp_only_ml_is_contaminated():
    assert is_contaminated("soccer", "moneyline", "sharp")


def test_soccer_sharp_capped_ml_is_contaminated():
    assert is_contaminated("soccer", "moneyline", "sharp(capped)+injury")


def test_soccer_ml_with_any_model_is_clean():
    assert not is_contaminated("soccer", "moneyline", "elo+poisson+sharp")
    assert not is_contaminated("soccer", "moneyline", "xg+sharp")


def test_soccer_total_sharp_only_is_clean():
    # The guard is moneyline-scoped; sharp-only totals are expected (no model
    # prices soccer totals) and not a code-state signature.
    assert not is_contaminated("soccer", "total", "sharp+total_dist")


def test_soccer_empty_sources_ml_is_contaminated():
    assert is_contaminated("soccer", "moneyline", None)
    assert is_contaminated("soccer", "moneyline", "")


# -------------------------------------------------------------------------
# metrics / promote — command level against a baseball fixture DB
# -------------------------------------------------------------------------


def _make_baseball_db(tmp_path: Path) -> Path:
    """Predictions DB with a mix of clean + contaminated baseball ML rows.

    3 clean (pitcher, no poisson) + 4 contaminated (poisson or no-pitcher),
    all resolved. Mirrors the real "3 clean of many" shape.
    """
    db_path = tmp_path / "predictions.db"
    conn = sqlite3.connect(str(db_path))
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
            UNIQUE(market_id, scan_date)
        );
        CREATE TABLE ev_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_id TEXT UNIQUE,
            event_id TEXT,
            outcome INTEGER
        );
        """
    )
    sd = date.today().isoformat()
    ed = (date.today() - timedelta(days=1)).isoformat()
    # (market_id, model_sources, outcome) — all baseball moneyline, EV 5%, price 0.50
    rows = [
        ("clean1", "elo+pitcher+sharp", 1),
        ("clean2", "elo+pitcher+sharp", 0),
        ("clean3", "elo+pitcher+sharp", 1),
        ("dirty_poisson1", "elo+pitcher+poisson+sharp", 1),
        ("dirty_poisson2", "elo+poisson+sharp", 0),
        ("dirty_nopitch1", "elo+sharp", 1),
        ("dirty_nopitch2", "elo+sharp", 0),
    ]
    for mid, src, outcome in rows:
        conn.execute(
            """INSERT INTO ev_predictions
               (scan_date, market_id, event_id, sector, yes_team, market_type,
                event_title, event_date, kalshi_yes_price, sharp_true_prob,
                blended_true_prob, ev_pct, kelly_fraction, volume_usd,
                model_sources, mode, captured_yes_price)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (sd, mid, f"baseball::game::{mid}", "baseball", "yes_team", "moneyline",
             "Team A vs Team B", ed, 0.50, 0.55, 0.55, 0.05, 0.02, 100, src, "shadow", 0.50),
        )
        conn.execute(
            "INSERT INTO ev_outcomes (market_id, event_id, outcome) VALUES (?,?,?)",
            (mid, f"baseball::game::{mid}", outcome),
        )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def baseball_db(tmp_path, monkeypatch):
    db_path = _make_baseball_db(tmp_path)
    from evmax.agents.cleanup import db as db_module

    monkeypatch.setattr(db_module, "DB_PATH", db_path)
    return db_path


def test_metrics_excludes_contaminated_by_default(baseball_db):
    result = runner.invoke(app, ["metrics", "--days", "30", "--category", "baseball"])
    assert result.exit_code == 0
    # 3 clean rows scored, 4 excluded.
    assert "Excl = rows from a superseded code state" in result.stdout
    assert "4 total" in result.stdout


def test_metrics_includes_contaminated_when_flagged(baseball_db):
    result = runner.invoke(
        app, ["metrics", "--days", "30", "--category", "baseball", "--include-contaminated"]
    )
    assert result.exit_code == 0
    # With the flag, nothing is dropped — the exclusion footer is absent.
    assert "superseded code state" not in result.stdout


def test_promote_refuses_on_thin_clean_sample(baseball_db, monkeypatch):
    # Point the YAML at a temp file with baseball in shadow mode so the
    # mode check passes and we reach the contamination gate.
    from evmax.cli.commands import shadow

    yaml = (
        "baseball:\n"
        '  display_name: "Baseball (MLB)"\n'
        "  mode: shadow\n"
        "  resolver: espn_scoreboard\n"
        "  status: shipped\n"
    )
    path = baseball_db.parent / "categories.yaml"
    path.write_text(yaml)
    monkeypatch.setattr(shadow, "_CATEGORIES_YAML", path)

    result = runner.invoke(app, ["promote", "baseball", "--yes"])
    assert result.exit_code == 1
    assert "current-code resolved row" in result.stdout
    # YAML untouched — still shadow.
    assert "mode: shadow" in path.read_text()


def test_promote_force_overrides_thin_sample(baseball_db, monkeypatch):
    from evmax.cli.commands import shadow

    yaml = (
        "baseball:\n"
        '  display_name: "Baseball (MLB)"\n'
        "  mode: shadow\n"
        "  resolver: espn_scoreboard\n"
        "  status: shipped\n"
    )
    path = baseball_db.parent / "categories.yaml"
    path.write_text(yaml)
    monkeypatch.setattr(shadow, "_CATEGORIES_YAML", path)

    with patch("evmax.categories.reload_registry", lambda *a, **k: None):
        with patch("evmax.categories.validate_registry", lambda *a, **k: None):
            result = runner.invoke(app, ["promote", "baseball", "--yes", "--force"])
    assert result.exit_code == 0
    assert "promoted to live" in result.stdout
    assert "mode: live" in path.read_text()
