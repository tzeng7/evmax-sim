"""Tests for `evmax cleanup shadow` commands.

Three targets:
  - _flip_mode_in_yaml (the promote YAML mutation helper)
  - `evmax cleanup shadow show` against an in-memory fixture DB
  - `evmax cleanup shadow metrics` against the same fixture
  - `evmax cleanup shadow promote` end-to-end using a temp YAML file

The promote YAML helper is the riskiest code because it text-edits the
registry YAML instead of parsing + re-serializing. Tests cover the
happy path, comment preservation, and refuses on wrong starting mode.
"""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from evmax.cli.commands.shadow import _flip_mode_in_yaml, app

runner = CliRunner()


# -------------------------------------------------------------------------
# _flip_mode_in_yaml — YAML mutation helper
# -------------------------------------------------------------------------


_SAMPLE_YAML = """# top comment

nba:
  display_name: "NBA"
  market_types: [moneyline, spread, total]
  mode: live  # inline comment
  resolver: espn_scoreboard
  status: shipped
  notes: null

nfl_props:
  display_name: "NFL player props"
  market_types: [player_prop]
  mode: shadow
  resolver: espn_boxscore
  status: blocked
  notes: "see MODEL-9"
"""


def test_flip_mode_happy_path():
    text, old = _flip_mode_in_yaml(_SAMPLE_YAML, "nfl_props", new_mode="live")
    assert old == "shadow"
    assert "  mode: live\n" in text  # nfl_props flipped
    # nba's mode line is untouched
    assert "  mode: live  # inline comment\n" in text


def test_flip_mode_preserves_comments():
    text, _ = _flip_mode_in_yaml(_SAMPLE_YAML, "nfl_props", new_mode="live")
    assert "# top comment" in text
    assert "# inline comment" in text  # other category's comment survives
    assert 'notes: "see MODEL-9"' in text


def test_flip_mode_preserves_all_other_fields():
    text, _ = _flip_mode_in_yaml(_SAMPLE_YAML, "nfl_props", new_mode="live")
    assert 'display_name: "NFL player props"' in text
    assert "market_types: [player_prop]" in text
    assert "resolver: espn_boxscore" in text
    assert "status: blocked" in text


def test_flip_mode_missing_category_returns_none():
    text, old = _flip_mode_in_yaml(_SAMPLE_YAML, "mystery", new_mode="live")
    assert old is None
    assert text == _SAMPLE_YAML  # unchanged


def test_flip_mode_strips_inline_comment_from_old_value():
    text, old = _flip_mode_in_yaml(_SAMPLE_YAML, "nba", new_mode="shadow")
    assert old == "live"  # comment stripped, just the value captured
    assert "mode: shadow" in text


def test_flip_mode_with_no_mode_line_in_block():
    yaml = """nba:
  display_name: "NBA"
  status: shipped
"""
    text, old = _flip_mode_in_yaml(yaml, "nba", new_mode="live")
    assert old is None


def test_flip_mode_leaves_other_categories_alone():
    text, _ = _flip_mode_in_yaml(_SAMPLE_YAML, "nfl_props", new_mode="live")
    # nba's mode stays "live  # inline comment"
    nba_section = text[text.index("nba:"):text.index("nfl_props:")]
    assert "mode: live  # inline comment" in nba_section


# -------------------------------------------------------------------------
# Promote command — end-to-end against a temp YAML
# -------------------------------------------------------------------------


@pytest.fixture
def temp_yaml(tmp_path, monkeypatch):
    """Point _CATEGORIES_YAML at a temp file seeded with the sample."""
    from evmax.cli.commands import shadow

    path = tmp_path / "categories.yaml"
    path.write_text(_SAMPLE_YAML)
    monkeypatch.setattr(shadow, "_CATEGORIES_YAML", path)
    # The promote contamination gate counts clean resolved rows against the
    # real predictions.db; these tests exercise only the YAML-flip mechanics,
    # so stub the count above the floor to isolate the flip from the gate.
    monkeypatch.setattr(shadow, "_clean_resolved_count", lambda *_a, **_k: 9999)
    # Also make reload_registry + validate_registry no-ops so we don't
    # need a valid full registry during the helper test
    with patch("evmax.categories.reload_registry", lambda *a, **k: None):
        with patch("evmax.categories.validate_registry", lambda *a, **k: None):
            yield path


def test_promote_flips_shadow_to_live(temp_yaml):
    result = runner.invoke(app, ["promote", "nfl_props", "--yes"])
    assert result.exit_code == 0
    assert "promoted to live" in result.stdout
    new_text = temp_yaml.read_text()
    # nfl_props mode is now live
    nfl_section = new_text[new_text.index("nfl_props:"):]
    assert "mode: live" in nfl_section
    assert "mode: shadow" not in nfl_section


def test_promote_refuses_if_already_live(temp_yaml):
    result = runner.invoke(app, ["promote", "nba", "--yes"])
    assert result.exit_code == 0  # not an error, just a no-op
    assert "already in `live`" in result.stdout


def test_promote_errors_on_unknown_category(temp_yaml):
    result = runner.invoke(app, ["promote", "mystery", "--yes"])
    assert result.exit_code == 1
    assert "Could not find" in result.stdout


# -------------------------------------------------------------------------
# show / metrics — require a fixture DB with shadow rows
# -------------------------------------------------------------------------


def _make_db_with_shadow_rows(tmp_path: Path) -> Path:
    """Build a tiny predictions DB with a mix of live and shadow rows."""
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
            event_date TEXT,
            sector TEXT,
            yes_team TEXT,
            outcome INTEGER,
            sharp_true_prob REAL,
            blended_true_prob REAL,
            resolved_at TEXT,
            result_source TEXT
        );
        """
    )
    sd = date.today().isoformat()
    ed = (date.today() - timedelta(days=1)).isoformat()
    # Two shadow rows (resolved, one win one loss), one live row (ignored by shadow show)
    shadow_rows = [
        ("m_s1", "nfl::2026::prop::QBA::passing_yards::249.5", "nfl", "QBA",
         "player_prop", "Team A vs Team B", ed, 0.45, 0.55, 0.55, 0.10, 0.02, 100, 0.45, "shadow"),
        ("m_s2", "nfl::2026::prop::QBB::passing_yards::299.5", "nfl", "QBB",
         "player_prop", "Team C vs Team D", ed, 0.30, 0.45, 0.45, 0.15, 0.01, 200, 0.30, "shadow"),
        ("m_live", "nba::2026::game:LAL-GSW", "nba", "LAL",
         "moneyline", "LAL vs GSW", ed, 0.40, 0.55, 0.55, 0.15, 0.03, 5000, 0.40, "live"),
    ]
    for (mid, eid, sector, team, mt, title, edate, kp, sp, bp, ev, kf, vol, cap, mode) in shadow_rows:
        conn.execute(
            """INSERT INTO ev_predictions
               (scan_date, market_id, event_id, sector, yes_team, market_type,
                event_title, event_date, kalshi_yes_price, sharp_true_prob,
                blended_true_prob, ev_pct, kelly_fraction, volume_usd,
                mode, captured_yes_price)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (sd, mid, eid, sector, team, mt, title, edate, kp, sp, bp, ev, kf, vol, mode, cap),
        )
    # Resolve m_s1 as a WIN, m_s2 as a LOSS, m_live as a WIN
    for mid, outcome in (("m_s1", 1), ("m_s2", 0), ("m_live", 1)):
        conn.execute(
            """INSERT INTO ev_outcomes
               (market_id, event_id, event_date, sector, yes_team, outcome)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (mid, "evt", ed, "nfl", "Team", outcome),
        )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def shadow_db(tmp_path, monkeypatch):
    db_path = _make_db_with_shadow_rows(tmp_path)
    from evmax.agents.cleanup import db as db_module

    monkeypatch.setattr(db_module, "DB_PATH", db_path)
    return db_path


def test_shadow_show_lists_only_shadow_rows(shadow_db):
    result = runner.invoke(app, ["show", "--days", "30"])
    assert result.exit_code == 0
    # The summary line survives rich truncation — check there:
    # "2 rows · 2 resolved · 1 wins" confirms only the two shadow rows
    # made it through the filter (the third, live NBA row, is excluded).
    assert "2 rows" in result.stdout
    assert "2 resolved" in result.stdout


def test_shadow_metrics_excludes_live_rows(shadow_db):
    result = runner.invoke(app, ["metrics", "--days", "30"])
    assert result.exit_code == 0
    # With 2 shadow rows (1 win, 1 loss), accuracy should be 50%
    assert "50.0%" in result.stdout or "50%" in result.stdout
    # n column shows 2
    assert "2 " in result.stdout or " 2\n" in result.stdout


def test_shadow_show_empty_prints_friendly_message(tmp_path, monkeypatch):
    # Fresh DB with no shadow rows
    db_path = tmp_path / "predictions.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """CREATE TABLE ev_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_date TEXT, market_id TEXT, event_id TEXT, sector TEXT,
            yes_team TEXT, market_type TEXT, event_title TEXT,
            event_date TEXT, kalshi_yes_price REAL, sharp_true_prob REAL,
            blended_true_prob REAL, ev_pct REAL, kelly_fraction REAL,
            volume_usd REAL, line REAL,
            mode TEXT NOT NULL DEFAULT 'live', captured_yes_price REAL,
            model_version TEXT
        );
        CREATE TABLE ev_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_id TEXT UNIQUE, outcome INTEGER
        );
        """
    )
    conn.commit()
    conn.close()
    from evmax.agents.cleanup import db as db_module

    monkeypatch.setattr(db_module, "DB_PATH", db_path)
    result = runner.invoke(app, ["show", "--days", "7"])
    assert result.exit_code == 0
    assert "No shadow predictions" in result.stdout
