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
    # Likewise isolate the CLV gate: default to a clearing sample so the
    # flip-mechanics tests don't depend on the real predictions.db. The CLV
    # gate itself is exercised in TestPromoteClvGate below.
    monkeypatch.setattr(
        shadow, "clv_stats",
        lambda *_a, **_k: {"n": 50, "mean_clv_pp": 2.0, "frac_positive": 0.7, "clears": True},
    )
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


# -------------------------------------------------------------------------
# CLV promotion gate — clv_clears (pure) + clv_stats (DB) + promote wiring
# -------------------------------------------------------------------------
from evmax.cli.commands.shadow import (  # noqa: E402
    CLV_MIN_FRAC_POSITIVE,
    MIN_CLV_RESOLVED,
    clv_clears,
    clv_stats,
)


class TestClvClears:
    def test_clears_with_positive_mean_and_majority(self):
        assert clv_clears(n=40, mean_clv_pp=1.5, frac_positive=0.70) is True

    def test_small_positive_mean_but_minority_positive_is_noise(self):
        # Mirrors the real WNBA-spread recent sample: +0.2pp mean but only 44%
        # of bets beat the close — a few outliers, not edge. Must NOT clear.
        assert clv_clears(n=34, mean_clv_pp=0.21, frac_positive=0.44) is False

    def test_negative_mean_never_clears(self):
        assert clv_clears(n=100, mean_clv_pp=-0.46, frac_positive=0.35) is False

    def test_too_few_rows_never_clears(self):
        assert clv_clears(n=MIN_CLV_RESOLVED - 1, mean_clv_pp=5.0, frac_positive=0.9) is False

    def test_frac_at_threshold_clears(self):
        assert clv_clears(n=30, mean_clv_pp=0.0, frac_positive=CLV_MIN_FRAC_POSITIVE) is True


def _make_clv_db(tmp_path: Path, rows: list[tuple]) -> Path:
    """rows: (market_id, scan_date, market_type, mode, kalshi_clv_pct, outcome[, line[, venue]]).

    The 7th element (line) and 8th (venue) are optional: line defaults to NULL
    (only the side-split tests need it) and venue defaults to 'kalshi' (only the
    venue-split tests need it).

    Deliberately omits the old `UNIQUE(market_id, scan_date)` constraint so
    get_connection()'s table-rebuild migration is skipped (it references base
    columns like logged_at this minimal fixture doesn't carry). kalshi_clv_pct
    is created directly; the additive ALTER for it then no-ops.
    """
    db_path = tmp_path / "predictions.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE ev_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_date TEXT, market_id TEXT, event_id TEXT, sector TEXT,
            market_type TEXT, line REAL, model_sources TEXT, mode TEXT,
            kalshi_clv_pct REAL, venue TEXT NOT NULL DEFAULT 'kalshi',
            placed INTEGER DEFAULT 0, placed_at TEXT
        );
        CREATE TABLE ev_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_id TEXT UNIQUE, outcome INTEGER
        );
        """
    )
    for row in rows:
        mid, sd, mt, mode, clv, outcome = row[:6]
        line = row[6] if len(row) > 6 else None
        venue = row[7] if len(row) > 7 else "kalshi"
        conn.execute(
            """INSERT INTO ev_predictions
               (scan_date, market_id, event_id, sector, market_type, mode,
                kalshi_clv_pct, line, venue)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (sd, mid, f"wnba::{sd}::a_vs_b::spread", "wnba", mt, mode, clv, line, venue),
        )
        if outcome is not None:
            conn.execute(
                "INSERT INTO ev_outcomes (market_id, outcome) VALUES (?, ?)",
                (mid, outcome),
            )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def _patch_db(monkeypatch):
    def _apply(db_path):
        from evmax.agents.cleanup import db as db_module
        monkeypatch.setattr(db_module, "DB_PATH", db_path)
    return _apply


class TestClvStats:
    def test_breakeven_sample_does_not_clear(self, tmp_path, _patch_db):
        # 32 resolved spread rows: 14 positive CLV (44%), small positive mean.
        rows = []
        for i in range(14):
            rows.append((f"p{i}", "2026-06-10", "spread", "live", 2.0, 1))
        for i in range(18):
            rows.append((f"n{i}", "2026-06-10", "spread", "live", -1.0, 0))
        _patch_db(_make_clv_db(tmp_path, rows))
        s = clv_stats("wnba", market_type="spread")
        assert s["n"] == 32
        assert round(s["frac_positive"], 2) == 0.44
        assert s["clears"] is False  # positive mean but minority => noise

    def test_since_filter_drops_old_stale_rows(self, tmp_path, _patch_db):
        # Old rows are strongly negative; recent rows are clean positives.
        rows = [(f"old{i}", "2026-05-10", "spread", "live", -5.0, 0) for i in range(20)]
        rows += [(f"new{i}", "2026-06-15", "spread", "live", 3.0, 1) for i in range(30)]
        _patch_db(_make_clv_db(tmp_path, rows))
        pooled = clv_stats("wnba", market_type="spread")
        recent = clv_stats("wnba", market_type="spread", since="2026-06-01")
        assert pooled["mean_clv_pp"] < 0           # dragged down by stale rows
        assert recent["n"] == 30
        assert recent["frac_positive"] == 1.0
        assert recent["clears"] is True            # clean recent edge clears

    def test_market_type_filter_isolates_spread(self, tmp_path, _patch_db):
        rows = [("s1", "2026-06-10", "spread", "live", 1.0, 1),
                ("ml1", "2026-06-10", "moneyline", "live", -9.0, 0)]
        _patch_db(_make_clv_db(tmp_path, rows))
        s = clv_stats("wnba", market_type="spread")
        assert s["n"] == 1 and s["mean_clv_pp"] == 1.0

    def test_side_filter_splits_lay_from_take(self, tmp_path, _patch_db):
        # Mirrors the 2026-07 WNBA spread audit: laying rows carry the CLV,
        # scan-time taking rows bleed. Promotion must judge each side alone.
        rows = [(f"lay{i}", "2026-06-10", "spread", "shadow", 2.0, 1, -6.5)
                for i in range(3)]
        rows += [(f"take{i}", "2026-06-10", "spread", "shadow", -3.0, 0, 5.5)
                 for i in range(2)]
        rows += [("noline", "2026-06-10", "spread", "shadow", 9.0, 1, None)]
        _patch_db(_make_clv_db(tmp_path, rows))
        lay = clv_stats("wnba", market_type="spread", side="lay")
        take = clv_stats("wnba", market_type="spread", side="take")
        pooled = clv_stats("wnba", market_type="spread")
        assert lay["n"] == 3 and lay["mean_clv_pp"] == 2.0
        assert take["n"] == 2 and take["mean_clv_pp"] == -3.0
        # NULL-line rows are excluded from side splits but kept in the pool
        assert pooled["n"] == 6

    def test_side_filter_rejects_bad_value(self, tmp_path, _patch_db):
        _patch_db(_make_clv_db(tmp_path, [("s1", "2026-06-10", "spread", "live", 1.0, 1)]))
        with pytest.raises(ValueError, match="side must be"):
            clv_stats("wnba", market_type="spread", side="left")

    def test_venue_filter_isolates_kalshi_from_polymarket(self, tmp_path, _patch_db):
        # 2026-07-12 WNBA spread audit: one thin PolyUS row at -18pp must not be
        # allowed to flip the sign of a Kalshi-sized sample. Judge each book alone.
        rows = [(f"k{i}", "2026-06-10", "spread", "shadow", 2.0, 1, -6.5, "kalshi")
                for i in range(3)]
        rows += [("pm1", "2026-06-10", "spread", "shadow", -18.0, 0, -6.5,
                  "polymarket_us")]
        _patch_db(_make_clv_db(tmp_path, rows))
        kalshi = clv_stats("wnba", market_type="spread", venue="kalshi")
        poly = clv_stats("wnba", market_type="spread", venue="polymarket_us")
        pooled = clv_stats("wnba", market_type="spread")
        assert kalshi["n"] == 3 and kalshi["mean_clv_pp"] == 2.0
        assert poly["n"] == 1 and poly["mean_clv_pp"] == -18.0
        # Pooled is dragged negative by the single PolyUS outlier — the bug the
        # venue filter exists to avoid.
        assert pooled["n"] == 4 and pooled["mean_clv_pp"] < 0

    def test_venue_defaults_to_pooled(self, tmp_path, _patch_db):
        # venue=None (default) scores every venue, unchanged from prior behaviour.
        rows = [("k1", "2026-06-10", "spread", "live", 1.0, 1, None, "kalshi"),
                ("pm1", "2026-06-10", "spread", "live", -1.0, 0, None,
                 "polymarket_us")]
        _patch_db(_make_clv_db(tmp_path, rows))
        assert clv_stats("wnba", market_type="spread")["n"] == 2

    def test_venue_composes_with_side(self, tmp_path, _patch_db):
        # venue filter stacks with side: only kalshi lay rows survive.
        rows = [("k_lay", "2026-06-10", "spread", "shadow", 2.0, 1, -6.5, "kalshi"),
                ("k_take", "2026-06-10", "spread", "shadow", -3.0, 0, 5.5, "kalshi"),
                ("pm_lay", "2026-06-10", "spread", "shadow", -18.0, 0, -6.5,
                 "polymarket_us")]
        _patch_db(_make_clv_db(tmp_path, rows))
        s = clv_stats("wnba", market_type="spread", side="lay", venue="kalshi")
        assert s["n"] == 1 and s["mean_clv_pp"] == 2.0

    def test_venue_filter_rejects_bad_value(self, tmp_path, _patch_db):
        _patch_db(_make_clv_db(tmp_path, [("s1", "2026-06-10", "spread", "live", 1.0, 1)]))
        with pytest.raises(ValueError, match="venue must be"):
            clv_stats("wnba", market_type="spread", venue="draftkings")

    def test_max_staleness_excludes_stale_capture_rows(self, tmp_path, _patch_db, monkeypatch):
        # 2026-07-12 audit: exact-zero CLV rows split ~68% stale-capture (close
        # snapshot hours before T-30) / 32% genuine flat. The filter drops the
        # stale ones so frac_positive reflects real near-tip price action.
        rows = [("fresh_a", "2026-06-10", "spread", "shadow", 3.0, 1, -6.5),
                ("fresh_b", "2026-06-10", "spread", "shadow", 0.0, 0, -6.5),
                ("stale_a", "2026-06-10", "spread", "shadow", 0.0, 0, -6.5),
                ("stale_b", "2026-06-10", "spread", "shadow", 0.0, 1, -6.5)]
        _patch_db(_make_clv_db(tmp_path, rows))
        staleness = {"fresh_a": 0.5, "fresh_b": 1.0, "stale_a": 12.0, "stale_b": 18.0}
        monkeypatch.setattr(
            "evmax.archiver.DataArchiver.get_kalshi_close_staleness_h",
            lambda self, ticker, event_id, not_before=None: staleness.get(ticker),
        )
        s = clv_stats("wnba", market_type="spread", max_staleness_h=3.0)
        assert s["n"] == 2               # only the two fresh rows survive
        assert s["excluded_stale"] == 2  # the two stale-capture zeros dropped
        assert s["mean_clv_pp"] == pytest.approx(1.5)   # (3.0 + 0.0) / 2

    def test_max_staleness_none_skips_archive_and_keeps_all(self, tmp_path, _patch_db, monkeypatch):
        # Default (None) must not touch archive.db at all — unchanged behaviour.
        rows = [("a", "2026-06-10", "spread", "shadow", 2.0, 1, -6.5),
                ("b", "2026-06-10", "spread", "shadow", 0.0, 0, -6.5)]
        _patch_db(_make_clv_db(tmp_path, rows))

        def _boom(*a, **k):  # would fire only if the archive path ran
            raise AssertionError("archive accessed with max_staleness_h=None")

        monkeypatch.setattr(
            "evmax.archiver.DataArchiver.get_kalshi_close_staleness_h", _boom
        )
        s = clv_stats("wnba", market_type="spread")
        assert s["n"] == 2 and s["excluded_stale"] == 0

    def test_max_staleness_excludes_rows_without_anchor(self, tmp_path, _patch_db, monkeypatch):
        # A None staleness (no tipoff anchor / no snapshot) = untrustworthy close,
        # excluded just like an over-threshold one.
        rows = [("has_anchor", "2026-06-10", "spread", "shadow", 2.0, 1, -6.5),
                ("no_anchor", "2026-06-10", "spread", "shadow", 0.0, 0, -6.5)]
        _patch_db(_make_clv_db(tmp_path, rows))
        monkeypatch.setattr(
            "evmax.archiver.DataArchiver.get_kalshi_close_staleness_h",
            lambda self, ticker, event_id, not_before=None: (
                0.5 if ticker == "has_anchor" else None
            ),
        )
        s = clv_stats("wnba", market_type="spread", max_staleness_h=3.0)
        assert s["n"] == 1 and s["excluded_stale"] == 1

    def test_max_staleness_leaves_polymarket_rows_untouched(self, tmp_path, _patch_db, monkeypatch):
        # PolyUS rows carry no Kalshi close snapshot; the staleness filter must
        # pass them through, never drop them for lacking a Kalshi anchor.
        rows = [("k1", "2026-06-10", "spread", "shadow", 2.0, 1, -6.5, "kalshi"),
                ("pm1", "2026-06-10", "spread", "shadow", 1.0, 1, -6.5,
                 "polymarket_us")]
        _patch_db(_make_clv_db(tmp_path, rows))
        monkeypatch.setattr(
            "evmax.archiver.DataArchiver.get_kalshi_close_staleness_h",
            lambda self, ticker, event_id, not_before=None: 0.5,
        )
        s = clv_stats("wnba", market_type="spread", max_staleness_h=3.0)
        assert s["n"] == 2 and s["excluded_stale"] == 0

    def test_max_staleness_rejects_negative(self, tmp_path, _patch_db):
        _patch_db(_make_clv_db(tmp_path, [("s1", "2026-06-10", "spread", "live", 1.0, 1)]))
        with pytest.raises(ValueError, match="max_staleness_h must be"):
            clv_stats("wnba", market_type="spread", max_staleness_h=-1.0)


class TestPromoteClvGate:
    """The promote command refuses a category whose CLV does not clear."""

    def _yaml(self, tmp_path, monkeypatch):
        from evmax.cli.commands import shadow
        path = tmp_path / "categories.yaml"
        path.write_text(_SAMPLE_YAML)
        monkeypatch.setattr(shadow, "_CATEGORIES_YAML", path)
        monkeypatch.setattr(shadow, "_clean_resolved_count", lambda *_a, **_k: 9999)
        monkeypatch.setattr("evmax.categories.reload_registry", lambda *a, **k: None)
        monkeypatch.setattr("evmax.categories.validate_registry", lambda *a, **k: None)
        return path

    def test_refuses_when_clv_does_not_clear(self, tmp_path, monkeypatch):
        from evmax.cli.commands import shadow
        path = self._yaml(tmp_path, monkeypatch)
        monkeypatch.setattr(
            shadow, "clv_stats",
            lambda *_a, **_k: {"n": 48, "mean_clv_pp": -0.46,
                               "frac_positive": 0.35, "clears": False},
        )
        result = runner.invoke(app, ["promote", "nfl_props", "--yes"])
        assert result.exit_code == 1
        assert "CLV does not clear" in result.stdout
        assert "mode: shadow" in path.read_text()  # unchanged

    def test_force_overrides_weak_clv(self, tmp_path, monkeypatch):
        from evmax.cli.commands import shadow
        path = self._yaml(tmp_path, monkeypatch)
        monkeypatch.setattr(
            shadow, "clv_stats",
            lambda *_a, **_k: {"n": 48, "mean_clv_pp": -0.46,
                               "frac_positive": 0.35, "clears": False},
        )
        result = runner.invoke(app, ["promote", "nfl_props", "--yes", "--force"])
        assert result.exit_code == 0
        assert "promoted to live" in result.stdout

    def test_no_clv_data_falls_through_to_count_gate(self, tmp_path, monkeypatch):
        # n < MIN_CLV_RESOLVED → CLV gate is not enforced (count gate already
        # passed via the 9999 stub), so the promote proceeds.
        from evmax.cli.commands import shadow
        path = self._yaml(tmp_path, monkeypatch)
        monkeypatch.setattr(
            shadow, "clv_stats",
            lambda *_a, **_k: {"n": 5, "mean_clv_pp": -3.0,
                               "frac_positive": 0.2, "clears": False},
        )
        result = runner.invoke(app, ["promote", "nfl_props", "--yes"])
        assert result.exit_code == 0
        assert "promoted to live" in result.stdout
