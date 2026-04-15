"""Comprehensive tests for the cleanup pipeline.

Covers:
  - Resolver matching helpers: _slug_teams, _to_fuzz, _match_espn, _match_bo3
  - MaintenanceReport / run_maintenance: all 6 check/fix rules, transaction commit,
    fixes_applied count
  - logger.log_gaps + get_logged_market_ids: insert, dedup, date isolation

Uses in-memory SQLite for all database tests (no file I/O, no HTTP calls).
"""

from __future__ import annotations

import sqlite3
from datetime import date
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from evmax.agents.cleanup.resolver import (
    _match_bo3,
    _match_espn,
    _slug_teams,
    _to_fuzz,
)
from evmax.agents.cleanup.maintenance import (
    MaintenanceReport,
    _check_duplicate_events,
    _check_prob_divergence,
    _check_rule_violations,
    _check_stale_markets,
    run_maintenance,
)


# ---------------------------------------------------------------------------
# Shared DB schema (minimal — same columns as production, no voided column
# needed since get_connection() handles migration at runtime)
# ---------------------------------------------------------------------------

_PREDICTIONS_DDL = """
CREATE TABLE IF NOT EXISTS ev_predictions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at           TEXT    NOT NULL DEFAULT (datetime('now')),
    scan_date           TEXT    NOT NULL,
    market_id           TEXT    NOT NULL,
    event_id            TEXT    NOT NULL,
    sector              TEXT    NOT NULL,
    yes_team            TEXT    NOT NULL,
    market_type         TEXT    NOT NULL,
    event_title         TEXT,
    event_date          TEXT,
    kalshi_yes_price    REAL    NOT NULL,
    sharp_true_prob     REAL    NOT NULL,
    blended_true_prob   REAL    NOT NULL,
    ev_pct              REAL    NOT NULL,
    kelly_fraction      REAL    NOT NULL,
    volume_usd          REAL,
    model_sources       TEXT,
    sharp_weight_used   REAL,
    bankroll_used       REAL,
    line                REAL,
    voided              INTEGER NOT NULL DEFAULT 0,
    placed              INTEGER NOT NULL DEFAULT 0,
    placed_at           TEXT,
    placed_price        REAL,
    placed_stake        REAL,
    clv_pct             REAL,
    -- ARCH-11 mode columns:
    mode                TEXT    NOT NULL DEFAULT 'live',
    captured_yes_price  REAL,
    model_version       TEXT,
    UNIQUE(market_id)
);
"""

_OUTCOMES_DDL = """
CREATE TABLE IF NOT EXISTS ev_outcomes (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id           TEXT    NOT NULL UNIQUE,
    event_id            TEXT    NOT NULL,
    event_date          TEXT    NOT NULL DEFAULT '',
    sector              TEXT    NOT NULL,
    yes_team            TEXT    NOT NULL,
    outcome             INTEGER,
    sharp_true_prob     REAL,
    blended_true_prob   REAL,
    pinnacle_close_prob REAL,
    resolved_at         TEXT,
    result_source       TEXT
);
"""


def _make_in_memory_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_PREDICTIONS_DDL + _OUTCOMES_DDL)
    conn.commit()
    return conn


def _insert_prediction(
    conn: sqlite3.Connection,
    *,
    market_id: str,
    scan_date: str = "2026-03-21",
    event_id: str = "nba::2026-03-21::team_a_vs_team_b",
    sector: str = "nba",
    yes_team: str = "team a",
    market_type: str = "moneyline",
    kalshi_yes_price: float = 0.45,
    sharp_true_prob: float = 0.55,
    blended_true_prob: float = 0.55,
    ev_pct: float = 0.05,
    kelly_fraction: float = 0.025,
) -> None:
    conn.execute(
        """INSERT INTO ev_predictions
           (scan_date, market_id, event_id, sector, yes_team, market_type,
            kalshi_yes_price, sharp_true_prob, blended_true_prob,
            ev_pct, kelly_fraction, volume_usd, model_sources, sharp_weight_used)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            scan_date, market_id, event_id, sector, yes_team, market_type,
            kalshi_yes_price, sharp_true_prob, blended_true_prob,
            ev_pct, kelly_fraction, 10_000.0, "sharp", 0.85,
        ),
    )
    conn.commit()


def _insert_outcome(
    conn: sqlite3.Connection,
    *,
    market_id: str,
    outcome: Optional[int] = 1,
    event_id: str = "nba::2026-03-21::team_a_vs_team_b",
    sector: str = "nba",
    yes_team: str = "team a",
) -> None:
    conn.execute(
        """INSERT INTO ev_outcomes
           (market_id, event_id, event_date, sector, yes_team, outcome, result_source)
           VALUES (?,?,?,?,?,?,?)""",
        (market_id, event_id, "2026-03-21", sector, yes_team, outcome, "espn"),
    )
    conn.commit()


def _make_report(conn: sqlite3.Connection, scan_date: str = "2026-03-21") -> MaintenanceReport:
    rows = conn.execute(
        "SELECT * FROM ev_predictions WHERE scan_date = ?", (scan_date,)
    ).fetchall()
    return MaintenanceReport(scan_date=scan_date, total_predictions=len(rows))


# ---------------------------------------------------------------------------
# _slug_teams
# ---------------------------------------------------------------------------

class TestSlugTeams:

    def test_soccer_multiword_teams(self):
        a, b = _slug_teams("soccer::2026-03-20::atletico_madrid_vs_real_madrid")
        assert a == "atletico madrid"
        assert b == "real madrid"

    def test_event_id_without_vs_returns_empty_strings(self):
        a, b = _slug_teams("nba::2026-03-21::unknown_game")
        assert a == ""
        assert b == ""

    def test_two_part_event_id_returns_empty_strings(self):
        a, b = _slug_teams("nba::2026-03-21")
        assert a == ""
        assert b == ""

    def test_empty_string_returns_empty_strings(self):
        a, b = _slug_teams("")
        assert a == ""
        assert b == ""

    def test_single_word_teams(self):
        a, b = _slug_teams("nba::2026-03-21::pistons_vs_wizards")
        assert a == "pistons"
        assert b == "wizards"

    def test_extra_suffix_still_parses(self):
        # parts[2] still has _vs_, extra parts after :: are ignored
        a, b = _slug_teams("nba::2026-03-21::pistons_vs_wizards::spread")
        assert a == "pistons"
        assert b == "wizards"


# ---------------------------------------------------------------------------
# _to_fuzz
# ---------------------------------------------------------------------------

class TestToFuzz:

    def test_lowercases_input(self):
        assert _to_fuzz("Los Angeles Lakers") == "los angeles lakers"

    @pytest.mark.parametrize("name,expected", [
        # _to_fuzz strips punctuation (.,&,',_,-) and lowercases;
        # it does NOT transliterate Unicode accents — that is intentional
        # (rapidfuzz token_set_ratio handles diacritics gracefully at threshold=72).
        ("Saint Mary's",       "saint marys"),
        ("Texas A&M",          "texas a m"),
        ("St. Louis Blues",    "st louis blues"),
        ("San-Antonio Spurs",  "san antonio spurs"),
        ("real_madrid",        "real madrid"),
        ("Atlético de Madrid", "atletico de madrid"),  # accent stripped by unicode normalisation
    ])
    def test_normalization(self, name: str, expected: str):
        assert _to_fuzz(name) == expected

    def test_strips_surrounding_whitespace(self):
        assert _to_fuzz("  warriors  ") == "warriors"

    def test_ampersand_not_in_result(self):
        result = _to_fuzz("Texas A&M Aggies")
        assert "&" not in result


# ---------------------------------------------------------------------------
# _match_espn helpers
# ---------------------------------------------------------------------------

def _score(home: str, home_pts: int, away: str, away_pts: int) -> dict:
    return {
        "home_name":  home,
        "away_name":  away,
        "home_score": home_pts,
        "away_score": away_pts,
        "home_won":   home_pts > away_pts,
    }


def _pred(event_id: str, yes_team: str) -> dict:
    return {"event_id": event_id, "yes_team": yes_team}


class TestMatchEspn:

    # --- YES = home, home wins ---
    def test_yes_home_team_wins(self):
        scores = [_score("Detroit Pistons", 117, "Washington Wizards", 95)]
        result = _match_espn(_pred("nba::2026-03-21::pistons_vs_wizards", "pistons"), scores)
        assert result == 1

    # --- YES = home, home loses ---
    def test_yes_home_team_loses(self):
        scores = [_score("Charlotte Hornets", 100, "Orlando Magic", 115)]
        result = _match_espn(_pred("nba::2026-03-21::hornets_vs_magic", "hornets"), scores)
        assert result == 0

    # --- YES = away, away wins ---
    def test_yes_away_team_wins(self):
        scores = [_score("Washington Wizards", 95, "Detroit Pistons", 117)]
        result = _match_espn(_pred("nba::2026-03-21::pistons_vs_wizards", "pistons"), scores)
        assert result == 1

    # --- YES = away, away loses ---
    def test_yes_away_team_loses(self):
        scores = [_score("Orlando Magic", 115, "Charlotte Hornets", 100)]
        result = _match_espn(_pred("nba::2026-03-21::hornets_vs_magic", "hornets"), scores)
        assert result == 0

    # --- Fuzzy name mismatch still matches (threshold=72) ---
    def test_fuzzy_name_variant_still_matches(self):
        # "Atletico Madrid" (no accent/de) vs "Atletico de Madrid" — should still match
        scores = [_score("Atletico de Madrid", 2, "Real Madrid", 1)]
        result = _match_espn(
            _pred("soccer::2026-03-21::atletico_madrid_vs_real_madrid", "atletico madrid"),
            scores,
        )
        assert result == 1

    # --- YES team not in any score → None ---
    def test_yes_team_not_in_any_score_returns_none(self):
        scores = [_score("Los Angeles Lakers", 110, "Golden State Warriors", 105)]
        result = _match_espn(_pred("nba::2026-03-21::pistons_vs_wizards", "pistons"), scores)
        assert result is None

    # --- Empty scores list → None ---
    def test_empty_scores_returns_none(self):
        result = _match_espn(_pred("nba::2026-03-21::pistons_vs_wizards", "pistons"), [])
        assert result is None

    # --- Both teams don't match any score → None ---
    def test_both_teams_absent_returns_none(self):
        scores = [_score("Bulls", 98, "Celtics", 102)]
        result = _match_espn(_pred("nba::2026-03-21::pistons_vs_wizards", "pistons"), scores)
        assert result is None

    # --- Event-identity gate: score with entirely different teams is skipped ---
    def test_event_gate_prevents_cross_match(self):
        # Score is for Lakers vs Warriors; pred is for pistons vs wizards.
        # The gate must reject this score so we get None instead of a wrong result.
        scores = [_score("Los Angeles Lakers", 120, "Golden State Warriors", 115)]
        result = _match_espn(
            _pred("nba::2026-03-21::pistons_vs_wizards", "pistons"),
            scores,
        )
        assert result is None

    # --- Multiple scores: correct game selected ---
    def test_multiple_scores_correct_game_selected(self):
        scores = [
            _score("Los Angeles Lakers", 120, "Golden State Warriors", 115),
            _score("Detroit Pistons", 117, "Washington Wizards", 95),
            _score("Chicago Bulls", 98, "Boston Celtics", 102),
        ]
        result = _match_espn(_pred("nba::2026-03-21::pistons_vs_wizards", "pistons"), scores)
        assert result == 1


# ---------------------------------------------------------------------------
# _match_bo3 helpers
# ---------------------------------------------------------------------------

def _bo3(t1: str, t1s: int, t2: str, t2s: int) -> dict:
    return {
        "team1_name":  t1,
        "team2_name":  t2,
        "team1_score": t1s,
        "team2_score": t2s,
        "team1_won":   t1s > t2s,
    }


class TestMatchBo3:

    # --- YES = team1, team1 wins ---
    def test_yes_team1_wins(self):
        scores = [_bo3("NaVi", 2, "Team Vitality", 0)]
        pred = {"event_id": "cs2::2026-03-21::navi_vs_vitality", "yes_team": "navi"}
        assert _match_bo3(pred, scores) == 1

    # --- YES = team2, team2 wins ---
    def test_yes_team2_wins(self):
        scores = [_bo3("NaVi", 0, "Team Vitality", 2)]
        pred = {"event_id": "cs2::2026-03-21::navi_vs_vitality", "yes_team": "vitality"}
        assert _match_bo3(pred, scores) == 1

    # --- YES = team1, team1 loses ---
    def test_yes_team1_loses(self):
        scores = [_bo3("NaVi", 0, "Team Vitality", 2)]
        pred = {"event_id": "cs2::2026-03-21::navi_vs_vitality", "yes_team": "navi"}
        assert _match_bo3(pred, scores) == 0

    # --- event_id without _vs_ → None ---
    def test_event_id_without_vs_returns_none(self):
        scores = [_bo3("NaVi", 2, "Team Vitality", 0)]
        pred = {"event_id": "cs2::2026-03-21::unknown", "yes_team": "navi"}
        assert _match_bo3(pred, scores) is None

    # --- yes_team doesn't match either slug → None ---
    def test_yes_team_no_slug_match_returns_none(self):
        scores = [_bo3("NaVi", 2, "Team Vitality", 0)]
        pred = {"event_id": "cs2::2026-03-21::navi_vs_vitality", "yes_team": "faze clan"}
        assert _match_bo3(pred, scores) is None

    # --- Wrong match in scores list (different teams) is skipped by gate ---
    def test_event_gate_skips_wrong_match(self):
        # Score is for Team Liquid vs G2; pred is for navi vs vitality
        scores = [_bo3("Team Liquid", 2, "G2 Esports", 1)]
        pred = {"event_id": "cs2::2026-03-21::navi_vs_vitality", "yes_team": "navi"}
        assert _match_bo3(pred, scores) is None

    # --- team1/team2 swapped in score data still resolved correctly ---
    def test_team1_team2_swapped_in_score(self):
        # event slug: navi (slug_a) vs vitality (slug_b)
        # score has team1=vitality, team2=navi (reversed order)
        scores = [_bo3("Team Vitality", 0, "NaVi", 2)]
        pred = {"event_id": "cs2::2026-03-21::navi_vs_vitality", "yes_team": "navi"}
        assert _match_bo3(pred, scores) == 1


# ---------------------------------------------------------------------------
# MaintenanceReport dataclass
# ---------------------------------------------------------------------------

class TestMaintenanceReport:

    def _make_issue(self, level="error", check="ev_below_threshold",
                    market_id="kalshi:X", event_id="ev::1"):
        from evmax.agents.cleanup.maintenance import MaintenanceIssue
        return MaintenanceIssue(
            level=level, check=check,
            market_id=market_id, event_id=event_id,
            detail="detail", fix="fix",
        )

    def test_ok_true_when_no_issues(self):
        report = MaintenanceReport(scan_date="2026-03-21", total_predictions=0)
        assert report.ok is True
        assert report.issues == []

    def test_ok_false_when_issues_present(self):
        report = MaintenanceReport(scan_date="2026-03-21", total_predictions=1,
                                   issues=[self._make_issue()])
        assert report.ok is False

    def test_errors_property_filters_level(self):
        report = MaintenanceReport(
            scan_date="2026-03-21", total_predictions=2,
            issues=[
                self._make_issue(level="error"),
                self._make_issue(level="warning", check="duplicate_event"),
            ],
        )
        assert len(report.errors) == 1
        assert len(report.warnings) == 1

    def test_summary_ok(self):
        report = MaintenanceReport(scan_date="2026-03-21", total_predictions=5)
        assert "OK" in report.summary()

    def test_summary_with_issues(self):
        report = MaintenanceReport(
            scan_date="2026-03-21", total_predictions=2, fixes_applied=1,
            issues=[self._make_issue()],
        )
        summary = report.summary()
        assert "error" in summary
        assert "fix" in summary


# ---------------------------------------------------------------------------
# run_maintenance via mocked get_connection
# ---------------------------------------------------------------------------

def _make_maintenance_ctx(conn: sqlite3.Connection):
    """Return a context manager that yields the given connection."""
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=conn)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


class TestRunMaintenance:

    SCAN_DATE = date(2026, 3, 21)
    SD = "2026-03-21"

    def _run(self, conn: sqlite3.Connection) -> MaintenanceReport:
        ctx = _make_maintenance_ctx(conn)
        with patch("evmax.agents.cleanup.maintenance.get_connection", return_value=ctx):
            return run_maintenance(self.SCAN_DATE)

    # ------------------------------------------------------------------
    # Empty table → no issues
    # ------------------------------------------------------------------

    def test_empty_predictions_table_returns_ok(self):
        conn = _make_in_memory_db()
        report = self._run(conn)
        assert report.ok is True
        assert report.issues == []
        assert report.total_predictions == 0

    # ------------------------------------------------------------------
    # Check 1: ev_below_threshold → row deleted
    # ------------------------------------------------------------------

    def test_ev_below_threshold_deletes_row(self):
        conn = _make_in_memory_db()
        _insert_prediction(conn, market_id="kalshi:LOW-EV", ev_pct=0.015,
                           scan_date=self.SD)

        report = self._run(conn)

        assert len(report.errors) == 1
        assert report.errors[0].check == "ev_below_threshold"
        assert report.fixes_applied == 1

        remaining = conn.execute(
            "SELECT * FROM ev_predictions WHERE market_id = ? AND scan_date = ?",
            ("kalshi:LOW-EV", self.SD),
        ).fetchall()
        assert len(remaining) == 0

    def test_ev_at_threshold_not_deleted(self):
        conn = _make_in_memory_db()
        _insert_prediction(conn, market_id="kalshi:OK-EV", ev_pct=0.02,
                           scan_date=self.SD)

        report = self._run(conn)

        # No ev_below_threshold issues
        below = [i for i in report.issues if i.check == "ev_below_threshold"]
        assert len(below) == 0
        row = conn.execute(
            "SELECT * FROM ev_predictions WHERE market_id = ?",
            ("kalshi:OK-EV",),
        ).fetchone()
        assert row is not None

    # ------------------------------------------------------------------
    # Check 2: kelly_above_cap → clamped to 0.05
    # ------------------------------------------------------------------

    def test_kelly_above_cap_clamped(self):
        conn = _make_in_memory_db()
        _insert_prediction(conn, market_id="kalshi:BIG-KELLY", kelly_fraction=0.07,
                           scan_date=self.SD)

        report = self._run(conn)

        kelly_issues = [i for i in report.issues if i.check == "kelly_above_cap"]
        assert len(kelly_issues) == 1
        assert report.fixes_applied >= 1

        row = conn.execute(
            "SELECT kelly_fraction FROM ev_predictions WHERE market_id = ?",
            ("kalshi:BIG-KELLY",),
        ).fetchone()
        assert row is not None
        assert row["kelly_fraction"] == pytest.approx(0.05)

    def test_kelly_at_cap_not_flagged(self):
        conn = _make_in_memory_db()
        _insert_prediction(conn, market_id="kalshi:CAP-KELLY", kelly_fraction=0.05,
                           scan_date=self.SD)

        report = self._run(conn)

        kelly_issues = [i for i in report.issues if i.check == "kelly_above_cap"]
        assert len(kelly_issues) == 0

    # ------------------------------------------------------------------
    # Check 3: invalid_probability → row deleted
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("bad_field,bad_value", [
        ("kalshi_yes_price",  1.5),
        ("kalshi_yes_price",  0.0),
        ("sharp_true_prob",   1.1),
        ("blended_true_prob", -0.1),
    ])
    def test_invalid_probability_deletes_row(self, bad_field: str, bad_value: float):
        conn = _make_in_memory_db()
        kwargs: dict = {
            "market_id": f"kalshi:BAD-PROB-{bad_field}",
            "scan_date": self.SD,
            bad_field: bad_value,
        }
        _insert_prediction(conn, **kwargs)

        report = self._run(conn)

        prob_issues = [i for i in report.issues if i.check == "invalid_probability"]
        assert len(prob_issues) >= 1

        remaining = conn.execute(
            "SELECT * FROM ev_predictions WHERE market_id = ?",
            (f"kalshi:BAD-PROB-{bad_field}",),
        ).fetchall()
        assert len(remaining) == 0

    # ------------------------------------------------------------------
    # Check 4: duplicate_event → lower-EV row deleted
    # ------------------------------------------------------------------

    def test_duplicate_event_keeps_higher_ev(self):
        conn = _make_in_memory_db()
        event_id = "nba::2026-03-21::team_a_vs_team_b"

        _insert_prediction(
            conn, market_id="kalshi:DUP-HI", scan_date=self.SD,
            event_id=event_id, sector="nba", yes_team="team a",
            ev_pct=0.08,
        )
        _insert_prediction(
            conn, market_id="kalshi:DUP-LO", scan_date=self.SD,
            event_id=event_id, sector="nba", yes_team="team a",
            ev_pct=0.04,
        )

        report = self._run(conn)

        dup_issues = [i for i in report.issues if i.check == "duplicate_event"]
        assert len(dup_issues) == 1
        assert report.fixes_applied >= 1

        # Higher EV row must survive
        hi = conn.execute(
            "SELECT * FROM ev_predictions WHERE market_id = ?",
            ("kalshi:DUP-HI",),
        ).fetchone()
        assert hi is not None

        # Lower EV row must be gone
        lo = conn.execute(
            "SELECT * FROM ev_predictions WHERE market_id = ?",
            ("kalshi:DUP-LO",),
        ).fetchone()
        assert lo is None

    def test_no_false_positive_duplicates_for_different_yes_teams(self):
        """Same event_id but different yes_team are NOT duplicates."""
        conn = _make_in_memory_db()
        event_id = "nba::2026-03-21::team_a_vs_team_b"

        _insert_prediction(
            conn, market_id="kalshi:SIDE-A", scan_date=self.SD,
            event_id=event_id, sector="nba", yes_team="team a", ev_pct=0.05,
        )
        _insert_prediction(
            conn, market_id="kalshi:SIDE-B", scan_date=self.SD,
            event_id=event_id, sector="nba", yes_team="team b", ev_pct=0.05,
        )

        report = self._run(conn)

        dup_issues = [i for i in report.issues if i.check == "duplicate_event"]
        assert len(dup_issues) == 0

    # ------------------------------------------------------------------
    # Check 5: stale_resolved_market → deleted from predictions
    # ------------------------------------------------------------------

    def test_stale_resolved_market_deleted(self):
        conn = _make_in_memory_db()
        _insert_prediction(conn, market_id="kalshi:RESOLVED", scan_date=self.SD)
        _insert_outcome(conn, market_id="kalshi:RESOLVED", outcome=1)

        report = self._run(conn)

        stale = [i for i in report.issues if i.check == "stale_resolved_market"]
        assert len(stale) == 1
        assert report.fixes_applied >= 1

        remaining = conn.execute(
            "SELECT * FROM ev_predictions WHERE market_id = ?",
            ("kalshi:RESOLVED",),
        ).fetchall()
        assert len(remaining) == 0

    def test_pending_outcome_not_deleted(self):
        """An ev_outcomes row with outcome=NULL is not yet resolved — keep prediction."""
        conn = _make_in_memory_db()
        _insert_prediction(conn, market_id="kalshi:PENDING", scan_date=self.SD)
        _insert_outcome(conn, market_id="kalshi:PENDING", outcome=None)

        report = self._run(conn)

        stale = [i for i in report.issues if i.check == "stale_resolved_market"]
        assert len(stale) == 0

        row = conn.execute(
            "SELECT * FROM ev_predictions WHERE market_id = ?",
            ("kalshi:PENDING",),
        ).fetchone()
        assert row is not None

    # ------------------------------------------------------------------
    # Check 6: prob_divergence → blended reset to sharp
    # ------------------------------------------------------------------

    def test_prob_divergence_resets_blended_to_sharp(self):
        conn = _make_in_memory_db()
        # blended=0.90, sharp=0.40 → drift = |0.90-0.40|/0.40 = 125% > 50%
        _insert_prediction(
            conn, market_id="kalshi:DRIFT", scan_date=self.SD,
            sharp_true_prob=0.40, blended_true_prob=0.90,
        )

        report = self._run(conn)

        drift_issues = [i for i in report.issues if i.check == "prob_divergence"]
        assert len(drift_issues) == 1
        assert report.fixes_applied >= 1

        row = conn.execute(
            "SELECT blended_true_prob FROM ev_predictions WHERE market_id = ?",
            ("kalshi:DRIFT",),
        ).fetchone()
        assert row is not None
        assert row["blended_true_prob"] == pytest.approx(0.40)

    def test_small_divergence_not_flagged(self):
        """blended within 50% of sharp must not trigger prob_divergence."""
        conn = _make_in_memory_db()
        # blended=0.55, sharp=0.50 → drift = 10% < 50%
        _insert_prediction(
            conn, market_id="kalshi:OK-BLEND", scan_date=self.SD,
            sharp_true_prob=0.50, blended_true_prob=0.55,
        )

        report = self._run(conn)

        drift_issues = [i for i in report.issues if i.check == "prob_divergence"]
        assert len(drift_issues) == 0

    # ------------------------------------------------------------------
    # Transaction atomicity: all fixes committed in single pass
    # ------------------------------------------------------------------

    def test_all_fixes_committed_in_single_transaction(self):
        """After run_maintenance the DB reflects every fix (re-query verifies commit)."""
        conn = _make_in_memory_db()

        # Row 1: ev below threshold → should be deleted
        _insert_prediction(conn, market_id="kalshi:DEL", ev_pct=0.01, scan_date=self.SD)

        # Row 2: kelly above cap → should be clamped
        _insert_prediction(conn, market_id="kalshi:CLAMP", kelly_fraction=0.09,
                           ev_pct=0.04,
                           event_id="nba::2026-03-21::team_c_vs_team_d",
                           yes_team="team c",
                           scan_date=self.SD)

        report = self._run(conn)

        # Verify deletions are visible in same connection post-commit
        deleted = conn.execute(
            "SELECT * FROM ev_predictions WHERE market_id = ?",
            ("kalshi:DEL",),
        ).fetchall()
        assert len(deleted) == 0

        # Verify clamp is visible
        clamped = conn.execute(
            "SELECT kelly_fraction FROM ev_predictions WHERE market_id = ?",
            ("kalshi:CLAMP",),
        ).fetchone()
        assert clamped is not None
        assert clamped["kelly_fraction"] == pytest.approx(0.05)

    # ------------------------------------------------------------------
    # fixes_applied count accuracy
    # ------------------------------------------------------------------

    def test_fixes_applied_count_matches_actual_fixes(self):
        conn = _make_in_memory_db()

        # 2 ev_below_threshold deletions
        _insert_prediction(conn, market_id="kalshi:FIX-1", ev_pct=0.005,
                           scan_date=self.SD)
        _insert_prediction(conn, market_id="kalshi:FIX-2", ev_pct=0.010,
                           scan_date=self.SD,
                           event_id="nba::2026-03-21::x_vs_y",
                           yes_team="x")

        # 1 kelly cap clamp
        _insert_prediction(conn, market_id="kalshi:FIX-3", kelly_fraction=0.10,
                           ev_pct=0.06,
                           event_id="nba::2026-03-21::p_vs_q",
                           yes_team="p",
                           scan_date=self.SD)

        report = self._run(conn)

        # 2 deletions + 1 clamp = 3 fixes minimum
        assert report.fixes_applied >= 3


# ---------------------------------------------------------------------------
# logger.log_gaps + get_logged_market_ids
# ---------------------------------------------------------------------------

def _make_ev_gap(
    market_id: str = "kalshi:GAP-001",
    event_id: str = "nba::2026-03-21::pistons_vs_wizards",
    sector: str = "nba",
    yes_team: str = "pistons",
    ev_pct: float = 0.05,
    sharp_true_prob: float = 0.55,
    blended_true_prob: float = 0.55,
):
    from evmax.agents.odds.ev_gap_agent import EVGap
    return EVGap(
        market_id=market_id,
        event_id=event_id,
        sector=sector,
        yes_team=yes_team,
        market_type="moneyline",
        kalshi_yes_price=0.40,
        sharp_true_prob=sharp_true_prob,
        blended_true_prob=blended_true_prob,
        ev_pct=ev_pct,
        kelly_full=0.10,
        kelly_fraction=0.025,
        match_confidence=0.95,
        volume_usd=10_000.0,
        spread_pct=0.02,
        event_date=None,
        model_sources="sharp",
        event_title="Test Game",
    )


def _logger_ctx(conn: sqlite3.Connection):
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=conn)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


class TestLogGaps:

    SCAN_DATE = date(2026, 3, 21)

    def _run_log(self, conn, gaps, scan_date=None):
        from evmax.agents.cleanup.logger import log_gaps
        ctx = _logger_ctx(conn)
        with patch("evmax.agents.cleanup.logger.get_connection", return_value=ctx):
            return log_gaps(gaps, scan_date=scan_date or self.SCAN_DATE)

    def test_logs_n_gaps_returns_n_inserted(self):
        conn = _make_in_memory_db()
        gaps = [_make_ev_gap(f"kalshi:G{i}") for i in range(5)]
        n = self._run_log(conn, gaps)
        assert n == 5

    def test_rows_present_after_insert(self):
        conn = _make_in_memory_db()
        gaps = [_make_ev_gap("kalshi:ROW-A"), _make_ev_gap("kalshi:ROW-B")]
        self._run_log(conn, gaps)
        rows = conn.execute("SELECT market_id FROM ev_predictions").fetchall()
        ids = {r["market_id"] for r in rows}
        assert "kalshi:ROW-A" in ids
        assert "kalshi:ROW-B" in ids

    def test_duplicate_market_id_same_date_returns_zero(self):
        conn = _make_in_memory_db()
        gap = _make_ev_gap("kalshi:DUPE")
        n1 = self._run_log(conn, [gap])
        n2 = self._run_log(conn, [gap])
        assert n1 == 1
        assert n2 == 0  # INSERT OR IGNORE

    def test_duplicate_persists_original_row(self):
        conn = _make_in_memory_db()
        gap = _make_ev_gap("kalshi:DUPE-KEEP", ev_pct=0.07)
        self._run_log(conn, [gap])

        # Try to insert again (same market_id + scan_date)
        self._run_log(conn, [gap])

        rows = conn.execute(
            "SELECT * FROM ev_predictions WHERE market_id = ?",
            ("kalshi:DUPE-KEEP",),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["ev_pct"] == pytest.approx(0.07)

    def test_empty_gaps_list_returns_zero(self):
        conn = _make_in_memory_db()
        n = self._run_log(conn, [])
        assert n == 0

    def test_rescan_on_later_date_is_noop_freeze_on_first_insert(self):
        """Under UNIQUE(market_id), the second scan is ignored entirely —
        the row's scan_date and snapshot fields stay frozen at first-flag time."""
        conn = _make_in_memory_db()
        gap1 = _make_ev_gap("kalshi:FREEZE", sharp_true_prob=0.60, blended_true_prob=0.65)
        gap2 = _make_ev_gap("kalshi:FREEZE", sharp_true_prob=0.70, blended_true_prob=0.75)

        n1 = self._run_log(conn, [gap1], scan_date=date(2026, 3, 20))
        n2 = self._run_log(conn, [gap2], scan_date=date(2026, 3, 21))

        assert n1 == 1
        assert n2 == 0  # INSERT OR IGNORE skipped

        row = conn.execute(
            "SELECT scan_date, sharp_true_prob, blended_true_prob FROM ev_predictions "
            "WHERE market_id = ?", ("kalshi:FREEZE",)
        ).fetchone()
        assert row["scan_date"] == "2026-03-20"
        assert row["sharp_true_prob"] == 0.60
        assert row["blended_true_prob"] == 0.65

    def test_placed_row_untouchable_by_later_scans(self):
        """If a market was placed (user hit 'pick'), a later re-log must not
        disturb the placed_* fields or the frozen snapshot."""
        conn = _make_in_memory_db()
        gap1 = _make_ev_gap("kalshi:PLACED", sharp_true_prob=0.55)
        self._run_log(conn, [gap1], scan_date=date(2026, 3, 20))

        # Simulate `pick` stamping the row
        conn.execute(
            "UPDATE ev_predictions SET placed = 1, placed_at = ?, "
            "placed_price = ?, placed_stake = ? WHERE market_id = ?",
            ("2026-03-20T18:00:00+00:00", 0.54, 25.0, "kalshi:PLACED"),
        )
        conn.commit()

        # Re-log the same market with different values
        gap2 = _make_ev_gap("kalshi:PLACED", sharp_true_prob=0.80)
        n2 = self._run_log(conn, [gap2], scan_date=date(2026, 3, 21))
        assert n2 == 0

        row = conn.execute(
            "SELECT sharp_true_prob, placed, placed_at, placed_price, placed_stake "
            "FROM ev_predictions WHERE market_id = ?", ("kalshi:PLACED",)
        ).fetchone()
        assert row["sharp_true_prob"] == 0.55  # frozen entry, not 0.80
        assert row["placed"] == 1
        assert row["placed_price"] == 0.54
        assert row["placed_stake"] == 25.0


class TestGetLoggedMarketIds:

    SCAN_DATE = date(2026, 3, 21)

    def _run_with_conn(self, conn, fn):
        ctx = _logger_ctx(conn)
        with patch("evmax.agents.cleanup.logger.get_connection", return_value=ctx):
            return fn()

    def test_returns_correct_set_for_scan_date(self):
        from evmax.agents.cleanup.logger import get_logged_market_ids, log_gaps

        conn = _make_in_memory_db()
        gaps = [_make_ev_gap("kalshi:ID-A"), _make_ev_gap("kalshi:ID-B")]

        ctx = _logger_ctx(conn)
        with patch("evmax.agents.cleanup.logger.get_connection", return_value=ctx):
            log_gaps(gaps, scan_date=self.SCAN_DATE)
            ids = get_logged_market_ids(self.SCAN_DATE)

        assert ids == {"kalshi:ID-A", "kalshi:ID-B"}

    def test_different_date_returns_empty_set(self):
        from evmax.agents.cleanup.logger import get_logged_market_ids, log_gaps

        conn = _make_in_memory_db()
        gaps = [_make_ev_gap("kalshi:ID-X")]

        ctx = _logger_ctx(conn)
        with patch("evmax.agents.cleanup.logger.get_connection", return_value=ctx):
            log_gaps(gaps, scan_date=date(2026, 3, 20))
            ids = get_logged_market_ids(date(2026, 3, 21))  # different date

        assert len(ids) == 0

    def test_empty_db_returns_empty_set(self):
        from evmax.agents.cleanup.logger import get_logged_market_ids

        conn = _make_in_memory_db()
        ctx = _logger_ctx(conn)
        with patch("evmax.agents.cleanup.logger.get_connection", return_value=ctx):
            ids = get_logged_market_ids(self.SCAN_DATE)

        assert ids == set()

    def test_returns_set_not_list(self):
        from evmax.agents.cleanup.logger import get_logged_market_ids

        conn = _make_in_memory_db()
        ctx = _logger_ctx(conn)
        with patch("evmax.agents.cleanup.logger.get_connection", return_value=ctx):
            ids = get_logged_market_ids(self.SCAN_DATE)

        assert isinstance(ids, set)


class TestUniqueMarketIdMigration:
    """Covers the _migrate_unique_market_id rebuild in db.py. Simulates a
    legacy database with UNIQUE(market_id, scan_date) and multiple rows per
    market, runs the migration, and asserts the new constraint is in place
    with the best row retained per market."""

    def _make_legacy_db(self, path) -> sqlite3.Connection:
        """Create a DB matching the pre-migration schema exactly."""
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        conn.executescript("""
            CREATE TABLE ev_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                logged_at TEXT NOT NULL DEFAULT (datetime('now')),
                scan_date TEXT NOT NULL,
                market_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                sector TEXT NOT NULL,
                yes_team TEXT NOT NULL,
                market_type TEXT NOT NULL,
                event_title TEXT,
                event_date TEXT,
                kalshi_yes_price REAL NOT NULL,
                sharp_true_prob REAL NOT NULL,
                blended_true_prob REAL NOT NULL,
                ev_pct REAL NOT NULL,
                kelly_fraction REAL NOT NULL,
                volume_usd REAL,
                model_sources TEXT,
                sharp_weight_used REAL,
                bankroll_used REAL,
                line REAL,
                voided INTEGER NOT NULL DEFAULT 0,
                placed INTEGER NOT NULL DEFAULT 0,
                placed_at TEXT,
                placed_price REAL,
                placed_stake REAL,
                clv_pct REAL,
                UNIQUE(market_id, scan_date)
            );
            CREATE TABLE ev_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                event_date TEXT NOT NULL,
                sector TEXT NOT NULL,
                yes_team TEXT NOT NULL,
                outcome INTEGER,
                sharp_true_prob REAL,
                blended_true_prob REAL,
                pinnacle_close_prob REAL,
                resolved_at TEXT,
                result_source TEXT,
                UNIQUE(market_id)
            );
            CREATE TABLE prop_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                logged_at TEXT NOT NULL DEFAULT (datetime('now')),
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
                UNIQUE(market_id, scan_date)
            );
        """)
        return conn

    def test_migration_rebuilds_table_and_keeps_best_row_per_market(self, tmp_path):
        db_path = tmp_path / "legacy.db"
        conn = self._make_legacy_db(db_path)

        # Seed a market scanned 3 times, placed on the middle scan
        for sd, sharp, placed in [
            ("2026-04-10", 0.60, 0),
            ("2026-04-11", 0.65, 1),  # placed
            ("2026-04-12", 0.70, 0),
        ]:
            conn.execute(
                "INSERT INTO ev_predictions (scan_date, market_id, event_id, sector, "
                "yes_team, market_type, kalshi_yes_price, sharp_true_prob, "
                "blended_true_prob, ev_pct, kelly_fraction, placed) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (sd, "kalshi:LEGACY-1", "nba::evt::1", "nba", "team", "moneyline",
                 0.5, sharp, sharp, 0.05, 0.025, placed),
            )
        # And a market scanned twice, never placed — oldest should win
        for sd, sharp in [("2026-04-10", 0.55), ("2026-04-11", 0.60)]:
            conn.execute(
                "INSERT INTO ev_predictions (scan_date, market_id, event_id, sector, "
                "yes_team, market_type, kalshi_yes_price, sharp_true_prob, "
                "blended_true_prob, ev_pct, kelly_fraction) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (sd, "kalshi:LEGACY-2", "nba::evt::2", "nba", "team", "moneyline",
                 0.5, sharp, sharp, 0.05, 0.025),
            )
        conn.commit()
        conn.close()

        # Run the migration via get_connection()
        with patch("evmax.agents.cleanup.db.DB_PATH", db_path):
            from evmax.agents.cleanup.db import get_connection
            conn = get_connection()

        # 1. New UNIQUE(market_id) constraint is in place
        ddl = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='ev_predictions'"
        ).fetchone()[0]
        normalized = " ".join(ddl.split())
        assert "UNIQUE(market_id)" in normalized
        assert "UNIQUE(market_id, scan_date)" not in normalized

        # 2. Exactly one row per market_id survives
        rows = conn.execute(
            "SELECT market_id, scan_date, sharp_true_prob, placed "
            "FROM ev_predictions ORDER BY market_id"
        ).fetchall()
        assert len(rows) == 2

        # 3. LEGACY-1: placed row (2026-04-11, 0.65) wins
        legacy_1 = rows[0]
        assert legacy_1["market_id"] == "kalshi:LEGACY-1"
        assert legacy_1["placed"] == 1
        assert legacy_1["scan_date"] == "2026-04-11"
        assert legacy_1["sharp_true_prob"] == 0.65

        # 4. LEGACY-2: oldest scan (2026-04-10, 0.55) wins — freeze-on-first-insert
        legacy_2 = rows[1]
        assert legacy_2["market_id"] == "kalshi:LEGACY-2"
        assert legacy_2["scan_date"] == "2026-04-10"
        assert legacy_2["sharp_true_prob"] == 0.55

        # 5. Subsequent INSERT OR IGNORE of the same market is a no-op
        conn.execute(
            "INSERT OR IGNORE INTO ev_predictions (scan_date, market_id, event_id, "
            "sector, yes_team, market_type, kalshi_yes_price, sharp_true_prob, "
            "blended_true_prob, ev_pct, kelly_fraction) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("2026-04-20", "kalshi:LEGACY-2", "nba::evt::2", "nba", "team",
             "moneyline", 0.5, 0.99, 0.99, 0.99, 0.99),
        )
        conn.commit()
        row = conn.execute(
            "SELECT scan_date, sharp_true_prob FROM ev_predictions "
            "WHERE market_id = ?", ("kalshi:LEGACY-2",)
        ).fetchone()
        assert row["scan_date"] == "2026-04-10"
        assert row["sharp_true_prob"] == 0.55

        conn.close()

    def test_migration_is_idempotent(self, tmp_path):
        """Running get_connection twice must not fail or alter state."""
        db_path = tmp_path / "legacy_idem.db"
        conn = self._make_legacy_db(db_path)
        conn.execute(
            "INSERT INTO ev_predictions (scan_date, market_id, event_id, sector, "
            "yes_team, market_type, kalshi_yes_price, sharp_true_prob, "
            "blended_true_prob, ev_pct, kelly_fraction) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("2026-04-10", "kalshi:IDEM", "nba::evt::3", "nba", "team",
             "moneyline", 0.5, 0.60, 0.60, 0.05, 0.025),
        )
        conn.commit()
        conn.close()

        with patch("evmax.agents.cleanup.db.DB_PATH", db_path):
            from evmax.agents.cleanup.db import get_connection
            c1 = get_connection()
            c1.close()
            c2 = get_connection()  # second call — must be a no-op
            rows = c2.execute("SELECT COUNT(*) FROM ev_predictions").fetchone()
            assert rows[0] == 1
            c2.close()
