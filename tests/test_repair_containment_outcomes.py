"""Tests for scripts/repair_containment_outcomes.py (no network, temp DBs)."""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "repair_containment_outcomes.py"
_spec = importlib.util.spec_from_file_location("repair_containment_outcomes", _SCRIPT)
repair = importlib.util.module_from_spec(_spec)
sys.modules["repair_containment_outcomes"] = repair  # dataclasses need the module registered
_spec.loader.exec_module(repair)


UTAH_EID = "ncaaf::2026-09-19::utah_vs_utah_state"
UTAH_MID = "kalshi:KXNCAAFGAME-26SEP19USUUTAH-USU"
UTAH_SCORE = {
    "home_name": "Utah Utes", "away_name": "Utah State Aggies",
    "home_score": 33, "away_score": 0, "home_won": True, "game_date": "2026-09-19",
}


def _build_dbs(tmp_path, monkeypatch) -> tuple[Path, Path]:
    import evmax.archiver as archiver_mod
    from evmax.agents.cleanup import db as cleanup_db

    pred_path = tmp_path / "predictions.db"
    arch_path = tmp_path / "archive.db"
    monkeypatch.setattr(cleanup_db, "DB_PATH", pred_path)
    monkeypatch.setattr(archiver_mod, "DB_PATH", arch_path)
    conn = cleanup_db.get_connection()  # real schema + migrations

    rows = [
        # (market_id, event_id, sector, yes_team, event_title, outcome, close, entry)
        (UTAH_MID, UTAH_EID, "ncaaf", "utah state", "Utah vs Utah State", 1, 0.97, 0.04),
        ("kalshi:KXNCAAFGAME-26SEP19USUUTAH-UTAH", UTAH_EID, "ncaaf", "utah",
         "Utah vs Utah State", 1, 0.97, 0.95),
        ("kalshi:KXNFLGAME-26SEP13BUFKC-KC", "nfl::2026-09-13::chiefs_vs_bills", "nfl",
         "chiefs", "Chiefs vs Bills", 0, 0.55, 0.52),
    ]
    for mid, eid, sector, yes, title, outcome, close, entry in rows:
        conn.execute(
            """INSERT INTO ev_predictions
               (scan_date, market_id, event_id, sector, yes_team, market_type, event_title,
                event_date, kalshi_yes_price, sharp_true_prob, blended_true_prob, ev_pct,
                kelly_fraction, pinnacle_drift_pct, mode)
               VALUES (?,?,?,?,?,'moneyline',?,?,?,0.5,0.5,0.05,0.0,?,'shadow')""",
            ("2026-09-18", mid, eid, sector, yes, title, eid.split("::")[1], entry,
             (close - entry) * 100),
        )
        conn.execute(
            """INSERT INTO ev_outcomes
               (market_id, event_id, event_date, sector, yes_team, outcome,
                result_source, pinnacle_close_prob)
               VALUES (?,?,?,?,?,?,'espn',?)""",
            (mid, eid, eid.split("::")[1], sector, yes, outcome, close),
        )
    conn.commit()
    conn.close()

    arch = sqlite3.connect(str(arch_path))
    arch.execute(
        """CREATE TABLE archived_sharp_odds (
               id INTEGER PRIMARY KEY, fetched_at TEXT, event_id TEXT,
               outcome_a_label TEXT, outcome_b_label TEXT,
               true_prob_a REAL, true_prob_b REAL, true_prob_draw REAL,
               spread_line REAL, total_line REAL, event_date TEXT)"""
    )
    arch.execute(
        """INSERT INTO archived_sharp_odds
           (fetched_at, event_id, outcome_a_label, outcome_b_label, true_prob_a, true_prob_b,
            event_date)
           VALUES ('2026-09-19T18:00:00+00:00', ?, 'Utah', 'Utah State', 0.97, 0.03,
                   '2026-09-19T19:00:00+00:00')""",
        (UTAH_EID,),
    )
    arch.execute(
        """INSERT INTO archived_sharp_odds
           (fetched_at, event_id, outcome_a_label, outcome_b_label, true_prob_a, true_prob_b,
            event_date)
           VALUES ('2026-09-13T16:00:00+00:00', 'nfl::2026-09-13::chiefs_vs_bills',
                   'Kansas City Chiefs', 'Buffalo Bills', 0.55, 0.45,
                   '2026-09-13T17:00:00+00:00')"""
    )
    arch.commit()
    arch.close()
    return pred_path, arch_path


def _ro(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


class TestCandidateSelection:
    def test_nested_and_non_literal_rows_are_candidates(self):
        assert repair.teams_nested("utah", "utah state", "ncaaf")
        assert repair.teams_nested("paris", "psg", "soccer")  # psg → paris saint germain
        assert not repair.teams_nested("chiefs", "bills", "nfl")
        base = {"market_type": "moneyline", "sector": "nfl"}
        assert repair.is_outcome_candidate(
            dict(base, event_id="nfl::2026-09-13::chiefs_vs_bills", yes_team="kc"))
        assert not repair.is_outcome_candidate(
            dict(base, event_id="nfl::2026-09-13::chiefs_vs_bills", yes_team="chiefs"))
        assert not repair.is_outcome_candidate(
            dict(base, event_id=UTAH_EID, sector="ncaaf", yes_team="over", market_type="total"))


class TestPlanAndApply:
    def test_plans_exactly_the_misgraded_rows_and_is_idempotent(self, tmp_path, monkeypatch):
        pred_path, arch_path = _build_dbs(tmp_path, monkeypatch)
        scores = {("ncaaf", "2026-09-19"): [UTAH_SCORE]}

        ro = _ro(pred_path)
        cands = repair.load_outcome_candidates(ro)
        assert {c["market_id"] for c in cands} == {
            UTAH_MID, "kalshi:KXNCAAFGAME-26SEP19USUUTAH-UTAH",
        }
        outcome_fixes, skipped = repair.plan_outcome_fixes(cands, scores)
        assert [(f.market_id, f.stored, f.correct) for f in outcome_fixes] == [(UTAH_MID, 1, 0)]
        assert skipped["unchanged"] == 1  # the Utah YES row was graded right

        close_fixes = repair.plan_close_fixes(ro, _ro(arch_path), tolerance=1.0)
        ro.close()
        assert [(f.market_id, f.stored_close, f.correct_close) for f in close_fixes] == [
            (UTAH_MID, 0.97, 0.03),
        ]
        (pred_id, old_drift, new_drift), = close_fixes[0].drift_updates
        assert old_drift == pytest.approx(93.0)
        assert new_drift == pytest.approx(-1.0)

        conn = sqlite3.connect(str(pred_path))
        counts = repair.apply_fixes(conn, outcome_fixes, close_fixes)
        conn.close()
        assert counts == {"outcomes": 1, "closes": 1, "drifts": 1}

        ro = _ro(pred_path)
        row = ro.execute(
            "SELECT outcome, pinnacle_close_prob FROM ev_outcomes WHERE market_id = ?",
            (UTAH_MID,),
        ).fetchone()
        assert (row["outcome"], row["pinnacle_close_prob"]) == (0, 0.03)
        # Untouched rows keep their values.
        kc = ro.execute(
            "SELECT outcome, pinnacle_close_prob FROM ev_outcomes WHERE market_id LIKE '%-KC'"
        ).fetchone()
        assert (kc["outcome"], kc["pinnacle_close_prob"]) == (0, 0.55)

        # Idempotent: a second plan finds nothing.
        again, _ = repair.plan_outcome_fixes(repair.load_outcome_candidates(ro), scores)
        assert again == []
        assert repair.plan_close_fixes(ro, _ro(arch_path), tolerance=1.0) == []
        ro.close()

    def test_game_on_another_date_is_not_rewritten(self, tmp_path, monkeypatch):
        pred_path, _ = _build_dbs(tmp_path, monkeypatch)
        other_day = dict(UTAH_SCORE, game_date="2026-09-18")
        ro = _ro(pred_path)
        fixes, skipped = repair.plan_outcome_fixes(
            repair.load_outcome_candidates(ro), {("ncaaf", "2026-09-19"): [other_day]},
        )
        ro.close()
        assert fixes == []
        assert skipped["game_date_mismatch"] == 2

    def test_guarded_update_skips_a_row_changed_since_planning(self, tmp_path, monkeypatch):
        pred_path, _ = _build_dbs(tmp_path, monkeypatch)
        ro = _ro(pred_path)
        fixes, _ = repair.plan_outcome_fixes(
            repair.load_outcome_candidates(ro), {("ncaaf", "2026-09-19"): [UTAH_SCORE]},
        )
        ro.close()
        conn = sqlite3.connect(str(pred_path))
        conn.execute("UPDATE ev_outcomes SET outcome = 0 WHERE market_id = ?", (UTAH_MID,))
        conn.commit()
        assert repair.apply_fixes(conn, fixes, [])["outcomes"] == 0
        conn.close()

    def test_backup_is_a_full_copy(self, tmp_path, monkeypatch):
        pred_path, _ = _build_dbs(tmp_path, monkeypatch)
        backup = repair.backup_database(pred_path)
        assert backup.exists() and backup.name.startswith("predictions.db.bak-containment-")
        n = sqlite3.connect(str(backup)).execute("SELECT COUNT(*) FROM ev_outcomes").fetchone()[0]
        assert n == 3
