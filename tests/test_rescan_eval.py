"""Tests for the near-close re-scan replay harness (rescan_eval.py).

Covers the pure scoring/window helpers plus the end-to-end paired replay
against synthetic predictions.db + archive.db fixtures: window selection,
paired CLV math against the SAME close, the NO-side flip convention, the
fresh-anchor EV gate (moneyline, spread, and total branches), coverage
counters for capture holes, and per-group summarization.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

from evmax.agents.cleanup.rescan_eval import (
    base_event_key,
    evaluate,
    pick_close,
    pick_rescan_entry,
    side_aligned,
    summarize,
)
from evmax.archiver import _get_connection

BASE_EVENT = "wnba::2026-07-02::mercury_vs_storm"
TIP = "2026-07-03T02:00:00+00:00"
TIP_DT = datetime.fromisoformat(TIP)


def _t(minutes_before_tip: float) -> datetime:
    from datetime import timedelta

    return TIP_DT - timedelta(minutes=minutes_before_tip)


def _iso(minutes_before_tip: float) -> str:
    return _t(minutes_before_tip).isoformat()


# ---------------------------------------------------------------- pure helpers


class TestSideAligned:
    def test_yes_side_passthrough(self):
        assert side_aligned(0.42, is_no_side=False) == pytest.approx(0.42)

    def test_no_side_flips(self):
        assert side_aligned(0.42, is_no_side=True) == pytest.approx(0.58)


class TestBaseEventKey:
    def test_moneyline_id_passes_through(self):
        assert base_event_key(BASE_EVENT) == BASE_EVENT

    def test_spread_suffix_stripped(self):
        assert base_event_key(BASE_EVENT + "::spread") == BASE_EVENT

    def test_total_suffix_and_line_stripped(self):
        assert base_event_key(BASE_EVENT + "::total::171.0") == BASE_EVENT

    def test_advance_suffix_kept(self):
        eid = "worldcup::2026-07-06::mexico_vs_england::advance"
        assert base_event_key(eid) == eid


class TestPickRescanEntry:
    def test_picks_first_snapshot_inside_window(self):
        trail = [(_t(600), 0.40), (_t(70), 0.45), (_t(30), 0.47)]
        got = pick_rescan_entry(trail, TIP_DT, window_lo_min=75, window_hi_min=10)
        assert got == (_t(70), 0.45)

    def test_window_bounds_are_inclusive(self):
        trail = [(_t(75), 0.44)]
        assert pick_rescan_entry(trail, TIP_DT, 75, 10) == (_t(75), 0.44)
        assert pick_rescan_entry([(_t(10), 0.5)], TIP_DT, 75, 10) == (_t(10), 0.5)

    def test_none_when_all_snapshots_outside_window(self):
        trail = [(_t(600), 0.40), (_t(5), 0.55)]  # too early / too late
        assert pick_rescan_entry(trail, TIP_DT, 75, 10) is None

    def test_none_on_empty_trail(self):
        assert pick_rescan_entry([], TIP_DT) is None


class TestPickClose:
    def test_last_pre_tip_snapshot_wins(self):
        trail = [(_t(600), 0.40), (_t(30), 0.47), (_t(15), 0.50)]
        assert pick_close(trail, TIP_DT) == (_t(15), 0.50)

    def test_post_tip_snapshots_ignored(self):
        trail = [(_t(30), 0.47), (_t(-20), 0.99)]  # -20 = 20min AFTER tip
        assert pick_close(trail, TIP_DT) == (_t(30), 0.47)

    def test_close_must_be_strictly_after_entry(self):
        trail = [(_t(600), 0.40), (_t(30), 0.47)]
        assert pick_close(trail, TIP_DT, after=_t(30)) is None
        assert pick_close(trail, TIP_DT, after=_t(60)) == (_t(30), 0.47)


# ------------------------------------------------------------------- fixtures


@pytest.fixture
def pred_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE ev_predictions (
               id INTEGER PRIMARY KEY,
               scan_date TEXT, market_id TEXT, event_id TEXT, sector TEXT,
               market_type TEXT, yes_team TEXT, line REAL, event_title TEXT,
               mode TEXT, kalshi_yes_price REAL, placed INTEGER DEFAULT 0,
               placed_price REAL, voided INTEGER DEFAULT 0
           )"""
    )
    yield conn
    conn.close()


@pytest.fixture
def arch_conn(tmp_path, monkeypatch):
    db_path = tmp_path / "archive.db"
    monkeypatch.setattr("evmax.archiver.DB_PATH", db_path)
    with _get_connection():  # creates schema
        pass
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


def insert_pred(conn, market_id, *, market_type="moneyline", yes_team="storm",
                line=None, price=0.40, event_id=BASE_EVENT, sector="wnba",
                scan_date="2026-07-02", placed=0, placed_price=None,
                mode="live", voided=0):
    conn.execute(
        """INSERT INTO ev_predictions
           (scan_date, market_id, event_id, sector, market_type, yes_team,
            line, event_title, mode, kalshi_yes_price, placed, placed_price,
            voided)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (scan_date, market_id, event_id, sector, market_type, yes_team, line,
         "Phoenix Mercury vs Seattle Storm", mode, price, placed, placed_price,
         voided),
    )


def insert_snap(conn, ticker, minutes_before_tip, yes_price, sector="wnba"):
    conn.execute(
        """INSERT INTO archived_kalshi_markets
           (session_id, fetched_at, sector, ticker, market_type, yes_price,
            no_price)
           VALUES (?,?,?,?,?,?,?)""",
        (f"s{minutes_before_tip}", _iso(minutes_before_tip), sector, ticker,
         "moneyline", yes_price, 1 - yes_price),
    )


def insert_ml_anchor(conn, minutes_before_tip, prob_a, event_id=BASE_EVENT,
                     a="Seattle Storm", b="Phoenix Mercury", prob_draw=None):
    conn.execute(
        """INSERT INTO archived_sharp_odds
           (session_id, fetched_at, sector, event_id, book, outcome_a_label,
            outcome_b_label, outcome_a_decimal, outcome_b_decimal,
            true_prob_a, true_prob_b, true_prob_draw, margin, event_date)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (f"a{minutes_before_tip}", _iso(minutes_before_tip), "wnba", event_id,
         "pinnacle", a, b, 2.0, 2.0, prob_a, 1 - prob_a - (prob_draw or 0.0),
         prob_draw, 0.03, TIP),
    )


def insert_spread_anchor(conn, minutes_before_tip, prob_a=0.5, spread_line=-4.5):
    conn.execute(
        """INSERT INTO archived_sharp_odds
           (session_id, fetched_at, sector, event_id, book, outcome_a_label,
            outcome_b_label, outcome_a_decimal, outcome_b_decimal,
            true_prob_a, true_prob_b, margin, spread_line, event_date)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (f"a{minutes_before_tip}", _iso(minutes_before_tip), "wnba",
         BASE_EVENT + "::spread", "pinnacle", "Seattle Storm",
         "Phoenix Mercury", 2.0, 2.0, prob_a, 1 - prob_a, 0.03, spread_line,
         TIP),
    )


def insert_total_anchor(conn, minutes_before_tip, total_line, p_over):
    conn.execute(
        """INSERT INTO archived_sharp_odds
           (session_id, fetched_at, sector, event_id, book, outcome_a_label,
            outcome_b_label, outcome_a_decimal, outcome_b_decimal,
            true_prob_a, true_prob_b, margin, total_line, true_prob_over,
            true_prob_under, event_date)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (f"a{minutes_before_tip}", _iso(minutes_before_tip), "wnba",
         f"{BASE_EVENT}::total::{total_line}", "pinnacle", "Over", "Under",
         2.0, 2.0, p_over, 1 - p_over, 0.03, total_line, p_over, 1 - p_over,
         TIP),
    )


# --------------------------------------------------------------- end-to-end


class TestEvaluateMoneyline:
    def test_paired_yes_side_scoring_and_gate(self, pred_conn, arch_conn):
        """Happy path: scan @0.40, rescan @0.45 (fair 0.50 -> enter), close 0.50."""
        insert_pred(pred_conn, "kalshi:T1", price=0.40)
        insert_snap(arch_conn, "T1", 600, 0.40)   # scan-time snapshot
        insert_snap(arch_conn, "T1", 60, 0.45)    # in-window rescan
        insert_snap(arch_conn, "T1", 15, 0.50)    # close
        insert_ml_anchor(arch_conn, 600, 0.50)
        insert_ml_anchor(arch_conn, 70, 0.50)     # fresh anchor at rescan
        pred_conn.commit(); arch_conn.commit()

        stats = evaluate(pred_conn, arch_conn)
        assert stats.rows == 1
        assert len(stats.pairs) == 1
        p = stats.pairs[0]
        assert p.actual_clv_pp == pytest.approx(10.0)
        assert p.rescan_price == pytest.approx(0.45)
        assert p.rescan_clv_pp == pytest.approx(5.0)
        assert p.close_price == pytest.approx(0.50)
        assert p.rescan_lead_min == pytest.approx(60.0)
        assert p.anchor_fair == pytest.approx(0.50)
        assert p.rescan_edge_pp == pytest.approx(5.0)
        assert p.rescan_enter is True
        assert p.event_title == "Phoenix Mercury vs Seattle Storm"
        assert p.outcome_label == "storm ML"

    def test_edge_below_gate_does_not_enter(self, pred_conn, arch_conn):
        insert_pred(pred_conn, "kalshi:T1", price=0.40)
        insert_snap(arch_conn, "T1", 600, 0.40)
        insert_snap(arch_conn, "T1", 60, 0.49)    # fair 0.50 -> edge 1pp < 2pp
        insert_snap(arch_conn, "T1", 15, 0.50)
        insert_ml_anchor(arch_conn, 70, 0.50)
        pred_conn.commit(); arch_conn.commit()

        stats = evaluate(pred_conn, arch_conn)
        p = stats.pairs[0]
        assert p.rescan_edge_pp == pytest.approx(1.0)
        assert p.rescan_enter is False

    def test_no_side_flip_convention(self, pred_conn, arch_conn):
        """':no' rows: stored entry is already NO-side; snapshots flip 1-yes."""
        insert_pred(pred_conn, "kalshi:T1:no", price=0.55)  # NO ask at scan
        insert_snap(arch_conn, "T1", 600, 0.45)
        insert_snap(arch_conn, "T1", 60, 0.55)    # NO rescan = 0.45
        insert_snap(arch_conn, "T1", 15, 0.60)    # NO close = 0.40
        insert_ml_anchor(arch_conn, 70, 0.70)     # fair NO = 0.30
        pred_conn.commit(); arch_conn.commit()

        stats = evaluate(pred_conn, arch_conn)
        p = stats.pairs[0]
        assert p.is_no_side is True
        assert p.actual_entry_price == pytest.approx(0.55)
        assert p.actual_clv_pp == pytest.approx(-15.0)   # 0.40 - 0.55
        assert p.rescan_price == pytest.approx(0.45)     # 1 - 0.55
        assert p.rescan_clv_pp == pytest.approx(-5.0)    # 0.40 - 0.45
        assert p.anchor_fair == pytest.approx(0.30)
        assert p.rescan_edge_pp == pytest.approx(-15.0)
        assert p.rescan_enter is False
        assert p.outcome_label == "NO storm ML"

    def test_placed_price_is_the_actual_entry(self, pred_conn, arch_conn):
        insert_pred(pred_conn, "kalshi:T1", price=0.40, placed=1,
                    placed_price=0.43)
        insert_snap(arch_conn, "T1", 60, 0.45)
        insert_snap(arch_conn, "T1", 15, 0.50)
        insert_ml_anchor(arch_conn, 70, 0.50)
        pred_conn.commit(); arch_conn.commit()

        p = evaluate(pred_conn, arch_conn).pairs[0]
        assert p.actual_entry_price == pytest.approx(0.43)
        assert p.actual_clv_pp == pytest.approx(7.0)

    def test_stale_anchor_keeps_pair_ungated(self, pred_conn, arch_conn):
        """Anchor 10h old at rescan time -> EV uncomputable, pair stays."""
        insert_pred(pred_conn, "kalshi:T1", price=0.40)
        insert_snap(arch_conn, "T1", 60, 0.45)
        insert_snap(arch_conn, "T1", 15, 0.50)
        insert_ml_anchor(arch_conn, 660, 0.50)   # way beyond 120min staleness
        pred_conn.commit(); arch_conn.commit()

        stats = evaluate(pred_conn, arch_conn)
        assert stats.no_fresh_anchor == 1
        p = stats.pairs[0]
        assert p.rescan_edge_pp is None
        assert p.rescan_enter is False
        assert p.rescan_clv_pp == pytest.approx(5.0)  # still scored ungated

    def test_no_window_snapshot_is_counted(self, pred_conn, arch_conn):
        insert_pred(pred_conn, "kalshi:T1", price=0.40)
        insert_snap(arch_conn, "T1", 600, 0.40)   # outside the window
        insert_snap(arch_conn, "T1", 5, 0.50)     # too close to tip
        insert_ml_anchor(arch_conn, 70, 0.50)
        pred_conn.commit(); arch_conn.commit()

        stats = evaluate(pred_conn, arch_conn)
        assert stats.no_window_snapshot == 1
        assert stats.pairs == []

    def test_no_close_after_rescan_is_counted(self, pred_conn, arch_conn):
        insert_pred(pred_conn, "kalshi:T1", price=0.40)
        insert_snap(arch_conn, "T1", 60, 0.45)    # in window, but LAST snapshot
        insert_ml_anchor(arch_conn, 70, 0.50)
        pred_conn.commit(); arch_conn.commit()

        stats = evaluate(pred_conn, arch_conn)
        assert stats.no_close_after == 1
        assert stats.pairs == []

    def test_no_tip_is_counted(self, pred_conn, arch_conn):
        insert_pred(pred_conn, "kalshi:T1", price=0.40)
        insert_snap(arch_conn, "T1", 60, 0.45)   # no sharp rows at all
        pred_conn.commit(); arch_conn.commit()

        stats = evaluate(pred_conn, arch_conn)
        assert stats.no_tip == 1

    def test_multi_day_rows_deduped_to_first_scan(self, pred_conn, arch_conn):
        insert_pred(pred_conn, "kalshi:T1", price=0.40, scan_date="2026-07-01")
        insert_pred(pred_conn, "kalshi:T1", price=0.44, scan_date="2026-07-02")
        insert_snap(arch_conn, "T1", 60, 0.45)
        insert_snap(arch_conn, "T1", 15, 0.50)
        insert_ml_anchor(arch_conn, 70, 0.50)
        pred_conn.commit(); arch_conn.commit()

        stats = evaluate(pred_conn, arch_conn)
        assert stats.rows == 1
        assert stats.pairs[0].actual_entry_price == pytest.approx(0.40)

    def test_voided_rows_skipped(self, pred_conn, arch_conn):
        insert_pred(pred_conn, "kalshi:T1", price=0.40, voided=1)
        pred_conn.commit()
        stats = evaluate(pred_conn, arch_conn)
        assert stats.rows == 0


class TestEvaluateSpreadAndTotal:
    def test_spread_gate_prices_kalshi_line_off_primary_anchor(
        self, pred_conn, arch_conn
    ):
        # storm -4.5 @ 0.50 -> P(storm covers -6.5) ~ 0.4364 (see listings test)
        insert_pred(pred_conn, "kalshi:S1", market_type="spread",
                    yes_team="storm", line=-6.5, price=0.30,
                    event_id=BASE_EVENT + "::spread")
        insert_snap(arch_conn, "S1", 60, 0.40)
        insert_snap(arch_conn, "S1", 15, 0.42)
        insert_spread_anchor(arch_conn, 70, prob_a=0.5, spread_line=-4.5)
        pred_conn.commit(); arch_conn.commit()

        p = evaluate(pred_conn, arch_conn).pairs[0]
        assert p.anchor_fair == pytest.approx(0.4364, abs=0.001)
        assert p.rescan_edge_pp == pytest.approx(3.64, abs=0.1)
        assert p.rescan_enter is True
        assert p.outcome_label == "storm -6.5"

    def test_total_gate_requires_matching_ladder_line(self, pred_conn, arch_conn):
        insert_pred(pred_conn, "kalshi:O1", market_type="total",
                    yes_team="over", line=164.5, price=0.50,
                    event_id=BASE_EVENT + "::total::164.5")
        insert_snap(arch_conn, "O1", 60, 0.52)
        insert_snap(arch_conn, "O1", 15, 0.55)
        insert_total_anchor(arch_conn, 70, total_line=164.5, p_over=0.60)
        pred_conn.commit(); arch_conn.commit()

        p = evaluate(pred_conn, arch_conn).pairs[0]
        assert p.anchor_fair == pytest.approx(0.60)
        assert p.rescan_edge_pp == pytest.approx(8.0, abs=0.01)
        assert p.rescan_enter is True
        assert p.outcome_label == "Over 164.5"

    def test_total_far_ladder_line_is_no_anchor(self, pred_conn, arch_conn):
        insert_pred(pred_conn, "kalshi:O1", market_type="total",
                    yes_team="over", line=170.5, price=0.50,
                    event_id=BASE_EVENT + "::total::170.5")
        insert_snap(arch_conn, "O1", 60, 0.52)
        insert_snap(arch_conn, "O1", 15, 0.55)
        insert_total_anchor(arch_conn, 70, total_line=164.5, p_over=0.60)
        pred_conn.commit(); arch_conn.commit()

        stats = evaluate(pred_conn, arch_conn)
        assert stats.no_fresh_anchor == 1
        assert stats.pairs[0].rescan_enter is False


class TestSummarize:
    def test_groups_and_all_row(self, pred_conn, arch_conn):
        insert_pred(pred_conn, "kalshi:T1", price=0.40)
        insert_snap(arch_conn, "T1", 60, 0.45)
        insert_snap(arch_conn, "T1", 15, 0.50)
        insert_ml_anchor(arch_conn, 70, 0.50)
        pred_conn.commit(); arch_conn.commit()

        groups = summarize(evaluate(pred_conn, arch_conn))
        assert [(g.sector, g.market_type) for g in groups] == [
            ("wnba", "moneyline"), ("ALL", "all"),
        ]
        g = groups[0]
        assert g.n == 1
        assert g.actual_mean_clv == pytest.approx(10.0)
        assert g.rescan_mean_clv == pytest.approx(5.0)
        assert g.delta_mean == pytest.approx(-5.0)
        assert g.n_gated == 1
        assert g.gated_mean_clv == pytest.approx(5.0)
        assert g.gated_delta_mean == pytest.approx(-5.0)

    def test_empty_stats_summarize_to_empty_list(self, pred_conn, arch_conn):
        assert summarize(evaluate(pred_conn, arch_conn)) == []
