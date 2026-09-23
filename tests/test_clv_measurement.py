"""CLV measurement correctness (2026-09-22 audit).

Every promotion gate reads ``kalshi_clv_pct`` and ``pinnacle_close_prob``, so a
biased measurement biases every promote/reject call. Locked in here:

1. NO-side rows are scored ask-to-ask (archived NO ask), not NO-ask entry vs
   ``1 − YES ask`` close (the NO *bid*, which charged every NO row the spread).
2. CLV is measured FORWARD from each row's own entry (placed_at, else the
   first-log scan) — an at/after-tip entry never scores against an earlier close.
3. Recompute mode re-derives stored values; dry-run writes nothing.
4. A devigged close of exactly 0/1 is never a price: the moneyline aligner
   rejects it and the orphan backfill heals stored zeros; alt-rung
   ``::total::<line>`` ids are out of its moneyline scope.
5. The CLV lenses drop cancelled voids and at-tip entries but KEEP
   ``stale_reverted`` voids, and the gate counts independent GAMES.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from evmax.archiver import DataArchiver
from evmax.models.odds import SharpBook, SharpOdds

from tests.test_cli_shadow import _make_clv_db
from tests.test_resolver import TestBackfillClvNoSide

TIP = datetime(2026, 5, 25, 23, 0, tzinfo=timezone.utc)
EVENT_TOTAL = "baseball::2026-05-25::yankees_vs_redsox::total::8.5"
TICKER = "KXMLBGAME-T"


@pytest.fixture
def archive(tmp_path, monkeypatch):
    monkeypatch.setattr("evmax.archiver.DB_PATH", tmp_path / "archive.db")
    return DataArchiver()


def _tipoff_anchor(archiver: DataArchiver, event_id: str) -> None:
    archiver.open_session("so", ["baseball"], "test")
    archiver.archive_sharp_odds("so", "baseball", [SharpOdds(
        event_id=event_id, book=SharpBook.pinnacle, sector="baseball",
        outcome_a_label="yankees", outcome_b_label="redsox",
        outcome_a_decimal=1.9, outcome_b_decimal=1.9,
        true_prob_a=0.5, true_prob_b=0.5, margin=0.04,
        event_date=TIP, fetched_at=TIP - timedelta(hours=4),
    )])


def _snapshot(archiver, yes, no, fetched_at, event_id=EVENT_TOTAL, session="k1"):
    snap = {"ticker": TICKER, "yes_price": yes, "event_id": event_id,
            "market_type": "total"}
    if no is not None:
        snap["no_price"] = no
    archiver.archive_kalshi_snapshot(session, "baseball", [snap], fetched_at=fetched_at)


# --- 1. side-aware close ----------------------------------------------------

def test_no_side_close_is_the_no_ask(archive):
    _tipoff_anchor(archive, EVENT_TOTAL)
    # YES ask 0.55 / NO ask 0.49: a 4c spread. The NO bid would be 0.45.
    _snapshot(archive, 0.55, 0.49, TIP - timedelta(hours=1))
    assert archive.get_kalshi_close_price(TICKER, EVENT_TOTAL) == pytest.approx(0.55)
    assert archive.get_kalshi_close_price(TICKER, EVENT_TOTAL, side="no") == pytest.approx(0.49)


def test_no_side_close_skips_a_synthesized_no_ask(archive):
    """A YES-only capture stores no_price = 1 − yes (the NO BID). A NO row must
    never close there — no real NO ask means no close, not a bid-valued one."""
    _tipoff_anchor(archive, EVENT_TOTAL)
    _snapshot(archive, 0.55, None, TIP - timedelta(hours=1))  # archiver derives 0.45
    assert archive.get_kalshi_close_price(TICKER, EVENT_TOTAL) == pytest.approx(0.55)
    assert archive.get_kalshi_close_price(TICKER, EVENT_TOTAL, side="no") is None


def test_no_side_close_takes_the_latest_REAL_no_ask(archive):
    _tipoff_anchor(archive, EVENT_TOTAL)
    _snapshot(archive, 0.52, 0.51, TIP - timedelta(hours=3), session="real")  # two-sided
    _snapshot(archive, 0.55, None, TIP - timedelta(hours=1), session="synth")  # YES-only
    assert archive.get_kalshi_close_price(TICKER, EVENT_TOTAL, side="no") == pytest.approx(0.51)
    assert archive.get_kalshi_close_price(TICKER, EVENT_TOTAL, side="yes") == pytest.approx(0.55)


def test_unplaced_kalshi_rows_never_close_inside_the_near_tip_window(archive):
    """Logged at T-20 with the only later print at T-10: an unplaced Kalshi row
    gets no close (tennis in-play risk); a placed fill or a PolyUS row may use it."""
    _tipoff_anchor(archive, EVENT_TOTAL)
    _snapshot(archive, 0.58, 0.45, TIP - timedelta(minutes=10))
    entry = TIP - timedelta(minutes=20)
    assert archive.get_kalshi_close_price(TICKER, EVENT_TOTAL, not_before=entry,
                                          near_tip_ok=False) is None
    assert archive.get_kalshi_close_price(TICKER, EVENT_TOTAL, not_before=entry) == pytest.approx(0.58)


def test_near_tip_fallback_needs_near_tip_ok(archive):
    """With no snapshot at/before T-30 at all, only near-tip-eligible rows may
    fall back to a (T-30, T) print."""
    _tipoff_anchor(archive, EVENT_TOTAL)
    _snapshot(archive, 0.58, 0.45, TIP - timedelta(minutes=10))
    assert archive.get_kalshi_close_price(TICKER, EVENT_TOTAL, near_tip_ok=False) is None
    assert archive.get_kalshi_close_price(TICKER, EVENT_TOTAL) == pytest.approx(0.58)


def test_clv_near_tip_ok_rule():
    from evmax.agents.cleanup.resolver import clv_near_tip_ok

    assert clv_near_tip_ok(1, "KXNBAGAME-X") is True
    assert clv_near_tip_ok(0, "polymarket_us:aec-wnba-x") is True
    assert clv_near_tip_ok(0, "KXATPMATCH-X") is False
    assert clv_near_tip_ok(0, None) is False


def test_close_side_is_validated(archive):
    with pytest.raises(ValueError):
        archive.get_kalshi_close_price(TICKER, EVENT_TOTAL, side="under")


# --- 2/3. backfill_clv: side, forward anchor, recompute ---------------------

class _Helpers(TestBackfillClvNoSide):
    __test__ = False  # borrow the seed helpers without re-collecting the tests


def _backfill(conn, **kw):
    from evmax.agents.cleanup import resolver

    with patch.object(resolver, "get_connection", return_value=conn):
        return resolver.backfill_clv(**kw)


def _clv(conn, mid):
    return conn.execute(
        "SELECT kalshi_clv_pct FROM ev_predictions WHERE market_id = ?", (mid,)
    ).fetchone()["kalshi_clv_pct"]


def test_backfill_scores_no_row_against_no_ask(tmp_path, monkeypatch, archive):
    h = _Helpers()
    _tipoff_anchor(archive, EVENT_TOTAL)
    _snapshot(archive, 0.30, 0.66, TIP - timedelta(hours=1))
    conn = h._make_predictions_db(tmp_path, monkeypatch)
    h._seed(conn, f"kalshi:{TICKER}:no", "under", "total", 0.60, 8.5, EVENT_TOTAL)
    _backfill(conn)
    # ask-to-ask: 0.66 − 0.60 = +6.0pp (the old 1 − yes flip read +10.0)
    assert _clv(conn, f"kalshi:{TICKER}:no") == pytest.approx(6.0)


def test_backfill_never_scores_a_close_that_precedes_the_entry(tmp_path, monkeypatch, archive):
    h = _Helpers()
    _tipoff_anchor(archive, EVENT_TOTAL)
    _snapshot(archive, 0.40, 0.62, TIP - timedelta(hours=2))  # only snapshot: T-2h
    conn = h._make_predictions_db(tmp_path, monkeypatch)
    # logged 1h before tip — AFTER the only snapshot → no forward close exists
    h._seed(conn, f"kalshi:{TICKER}", "over", "total", 0.45, 8.5, EVENT_TOTAL,
            logged_at=(TIP - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"))
    _backfill(conn)
    assert _clv(conn, f"kalshi:{TICKER}") is None


def test_recompute_dry_run_reports_without_writing(tmp_path, monkeypatch, archive):
    h = _Helpers()
    _tipoff_anchor(archive, EVENT_TOTAL)
    _snapshot(archive, 0.30, 0.66, TIP - timedelta(hours=1))
    conn = h._make_predictions_db(tmp_path, monkeypatch)
    mid = f"kalshi:{TICKER}:no"
    h._seed(conn, mid, "under", "total", 0.60, 8.5, EVENT_TOTAL)
    conn.execute("UPDATE ev_predictions SET kalshi_clv_pct = 10.0, pinnacle_drift_pct = 0.0")
    conn.commit()

    res = _backfill(conn, recompute_kalshi_clv=True, dry_run=True)
    assert res["dry_run"] is True
    assert res["changed"] == 1
    assert res["mean_delta_pp"] == pytest.approx(-4.0)   # 10.0 → 6.0
    assert _clv(conn, mid) == pytest.approx(10.0)       # untouched

    _backfill(conn, recompute_kalshi_clv=True)
    assert _clv(conn, mid) == pytest.approx(6.0)


def test_plain_backfill_leaves_measured_rows_alone(tmp_path, monkeypatch, archive):
    h = _Helpers()
    _tipoff_anchor(archive, EVENT_TOTAL)
    _snapshot(archive, 0.30, 0.66, TIP - timedelta(hours=1))
    conn = h._make_predictions_db(tmp_path, monkeypatch)
    mid = f"kalshi:{TICKER}:no"
    h._seed(conn, mid, "under", "total", 0.60, 8.5, EVENT_TOTAL)
    conn.execute("UPDATE ev_predictions SET kalshi_clv_pct = 10.0")
    conn.commit()
    _backfill(conn)
    assert _clv(conn, mid) == pytest.approx(10.0)


def test_clv_not_before_prefers_fill_then_first_log():
    from evmax.agents.cleanup.resolver import clv_not_before

    assert clv_not_before(1, "2026-05-25T22:00:00", "2026-05-25 12:00:00") == "2026-05-25T22:00:00"
    assert clv_not_before(0, None, "2026-05-25 12:00:00") == "2026-05-25 12:00:00"
    assert clv_not_before(1, None, "2026-05-25 12:00:00") == "2026-05-25 12:00:00"
    assert clv_not_before(0, None, None) is None


# --- 4. zero closes ---------------------------------------------------------

def _total_record(archiver, event_id, line, p_over):
    archiver.open_session("st", ["nfl"], "test")
    archiver.archive_sharp_odds("st", "nfl", [SharpOdds(
        event_id=event_id, book=SharpBook.pinnacle, sector="nfl",
        outcome_a_label="over", outcome_b_label="under",
        outcome_a_decimal=1.9, outcome_b_decimal=1.9,
        true_prob_a=0.0, true_prob_b=0.0, margin=0.04,
        total_line=line, true_prob_over=p_over, true_prob_under=1 - p_over,
        event_date=TIP, fetched_at=TIP - timedelta(hours=3),
    )])


def test_moneyline_aligner_rejects_totals_record(archive):
    ev = "nfl::2026-09-09::seahawks_vs_patriots::total::47.0"
    _total_record(archive, ev, 47.0, 0.39)
    assert archive.get_closing_line_aligned(ev, "over") is None


def test_orphan_backfill_skips_alt_rung_totals_and_heals_zero(tmp_path, monkeypatch, archive):
    from evmax.agents.cleanup.db import get_connection
    from evmax.agents.cleanup.resolver import backfill_outcome_closes

    monkeypatch.setattr("evmax.agents.cleanup.db.DB_PATH", tmp_path / "predictions.db")
    ev = "nfl::2026-09-09::seahawks_vs_patriots::total::47.0"
    _total_record(archive, ev, 47.0, 0.39)
    conn = get_connection()
    conn.execute(
        """INSERT INTO ev_outcomes (market_id, event_id, event_date, sector, yes_team,
               outcome, sharp_true_prob, blended_true_prob, pinnacle_close_prob)
           VALUES ('kalshi:TOT', ?, '2026-09-09', 'nfl', 'over', 1, 0.4, 0.4, 0.0)""",
        (ev,),
    )
    conn.commit()
    conn.close()

    dry = backfill_outcome_closes(sector="nfl", dry_run=True)
    assert dry["healed_invalid_closes"] == 1
    res = backfill_outcome_closes(sector="nfl")
    assert res["healed_invalid_closes"] == 1
    assert res["candidates"] == 0   # ::total::<line> is out of moneyline scope
    conn = get_connection()
    close = conn.execute("SELECT pinnacle_close_prob FROM ev_outcomes").fetchone()[0]
    assert close is None             # 0.0 cleared for the line-aware path
    conn.close()


# --- 5. CLV lenses: voids, at-tip entries, game counting --------------------

def _add_cols(db_path, updates):
    conn = sqlite3.connect(str(db_path))
    for col, typ in (("voided", "INTEGER DEFAULT 0"), ("void_reason", "TEXT"),
                     ("minutes_to_tipoff", "INTEGER"), ("logged_at", "TEXT")):
        try:
            conn.execute(f"ALTER TABLE ev_predictions ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass
    for mid, col, val in updates:
        conn.execute(f"UPDATE ev_predictions SET {col} = ? WHERE market_id = ?", (val, mid))
    conn.commit()
    conn.close()


def test_clv_lens_drops_cancelled_and_at_tip_rows_but_keeps_stale_reverted(tmp_path, monkeypatch):
    from evmax.cli.commands.shadow import clv_stats

    rows = [(f"ok{i}", "2026-06-10", "moneyline", "live", 1.0, 1) for i in range(3)]
    rows += [("cancel", "2026-06-10", "moneyline", "live", 40.0, 1),
             ("attip", "2026-06-10", "moneyline", "live", 30.0, 1),
             ("pruned", "2026-06-10", "moneyline", "live", -5.0, 0)]
    db = _make_clv_db(tmp_path, rows)
    _add_cols(db, [("cancel", "voided", 1),
                   ("attip", "minutes_to_tipoff", 0),
                   ("pruned", "voided", 1), ("pruned", "void_reason", "stale_reverted")])
    monkeypatch.setattr("evmax.agents.cleanup.db.DB_PATH", db)
    s = clv_stats("wnba", market_type="moneyline")
    assert s["n"] == 4                                   # 3 ok + stale_reverted
    assert s["mean_clv_pp"] == pytest.approx((3 * 1.0 - 5.0) / 4)


def test_game_key_strips_market_and_rung():
    from evmax.cli.commands.shadow import game_key

    base = "nfl::2026-09-13::bills_vs_jets"
    for suffix in ("", "::spread", "::spread::-3.5", "::total::47.0", "::advance"):
        assert game_key(base + suffix) == base


def test_gate_counts_games_not_rungs(tmp_path, monkeypatch):
    from evmax.cli.commands.shadow import clv_stats

    # 40 all-positive rungs spread over only 2 games: must NOT clear n>=30.
    db = _make_clv_db(tmp_path, [(f"r{i}", "2026-06-10", "spread", "live", 2.0, 1)
                                 for i in range(40)])
    conn = sqlite3.connect(str(db))
    conn.execute(
        "UPDATE ev_predictions SET event_id = "
        "'wnba::2026-06-10::g' || (id % 2) || '::spread::' || id"
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr("evmax.agents.cleanup.db.DB_PATH", db)
    s = clv_stats("wnba", market_type="spread")
    assert s["n"] == 40 and s["games"] == 2
    assert s["frac_positive"] == 1.0
    assert s["clears"] is False


# --- watch-closes quote capture ---------------------------------------------

def test_quotes_batch_rest_fallback_keeps_yes_and_leaves_no_unknown(monkeypatch):
    import asyncio

    from evmax.clients import kalshi as kalshi_mod

    class _S:
        kalshi_ws_enabled = False

    monkeypatch.setattr(kalshi_mod, "get_settings", lambda: _S())
    client = kalshi_mod.KalshiClient.__new__(kalshi_mod.KalshiClient)

    async def _ask(t):
        return {"A": 0.42, "B": None}[t]

    client.get_market_ask = _ask
    out = asyncio.run(client.get_market_quotes_batch(["A", "B"]))
    assert out == {"A": (0.42, None), "B": (None, None)}


def test_board_verdict_too_few_games_is_collecting_not_failing():
    from evmax.agents.cleanup.promotion_board import _verdict

    clv = {"n": 40, "games": 2, "mean_clv_pp": 1.8, "frac_positive": 0.78, "clears": False}
    assert _verdict("total", None, 40, clv, "shadow", 30) == "COLLECTING 2/30g"
    failing = {"n": 60, "games": 35, "mean_clv_pp": -0.4, "frac_positive": 0.40, "clears": False}
    assert _verdict("total", None, 60, failing, "shadow", 30) == "FAILING-CLV"


def test_recompute_writes_a_backup_first(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from evmax.agents.cleanup import db as db_module
    from evmax.agents.cleanup import resolver
    from evmax.cli.commands.cleanup import app

    db = tmp_path / "predictions.db"
    sqlite3.connect(str(db)).close()
    monkeypatch.setattr(db_module, "DB_PATH", db)
    monkeypatch.setattr(resolver, "backfill_clv", lambda **kw: {
        "updated": 0, "skipped": 0, "changed": 0, "mean_delta_pp": 0.0, "dry_run": kw["dry_run"]})

    assert CliRunner().invoke(app, ["backfill-clv", "--recompute", "--dry-run"]).exit_code == 0
    assert not list(tmp_path.glob("predictions_backup_before_clv_recompute_*.db"))
    assert CliRunner().invoke(app, ["backfill-clv", "--recompute"]).exit_code == 0
    assert len(list(tmp_path.glob("predictions_backup_before_clv_recompute_*.db"))) == 1


def test_ws_fetch_asks_tolerates_a_ticker_without_a_snapshot(monkeypatch):
    """fetch_quotes marks a snapshot-less ticker None; fetch_asks used to index it
    (TypeError) and take down the live pick / prune-stale path."""
    import asyncio

    from evmax.clients import kalshi as kalshi_mod

    ws_cls = next(v for k, v in vars(kalshi_mod).items()
                  if isinstance(v, type) and hasattr(v, "fetch_quotes") and hasattr(v, "fetch_asks"))
    ws = ws_cls.__new__(ws_cls)

    async def _quotes(self, tickers):
        return {"A": (0.41, 0.61), "B": None}

    monkeypatch.setattr(ws_cls, "fetch_quotes", _quotes)
    assert asyncio.run(ws.fetch_asks(["A", "B"])) == {"A": 0.41, "B": None}
