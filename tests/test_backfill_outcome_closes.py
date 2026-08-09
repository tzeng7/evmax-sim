"""Tests for resolver.backfill_outcome_closes.

This backfill reconstructs ``ev_outcomes.pinnacle_close_prob`` directly from
archive.db for resolved MONEYLINE rows, reaching orphaned outcome rows that
``backfill_clv`` (driven by an ev_predictions join) cannot. The scope guard is
the important behaviour to lock in: spread, total, and player-prop rows must be
excluded, since only moneyline games can be aligned to a moneyline close.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from evmax.archiver import DataArchiver
from evmax.models.odds import SharpBook, SharpOdds
from evmax.agents.cleanup.resolver import backfill_outcome_closes


@pytest.fixture
def temp_dbs(tmp_path, monkeypatch):
    """Point both the archive and predictions DBs at fresh temp files."""
    monkeypatch.setattr("evmax.archiver.DB_PATH", tmp_path / "archive.db")
    monkeypatch.setattr("evmax.agents.cleanup.db.DB_PATH", tmp_path / "predictions.db")
    return tmp_path


def _ml_snapshot(archiver, session, event_id, event_date, prob_a,
                 a_label="Atlanta Hawks", b_label="Orlando Magic"):
    archiver.open_session(session, ["nba"], "test")
    archiver.archive_sharp_odds(session, "nba", [SharpOdds(
        event_id=event_id,
        book=SharpBook.pinnacle,
        sector="nba",
        outcome_a_label=a_label,
        outcome_b_label=b_label,
        outcome_a_decimal=1.67,
        outcome_b_decimal=2.5,
        true_prob_a=prob_a,
        true_prob_b=1.0 - prob_a,
        margin=0.0,
        event_date=event_date,
        fetched_at=event_date - timedelta(hours=3),
    )])


def _seed_outcome(conn, market_id, event_id, yes_team):
    conn.execute(
        """INSERT INTO ev_outcomes
           (market_id, event_id, event_date, sector, yes_team, outcome,
            sharp_true_prob, blended_true_prob, pinnacle_close_prob)
           VALUES (?,?,?,?,?,?,?,?,NULL)""",
        (market_id, event_id, "2026-04-01", "nba", yes_team, 1, 0.5, 0.5),
    )


def test_fills_moneyline_only_and_aligns_yes_side(temp_dbs):
    from evmax.agents.cleanup.db import get_connection

    event_id = "nba::2026-04-01::magic_vs_hawks"
    event_date = datetime(2026, 4, 1, 23, 0, tzinfo=timezone.utc)
    _ml_snapshot(DataArchiver(), "s1", event_id, event_date, prob_a=0.60)

    conn = get_connection()
    _seed_outcome(conn, "kalshi:ML", event_id, "hawks")              # moneyline -> fill
    _seed_outcome(conn, "kalshi:SP", event_id + "::spread", "hawks")  # spread -> skip
    _seed_outcome(conn, "kalshi:TO", event_id + "::total", "over")    # total  -> skip
    _seed_outcome(conn, "kalshi:PR",
                  "nba::2026-04-01::prop::maxey::points::25.5", "maxey")  # prop -> skip
    conn.commit()
    conn.close()

    res = backfill_outcome_closes(sector="nba")
    assert res["candidates"] == 1          # only the moneyline row qualifies
    assert res["filled"] == 1
    assert res["no_archive_close"] == 0

    conn = get_connection()
    closes = dict(conn.execute(
        "SELECT market_id, pinnacle_close_prob FROM ev_outcomes"
    ).fetchall())
    conn.close()
    # yes_team "hawks" substring-matches outcome_a "Atlanta Hawks" -> true_prob_a
    assert closes["kalshi:ML"] == pytest.approx(0.60)
    assert closes["kalshi:SP"] is None
    assert closes["kalshi:TO"] is None
    assert closes["kalshi:PR"] is None


def test_yes_side_b_alignment(temp_dbs):
    from evmax.agents.cleanup.db import get_connection

    event_id = "nba::2026-04-01::magic_vs_hawks_b"
    event_date = datetime(2026, 4, 1, 23, 0, tzinfo=timezone.utc)
    _ml_snapshot(DataArchiver(), "s2", event_id, event_date, prob_a=0.60)

    conn = get_connection()
    _seed_outcome(conn, "kalshi:MLb", event_id, "magic")  # aligns to outcome_b
    conn.commit()
    conn.close()

    backfill_outcome_closes(sector="nba")

    conn = get_connection()
    val = conn.execute(
        "SELECT pinnacle_close_prob FROM ev_outcomes WHERE market_id='kalshi:MLb'"
    ).fetchone()[0]
    conn.close()
    assert val == pytest.approx(0.40)  # 1 - true_prob_a


def test_dry_run_writes_nothing(temp_dbs):
    from evmax.agents.cleanup.db import get_connection

    event_id = "nba::2026-04-02::heat_vs_celtics"
    event_date = datetime(2026, 4, 2, 23, 0, tzinfo=timezone.utc)
    _ml_snapshot(DataArchiver(), "s3", event_id, event_date, prob_a=0.65,
                 a_label="Boston Celtics", b_label="Miami Heat")

    conn = get_connection()
    _seed_outcome(conn, "kalshi:ML3", event_id, "celtics")
    conn.commit()
    conn.close()

    res = backfill_outcome_closes(sector="nba", dry_run=True)
    assert res["candidates"] == 1
    assert res["filled"] == 1  # would fill

    conn = get_connection()
    val = conn.execute(
        "SELECT pinnacle_close_prob FROM ev_outcomes WHERE market_id='kalshi:ML3'"
    ).fetchone()[0]
    conn.close()
    assert val is None  # ...but dry-run did not write


def test_idempotent_skips_already_filled(temp_dbs):
    from evmax.agents.cleanup.db import get_connection

    event_id = "nba::2026-04-03::hornets_vs_pacers"
    event_date = datetime(2026, 4, 3, 23, 0, tzinfo=timezone.utc)
    _ml_snapshot(DataArchiver(), "s4", event_id, event_date, prob_a=0.7,
                 a_label="Charlotte Hornets", b_label="Indiana Pacers")

    conn = get_connection()
    _seed_outcome(conn, "kalshi:ML4", event_id, "hornets")
    conn.commit()
    conn.close()

    assert backfill_outcome_closes(sector="nba")["filled"] == 1
    # second pass: row already has a close, so it is no longer a candidate
    assert backfill_outcome_closes(sector="nba")["candidates"] == 0
