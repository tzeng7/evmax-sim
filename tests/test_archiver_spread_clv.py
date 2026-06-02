"""Tests for the spread-aware closing-line helper added so spread bets can
have ``pinnacle_close_prob`` populated for CLV tracking.

Before this change, ``DataArchiver.get_closing_line_aligned`` filtered to
``spread_line IS NULL`` (moneyline only), so every spread bet got NULL in
``ev_outcomes.pinnacle_close_prob`` and CLV metrics were unmeasurable for
all spread markets. The new ``get_spread_closing_line_aligned`` reads the
spread-snapshot rows and aligns the YES/NO probability to ``yes_team``.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from evmax.archiver import DataArchiver
from evmax.models.odds import SharpBook, SharpOdds


@pytest.fixture
def temp_archive_db(tmp_path, monkeypatch):
    db_path = tmp_path / "archive_clv_test.db"
    monkeypatch.setattr("evmax.archiver.DB_PATH", db_path)
    return db_path


def _spread_sharp(
    event_id: str,
    spread_line: float,
    true_prob_a: float,
    fetched_at: datetime,
    event_date: datetime,
    outcome_a_label: str = "mercury",
    outcome_b_label: str = "lynx",
) -> SharpOdds:
    return SharpOdds(
        event_id=event_id,
        book=SharpBook.pinnacle,
        sector="wnba",
        outcome_a_label=outcome_a_label,
        outcome_b_label=outcome_b_label,
        outcome_a_decimal=1.91,
        outcome_b_decimal=1.91,
        true_prob_a=true_prob_a,
        true_prob_b=1.0 - true_prob_a,
        margin=0.04,
        spread_line=spread_line,
        event_date=event_date,
        fetched_at=fetched_at,
    )


def _ml_sharp(event_id: str, true_prob_a: float, fetched_at: datetime, event_date: datetime) -> SharpOdds:
    """Moneyline snapshot for the same event — used to verify the new spread
    helper does NOT pick up moneyline rows."""
    return SharpOdds(
        event_id=event_id,
        book=SharpBook.pinnacle,
        sector="wnba",
        outcome_a_label="mercury",
        outcome_b_label="lynx",
        outcome_a_decimal=1.85,
        outcome_b_decimal=1.95,
        true_prob_a=true_prob_a,
        true_prob_b=1.0 - true_prob_a,
        margin=0.04,
        event_date=event_date,
        fetched_at=fetched_at,
    )


def test_spread_close_returns_aligned_prob(temp_archive_db):
    archiver = DataArchiver()
    event_id = "wnba::2026-05-25::mercury_vs_lynx"
    event_date = datetime(2026, 5, 25, 23, 0, tzinfo=timezone.utc)

    # Two pre-tipoff snapshots: the later one is the "close". UNIQUE constraint
    # is (session_id, event_id, book) so each scan goes in its own session —
    # mirrors how the live scanner persists per cycle.
    archiver.open_session("s1a", ["wnba"], "test")
    archiver.archive_sharp_odds("s1a", "wnba", [
        _spread_sharp(event_id, -5.5, 0.55, event_date - timedelta(hours=3), event_date),
    ])
    archiver.open_session("s1b", ["wnba"], "test")
    archiver.archive_sharp_odds("s1b", "wnba", [
        _spread_sharp(event_id, -5.5, 0.62, event_date - timedelta(minutes=20), event_date),
    ])

    # yes_team aligns to outcome_a → expect true_prob_a (0.62)
    p = archiver.get_spread_closing_line_aligned(event_id, "mercury", -5.5)
    assert p == pytest.approx(0.62)

    # yes_team aligns to outcome_b → expect 1 - true_prob_a (0.38)
    p_b = archiver.get_spread_closing_line_aligned(event_id, "lynx", 5.5)
    assert p_b == pytest.approx(0.38)


def test_spread_close_ignores_moneyline_rows(temp_archive_db):
    """The new helper must not return moneyline probs even when both ML and
    spread snapshots exist for the same event — that was the original bug."""
    archiver = DataArchiver()
    event_id = "wnba::2026-05-25::mercury_vs_lynx_2"
    event_date = datetime(2026, 5, 25, 23, 0, tzinfo=timezone.utc)
    fetched = event_date - timedelta(minutes=20)

    # ML and spread go in separate sessions because UNIQUE(session_id, event_id,
    # book) would otherwise collapse them to one row
    archiver.open_session("s2_ml", ["wnba"], "test")
    archiver.archive_sharp_odds("s2_ml", "wnba", [
        _ml_sharp(event_id, 0.70, fetched, event_date),
    ])
    archiver.open_session("s2_spread", ["wnba"], "test")
    archiver.archive_sharp_odds("s2_spread", "wnba", [
        _spread_sharp(event_id, -5.5, 0.52, fetched, event_date),
    ])

    p = archiver.get_spread_closing_line_aligned(event_id, "mercury", -5.5)
    assert p == pytest.approx(0.52)  # spread, not ML
    # And the legacy ML helper still skips spread rows
    p_ml = archiver.get_closing_line_aligned(event_id, "mercury")
    assert p_ml == pytest.approx(0.70)


def test_spread_close_falls_back_when_line_mismatches(temp_archive_db):
    """When the bet's line doesn't match any archived snapshot, fall back to
    the latest spread row rather than returning None — pre-tipoff line drift
    of ±0.5pt is routine and shouldn't kill CLV measurement entirely."""
    archiver = DataArchiver()
    archiver.open_session("s3", ["wnba"], "test")
    event_id = "wnba::2026-05-25::mercury_vs_lynx_3"
    event_date = datetime(2026, 5, 25, 23, 0, tzinfo=timezone.utc)
    fetched = event_date - timedelta(minutes=20)

    # Pinnacle line drifted to -6.5 but our bet was placed at -5.5
    archiver.archive_sharp_odds("s3", "wnba", [
        _spread_sharp(event_id, -6.5, 0.58, fetched, event_date),
    ])

    p = archiver.get_spread_closing_line_aligned(event_id, "mercury", -5.5)
    assert p == pytest.approx(0.58)


def test_spread_close_excludes_post_tipoff_snapshots(temp_archive_db):
    """Snapshots fetched after tipoff (live in-game) collapse toward outcome
    and must be excluded from the close calculation."""
    archiver = DataArchiver()
    event_id = "wnba::2026-05-25::mercury_vs_lynx_4"
    event_date = datetime(2026, 5, 25, 23, 0, tzinfo=timezone.utc)

    archiver.open_session("s4a", ["wnba"], "test")
    archiver.archive_sharp_odds("s4a", "wnba", [
        _spread_sharp(event_id, -5.5, 0.55, event_date - timedelta(minutes=15), event_date),
    ])
    archiver.open_session("s4b", ["wnba"], "test")
    archiver.archive_sharp_odds("s4b", "wnba", [
        # Post-tipoff sample with extreme prob — must be ignored
        _spread_sharp(event_id, -5.5, 0.95, event_date + timedelta(minutes=30), event_date),
    ])

    p = archiver.get_spread_closing_line_aligned(event_id, "mercury", -5.5)
    assert p == pytest.approx(0.55)


def test_spread_close_returns_none_when_no_snapshots(temp_archive_db):
    archiver = DataArchiver()
    archiver.open_session("s5", ["wnba"], "test")
    p = archiver.get_spread_closing_line_aligned("nonexistent::event", "mercury", -5.5)
    assert p is None


# ---------------------------------------------------------------------------
# Totals close-line alignment. Baseball totals are framed YES = OVER on Kalshi,
# so the close prob must align by over/under side (yes_team "over"/"under"),
# NOT by team label. Before this wiring, totals fell into resolver.py's
# `else: pinn_close = None` branch and every totals CLV stayed NULL.
# ---------------------------------------------------------------------------

def _total_sharp(
    event_id: str,
    total_line: float,
    prob_over: float,
    fetched_at: datetime,
    event_date: datetime,
) -> SharpOdds:
    return SharpOdds(
        event_id=event_id,
        book=SharpBook.pinnacle,
        sector="baseball",
        outcome_a_label="over",
        outcome_b_label="under",
        outcome_a_decimal=1.91,
        outcome_b_decimal=1.91,
        true_prob_a=0.0,  # totals-only record: ML probs zeroed (validator skips sum check)
        true_prob_b=0.0,
        margin=0.04,
        total_line=total_line,
        true_prob_over=prob_over,
        true_prob_under=1.0 - prob_over,
        event_date=event_date,
        fetched_at=fetched_at,
    )


def test_total_close_returns_over_under_aligned_prob(temp_archive_db):
    archiver = DataArchiver()
    event_id = "baseball::2026-05-25::yankees_vs_redsox::total::8.5"
    event_date = datetime(2026, 5, 25, 23, 0, tzinfo=timezone.utc)

    archiver.open_session("t1a", ["baseball"], "test")
    archiver.archive_sharp_odds("t1a", "baseball", [
        _total_sharp(event_id, 8.5, 0.58, event_date - timedelta(hours=3), event_date),
    ])
    archiver.open_session("t1b", ["baseball"], "test")
    archiver.archive_sharp_odds("t1b", "baseball", [
        _total_sharp(event_id, 8.5, 0.63, event_date - timedelta(minutes=20), event_date),
    ])

    # OVER side → latest pre-tipoff true_prob_over (0.63)
    p_over = archiver.get_total_closing_line_aligned(event_id, "over", 8.5)
    assert p_over == pytest.approx(0.63)

    # UNDER side (the Kalshi NO side) → 1 - over (0.37)
    p_under = archiver.get_total_closing_line_aligned(event_id, "under", 8.5)
    assert p_under == pytest.approx(0.37)


def test_total_close_returns_none_for_non_over_under_label(temp_archive_db):
    """A team label (not over/under) must not resolve a totals close."""
    archiver = DataArchiver()
    event_id = "baseball::2026-05-25::yankees_vs_redsox::total::8.5"
    event_date = datetime(2026, 5, 25, 23, 0, tzinfo=timezone.utc)
    archiver.open_session("t2", ["baseball"], "test")
    archiver.archive_sharp_odds("t2", "baseball", [
        _total_sharp(event_id, 8.5, 0.58, event_date - timedelta(hours=2), event_date),
    ])
    assert archiver.get_total_closing_line_aligned(event_id, "yankees", 8.5) is None
