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


def test_spread_close_falls_back_within_tolerance(temp_archive_db):
    """When the bet's line doesn't exactly match but is within ±1.0pt of an
    archived snapshot, fall back to it — pre-tipoff line drift of ±0.5–1.0pt is
    routine and shouldn't kill CLV measurement entirely."""
    archiver = DataArchiver()
    archiver.open_session("s3", ["wnba"], "test")
    event_id = "wnba::2026-05-25::mercury_vs_lynx_3"
    event_date = datetime(2026, 5, 25, 23, 0, tzinfo=timezone.utc)
    fetched = event_date - timedelta(minutes=20)

    # Pinnacle line drifted to -6.5 but our bet was placed at -5.5 (exactly 1pt)
    archiver.archive_sharp_odds("s3", "wnba", [
        _spread_sharp(event_id, -6.5, 0.58, fetched, event_date),
    ])

    p = archiver.get_spread_closing_line_aligned(event_id, "mercury", -5.5)
    assert p == pytest.approx(0.58)


def test_spread_close_returns_none_when_line_out_of_tolerance(temp_archive_db):
    """An alternate-spread bet several points off the only archived (primary)
    line must NOT be matched to it. The archive only captures Pinnacle's
    primary ~0.50 line per game; matching a -5.5 alt bet to a -8.0 main line
    would record a fabricated ~0.50 'close'. Out of tolerance → None."""
    archiver = DataArchiver()
    archiver.open_session("s3b", ["wnba"], "test")
    event_id = "wnba::2026-05-25::mercury_vs_lynx_3b"
    event_date = datetime(2026, 5, 25, 23, 0, tzinfo=timezone.utc)
    fetched = event_date - timedelta(minutes=20)

    # Only the -8.0 primary line is archived; our bet was the -5.5 alt (2.5pt off)
    archiver.archive_sharp_odds("s3b", "wnba", [
        _spread_sharp(event_id, -8.0, 0.50, fetched, event_date),
    ])

    assert archiver.get_spread_closing_line_aligned(event_id, "mercury", -5.5) is None


def test_spread_close_picks_nearest_within_tolerance(temp_archive_db):
    """With multiple in-tolerance snapshots, prefer the closest line, not just
    the latest fetched_at."""
    archiver = DataArchiver()
    event_id = "wnba::2026-05-25::mercury_vs_lynx_3c"
    event_date = datetime(2026, 5, 25, 23, 0, tzinfo=timezone.utc)

    # -6.0 (0.5pt off) fetched earlier; -6.5 (1.0pt off) fetched later.
    # Nearest-line wins → 0.60, not the later-but-farther 0.58.
    archiver.open_session("s3c_a", ["wnba"], "test")
    archiver.archive_sharp_odds("s3c_a", "wnba", [
        _spread_sharp(event_id, -6.0, 0.60, event_date - timedelta(hours=2), event_date),
    ])
    archiver.open_session("s3c_b", ["wnba"], "test")
    archiver.archive_sharp_odds("s3c_b", "wnba", [
        _spread_sharp(event_id, -6.5, 0.58, event_date - timedelta(minutes=20), event_date),
    ])

    p = archiver.get_spread_closing_line_aligned(event_id, "mercury", -5.5)
    assert p == pytest.approx(0.60)


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


def test_total_close_returns_none_when_line_out_of_tolerance(temp_archive_db):
    """A total bet several points off the only archived line must not match it,
    mirroring the spread tolerance bound."""
    archiver = DataArchiver()
    event_id = "baseball::2026-05-25::yankees_vs_redsox::total::8.5"
    event_date = datetime(2026, 5, 25, 23, 0, tzinfo=timezone.utc)
    archiver.open_session("t3", ["baseball"], "test")
    # Only an 8.5 snapshot archived; our bet was the 11.5 alt total (3pt off).
    archiver.archive_sharp_odds("t3", "baseball", [
        _total_sharp(event_id, 8.5, 0.58, event_date - timedelta(hours=2), event_date),
    ])
    assert archiver.get_total_closing_line_aligned(event_id, "over", 11.5) is None
    # But a 9.0 bet (0.5pt off) still falls back within tolerance.
    assert archiver.get_total_closing_line_aligned(event_id, "over", 9.0) == pytest.approx(0.58)


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


# ---------------------------------------------------------------------------
# Placement-aware Kalshi close anchor (get_kalshi_close_price).
#
# The "close" must be a snapshot AT OR AFTER our fill so placed-bet CLV measures
# forward from entry, not against a price that preceded it. For a fill inside the
# T-30 window the upper bound relaxes to tipoff so a genuine post-entry snapshot
# can still be found; with no post-entry snapshot we return None rather than
# fabricate a backward CLV. not_before=None preserves the legacy T-30 behaviour.
# ---------------------------------------------------------------------------

EVENT_ID = "nba::2026-05-25::lakers_vs_celtics"
TICKER = "KXNBAGAME-26MAY25LAC-LAC"


def _seed_kalshi_clv_fixture(archiver):
    """Tipoff anchor + three Kalshi snapshots at tip-3h, tip-1h, tip-20m."""
    tip = datetime(2026, 5, 25, 23, 0, tzinfo=timezone.utc)
    # Pinnacle row gives the tipoff anchor (event_date) for the close window.
    archiver.open_session("so", ["nba"], "test")
    archiver.archive_sharp_odds("so", "nba", [_ml_sharp(EVENT_ID, 0.5, tip - timedelta(hours=4), tip)])
    # Distinct sessions because UNIQUE(session_id, ticker) — one ticker per sweep.
    archiver.archive_kalshi_snapshot("k1", "nba", [{"ticker": TICKER, "yes_price": 0.40, "event_id": EVENT_ID}], fetched_at=tip - timedelta(hours=3))
    archiver.archive_kalshi_snapshot("k2", "nba", [{"ticker": TICKER, "yes_price": 0.50, "event_id": EVENT_ID}], fetched_at=tip - timedelta(hours=1))
    archiver.archive_kalshi_snapshot("k3", "nba", [{"ticker": TICKER, "yes_price": 0.55, "event_id": EVENT_ID}], fetched_at=tip - timedelta(minutes=20))
    return tip


def test_kalshi_close_legacy_no_anchor(temp_archive_db):
    """not_before=None → latest snapshot at or before T-30 (here tip-1h=0.50);
    the tip-20m print is inside the T-30 window and excluded."""
    archiver = DataArchiver()
    _seed_kalshi_clv_fixture(archiver)
    assert archiver.get_kalshi_close_price(TICKER, EVENT_ID) == pytest.approx(0.50)


def test_kalshi_close_excludes_pre_entry_snapshot(temp_archive_db):
    """A fill at tip-2h must skip the tip-3h pre-entry snapshot (0.40) and use
    the post-entry tip-1h snapshot (0.50)."""
    archiver = DataArchiver()
    tip = _seed_kalshi_clv_fixture(archiver)
    p = archiver.get_kalshi_close_price(TICKER, EVENT_ID, not_before=tip - timedelta(hours=2))
    assert p == pytest.approx(0.50)


def test_kalshi_close_relaxes_upper_for_late_fill(temp_archive_db):
    """A fill at tip-25m is already past T-30; the upper bound relaxes to tipoff
    so the tip-20m snapshot (0.55) is found instead of returning None."""
    archiver = DataArchiver()
    tip = _seed_kalshi_clv_fixture(archiver)
    p = archiver.get_kalshi_close_price(TICKER, EVENT_ID, not_before=tip - timedelta(minutes=25))
    assert p == pytest.approx(0.55)


def test_kalshi_close_none_when_no_post_entry_snapshot(temp_archive_db):
    """A fill at tip-5m is after every snapshot — no forward close exists, so we
    return None rather than measure CLV backward."""
    archiver = DataArchiver()
    tip = _seed_kalshi_clv_fixture(archiver)
    assert archiver.get_kalshi_close_price(TICKER, EVENT_ID, not_before=tip - timedelta(minutes=5)) is None


def test_archive_kalshi_snapshot_round_trips(temp_archive_db):
    """The lightweight snapshot primitive lands a row get_kalshi_close_price reads."""
    archiver = DataArchiver()
    tip = datetime(2026, 5, 25, 23, 0, tzinfo=timezone.utc)
    archiver.open_session("so", ["nba"], "test")
    archiver.archive_sharp_odds("so", "nba", [_ml_sharp(EVENT_ID, 0.5, tip - timedelta(hours=4), tip)])
    n = archiver.archive_kalshi_snapshot(
        "snap", "nba",
        [{"ticker": TICKER, "yes_price": 0.62, "event_id": EVENT_ID}],
        fetched_at=tip - timedelta(minutes=20),
    )
    assert n == 1
    # tip-20m is inside T-30; anchor a late fill so the relax path surfaces it.
    p = archiver.get_kalshi_close_price(TICKER, EVENT_ID, not_before=tip - timedelta(minutes=25))
    assert p == pytest.approx(0.62)


def test_archive_kalshi_snapshot_empty_is_noop(temp_archive_db):
    archiver = DataArchiver()
    assert archiver.archive_kalshi_snapshot("snap", "nba", []) == 0


# ---------------------------------------------------------------------------
# Close-snapshot STALENESS (get_kalshi_close_staleness_h).
#
# The gap between the T-30 close target and the snapshot we actually scored as
# "close." Small = a genuine near-tip read; large = a watch-closes capture gap
# where "close" is a stale mid-day price. Feeds the shadow CLV gate's optional
# stale-capture exclusion.
# ---------------------------------------------------------------------------
def test_close_staleness_zero_at_target(temp_archive_db):
    """The tip-30m snapshot IS the T-30 target → staleness 0h."""
    archiver = DataArchiver()
    tip = datetime(2026, 5, 25, 23, 0, tzinfo=timezone.utc)
    archiver.open_session("so", ["nba"], "test")
    archiver.archive_sharp_odds(
        "so", "nba", [_ml_sharp(EVENT_ID, 0.5, tip - timedelta(hours=4), tip)]
    )
    archiver.archive_kalshi_snapshot(
        "k1", "nba", [{"ticker": TICKER, "yes_price": 0.5, "event_id": EVENT_ID}],
        fetched_at=tip - timedelta(minutes=30),
    )
    assert archiver.get_kalshi_close_staleness_h(TICKER, EVENT_ID) == pytest.approx(0.0)


def test_close_staleness_measures_capture_gap(temp_archive_db):
    """Only a tip-3h snapshot exists before T-30 → close is 2.5h stale."""
    archiver = DataArchiver()
    tip = datetime(2026, 5, 25, 23, 0, tzinfo=timezone.utc)
    archiver.open_session("so", ["nba"], "test")
    archiver.archive_sharp_odds(
        "so", "nba", [_ml_sharp(EVENT_ID, 0.5, tip - timedelta(hours=4), tip)]
    )
    archiver.archive_kalshi_snapshot(
        "k1", "nba", [{"ticker": TICKER, "yes_price": 0.4, "event_id": EVENT_ID}],
        fetched_at=tip - timedelta(hours=3),
    )
    # target = tip-30m; snapshot = tip-3h; gap = 2.5h.
    assert archiver.get_kalshi_close_staleness_h(TICKER, EVENT_ID) == pytest.approx(2.5)


def test_close_staleness_matches_selected_close(temp_archive_db):
    """Staleness anchors on the SAME snapshot get_kalshi_close_price picks: with
    three snapshots (tip-3h/-1h/-20m) the T-30 close is tip-1h → staleness 0.5h."""
    archiver = DataArchiver()
    tip = _seed_kalshi_clv_fixture(archiver)
    assert archiver.get_kalshi_close_price(TICKER, EVENT_ID) == pytest.approx(0.50)
    assert archiver.get_kalshi_close_staleness_h(TICKER, EVENT_ID) == pytest.approx(0.5)


def test_close_staleness_none_without_anchor(temp_archive_db):
    """No Pinnacle tipoff row → no close target → None (treat as unknown)."""
    archiver = DataArchiver()
    assert archiver.get_kalshi_close_staleness_h(TICKER, EVENT_ID) is None


# ---------------------------------------------------------------------------
# launchd watch-closes lookahead vs the T-30 close target (window alignment).
#
# 2026-07-12 root-cause finding: with --lookahead 30 the watch-closes capture
# window only OPENED at tipoff-30min, while get_kalshi_close_price selects the
# latest snapshot AT OR BEFORE tipoff-30min — disjoint windows, so watch-closes
# snapshots never qualified as the close for unplaced (shadow) bets and every
# close fell back to the last hourly watch-listings sweep (up to hours stale).
# The lookahead must stay STRICTLY GREATER than minutes_before or the near-tip
# capture service silently contributes nothing to close-price measurement.
# ---------------------------------------------------------------------------
def test_watch_closes_lookahead_exceeds_close_target():
    import inspect
    import plistlib
    from pathlib import Path

    plist_path = (
        Path.home() / "Library" / "LaunchAgents" / "com.evmax.watch-closes.plist"
    )
    if not plist_path.exists():
        pytest.skip("machine-local launchd plist not present (CI or other host)")

    with plist_path.open("rb") as f:
        args = plistlib.load(f)["ProgramArguments"]
    assert "--lookahead" in args, "watch-closes plist no longer passes --lookahead"
    lookahead_min = int(args[args.index("--lookahead") + 1])

    minutes_before = inspect.signature(
        DataArchiver.get_kalshi_close_price
    ).parameters["minutes_before"].default

    assert lookahead_min > minutes_before, (
        f"watch-closes --lookahead ({lookahead_min}m) must exceed the "
        f"get_kalshi_close_price T-{minutes_before} close target, or its "
        "snapshots all land inside the excluded window and the close silently "
        "falls back to stale hourly watch-listings sweeps"
    )


def test_candlebf_snapshot_selected_as_close_over_stale_watchlist(temp_archive_db):
    """Backfilled candlestick rows (session_id 'candlebf-*') participate in
    close selection like any other snapshot: a candle ending at T-45m beats a
    watchlist sweep from hours earlier, killing the stale-close artifact."""
    archiver = DataArchiver()
    tip = datetime(2026, 5, 25, 23, 0, tzinfo=timezone.utc)
    archiver.open_session("so", ["nba"], "test")
    archiver.archive_sharp_odds("so", "nba", [_ml_sharp(EVENT_ID, 0.5, tip - timedelta(hours=4), tip)])
    archiver.archive_kalshi_snapshot(
        "watchlist-20260525T1200", "nba",
        [{"ticker": TICKER, "yes_price": 0.40, "event_id": EVENT_ID}],
        fetched_at=tip - timedelta(hours=6),
    )
    archiver.archive_kalshi_snapshot(
        "candlebf-1782428100", "nba",
        [{"ticker": TICKER, "yes_price": 0.52, "event_id": EVENT_ID}],
        fetched_at=tip - timedelta(minutes=45),
    )
    assert archiver.get_kalshi_close_price(TICKER, EVENT_ID) == pytest.approx(0.52)
