"""NBA season labels (evmax/clients/nba_season.py) — the rollover rule every
NBA fetch and model guard uses."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from evmax.clients.nba_season import (
    as_date,
    nba_espn_season_year,
    nba_season_label,
    nba_season_start_year,
    season_matches,
)


@pytest.mark.parametrize("d, label", [
    (date(2026, 6, 30), "2025-26"),   # Finals window: still last season
    (date(2026, 7, 1), "2026-27"),    # rollover day
    (date(2026, 10, 6), "2026-27"),   # preseason
    (date(2026, 10, 20), "2026-27"),  # opening night
    (date(2027, 2, 15), "2026-27"),
    (date(2027, 6, 15), "2026-27"),
    (date(1999, 11, 2), "1999-00"),   # two-digit end year wraps the century
])
def test_season_label_rolls_over_on_july_1(d, label):
    assert nba_season_label(d) == label


def test_espn_season_year_is_the_end_year():
    assert nba_espn_season_year(date(2026, 10, 20)) == 2027
    assert nba_espn_season_year(date(2027, 4, 1)) == 2027
    assert nba_season_start_year(date(2027, 4, 1)) == 2026


def test_as_date_accepts_market_event_date_shapes():
    # PredictionMarket.event_date: Kalshi tickers anchor at noon UTC.
    assert as_date(datetime(2026, 10, 20, 12, tzinfo=timezone.utc)) == date(2026, 10, 20)
    assert as_date(date(2026, 10, 20)) == date(2026, 10, 20)
    assert as_date("2026-10-20") == date(2026, 10, 20)
    assert as_date("2026-10-20T23:30:00+00:00") == date(2026, 10, 20)


def test_as_date_falls_back_to_today():
    assert as_date(None) == date.today()
    assert as_date("") == date.today()
    assert as_date("not-a-date") == date.today()


def test_season_matches_requires_the_game_season():
    assert season_matches("2026-27", date(2026, 10, 20))
    assert not season_matches("2025-26", date(2026, 10, 20))   # last season's state
    assert not season_matches("2026-27", date(2026, 4, 20))    # last season's game


def test_unstamped_state_never_matches():
    # States written before the stamp existed came from the hard-coded
    # "2025-26" fetch — never trust them.
    assert not season_matches(None, date(2026, 10, 20))
    assert not season_matches("", date(2026, 10, 20))


def test_season_matches_defaults_to_today():
    assert season_matches(nba_season_label(), None)
