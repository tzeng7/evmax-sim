"""Tests for evmax.clients.time_util.kalshi_game_day.

The bug this guards against: Pinnacle's `commence_time` is UTC. For US
evening games (e.g. 10pm ET tipoff) the UTC timestamp is on the NEXT
calendar day, which caused event_ids and stored event_dates to be
off-by-one relative to Kalshi's ticker date (e.g. ticker says APR20,
stored date says APR21).
"""

from __future__ import annotations

from datetime import datetime, timezone

from evmax.clients.time_util import kalshi_game_day


def test_us_sport_evening_tipoff_stays_on_game_day():
    """Nuggets vs Timberwolves G2: ~9:30pm ET tipoff = 01:30 UTC next day.
    Under naive UTC strftime this would read as the next day. Under the
    US/Eastern conversion it must stay on the actual game date."""
    commence = datetime(2026, 4, 21, 1, 30, tzinfo=timezone.utc)
    assert kalshi_game_day(commence, "nba") == "2026-04-20"


def test_us_sport_late_pacific_tipoff():
    """A 10pm PT Lakers home game = 05:00 UTC next day. 01:00 ET is
    still the 'next day' in Eastern, so ET-based formatting pushes to
    APR21 — which matches how Kalshi encodes late Pacific games."""
    commence = datetime(2026, 4, 21, 5, 0, tzinfo=timezone.utc)
    assert kalshi_game_day(commence, "nba") == "2026-04-21"


def test_us_sport_daytime_game_unaffected():
    """A noon ET game = 16:00 UTC same day. Both conventions agree."""
    commence = datetime(2026, 4, 20, 16, 0, tzinfo=timezone.utc)
    assert kalshi_game_day(commence, "nfl") == "2026-04-20"


def test_non_us_sector_uses_utc_strftime():
    """Soccer / tennis / esports stay on the existing UTC convention —
    we don't know the home league TZ and the previous behavior was UTC."""
    commence = datetime(2026, 4, 21, 1, 30, tzinfo=timezone.utc)
    assert kalshi_game_day(commence, "soccer") == "2026-04-21"
    assert kalshi_game_day(commence, "tennis") == "2026-04-21"
    assert kalshi_game_day(commence, "lol") == "2026-04-21"


def test_none_returns_unknown():
    assert kalshi_game_day(None, "nba") == "unknown"


def test_us_sectors_coverage():
    """Every US game sector must be routed through US/Eastern."""
    commence = datetime(2026, 4, 21, 1, 30, tzinfo=timezone.utc)
    # NOTE: "baseball" is the actual sector key for MLB across the codebase
    # (sectors/registry.py, data/categories.yaml). "mlb" is kept as an alias
    # for legacy callers but the production path goes through "baseball".
    for sector in ("nba", "wnba", "nfl", "mlb", "baseball", "nhl", "ncaab", "ncaaw"):
        assert kalshi_game_day(commence, sector) == "2026-04-20", sector


def test_baseball_evening_game_uses_eastern_calendar():
    """Regression: MLB sector key is 'baseball', not 'mlb'. An 8:10pm CT
    Astros home game (May 12) = 01:10 UTC May 13. Under the buggy mapping
    this returned '2026-05-13', so Pinnacle's game-1 event_id was off-by-one
    relative to Kalshi's 'MAY12' ticker — causing the matching engine to
    pair Kalshi's game 1 with Pinnacle's game 2 in a back-to-back series."""
    commence = datetime(2026, 5, 13, 1, 10, tzinfo=timezone.utc)
    assert kalshi_game_day(commence, "baseball") == "2026-05-12"
