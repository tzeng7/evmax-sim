"""Date/time helpers shared across market clients.

The problem these solve: Pinnacle returns `commence_time` as a UTC ISO
timestamp. For US evening games that tip after 8pm local, the UTC
timestamp lands on the NEXT calendar day. If we strftime that UTC
datetime directly, the event_id date is off-by-one relative to Kalshi's
ticker date (which uses US calendar day convention), breaking
canonical-key matching and surfacing a wrong date on the dashboard.

`kalshi_game_day` converts a commence_time to a date string that aligns
with Kalshi's ticker convention, sector-by-sector.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

# Sectors whose Kalshi ticker date follows US calendar day (most US sports).
# For these, convert commence_time to US/Eastern before formatting so a
# 10pm ET game on April 20 is "2026-04-20", not "2026-04-21".
_US_SECTORS = frozenset({"nba", "nfl", "mlb", "nhl", "ncaab", "ncaaw"})
_US_TZ = ZoneInfo("America/New_York")


def kalshi_game_day(event_date: datetime | None, sector: str) -> str:
    """Return a YYYY-MM-DD string for `event_date` using the sector's
    canonical game-day convention.

    US sports → America/New_York. Everything else falls back to whatever
    tz the datetime already carries (typically UTC), which matches our
    prior behavior for soccer/tennis/esports.
    """
    if event_date is None:
        return "unknown"
    if sector in _US_SECTORS:
        return event_date.astimezone(_US_TZ).strftime("%Y-%m-%d")
    return event_date.strftime("%Y-%m-%d")
