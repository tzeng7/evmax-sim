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
#
# IMPORTANT: keys must match the *sector* names used throughout the codebase
# (see evmax/sectors/registry.py and data/categories.yaml), not the league
# brand. MLB is sector="baseball" — using "mlb" here silently fell through
# to the UTC branch and caused 8pm+ ET MLB games to land on the next calendar
# day on the Pinnacle side, so Kalshi's game 1 in a series matched Pinnacle's
# game 2 (off-by-one date in the canonical event key).
_US_SECTORS = frozenset(
    {"nba", "wnba", "nfl", "mlb", "baseball", "nhl", "ncaab", "ncaaw", "ncaaf", "ufc"}
)
# "ufc": Kalshi tickers/titles date UFC cards on the US calendar day
# (KXUFCFIGHT-26JUL11… / "scheduled for Jul 11"), but a US main card that
# tips after ~8pm ET carries a next-day UTC startTime on Pinnacle
# (2026-07-12T03:20Z) — the exact off-by-one this table exists to fix.
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
