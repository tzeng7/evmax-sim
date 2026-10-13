"""NBA season labels — the one rule every NBA data fetch and model guard uses.

nba_api names a season by its START year ("2026-27"); ESPN's athlete
endpoints take the END year (2027). The Finals end in June and the next
preseason opens in early October, so every date from July 1 on belongs to the
next season. From July to opening night a fetch for the new season returns no
regular-season rows, and the NBA team-stat models abstain — there are no
games to price.

The NBA stack shipped 2026-04-19 with "2025-26" hard-coded at six fetch
sites. At the 2026-27 rollover those fetches kept returning last season's
complete stats (gp 82-105) and the models fired them at full confidence into
new-season games — the WNBA opening-month chalk-bias failure (+24pp). States
now carry a ``season`` stamp, and a state stamped for another season (or
unstamped) never prices a game.

Leaf module (stdlib only) so clients and model agents can both import it.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional, Union

# Month on which the NBA season label rolls over (July 1).
SEASON_ROLLOVER_MONTH = 7

DateLike = Union[date, datetime, str, None]


def as_date(value: DateLike) -> date:
    """Coerce a game date to a ``date``; None or an unparseable value → today.

    Accepts ``PredictionMarket.event_date`` (a datetime), a ``date``, or an
    ISO string (``YYYY-MM-DD`` or a full timestamp).
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value:
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            pass
    return date.today()


def nba_season_start_year(value: DateLike = None) -> int:
    """Start year of the NBA season a game on ``value`` belongs to."""
    d = as_date(value)
    return d.year if d.month >= SEASON_ROLLOVER_MONTH else d.year - 1


def nba_season_label(value: DateLike = None) -> str:
    """nba_api season label: 2026-10-20 → "2026-27", 2027-06-10 → "2026-27"."""
    start = nba_season_start_year(value)
    return f"{start}-{(start + 1) % 100:02d}"


def nba_espn_season_year(value: DateLike = None) -> int:
    """ESPN season year (the END year): 2026-10-20 → 2027."""
    return nba_season_start_year(value) + 1


def season_matches(state_season: Optional[str], game_date: DateLike = None) -> bool:
    """True only when a state's ``season`` stamp is the season of ``game_date``.

    ``game_date`` defaults to today. An unstamped state was written by the
    pre-2026-10-13 hard-coded "2025-26" fetch, so it is treated as stale.
    """
    return bool(state_season) and state_season == nba_season_label(game_date)
