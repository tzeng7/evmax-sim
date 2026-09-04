"""nflverse depth charts → PRE-game QB starters for ``nfl_qb_elo``.

Why: ``nfl_qb_elo``'s ``current_starters`` map is derived from play-by-play
(the max-attempts passer of each team's LAST game), so a starter change
announced mid-week is invisible until AFTER the game it matters for — exactly
the game the per-QB delta layer exists for. nflverse publishes team depth
charts (2025→: near-daily ``dt`` snapshots; ≤2024: weekly rows) and QB1
changes land there days before kickoff — KC 2025: Mahomes → Minshew listed
12-16 for the 12-21 game, → Oladokun listed 12-23. Layering the ESPN injury
report on top (QB Out / IR / Suspension → skip to the next healthy QB on the
chart) gives a genuine pre-game starter.

Two nflverse schemas are normalized to :class:`QbChartRow`:
  - snapshot (2025→): ``dt, team, player_name, pos_abb, pos_rank``
  - weekly   (≤2024): ``season, club_code, week, game_type, depth_team,
    position, full_name``

Everything here is fail-soft: a fetch/parse failure yields no overrides and
the agent falls back to its last-game starters.
"""

from __future__ import annotations

import re
import threading
import time
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Iterable, Optional

import structlog

from evmax.agents.models.nfl_efficiency_agent import NFL_ABBREV_TO_NAME

logger = structlog.get_logger(__name__)

# nflverse / ESPN club codes that differ from the abbreviations in
# NFL_ABBREV_TO_NAME (which uses LA / WAS / JAX / LV / LAC).
_ABBR_ALIASES: dict[str, str] = {
    "LAR": "LA", "WSH": "WAS", "JAC": "JAX", "OAK": "LV", "SD": "LAC", "STL": "LA",
}
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
_CACHE_TTL_S = 6 * 3600


@dataclass(frozen=True)
class QbChartRow:
    team: str                 # full team name key, e.g. "kansas city chiefs"
    player: str               # full name as listed on the chart
    rank: int                 # 1 = QB1
    as_of: Optional[datetime]  # snapshot timestamp (snapshot schema) or None
    week: Optional[int]       # NFL week (weekly schema) or None


def _strip_suffix(tokens: list[str]) -> list[str]:
    while len(tokens) > 1 and tokens[-1].lower().strip(".") in _SUFFIXES:
        tokens.pop()
    return tokens


def pbp_passer_name(full_name: str) -> str:
    """Full name → nflverse ``passer_player_name`` form: ``"Michael Penix Jr." → "M.Penix"``.

    First initial + "." + everything after the first name, suffix dropped
    (``"Gardner Minshew II" → "G.Minshew"``, ``"Aidan O'Connell" → "A.O'Connell"``,
    ``"Amon-Ra St. Brown" → "A.St. Brown"``). Matches the keys ``nfl_qb_elo``
    stores in ``qb_deltas`` / ``current_starters``.
    """
    tokens = _strip_suffix([t for t in (full_name or "").replace(",", " ").split() if t])
    if not tokens:
        return ""
    if len(tokens) == 1:
        return tokens[0]
    return f"{tokens[0][0]}.{' '.join(tokens[1:])}"


def normalize_person(name: str) -> str:
    """Lowercase ASCII, suffix + punctuation stripped — the injury-report join key."""
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    tokens = _strip_suffix([t for t in re.sub(r"[^a-z0-9' ]+", " ", s.lower()).split() if t])
    return " ".join(t.replace("'", "") for t in tokens)


def team_key(abbr: str) -> Optional[str]:
    a = (abbr or "").upper().strip()
    a = _ABBR_ALIASES.get(a, a)
    return NFL_ABBREV_TO_NAME.get(a)


def _parse_dt(raw) -> Optional[datetime]:
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def rows_from_frame(df) -> list[QbChartRow]:
    """Normalize a nflverse depth-chart frame (either schema) to QB rows."""
    import polars as pl

    cols = set(df.columns)
    rows: list[QbChartRow] = []
    if "dt" in cols:  # snapshot schema (2025→)
        sub = df.filter(pl.col("pos_abb") == "QB")
        for r in sub.iter_rows(named=True):
            team = team_key(r.get("team"))
            rank = r.get("pos_rank")
            as_of = _parse_dt(r.get("dt"))
            if not team or rank is None or as_of is None or not r.get("player_name"):
                continue
            rows.append(QbChartRow(team, str(r["player_name"]), int(rank), as_of, None))
    else:  # weekly schema (≤2024)
        sub = df.filter(pl.col("position") == "QB")
        for r in sub.iter_rows(named=True):
            team = team_key(r.get("club_code"))
            wk = r.get("week")
            try:
                rank = int(r.get("depth_team") or 0)
            except (TypeError, ValueError):
                continue
            if not team or wk is None or rank <= 0 or not r.get("full_name"):
                continue
            rows.append(QbChartRow(team, str(r["full_name"]), rank, None, int(wk)))
    return rows


_cache: dict[int, tuple[float, list[QbChartRow]]] = {}
_lock = threading.Lock()


def load_qb_chart_rows(season: int, *, refresh: bool = False) -> list[QbChartRow]:
    """QB depth-chart rows for `season` via nflreadpy (TTL-cached in-process).

    Returns ``[]`` on any failure so callers degrade to last-game starters.
    """
    with _lock:
        hit = _cache.get(season)
        if hit and not refresh and time.time() - hit[0] < _CACHE_TTL_S:
            return hit[1]
    try:
        import nflreadpy as nfl
        df = nfl.load_depth_charts(seasons=[season])
        rows = rows_from_frame(df)
    except Exception as e:  # noqa: BLE001 — fail-soft by design
        logger.warning("nfl_depth_charts_unavailable", season=season, error=str(e))
        return []
    with _lock:
        _cache[season] = (time.time(), rows)
    logger.info("nfl_depth_charts_loaded", season=season, qb_rows=len(rows),
                schema="snapshot" if rows and rows[0].as_of is not None else "weekly")
    return rows


def qb_depth_as_of(
    rows: Iterable[QbChartRow],
    *,
    as_of: Optional[date] = None,
    week: Optional[int] = None,
) -> dict[str, list[str]]:
    """Per team, QB full names in rank order from the applicable chart.

    Snapshot rows: the latest snapshot whose date is STRICTLY before `as_of`
    (``as_of=None`` → the latest snapshot available — the live path). Weekly
    rows: the rows for `week` (``None`` → nothing).
    """
    rows = list(rows)
    if not rows:
        return {}
    latest: dict[str, tuple[datetime, dict[int, str]]] = {}
    if rows[0].as_of is not None:
        for r in rows:
            if r.as_of is None:
                continue
            if as_of is not None and r.as_of.date() >= as_of:
                continue
            cur = latest.get(r.team)
            if cur is None or r.as_of > cur[0]:
                latest[r.team] = (r.as_of, {r.rank: r.player})
            elif r.as_of == cur[0]:
                cur[1].setdefault(r.rank, r.player)
        return {team: [n for _, n in sorted(qbs.items())] for team, (_, qbs) in latest.items()}
    if week is None:
        return {}
    by_team: dict[str, dict[int, str]] = {}
    for r in rows:
        if r.week == week:
            by_team.setdefault(r.team, {}).setdefault(r.rank, r.player)
    return {team: [n for _, n in sorted(qbs.items())] for team, qbs in by_team.items()}


def resolve_pregame_starters(
    season: int,
    *,
    as_of: Optional[date] = None,
    week: Optional[int] = None,
    out_players: Optional[dict[str, Iterable[str]]] = None,
    rows: Optional[list[QbChartRow]] = None,
) -> dict[str, dict]:
    """{team full name → {"starter": pbp name, "full_name", "rank", "skipped": [...]}}.

    Walks each team's chart in rank order and takes the first QB not listed
    in `out_players` (team → names from the injury report, any name form).
    """
    rows = load_qb_chart_rows(season) if rows is None else rows
    depth = qb_depth_as_of(rows, as_of=as_of, week=week)
    out = {
        str(team).lower().strip(): {normalize_person(n) for n in names}
        for team, names in (out_players or {}).items()
    }
    result: dict[str, dict] = {}
    for team, qbs in depth.items():
        skipped: list[str] = []
        for rank, name in enumerate(qbs, start=1):
            if normalize_person(name) in out.get(team, set()):
                skipped.append(name)
                continue
            result[team] = {
                "starter": pbp_passer_name(name), "full_name": name,
                "rank": rank, "skipped": skipped,
            }
            break
    return result
