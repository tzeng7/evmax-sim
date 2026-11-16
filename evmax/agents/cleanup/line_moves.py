"""Pinnacle vs venue moneyline line movement, read from archive.db.

Read-only lens behind ``evmax cleanup line-moves``. It answers four questions
for one sector and one venue:

  A. How far is the venue's mid from Pinnacle's devigged fair price at each
     time-to-tip horizon, and how far is each from its own close?
  B. When a gap exists, which side closes it by the next snapshot?
  C. Does one venue's move predict the other's NEXT move (lead-lag)?
  D. Is there a directional drift from the first snapshot to the close
     (do favorites shorten)?

Pairing rules (each one prevents a known bad join):

  * A venue snapshot pairs with the NEAREST Pinnacle snapshot of the same game
    within ``pair_tol_min`` minutes. Scan sessions write both books seconds
    apart; watch-closes / watch-listings sessions write only the venue.
  * The game is matched on the venue market's ``yes_team`` and the game date.
    ET-dated sectors (``uses_et_game_day``) need the SAME ET day: a ±1 window
    pairs the next game of a series with today's line (the 2026-09-23 MLB
    cross-match). Other sectors accept ±1 day, exact day first.
  * Only strictly pre-tip snapshots are used. Tip = the latest Pinnacle
    ``event_date`` seen for the game.
  * A venue quote is dropped when its ask is outside (0.02, 0.98) or its
    bid/ask spread is negative or wider than ``max_spread``: settled, empty
    and placeholder books are not prices.

Limits: moneyline only (spread/total rungs need line matching), and the
resolution is the archive's snapshot cadence — a venue lag of minutes after a
Pinnacle move is invisible when snapshots are hours apart.
"""

from __future__ import annotations

import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")

# (lo_hours, hi_hours, label), far to near.
BUCKETS: tuple[tuple[float, float, str], ...] = (
    (48.0, math.inf, "48h+"),
    (24.0, 48.0, "24-48h"),
    (12.0, 24.0, "12-24h"),
    (6.0, 12.0, "6-12h"),
    (3.0, 6.0, "3-6h"),
    (1.0, 3.0, "1-3h"),
    (0.0, 1.0, "<1h"),
)

# A game's last snapshot must be this close to tip to count as its "close".
CLOSE_MAX_LEAD_H = 1.5
# Consecutive snapshots closer than this are the same observation twice.
MIN_INTERVAL_H = 0.25
# Consecutive snapshots further apart than this are not one "next move".
MAX_INTERVAL_H = 12.0
# Drift needs a first snapshot at least this far from tip.
DRIFT_MIN_FIRST_LEAD_H = 6.0


@dataclass(frozen=True)
class Point:
    """One paired observation of a game. Probabilities are for ``team``."""

    hours_to_tip: float
    pinnacle: float      # devigged Pinnacle fair probability
    venue_mid: float     # (ask + bid) / 2
    venue_spread: float  # ask - bid


@dataclass(frozen=True)
class GameSeries:
    event_id: str
    team: str            # the side every probability refers to
    opponent: str
    tip: datetime
    points: tuple[Point, ...]  # ordered far -> near


def _ts(raw: str) -> datetime:
    d = datetime.fromisoformat(raw)
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _game_days(tip: datetime, sector: str) -> list[str]:
    """Venue game dates that may carry this game, most trusted first."""
    from evmax.clients.time_util import uses_et_game_day

    if uses_et_game_day(sector):
        return [tip.astimezone(_ET).date().isoformat()]
    base = tip.astimezone(timezone.utc).date()
    return [(base + timedelta(days=d)).isoformat() for d in (0, -1, 1)]


def load_series(
    conn: sqlite3.Connection,
    sector: str,
    *,
    venue: str = "kalshi",
    since: Optional[str] = None,
    until: Optional[str] = None,
    pair_tol_min: float = 20.0,
    max_spread: float = 0.10,
    min_points: int = 3,
    normalize: Optional[Callable[[str], str]] = None,
) -> list[GameSeries]:
    """Build one paired Pinnacle/venue moneyline series per game."""
    if normalize is None:
        from evmax.matching.normalizer import NameNormalizer

        normalize = NameNormalizer(sector).normalize

    where, params = ["sector = ?"], [sector]
    if since:
        where.append("fetched_at >= ?")
        params.append(since)
    if until:
        where.append("fetched_at < ?")
        params.append(until)
    clause = " AND ".join(where)

    # game -> [(fetched, tip, label_a, label_b, prob_a)]
    pin: dict[str, list[tuple]] = defaultdict(list)
    for eid, ed, fa, la, lb, pa in conn.execute(
        f"""SELECT event_id, event_date, fetched_at, outcome_a_label,
                   outcome_b_label, true_prob_a
            FROM archived_sharp_odds
            WHERE {clause} AND book = 'pinnacle'
              AND spread_line IS NULL AND total_line IS NULL
              AND prop_player_name IS NULL
              AND event_id NOT LIKE '%::spread%' AND event_id NOT LIKE '%::total%'
              AND event_id NOT LIKE '%::prop::%' AND event_id NOT LIKE '%::advance%'
              AND event_date IS NOT NULL AND outcome_a_label IS NOT NULL
              AND outcome_b_label IS NOT NULL""",
        params,
    ):
        pin[eid].append((_ts(fa), _ts(ed), la, lb, pa))

    pm = "ticker LIKE 'polymarket_us:%'"
    venue_clause = pm if venue == "polymarket_us" else f"NOT ({pm})"
    # (game date, team) -> [(fetched, ask, bid)]
    ven: dict[tuple[str, str], list[tuple]] = defaultdict(list)
    for ed, fa, yt, yp, npx in conn.execute(
        f"""SELECT event_date, fetched_at, yes_team, yes_price, no_price
            FROM archived_kalshi_markets
            WHERE {clause} AND market_type = 'moneyline' AND {venue_clause}
              AND yes_team IS NOT NULL AND event_date IS NOT NULL""",
        params,
    ):
        ask, bid = yp, 1.0 - npx
        if not (0.02 < ask < 0.98) or not (0.0 <= ask - bid <= max_spread):
            continue
        ven[(ed[:10], normalize(yt))].append((_ts(fa), ask, bid))

    tol = timedelta(minutes=pair_tol_min)
    out: list[GameSeries] = []
    for eid, snaps in pin.items():
        snaps.sort(key=lambda s: s[0])
        tip = max(s[1] for s in snaps)
        team, opp = normalize(snaps[0][2]), normalize(snaps[0][3])
        if not team or not opp or team == opp:
            continue
        quotes, flip = None, False
        for day in _game_days(tip, sector):
            if (day, team) in ven:
                quotes = ven[(day, team)]
                break
            if (day, opp) in ven:
                # Only the opponent's contract is archived: read its complement.
                quotes, flip = ven[(day, opp)], True
                break
        if not quotes:
            continue
        pts: list[Point] = []
        for t, ask, bid in sorted(quotes):
            if t >= tip:
                continue
            near = min(snaps, key=lambda s: abs(s[0] - t))
            if abs(near[0] - t) > tol or near[0] >= tip:
                continue
            p = near[4] if normalize(near[2]) == team else 1.0 - near[4]
            mid = (ask + bid) / 2.0
            pts.append(Point(
                hours_to_tip=(tip - t).total_seconds() / 3600.0,
                pinnacle=p,
                venue_mid=(1.0 - mid) if flip else mid,
                venue_spread=ask - bid,
            ))
        if len(pts) >= min_points:
            pts.sort(key=lambda x: -x.hours_to_tip)
            out.append(GameSeries(eid, team, opp, tip, tuple(pts)))
    out.sort(key=lambda g: g.tip)
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def clustered_slope(
    xs: list[float], ys: list[float], groups: list[str]
) -> Optional[dict]:
    """OLS slope of y on x with a game-clustered standard error.

    Consecutive intervals of one game share its news, so they are not
    independent observations; the cluster is the game.
    """
    n = len(xs)
    if n < 3 or len(set(groups)) < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    score: dict[str, float] = defaultdict(float)
    for x, y, g in zip(xs, ys, groups):
        score[g] += (x - mx) * (y - my - slope * (x - mx))
    se = math.sqrt(sum(v * v for v in score.values())) / sxx
    return {
        "slope": slope,
        "t": (slope / se) if se > 0 else None,
        "n": n,
        "games": len(set(groups)),
    }


def gap_by_bucket(series: list[GameSeries]) -> list[dict]:
    """Per time-to-tip bucket: venue-vs-Pinnacle gap and distance to close.

    One value per game per bucket (its LAST snapshot in the bucket), and only
    games whose final snapshot is a usable close (within CLOSE_MAX_LEAD_H).
    """
    rows = []
    for lo, hi, label in BUCKETS:
        vals = []
        for g in series:
            close = g.points[-1]
            if close.hours_to_tip >= CLOSE_MAX_LEAD_H:
                continue
            inside = [p for p in g.points if lo <= p.hours_to_tip < hi]
            if not inside:
                continue
            x = inside[-1]
            vals.append((
                abs(x.venue_mid - x.pinnacle),
                x.venue_mid - x.pinnacle,
                x.venue_spread,
                abs(x.pinnacle - close.pinnacle),
                abs(x.venue_mid - close.venue_mid),
            ))
        if not vals:
            continue
        n = len(vals)
        mean = lambda i: sum(v[i] for v in vals) / n * 100.0  # noqa: E731
        rows.append({
            "bucket": label,
            "games": n,
            "abs_gap_pp": mean(0),
            "signed_gap_pp": mean(1),
            "venue_spread_pp": mean(2),
            "pinnacle_to_close_pp": mean(3),
            "venue_to_close_pp": mean(4),
        })
    return rows


def _intervals(g: GameSeries):
    """Consecutive snapshot pairs that count as one 'next move'."""
    pts = g.points
    for a, b in zip(pts, pts[1:]):
        dt = a.hours_to_tip - b.hours_to_tip
        if MIN_INTERVAL_H <= dt <= MAX_INTERVAL_H:
            yield a, b


def gap_closure(
    series: list[GameSeries], lo: float = 0.0, hi: float = math.inf
) -> Optional[dict]:
    """Share of the current gap each side closes by the next snapshot.

    gap = Pinnacle fair − venue mid. ``venue_toward_pinnacle`` is the slope of
    the venue's next move on the gap; ``pinnacle_toward_venue`` is minus the
    slope of Pinnacle's next move on it. Both are positive when the two prices
    converge. Quote noise (a one-cent tick) mean-reverts on its own, so two
    similar positive slopes mean "no clear follower", not "both follow".
    """
    gx, dv, dp, gg = [], [], [], []
    for g in series:
        for a, b in _intervals(g):
            if not (lo <= a.hours_to_tip < hi):
                continue
            gx.append(a.pinnacle - a.venue_mid)
            dv.append(b.venue_mid - a.venue_mid)
            dp.append(b.pinnacle - a.pinnacle)
            gg.append(g.event_id)
    v, p = clustered_slope(gx, dv, gg), clustered_slope(gx, dp, gg)
    if v is None or p is None:
        return None
    return {
        "n": v["n"],
        "games": v["games"],
        "venue_toward_pinnacle": v["slope"],
        "venue_t": v["t"],
        "pinnacle_toward_venue": -p["slope"],
        "pinnacle_t": (-p["t"]) if p["t"] is not None else None,
    }


def lead_lag(series: list[GameSeries]) -> Optional[dict]:
    """Does one book's move predict the other's move in the NEXT interval?"""
    x_p, y_v, x_v, y_p, gg = [], [], [], [], []
    for g in series:
        iv = list(_intervals(g))
        for (a, b), (c, d) in zip(iv, iv[1:]):
            if b is not c:  # the two intervals must be adjacent
                continue
            x_p.append(b.pinnacle - a.pinnacle)
            y_v.append(d.venue_mid - c.venue_mid)
            x_v.append(b.venue_mid - a.venue_mid)
            y_p.append(d.pinnacle - c.pinnacle)
            gg.append(g.event_id)
    pv, vp = clustered_slope(x_p, y_v, gg), clustered_slope(x_v, y_p, gg)
    if pv is None or vp is None:
        return None
    return {"pinnacle_leads_venue": pv, "venue_leads_pinnacle": vp}


def drift(series: list[GameSeries], top: int = 6) -> Optional[dict]:
    """First-snapshot-to-close drift, signed toward the opening favorite."""
    fav, moves, big = [], [], []
    for g in series:
        first, close = g.points[0], g.points[-1]
        if (close.hours_to_tip >= CLOSE_MAX_LEAD_H
                or first.hours_to_tip < DRIFT_MIN_FIRST_LEAD_H):
            continue
        d = close.pinnacle - first.pinnacle
        fav.append(d if first.pinnacle >= 0.5 else -d)
        moves.append(abs(d))
        big.append({
            "event_id": g.event_id,
            "team": g.team,
            "opponent": g.opponent,
            "pinnacle_first": first.pinnacle,
            "pinnacle_close": close.pinnacle,
            "venue_first": first.venue_mid,
            "venue_close": close.venue_mid,
            "first_lead_h": first.hours_to_tip,
        })
    n = len(fav)
    if n < 2:
        return None
    mean = sum(fav) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in fav) / (n - 1))
    big.sort(key=lambda r: -abs(r["pinnacle_close"] - r["pinnacle_first"]))
    return {
        "games": n,
        "mean_abs_move_pp": sum(moves) / n * 100.0,
        "median_abs_move_pp": sorted(moves)[n // 2] * 100.0,
        "moves_ge_3pp": sum(1 for v in moves if v >= 0.03),
        "favorite_drift_pp": mean * 100.0,
        "favorite_drift_z": (mean / (sd / math.sqrt(n))) if sd > 0 else None,
        "favorite_shortened_pct": sum(1 for v in fav if v > 0) / n * 100.0,
        "largest": big[:top],
    }


def analyze(
    conn: sqlite3.Connection,
    sector: str,
    *,
    venue: str = "kalshi",
    since: Optional[str] = None,
    until: Optional[str] = None,
    pair_tol_min: float = 20.0,
    top: int = 6,
    normalize: Optional[Callable[[str], str]] = None,
) -> dict:
    """Run every lens and return a JSON-serializable report."""
    series = load_series(
        conn, sector, venue=venue, since=since, until=until,
        pair_tol_min=pair_tol_min, normalize=normalize,
    )
    counts = sorted(len(g.points) for g in series)
    return {
        "sector": sector,
        "venue": venue,
        "since": since,
        "until": until,
        "games": len(series),
        "snapshots": sum(counts),
        "median_snapshots_per_game": counts[len(counts) // 2] if counts else 0,
        "buckets": gap_by_bucket(series),
        "closure": {
            "all": gap_closure(series),
            "over_6h": gap_closure(series, 6.0, math.inf),
            "under_6h": gap_closure(series, 0.0, 6.0),
        },
        "lead_lag": lead_lag(series),
        "drift": drift(series, top=top),
    }
