"""Offline venue-lag lens for game-winner (moneyline) markets.

Pre-registration: docs/nfl-ml-lag-prereg.md. The promotion gate only counts
rows the scanner flagged at EV>=2%, which for a sharp-passthrough moneyline is
a handful a week. This lens scores EVERY matched (game, team) contract instead:
enter at the first sweep with a concurrent Pinnacle anchor, mark to the last
pre-tip snapshot, and ask whether the Kalshi ask converges toward the devigged
Pinnacle price (CLV regressed on entry edge) and whether that survives the fee.

Pure read-only diagnostics over archive.db: never writes, cannot touch the
shadow stream.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from math import sqrt
from statistics import mean, median
from typing import Optional

from evmax.fees import kalshi_fee_prob
from evmax.sectors.registry import get_handler

ANCHOR_MAX_STALE_H = 2.0
ANCHOR_FORWARD_SKEW_H = 0.25
MIN_ENTRY_LEAD_H = 0.5
MIN_HOLD_H = 1.0
MAX_CLOSE_LEAD_H = 3.0
PLACEHOLDER_LO, PLACEHOLDER_HI = 0.01, 0.99
FLAG_EDGE_PP = 2.0
FLAG_DEPTH_USD = 50.0
SLOPE_Z_GATE = 1.64
MIN_GAMES_SUPPORTED = 30
MIN_GAMES_REFUTED = 60

_GameKey = tuple[str, frozenset[str]]


@dataclass
class Unit:
    """One (game, team YES contract) hypothetical entry."""

    ticker: str
    game: _GameKey
    entry_at: datetime
    entry_ask: float
    fair: float
    edge_pp: float
    depth_usd: Optional[float]
    tip_at: datetime
    entry_lead_h: float
    close_ask: Optional[float] = None
    close_lead_h: Optional[float] = None
    gross_clv_pp: Optional[float] = None
    net_clv_pp: Optional[float] = None


@dataclass
class LagStats:
    tickers: int = 0
    never_anchored: int = 0
    side_unresolved: int = 0
    placeholder_entry: int = 0
    units: list[Unit] = field(default_factory=list)


def _ts(raw: str) -> datetime:
    return datetime.fromisoformat(raw)


def load_ml_anchors(conn, sector: str) -> dict[_GameKey, dict]:
    """Per-game Pinnacle moneyline series: {game: {"tip", "series": [(t, {team: p})]}}."""
    games: dict[_GameKey, dict] = {}
    rows = conn.execute(
        """SELECT fetched_at, event_id, true_prob_a, true_prob_b, event_date
           FROM archived_sharp_odds
           WHERE sector = ? AND event_date IS NOT NULL
             AND event_id NOT LIKE '%::%::%::%'
           ORDER BY fetched_at""",
        (sector,),
    ).fetchall()
    for r in rows:
        parts = r["event_id"].split("::")
        if len(parts) != 3 or "_vs_" not in parts[2]:
            continue
        a, _, b = parts[2].partition("_vs_")
        key = (parts[1], frozenset((a, b)))
        g = games.setdefault(key, {"tip": None, "series": []})
        tip = _ts(r["event_date"])
        if g["tip"] is None or tip < g["tip"]:
            g["tip"] = tip
        g["series"].append((_ts(r["fetched_at"]), {a: r["true_prob_a"], b: r["true_prob_b"]}))
    return games


def _anchor_at(series: list, snap_t: datetime) -> Optional[dict]:
    best = None
    for t, probs in series:
        age_h = (snap_t - t).total_seconds() / 3600
        if -ANCHOR_FORWARD_SKEW_H <= age_h <= ANCHOR_MAX_STALE_H:
            if best is None or abs(age_h) < best[0]:
                best = (abs(age_h), probs)
    return best[1] if best else None


def _valid_ask(p) -> bool:
    return p is not None and PLACEHOLDER_LO < p < PLACEHOLDER_HI


def evaluate_moneyline(
    conn,
    sector: str,
    ticker_prefix: str,
    since: Optional[str] = None,
    session_prefixes: tuple[str, ...] = ("watchlist", "watchclos"),
) -> LagStats:
    handler = get_handler(sector)
    norm = lambda s: handler.normalize_team(s or "")  # noqa: E731
    anchors = load_ml_anchors(conn, sector)

    prefix_clause = " OR ".join("session_id LIKE ?" for _ in session_prefixes)
    params: list = [sector, f"{ticker_prefix}%", *[f"{p}%" for p in session_prefixes]]
    since_clause = ""
    if since:
        since_clause = " AND fetched_at >= ?"
        params.append(since)
    snaps = conn.execute(
        f"""SELECT session_id, fetched_at, ticker, yes_price, yes_team,
                   team_home, team_away, event_date
            FROM archived_kalshi_markets
            WHERE sector = ? AND market_type = 'moneyline'
              AND ticker LIKE ? AND ({prefix_clause}){since_clause}
            ORDER BY ticker, fetched_at""",
        params,
    ).fetchall()
    depth = {
        (r["session_id"], r["ticker"]): r["yes_ask_depth_usd"]
        for r in conn.execute(
            "SELECT session_id, ticker, yes_ask_depth_usd FROM archived_orderbook_depth WHERE sector = ?",
            (sector,),
        )
    }

    trails: dict[str, list] = defaultdict(list)
    for r in snaps:
        trails[r["ticker"]].append(r)

    stats = LagStats()
    for ticker, trail in trails.items():
        first = trail[0]
        home, away = norm(first["team_home"]), norm(first["team_away"])
        yes = norm(first["yes_team"])
        if not home or not away or not first["event_date"]:
            continue
        stats.tickers += 1
        game = (first["event_date"][:10], frozenset((home, away)))
        g = anchors.get(game)
        if g is None:
            stats.never_anchored += 1
            continue
        if yes not in game[1]:
            stats.side_unresolved += 1
            continue
        tip = g["tip"]
        pre = [r for r in trail if _ts(r["fetched_at"]) <= tip]

        entry = None
        for snap in pre:
            t = _ts(snap["fetched_at"])
            lead_h = (tip - t).total_seconds() / 3600
            if lead_h < MIN_ENTRY_LEAD_H:
                continue
            probs = _anchor_at(g["series"], t)
            if probs is None or yes not in probs:
                continue
            if not _valid_ask(snap["yes_price"]):
                stats.placeholder_entry += 1
                continue
            entry = (snap, t, lead_h, probs[yes])
            break
        if entry is None:
            stats.never_anchored += 1
            continue

        snap, t, lead_h, fair = entry
        ask = snap["yes_price"]
        u = Unit(
            ticker=ticker, game=game, entry_at=t, entry_ask=ask, fair=fair,
            edge_pp=(fair - ask) * 100, depth_usd=depth.get((snap["session_id"], ticker)),
            tip_at=tip, entry_lead_h=lead_h,
        )
        for later in reversed(pre):
            lt = _ts(later["fetched_at"])
            close_lead = (tip - lt).total_seconds() / 3600
            if close_lead > MAX_CLOSE_LEAD_H:
                break  # earlier snapshots are only staler
            if (lt - t).total_seconds() / 3600 < MIN_HOLD_H:
                break
            if not _valid_ask(later["yes_price"]):
                continue
            u.close_ask = later["yes_price"]
            u.close_lead_h = close_lead
            u.gross_clv_pp = (u.close_ask - ask) * 100
            u.net_clv_pp = u.gross_clv_pp - kalshi_fee_prob(ask) * 100
            break
        stats.units.append(u)
    return stats


# ---------------------------------------------------------------- statistics


def cluster_slope(xs: list[float], ys: list[float], groups: list) -> Optional[tuple[float, float, float, int]]:
    """OLS y = a + b*x with game-clustered (CR0) SE on b. Returns (b, se, z, n_clusters)."""
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    score: dict = defaultdict(float)
    for x, y, g in zip(xs, ys, groups):
        score[g] += (x - mx) * (y - a - b * x)
    se = sqrt(sum(u * u for u in score.values())) / sxx
    if se <= 0:
        return None
    return b, se, b / se, len(score)


def game_means(units: list[Unit], attr: str) -> list[float]:
    per: dict = defaultdict(list)
    for u in units:
        v = getattr(u, attr)
        if v is not None:
            per[u.game].append(v)
    return [mean(v) for v in per.values()]


def mean_t(vals: list[float]) -> Optional[tuple[float, float, float]]:
    """(mean, t, pct_positive) over independent values (one per game)."""
    n = len(vals)
    if n < 2:
        return None
    m = mean(vals)
    sd = sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
    t = m / (sd / sqrt(n)) if sd > 0 else 0.0
    return m, t, 100 * sum(1 for v in vals if v > 0) / n


@dataclass
class LagVerdict:
    verdict: str
    n_units: int
    n_games: int
    slope: Optional[tuple[float, float, float, int]]
    flagged_net: Optional[tuple[float, float, float]]
    flagged_games: int


def verdict(stats: LagStats) -> LagVerdict:
    scored = [u for u in stats.units if u.gross_clv_pp is not None]
    games = {u.game for u in scored}
    slope = cluster_slope(
        [u.edge_pp for u in scored], [u.gross_clv_pp for u in scored], [u.game for u in scored]
    )
    flagged = [u for u in scored if u.edge_pp >= FLAG_EDGE_PP]
    fl_means = game_means(flagged, "net_clv_pp")
    fl = mean_t(fl_means)
    n_games = len(games)

    slope_ok = slope is not None and slope[0] > 0 and slope[2] >= SLOPE_Z_GATE
    if slope_ok:
        if fl is not None and len(fl_means) >= MIN_GAMES_SUPPORTED and fl[0] >= 0 and fl[1] >= SLOPE_Z_GATE:
            v = "SUPPORTED"
        elif fl is not None and fl[0] < 0:
            v = "LAG-BUT-FEE-NEGATIVE"
        else:
            v = "UNDERPOWERED"
    elif slope is not None and n_games >= MIN_GAMES_REFUTED:
        v = "REFUTED"
    else:
        v = "UNDERPOWERED"
    return LagVerdict(v, len(scored), n_games, slope, fl, len(fl_means))


EDGE_BUCKETS = [(-99, -2), (-2, 0), (0, 1), (1, 2), (2, 99)]


def bucket_rows(stats: LagStats) -> list[tuple[str, int, int, Optional[float], Optional[float]]]:
    out = []
    scored = [u for u in stats.units if u.gross_clv_pp is not None]
    for lo, hi in EDGE_BUCKETS:
        us = [u for u in scored if lo <= u.edge_pp < hi]
        label = f"[{lo:g},{hi:g})".replace("-99", "-inf").replace("99", "inf")
        out.append((
            label, len(us), len({u.game for u in us}),
            mean(u.gross_clv_pp for u in us) if us else None,
            mean(u.net_clv_pp for u in us) if us else None,
        ))
    return out


def lead_summary(stats: LagStats) -> tuple[Optional[float], Optional[float]]:
    sc = [u for u in stats.units if u.gross_clv_pp is not None]
    if not sc:
        return None, None
    return median(u.entry_lead_h for u in sc), median(u.close_lead_h for u in sc)
