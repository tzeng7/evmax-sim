"""Offline anchored-entry evaluator over archive.db watch-listings captures.

Replays the hourly listing-window capture and asks: if we had entered each
laddered market at its FIRST ANCHORED SWEEP — the first watch-listings snapshot
with a concurrent devigged-Pinnacle anchor that can price the ticker's line —
gated on edge >= ev_min at the crossable price AND resting depth >= depth_min,
what Kalshi CLV would those entries have earned by the last pre-tip snapshot?

Why "first anchored" and not "at listing": Kalshi lists WNBA/NBA/MLB ladders
~T-28-48h but Pinnacle only posts ~T-17-24h, so the listing window opens with
no sharp anchor — EV is uncomputable there and the observed price moves are MM
placeholder-spread tightening (crossing either side pays ~half the spread).
The first sweep with an anchor is the earliest moment edge is measurable.

Both sides of every ticker are evaluated independently:
  YES side (spread 'lay' / total 'over'):  cross the ask;  CLV = ask_close - ask_entry
  NO  side (spread 'take' / total 'under'): cross 1 - yes_bid; CLV = bid_entry - bid_close

Pure read-only diagnostics — never writes, cannot contaminate the shadow stream.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean, median
from typing import Optional

from evmax.models.odds import SharpBook, SharpOdds
from evmax.models_ml.spread_distribution import SpreadDistributionModel
from evmax.sectors.registry import get_handler

# An anchor is "concurrent" with a Kalshi snapshot when it was fetched at most
# this many hours BEFORE the snapshot (or within the same sweep — small forward
# skew tolerated because a sweep fetches Kalshi and Pinnacle seconds apart).
ANCHOR_MAX_STALE_H = 2.0
ANCHOR_FORWARD_SKEW_H = 0.25

# Totals anchor requires the archived Pinnacle ladder to carry (nearly) the
# exact Kalshi line — half-point neighbours are a different market.
TOTAL_LINE_TOLERANCE = 0.51

_GameKey = tuple[str, frozenset[str]]


@dataclass
class AnchoredEntry:
    """One hypothetical entry produced by the first-anchored-sweep rule."""

    ticker: str
    sector: str
    market_type: str
    side: str                  # spread: lay/take · total: over/under
    line: float
    event_label: str           # "storm vs mercury"
    outcome_label: str         # e.g. "storm -6.5" / "Under 179.5"
    entry_at: datetime
    entry_price: float         # crossable price actually paid
    entry_depth_usd: float
    fair_prob: float           # sharp fair value of the side bought
    edge_pp: float             # (fair - entry_price) * 100
    tip_at: datetime
    entry_lead_h: float
    close_at: Optional[datetime] = None
    close_price: Optional[float] = None
    clv_pp: Optional[float] = None   # None = close-side price not captured


@dataclass
class EvalStats:
    """Coverage counters so silent drops are visible in the report."""

    tickers: int = 0
    never_anchored: int = 0
    anchored_no_entry: int = 0
    entries: list[AnchoredEntry] = field(default_factory=list)


def _parse_ts(raw: str) -> datetime:
    return datetime.fromisoformat(raw)


def _sharp_game_key(event_id: str) -> Optional[_GameKey]:
    parts = event_id.split("::")
    if len(parts) < 3 or "_vs_" not in parts[2]:
        return None
    a, _, b = parts[2].partition("_vs_")
    return (parts[1], frozenset((a, b)))


def load_anchors(conn, sector: str) -> dict[_GameKey, dict]:
    """Per-game Pinnacle anchor time series from archived_sharp_odds.

    Returns {game_key: {"tip": datetime,
                        "spread": [(t, SharpOdds), ...]        # primary line
                        "totals": [(t, {line: (p_over, p_under)}), ...]}}
    """
    games: dict[_GameKey, dict] = {}
    rows = conn.execute(
        """SELECT fetched_at, event_id, outcome_a_label, outcome_b_label,
                  true_prob_a, true_prob_b, spread_line, total_line,
                  true_prob_over, true_prob_under, event_date,
                  outcome_a_decimal, outcome_b_decimal
           FROM archived_sharp_odds
           WHERE sector = ? AND event_date IS NOT NULL
           ORDER BY fetched_at""",
        (sector,),
    ).fetchall()
    total_acc: dict[tuple[_GameKey, str], dict] = defaultdict(dict)
    for r in rows:
        key = _sharp_game_key(r["event_id"])
        if key is None:
            continue
        g = games.setdefault(key, {"tip": None, "spread": [], "totals": []})
        tip = _parse_ts(r["event_date"])
        if g["tip"] is None or tip < g["tip"]:
            g["tip"] = tip
        t = _parse_ts(r["fetched_at"])
        if "::total::" in r["event_id"] and r["total_line"] is not None:
            if r["true_prob_over"] is not None:
                # ladders arrive as many rows per sweep — bucket by fetch time
                bucket = total_acc[(key, r["fetched_at"][:16])]
                bucket.setdefault("_t", t)
                bucket[r["total_line"]] = (r["true_prob_over"], r["true_prob_under"])
        elif r["event_id"].endswith("::spread") and r["spread_line"] is not None:
            so = SharpOdds(
                event_id=r["event_id"],
                book=SharpBook.pinnacle,
                sector=sector,
                outcome_a_label=r["outcome_a_label"] or "",
                outcome_b_label=r["outcome_b_label"] or "",
                outcome_a_decimal=r["outcome_a_decimal"] or 2.0,
                outcome_b_decimal=r["outcome_b_decimal"] or 2.0,
                true_prob_a=r["true_prob_a"],
                true_prob_b=r["true_prob_b"],
                spread_line=r["spread_line"],
            )
            g["spread"].append((t, so))
    for (key, _), bucket in total_acc.items():
        t = bucket.pop("_t")
        games[key]["totals"].append((t, bucket))
    for g in games.values():
        g["totals"].sort(key=lambda x: x[0])
    return games


def _nearest_anchor(series: list, snap_t: datetime):
    """Latest anchor within [snap_t - ANCHOR_MAX_STALE_H, snap_t + skew]."""
    best = None
    for t, payload in series:
        age_h = (snap_t - t).total_seconds() / 3600
        if -ANCHOR_FORWARD_SKEW_H <= age_h <= ANCHOR_MAX_STALE_H:
            if best is None or abs(age_h) < best[0]:
                best = (abs(age_h), payload)
    return best[1] if best else None


def _fair_yes_spread(
    so: SharpOdds, sector: str, yes_nick: str, line: float, normalize
) -> Optional[float]:
    """P(yes_nick covers Kalshi `line`) via the live alt-line pricing model."""
    nick_a = normalize(so.outcome_a_label)
    nick_b = normalize(so.outcome_b_label)
    if yes_nick == nick_a:
        yes_is_underdog = False
    elif yes_nick == nick_b:
        yes_is_underdog = True
    else:
        return None
    pred = SpreadDistributionModel().predict(
        so, target_line=line, sector=sector, yes_is_underdog=yes_is_underdog
    )
    return pred.true_prob if pred else None


def _fair_yes_total(ladder: dict, yes_side: str, line: float) -> Optional[float]:
    if not ladder:
        return None
    close_line, probs = min(ladder.items(), key=lambda kv: abs(kv[0] - line))
    if abs(close_line - line) > TOTAL_LINE_TOLERANCE:
        return None
    return probs[0] if yes_side == "over" else probs[1]


def evaluate_sector(
    conn,
    sector: str,
    ev_min_pp: float = 2.0,
    depth_min_usd: float = 50.0,
    market_types: tuple[str, ...] = ("spread", "total"),
    since: Optional[str] = None,
) -> EvalStats:
    """Run the first-anchored-sweep entry rule over one sector's captures."""
    handler = get_handler(sector)
    normalize = lambda s: handler.normalize_team(s or "")  # noqa: E731
    anchors = load_anchors(conn, sector)

    placeholders = ",".join("?" * len(market_types))
    params: list = [sector, *market_types]
    since_clause = ""
    if since:
        since_clause = " AND fetched_at >= ?"
        params.append(since)
    snaps = conn.execute(
        f"""SELECT session_id, fetched_at, ticker, market_type, yes_price, line,
                   yes_team, team_home, team_away, event_date
            FROM archived_kalshi_markets
            WHERE sector = ? AND market_type IN ({placeholders})
              AND session_id LIKE 'watchlist%' AND line IS NOT NULL{since_clause}
            ORDER BY ticker, fetched_at""",
        params,
    ).fetchall()
    depth = {
        (r["session_id"], r["ticker"]): r
        for r in conn.execute(
            """SELECT session_id, ticker, yes_ask, yes_ask_depth_usd,
                      yes_bid, yes_bid_depth_usd
               FROM archived_orderbook_depth WHERE sector = ?""",
            (sector,),
        )
    }

    trails: dict[str, list] = defaultdict(list)
    for r in snaps:
        trails[r["ticker"]].append(r)

    stats = EvalStats()
    for ticker, trail in trails.items():
        first = trail[0]
        home = normalize(first["team_home"])
        away = normalize(first["team_away"])
        if not home or not away or not first["event_date"]:
            continue
        game = (first["event_date"][:10], frozenset((home, away)))
        g = anchors.get(game)
        stats.tickers += 1
        if g is None:
            stats.never_anchored += 1
            continue
        tip = g["tip"]
        pre = [r for r in trail if _parse_ts(r["fetched_at"]) <= tip]
        if not pre:
            stats.never_anchored += 1
            continue

        mt = first["market_type"]
        yes_team = (first["yes_team"] or "").strip().lower()
        line = first["line"]
        event_label = f"{away} vs {home}"

        def fair_at(snap_t: datetime) -> Optional[float]:
            if mt == "spread":
                so = _nearest_anchor(g["spread"], snap_t)
                if so is None:
                    return None
                return _fair_yes_spread(so, sector, yes_team, line, normalize)
            ladder = _nearest_anchor(g["totals"], snap_t)
            if ladder is None:
                return None
            return _fair_yes_total(ladder, yes_team, line)

        close = pre[-1]
        close_t = _parse_ts(close["fetched_at"])
        close_d = depth.get((close["session_id"], ticker))
        close_ask = close_d["yes_ask"] if close_d and close_d["yes_ask"] is not None else close["yes_price"]
        close_bid = close_d["yes_bid"] if close_d else None

        anchored_any = False
        found_yes = found_no = False
        for snap in pre:
            if found_yes and found_no:
                break
            snap_t = _parse_ts(snap["fetched_at"])
            fair = fair_at(snap_t)
            if fair is None:
                continue
            anchored_any = True
            d = depth.get((snap["session_id"], ticker))
            ask = d["yes_ask"] if d and d["yes_ask"] is not None else snap["yes_price"]
            bid = d["yes_bid"] if d else None

            if not found_yes and ask is not None and 0 < ask < 1:
                edge = (fair - ask) * 100
                depth_usd = (d["yes_ask_depth_usd"] if d else None) or 0.0
                if edge >= ev_min_pp and depth_usd >= depth_min_usd:
                    side = "lay" if mt == "spread" else yes_team  # over/under
                    outcome = (
                        f"{yes_team} {line:+g}" if mt == "spread"
                        else f"{yes_team.capitalize()} {line:g}"
                    )
                    stats.entries.append(AnchoredEntry(
                        ticker=ticker, sector=sector, market_type=mt, side=side,
                        line=line, event_label=event_label, outcome_label=outcome,
                        entry_at=snap_t, entry_price=ask, entry_depth_usd=depth_usd,
                        fair_prob=fair, edge_pp=edge, tip_at=tip,
                        entry_lead_h=(tip - snap_t).total_seconds() / 3600,
                        close_at=close_t, close_price=close_ask,
                        clv_pp=(close_ask - ask) * 100 if close_ask is not None else None,
                    ))
                    found_yes = True

            if not found_no and bid is not None and 0 < bid < 1:
                no_price = 1.0 - bid
                edge = (bid - fair) * 100   # = fair_no - no_price
                depth_usd = (d["yes_bid_depth_usd"] if d else None) or 0.0
                if edge >= ev_min_pp and depth_usd >= depth_min_usd:
                    side = "take" if mt == "spread" else ("under" if yes_team == "over" else "over")
                    opp = next(iter(frozenset((home, away)) - {yes_team}), "")
                    outcome = (
                        f"{opp} +{abs(line):g}" if mt == "spread"
                        else f"{side.capitalize()} {line:g}"
                    )
                    clv = (bid - close_bid) * 100 if close_bid is not None else None
                    stats.entries.append(AnchoredEntry(
                        ticker=ticker, sector=sector, market_type=mt, side=side,
                        line=line, event_label=event_label, outcome_label=outcome,
                        entry_at=snap_t, entry_price=no_price, entry_depth_usd=depth_usd,
                        fair_prob=1.0 - fair, edge_pp=edge, tip_at=tip,
                        entry_lead_h=(tip - snap_t).total_seconds() / 3600,
                        close_at=close_t,
                        close_price=(1.0 - close_bid) if close_bid is not None else None,
                        clv_pp=clv,
                    ))
                    found_no = True

        if anchored_any and not (found_yes or found_no):
            stats.anchored_no_entry += 1
        elif not anchored_any:
            stats.never_anchored += 1

    return stats


@dataclass
class SideSummary:
    market_type: str
    side: str
    n: int
    n_with_clv: int
    mean_edge_pp: float
    mean_clv_pp: Optional[float]
    median_clv_pp: Optional[float]
    pct_clv_pos: Optional[float]
    median_depth_usd: float
    median_lead_h: float


def summarize(stats: EvalStats) -> list[SideSummary]:
    by_side: dict[tuple[str, str], list[AnchoredEntry]] = defaultdict(list)
    for e in stats.entries:
        by_side[(e.market_type, e.side)].append(e)
    out = []
    for (mt, side), es in sorted(by_side.items()):
        clvs = [e.clv_pp for e in es if e.clv_pp is not None]
        out.append(SideSummary(
            market_type=mt,
            side=side,
            n=len(es),
            n_with_clv=len(clvs),
            mean_edge_pp=mean(e.edge_pp for e in es),
            mean_clv_pp=mean(clvs) if clvs else None,
            median_clv_pp=median(clvs) if clvs else None,
            pct_clv_pos=(100 * sum(1 for c in clvs if c > 0) / len(clvs)) if clvs else None,
            median_depth_usd=median(e.entry_depth_usd for e in es),
            median_lead_h=median(e.entry_lead_h for e in es),
        ))
    return out
