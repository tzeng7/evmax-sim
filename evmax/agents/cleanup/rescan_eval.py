"""Near-close re-scan replay: paired entry-CLV comparison over archived snapshots.

Answers the validation question behind the near-close re-scan workflow: for each
historical ``ev_predictions`` row, had we re-scanned inside the
``[T-window_lo, T-window_hi]`` pre-tip window and entered at the archived
near-tip crossable price — gated on a fresh as-of Pinnacle anchor still showing
edge >= ``ev_min_pp`` — how would Kalshi entry→close CLV compare to the actual
scan-time entry?

Design decisions (mirror the production CLV conventions in ``resolver.py``):

* **Paired scoring.** Both entries are scored against the SAME close — the last
  archived venue snapshot at or before tip that lies strictly after the rescan
  entry — so the delta is a paired measurement, never two different closes.
* **NO-side rows** (market_id ``:no`` suffix) follow the ``backfill_clv``
  convention: ``ev_predictions.kalshi_yes_price`` already stores the NO-side
  entry price, archived snapshots store the YES price and are flipped to
  ``1 - yes`` before use. The flipped rescan entry is an optimistic crossable
  proxy (true NO ask = 1 - yes_bid, we only have the YES ask) — the same
  approximation the production CLV backfill makes on the close side, applied
  consistently to entry and close so the paired delta is unaffected.
* **Venue-aware tickers** via ``resolver.close_lookup_ticker``: ``kalshi:``
  prefixes are stripped (the archive stores raw tickers), ``polymarket_us:``
  ids are kept (snapshots are archived under the full prefixed id).
* **Fresh-anchor gate.** The simulated rescan only "enters" when an archived
  Pinnacle anchor no older than ``anchor_max_stale_min`` exists at the rescan
  snapshot time AND prices the bet's side at ``>= ev_min_pp`` edge over the
  crossable price. Rows with a near-tip snapshot but no fresh anchor stay in
  the ungated comparison and are counted separately.

Pure read-only diagnostics — callers pass sqlite3 connections (open the real
DBs with URI ``mode=ro``). Never writes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import mean, median
from typing import Optional

from evmax.agents.cleanup.listings_eval import _fair_yes_spread, _fair_yes_total
from evmax.agents.cleanup.resolver import (
    close_lookup_ticker,
    clv_entry_price,
    yes_aligned_close_prob,
)
from evmax.models.odds import SharpBook, SharpOdds
from evmax.sectors.registry import get_handler

# Rescan entry window, minutes before tip. The window opens at T-75 (a rescan
# loop on a 15-min cadence reliably lands one pass in [T-75, T-60]) and closes
# at T-10 (inside that, quotes start absorbing lineup/in-play information).
WINDOW_LO_MIN = 75.0
WINDOW_HI_MIN = 10.0

# A Pinnacle anchor older than this at rescan time is stale — EV is
# uncomputable and the simulated rescan does not enter. Matches the 2h
# tolerance the listings-eval harness uses (hourly watch-listings sweeps).
ANCHOR_MAX_STALE_MIN = 120.0

EV_MIN_PP = 2.0


def _parse_ts(raw: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(raw)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def side_aligned(yes_price: float, is_no_side: bool) -> float:
    """Align an archived YES-market price to the bet's side of the book."""
    return 1.0 - yes_price if is_no_side else yes_price


def base_event_key(event_id: str) -> str:
    """Strip market-type suffixes back to ``sector::date::a_vs_b``.

    ``::advance`` is deliberately KEPT — advance markets have their own
    archived sharp rows under the suffixed id (see the World Cup advance
    notes) and must never cross-match the regulation 3-way.
    """
    for marker in ("::spread", "::total::"):
        idx = event_id.find(marker)
        if idx != -1:
            return event_id[:idx]
    return event_id


def pick_rescan_entry(
    trail: list[tuple[datetime, float]],
    tip: datetime,
    window_lo_min: float = WINDOW_LO_MIN,
    window_hi_min: float = WINDOW_HI_MIN,
) -> Optional[tuple[datetime, float]]:
    """First snapshot inside [tip - window_lo, tip - window_hi].

    "First" simulates a rescan daemon entering at its earliest pass through
    the window rather than cherry-picking the best price.
    """
    for t, price in trail:
        lead_min = (tip - t).total_seconds() / 60.0
        if window_hi_min <= lead_min <= window_lo_min:
            return (t, price)
    return None


def pick_close(
    trail: list[tuple[datetime, float]],
    tip: datetime,
    after: Optional[datetime] = None,
) -> Optional[tuple[datetime, float]]:
    """Last snapshot at or before tip, strictly after ``after`` when given.

    Kalshi never truly closes; the last archived pre-tip snapshot is the close
    proxy. Requiring ``t > after`` keeps CLV a forward measurement from the
    entry — a close that precedes the entry would be fabricated CLV.
    """
    best = None
    for t, price in trail:
        if t > tip:
            continue
        if after is not None and t <= after:
            continue
        if best is None or t > best[0]:
            best = (t, price)
    return best


@dataclass
class PairedComparison:
    """One ev_predictions row scored under both entry rules."""

    market_id: str
    sector: str
    market_type: str
    event_title: str
    outcome_label: str
    is_no_side: bool
    mode: str
    tip_at: datetime
    close_at: datetime
    close_price: float          # side-aligned
    actual_entry_price: float   # side-aligned (as stored on the row)
    actual_clv_pp: float
    rescan_at: datetime
    rescan_lead_min: float
    rescan_price: float         # side-aligned crossable proxy
    rescan_clv_pp: float
    anchor_fair: Optional[float] = None      # side-aligned fair prob at rescan
    anchor_age_min: Optional[float] = None
    rescan_edge_pp: Optional[float] = None   # None = no fresh anchor
    rescan_enter: bool = False               # fresh anchor AND edge >= ev_min


@dataclass
class RescanEvalStats:
    """Coverage counters so capture holes are visible, never silent."""

    rows: int = 0
    no_ticker: int = 0
    no_tip: int = 0
    bad_entry_price: int = 0
    no_window_snapshot: int = 0
    no_close_after: int = 0
    no_fresh_anchor: int = 0     # paired but EV-ungateable at rescan time
    pairs: list[PairedComparison] = field(default_factory=list)


def _load_snapshot_trail(arch_conn, ticker: str) -> list[tuple[datetime, float]]:
    rows = arch_conn.execute(
        """SELECT fetched_at, yes_price FROM archived_kalshi_markets
           WHERE ticker = ? AND yes_price IS NOT NULL
           ORDER BY fetched_at""",
        (ticker,),
    ).fetchall()
    out = []
    for r in rows:
        t = _parse_ts(r["fetched_at"])
        if t is not None and 0 < r["yes_price"] < 1:
            out.append((t, r["yes_price"]))
    return out


def _tip_time(arch_conn, event_id: str) -> Optional[datetime]:
    base = base_event_key(event_id)
    row = arch_conn.execute(
        """SELECT event_date FROM archived_sharp_odds
           WHERE (event_id = ? OR event_id LIKE ?) AND event_date IS NOT NULL
           ORDER BY fetched_at DESC LIMIT 1""",
        (base, base + "::%"),
    ).fetchone()
    return _parse_ts(row["event_date"]) if row else None


def _fair_yes_at(
    arch_conn,
    row,
    snap_t: datetime,
    anchor_max_stale_min: float = ANCHOR_MAX_STALE_MIN,
) -> tuple[Optional[float], Optional[float]]:
    """As-of Pinnacle fair YES prob for the row's market at ``snap_t``.

    Returns ``(fair_yes, anchor_age_min)`` — ``(None, None)`` when no archived
    anchor fresh enough exists or the anchor cannot price the bet's line/side.
    """
    mt = (row["market_type"] or "").lower()
    base = base_event_key(row["event_id"])
    yes_team = (row["yes_team"] or "").strip().lower()
    hi = snap_t.isoformat()

    if mt == "spread":
        r = arch_conn.execute(
            """SELECT fetched_at, outcome_a_label, outcome_b_label,
                      outcome_a_decimal, outcome_b_decimal,
                      true_prob_a, true_prob_b, spread_line
               FROM archived_sharp_odds
               WHERE event_id = ? AND fetched_at <= ? AND spread_line IS NOT NULL
               ORDER BY fetched_at DESC LIMIT 1""",
            (base + "::spread", hi),
        ).fetchone()
        if r is None:
            return None, None
        t = _parse_ts(r["fetched_at"])
        age_min = (snap_t - t).total_seconds() / 60.0
        if age_min > anchor_max_stale_min:
            return None, None
        so = SharpOdds(
            event_id=base + "::spread",
            book=SharpBook.pinnacle,
            sector=row["sector"],
            outcome_a_label=r["outcome_a_label"] or "",
            outcome_b_label=r["outcome_b_label"] or "",
            outcome_a_decimal=r["outcome_a_decimal"] or 2.0,
            outcome_b_decimal=r["outcome_b_decimal"] or 2.0,
            true_prob_a=r["true_prob_a"],
            true_prob_b=r["true_prob_b"],
            spread_line=r["spread_line"],
        )
        handler = get_handler(row["sector"])
        normalize = lambda s: handler.normalize_team(s or "")  # noqa: E731
        fair = _fair_yes_spread(so, row["sector"], yes_team, row["line"], normalize)
        return (fair, age_min) if fair is not None else (None, None)

    if mt == "total":
        rows = arch_conn.execute(
            """SELECT fetched_at, total_line, true_prob_over, true_prob_under
               FROM archived_sharp_odds
               WHERE event_id LIKE ? AND fetched_at <= ?
                 AND total_line IS NOT NULL AND true_prob_over IS NOT NULL
               ORDER BY fetched_at DESC LIMIT 40""",
            (base + "::total::%", hi),
        ).fetchall()
        if not rows:
            return None, None
        t = _parse_ts(rows[0]["fetched_at"])
        age_min = (snap_t - t).total_seconds() / 60.0
        if age_min > anchor_max_stale_min:
            return None, None
        # Ladder rows land seconds apart within one sweep — keep the sweep of
        # the latest row (minute-bucketed, same rule as listings_eval).
        sweep_key = rows[0]["fetched_at"][:16]
        ladder = {
            r["total_line"]: (r["true_prob_over"], r["true_prob_under"])
            for r in rows
            if r["fetched_at"][:16] == sweep_key
        }
        fair = _fair_yes_total(ladder, yes_team, row["line"])
        return (fair, age_min) if fair is not None else (None, None)

    # moneyline / advance / anything two-or-three-way with plain probs.
    # Advance ids keep their ::advance suffix (own derived anchor rows).
    event_id = row["event_id"] if mt == "advance" else base
    r = arch_conn.execute(
        """SELECT fetched_at, outcome_a_label, outcome_b_label,
                  true_prob_a, true_prob_b, true_prob_draw
           FROM archived_sharp_odds
           WHERE event_id = ? AND fetched_at <= ?
             AND spread_line IS NULL AND total_line IS NULL
           ORDER BY fetched_at DESC LIMIT 1""",
        (event_id, hi),
    ).fetchone()
    if r is None or r["true_prob_a"] is None:
        return None, None
    t = _parse_ts(r["fetched_at"])
    age_min = (snap_t - t).total_seconds() / 60.0
    if age_min > anchor_max_stale_min:
        return None, None
    fair = yes_aligned_close_prob(
        yes_team=yes_team,
        outcome_a_label=r["outcome_a_label"],
        outcome_b_label=r["outcome_b_label"],
        true_prob_a=r["true_prob_a"],
        true_prob_b=r["true_prob_b"],
        true_prob_draw=r["true_prob_draw"],
    )
    return (fair, age_min) if fair is not None else (None, None)


def evaluate(
    pred_conn,
    arch_conn,
    since: Optional[str] = None,
    window_lo_min: float = WINDOW_LO_MIN,
    window_hi_min: float = WINDOW_HI_MIN,
    ev_min_pp: float = EV_MIN_PP,
    anchor_max_stale_min: float = ANCHOR_MAX_STALE_MIN,
    sectors: Optional[list[str]] = None,
) -> RescanEvalStats:
    """Replay every eligible ev_predictions row through the rescan entry rule."""
    params: list = []
    clauses = ["p.voided = 0"]
    if since:
        clauses.append("p.scan_date >= ?")
        params.append(since)
    if sectors:
        clauses.append(f"p.sector IN ({','.join('?' * len(sectors))})")
        params.extend(sectors)
    rows = pred_conn.execute(
        f"""SELECT p.id, p.market_id, p.event_id, p.sector, p.market_type,
                   p.yes_team, p.line, p.event_title, p.mode,
                   p.kalshi_yes_price, p.placed, p.placed_price
            FROM ev_predictions p
            WHERE {' AND '.join(clauses)}
            ORDER BY p.scan_date, p.id""",
        params,
    ).fetchall()

    stats = RescanEvalStats()
    seen: set[str] = set()
    trail_cache: dict[str, list] = {}
    tip_cache: dict[str, Optional[datetime]] = {}

    for row in rows:
        # One comparison per market — a market scanned on consecutive days has
        # one row per scan_date; keep the earliest (true first-entry price).
        if row["market_id"] in seen:
            continue
        seen.add(row["market_id"])
        stats.rows += 1

        ticker, is_no_side = close_lookup_ticker(row["market_id"])
        if not ticker:
            stats.no_ticker += 1
            continue

        if row["event_id"] not in tip_cache:
            tip_cache[row["event_id"]] = _tip_time(arch_conn, row["event_id"])
        tip = tip_cache[row["event_id"]]
        if tip is None:
            stats.no_tip += 1
            continue

        actual_entry = clv_entry_price(
            placed=row["placed"],
            placed_price=row["placed_price"],
            scan_price=row["kalshi_yes_price"],
        )
        if actual_entry is None or not 0 < actual_entry < 1:
            stats.bad_entry_price += 1
            continue

        if ticker not in trail_cache:
            trail_cache[ticker] = _load_snapshot_trail(arch_conn, ticker)
        trail = trail_cache[ticker]

        entry = pick_rescan_entry(trail, tip, window_lo_min, window_hi_min)
        if entry is None:
            stats.no_window_snapshot += 1
            continue
        rescan_t, rescan_yes = entry

        close = pick_close(trail, tip, after=rescan_t)
        if close is None:
            stats.no_close_after += 1
            continue
        close_t, close_yes = close

        close_side = side_aligned(close_yes, is_no_side)
        rescan_side = side_aligned(rescan_yes, is_no_side)

        fair_yes, anchor_age = _fair_yes_at(
            arch_conn, row, rescan_t, anchor_max_stale_min
        )
        edge_pp: Optional[float] = None
        enter = False
        if fair_yes is not None:
            fair_side = side_aligned(fair_yes, is_no_side)
            edge_pp = (fair_side - rescan_side) * 100
            enter = edge_pp >= ev_min_pp
        else:
            stats.no_fresh_anchor += 1

        yes_team = (row["yes_team"] or "").strip()
        if row["market_type"] == "spread" and row["line"] is not None:
            outcome = f"{yes_team} {row['line']:+g}"
        elif row["market_type"] == "total" and row["line"] is not None:
            outcome = f"{yes_team.capitalize()} {row['line']:g}"
        else:
            outcome = f"{yes_team} ML" if row["market_type"] == "moneyline" else (
                f"{yes_team} {row['market_type']}"
            )
        if is_no_side:
            outcome = f"NO {outcome}"

        stats.pairs.append(PairedComparison(
            market_id=row["market_id"],
            sector=row["sector"],
            market_type=row["market_type"],
            event_title=row["event_title"] or base_event_key(row["event_id"]),
            outcome_label=outcome,
            is_no_side=is_no_side,
            mode=row["mode"] or "live",
            tip_at=tip,
            close_at=close_t,
            close_price=close_side,
            actual_entry_price=actual_entry,
            actual_clv_pp=(close_side - actual_entry) * 100,
            rescan_at=rescan_t,
            rescan_lead_min=(tip - rescan_t).total_seconds() / 60.0,
            rescan_price=rescan_side,
            rescan_clv_pp=(close_side - rescan_side) * 100,
            anchor_fair=side_aligned(fair_yes, is_no_side) if fair_yes is not None else None,
            anchor_age_min=anchor_age,
            rescan_edge_pp=edge_pp,
            rescan_enter=enter,
        ))
    return stats


@dataclass
class GroupSummary:
    sector: str
    market_type: str
    n: int
    actual_mean_clv: float
    actual_pct_pos: float
    rescan_mean_clv: float
    rescan_pct_pos: float
    delta_mean: float                     # mean(rescan - actual), paired
    median_lead_min: float
    n_gated: int
    gated_mean_clv: Optional[float]
    gated_pct_pos: Optional[float]
    gated_delta_mean: Optional[float]


def _pct_pos(vals: list[float]) -> float:
    return 100.0 * sum(1 for v in vals if v > 0) / len(vals)


def _group(sector: str, mt: str, pairs: list[PairedComparison]) -> GroupSummary:
    actual = [p.actual_clv_pp for p in pairs]
    rescan = [p.rescan_clv_pp for p in pairs]
    gated = [p for p in pairs if p.rescan_enter]
    return GroupSummary(
        sector=sector,
        market_type=mt,
        n=len(pairs),
        actual_mean_clv=mean(actual),
        actual_pct_pos=_pct_pos(actual),
        rescan_mean_clv=mean(rescan),
        rescan_pct_pos=_pct_pos(rescan),
        delta_mean=mean(r - a for r, a in zip(rescan, actual)),
        median_lead_min=median(p.rescan_lead_min for p in pairs),
        n_gated=len(gated),
        gated_mean_clv=mean(p.rescan_clv_pp for p in gated) if gated else None,
        gated_pct_pos=_pct_pos([p.rescan_clv_pp for p in gated]) if gated else None,
        gated_delta_mean=(
            mean(p.rescan_clv_pp - p.actual_clv_pp for p in gated) if gated else None
        ),
    )


def summarize(stats: RescanEvalStats) -> list[GroupSummary]:
    """Per (sector, market_type) groups plus an ALL row, sorted by n desc."""
    by_group: dict[tuple[str, str], list[PairedComparison]] = defaultdict(list)
    for p in stats.pairs:
        by_group[(p.sector, p.market_type)].append(p)
    out = [
        _group(sector, mt, pairs)
        for (sector, mt), pairs in sorted(
            by_group.items(), key=lambda kv: -len(kv[1])
        )
    ]
    if stats.pairs:
        out.append(_group("ALL", "all", stats.pairs))
    return out
