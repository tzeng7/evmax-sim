"""Backfill Polymarket US closing prices from the public price-history API.

PolyUS closes otherwise come only from ``cleanup watch-closes`` snapshots,
which are near-tip only and missing whenever the sweep did not run. The
gateway's ``GET /v1/price-history`` is public and keeps minute-level display
prices for settled markets, so any PolyUS market we ever logged can have its
pre-tip close reconstructed after the fact.

The reconstructed points are written to ``archived_kalshi_markets`` under the
SAME venue-prefixed ticker the live snapshots use, with ``fetched_at`` set to
the history point's own timestamp. ``DataArchiver._select_kalshi_close`` /
``resolver.backfill_clv`` therefore pick them with the unchanged close rules:
the latest point at/before T-30 (session ``polyhist-t30``) and the latest
strictly pre-tip point (session ``polyhist-pretip``, the late-entry fallback).
In-play points (>= tip-off) are never written. Idempotent: the archive's
UNIQUE(session_id, ticker) makes a re-run a no-op.

Price mapping (a side's YES = its own ask): for the long side
``yes=longPrice, no=shortPrice``; for the short side they swap. A market id
without a ``:side`` suffix (spread / total / drawable outcome) is the long side.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

SESSION_T30 = "polyhist-t30"
SESSION_PRETIP = "polyhist-pretip"
WINDOW_HOURS = 6          # minute-level window ending at tip-off (docs: <= 24h)
SKIP_IF_STALENESS_H = 0.5  # an existing close this fresh is left alone
DEGENERATE = 0.99          # mirrors capture_polymarket_us_closes' empty-book rule


@dataclass(frozen=True)
class Target:
    ticker: str          # venue-prefixed archive ticker, ':no' stripped
    slug: str
    side_key: Optional[str]
    event_id: str
    sector: str
    market_type: str
    yes_team: Optional[str]
    line: Optional[float]
    event_date: Optional[str]


def parse_ticker(ticker: str) -> tuple[str, Optional[str]]:
    """``polymarket_us:{slug}[:{side}]`` -> (slug, side_key)."""
    body = ticker.removeprefix("polymarket_us:")
    slug, _, side = body.partition(":")
    return slug, (side or None)


def pick_points(history: list[dict], tip_ts: float, minutes_before: int = 30):
    """(t30_point, pretip_point) from a history list, never touching in-play.

    t30 = latest point with ts <= tip - 30min; pretip = latest with ts < tip.
    """
    pts = sorted(history, key=lambda h: h["timestamp"])
    t30_cut = tip_ts - minutes_before * 60
    t30 = next((h for h in reversed(pts) if h["timestamp"] <= t30_cut), None)
    pre = next((h for h in reversed(pts) if h["timestamp"] < tip_ts), None)
    return t30, pre


def side_prices(point: dict, is_long: bool) -> Optional[tuple[float, float]]:
    """(yes, no) asks of the contract's own side; None for an unusable book."""
    lp, sp = point.get("longPrice"), point.get("shortPrice")
    if lp is None or sp is None:
        return None
    yes, no = (lp, sp) if is_long else (sp, lp)
    if not (0.0 < yes < DEGENERATE) or not (0.0 < no <= 1.0):
        return None
    return float(yes), float(no)


def resolve_is_long(side_key: Optional[str], sides: list[dict]) -> Optional[bool]:
    """Is the contract the market's long side? None when it cannot be told."""
    if side_key is None:
        return True
    for sd in sides:
        abbr = (sd.get("team") or {}).get("abbreviation")
        if abbr is not None and abbr == side_key:
            return bool(sd.get("long"))
    if side_key in ("a", "b"):  # fallback key used when a side has no abbreviation
        return side_key == "a"
    return None


def tip_from_archive(event_id: str) -> Optional[datetime]:
    """Tip-off the CLV selector itself uses (archived_sharp_odds.event_date)."""
    from evmax.archiver import _get_connection

    with _get_connection() as conn:
        row = conn.execute(
            """SELECT event_date FROM archived_sharp_odds
               WHERE event_id = ? AND event_date IS NOT NULL
               ORDER BY fetched_at DESC LIMIT 1""",
            (event_id,),
        ).fetchone()
    if not row:
        return None
    try:
        dt = datetime.fromisoformat(row["event_date"])
    except (ValueError, TypeError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def load_targets(since: Optional[str] = None, until: Optional[str] = None) -> list[Target]:
    """Distinct PolyUS archive tickers behind logged ev_predictions rows."""
    from evmax.agents.cleanup.db import get_connection

    sql = """SELECT market_id, event_id, sector, market_type, yes_team, line, event_date
             FROM ev_predictions WHERE venue = 'polymarket_us'"""
    params: list = []
    if since:
        sql += " AND event_date >= ?"
        params.append(since)
    if until:
        sql += " AND event_date <= ?"
        params.append(until)
    sql += " ORDER BY market_id"
    seen: dict[str, Target] = {}
    with get_connection() as conn:
        for r in conn.execute(sql, params).fetchall():
            ticker = r["market_id"].removesuffix(":no")
            if not ticker.startswith("polymarket_us:") or ticker in seen:
                continue
            slug, side = parse_ticker(ticker)
            seen[ticker] = Target(ticker, slug, side, r["event_id"], r["sector"],
                                  r["market_type"], r["yes_team"], r["line"], r["event_date"])
    return list(seen.values())


def needs_backfill(t: Target, force: bool = False) -> bool:
    if force:
        return True
    from evmax.archiver import DataArchiver

    st = DataArchiver().get_kalshi_close_staleness_h(t.ticker, t.event_id)
    return st is None or st > SKIP_IF_STALENESS_H


async def backfill(
    targets: list[Target], client, dry_run: bool = False, force: bool = False,
    concurrency: int = 6,
) -> dict:
    """Fetch history per slug, write T-30 / pre-tip points. Returns a tally."""
    import asyncio

    from evmax.archiver import DataArchiver

    tally = {"targets": len(targets), "skipped_fresh": 0, "no_tip": 0, "no_history": 0,
             "no_side": 0, "unusable_book": 0, "written_t30": 0, "written_pretip": 0,
             "errors": 0}
    todo = []
    for t in targets:
        if needs_backfill(t, force):
            todo.append(t)
        else:
            tally["skipped_fresh"] += 1

    by_slug: dict[str, list[Target]] = {}
    for t in todo:
        by_slug.setdefault(t.slug, []).append(t)

    sem = asyncio.Semaphore(concurrency)
    out_t30: dict[str, list[dict]] = {}
    out_pre: dict[str, list[dict]] = {}
    stamp: dict[tuple[str, str], str] = {}

    async def one(slug: str, ts: list[Target]) -> None:
        async with sem:
            try:
                tip = tip_from_archive(ts[0].event_id)
                if tip is None:
                    tally["no_tip"] += len(ts)
                    return
                tip_ts = tip.timestamp()
                hist = await client.get_price_history(
                    slug, int(tip_ts - WINDOW_HOURS * 3600), int(tip_ts))
                if not hist:
                    tally["no_history"] += len(ts)
                    return
                t30, pre = pick_points(hist, tip_ts)
                sides = None
                for t in ts:
                    if t.side_key is not None and sides is None:
                        sides = await client.get_market_sides(slug)
                    is_long = resolve_is_long(t.side_key, sides or [])
                    if is_long is None:
                        tally["no_side"] += 1
                        continue
                    for kind, pt, bucket, key in (
                        ("t30", t30, out_t30, "written_t30"),
                        ("pretip", pre, out_pre, "written_pretip"),
                    ):
                        if pt is None:
                            continue
                        px = side_prices(pt, is_long)
                        if px is None:
                            tally["unusable_book"] += 1
                            continue
                        bucket.setdefault(t.sector, []).append({
                            "ticker": t.ticker, "yes_price": px[0], "no_price": px[1],
                            "event_id": t.event_id, "event_date": t.event_date,
                            "market_type": t.market_type, "yes_team": t.yes_team,
                            "line": t.line,
                        })
                        stamp[(kind, t.ticker)] = datetime.fromtimestamp(
                            pt["timestamp"], timezone.utc).isoformat()
                        tally[key] += 1
            except Exception:  # noqa: BLE001 — one bad slug never aborts the run
                tally["errors"] += len(ts)

    await asyncio.gather(*(one(slug, ts) for slug, ts in by_slug.items()))

    if dry_run:
        return tally
    archiver = DataArchiver()
    for session, bucket, kind in ((SESSION_T30, out_t30, "t30"), (SESSION_PRETIP, out_pre, "pretip")):
        for sector, snaps in bucket.items():
            for s in snaps:  # fetched_at is per point, so write one by one
                archiver.archive_kalshi_snapshot(
                    session, sector, [s], fetched_at=stamp[(kind, s["ticker"])])
    return tally
