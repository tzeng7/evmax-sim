"""Repair rows graded on the WRONG game of a multi-day series.

Before 2026-09-22 the ESPN resolve path pooled the event_date−1 / event_date /
event_date+1 scoreboards (to absorb stored-date off-by-ones) and fetched
COMPLETED games only. A row resolved before its own game finished — the
morning resolve picks up rows scanned yesterday for games today — found only
yesterday's completed game of the same series in the pool and was graded on
it (≈192 rows, mostly baseball). The fixed resolver sees the own-day game even
when it is not final and never falls back past it.

This script re-grades every ESPN-resolved game row (the ``ESPN_SPORT_MAP``
sectors — series sports) against freshly fetched scoreboards with the fixed
``_match_espn`` and plans a rewrite only when:
  * the venue CONTRACT is for the row's own ``event_date`` — the Kalshi ticker
    date / Polymarket US slug date matches it. A separate matcher bug priced
    some rows against the adjacent game of the series (ticker dated
    event_date±1); for those the contract settles on the OTHER game, so
    re-grading onto event_date would overwrite a correct grade. They are
    listed separately and never written;
  * the fixed matcher selects a COMPLETED game dated on the row's own
    ``event_date`` (so a genuinely mis-dated row is never "corrected" onto a
    different game), and
  * that grade differs from the stored outcome.

``portfolio_bets`` copies of an outcome are only ever filled while NULL
(``portfolios.sync_portfolio_outcomes``), so the script reports how many
portfolio rows carry a changed outcome; fix those deliberately if needed.

``--dry-run`` (default) is read-only on both databases (ESPN GETs only).
``--apply`` first writes a full SQLite backup next to predictions.db
(``predictions.db.bak-series-<UTC timestamp>``), then updates each planned row
inside one transaction, guarded on the old value. Idempotent: a re-run after
``--apply`` plans nothing.

    python scripts/repair_series_game_outcomes.py [--sector baseball] [--since 2026-03-01]
    python scripts/repair_series_game_outcomes.py --apply
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evmax.agents.cleanup.resolver import (  # noqa: E402
    ESPN_SPORT_MAP,
    _ESPN_HTTP_UA,
    _fetch_espn_scores,
    _match_espn,
    _select_espn_game,
)

DEFAULT_DB = ROOT / "data" / "predictions.db"


@dataclass
class SeriesFix:
    market_id: str
    sector: str
    event_date: str
    event_title: str
    yes_team: str
    market_type: str
    mode: str
    placed: int
    stored: int
    correct: int


_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}
_KALSHI_DATE_RE = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})")
_SLUG_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def contract_date(market_id: Optional[str]) -> Optional[str]:
    """ISO game date the venue contract is for, from the Kalshi ticker
    (``…-26JUL11…``) or the Polymarket US slug (``…-2026-07-10-…``)."""
    if not market_id:
        return None
    if market_id.startswith("polymarket_us:"):
        m = _SLUG_DATE_RE.search(market_id)
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None
    m = _KALSHI_DATE_RE.search(market_id.upper())
    if not m or m.group(2) not in _MONTHS:
        return None
    return f"20{m.group(1)}-{_MONTHS[m.group(2)]:02d}-{m.group(3)}"


def contract_date_mismatch(row: dict) -> bool:
    cd = contract_date(row.get("market_id"))
    return cd is not None and cd != (row.get("event_date") or "")[:10]


def _window(d: str) -> list[str]:
    day = date.fromisoformat(d[:10])
    return [(day + timedelta(days=k)).strftime("%Y%m%d") for k in (-1, 0, 1)]


def load_rows(conn: sqlite3.Connection, sector: Optional[str], since: Optional[str]) -> list[dict]:
    where = ["o.outcome IN (0, 1)", "o.result_source = 'espn'",
             "p.event_id NOT LIKE '%::prop::%'"]
    params: list = []
    sectors = [sector] if sector else sorted(ESPN_SPORT_MAP)
    where.append(f"p.sector IN ({','.join('?' * len(sectors))})")
    params += sectors
    if since:
        where.append("p.event_date >= ?")
        params.append(since)
    sql = f"""SELECT p.market_id, p.event_id, p.sector, p.yes_team, p.market_type,
                     p.line, p.event_date, p.event_title, p.mode, p.placed,
                     o.outcome AS stored
              FROM ev_predictions p JOIN ev_outcomes o ON o.market_id = p.market_id
              WHERE {' AND '.join(where)}"""
    conn.row_factory = sqlite3.Row
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def plan_fix(row: dict, scores: list[dict]) -> Optional[SeriesFix]:
    """The fix for one row, or None when its stored grade stands."""
    selected = _select_espn_game(row, scores)
    if selected is None:
        return None
    game, _ = selected
    if (game.get("game_date") or "")[:10] != (row.get("event_date") or "")[:10]:
        return None  # only ever re-grade onto the row's OWN-date game
    correct = _match_espn(row, scores)
    if correct is None or correct == row["stored"]:
        return None
    return SeriesFix(
        market_id=row["market_id"], sector=row["sector"], event_date=row["event_date"],
        event_title=row.get("event_title") or "", yes_team=row.get("yes_team") or "",
        market_type=row.get("market_type") or "", mode=row.get("mode") or "",
        placed=int(row.get("placed") or 0), stored=row["stored"], correct=correct,
    )


async def plan(rows: list[dict]) -> tuple[list[SeriesFix], list[SeriesFix]]:
    """(fixes to write, contract-date-mismatch rows that would change — never written)."""
    cache: dict = {}
    fixes: list[SeriesFix] = []
    skipped: list[SeriesFix] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(20.0),
                                 headers={"User-Agent": _ESPN_HTTP_UA},
                                 follow_redirects=True) as client:
        by_key: dict[tuple[str, str], list[dict]] = {}
        for r in rows:
            if r.get("event_date"):
                by_key.setdefault((r["sector"], r["event_date"][:10]), []).append(r)
        for (sector, day), group in sorted(by_key.items()):
            sport, league, extra = ESPN_SPORT_MAP[sector]
            per_day = await asyncio.gather(*(
                _fetch_espn_scores(client, sport, league, d, extra, cache=cache,
                                   include_incomplete=True)
                for d in _window(day)
            ))
            scores = [s for daylist in per_day for s in daylist]
            for r in group:
                fix = plan_fix(r, scores)
                if fix is None:
                    continue
                (skipped if contract_date_mismatch(r) else fixes).append(fix)
    return fixes, skipped


def portfolio_copies(conn: sqlite3.Connection, fixes: list[SeriesFix]) -> int:
    """portfolio_bets rows carrying an outcome this repair changes."""
    if not fixes:
        return 0
    try:
        ids = [f.market_id for f in fixes]
        q = f"SELECT COUNT(*) FROM portfolio_bets WHERE market_id IN ({','.join('?' * len(ids))}) AND outcome IS NOT NULL"
        return conn.execute(q, ids).fetchone()[0]
    except sqlite3.Error:
        return 0


def apply(db: Path, fixes: list[SeriesFix]) -> tuple[Path, int]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = db.with_name(f"{db.name}.bak-series-{stamp}")
    src = sqlite3.connect(str(db))
    dst = sqlite3.connect(str(backup))
    with dst:
        src.backup(dst)
    dst.close()
    changed = 0
    with src:
        for f in fixes:
            changed += src.execute(
                "UPDATE ev_outcomes SET outcome = ? WHERE market_id = ? AND outcome = ?",
                (f.correct, f.market_id, f.stored),
            ).rowcount
    src.close()
    return backup, changed


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--apply", action="store_true", help="back up predictions.db, then write")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--sector", default=None)
    p.add_argument("--since", default=None, help="event_date lower bound (YYYY-MM-DD)")
    args = p.parse_args(argv)

    uri = f"file:{args.db}?mode=ro" if not args.apply else str(args.db)
    conn = sqlite3.connect(uri, uri=not args.apply)
    rows = load_rows(conn, args.sector, args.since)
    fixes, skipped = asyncio.run(plan(rows))
    n_portfolio = portfolio_copies(conn, fixes)
    conn.close()

    print(f"checked {len(rows)} ESPN-graded game rows; {len(fixes)} graded on the wrong game")
    by_sector: dict[str, int] = {}
    for f in fixes:
        by_sector[f.sector] = by_sector.get(f.sector, 0) + 1
    for s, n in sorted(by_sector.items()):
        live = sum(1 for f in fixes if f.sector == s and f.mode == "live")
        placed = sum(1 for f in fixes if f.sector == s and f.placed)
        print(f"  {s:10s} {n:4d} rows  (live {live}, placed {placed})")
    for f in fixes[:25]:
        print(f"  {f.event_date} {f.sector:8s} {f.market_type:9s} {f.event_title[:40]:40s} "
              f"YES={f.yes_team[:18]:18s} {f.stored}->{f.correct} {f.mode}{' PLACED' if f.placed else ''}")
    if len(fixes) > 25:
        print(f"  … {len(fixes) - 25} more")
    if skipped:
        print(f"skipped {len(skipped)} rows whose venue contract is dated off event_date "
              f"(priced against a different game of the series — NOT written):")
        for f in skipped[:25]:
            print(f"  {f.event_date} {f.sector:8s} {f.market_id[:60]}  stored {f.stored}")
    if n_portfolio:
        print(f"note: {n_portfolio} portfolio_bets rows copy an outcome this repair changes; "
              "sync only fills NULL outcomes, so fix those deliberately if needed")
    if args.apply and fixes:
        backup, changed = apply(args.db, fixes)
        print(f"applied {changed} of {len(fixes)} planned fixes; backup at {backup}")
    elif not args.apply:
        print("(dry-run — nothing written; pass --apply to back up and write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
