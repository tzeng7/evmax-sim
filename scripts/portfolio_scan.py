#!/usr/bin/env python3
"""Scheduled portfolio scanner — run via crontab.

Scans all active portfolios, logs today's +EV bets, and syncs
any pending outcomes from ev_outcomes.

Usage (standalone):
    python scripts/portfolio_scan.py

Crontab (2 AM and 3 PM ET daily):
    0 2 * * *  cd /Users/ktzeng/Projects/evmax && /Users/ktzeng/Projects/evmax/.venv/bin/python scripts/portfolio_scan.py >> data/logs/portfolio_scan.log 2>&1
    0 15 * * * cd /Users/ktzeng/Projects/evmax && /Users/ktzeng/Projects/evmax/.venv/bin/python scripts/portfolio_scan.py >> data/logs/portfolio_scan.log 2>&1
"""

from __future__ import annotations

import asyncio
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


async def main() -> None:
    from evmax.agents.coordinator import AgentCoordinator
    from evmax.portfolios import (
        is_excluded_from_portfolio,
        list_portfolios,
        log_portfolio_bet,
        sync_portfolio_outcomes,
    )

    now = datetime.now(timezone.utc)
    today_str = date.today().isoformat()
    print(f"\n{'='*60}")
    print(f"Portfolio scan — {now.isoformat()}")
    print(f"{'='*60}")

    # Sync any pending outcomes first
    synced = sync_portfolio_outcomes()
    if synced:
        print(f"Synced {synced} outcome(s) from ev_outcomes")

    portfolios = list_portfolios(active_only=True)
    if not portfolios:
        print("No active portfolios — nothing to scan")
        return

    all_sectors = sorted(set(s for p in portfolios for s in p.sectors))
    print(f"Scanning {len(portfolios)} portfolios across sectors: {', '.join(all_sectors)}")

    coord = AgentCoordinator(sectors=all_sectors, bankroll=500, kelly_fraction=0.5)
    cycle = await coord.run_cycle()

    print(f"Markets fetched: {cycle.markets_fetched}, matched: {cycle.markets_matched}")
    print(f"Total +EV gaps: {len(cycle.top_gaps)}")

    # plays() drops partial-blend (shadow, $0.00-stake) gaps + map_handicap —
    # the same actionable-play definition the CLI and dashboard use, so the
    # cron path can't silently log blocked tennis gaps as real stake.
    playable = cycle.plays(require_full_blend=True, drop_map_handicap=True)

    total_logged = 0
    for portfolio in portfolios:
        count = 0
        for g in playable:
            if g.sector not in portfolio.sectors:
                continue
            if is_excluded_from_portfolio(g.sector, g.market_type):
                continue

            event_date_str = str(g.event_date.astimezone().strftime("%Y-%m-%d") if g.event_date else "")
            # Drop only strictly-past games (keep today + future). The old
            # "== today" silently dropped Finals/series games Kalshi lists 2-4
            # days out; the web fan-out has no date floor at all.
            if event_date_str and event_date_str < today_str:
                continue

            gap_dict = {
                "market_id": g.market_id or "",
                "event_id": g.event_id or "",
                "event_title": g.event_title or "",
                "yes_team": g.yes_team or "",
                "market_type": g.market_type or "",
                "display_label": getattr(g, "display_label", "") or "",
                "sector": g.sector or "",
                "kalshi_yes_price": round(g.kalshi_yes_price, 4),
                "blended_true_prob": round(g.blended_true_prob, 4),
                "ev_pct": round(g.ev_pct, 4),
                "kelly_fraction": round(g.kelly_fraction, 4),
                "event_date": event_date_str,
                "volume_usd": g.volume_usd or 0,
                "model_sources": g.model_sources or "",
                "scan_date": today_str,
            }
            log_portfolio_bet(portfolio.id, gap_dict, portfolio.current_bankroll, portfolio.kelly_fraction)
            count += 1

        if count:
            print(f"  {portfolio.id}: {count} bets logged (bankroll=${portfolio.current_bankroll:.2f})")
        total_logged += count

    print(f"\nDone — {total_logged} total bets across {len(portfolios)} portfolios")


if __name__ == "__main__":
    asyncio.run(main())
