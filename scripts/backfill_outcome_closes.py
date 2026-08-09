#!/usr/bin/env python3
"""Reconstruct ``ev_outcomes.pinnacle_close_prob`` directly from archive.db.

Thin CLI over
:func:`evmax.agents.cleanup.resolver.backfill_outcome_closes`.

``backfill_clv`` is driven by an ``ev_predictions INNER JOIN ev_outcomes``, so
resolved outcome rows whose ``ev_predictions`` partner was pruned by
maintenance (the stale-delete that orphans outcome rows) keep
``pinnacle_close_prob = NULL`` even though the Pinnacle close is still
archived. This backfill drives from ``ev_outcomes`` instead, so it reaches
orphans, reusing the same archive close-selector (moneyline games only —
see the function docstring for why spread/total/prop are excluded).

Examples
--------
    # preview NBA
    python scripts/backfill_outcome_closes.py --sector nba --dry-run
    # reconstruct NBA closes
    python scripts/backfill_outcome_closes.py --sector nba
"""

from __future__ import annotations

import argparse

from evmax.agents.cleanup.resolver import backfill_outcome_closes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sector", default=None, help="restrict to one sector, e.g. nba")
    ap.add_argument("--since", default=None, help="event_date lower bound YYYY-MM-DD")
    ap.add_argument("--until", default=None, help="event_date upper bound YYYY-MM-DD")
    ap.add_argument("--dry-run", action="store_true", help="report without writing")
    args = ap.parse_args()

    result = backfill_outcome_closes(
        sector=args.sector, since=args.since, until=args.until, dry_run=args.dry_run
    )
    mode = "DRY-RUN" if args.dry_run else "WROTE"
    print(
        f"[{mode}] sector={args.sector or 'ALL'} "
        f"candidates={result['candidates']} "
        f"filled={result['filled']} "
        f"no_archive_close={result['no_archive_close']}"
    )


if __name__ == "__main__":
    main()
