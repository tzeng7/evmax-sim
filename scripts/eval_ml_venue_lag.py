"""Venue-lag lens for game-winner markets (pre-registered: docs/nfl-ml-lag-prereg.md).

Scores EVERY matched (game, team) Kalshi moneyline contract, not just scanner-
flagged rows: entry at the first anchored sweep, mark to the last pre-tip
snapshot, CLV regressed on entry edge vs devigged Pinnacle, net of the Kalshi
taker fee. Read-only against archive.db.

Usage: python scripts/eval_ml_venue_lag.py [--sector nfl] [--prefix KXNFLGAME]
                                           [--since YYYY-MM-DD] [--archive-db PATH]
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

import structlog

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evmax.agents.cleanup.moneyline_lag_eval import (  # noqa: E402
    FLAG_DEPTH_USD, FLAG_EDGE_PP, bucket_rows, evaluate_moneyline, game_means,
    lead_summary, mean_t, verdict,
)


def fmt(x, nd=2):
    return "n/a" if x is None else f"{x:+.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sector", default="nfl")
    ap.add_argument("--prefix", default="KXNFLGAME")
    ap.add_argument("--since", default=None)
    ap.add_argument("--archive-db", default=str(REPO / "data" / "archive.db"))
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.archive_db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    stats = evaluate_moneyline(conn, args.sector, args.prefix, since=args.since)
    v = verdict(stats)
    scored = [u for u in stats.units if u.gross_clv_pp is not None]
    el, cl = lead_summary(stats)

    print(f"{args.sector} moneyline venue-lag lens  (pre-reg: docs/nfl-ml-lag-prereg.md)")
    print(f"  tickers {stats.tickers} · entered {len(stats.units)} · scored {len(scored)} "
          f"in {v.n_games} games · never-anchored {stats.never_anchored} · "
          f"side-unresolved {stats.side_unresolved} · placeholder-asks skipped {stats.placeholder_entry}")
    print(f"  median entry lead {fmt(el, 1)}h · median close lead {fmt(cl, 1)}h")
    if scored:
        print(f"  all units: mean edge {fmt(sum(u.edge_pp for u in scored)/len(scored))}pp · "
              f"mean gross CLV {fmt(sum(u.gross_clv_pp for u in scored)/len(scored))}pp · "
              f"mean net CLV {fmt(sum(u.net_clv_pp for u in scored)/len(scored))}pp")
    if v.slope:
        b, se, z, g = v.slope
        print(f"  convergence slope CLV~edge: b={b:+.3f} se={se:.3f} z={z:+.2f} ({g} game clusters)")
    print("\n  edge bucket     units games  grossCLV  netCLV")
    for label, n, g, gross, net in bucket_rows(stats):
        print(f"  {label:<13} {n:>6} {g:>5}  {fmt(gross):>8}  {fmt(net):>7}")

    print(f"\n  flagged analogue (edge >= {FLAG_EDGE_PP:g}pp):")
    for name, units in (
        ("no depth gate", [u for u in scored if u.edge_pp >= FLAG_EDGE_PP]),
        (f"depth >= ${FLAG_DEPTH_USD:g}", [u for u in scored if u.edge_pp >= FLAG_EDGE_PP
                                           and (u.depth_usd or 0) >= FLAG_DEPTH_USD]),
    ):
        for metric in ("gross_clv_pp", "net_clv_pp"):
            r = mean_t(game_means(units, metric))
            if r:
                print(f"    {name:<16} {metric[:-7]:<5} {len(game_means(units, metric)):>3} games  "
                      f"mean {r[0]:+.2f}pp  t {r[1]:+.2f}  %pos {r[2]:.0f}")
            else:
                print(f"    {name:<16} {metric[:-7]:<5} too few games")
    print(f"\n  VERDICT: {v.verdict}")


if __name__ == "__main__":
    main()
