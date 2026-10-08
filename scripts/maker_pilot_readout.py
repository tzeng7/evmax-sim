"""Maker-only pilot checkpoint readout (evmax/ev/maker_pilot.py). Read-only.

Prints the fill rate of pilot candidates on resolved games, then scores the
FILLED pilot rows (``record_maker_fill`` flips them to ``mode='live'``) net of
the maker fee and prints the checkpoint verdict:

  COLLECTING  fewer than 15 filled games
  STOP        mean net CLV per game < 0          -> turn the pilot off
  HOLD        positive, < 55% of rows positive   -> keep ¼ Kelly
  STEP-UP     positive, >= 55% rows positive     -> MAKER_PILOT_KELLY_MULT = 0.5
  FULL-SIZE   30+ games, gate cleared            -> MAKER_PILOT_KELLY_MULT = 1.0

Usage: python scripts/maker_pilot_readout.py [--sector nfl] [--max-staleness-h 3] [--detail]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from evmax.agents.cleanup.db import get_connection  # noqa: E402
from evmax.cli.commands.shadow import _fetch_clv_rows  # noqa: E402
from evmax.ev.maker_pilot import (  # noqa: E402
    MAKER_PILOT_KELLY_MULT,
    MAKER_PILOT_SECTORS,
    MAKER_PILOT_TOKEN,
    checkpoint_verdict,
)
from evmax.formatting import format_outcome_label_for_row  # noqa: E402


def fill_rate(sector: str) -> tuple[int, int, int]:
    """(candidates on resolved games, of which filled, candidates still open)."""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT p.market_id, MAX(p.placed) AS placed,
                      MAX(o.outcome IS NOT NULL) AS resolved
               FROM ev_predictions p
               LEFT JOIN ev_outcomes o ON o.market_id = p.market_id
               WHERE p.sector = ? AND p.model_sources LIKE ?
                 AND COALESCE(p.voided, 0) = 0
               GROUP BY p.market_id""",
            (sector, f"%{MAKER_PILOT_TOKEN}%"),
        ).fetchall()
    resolved = [r for r in rows if r["resolved"]]
    return len(resolved), sum(1 for r in resolved if r["placed"]), len(rows) - len(resolved)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sector", default="nfl")
    ap.add_argument("--max-staleness-h", type=float, default=3.0,
                    help="Drop rows whose Kalshi close snapshot is stale (gate convention). <0 disables.")
    ap.add_argument("--detail", action="store_true", help="List every filled pilot row.")
    args = ap.parse_args()
    staleness = args.max_staleness_h if args.max_staleness_h >= 0 else None

    print(f"Maker pilot — sector={args.sector}  enabled={args.sector in MAKER_PILOT_SECTORS}  "
          f"kelly_mult={MAKER_PILOT_KELLY_MULT}")
    n_res, n_filled, n_open = fill_rate(args.sector)
    pct = f"{n_filled / n_res * 100:.0f}%" if n_res else "—"
    print(f"Candidates on resolved games: {n_res}   filled: {n_filled} ({pct})   still open: {n_open}")

    rows, excluded = _fetch_clv_rows(
        args.sector, market_type="spread", mode="live", venue="kalshi",
        sources_token=MAKER_PILOT_TOKEN, max_staleness_h=staleness,
    )
    rows = [dict(r) for r in rows]
    if args.detail and rows:
        # The CLV row fetch carries no yes_team; look it up for the Outcome label.
        ids = [r["market_id"] for r in rows]
        with get_connection() as conn:
            teams = dict(conn.execute(
                f"SELECT market_id, yes_team FROM ev_predictions WHERE market_id IN "
                f"({','.join('?' * len(ids))})", ids,
            ).fetchall())
        for r in rows:
            r["yes_team"] = teams.get(r["market_id"])
        print(f"\n{'Event':<40} {'Outcome':<26} {'Fill':>5} {'CLV pp':>7}")
        for r in sorted(rows, key=lambda r: r["event_id"] or ""):
            print(f"{(r['event_title'] or '')[:40]:<40} {format_outcome_label_for_row(r)[:26]:<26} "
                  f"{(r['placed_price'] or 0) * 100:>4.0f}c {r['kalshi_clv_pct']:>+7.2f}")

    v = checkpoint_verdict(rows)
    print()
    if v["games"]:
        z = f"{v['z']:+.2f}" if v["z"] is not None else "n/a"
        print(f"Filled: {v['games']} games / {v['rows']} rows   net-of-maker-fee CLV per game "
              f"{v['mean_net_pp']:+.2f}pp (z {z})   rows positive {v['frac_positive'] * 100:.0f}%"
              + (f"   [{excluded} stale-close rows dropped]" if excluded else ""))
    print(f"Verdict: {v['verdict']} — {v['action']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
