#!/usr/bin/env python3
"""Near-close re-scan validation harness — READ-ONLY replay over the real DBs.

Answers the Phase-1 gate question for the near-close re-scan workflow:

    "If we had re-scanned at T-75..T-10 before tip and entered at the live
    near-tip ask with a fresh Pinnacle anchor, how would entry CLV compare
    to the actual scan-time entry?"

Two lenses, both pure replay (the DBs are opened with sqlite URI mode=ro and
are NEVER written):

1. PAIRED ev_predictions lens (primary) — every logged bet that has an
   archived near-tip snapshot inside the window AND a later pre-tip close is
   scored under both entry rules against the SAME close.
   See evmax/agents/cleanup/rescan_eval.py for the conventions (NO-side flip,
   venue-prefixed tickers, fresh-anchor EV gate).

2. ARCHIVE-WIDE listings lens — for the currently-gated laddered market types
   (baseball total/spread, wnba spread/total) there are no ev_predictions
   rows, so the watch-listings capture is replayed twice through the
   listings-eval entry rule: unrestricted first-anchored-sweep (baseline ≈
   what a scan catches) vs entries restricted to the near-tip window. Same
   EV/depth gates, same last-pre-tip close.

Gate (from the mission): near-tip entry improves Kalshi CLV by >= 1pp on
n >= 50 overall, OR shows a clear positive-CLV path for at least one
currently-gated market type (baseball total, wnba total/spread).

Usage:
    python scripts/eval_near_close_rescan.py [--data-dir DIR] [--since DATE]
        [--window-lo 75] [--window-hi 10] [--ev-min 2.0] [--depth-min 50]
        [--sectors wnba,baseball] [--detail]
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

# Ensure the repo this script lives in wins over any editable install that
# points at a different checkout (worktree gotcha).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging

import structlog
from rich.console import Console
from rich.table import Table

from evmax.agents.cleanup import listings_eval, rescan_eval

# The spread-distribution model logs a debug line per unpriceable alt line —
# thousands of them over a full replay. Warnings and up only.
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
)

console = Console()

GATED_MARKETS = [("baseball", "total"), ("baseball", "spread"),
                 ("wnba", "total"), ("wnba", "spread")]


def open_ro(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise SystemExit(f"missing database: {path}")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def fmt(v, digits=2, suffix=""):
    return "—" if v is None else f"{v:+.{digits}f}{suffix}" if suffix == "pp" else (
        f"{v:.{digits}f}" if not isinstance(v, str) else v
    )


def run_paired_lens(args, pred_conn, arch_conn) -> list[rescan_eval.GroupSummary]:
    stats = rescan_eval.evaluate(
        pred_conn,
        arch_conn,
        since=args.since,
        window_lo_min=args.window_lo,
        window_hi_min=args.window_hi,
        ev_min_pp=args.ev_min,
        sectors=args.sectors.split(",") if args.sectors else None,
    )
    console.print(
        f"\n[bold]Lens 1 — paired ev_predictions replay[/bold]  "
        f"(window T-{args.window_lo:g}m..T-{args.window_hi:g}m, EV gate ≥ {args.ev_min:g}pp)"
    )
    console.print(
        f"[dim]rows={stats.rows}  paired={len(stats.pairs)}  "
        f"no_tip={stats.no_tip}  no_window_snapshot={stats.no_window_snapshot}  "
        f"no_close_after={stats.no_close_after}  no_ticker={stats.no_ticker}  "
        f"bad_entry={stats.bad_entry_price}  "
        f"paired_but_no_fresh_anchor={stats.no_fresh_anchor}[/dim]"
    )

    if args.detail:
        dt = Table(title="Paired comparisons (detail)")
        for col in ("Event", "Outcome", "Sector", "MT", "Lead(m)", "Actual entry",
                    "Rescan entry", "Close", "Actual CLV", "Rescan CLV",
                    "Rescan edge", "Enter?"):
            dt.add_column(col, no_wrap=False)
        for p in sorted(stats.pairs, key=lambda x: (x.sector, x.market_type)):
            dt.add_row(
                p.event_title, p.outcome_label, p.sector, p.market_type,
                f"{p.rescan_lead_min:.0f}",
                f"{p.actual_entry_price:.3f}", f"{p.rescan_price:.3f}",
                f"{p.close_price:.3f}",
                f"{p.actual_clv_pp:+.2f}", f"{p.rescan_clv_pp:+.2f}",
                "—" if p.rescan_edge_pp is None else f"{p.rescan_edge_pp:+.2f}",
                "Y" if p.rescan_enter else "n",
            )
        console.print(dt)

    groups = rescan_eval.summarize(stats)
    t = Table(title="Entry CLV: actual scan-time entry vs simulated near-tip re-scan (paired, pp)")
    for col in ("Sector", "MT", "n", "Actual CLV", "Act %+", "Rescan CLV",
                "Res %+", "Δ(res−act)", "n gated", "Gated CLV", "Gated Δ",
                "Med lead(m)"):
        t.add_column(col, no_wrap=False)
    for g in groups:
        style = "bold" if g.sector == "ALL" else ""
        t.add_row(
            g.sector, g.market_type, str(g.n),
            f"{g.actual_mean_clv:+.2f}", f"{g.actual_pct_pos:.0f}%",
            f"{g.rescan_mean_clv:+.2f}", f"{g.rescan_pct_pos:.0f}%",
            f"{g.delta_mean:+.2f}",
            str(g.n_gated),
            "—" if g.gated_mean_clv is None else f"{g.gated_mean_clv:+.2f}",
            "—" if g.gated_delta_mean is None else f"{g.gated_delta_mean:+.2f}",
            f"{g.median_lead_min:.0f}",
            style=style,
        )
    console.print(t)
    return groups


def run_listings_lens(args, arch_conn) -> dict:
    """Baseline vs near-tip-window replay of the listings-eval entry rule."""
    console.print(
        f"\n[bold]Lens 2 — archive-wide listings replay for gated market types[/bold]  "
        f"(EV ≥ {args.ev_min:g}pp, depth ≥ ${args.depth_min:g})"
    )
    results = {}
    max_lead_h = args.window_lo / 60.0
    min_lead_h = args.window_hi / 60.0
    t = Table(title="First-anchored-sweep entry: any-time baseline vs near-tip window (Kalshi CLV, pp)")
    for col in ("Sector", "MT", "Side", "Base n(clv)", "Base CLV", "Base %+",
                "Near n(clv)", "Near CLV", "Near %+", "Δ(near−base)"):
        t.add_column(col, no_wrap=False)
    sectors = sorted({s for s, _ in GATED_MARKETS})
    for sector in sectors:
        mts = tuple(mt for s, mt in GATED_MARKETS if s == sector)
        base = listings_eval.evaluate_sector(
            arch_conn, sector, ev_min_pp=args.ev_min,
            depth_min_usd=args.depth_min, market_types=mts, since=args.since,
        )
        near = listings_eval.evaluate_sector(
            arch_conn, sector, ev_min_pp=args.ev_min,
            depth_min_usd=args.depth_min, market_types=mts, since=args.since,
            max_lead_h=max_lead_h, min_lead_h=min_lead_h,
        )
        base_sum = {(s.market_type, s.side): s for s in listings_eval.summarize(base)}
        near_sum = {(s.market_type, s.side): s for s in listings_eval.summarize(near)}
        results[sector] = {"base": base_sum, "near": near_sum}
        for key in sorted(set(base_sum) | set(near_sum)):
            b, nr = base_sum.get(key), near_sum.get(key)
            delta = (
                nr.mean_clv_pp - b.mean_clv_pp
                if b and nr and b.mean_clv_pp is not None and nr.mean_clv_pp is not None
                else None
            )
            t.add_row(
                sector, key[0], key[1],
                f"{b.n}({b.n_with_clv})" if b else "0",
                "—" if not b or b.mean_clv_pp is None else f"{b.mean_clv_pp:+.2f}",
                "—" if not b or b.pct_clv_pos is None else f"{b.pct_clv_pos:.0f}%",
                f"{nr.n}({nr.n_with_clv})" if nr else "0",
                "—" if not nr or nr.mean_clv_pp is None else f"{nr.mean_clv_pp:+.2f}",
                "—" if not nr or nr.pct_clv_pos is None else f"{nr.pct_clv_pos:.0f}%",
                "—" if delta is None else f"{delta:+.2f}",
            )
    console.print(t)
    return results


def verdict(groups: list[rescan_eval.GroupSummary], listings: dict) -> None:
    console.print("\n[bold]Gate check[/bold]")
    overall = next((g for g in groups if g.sector == "ALL"), None)
    passed = False
    if overall:
        # The ungated paired Δ is mechanically (scan price − near-tip price):
        # +EV bets converge toward sharp fair pre-tip, so buying the same side
        # later is worse BY CONSTRUCTION on average. The workflow's entry rule
        # is the GATED subset — enter only when a fresh anchor still shows
        # edge ≥ ev_min at the near-tip crossable price — so the gate is
        # judged on gated entries.
        ok = (
            overall.n_gated >= 50
            and overall.gated_delta_mean is not None
            and overall.gated_delta_mean >= 1.0
        )
        console.print(
            f"  overall paired (ungated, drift diagnostic): n={overall.n}, "
            f"Δ(rescan−actual)={overall.delta_mean:+.2f}pp"
        )
        gclv = "—" if overall.gated_mean_clv is None else f"{overall.gated_mean_clv:+.2f}pp"
        gd = "—" if overall.gated_delta_mean is None else f"{overall.gated_delta_mean:+.2f}pp"
        console.print(
            f"  overall GATED entry rule (EV≥gate at near-tip price): "
            f"n={overall.n_gated}, CLV={gclv}, Δ={gd} "
            f"→ {'[green]PASS[/green]' if ok else '[red]fail[/red]'} (need Δ≥+1pp on n≥50)"
        )
        passed = passed or ok
    else:
        console.print("  overall paired: [red]no paired rows at all[/red]")
    for sector, mt in GATED_MARKETS:
        g = next((g for g in groups if g.sector == sector and g.market_type == mt), None)
        near = listings.get(sector, {}).get("near", {})
        near_rows = [s for (m, _), s in near.items() if m == mt]
        n_clv = sum(s.n_with_clv for s in near_rows)
        clvs = [s.mean_clv_pp * s.n_with_clv for s in near_rows if s.mean_clv_pp is not None]
        near_clv = (sum(clvs) / n_clv) if n_clv else None
        ok = (
            g is not None
            and g.n_gated >= 30
            and g.gated_mean_clv is not None
            and g.gated_mean_clv >= 1.0
        ) or (near_clv is not None and n_clv >= 30 and near_clv >= 1.0)
        bits = []
        if g:
            gclv = "—" if g.gated_mean_clv is None else f"{g.gated_mean_clv:+.2f}pp"
            bits.append(
                f"paired n={g.n} (gated n={g.n_gated} CLV={gclv})"
            )
        if near_clv is not None:
            bits.append(f"listings-near n(clv)={n_clv} CLV={near_clv:+.2f}pp")
        console.print(
            f"  {sector} {mt}: {'; '.join(bits) or 'no data'} "
            f"→ {'[green]positive path[/green]' if ok else '[red]no clear path[/red]'} "
            f"(need gated CLV≥+1pp on n≥30)"
        )
        passed = passed or ok
    console.print(
        f"\n[bold]{'VERDICT: VALIDATED' if passed else 'VERDICT: REJECTED'}[/bold] — "
        + ("near-tip re-scan entry clears the CLV gate."
           if passed else
           "near-tip re-scan entry does NOT clear the CLV gate on current capture data.")
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", default="/Users/ktzeng/Projects/evmax/data",
                    help="Directory holding predictions.db + archive.db (opened read-only)")
    ap.add_argument("--since", default="2026-06-20",
                    help="Earliest scan_date / capture date (watch-closes launched 06-20)")
    ap.add_argument("--window-lo", type=float, default=75.0,
                    help="Entry window opens this many minutes before tip")
    ap.add_argument("--window-hi", type=float, default=10.0,
                    help="Entry window closes this many minutes before tip")
    ap.add_argument("--ev-min", type=float, default=2.0, help="EV gate in pp")
    ap.add_argument("--depth-min", type=float, default=50.0,
                    help="Resting depth gate for the listings lens (USD)")
    ap.add_argument("--sectors", default=None,
                    help="Comma list to restrict the paired lens (default: all)")
    ap.add_argument("--detail", action="store_true", help="Print per-bet rows")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    pred_conn = open_ro(data_dir / "predictions.db")
    arch_conn = open_ro(data_dir / "archive.db")
    try:
        groups = run_paired_lens(args, pred_conn, arch_conn)
        listings = run_listings_lens(args, arch_conn)
        verdict(groups, listings)
    finally:
        pred_conn.close()
        arch_conn.close()


if __name__ == "__main__":
    main()
