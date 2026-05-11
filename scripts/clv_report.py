"""Honest CLV report — strips both-sides cancellation, adds Brier check.

The raw aggregate Kalshi-CLV on ev_predictions is polluted by two effects:

  1. Both-sides betting. When we log YES on Team A AND YES on Team B for
     the same event, the two CLVs move opposite each other by construction.
     They roughly cancel but leave a noisy residual from Kalshi vig.
     Net effect: 30% of NBA CLV rows contribute zero edge signal.

  2. No symmetric calibration check. Brier is sign-agnostic and works
     uniformly across single- and two-sided bets; comparing our Brier
     to Kalshi's Brier on the same outcomes tells us whether our
     prediction beat the soft book's prediction.

This script reports four numbers per sector:
  - raw_clv      : naive aggregate (current dashboard number)
  - single_clv   : CLV restricted to single-side-per-event bets only
  - top_side_clv : two-sided events keep ONLY their higher-EV side
  - brier_delta  : kalshi_brier − model_brier (positive = our model
                   was sharper than Kalshi's implied price)

Run:
  .venv/bin/python scripts/clv_report.py
  .venv/bin/python scripts/clv_report.py --sector nba
  .venv/bin/python scripts/clv_report.py --since 2026-04-01
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "data" / "predictions.db"


def _load(sector: str | None, since: str | None):
    sql = """
        SELECT p.id, p.event_id, p.sector, p.yes_team, p.market_type,
               p.kalshi_yes_price, p.sharp_true_prob, p.ev_pct,
               p.kalshi_clv_pct, p.pinnacle_drift_pct,
               p.model_sources, p.minutes_to_tipoff,
               o.outcome
        FROM ev_predictions p
        JOIN ev_outcomes o ON p.market_id = o.market_id
        WHERE p.voided=0 AND p.mode='live' AND o.outcome IS NOT NULL
    """
    params: list = []
    if sector:
        sql += " AND p.sector = ?"
        params.append(sector)
    if since:
        sql += " AND p.scan_date >= ?"
        params.append(since)
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def _summarize_sector(rows: list[dict], sector: str) -> dict:
    # Group by event for the both-sides analysis
    by_event: dict[str, list[dict]] = {}
    for r in rows:
        by_event.setdefault(r["event_id"], []).append(r)

    single_side_bets: list[dict] = []
    top_side_bets: list[dict] = []
    two_sided_events = 0
    for evt, bets in by_event.items():
        sides = {b["yes_team"] for b in bets}
        if len(sides) <= 1:
            single_side_bets.extend(bets)
            top_side_bets.extend(bets)
        else:
            two_sided_events += 1
            # Pick the highest-EV side per event for the top_side aggregate
            top = max(bets, key=lambda b: b["ev_pct"] or 0)
            top_side_bets.append(top)

    def _avg_clv(b_list: list[dict]) -> tuple[float, int]:
        vals = [b["kalshi_clv_pct"] for b in b_list if b["kalshi_clv_pct"] is not None]
        return (round(mean(vals), 2), len(vals)) if vals else (float("nan"), 0)

    raw_clv, n_raw = _avg_clv(rows)
    single_clv, n_single = _avg_clv(single_side_bets)
    top_clv, n_top = _avg_clv(top_side_bets)

    # Brier delta: kalshi implied vs our model, both against actual outcome.
    # Positive = our sharp_true_prob was more accurate than the Kalshi price.
    valid_brier = [r for r in rows if r["sharp_true_prob"] is not None]
    if valid_brier:
        kalshi_brier = mean((r["kalshi_yes_price"] - r["outcome"]) ** 2 for r in valid_brier)
        model_brier  = mean((r["sharp_true_prob"]  - r["outcome"]) ** 2 for r in valid_brier)
        brier_delta = kalshi_brier - model_brier  # positive = model beats kalshi
    else:
        kalshi_brier = model_brier = brier_delta = float("nan")

    return {
        "sector": sector,
        "total_bets": len(rows),
        "two_sided_events": two_sided_events,
        "raw_clv": raw_clv,
        "single_clv": single_clv,
        "single_n": n_single,
        "top_clv": top_clv,
        "top_n": n_top,
        "kalshi_brier": round(kalshi_brier, 4) if kalshi_brier == kalshi_brier else float("nan"),
        "model_brier":  round(model_brier, 4)  if model_brier  == model_brier  else float("nan"),
        "brier_delta":  round(brier_delta, 4)  if brier_delta  == brier_delta  else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sector", help="Restrict to one sector")
    parser.add_argument("--since", help="YYYY-MM-DD lower bound on scan_date")
    args = parser.parse_args()

    rows = _load(args.sector, args.since)
    if not rows:
        print("No matching rows.")
        return 1

    by_sector: dict[str, list[dict]] = {}
    for r in rows:
        by_sector.setdefault(r["sector"], []).append(r)

    summaries = [_summarize_sector(rs, s) for s, rs in by_sector.items()]
    summaries.sort(key=lambda x: -x["total_bets"])

    print(f"\n{'sector':<10} {'n':>5} {'2-side ev':>9} "
          f"{'raw CLV':>8} {'1-side CLV':>11} {'top-side CLV':>13} "
          f"{'k_Brier':>8} {'m_Brier':>8} {'Δ Brier':>8}")
    print("-" * 99)
    for s in summaries:
        print(
            f"{s['sector']:<10} {s['total_bets']:>5} {s['two_sided_events']:>9} "
            f"{s['raw_clv']:>8.2f} {s['single_clv']:>11.2f} {s['top_clv']:>13.2f} "
            f"{s['kalshi_brier']:>8.4f} {s['model_brier']:>8.4f} {s['brier_delta']:>+8.4f}"
        )

    # Scan-timing stratification. Late-news edge can ONLY manifest in late
    # scans — scans done close to tipoff have already absorbed any breaking
    # news through Pinnacle's quick adjustment, leaving Kalshi as the lagging
    # side we can catch. Scans done hours before tipoff are dominated by
    # post-scan market drift, which is mostly noise from our perspective
    # (the news happens BETWEEN scan and close, so we couldn't have
    # anticipated it). If our late-news thesis holds, the <30 min bucket
    # should show systematically higher Kalshi-CLV than the 24h+ bucket.
    BUCKETS = [
        ("<30 min",     0,    30),
        ("30 min–6h",  30,   360),
        ("6h–24h",    360,  1440),
        ("24h+",      1440, None),
    ]
    clv_eligible = [r for r in rows if r["kalshi_clv_pct"] is not None]
    print()
    print("Scan-timing stratification (cross-sector, Kalshi-CLV):")
    print(f"  {'bucket':<14} {'n':>5} {'avg CLV':>10}")
    for label, lo, hi in BUCKETS:
        if hi is None:
            sub = [r for r in clv_eligible
                   if r.get("minutes_to_tipoff") is not None
                   and r["minutes_to_tipoff"] >= lo]
        else:
            sub = [r for r in clv_eligible
                   if r.get("minutes_to_tipoff") is not None
                   and lo <= r["minutes_to_tipoff"] < hi]
        if not sub:
            print(f"  {label:<14} {0:>5}  {'—':>10}")
            continue
        avg = mean(r["kalshi_clv_pct"] for r in sub)
        print(f"  {label:<14} {len(sub):>5}  {avg:>+9.2f}pp")
    no_tipoff = [r for r in clv_eligible if r.get("minutes_to_tipoff") is None]
    if no_tipoff:
        avg = mean(r["kalshi_clv_pct"] for r in no_tipoff)
        print(f"  (no timing)    {len(no_tipoff):>5}  {avg:>+9.2f}pp  ← pre-2026-05-10 rows")

    # Late-news tag slice — only meaningful WHEN COMBINED with the
    # <30 min bucket. The tag fires on news within 6 hours of scan, but
    # if the scan itself was 24h pre-tipoff that news is already
    # priced in by Kalshi long before close.
    late_tagged_short = [
        r for r in clv_eligible
        if "late_news" in (r.get("model_sources") or "")
        and r.get("minutes_to_tipoff") is not None
        and r["minutes_to_tipoff"] < 60
    ]
    untagged_short = [
        r for r in clv_eligible
        if "late_news" not in (r.get("model_sources") or "")
        and r.get("minutes_to_tipoff") is not None
        and r["minutes_to_tipoff"] < 60
    ]
    print()
    print("Late-news edge (conditional on late scan, < 60 min to tipoff):")
    if late_tagged_short:
        print(f"  tagged + late scan  n={len(late_tagged_short):>4}  "
              f"avg Kalshi-CLV {mean(r['kalshi_clv_pct'] for r in late_tagged_short):+.2f}pp")
    else:
        print(f"  tagged + late scan  n=0  (no late-scan resolved bets yet)")
    if untagged_short:
        print(f"  untagged late scan  n={len(untagged_short):>4}  "
              f"avg Kalshi-CLV {mean(r['kalshi_clv_pct'] for r in untagged_short):+.2f}pp")
    else:
        print(f"  untagged late scan  n=0")

    # Interpretation hints
    print()
    print("Reads:")
    print("  raw_clv         — every CLV-eligible bet, naive aggregate")
    print("  1-side CLV      — events where we bet only one side (no mutual cancellation)")
    print("  top-side        — 2-sided events keep only the higher-EV bet")
    print("  Δ Brier         — positive means model beats Kalshi's price as a predictor")
    print("                    (CI on Brier ≈ ±0.01 for n=100, ±0.003 for n=500)")
    print("  Scan timing     — Kalshi-CLV stratified by minutes-to-tipoff at scan")
    print("                    If late-news edge is real, <30min bucket > 24h+ bucket")
    print("  Late-news tag   — only meaningful in conjunction with late scan")
    return 0


if __name__ == "__main__":
    sys.exit(main())
