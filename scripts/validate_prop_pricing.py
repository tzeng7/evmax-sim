"""Validate Pinnacle anchor pricing against Kalshi market.

The validation gate for re-enabling _evaluate_prop_pair (see
evmax/agents/odds/ev_gap_agent.py:1011 and the project_props_validation_gate
memory): Pinnacle anchor pricing must beat both

  (a) the Kalshi implied price (i.e. sharper than the market we're betting
      against — required for any betting edge), and
  (b) the predict-37%-for-everything baseline that beat L15 in April.

Reads resolved rows from `prop_observations`, partitions by `model_version`
and stat, computes per-row Brier:

    brier_model  = (sharp_prob   - outcome) ²
    brier_kalshi = (kalshi_price - outcome) ²
    brier_naive  = (0.37         - outcome) ²

Reports averages per (model_version, stat) and an overall verdict — anchor
pricing must win on a stat-by-stat basis to clear the gate.

Run when ≥14 days of resolved 'pinnacle-anchor-v1' rows have accumulated:
  .venv/bin/python scripts/validate_prop_pricing.py
  .venv/bin/python scripts/validate_prop_pricing.py --version pinnacle-anchor-v1
  .venv/bin/python scripts/validate_prop_pricing.py --since 2026-05-10
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from statistics import mean
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "data" / "predictions.db"

NAIVE_PROB = 0.37  # the predict-37% baseline that beat L15 in April
PROMOTION_BAR_BRIER = 0.000  # anchor must at least tie Kalshi; positive = strictly better


def _load_rows(
    version: Optional[str],
    since: Optional[str],
    sector: str,
) -> list[dict]:
    sql = """
        SELECT scan_date, sector, stat_type, line, kalshi_price, sharp_prob,
               actual_value, outcome, model_version, mode
        FROM prop_observations
        WHERE outcome IS NOT NULL
          AND sharp_prob IS NOT NULL
          AND kalshi_price IS NOT NULL
          AND sector = ?
    """
    params: list = [sector]
    if version == "legacy":
        # NULL model_version — pre-tagging rows from the L15-era pipeline.
        # Useful for retroactive sanity check against the predict-37% baseline.
        sql += " AND model_version IS NULL"
    elif version is not None:
        sql += " AND model_version = ?"
        params.append(version)
    if since is not None:
        sql += " AND scan_date >= ?"
        params.append(since)
    sql += " ORDER BY scan_date"

    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def _brier(p: float, outcome: int) -> float:
    return (p - float(outcome)) ** 2


def report_per_stat(rows: list[dict]) -> None:
    by_stat: dict[str, list[dict]] = {}
    for r in rows:
        by_stat.setdefault(r["stat_type"], []).append(r)

    print(f"\n{'stat':<10} {'n':>6} {'B_model':>9} {'B_kalshi':>9} {'B_naive':>9} "
          f"{'Δ_kalshi':>10} {'Δ_naive':>9}  verdict")
    print("-" * 86)
    for stat, srows in sorted(by_stat.items()):
        bm = mean(_brier(r["sharp_prob"],   r["outcome"]) for r in srows)
        bk = mean(_brier(r["kalshi_price"], r["outcome"]) for r in srows)
        bn = mean(_brier(NAIVE_PROB,        r["outcome"]) for r in srows)
        d_kalshi = bk - bm   # positive = model better than Kalshi
        d_naive  = bn - bm   # positive = model better than naive
        if d_kalshi > PROMOTION_BAR_BRIER and d_naive > 0:
            verdict = "✓ beats both"
        elif d_naive > 0:
            verdict = "  beats naive only"
        else:
            verdict = "✗ worse than naive"
        print(f"{stat:<10} {len(srows):>6} {bm:>9.4f} {bk:>9.4f} {bn:>9.4f} "
              f"{d_kalshi:>+10.5f} {d_naive:>+9.5f}  {verdict}")


def report_actionable_buckets(rows: list[dict]) -> None:
    """The L15 trap: the model can be well-calibrated ON AVERAGE while still
    biasing UP in the long-shot bucket where we'd actually place bets. Slice
    by Kalshi price (proxy for which markets we'd bet on) and report bias
    direction in each bucket — the bucket-level pattern is what killed L15."""
    BUCKETS = [
        ("<10c (deep dog)", 0.00, 0.10),
        ("10-20c",          0.10, 0.20),
        ("20-40c",          0.20, 0.40),
        ("40-60c (close)",  0.40, 0.60),
        ("60-80c",          0.60, 0.80),
        (">80c (chalk)",    0.80, 1.01),
    ]
    print(f"\nKalshi-price-bucket diagnostic (where L15 died):")
    print(f"{'bucket':<18} {'n':>5} {'avg_kalshi':>10} {'avg_model':>10} "
          f"{'avg_actual':>11} {'bias':>8} {'B_model':>8} {'B_kalshi':>9}")
    print("-" * 92)
    for label, lo, hi in BUCKETS:
        sub = [r for r in rows if lo <= r["kalshi_price"] < hi]
        if not sub:
            continue
        avg_kalshi = mean(r["kalshi_price"] for r in sub)
        avg_model  = mean(r["sharp_prob"]   for r in sub)
        avg_actual = mean(r["outcome"]      for r in sub)
        bias = avg_model - avg_actual  # positive = model over-predicts overs
        bm = mean(_brier(r["sharp_prob"],   r["outcome"]) for r in sub)
        bk = mean(_brier(r["kalshi_price"], r["outcome"]) for r in sub)
        flag = "  "
        if abs(bias) > 0.05:
            flag = "⚠ "
        print(f"{flag}{label:<16} {len(sub):>5} {avg_kalshi:>10.3f} {avg_model:>10.3f} "
              f"{avg_actual:>11.3f} {bias:>+8.3f} {bm:>8.4f} {bk:>9.4f}")
    print(
        "\n  bias > 0 means model says OVER more often than reality → bets on "
        "this side lose. Watch the <10c row especially: that's where L15 "
        "produced 73% of its losses."
    )


def report_overall(rows: list[dict]) -> tuple[float, float, float]:
    bm = mean(_brier(r["sharp_prob"],   r["outcome"]) for r in rows)
    bk = mean(_brier(r["kalshi_price"], r["outcome"]) for r in rows)
    bn = mean(_brier(NAIVE_PROB,        r["outcome"]) for r in rows)
    print(f"\nOverall (n={len(rows)}):")
    print(f"  model  Brier:  {bm:.5f}")
    print(f"  kalshi Brier:  {bk:.5f}")
    print(f"  naive  Brier:  {bn:.5f}   (predict {NAIVE_PROB} for all)")
    print(f"  Δ vs kalshi:   {bk - bm:+.5f}  ({'better' if bm < bk else 'worse'})")
    print(f"  Δ vs naive:    {bn - bm:+.5f}  ({'better' if bm < bn else 'worse'})")
    return bm, bk, bn


def report_eligibility_count(version: str, sector: str) -> None:
    """Show whether we have enough resolved data to draw conclusions."""
    if version == "legacy":
        version_clause, version_args = "model_version IS NULL", ()
        version_label = "legacy (NULL model_version)"
    else:
        version_clause, version_args = "model_version = ?", (version,)
        version_label = repr(version)
    with sqlite3.connect(str(DB_PATH)) as conn:
        total = conn.execute(
            f"SELECT COUNT(*) FROM prop_observations "
            f"WHERE {version_clause} AND sector=?",
            (*version_args, sector),
        ).fetchone()[0]
        resolved = conn.execute(
            f"SELECT COUNT(*) FROM prop_observations "
            f"WHERE {version_clause} AND sector=? AND outcome IS NOT NULL",
            (*version_args, sector),
        ).fetchone()[0]
    print(f"  rows for {version_label}: total={total}  resolved={resolved}")
    if resolved < 200:
        print(f"  ⚠ Need ≥200 resolved rows for stable Brier estimate (n=√n CI shrinks below ±3%).")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="pinnacle-anchor-v1",
                        help="model_version to validate (default: pinnacle-anchor-v1)")
    parser.add_argument("--since", default=None,
                        help="Only include scan_date >= this (YYYY-MM-DD)")
    parser.add_argument("--sector", default="nba",
                        help="Sector to validate (default: nba)")
    args = parser.parse_args()

    print(f"=== validate_prop_pricing  version={args.version}  sector={args.sector} ===")
    report_eligibility_count(args.version, args.sector)

    rows = _load_rows(args.version, args.since, args.sector)
    if not rows:
        print("\nNo resolved rows yet. Re-run after games have settled.")
        return 0

    bm, bk, bn = report_overall(rows)
    report_per_stat(rows)
    report_actionable_buckets(rows)

    # Long-shot bias check: positive bias in the <10c bucket is the L15 trap —
    # the model was well-calibrated overall but over-predicted in the deep-dog
    # bucket where the high-EV signals lived, leading to systematic losses.
    deep_dog = [r for r in rows if r["kalshi_price"] < 0.10]
    if deep_dog and len(deep_dog) >= 30:
        avg_model_dd  = mean(r["sharp_prob"] for r in deep_dog)
        avg_actual_dd = mean(r["outcome"]    for r in deep_dog)
        deep_dog_bias = avg_model_dd - avg_actual_dd
    else:
        deep_dog_bias = None

    print("\n--- Promotion verdict ---")
    overall_pass = bm < bk and bm < bn
    long_shot_pass = deep_dog_bias is None or abs(deep_dog_bias) < 0.05
    if overall_pass and long_shot_pass:
        print("✓ Anchor pricing beats Kalshi AND predict-37% AND deep-dog bias < 5pp.")
        print("  Safe to re-enable _evaluate_prop_pair.")
    elif overall_pass and not long_shot_pass:
        print(f"⚠ Overall Brier wins, but deep-dog bias is {deep_dog_bias:+.3f} — "
              f"the same shape that killed L15.")
        print("  Do NOT re-enable until long-shot bucket is calibrated.")
    elif bm < bn:
        print("○ Beats predict-37% but not Kalshi. No betting edge — keep evaluator off.")
    else:
        print("✗ Worse than naive. Same trap that killed L15 — investigate before re-enabling.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
