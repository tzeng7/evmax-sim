"""Diagnostic: break down ROI losses by model_prob bucket + closing_price bucket.

Stage 4 + MODEL-8 step 1-3 produced a model that's calibrated on average
(Brier 0.179) but loses −88% at the ROI filter. The pooled calibration
chart doesn't explain where the losses come from — this script slices
the ROI-flagged bets by predicted-probability bucket so we can see
whether the losses are concentrated in tails, middle, or everywhere.
"""

from __future__ import annotations

import sys
from collections import defaultdict

from evmax.backtest.sources.nfl_props import load_nfl_props


def bucket_of(prob: float) -> str:
    lo = int(prob * 10) * 10
    if lo >= 100:
        lo = 90
    return f"{lo}-{lo+10}%"


def main() -> int:
    rows = load_nfl_props(stats_filter=["passing_yards", "passing_tds"])
    with_price = [r for r in rows if r.closing_yes_price is not None]
    print(f"total scored: {len(rows)}, with closing price: {len(with_price)}")

    # Filter: valid ROI universe — closing price in [0.05, 0.95]
    universe = [
        r for r in with_price if 0.05 <= r.closing_yes_price <= 0.95
    ]
    print(f"universe after price filter [0.05, 0.95]: {len(universe)}")

    # Bets: model_prob - closing_yes_price >= 0.02
    bets = [r for r in universe if r.model_prob - r.closing_yes_price >= 0.02]
    print(f"bets at ev>=2%: {len(bets)}")
    wins = sum(1 for r in bets if r.actual_result == 1)
    print(f"wins: {wins} ({wins / max(len(bets),1) * 100:.1f}%)")
    print()

    # Breakdown by model_prob bucket
    buckets = defaultdict(list)
    for r in bets:
        buckets[bucket_of(r.model_prob)].append(r)
    print(f"{'model_prob':<12} {'n':>6} {'win_rate':>10} {'mean_price':>12} {'mean_model':>12} {'mean_edge':>10} {'roi':>10}")
    for k in sorted(buckets.keys(), key=lambda s: int(s.split("-")[0])):
        sub = buckets[k]
        wr = sum(1 for r in sub if r.actual_result == 1) / len(sub)
        mp = sum(r.closing_yes_price for r in sub) / len(sub)
        mm = sum(r.model_prob for r in sub) / len(sub)
        me = mm - mp
        pnl = 0.0
        for r in sub:
            if r.actual_result == 1:
                pnl += (1.0 / r.closing_yes_price) - 1.0
            else:
                pnl -= 1.0
        roi = pnl / len(sub)
        print(f"{k:<12} {len(sub):>6} {wr*100:>9.1f}% {mp:>12.3f} {mm:>12.3f} {me*100:>9.1f}% {roi*100:>9.1f}%")

    print()
    # Flip: breakdown by closing price bucket instead
    print("By closing price bucket:")
    pbuckets = defaultdict(list)
    for r in bets:
        pbuckets[bucket_of(r.closing_yes_price)].append(r)
    print(f"{'close_price':<12} {'n':>6} {'win_rate':>10} {'mean_model':>12} {'roi':>10}")
    for k in sorted(pbuckets.keys(), key=lambda s: int(s.split("-")[0])):
        sub = pbuckets[k]
        wr = sum(1 for r in sub if r.actual_result == 1) / len(sub)
        mm = sum(r.model_prob for r in sub) / len(sub)
        pnl = 0.0
        for r in sub:
            if r.actual_result == 1:
                pnl += (1.0 / r.closing_yes_price) - 1.0
            else:
                pnl -= 1.0
        roi = pnl / len(sub)
        print(f"{k:<12} {len(sub):>6} {wr*100:>9.1f}% {mm:>12.3f} {roi*100:>9.1f}%")

    # ALSO: what are the losses when we flip side and bet NO when model thinks market is too high?
    print()
    print("Flip test: bet NO on markets where closing_yes_price - model_prob >= 0.02")
    no_bets = [r for r in universe if r.closing_yes_price - r.model_prob >= 0.02]
    print(f"no-bets: {len(no_bets)}")
    no_wins = sum(1 for r in no_bets if r.actual_result == 0)
    pnl = 0.0
    for r in no_bets:
        no_price = 1.0 - r.closing_yes_price
        if no_price <= 0:
            continue
        if r.actual_result == 0:
            pnl += (1.0 / no_price) - 1.0
        else:
            pnl -= 1.0
    print(f"no-bet wins: {no_wins} ({no_wins/max(len(no_bets),1)*100:.1f}%)")
    print(f"no-bet ROI: {pnl / max(len(no_bets),1) * 100:.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
