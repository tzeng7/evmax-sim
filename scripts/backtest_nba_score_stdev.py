"""Backtest NBA SCORE_STDEV (margin sigma) against resolved spread bets.

The 4-week metrics report showed NBA model is over-confident at the tails:
  >80% bucket (n=17): predicted 90.7%, actual 82.4% (+8.3pp)
  <20% bucket (n=11): predicted 16.6%, actual 27.3% (-10.7pp)

Both biases mean the underlying margin sigma used for spread CDF math is
TOO SMALL (sharp tails). Increasing sigma shrinks all probs toward 0.5.

Method:
  Recover implied z-score from each stored spread `blended_true_prob`,
  treat it as `(threshold - mean) / sigma_old`, then re-evaluate at
  several candidate sigmas via:
      prob_new = Φ(z_old × sigma_old / sigma_new)
  This is mathematically the same as keeping (mean, threshold) fixed and
  swapping sigma — exactly what changing _SECTOR_SIGMA["nba"] would do at
  the spread_distribution layer (and the possession_sim cover_probability
  layer, which also hard-codes sigma=11.5 for NBA).

Caveat: the stored `blended_true_prob` is `0.65 × spread_dist + 0.35 × sim`
(see ev_gap_agent.py step 2a). Both internal probs use sigma=11.5, so the
z-shrink approximates the unified bump well — but it is an approximation,
not exact recomputation. Treat the recommended sigma as a tuning hint, not
a definitive answer; confirm via walk-forward before promoting.

Usage:
    .venv/bin/python scripts/backtest_nba_score_stdev.py
    .venv/bin/python scripts/backtest_nba_score_stdev.py --weeks 12
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
from scipy.stats import norm

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "predictions.db"
SIGMA_OLD = 11.5  # current _SECTOR_SIGMA["nba"]
CANDIDATES = [10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0]


def _fetch_spreads(weeks: int) -> list[tuple[float, int]]:
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        f"""
        SELECT p.blended_true_prob, o.outcome
        FROM ev_predictions p
        JOIN ev_outcomes o ON p.market_id = o.market_id
        WHERE p.sector = 'nba'
          AND p.market_type = 'spread'
          AND o.outcome IS NOT NULL
          AND p.scan_date >= date('now', '-{int(weeks * 7)} days')
          AND (p.mode IS NULL OR p.mode = 'live')
        """
    ).fetchall()
    conn.close()
    return [(float(p), int(o)) for p, o in rows]


def _z_shrink(prob: float, sigma_old: float, sigma_new: float) -> float:
    prob = max(1e-6, min(1.0 - 1e-6, prob))
    z = norm.ppf(prob)
    z_new = z * sigma_old / sigma_new
    return float(norm.cdf(z_new))


def _brier(probs: list[float], outcomes: list[int]) -> float:
    return float(np.mean([(p - o) ** 2 for p, o in zip(probs, outcomes)]))


def _bucket_bias(probs: list[float], outcomes: list[int], lo: float, hi: float) -> tuple[int, float, float]:
    sub = [(p, o) for p, o in zip(probs, outcomes) if lo <= p < hi]
    if not sub:
        return 0, 0.0, 0.0
    n = len(sub)
    avg_p = sum(p for p, _ in sub) / n
    avg_o = sum(o for _, o in sub) / n
    return n, avg_p - avg_o, avg_o


def main(weeks: int) -> None:
    rows = _fetch_spreads(weeks)
    n = len(rows)
    if n < 20:
        print(f"Only {n} resolved NBA spread bets in last {weeks}w — need ≥20 for a stable sweep. Aborting.")
        return

    probs_old = [p for p, _ in rows]
    outcomes = [o for _, o in rows]
    base_brier = _brier(probs_old, outcomes)

    print(f"=== backtest_nba_score_stdev (last {weeks}w, n={n} resolved spread bets) ===\n")
    print(f"{'sigma':>8} {'Brier':>10} {'ΔBrier':>10} {'Hit Rate':>10}")
    print("-" * 42)

    best_sigma = SIGMA_OLD
    best_brier = base_brier
    for sigma in CANDIDATES:
        probs_new = [_z_shrink(p, SIGMA_OLD, sigma) for p in probs_old]
        b = _brier(probs_new, outcomes)
        delta = b - base_brier
        marker = "  ←" if sigma == SIGMA_OLD else ("  *" if b < best_brier else "")
        if b < best_brier:
            best_brier = b
            best_sigma = sigma
        avg_p = sum(probs_new) / n
        print(f"{sigma:>8.1f} {b:>10.5f} {delta:>+10.5f} {avg_p:>9.1%}{marker}")

    print(f"\nbase  (sigma={SIGMA_OLD}) Brier: {base_brier:.5f}")
    print(f"best  (sigma={best_sigma}) Brier: {best_brier:.5f}")
    print(f"delta: {best_brier - base_brier:+.5f}")

    print("\n--- Tail-bucket bias before/after at recommended sigma ---")
    print(f"{'Bucket':<10} {'N':>4} {'BiasOld':>10} {'BiasNew':>10}")
    for lo, hi, label in [(0, 0.20, "<20%"), (0.20, 0.40, "20-40%"),
                          (0.40, 0.60, "40-60%"), (0.60, 0.80, "60-80%"),
                          (0.80, 1.01, ">80%")]:
        n_old, bias_old, _ = _bucket_bias(probs_old, outcomes, lo, hi)
        probs_new = [_z_shrink(p, SIGMA_OLD, best_sigma) for p in probs_old]
        n_new, bias_new, _ = _bucket_bias(probs_new, outcomes, lo, hi)
        print(f"{label:<10} {n_old:>4} {bias_old*100:>+9.1f}pp {bias_new*100:>+9.1f}pp")

    if best_sigma != SIGMA_OLD and (base_brier - best_brier) >= 0.001:
        print(
            f"\nRECOMMEND: bump _SECTOR_SIGMA['nba'] (and possession_sim sigma) "
            f"{SIGMA_OLD} → {best_sigma} (Brier −{base_brier - best_brier:.5f})."
        )
    else:
        print(f"\nKEEP sigma={SIGMA_OLD} (no candidate beats current by ≥0.001).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weeks", type=int, default=8, help="Look-back window (default: 8w)")
    args = parser.parse_args()
    main(args.weeks)
