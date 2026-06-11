"""Bucket comparison: full tennis model blend vs ranking_trend-only, both +sharp.

Reuses the leak-free walk-forward harness in backtest_tennis_walkforward.py
(seeded state from disk, in-memory updates only, eval on 2025+2026) and builds
two production-style blends:

  full     — all 6 tennis agents, SECTOR_WEIGHT_OVERRIDES["tennis"] weights,
             confidence gate 0.45, then sharp_weight 0.85 + the DEFAULT
             disagreement ramp (0.10 / 0.30 / 0.95) — i.e. what the live
             ensemble does today (the tennis-specific ramp was removed
             2026-05-02).
  ranking  — tennis_ranking_trend as the ONLY model, same gate, same 0.85
             base + default ramp. Rows where the trend agent doesn't fire
             fall back to sharp, exactly like the ensemble would.

Reports:
  1. Overall Brier / accuracy / coverage for sharp, full, ranking blends
  2. Calibration buckets (predicted-prob bins): n, avg predicted, actual
     win rate, gap, per-bucket Brier — side by side
  3. Disagreement buckets (|model - sharp|): per-bucket Brier of each blend
     vs sharp, so you can see where each blend earns or loses its keep

USAGE
    python scripts/compare_tennis_blend_buckets.py
    python scripts/compare_tennis_blend_buckets.py --train 2024 --eval 2025,2026
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
sys.path.insert(0, str(_SCRIPTS_DIR.parent))

from backtest_tennis_walkforward import (  # noqa: E402
    CONFIDENCE_GATE,
    TennisWalkRow,
    disagreement_ramp,
    filter_eval,
    load_tennis_matches,
    model_blend,
    walk_forward,
)
from evmax.backtest.metrics import accuracy_score, brier_score  # noqa: E402

# Production params: data/model_config.json sharp_weight_by_sector["tennis"]
# + the ensemble's DEFAULT ramp (tennis override removed 2026-05-02).
PROD_SHARP_WEIGHT = 0.85
DEFAULT_RAMP = dict(threshold=0.10, saturate_at=0.30, cap=0.95)


def ranking_model(row: TennisWalkRow) -> Optional[float]:
    """tennis_ranking_trend as the sole model, same confidence gate."""
    if row.trend_prob_a is None or row.trend_conf is None:
        return None
    if row.trend_conf <= CONFIDENCE_GATE:
        return None
    return row.trend_prob_a


def blend_with_sharp(model_a: Optional[float], sharp_a: float) -> tuple[float, bool]:
    """Production-style blend. Returns (blended_prob, model_contributed)."""
    if model_a is None:
        return sharp_a, False
    sw = disagreement_ramp(PROD_SHARP_WEIGHT, model_a, sharp_a, **DEFAULT_RAMP)
    return sw * sharp_a + (1.0 - sw) * model_a, True


def overall_summary(rows: list[TennisWalkRow]) -> None:
    print("\n=== Overall (eval period) ===")
    print(f"{'Blend':22s}  {'n':>6s}  {'fired':>6s}  {'Brier':>8s}  {'Acc':>7s}  {'Δ vs sharp':>12s}")
    print("-" * 72)
    sharp_b = brier_score([r.sharp_prob_a for r in rows], [r.a_won for r in rows])

    for name, model_fn in [
        ("sharp_only", lambda r: None),
        ("full_blend+sharp", model_blend),
        ("ranking+sharp", ranking_model),
    ]:
        ps, acts, fired = [], [], 0
        for r in rows:
            p, contributed = blend_with_sharp(model_fn(r), r.sharp_prob_a)
            ps.append(p)
            acts.append(r.a_won)
            fired += contributed
        b = brier_score(ps, acts)
        a = accuracy_score(ps, acts)
        delta = f"{(sharp_b - b) * 1000:+8.3f}/1000" if name != "sharp_only" else ""
        print(f"  {name:20s}  {len(ps):6d}  {fired:6d}  {b:8.4f}  {a * 100:6.1f}%  {delta}")

    # Standalone trend model on the rows where it fired, vs sharp on the SAME rows
    fired_rows = [(ranking_model(r), r) for r in rows]
    fired_rows = [(m, r) for m, r in fired_rows if m is not None]
    if fired_rows:
        acts = [r.a_won for _, r in fired_rows]
        b_trend = brier_score([m for m, _ in fired_rows], acts)
        b_sharp_sub = brier_score([r.sharp_prob_a for _, r in fired_rows], acts)
        a_trend = accuracy_score([m for m, _ in fired_rows], acts)
        print(f"  {'trend standalone':20s}  {len(fired_rows):6d}  {len(fired_rows):6d}  "
              f"{b_trend:8.4f}  {a_trend * 100:6.1f}%  "
              f"{(b_sharp_sub - b_trend) * 1000:+8.3f}/1000 (same-row sharp {b_sharp_sub:.4f})")


def calibration_buckets(rows: list[TennisWalkRow]) -> None:
    """Predicted-prob bins → actual win rate, for each blend."""
    edges = [(0.0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
             (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]

    print("\n=== Calibration buckets (predicted prob → actual win rate) ===")
    for name, model_fn in [("full_blend+sharp", model_blend),
                           ("ranking+sharp", ranking_model)]:
        print(f"\n  --- {name} ---")
        print(f"  {'Bucket':>10s}  {'n':>5s}  {'avg pred':>8s}  {'actual':>7s}  "
              f"{'gap':>7s}  {'Brier':>8s}")
        buckets: dict[tuple, list[tuple[float, bool]]] = defaultdict(list)
        for r in rows:
            p, _ = blend_with_sharp(model_fn(r), r.sharp_prob_a)
            for lo, hi in edges:
                if lo <= p < hi:
                    buckets[(lo, hi)].append((p, r.a_won))
                    break
        for lo, hi in edges:
            sub = buckets.get((lo, hi), [])
            if not sub:
                print(f"  {lo:.1f}-{hi:.1f}{'':>3s}  (empty)")
                continue
            ps = [p for p, _ in sub]
            acts = [a for _, a in sub]
            avg_p = sum(ps) / len(ps)
            actual = sum(acts) / len(acts)
            b = brier_score(ps, acts)
            print(f"  {lo:5.1f}-{hi:.1f}  {len(sub):5d}  {avg_p:8.3f}  {actual:7.3f}  "
                  f"{actual - avg_p:+7.3f}  {b:8.4f}")


def disagreement_buckets(rows: list[TennisWalkRow]) -> None:
    """|model - sharp| buckets → per-bucket Brier of blend vs sharp."""
    edges = [(0.00, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 1.00)]

    print("\n=== Disagreement buckets (|model − sharp| → Brier, blend vs sharp) ===")
    for name, model_fn in [("full_blend+sharp", model_blend),
                           ("ranking+sharp", ranking_model)]:
        print(f"\n  --- {name} ---")
        print(f"  {'Bucket':>10s}  {'n':>5s}  {'sharp':>8s}  {'blend':>8s}  "
              f"{'Δ/1000':>8s}  best")
        buckets: dict[tuple, list[TennisWalkRow]] = defaultdict(list)
        skipped = 0
        for r in rows:
            m = model_fn(r)
            if m is None:
                skipped += 1
                continue
            diff = abs(m - r.sharp_prob_a)
            for lo, hi in edges:
                if lo <= diff < hi:
                    buckets[(lo, hi)].append(r)
                    break
        for lo, hi in edges:
            sub = buckets.get((lo, hi), [])
            if not sub:
                print(f"  {lo:.2f}-{hi:.2f}  (empty)")
                continue
            acts = [r.a_won for r in sub]
            sharp_ps = [r.sharp_prob_a for r in sub]
            blend_ps = [blend_with_sharp(model_fn(r), r.sharp_prob_a)[0] for r in sub]
            b_sharp = brier_score(sharp_ps, acts)
            b_blend = brier_score(blend_ps, acts)
            best = "blend" if b_blend < b_sharp else ("sharp" if b_sharp < b_blend else "tie")
            print(f"  {lo:.2f}-{hi:.2f}  {len(sub):5d}  {b_sharp:8.4f}  {b_blend:8.4f}  "
                  f"{(b_sharp - b_blend) * 1000:+8.3f}  {best}")
        print(f"  (model didn't fire on {skipped} rows — those fall back to sharp)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, default="2024")
    ap.add_argument("--eval", type=str, default="2025,2026")
    args = ap.parse_args()

    train_years = [int(y) for y in args.train.split(",") if y]
    eval_years = {int(y) for y in args.eval.split(",") if y}
    all_years = sorted(set(train_years) | eval_years)

    print(f"Loading tennis xlsx for years {all_years}...")
    matches = load_tennis_matches(all_years)
    print(f"Total matches: {len(matches)}")
    if not matches:
        return 1

    print("\nWalk-forward replay (seeded state, in-memory only)...")
    rows = walk_forward(matches)
    eval_rows = filter_eval(rows, eval_years)
    print(f"Eval slice ({sorted(eval_years)}): {len(eval_rows)} matches")

    overall_summary(eval_rows)
    calibration_buckets(eval_rows)
    disagreement_buckets(eval_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
