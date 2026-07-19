"""pitcher_v2 A/B — walk-forward comparison of pitcher-model variants.

Runs the ESPN baseball walk-forward once per variant over a CONTINUOUS
2025→2026 span (running totals accumulate across the boundary, matching how
the live model carries a warm seed into a season), then scores each variant
on the 2025 train year and the 2026 holdout separately. The holdout is
touched once; nothing is swept on it.

Variants: v1 (starter-only, the incumbent) · v2 (pen+park+off+xera) · any
comma-set of components (e.g. "park,off,xera" — pen has a prior null result
in this harness and must independently beat it to ship).

Ship gate (from the diversification plan, judged on the 2026 holdout):
  (a) v2 BLEND Brier <= v1 blend Brier
  (b) v2 pitcher STANDALONE Brier better than v1 by >= 2.0/1000
  (c) v2 coverage (games the pitcher model fires) >= 95% of v1's
If (b) fails for the composite, ship only the components that individually
clear it — each is independently switchable in the live agent.

Usage:
    python scripts/backtest_baseball_ab.py                     # v1 vs v2
    python scripts/backtest_baseball_ab.py --variants v1,v2,park,off,xera,pen
    python scripts/backtest_baseball_ab.py --quick             # 2 months/split (smoke)

First uncached run fetches ~2,400 ESPN boxscores (~1-2h); cached re-runs
take minutes (data/backtest/cache/espn_boxscore/).
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evmax.backtest.sources.espn_walkforward import (  # noqa: E402
    BASEBALL_WEIGHT_OVERRIDES,
    WalkForwardResult,
    run_walkforward,
)

TRAIN_MONTHS = [f"2025{m:02d}" for m in range(3, 11)]
HOLDOUT_MONTHS = [f"2026{m:02d}" for m in range(3, 8)]  # season-to-date
HOLDOUT_START = date(2026, 1, 1)


def _brier(rows: list[WalkForwardResult], attr: str) -> tuple[float, int]:
    vals = [
        (getattr(r, attr), r.home_won)
        for r in rows
        if getattr(r, attr) is not None
    ]
    if not vals:
        return float("nan"), 0
    return sum((p - (1.0 if won else 0.0)) ** 2 for p, won in vals) / len(vals), len(vals)


def _split(results: list[WalkForwardResult]) -> tuple[list, list]:
    train = [r for r in results if r.date < HOLDOUT_START]
    holdout = [r for r in results if r.date >= HOLDOUT_START]
    return train, holdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants", default="v1;v2",
        help="Semicolon-separated variants; each is 'v1', 'v2', or a comma-set "
             "of components, e.g. 'v1;v2;park,off,xera;pen'.",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="2 months per split — plumbing smoke test, not a verdict.",
    )
    args = parser.parse_args()

    train_months = TRAIN_MONTHS[:2] if args.quick else TRAIN_MONTHS
    holdout_months = HOLDOUT_MONTHS[:2] if args.quick else HOLDOUT_MONTHS
    months = train_months + holdout_months
    variants = [v.strip() for v in args.variants.split(";") if v.strip()]

    print(f"pitcher_v2 A/B — months {months[0]}..{months[-1]}  variants: {variants}")
    print(f"blend weights: {BASEBALL_WEIGHT_OVERRIDES}\n")

    stats: dict[str, dict] = {}
    for variant in variants:
        print(f"=== variant {variant!r} — running walk-forward ...")
        report = run_walkforward("baseball", months, pitcher_variant=variant)
        train, holdout = _split(report.results)
        row: dict = {}
        for label, rows in (("train", train), ("holdout", holdout)):
            pb, pn = _brier(rows, "pitcher_prob_home")
            eb, en = _brier(rows, "ensemble_prob_home")
            row[label] = {
                "pitcher_brier": pb, "pitcher_n": pn,
                "blend_brier": eb, "blend_n": en,
                "games": len(rows),
            }
        stats[variant] = row
        for label in ("train", "holdout"):
            s = row[label]
            print(
                f"  {label:8s} games={s['games']:4d}  "
                f"pitcher {s['pitcher_brier']:.4f} (n={s['pitcher_n']})  "
                f"blend {s['blend_brier']:.4f} (n={s['blend_n']})"
            )
        print()

    if "v1" in stats:
        v1 = stats["v1"]
        print("=== deltas vs v1 (negative pitcher/blend Δ = better) ===")
        for variant, row in stats.items():
            if variant == "v1":
                continue
            for label in ("train", "holdout"):
                s, b = row[label], v1[label]
                d_pitch = (s["pitcher_brier"] - b["pitcher_brier"]) * 1000
                d_blend = (s["blend_brier"] - b["blend_brier"]) * 1000
                cov = s["pitcher_n"] / b["pitcher_n"] * 100 if b["pitcher_n"] else 0
                print(
                    f"  {variant:14s} {label:8s} Δpitcher {d_pitch:+7.2f}/1000  "
                    f"Δblend {d_blend:+7.2f}/1000  coverage {cov:5.1f}%"
                )
        print()

        if "v2" in stats and not args.quick:
            h2, h1 = stats["v2"]["holdout"], v1["holdout"]
            gate_a = h2["blend_brier"] <= h1["blend_brier"] + 1e-9
            gate_b = (h1["pitcher_brier"] - h2["pitcher_brier"]) * 1000 >= 2.0
            gate_c = (
                h1["pitcher_n"] == 0
                or h2["pitcher_n"] / h1["pitcher_n"] >= 0.95
            )
            print("=== ship gate (2026 holdout) ===")
            print(f"  (a) blend <= v1 blend:            {'PASS' if gate_a else 'FAIL'}")
            print(f"  (b) standalone >= 2.0/1000 better: {'PASS' if gate_b else 'FAIL'}")
            print(f"  (c) coverage >= 95% of v1:         {'PASS' if gate_c else 'FAIL'}")
            print(
                "  → SHIP v2" if gate_a and gate_b and gate_c else
                "  → do NOT ship the composite — check per-component variants"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
