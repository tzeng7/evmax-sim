"""Per-disagreement-bucket performance: where does the production blend
(V3 override + xG) actually beat sharp-only, and where would we be better
off skipping the blend?

For each row, we:
  1. Compute the model-side blend (elo+form+poisson+xg, with SOCCER_W weights)
  2. Place the row in a disagreement bucket by signed gap (model_h - sharp_h)
  3. Score the SAME row under three blends: V3 production, sharp-only, stat-only
  4. Aggregate Brier per bucket

Reports both binary Brier (P(home) vs home_won — what the prop bet sees)
and 3-way Brier (full distribution — calibration view).

Usage:
    .venv/bin/python scripts/per_bucket_disagreement_perf.py
    .venv/bin/python scripts/per_bucket_disagreement_perf.py --holdout 2526
    .venv/bin/python scripts/per_bucket_disagreement_perf.py --holdout 2425+2526
"""

from __future__ import annotations

import argparse
from typing import Optional

from evmax.agents.models.ensemble_agent import EnsembleModelAgent
from evmax.backtest.loader import SOCCER_LEAGUES, fetch_soccer_csv
from evmax.backtest.models import BacktestRow
from evmax.backtest.sources.soccer_csv import parse_soccer_csv
from evmax.backtest.sources.soccer_walkforward import (
    SoccerWalkForwardRow,
    _model_side_blend,
    base_sharp_weight_for,
    stat_ensemble,
    walk_forward_predictions,
)


def _load_rows(seasons: list[str]) -> list[BacktestRow]:
    rows: list[BacktestRow] = []
    for season in seasons:
        for league in SOCCER_LEAGUES:
            try:
                path = fetch_soccer_csv(season, league)
            except Exception:
                continue
            rows.extend(parse_soccer_csv(path, league, season))
    return rows


def _v3_blend(r: SoccerWalkForwardRow) -> Optional[tuple[float, float, float]]:
    return _custom_blend(r, threshold=0.04, saturate_at=0.15, cap=0.99)


def _v4_blend(r: SoccerWalkForwardRow) -> Optional[tuple[float, float, float]]:
    """V4 — more aggressive ramp: thr=0.04, sat=0.10, cap=1.00."""
    return _custom_blend(r, threshold=0.04, saturate_at=0.10, cap=1.00)


def _custom_blend(
    r: SoccerWalkForwardRow,
    threshold: float, saturate_at: float, cap: float,
) -> Optional[tuple[float, float, float]]:
    side = _model_side_blend(r)
    if side is None or r.sharp_ph is None:
        return None
    model_h, model_d, model_a = side
    eff_sw = EnsembleModelAgent._disagreement_sharp_weight(
        model_h, model_a, model_d,
        r.sharp_ph, r.sharp_pa, r.sharp_pd,
        base_sharp_weight=base_sharp_weight_for(r),
        threshold=threshold, saturate_at=saturate_at, cap=cap,
    )
    mw = 1.0 - eff_sw
    ph = eff_sw * r.sharp_ph + mw * model_h
    pd = eff_sw * r.sharp_pd + mw * model_d
    pa = eff_sw * r.sharp_pa + mw * model_a
    s = ph + pd + pa
    if s <= 1e-9:
        return None
    return ph / s, pd / s, pa / s


def _three_way_brier(ph: float, pd: float, pa: float, result: str) -> float:
    yh = 1.0 if result == "H" else 0.0
    yd = 1.0 if result == "D" else 0.0
    ya = 1.0 if result == "A" else 0.0
    return (ph - yh) ** 2 + (pd - yd) ** 2 + (pa - ya) ** 2


def _binary_brier(ph: float, result: str) -> float:
    yh = 1.0 if result == "H" else 0.0
    return (ph - yh) ** 2


def _bucket_label(gap_h: float) -> str:
    if gap_h >= 0.07:
        return "model > sharp by 7+pts"
    if gap_h >= 0.04:
        return "model > sharp by 4-7pts"
    if gap_h <= -0.07:
        return "model < sharp by 7+pts"
    if gap_h <= -0.04:
        return "model < sharp by 4-7pts"
    return "within 4pts"


BUCKET_ORDER = [
    "model > sharp by 7+pts",
    "model > sharp by 4-7pts",
    "within 4pts",
    "model < sharp by 4-7pts",
    "model < sharp by 7+pts",
]


def _filter_holdout(
    predictions: list[SoccerWalkForwardRow], spec: str,
) -> list[SoccerWalkForwardRow]:
    """spec like '2526' or '2425+2526' — match by season-aware date window.
    A football season SSEE runs Aug-20SS to May-20EE (e.g. 2526 = 2025-08 → 2026-05)."""
    windows: list[tuple[int, int]] = []
    for season in spec.split("+"):
        season = season.strip()
        if len(season) != 4:
            raise ValueError(f"bad season spec: {season!r}")
        start_year = 2000 + int(season[:2])
        end_year = 2000 + int(season[2:])
        windows.append((start_year, end_year))

    out = []
    for p in predictions:
        for start_year, end_year in windows:
            if (p.date.year == start_year and p.date.month >= 7) or (
                p.date.year == end_year and p.date.month <= 6
            ):
                out.append(p)
                break
    return out


def _print_bucket_table(
    bucket_rows: dict[str, list[SoccerWalkForwardRow]],
    metric: str,  # "binary" or "3way"
) -> None:
    if metric == "binary":
        scorer = lambda probs, r: _binary_brier(probs[0], r.result)
        ceiling = 0.25  # binary baseline (random 50/50)
    else:
        scorer = lambda probs, r: _three_way_brier(*probs, r.result)
        ceiling = 2 / 3  # 3-way (random 1/3 each)

    print(f"  {'bucket':<25} {'n':>5} {'V3 (prod)':>10} {'V4':>10} "
          f"{'sharp-only':>11} {'stat-only':>10} {'best':>7}")
    print(f"  {'-' * 25} {'-' * 5} {'-' * 10} {'-' * 10} {'-' * 11} "
          f"{'-' * 10} {'-' * 7}")

    totals = {"v3": 0.0, "v4": 0.0, "sharp": 0.0, "stat": 0.0, "n": 0}

    for bucket in BUCKET_ORDER:
        items = bucket_rows[bucket]
        if not items:
            continue
        v3_b = v4_b = sharp_b = stat_b = 0.0
        n = 0
        for r in items:
            v3 = _v3_blend(r); v4 = _v4_blend(r); stat = stat_ensemble(r)
            if v3 is None or v4 is None or stat[0] is None:
                continue
            n += 1
            v3_b += scorer(v3, r)
            v4_b += scorer(v4, r)
            sharp_b += scorer((r.sharp_ph, r.sharp_pd, r.sharp_pa), r)
            stat_b += scorer(stat, r)

        if n == 0:
            continue
        v3_avg = v3_b / n; v4_avg = v4_b / n; sharp_avg = sharp_b / n; stat_avg = stat_b / n
        ranking = [("V3", v3_avg), ("V4", v4_avg), ("sharp", sharp_avg), ("stat", stat_avg)]
        winner = min(ranking, key=lambda x: x[1])[0]

        print(f"  {bucket:<25} {n:>5d} "
              f"{v3_avg:>10.5f} {v4_avg:>10.5f} "
              f"{sharp_avg:>11.5f} {stat_avg:>10.5f} {winner:>7}")

        totals["v3"] += v3_b
        totals["v4"] += v4_b
        totals["sharp"] += sharp_b
        totals["stat"] += stat_b
        totals["n"] += n

    print(f"  {'-' * 25} {'-' * 5} {'-' * 10} {'-' * 10} {'-' * 11} "
          f"{'-' * 10} {'-' * 7}")
    if totals["n"] > 0:
        n = totals["n"]
        ranking = [
            ("V3", totals["v3"] / n),
            ("V4", totals["v4"] / n),
            ("sharp", totals["sharp"] / n),
            ("stat", totals["stat"] / n),
        ]
        winner = min(ranking, key=lambda x: x[1])[0]
        print(f"  {'OVERALL':<25} {n:>5d} "
              f"{ranking[0][1]:>10.5f} {ranking[1][1]:>10.5f} "
              f"{ranking[2][1]:>11.5f} {ranking[3][1]:>10.5f} {winner:>7}")
        print(f"  (random baseline at this metric: {ceiling:.4f})")


def main(holdout_spec: str) -> None:
    print(f"=== per-disagreement-bucket performance (xG wired in) ===")
    print(f"holdout: {holdout_spec}\n")

    print("loading 2324 + 2425 + 2526 (full corpus for walk-forward) ...")
    rows = _load_rows(["2324", "2425", "2526"])
    print(f"  {len(rows)} rows total")
    predictions = walk_forward_predictions(rows)
    holdout = _filter_holdout(predictions, holdout_spec)
    print(f"  {len(predictions)} walk-forward predictions"
          f" → {len(holdout)} in holdout window\n")

    bucket_rows: dict[str, list[SoccerWalkForwardRow]] = {b: [] for b in BUCKET_ORDER}
    skipped = 0
    for r in holdout:
        side = _model_side_blend(r)
        if side is None or r.sharp_ph is None:
            skipped += 1
            continue
        gap = side[0] - r.sharp_ph
        bucket_rows[_bucket_label(gap)].append(r)
    print(f"  bucketed: {len(holdout) - skipped} rows ({skipped} skipped)\n")

    print("--- BINARY Brier per bucket (P(home) vs home_won) ---")
    _print_bucket_table(bucket_rows, "binary")

    print("\n--- 3-WAY Brier per bucket (sum over H/D/A legs) ---")
    _print_bucket_table(bucket_rows, "3way")

    print()
    print("Reading the table:")
    print("  - V3       = production blend (thr=0.04, sat=0.15, cap=0.99) + xG")
    print("  - V4       = more aggressive ramp (thr=0.04, sat=0.10, cap=1.00) + xG")
    print("  - sharp    = devigged Pinnacle, no model contribution")
    print("  - stat     = elo+form+poisson+xg, no sharp contribution")
    print("  - best     = lowest Brier in that bucket (lower = better)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--holdout", default="2526",
        help="season spec, e.g. '2526' or '2425+2526'",
    )
    args = parser.parse_args()
    main(args.holdout)
