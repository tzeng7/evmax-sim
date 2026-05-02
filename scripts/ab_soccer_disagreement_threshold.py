"""A/B walk-forward: soccer _disagreement_sharp_weight threshold 0.10 vs 0.05.

Hypothesis (from live predictions.db, n=89 resolved):
  - "model > sharp by 5+pts" bucket has −9.2pp gap (model overconfident).
  - The current disagreement-ramp threshold (0.10) doesn't kick in there, so
    the blend keeps the model's view at exactly the disagreement level where
    it's losing.
  - Lowering the threshold to 0.05 should pull weight back toward sharp on
    that subset and tighten holdout Brier.

This script:
  1. Reuses walk_forward_predictions to replay 2324–2526 chronologically.
  2. For each game, computes the model-side blend and the sharp prob.
  3. Reports the model-vs-sharp disagreement profile (how much of the corpus
     sits in each |model − sharp| bucket, and what the actual win rate is
     in each — directly testing whether sharp is right where the two
     disagree).
  4. Re-runs the blended_ensemble_3way blend twice — once with threshold=0.10
     (current production) and once with threshold=0.05 — and compares
     holdout Brier on 2526 alongside per-bucket calibration.

This is a read-only diagnostic — does NOT modify any production code or
state files. Intended to be run before shipping the threshold change.

Usage:
    .venv/bin/python scripts/ab_soccer_disagreement_threshold.py
"""

from __future__ import annotations

from typing import Optional

from evmax.agents.models.ensemble_agent import EnsembleModelAgent
from evmax.backtest.loader import SOCCER_LEAGUES, fetch_soccer_csv
from evmax.backtest.models import BacktestRow
from evmax.backtest.sources.soccer_csv import parse_soccer_csv
from evmax.backtest.sources.soccer_walkforward import (
    SoccerWalkForwardRow,
    _model_side_blend,
    base_sharp_weight_for,
    walk_forward_predictions,
)


def _load_rows(seasons: list[str]) -> list[BacktestRow]:
    rows: list[BacktestRow] = []
    for season in seasons:
        for league in SOCCER_LEAGUES:
            try:
                path = fetch_soccer_csv(season, league)
            except Exception as e:
                print(f"  skip {season}/{league}: {e}")
                continue
            rows.extend(parse_soccer_csv(path, league, season))
    return rows


def _blend_with_params(
    r: SoccerWalkForwardRow,
    threshold: float,
    saturate_at: float,
    cap: float,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """blended_ensemble_3way with custom disagreement-ramp params."""
    side = _model_side_blend(r)
    if side is None or r.sharp_ph is None:
        return (None, None, None)
    model_h, model_d, model_a = side
    eff_sw = EnsembleModelAgent._disagreement_sharp_weight(
        model_h, model_a, model_d,
        r.sharp_ph, r.sharp_pa, r.sharp_pd,
        base_sharp_weight=base_sharp_weight_for(r),
        threshold=threshold,
        saturate_at=saturate_at,
        cap=cap,
    )
    mw = 1.0 - eff_sw
    ph = eff_sw * r.sharp_ph + mw * model_h
    pd = eff_sw * r.sharp_pd + mw * model_d
    pa = eff_sw * r.sharp_pa + mw * model_a
    s = ph + pd + pa
    return (ph / s, pd / s, pa / s) if s > 1e-9 else (None, None, None)


def _team_win_pairs_blend(
    predictions: list[SoccerWalkForwardRow],
    threshold: float,
    saturate_at: float,
    cap: float,
) -> tuple[list[float], list[int]]:
    probs: list[float] = []
    outcomes: list[int] = []
    for r in predictions:
        ph, _pd, pa = _blend_with_params(r, threshold, saturate_at, cap)
        if ph is None or pa is None:
            continue
        probs.append(ph)
        outcomes.append(1 if r.result == "H" else 0)
        probs.append(pa)
        outcomes.append(1 if r.result == "A" else 0)
    return probs, outcomes


def _team_win_pairs_sharp_only(
    predictions: list[SoccerWalkForwardRow],
) -> tuple[list[float], list[int]]:
    probs: list[float] = []
    outcomes: list[int] = []
    for r in predictions:
        if r.sharp_ph is None or r.sharp_pa is None:
            continue
        probs.append(r.sharp_ph)
        outcomes.append(1 if r.result == "H" else 0)
        probs.append(r.sharp_pa)
        outcomes.append(1 if r.result == "A" else 0)
    return probs, outcomes


def _brier(probs: list[float], outcomes: list[int]) -> float:
    return sum((p - o) ** 2 for p, o in zip(probs, outcomes)) / max(len(probs), 1)


def _disagreement_profile(predictions: list[SoccerWalkForwardRow]) -> None:
    """Tests the core hypothesis: when stat-blend disagrees with sharp, who is right?

    For each game, compute home leg model prob and home leg sharp prob.
    Bucket by signed gap (model_h - sharp_h). In each bucket report:
      n, mean(model_h), mean(sharp_h), actual home win rate.
    If sharp is right where they disagree, the actual rate sits next to sharp."""
    buckets: dict[str, list[tuple[float, float, int]]] = {
        "model > sharp by 7+pts": [],
        "model > sharp by 4-7pts": [],
        "within 4pts": [],
        "model < sharp by 4-7pts": [],
        "model < sharp by 7+pts": [],
    }
    for r in predictions:
        side = _model_side_blend(r)
        if side is None or r.sharp_ph is None:
            continue
        model_h, _model_d, _model_a = side
        gap = model_h - r.sharp_ph
        won_h = 1 if r.result == "H" else 0
        if gap >= 0.07:
            buckets["model > sharp by 7+pts"].append((model_h, r.sharp_ph, won_h))
        elif gap >= 0.04:
            buckets["model > sharp by 4-7pts"].append((model_h, r.sharp_ph, won_h))
        elif gap <= -0.07:
            buckets["model < sharp by 7+pts"].append((model_h, r.sharp_ph, won_h))
        elif gap <= -0.04:
            buckets["model < sharp by 4-7pts"].append((model_h, r.sharp_ph, won_h))
        else:
            buckets["within 4pts"].append((model_h, r.sharp_ph, won_h))

    print(f"  {'bucket':<25} {'n':>6} {'model_h':>8} {'sharp_h':>8} {'actual':>8}")
    print(f"  {'-' * 25} {'-' * 6} {'-' * 8} {'-' * 8} {'-' * 8}")
    for label, items in buckets.items():
        if not items:
            continue
        n = len(items)
        mean_model = sum(x[0] for x in items) / n
        mean_sharp = sum(x[1] for x in items) / n
        actual = sum(x[2] for x in items) / n
        print(f"  {label:<25} {n:>6d} {mean_model:>8.3f} {mean_sharp:>8.3f} {actual:>8.3f}")


_BUCKETS = [(0.0, 0.20), (0.20, 0.35), (0.35, 0.50), (0.50, 0.65), (0.65, 1.00)]


def _bucket_table(probs: list[float], outcomes: list[int]) -> None:
    print(f"  {'bucket':<11} {'n':>6} {'pred':>7} {'actual':>7} {'gap':>7}")
    for lo, hi in _BUCKETS:
        idx = [i for i, p in enumerate(probs) if lo <= p < hi]
        if not idx:
            continue
        n = len(idx)
        p = sum(probs[i] for i in idx) / n
        a = sum(outcomes[i] for i in idx) / n
        print(f"  {lo:.2f}-{hi:.2f}    {n:>6d} {p:>7.3f} {a:>7.3f} {a - p:>+7.3f}")


def main() -> None:
    print("=== A/B: soccer disagreement threshold 0.10 (current) vs 0.05 ===\n")

    print("loading 2324 + 2425 + 2526 ...")
    rows = _load_rows(["2324", "2425", "2526"])
    print(f"  {len(rows)} rows total")
    predictions = walk_forward_predictions(rows)
    holdout = [p for p in predictions if p.date.year >= 2025]  # 2526 season
    print(f"  {len(predictions)} walk-forward predictions ({len(holdout)} in 2526 holdout)\n")

    print("=== disagreement profile (full corpus) ===")
    print("If sharp is right when they disagree, actual rate should sit next to sharp_h:\n")
    _disagreement_profile(predictions)
    print()

    print("=== holdout Brier (2526) ===")
    print("\nsharp_only (theoretical ceiling — no model contribution at all):")
    sp, so = _team_win_pairs_sharp_only(holdout)
    print(f"  Brier = {_brier(sp, so):.5f}  (n={len(sp)})")
    _bucket_table(sp, so)

    variants = [
        ("current  (thr=0.10, sat=0.30, cap=0.95)", 0.10, 0.30, 0.95),
        ("V1       (thr=0.05, sat=0.30, cap=0.95)", 0.05, 0.30, 0.95),
        ("V2       (thr=0.05, sat=0.20, cap=0.99)", 0.05, 0.20, 0.99),
        ("V3       (thr=0.04, sat=0.15, cap=0.99)", 0.04, 0.15, 0.99),
        ("V4       (thr=0.04, sat=0.10, cap=1.00)", 0.04, 0.10, 1.00),
    ]
    for label, thr, sat, cap in variants:
        probs, outcomes = _team_win_pairs_blend(holdout, thr, sat, cap)
        brier = _brier(probs, outcomes)
        print(f"\n{label}: Brier = {brier:.5f}  (n={len(probs)})")
        _bucket_table(probs, outcomes)


if __name__ == "__main__":
    main()
