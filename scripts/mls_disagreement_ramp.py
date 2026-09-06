"""MLS-only disagreement-ramp sweep — pick `disagreement_ramp` for the
`secondary` tier in data/soccer_league_tiers.yaml.

Why a separate run: the sector-level soccer ramp (0.04 / 0.10 / 1.00) was
calibrated on POOLED top-5 + UEFA data (n=4406) where Pinnacle is the
informational ceiling. MLS is priced at sharp_weight 0.40 precisely because
that ceiling is assumed NOT to hold there — but the pooled ramp still fires on
every MLS game with a >=4pt model/sharp gap and is 100% sharp at 10pt, which
erases the 0.60 model share on exactly the games where a secondary-league edge
could show (live MLS divergence 0.8pp at sharp_weight 0.40).

Protocol (walk-forward, MLS rows only, models warm up on 2024):
  train   = calendar 2025  (Pinnacle close anchor)
  holdout = calendar 2026  (CONSENSUS close anchor — USA.csv carries no
            Pinnacle columns for the in-progress season; see
            parse_soccer_extra_csv. Numbers are "vs the close", not Pinnacle.)
Every candidate (threshold, saturate_at, cap) is scored with the SAME
EnsembleModelAgent._disagreement_sharp_weight the live blend uses, on top of
the MLS base sharp_weight 0.40. Reference columns: `sector` = the pooled ramp
MLS inherits today, `flat` = no ramp (pure 0.40), `sharp` = close only,
`stat` = model side only. Paired Brier deltas vs sharp-only carry a z-score.

Usage:
    .venv/bin/python scripts/mls_disagreement_ramp.py [--metric 3way|binary]
                                                       [--train 2025] [--holdout 2026]
"""

from __future__ import annotations

import argparse
import math
from itertools import product
from pathlib import Path
from typing import Optional

from evmax.agents.models.ensemble_agent import EnsembleModelAgent
from evmax.backtest.loader import fetch_soccer_extra_csv
from evmax.backtest.sources.soccer_csv import parse_soccer_extra_csv
from evmax.backtest.sources.soccer_walkforward import (
    SoccerWalkForwardRow,
    _model_side_blend,
    base_sharp_weight_for,
    stat_ensemble,
    walk_forward_predictions,
)

SECTOR_RAMP = EnsembleModelAgent.DISAGREEMENT_OVERRIDES["soccer"]
# A ramp that never fires — threshold at 1.0 can't be exceeded by a probability gap.
FLAT_RAMP = (1.0, 1.0, 1.0)

GRID_THRESHOLD = (0.02, 0.04, 0.06, 0.08, 0.10, 0.15)
GRID_SATURATE = (0.10, 0.15, 0.20, 0.30)
GRID_CAP = (0.70, 0.85, 0.95, 1.00)


def load_mls_rows(seasons: list[str]) -> list:
    cached = Path("data/backtest/soccer/extra/USA.csv")
    try:
        path = fetch_soccer_extra_csv("USA")
    except Exception as e:  # football-data.co.uk 503s intermittently
        if not cached.exists():
            raise
        print(f"  (fetch failed: {e!s:.60} — using cached {cached})")
        path = cached
    return parse_soccer_extra_csv(path, "USA", seasons)


def blend_with_ramp(
    r: SoccerWalkForwardRow, ramp: tuple[float, float, float],
) -> Optional[tuple[float, float, float]]:
    side = _model_side_blend(r)
    if side is None or r.sharp_ph is None:
        return None
    model_h, model_d, model_a = side
    eff_sw = EnsembleModelAgent._disagreement_sharp_weight(
        model_h, model_a, model_d,
        r.sharp_ph, r.sharp_pa, r.sharp_pd,
        base_sharp_weight=base_sharp_weight_for(r),
        params=ramp,
    )
    mw = 1.0 - eff_sw
    ph = eff_sw * r.sharp_ph + mw * model_h
    pd = eff_sw * r.sharp_pd + mw * model_d
    pa = eff_sw * r.sharp_pa + mw * model_a
    s = ph + pd + pa
    if s <= 1e-9:
        return None
    return ph / s, pd / s, pa / s


def brier(probs: tuple[float, float, float], result: str, metric: str) -> float:
    ph, pd, pa = probs
    yh, yd, ya = (result == "H"), (result == "D"), (result == "A")
    if metric == "binary":
        return (ph - yh) ** 2
    return (ph - yh) ** 2 + (pd - yd) ** 2 + (pa - ya) ** 2


def score(rows: list[SoccerWalkForwardRow], ramp, metric: str) -> tuple[float, float, float, int]:
    """(mean Brier, mean paired delta vs sharp, z of that delta, n)."""
    diffs, total = [], 0.0
    for r in rows:
        p = blend_with_ramp(r, ramp)
        if p is None:
            continue
        b = brier(p, r.result, metric)
        s = brier((r.sharp_ph, r.sharp_pd, r.sharp_pa), r.result, metric)
        total += b
        diffs.append(b - s)
    n = len(diffs)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean_d = sum(diffs) / n
    var = sum((d - mean_d) ** 2 for d in diffs) / max(n - 1, 1)
    z = mean_d / math.sqrt(var / n) if var > 0 else 0.0
    return total / n, mean_d, z, n


def score_fixed(rows, kind: str, metric: str) -> tuple[float, int]:
    total, n = 0.0, 0
    for r in rows:
        if _model_side_blend(r) is None or r.sharp_ph is None:
            continue
        if kind == "sharp":
            p = (r.sharp_ph, r.sharp_pd, r.sharp_pa)
        else:
            p = stat_ensemble(r)
            if p[0] is None:
                continue
        total += brier(p, r.result, metric)
        n += 1
    return (total / n if n else float("nan")), n


def bucket(gap: float) -> str:
    if gap >= 0.07:
        return "model>sharp 7+"
    if gap >= 0.04:
        return "model>sharp 4-7"
    if gap <= -0.07:
        return "model<sharp 7+"
    if gap <= -0.04:
        return "model<sharp 4-7"
    return "within 4"


BUCKETS = ["model>sharp 7+", "model>sharp 4-7", "within 4", "model<sharp 4-7", "model<sharp 7+"]


def bucket_table(rows, metric: str, ramps: dict[str, tuple]) -> None:
    by: dict[str, list] = {b: [] for b in BUCKETS}
    for r in rows:
        side = _model_side_blend(r)
        if side is None or r.sharp_ph is None:
            continue
        by[bucket(side[0] - r.sharp_ph)].append(r)
    names = list(ramps)
    print(f"  {'bucket':<16}{'n':>5}" + "".join(f"{nm:>10}" for nm in names) + f"{'sharp':>10}{'stat':>10}")
    for b in BUCKETS:
        items = by[b]
        if not items:
            continue
        cells = [score(items, ramps[nm], metric)[0] for nm in names]
        sh, _ = score_fixed(items, "sharp", metric)
        st, _ = score_fixed(items, "stat", metric)
        print(f"  {b:<16}{len(items):>5}" + "".join(f"{c:>10.4f}" for c in cells) + f"{sh:>10.4f}{st:>10.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="3way", choices=["3way", "binary"])
    ap.add_argument("--train", type=int, default=2025)
    ap.add_argument("--holdout", type=int, default=2026)
    args = ap.parse_args()
    metric = args.metric

    print("=== MLS-only disagreement-ramp sweep ===")
    rows = load_mls_rows(["2324", "2425", "2526"])
    print(f"  {len(rows)} MLS rows loaded")
    preds = walk_forward_predictions(rows)
    train = [p for p in preds if p.date.year == args.train]
    hold = [p for p in preds if p.date.year == args.holdout]
    print(f"  walk-forward: {len(preds)} predictions → train {args.train}: {len(train)}, "
          f"holdout {args.holdout}: {len(hold)} (holdout anchor = consensus close)\n")

    refs = {"sector": SECTOR_RAMP, "flat": FLAT_RAMP}
    for label, rs in (("TRAIN", train), ("HOLDOUT", hold)):
        print(f"--- {label} {args.train if label == 'TRAIN' else args.holdout} — {metric} Brier, per disagreement bucket ---")
        bucket_table(rs, metric, refs)
        print()

    # Sweep on train, report holdout for every candidate.
    print(f"--- sweep ({metric} Brier; Δ = blend − sharp-only, negative = beats sharp) ---")
    cands = [("sector", SECTOR_RAMP), ("flat", FLAT_RAMP)]
    for thr, sat, cap in product(GRID_THRESHOLD, GRID_SATURATE, GRID_CAP):
        if sat <= thr:
            continue
        cands.append((f"{thr:.2f}/{sat:.2f}/{cap:.2f}", (thr, sat, cap)))
    results = []
    for name, ramp in cands:
        tb, td, tz, tn = score(train, ramp, metric)
        hb, hd, hz, hn = score(hold, ramp, metric)
        results.append((name, ramp, tb, td, tz, tn, hb, hd, hz, hn))
    sharp_t, _ = score_fixed(train, "sharp", metric)
    sharp_h, _ = score_fixed(hold, "sharp", metric)
    stat_t, _ = score_fixed(train, "stat", metric)
    stat_h, _ = score_fixed(hold, "stat", metric)
    print(f"  sharp-only: train {sharp_t:.5f}  holdout {sharp_h:.5f}")
    print(f"  stat-only : train {stat_t:.5f}  holdout {stat_h:.5f}")
    print(f"  {'ramp':<18}{'train':>9}{'Δ/1000':>8}{'z':>6}{'holdout':>10}{'Δ/1000':>8}{'z':>6}")
    ranked = sorted(results, key=lambda x: x[2])
    shown = [r for r in results if r[0] in ("sector", "flat")] + ranked[:12]
    seen = set()
    for name, ramp, tb, td, tz, tn, hb, hd, hz, hn in shown:
        if name in seen:
            continue
        seen.add(name)
        print(f"  {name:<18}{tb:>9.5f}{td*1000:>8.2f}{tz:>6.2f}{hb:>10.5f}{hd*1000:>8.2f}{hz:>6.2f}")
    best = ranked[0]
    sector = next(r for r in results if r[0] == "sector")
    flat = next(r for r in results if r[0] == "flat")
    print()
    print("Verdict inputs:")
    print(f"  train-best {best[0]}: train Δ {best[3]*1000:+.2f}/1000 (z {best[4]:.2f}), "
          f"holdout Δ {best[7]*1000:+.2f}/1000 (z {best[8]:.2f})")
    print(f"  sector ramp (inherited today): train Δ {sector[3]*1000:+.2f}, holdout Δ {sector[7]*1000:+.2f}")
    print(f"  flat 0.40 (no ramp)          : train Δ {flat[3]*1000:+.2f}, holdout Δ {flat[7]*1000:+.2f}")
    print("  Ship rule: a candidate replaces the inherited sector ramp only if it beats it on")
    print("  BOTH train and holdout; otherwise keep inheriting (blend-neutral).")


if __name__ == "__main__":
    main()
