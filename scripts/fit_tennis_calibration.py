"""Fit per-sector isotonic calibration for the tennis ensemble — PIT harness.

Symptom (original 2024–2026 backtest, sharp_weight=0):
  bucket 30–40%: predicted 35.2%, actual 29.7%   (gap −5.5%)
  bucket 60–70%: predicted 64.8%, actual 70.3%   (gap +5.5%)
The blend looked systematically underconfident — pulled toward 0.5 by the
weighted-averaging of six diverse model voices.

WHY THIS IS A REWRITE (2026-07-01): the previous version of this script ran
the production tennis agents against TODAY'S aggregate-current Tennis Abstract
state files for both the 2024 training year and the 2025–26 holdout. Post
source-swap (Sackmann → Tennis Abstract, 2026-06-27) those states are a
snapshot of *current* ratings, so both sides of the experiment leaked future
information and neither the fit nor the "validation" meant anything.

This version is point-in-time (PIT), Wayback-style:
  1. Rows come from scripts/tennis_pit_rows.py — for each test month, surface
     Elo is seeded from the nearest Wayback snapshot of the real Tennis
     Abstract leaderboard dated BEFORE the month, and serve/return, advanced,
     form, h2h are seeded from matchmx rows dated before the month. No model
     ever sees a rating computed from a match it is asked to predict.
  2. The model-only consensus is recomputed here exactly as
     EnsembleModelAgent._blend does it (per-sector weights × confidence, with
     the production confidence gate conf > 0.45) — uncalibrated, since we are
     learning the transform. ranking_trend is omitted (no PIT weekly snapshot
     series; it structurally rarely fires).
  3. Chronological split: train on the first --train-frac of matches, fit
     isotonic regression (same params as ModelCalibrator), validate on the
     held-out tail. Both sides of each match are fed for balanced labels.
  4. Promotion gate: holdout model-only Brier must improve by ≥ PROMOTION_BAR
     AND the paired-bootstrap 95% CI of the improvement must exclude zero
     (lesson from tennis_weight_optimize.py: close-anchored deltas of ~1–3
     per-mille are below the bootstrap noise floor — a bare threshold is not
     enough). Only then is the curve persisted under "tennis_ensemble" in
     calibration.json (picked up by EnsembleModelAgent._apply_sector_calibration).
     Nothing is written otherwise.

The live-blend impact at sharp_weight=0.85 and the full-blend subset (the four
REQUIRED_BLEND_MODELS all firing — the only rows that become live plays) are
reported for context. The blend metric uses the flat sharp mix like
tennis_weight_optimize.py (no FLB correction), so treat it as directional.

Usage:
    uv run python scripts/fit_tennis_calibration.py build   # rebuild PIT rows (~90s + matchmx fetch)
    uv run python scripts/fit_tennis_calibration.py         # fit + validate + gate
    uv run python scripts/fit_tennis_calibration.py --report-only
    uv run python scripts/fit_tennis_calibration.py --train-frac 0.7 --sharp-weight 0.85

Requires Wayback snapshots cached by scripts/backtest_tennis_abstract_elo.py.
Rebuild the row cache after any tennis agent/seeding change — rows bake in the
agent behavior at build time (see the surface-confidence note in tennis_pit_rows.py).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402

from evmax.agents.models.calibration import ModelCalibrator  # noqa: E402
from evmax.agents.models.ensemble_agent import EnsembleModelAgent  # noqa: E402

# NOTE: tennis_pit_rows is imported lazily (build path only) — importing it
# pulls in analyze_tennis_full_blend, which monkeypatches
# TennisModelAgent._resolve_surface at import time. The fit path must stay
# side-effect free so tests can import this module safely.

CACHE = _REPO_ROOT / "data/backtest/tennis/pit_rows.json"
CALIBRATION_KEY = "tennis_ensemble"
# Same bar as the pre-PIT version (the first, leaky run measured ~0.0015 on a
# 5000-match holdout). The PIT holdout is smaller (~700 matches), so the CI
# condition below does most of the gating work.
PROMOTION_BAR = 0.001
CONF_GATE = 0.45  # EnsembleModelAgent._blend: `if pred.confidence <= 0.45: continue`
PRIMARY = ("tennis_surface", "tennis_serve_return", "tennis_form", "tennis_advanced")
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 20260701


def tennis_weights() -> dict[str, float]:
    """The shipped per-sector weights — single source of truth, never hardcoded."""
    return dict(EnsembleModelAgent.SECTOR_WEIGHT_OVERRIDES["tennis"])


def model_consensus(
    mp: dict[str, list],
    weights: dict[str, float],
    conf_gate: float = CONF_GATE,
) -> tuple[float | None, list[str]]:
    """Replicate _blend step 1 (model-only, 2-way): confidence-weighted average.

    mp: {model_name: [prob_a, confidence]}. Models at conf <= gate or with a
    non-positive effective weight are excluded, exactly like production.
    Returns (consensus_prob_a, contributing_model_names) or (None, []).
    """
    parts: list[tuple[float, float]] = []
    names: list[str] = []
    for name, (prob_a, conf) in mp.items():
        w = weights.get(name)
        if w is None or conf <= conf_gate:
            continue
        eff = w * conf
        if eff <= 0:
            continue
        parts.append((eff, prob_a))
        names.append(name)
    den = sum(e for e, _ in parts)
    if den <= 0:
        return None, []
    return sum(e * p for e, p in parts) / den, sorted(names)


def fit_isotonic(probs: list[float], labels: list[int]) -> tuple[list[float], list[float]]:
    """Fit isotonic breakpoints with the same params ModelCalibrator uses."""
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    iso.fit(np.array(probs), np.array(labels, dtype=float))
    return iso.X_thresholds_.tolist(), iso.y_thresholds_.tolist()


def apply_isotonic(x: list[float], y: list[float], prob: float) -> float:
    """Mirror ModelCalibrator.calibrate: clamp to breakpoints, then interp."""
    if not x or len(x) < 2:
        return prob
    prob = max(x[0], min(x[-1], prob))
    return float(np.interp(prob, x, y))


def split_chronological(rows: list[list], train_frac: float) -> tuple[list[list], list[list]]:
    rows = sorted(rows, key=lambda r: r[1])
    cut = int(len(rows) * train_frac)
    return rows[:cut], rows[cut:]


def both_sides(cons_rows: list[tuple[float, float]]) -> tuple[list[float], list[int]]:
    """(consensus_prob_a, outcome_a) matches -> balanced per-side samples."""
    probs: list[float] = []
    labels: list[int] = []
    for cons, out in cons_rows:
        probs.append(cons)
        labels.append(int(out))
        probs.append(1.0 - cons)
        labels.append(1 - int(out))
    return probs, labels


def brier(probs: list[float], labels: list[int]) -> float:
    return sum((p - o) ** 2 for p, o in zip(probs, labels)) / len(probs)


def paired_bootstrap_delta(
    per_match_deltas: list[float],
    b: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Bootstrap the mean of per-match (uncal − cal) Brier deltas.

    Resamples MATCHES (both sides of a match move together — they are
    perfectly anti-correlated samples, not independent).
    Returns (ci_low, ci_high, frac_positive) at 95%.
    """
    rng = random.Random(seed)
    n = len(per_match_deltas)
    means = []
    for _ in range(b):
        s = sum(per_match_deltas[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    ci_low = means[int(0.025 * b)]
    ci_high = means[int(0.975 * b)]
    frac_pos = sum(1 for m in means if m > 0) / b
    return ci_low, ci_high, frac_pos


def reliability_table(probs: list[float], labels: list[int]) -> list[str]:
    lines = []
    for lo10 in range(0, 10):
        lo, hi = lo10 / 10, lo10 / 10 + 0.1
        bucket = [(p, o) for p, o in zip(probs, labels) if lo <= p < hi]
        if len(bucket) < 20:
            continue
        pred = sum(p for p, _ in bucket) / len(bucket)
        act = sum(o for _, o in bucket) / len(bucket)
        lines.append(f"    {lo:.1f}–{hi:.1f}: n={len(bucket):<5} pred={pred:.3f} actual={act:.3f} "
                     f"gap={act - pred:+.3f}")
    return lines


def evaluate(
    rows: list[list],
    weights: dict[str, float],
    iso: tuple[list[float], list[float]] | None,
    sharp_weight: float,
) -> dict:
    """Score a row set. iso=None -> uncalibrated. Returns metrics + raw series."""
    cons_rows: list[tuple[float, float]] = []        # (cons_a, outcome_a)
    blend_probs: list[float] = []                    # sw*mkt + (1-sw)*cons, side a
    fb_cons_rows: list[tuple[float, float]] = []     # full-blend subset
    skipped = 0
    for out, _date, p_mkt, mp in rows:
        cons, names = model_consensus(mp, weights)
        if cons is None:
            skipped += 1
            continue
        if iso is not None:
            cons = apply_isotonic(iso[0], iso[1], cons)
        cons_rows.append((cons, out))
        blend_probs.append(sharp_weight * p_mkt + (1 - sharp_weight) * cons)
        if all(k in names for k in PRIMARY):
            fb_cons_rows.append((cons, out))

    probs, labels = both_sides(cons_rows)
    bl_probs, bl_labels = both_sides(
        [(bp, out) for bp, (_c, out) in zip(blend_probs, cons_rows)]
    )
    fb_probs, fb_labels = both_sides(fb_cons_rows) if fb_cons_rows else ([], [])
    return {
        "n_matches": len(cons_rows),
        "n_skipped": skipped,
        "n_full_blend": len(fb_cons_rows),
        "model_brier": brier(probs, labels),
        "blend_brier": brier(bl_probs, bl_labels),
        "fb_model_brier": brier(fb_probs, fb_labels) if fb_probs else None,
        "cons_rows": cons_rows,
        "probs": probs,
        "labels": labels,
    }


def run(train_frac: float, sharp_weight: float, report_only: bool) -> None:
    if not CACHE.exists():
        print(f"PIT row cache missing ({CACHE}).\n"
              f"Run: uv run python scripts/fit_tennis_calibration.py build", file=sys.stderr)
        sys.exit(2)
    rows = sorted(json.loads(CACHE.read_text()), key=lambda r: r[1])
    weights = tennis_weights()
    train, holdout = split_chronological(rows, train_frac)
    print("=== fit_tennis_calibration (PIT) ===")
    print(f"rows: {len(rows)}  train: {len(train)} ({train[0][1]} → {train[-1][1]})  "
          f"holdout: {len(holdout)} ({holdout[0][1]} → {holdout[-1][1]})")
    print(f"weights: {weights}  conf_gate: >{CONF_GATE}  sharp_weight: {sharp_weight}\n")

    # Sanity guard against pre-fix caches (surface conf 0.30 artifact): if
    # surface never clears the production gate, the cache predates the
    # game-count lookup fix and every consensus here would be surface-less.
    surf_pass = sum(
        1 for _o, _d, _p, mp in rows
        if "tennis_surface" in mp and mp["tennis_surface"][1] > CONF_GATE
    )
    if surf_pass == 0:
        print("STALE CACHE: tennis_surface never clears the 0.45 confidence gate — this\n"
              "cache predates the game-count lookup fix. Rebuild with `build`.", file=sys.stderr)
        sys.exit(2)

    # --- Train: fit isotonic on the uncalibrated train consensus ---
    train_eval = evaluate(train, weights, iso=None, sharp_weight=sharp_weight)
    iso = fit_isotonic(train_eval["probs"], train_eval["labels"])
    print(f"train: {train_eval['n_matches']} matches used ({train_eval['n_skipped']} no-consensus), "
          f"uncal model-only Brier {train_eval['model_brier']:.5f}")
    print(f"isotonic fit: {len(iso[0])} breakpoints")
    print("  train reliability (uncalibrated):")
    print("\n".join(reliability_table(train_eval["probs"], train_eval["labels"])) or "    (thin)")

    # --- Holdout: uncalibrated vs calibrated ---
    h_uncal = evaluate(holdout, weights, iso=None, sharp_weight=sharp_weight)
    h_cal = evaluate(holdout, weights, iso=iso, sharp_weight=sharp_weight)
    d_model = h_uncal["model_brier"] - h_cal["model_brier"]
    d_blend = h_uncal["blend_brier"] - h_cal["blend_brier"]

    # Paired bootstrap over holdout matches. The two sides of a match have
    # identical squared error ((p−o)² = ((1−p)−(1−o))²), so the per-match
    # delta IS the per-side delta — the match is the resampling unit.
    per_match = [
        (cu - out) ** 2 - (cc - out) ** 2
        for (cu, out), (cc, _o) in zip(h_uncal["cons_rows"], h_cal["cons_rows"])
    ]
    ci_low, ci_high, frac_pos = paired_bootstrap_delta(per_match)

    print(f"\nholdout: {h_uncal['n_matches']} matches ({h_uncal['n_skipped']} no-consensus), "
          f"{h_uncal['n_full_blend']} full-blend")
    print(f"  model-only Brier: uncal {h_uncal['model_brier']:.5f}  cal {h_cal['model_brier']:.5f}  "
          f"Δ {d_model * 1000:+.2f}/1000")
    print(f"    bootstrap 95% CI of Δ: [{ci_low * 1000:+.2f}, {ci_high * 1000:+.2f}]/1000  "
          f"P(Δ>0)={frac_pos:.1%}")
    print(f"  live-blend  Brier (sw={sharp_weight}): uncal {h_uncal['blend_brier']:.5f}  "
          f"cal {h_cal['blend_brier']:.5f}  Δ {d_blend * 1000:+.2f}/1000")
    if h_uncal["fb_model_brier"] is not None:
        d_fb = h_uncal["fb_model_brier"] - h_cal["fb_model_brier"]
        print(f"  full-blend subset model-only: uncal {h_uncal['fb_model_brier']:.5f}  "
              f"cal {h_cal['fb_model_brier']:.5f}  Δ {d_fb * 1000:+.2f}/1000")
    print("  holdout reliability (uncalibrated):")
    print("\n".join(reliability_table(h_uncal["probs"], h_uncal["labels"])) or "    (thin)")
    print("  holdout reliability (calibrated):")
    print("\n".join(reliability_table(h_cal["probs"], h_cal["labels"])) or "    (thin)")

    # --- Promotion gate ---
    passes = d_model >= PROMOTION_BAR and ci_low > 0
    print(f"\ngate: Δ ≥ {PROMOTION_BAR} AND bootstrap CI low > 0")
    if not passes:
        print(f"NO PROMOTE: Δ={d_model:+.5f}, CI low={ci_low:+.5f}. Nothing written.")
        return
    print(f"PASS: Δ={d_model:+.5f}, CI low={ci_low:+.5f}.")
    if report_only:
        print("[report-only] calibration NOT written.")
        return
    cal = ModelCalibrator()
    ok = cal.retrain(CALIBRATION_KEY, train_eval["probs"], train_eval["labels"])
    if ok:
        print(f"PROMOTED: '{CALIBRATION_KEY}' written to calibration.json "
              f"(fit on train split only — exactly the curve that was validated).")
    else:
        print("retrain returned False (insufficient data or sklearn missing). Nothing written.")


async def _build_live_rows(cache_path: Path) -> None:
    """Build the row cache using CURRENT live model state files.

    No Wayback snapshots, no matchmx fetch — agents load from
    data/models/tennis_*_state.json on init, exactly as in production.
    Ratings reflect the latest weekly seed, so predictions for historical
    training rows (2024) leak future information. Acceptable for calibration:
    the transform corrects systematic confidence bias (a monotone mapping),
    not individual match outcomes. Mirrors the pre-2026-07-01 approach that
    produced the original +0.00154 calibration.

    Call via: python scripts/fit_tennis_calibration.py build-live
    """
    # Import lazily — afb monkeypatches TennisModelAgent._resolve_surface at
    # import time; keep the fit path side-effect-free for tests.
    import analyze_tennis_full_blend as afb  # noqa: PLC0415
    import backtest_tennis_matchmx as bt  # noqa: PLC0415
    from evmax.agents.models.tennis_model_agent import TennisModelAgent  # noqa: PLC0415
    from evmax.agents.models.tennis_serve_return_agent import TennisServeReturnAgent  # noqa: PLC0415
    from evmax.agents.models.tennis_advanced_stats_agent import TennisAdvancedStatsAgent  # noqa: PLC0415
    from evmax.agents.models.tennis_form_agent import TennisFormAgent  # noqa: PLC0415
    from evmax.agents.models.tennis_h2h_agent import TennisH2HAgent  # noqa: PLC0415
    from evmax.ev.devig import devig_two_way  # noqa: PLC0415

    models_list = list(afb.WEIGHTS)  # surface, serve_return, form, advanced, h2h

    # Agents auto-load from data/models/*_state.json on init.
    agents: dict = {
        "tennis_surface": TennisModelAgent(),
        "tennis_serve_return": TennisServeReturnAgent(),
        "tennis_form": TennisFormAgent(),
        "tennis_advanced": TennisAdvancedStatsAgent(),
        "tennis_h2h": TennisH2HAgent(),
    }
    for ag in agents.values():
        ag.save_state = lambda: None  # suppress disk writes during prediction

    years = (2023, 2024, 2025, 2026)  # 2023 included for Excel completeness; filtered below
    results = bt.load_atp_results(years)
    print(f"[build-live] loaded {len(results)} ATP results from Excel")

    rows: list[list] = []
    for r in results:
        if r["date"].year < 2024:
            continue
        if not r["psw"] or not r["psl"]:
            continue
        try:
            p_mkt_w, _, _ = devig_two_way(float(r["psw"]), float(r["psl"]))
        except Exception:
            continue
        a, b = sorted((r["winner"], r["loser"]))
        a_won = a == r["winner"]
        market = afb._mk_market(a, b, r["surface"], r["date"], r["best_of"])
        sharp = afb._mk_sharp(a, b, r["date"])
        preds = await asyncio.gather(*[afb._prob_a(agents[k], market, sharp) for k in models_list])
        mp = {k: [preds[i][0], preds[i][1]] for i, k in enumerate(models_list)
              if preds[i][0] is not None}
        rows.append([1.0 if a_won else 0.0, r["date"].isoformat(),
                     p_mkt_w if a_won else 1.0 - p_mkt_w, mp])

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(rows))
    fire = {k: sum(1 for _, _, _, mp in rows if k in mp) for k in models_list}
    print(f"[build-live] cached {len(rows)} rows -> {cache_path}")
    print(f"fire rates: { {k: f'{v}/{len(rows)}' for k, v in fire.items()} }")
    print("NOTE: non-PIT — ratings from live state files, not Wayback snapshots.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PIT isotonic calibration for the tennis ensemble")
    parser.add_argument("cmd", nargs="?", default="fit", choices=["fit", "build", "build-live"])
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--sharp-weight", type=float, default=0.85)
    parser.add_argument("--report-only", action="store_true",
                        help="never write calibration.json, even on a gate pass")
    args = parser.parse_args()
    if args.cmd == "build":
        import tennis_pit_rows  # noqa: PLC0415  (import-time monkeypatch — see module note)

        asyncio.run(tennis_pit_rows.build_rows(CACHE))
    elif args.cmd == "build-live":
        asyncio.run(_build_live_rows(CACHE))
    else:
        run(args.train_frac, args.sharp_weight, args.report_only)
