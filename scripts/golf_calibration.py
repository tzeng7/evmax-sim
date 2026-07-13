#!/usr/bin/env python
"""Recalibrate the golf model, validate out-of-sample, then reality-check edge.

Two questions, in order:

  1. CAN the model be made honest? Fit an isotonic calibration map on 2025
     walk-forward predictions and test it on 2026 (a temporal holdout the map
     never sees). If holdout Brier drops, the under-confidence is real and
     fixable and it generalises.

  2. Does any EDGE survive calibration? Apply the calibrated model to the live
     Open field and compare to the market. If calibration pushes the model's
     favourites UP toward the market (which already had them high), the gap
     CLOSES — that is the null result: the model just reproduces the market.
     Edge would mean the calibrated model still disagrees with the market in
     specific spots. (We cannot score historical model-vs-market — we never
     captured historical golf odds — so this step is a live, forward-looking
     read, not a backtest.)

    python scripts/golf_calibration.py

Free ESPN + the captured PolyUS Open fixture. Score-proxy skill path throughout,
so the calibration (fit on score-proxy) applies to the same model family.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evmax.golf.backtest import WalkForwardConfig, collect_predictions  # noqa: E402
from evmax.golf.calibration import IsotonicCalibrator, brier  # noqa: E402
from evmax.golf.data.espn import parse_scoreboard, result_to_sg_proxy  # noqa: E402
from evmax.golf.devig import devig_field  # noqa: E402
from evmax.golf.models import GolfMarketType  # noqa: E402
from evmax.golf.names import canonical  # noqa: E402
from evmax.golf.simulator import SimConfig, simulate_field  # noqa: E402
from evmax.golf.skill import SkillConfig, skill_from_records, to_sim_inputs  # noqa: E402
from evmax.golf.venues.polymarket_golf import parse_golf_events  # noqa: E402

_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/golf/{tour}/scoreboard?dates={year}"
_UA = "Mozilla/5.0 (evmax-golf calibration)"
_ROOT = Path(__file__).resolve().parents[1]
_CACHE = _ROOT / "data" / "golf_backtest_cache"
_FIXTURES = _ROOT / "tests" / "golf" / "fixtures"
_MARKETS = (GolfMarketType.win, GolfMarketType.top_10, GolfMarketType.make_cut)


def fetch_season(year: int, tour: str = "pga", refresh: bool = False) -> dict:
    _CACHE.mkdir(parents=True, exist_ok=True)
    path = _CACHE / f"{tour}_{year}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    req = urllib.request.Request(_SCOREBOARD.format(tour=tour, year=year),
                                 headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=40) as resp:  # noqa: S310
        payload = json.load(resp)
    path.write_text(json.dumps(payload))
    return payload


def _pairs(preds, key):
    return [(p.model[key], p.actual[key]) for p in preds
            if p.had_prior and p.model.get(key) is not None and p.actual.get(key) is not None]


def _cal_table(probs, outcomes, bins=((0, .02), (.02, .05), (.05, .1), (.1, .2), (.2, .4), (.4, 1.01))):
    rows = []
    for lo, hi in bins:
        sel = [(m, a) for m, a in zip(probs, outcomes) if lo <= m < hi]
        if sel:
            mp = sum(m for m, _ in sel) / len(sel)
            ma = sum(a for _, a in sel) / len(sel)
            rows.append((lo, hi, len(sel), mp, ma))
    return rows


def part1_calibration(train, holdout):
    print("=" * 68)
    print("PART 1 — can the model be made honest?  (fit 2025 → test 2026)")
    print("=" * 68)
    calibrators = {}
    for mk in _MARKETS:
        key = mk.value
        tr = _pairs(train, key)
        ho = _pairs(holdout, key)
        if len(tr) < 50 or len(ho) < 50:
            print(f"\n{key}: insufficient data (train={len(tr)}, holdout={len(ho)}) — skip")
            continue
        cal = IsotonicCalibrator.fit([m for m, _ in tr], [a for _, a in tr])
        calibrators[key] = cal
        ho_p = [m for m, _ in ho]
        ho_y = [a for _, a in ho]
        raw_b = brier(ho_p, ho_y)
        cal_b = brier(cal.transform(ho_p), ho_y)
        impr = (raw_b - cal_b) / raw_b if raw_b else 0.0
        verdict = "IMPROVES" if cal_b < raw_b else "no help"
        print(f"\n{key}  (train n={len(tr)}, holdout n={len(ho)})")
        print(f"  holdout Brier:  raw {raw_b:.4f}  →  calibrated {cal_b:.4f}   "
              f"({impr:+.1%})  [{verdict}]")
        print("  holdout calibration (pred → actual), raw vs calibrated:")
        for lo, hi, n, mp, ma in _cal_table(ho_p, ho_y):
            cmp = cal.transform_one(mp)
            print(f"    [{lo:.0%},{hi:.0%})  n={n:4d}  raw {mp:5.1%}→act {ma:5.1%}   "
                  f"calibrated {cmp:5.1%}")
    return calibrators


def part2_edge_reality_check(all_results, calibrators):
    print("\n" + "=" * 68)
    print("PART 2 — does any edge survive?  (calibrated model vs LIVE market)")
    print("=" * 68)
    win_cal = calibrators.get(GolfMarketType.win.value)
    if win_cal is None:
        print("no win calibrator (win outcomes too sparse) — cannot run edge check")
        return

    # Score-proxy skill from ALL completed results to date.
    records = []
    for r in sorted(all_results, key=lambda e: e.date):
        records.extend(result_to_sg_proxy(r, min_field=30))
    skills = skill_from_records(records, cfg=SkillConfig())

    # Live Open field + market (captured fixture).
    payload = json.loads((_FIXTURES / "polyus_open_winner.json").read_text())
    win_field = next(f for f in parse_golf_events(payload)
                     if f.market_type is GolfMarketType.win)
    devig_field(win_field)
    roster = sorted({m.golfer for m in win_field.markets})

    names, means, sds = to_sim_inputs(roster, skills, SkillConfig())
    sims = {r.player: r for r in simulate_field(
        names, means, sds, SimConfig(n_sims=30000, seed=31, cut_size=min(65, len(roster) // 2)))}

    rows = []
    for g in roster:
        if g not in sims:
            continue
        raw = sims[g].win
        cal = win_cal.transform_one(raw)
        mkt = win_field.devigged.get(g, 0.0)
        rows.append((g, raw, cal, mkt))
    rows.sort(key=lambda r: -r[3])  # by market prob

    print("\n(top of the market — winner market)")
    print(f"  {'golfer':22s} {'raw model':>9s} {'calibrated':>10s} {'market':>8s} "
          f"{'cal−mkt':>8s}")
    gap_raw = gap_cal = 0.0
    n_cov = 0
    for g, raw, cal, mkt in rows[:12]:
        print(f"  {g:22s} {raw:9.1%} {cal:10.1%} {mkt:8.1%} {cal-mkt:+8.1%}")
    for g, raw, cal, mkt in rows:
        if mkt > 0.01:  # focus on real contenders
            gap_raw += abs(raw - mkt)
            gap_cal += abs(cal - mkt)
            n_cov += 1
    if n_cov:
        print(f"\n  mean |model − market| over {n_cov} contenders:")
        print(f"    raw model:        {gap_raw/n_cov:.2%}")
        print(f"    calibrated model: {gap_cal/n_cov:.2%}")
        if gap_cal < gap_raw:
            print("  → calibration CLOSES the gap to the market. The model is not"
                  " disagreeing-\n    and-right; it was just under-confident. This is"
                  " the NULL result:\n    no edge on the liquid winner market.")
        else:
            print("  → calibration does NOT close the gap — the calibrated model still"
                  "\n    systematically disagrees with the market. Worth investigating"
                  " which\n    side is right (needs outcomes / CLV to settle).")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    r2025 = [r for r in parse_scoreboard(fetch_season(2025, refresh=args.refresh)) if r.completed]
    r2026 = [r for r in parse_scoreboard(fetch_season(2026, refresh=args.refresh)) if r.completed]
    print(f"2025 completed: {len(r2025)}  |  2026 completed: {len(r2026)}")

    cfg = WalkForwardConfig(min_prior_events=8, sim=SimConfig(n_sims=12000, seed=99))
    # Predictions for the WHOLE timeline (leak-free); split by season for the
    # calibration train/holdout so the map is judged on unseen 2026 outcomes.
    preds, _, _ = collect_predictions(r2025 + r2026, cfg)
    train = [p for p in preds if p.date < "2026-01-01"]
    holdout = [p for p in preds if p.date >= "2026-01-01"]
    print(f"walk-forward predictions: train(2025)={len(train)}  holdout(2026)={len(holdout)}\n")

    calibrators = part1_calibration(train, holdout)
    part2_edge_reality_check(r2025 + r2026, calibrators)


if __name__ == "__main__":
    main()
