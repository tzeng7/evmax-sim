"""Full tennis ensemble-weight optimizer with a FAITHFUL point-in-time harness.

The earlier signal (tennis_weight_signal.py) seeded surface Elo from a crude
matchmx-replay, so surface/advanced weights couldn't be trusted. This harness fixes
the big gap: surface Elo is seeded from the **real Tennis Abstract Elo**, point-in-time,
via the nearest Wayback snapshot dated before each test month — matching live behavior.
serve/return, advanced, form, h2h are seeded from matchmx pre-month (same as live).

Then it coordinate-descends the 5 tennis weights to minimize TRAIN Brier (chronological
70/30 split), subject to the full-blend gate constraint: the four REQUIRED_BLEND_MODELS
(surface, serve_return, form, advanced) must stay >= FLOOR so they don't drop out of
model_sources; h2h may go to 0. Holdout is reported but never optimized on.

    uv run python scripts/tennis_weight_optimize.py build      # ~90s, once
    uv run python scripts/tennis_weight_optimize.py            # optimize + report

Requires Wayback snapshots cached by scripts/backtest_tennis_abstract_elo.py
(/tmp/evmax_ta_elo_wayback/atp_*.html).

FINDING (2026-06-27) — close-Brier CANNOT tune these weights, and we did NOT apply
the optimizer's output. With faithful surface Elo, on the holdout (n=703):
  blend 0.21151  vs  pure sharp 0.21011  ->  blend −sharp = +1.39/1000 (blend WORSE)
  bootstrap 95% CI of (blend−sharp): [−0.23, +2.99]/1000; blend beats sharp in 4.5%.
The blend does NOT beat the Pinnacle close out-of-sample, the train/holdout sign flips
(blend wins train, loses holdout) = no real signal, and the differences (~1−3/1000) are
below the bootstrap noise (~±3/1000). The coordinate descent goes degenerate (pumps h2h
to 0.50 on 84 firing matches, floors surface) = fitting noise. The matchmx-replay harness
that suggested a blend edge was the crude surface proxy; with the real TA Elo it vanishes.
Conclusion: weights are not finely tunable on any offline close-anchored signal — the
models' edge (if any) is CLV vs the stale live line, not the close. Keep principled
weights; use CLV as a guardrail (cleanup shadow clv), not a fine-tuning objective.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import analyze_tennis_full_blend as afb  # noqa: E402  (runs the _resolve_surface monkeypatch)
import tennis_pit_rows  # noqa: E402

CACHE = Path("/tmp/evmax_tennis_optimize_cache.json")
TRAIN_FRAC = 0.70
PRIMARY = ("tennis_surface", "tennis_serve_return", "tennis_form", "tennis_advanced")
FLOOR = 0.05            # required-model floor (stay in model_sources)
SHARP_WEIGHT = afb.SHARP_WEIGHT
MODELS = list(afb.WEIGHTS)  # surface, serve, form, advanced, h2h


async def build():
    # Row building factored into tennis_pit_rows.py (2026-07-01), shared with
    # fit_tennis_calibration.py. NOTE: caches rebuilt after the tennis game-count
    # lookup fix carry surface confidence 0.61, not the 0.30 artifact the
    # 2026-06-27 FINDING above was computed on.
    await tennis_pit_rows.build_rows(CACHE)


def _brier(subset, w, sw=SHARP_WEIGHT):
    tot = 0.0
    for o, _d, p_mkt, mp in subset:
        parts = [(w[k], c, p) for k, (p, c) in mp.items() if k in w and w[k] > 0]
        den = sum(x * c for x, c, _ in parts)
        cons = (sum(x * c * p for x, c, p in parts) / den) if den else None
        p = p_mkt if cons is None else sw * p_mkt + (1 - sw) * cons
        tot += (p - o) ** 2
    return tot / len(subset)


def optimize():
    rows = sorted(json.loads(CACHE.read_text()), key=lambda r: r[1])
    cut = int(len(rows) * TRAIN_FRAC)
    train, holdout = rows[:cut], rows[cut:]
    grid = [round(0.05 * i, 2) for i in range(0, 11)]  # 0.00 .. 0.50

    cur = {"tennis_surface": 0.30, "tennis_serve_return": 0.10,
           "tennis_form": 0.35, "tennis_advanced": 0.15, "tennis_h2h": 0.05}
    base_tb, base_hb = _brier(train, cur), _brier(holdout, cur)
    print(f"current weights {cur}")
    print(f"  train={base_tb:.6f} holdout={base_hb:.6f}\n")

    w = dict(cur)
    for it in range(6):
        improved = False
        for k in MODELS:
            lo = FLOOR if k in PRIMARY else 0.0
            best_v, best_tb = w[k], _brier(train, w)
            for v in grid:
                if v < lo:
                    continue
                cand = dict(w); cand[k] = v
                tb = _brier(train, cand)
                if tb < best_tb - 1e-9:
                    best_tb, best_v = tb, v
            if best_v != w[k]:
                w[k] = best_v; improved = True
        if not improved:
            break

    tb, hb = _brier(train, w), _brier(holdout, w)
    print(f"optimized weights {w}")
    print(f"  train={tb:.6f} (Δ {(tb-base_tb)*1000:+.2f}/1000)")
    print(f"  holdout={hb:.6f} (Δ {(hb-base_hb)*1000:+.2f}/1000)")
    print(f"  pure sharp: train={_brier(train, {}):.6f} holdout={_brier(holdout, {}):.6f}")
    # normalize for readability (sum of primary+h2h to ~1.0, sharp separate)
    tot = sum(w.values())
    print("  normalized:", {k: round(v / tot, 3) for k, v in w.items()})


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        asyncio.run(build())
    else:
        optimize()
