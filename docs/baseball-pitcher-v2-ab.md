# pitcher_v2 A/B — walk-forward verdict (2026-07-19)

Harness: `scripts/backtest_baseball_ab.py` — one continuous 2025-03 → 2026-07 ESPN
walk-forward per variant (running totals cross the season boundary, matching a warm live
seed), scored separately on the 2025 train year and the 2026 holdout. Holdout touched
once; nothing swept on it. Frozen PRIOR-season Savant xERA only (leak-free by
construction — the live model's fresh current-season xERA is strictly better-informed,
so these numbers UNDER-state v2 if anything). Anchor caveat: the walk-forward month
fetch caps ~200 games/month (~half the MLB slate) — a pre-existing harness property,
identical across arms, so the paired comparison stands.

## Results (Brier, lower is better)

| Variant | Train pitcher (n=545) | Train blend (n=1273) | Holdout pitcher (n=610) | Holdout blend (n=905) |
|---|---|---|---|---|
| v1 (starter-only) | 0.2457 | 0.2433 | 0.2612 | 0.2548 |
| **v2 (pen+park+off+xera)** | **0.2380** | 0.2434 | **0.2557** | **0.2546** |
| park,off,xera | 0.2403 | 0.2431 | 0.2607 | 0.2553 |
| pen only | 0.2415 | 0.2435 | 0.2539 | 0.2537 |

Deltas vs v1 (negative = better): v2 holdout Δpitcher **−5.46/1000**, Δblend −0.19/1000,
coverage 100%.

## Ship gate (2026 holdout) — ALL PASS → SHIP v2

- (a) blend ≤ v1 blend: **PASS** (0.2546 ≤ 0.2548)
- (b) standalone ≥ 2.0/1000 better: **PASS** (−5.46/1000)
- (c) coverage ≥ 95% of v1: **PASS** (100%)

## Component notes

- **pen** REVERSED its prior null result (the 2025-era experiment measured Δ+0.0012
  WORSE; the note survives at the walk-forward call site). On the 2025 train year here it
  is still blend-flat (+0.22/1000, consistent with the old null) — but on the 2026
  holdout it is the strongest single component (Δpitcher −7.28, Δblend −1.07/1000).
  Availability/fatigue is a fresh-season signal; the composite ships with it.
- **park,off,xera** without pen: clear train gain (−5.43) but holdout-marginal
  (−0.55 standalone, +0.49 blend) — keep inside the composite, not standalone-shippable.
- Blend deltas are small everywhere because elo+form dominate the blend rows where the
  pitcher abstains (n=1273 vs pitcher n=610); the standalone column is the model-quality
  signal, and it is consistent in sign across train AND holdout.

## What happens next

pitcher_v2 is live in code (baseball remains `mode: shadow`). The clean shadow sample
resets via the contamination signature (rows without `pitcher_v2` in model_sources are
dated out). Promotion is re-judged on fresh v2 rows only:
`evmax cleanup shadow clv baseball -m moneyline --max-staleness-h 3` must clear
n≥30 / mean≥0 / %pos≥55 — the v1 baseline was 29% pos on fresh closes; that is the bar
v2's improved pricing has to move. Expected ~3-4 weeks at the light-scan cadence.
