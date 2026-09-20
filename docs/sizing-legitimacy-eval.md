# Stake-sizing legitimacy audit and rebuild (2026-09-19)

Question asked: *should we cap our stake, and are the current sizing methods legitimate?*

Verdict up front: **yes, keep the cap.** On the system's own resolved data the 5% cap is
what separates positive growth from ruin, and it dominates a 10% cap on both median and
5th-percentile growth. But two of the four sizing layers were justified only by stale
docstrings. This change builds the evidence engine, routes every caller through one entry
point, and grounds each layer's verdict in a walk-forward growth test.

## The four layers and where they stood

| Layer | Was | Now |
|-------|-----|-----|
| Fractional Kelly (half) | half Kelly (CLI default), docstring said quarter | unchanged; sweep confirms half is at the aggressive end (boot5-optimal is 0.30–0.40) |
| Confidence discount | dead code fixed at 1.0, docstring described a live discount | removed from the docstring; principled successor (edge shrinkage) built as an opt-in layer |
| Liquidity discount | spread-over-mid proxy, ignores depth | depth-keyed `min(1, α·depth/stake)` added behind a flag; proxy is the fallback |
| Hard 5% cap | present, rationale "growth" | kept; harness shows it is the tail/error backstop, not a growth knob |
| Exposure guard (8%/game, same-side 0.5) | ρ≈0.8 assumed | measurement script added; joint-Kelly is the principled replacement |

## Evidence (sizing replay harness, `scripts/backtest_sizing.py`)

Replays resolved rows with day-batched simultaneous settlement, the per-game exposure
cap inside the loop, contamination rows excluded by rule, fees at the effective price,
and a week-block bootstrap. Live rows, 180 days (n≈815).

**Cap sweep (clean sample, excl. lol/cs2 alignment-bug rows):** cap 5% beats cap 10% at
every base fraction on both median and 5th-percentile log growth. base 0.30–0.40 is the
boot5 sweet spot; base 0.50 (the CLI default) is defensible but past the left-tail
optimum. Uncapped half-Kelly risks ruin on the raw sample (a handful of mis-aligned rows
take 68% of the bankroll).

**Edge shrinkage — WALK-FORWARD REJECTED.** The two-input logistic
`P(win)=σ(a+s_b·logit(blended)+s_p·logit(price))` and the single-λ edge-shrink variant
both lose out of sample: walk-forward the logistic turns baseline +0.74 into −0.38. The
pooled fit puts almost all weight on price (`s_b≈0.06`), which over-shrinks the sectors
that carry real edge (nba realized/predicted edge ≈ 2.2). Only tennis and wnba have
`n≥300` for a per-sector fit. The λ-form improves the in-sample left tail
(boot5 −0.29 vs −0.65) but still loses walk-forward. **Kept OFF.** Re-run
`scripts/fit_edge_shrinkage.py` + the harness as per-sector samples grow.

**Exposure correlation (`scripts/measure_outcome_correlation.py`).** Deduped same-game
same-side outcome correlation is ~0.05–0.06, far below the ρ≈0.8 the 0.5 same-side
discount assumes — but the measure is confounded by spread-magnitude mix and understates
Kelly-return correlation. Not enough to hand-change 0.5; the joint-Kelly path
(`joint_kelly_enabled`) is the principled fix and should be validated against this.

## What shipped

- `evmax/ev/sizing.py` — the single sizing entry point `size_position`, composing edge
  shrinkage, fractional Kelly, depth-keyed liquidity, and the cap. **Identity when all
  flags are off** (default), so the live path is byte-identical until a layer is enabled.
- `evmax/backtest/sizing.py` + `scripts/backtest_sizing.py` — the replay harness. Every
  sizing verdict above is reproducible from it.
- `scripts/fit_edge_shrinkage.py` — fits the shrinkage coefficients (no state file is
  shipped; flag-on with no file safely no-ops to identity).
- Pre-persist **quarantine gate** (`sizing_quarantine_pp`, default off): demotes to
  shadow any live moneyline-family row whose blended prob is >pp from BOTH the sharp
  anchor and the price — the explicit form of the error containment the cap does bluntly
  (it would have caught the April esports alignment rows before sizing).
- Depth-keyed liquidity discount wired through the coordinator (candidate-only depth
  fetch, `liquidity_depth_enabled`, default off); validate against anchored-entry fills
  before enabling.
- All seven historical `compute_kelly` call sites (five in the EV agent, the pruner, the
  CLI pick path) now route through `size_position`, fixing the scan-vs-prune `spread_pct`
  divergence.

## Settings (all default OFF — live behavior unchanged)

`edge_shrinkage_enabled`, `liquidity_depth_enabled` (+ `liquidity_depth_alpha`,
`liquidity_depth_floor`), `sizing_quarantine_pp`.
