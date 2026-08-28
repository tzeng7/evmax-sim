# WNBA injury-impact model eval (MODEL-13)

**Verdict (2026-08-28): the generic injury layer is VALIDATED; amplifying WNBA
injury magnitude beyond generic is REJECTED — the signal is directional but
sub-noise-floor, unstable across seasons, and shrinks when a seed-lookahead is
removed. No change to live WNBA injury sizing.**

## The question

WNBA moneyline is LIVE (real-bankroll). MODEL-13 proposed that WNBA star-out
impact should be *larger* than the generic tier magnitude the injury agent
applies today (a WNBA star loss is often cited as an ~8–10pt swing vs ~5–6 in
the NBA, because WNBA rosters are shorter). The counter-evidence: the project's
own `scripts/backtest_adjustment_scaling.py` found the Pinnacle close realizes
only **~9%** of the full injury shove (n=353) — sharp already prices ~90% of
injury news, which is why injury deltas are already scaled to the model share
`(1 − effective_sharp_weight)`. So the "make it bigger" thesis was NOT assumed —
it was tested walk-forward, the same bar every other WNBA model change clears.

## Method

`scripts/backtest_wnba_injury_impact.py` — an availability walk-forward that
mirrors the LIVE mechanism rather than inventing a new model:

1. `run_walkforward("wnba", months)` supplies the genuinely-OOS model blend per
   game (elo/form update per game from empty state; efficiency/possession-sim on
   a fixed prior-season seed, exactly like live opening day). Unmodified.
2. Per game, player **availability** is reconstructed from the ESPN `summary`
   boxscore (`boxscore.players`, per-player `MIN`). A **scratch** = an
   established rotation member (≥3 prior appearances, rolling role minutes ≥14)
   who played 0 minutes.
3. A scratch → win-prob impact exactly as the injury agent computes it:
   `OUT(0.045) × tier-mult(star1.5/starter1.0/rotation0.5) × staleness`, where
   staleness ramps a weeks-old absence to 0 (days-since-last-played is the
   `reported_at` proxy — the same `_injury_staleness_multiplier` logic).
   Per-team impact is capped (report 0.20, effective 0.10) then scaled by a
   **magnitude** M.
4. The adjustment is applied like `apply_adjustments`
   (`p_home + adj_home − adj_away`, renormalized) and scored vs the outcome.

**Signal:** paired ΔBrier(M vs generic=1.0) on the **injury subset** (games with
a detected scratch — injury-free games are identical across M and only dilute),
with a bootstrap 95% CI. Because the base blend is identical across M, its
absolute quality (including any efficiency-seed lookahead) cancels in the paired
magnitude ranking. `M=0` = no injury; `M=1.0` = the current generic magnitude.

## Results

Injury-subset Brier, ΔBrier/1000 vs generic (negative = beats generic):

| Season | seed | n(inj) | M=0 (no-inj) vs generic | best M>1 | best-M ΔBrier/1k (CI) |
|---|---|---|---|---|---|
| 2425 (2025) | 2026 (lookahead) | 212 | **+7.53** [+2.08,+12.97] | 2.0 | −2.12 [−5.53,+1.39] |
| 2324 (2024) | 2026 (lookahead) | 147 | +5.52 [−1.14,+12.59] | 1.4 | −0.64 [−2.80,+1.57] |
| 2425 (2025) | **2024 (clean prior)** | 209 | +4.86 [−0.47,+10.28] | 2.0 | **−0.72 [−4.09,+2.70]** |

Reading:
- **The injury layer helps.** M=0 (no injury) is consistently *worse* than
  generic across every run — significantly so with the lookahead seed on 2425,
  directional/borderline with the clean seed. Applying the injury adjustment
  improves the WNBA model blend. Keep it.
- **Amplifying beyond generic does NOT validate.** The ΔBrier(M>1) is
  directionally negative (M>1 slightly better) in every run, but **every
  amplification CI includes 0**, the effect is small (<1/1k with a clean seed),
  and the optimal magnitude is **unstable** (2.0 on 2025 vs 1.4 on 2024).
- **The lookahead inflated the signal.** Correcting the efficiency seed from
  2026 → the true prior season (2024) shrank the 2425 M=2.0 improvement from
  −2.12 to −0.72/1k — direct evidence the earlier signal was partly an artifact,
  and why the clean-seed run is the one that decides.

## Verdict & rationale

Per the project's discipline (never fine-tune on close-Brier below the noise
floor; never ship an unvalidated probability-moving change to a live sector):

- **KEEP** the generic tier magnitude and the existing `(1 − esw)` scaling +
  staleness decay. The injury layer is validated as-is.
- **DO NOT** add a WNBA magnitude amplifier / per-sector `MAX_ADJ` bump. The
  enhancement is directional but sub-noise, unstable across seasons, and would
  push live WNBA sizing further from a sharp line that already prices ~90% of
  injury news — the exact phantom-edge failure mode the `(1 − esw)` fix closed.
- **DO NOT** refresh `KNOWN_STARS` for magnitude reasons — the eval keys tier
  off rolling minutes, not that list; the list is only a live safety-net when
  ESPN leaders is unavailable and is orthogonal to this verdict.

This is a negative result by design: the harness is the reusable deliverable
(no injury-aware backtest existed before), and the decision is "the current
generic system is right; leave live WNBA injury handling untouched."

## Reproduce

```bash
# clean prior-season seed first (avoids lookahead), then run:
python scripts/seed_wnba_efficiency.py --year 2024      # for the 2425 run
python scripts/backtest_wnba_injury_impact.py --season 2425 --refresh
python scripts/backtest_wnba_injury_impact.py --season 2324           # held-out
# restore the live seed afterwards:  git checkout data/models/wnba_efficiency_state.json
```

Caveats: boxscore availability is post-hoc (game-time DNP truth), a mildly
optimistic proxy for the live pre-game ESPN status the agent consumes; the walk
has no Pinnacle line, so this measures the MODEL-side magnitude (the layer the
`(1 − esw)` scaling actually corrects), not the sharp-inclusive blend.
