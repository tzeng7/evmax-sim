# MLS disagreement-ramp evaluation — 2026-09-05

**Verdict: REFUTED. Keep MLS inheriting the sector ramp (0.04 / 0.10 / 1.00). No MLS-specific ramp ships.**

## Question

The soccer disagreement ramp in `EnsembleModelAgent.DISAGREEMENT_OVERRIDES["soccer"]`
was calibrated on pooled top-5 + UEFA data (n=4406, 2324–2526) where Pinnacle's close is
the informational ceiling. MLS runs at tier sharp_weight 0.40 on the assumption that the
ceiling does NOT hold there, yet the same ramp fires on every MLS game with a ≥4pt
model/sharp gap and is 100% sharp at 10pt — erasing the 0.60 model share on exactly the
games where a secondary-league edge could show. Live MLS divergence sits at 0.8pp despite the
0.40 base. Does an MLS-specific ramp recover a real edge?

## Protocol

`scripts/mls_disagreement_ramp.py`. MLS rows only (football-data.co.uk `USA.csv`), walk-forward
through fresh Elo/Form/Poisson/xG (xG is structurally absent — the file has no shot columns).
Models warm on calendar 2024, **train = 2025 (n=540, Pinnacle close)**, **holdout = 2026 to
May 25 (n=218, CONSENSUS close — the file carries no Pinnacle columns for the in-progress
season)**. Every candidate `(threshold, saturate_at, cap)` uses the live
`_disagreement_sharp_weight` on the MLS base weight 0.40. Paired Brier deltas vs sharp-only
carry a z-score.

## Results (3-way Brier, Δ = blend − sharp-only in /1000; negative beats sharp)

| blend | train 2025 | Δ | z | holdout 2026 | Δ | z |
|---|---|---|---|---|---|---|
| sharp-only | 0.61466 | — | — | 0.59640 | — | — |
| stat-only (elo+form+poisson) | 0.63186 | +17.2 | | 0.61794 | +21.5 | |
| **sector ramp (inherited today)** | 0.61513 | +0.47 | 0.47 | 0.59776 | +1.36 | 0.80 |
| flat 0.40, no ramp | 0.61973 | +5.07 | 1.54 | 0.61020 | **+13.80** | **2.92** |
| train-best 0.02/0.10/1.00 | 0.61505 | +0.39 | 0.50 | 0.59738 | +0.98 | 0.73 |

Per bucket, the bucket the thesis cared about — "model > sharp by 7+ pts" — is where flat
0.40 is worst (train 0.6669 vs sharp 0.6417; holdout 0.6321 vs 0.5864): when the MLS models
disagree hard with the close, the close is right. Binary Brier (P(home)) tells the same
story (flat holdout +9.69/1000, z 3.51; sector ramp +1.43, z 1.42).

### Base-weight sweep (sector ramp vs flat, 3-way)

| base sharp_weight | ramp | train Δ | holdout Δ (z) |
|---|---|---|---|
| 0.40 | sector | +0.47 | +1.36 (0.80) |
| 0.40 | flat | +5.07 | +13.80 (2.92) |
| 0.70 | sector | +0.13 | +0.56 (0.65) |
| 0.85 | sector | +0.04 | +0.25 (0.58) |
| 0.95 | sector | +0.01 | +0.08 (0.54) |

Monotone: the closer to sharp-only, the better. Nothing beats the close.

## Reading

1. **MLS is also a sharp-ceiling league on close-Brier.** The "secondary tier where our
   models might help" rationale in `data/soccer_league_tiers.yaml` is not supported by this
   walk-forward. The 0.40 base is held in check only by the inherited ramp.
2. **The ramp is doing its job on MLS, not suppressing an edge.** The train-best candidate
   beats the inherited ramp by 0.08/1000 (train) and 0.38/1000 (holdout) — noise (z<1).
   Ship rule was "beat the inherited ramp on BOTH windows outside noise"; it fails.
3. **Do not raise the MLS base weight on this evidence alone.** The soccer sector's edge (if
   any) is CLV vs the stale live line, not close-Brier (tennis lesson — see the Tennis blend
   note in CLAUDE.md). A base-weight change alters what gets FLAGGED, so judge it on
   `evmax cleanup shadow clv-leagues soccer` once the MLS league bucket has n≥30 clean
   resolved rows with CLV. The structural plumbing (`league` column, `--league` filters,
   per-tier `disagreement_ramp`) exists precisely so that decision can be made per league.

## Caveats

- Holdout is anchored to the consensus close, not Pinnacle (`parse_soccer_extra_csv`
  fallback). Numbers are "vs the close"; don't quote them as Pinnacle CLV.
- Holdout is small (n=218) and ends 2026-05-25 (football-data.co.uk was 503 on this run;
  the cached file from 2026-08-30 was used). Re-run once the 2026 season completes.
- xG never fires for MLS in this harness (no shot columns), so the stat side is
  elo+form+poisson only; live MLS does get xG from ESPN. This can only make the live stat
  side better than shown, but it does not change the bucket-level conclusion that flat
  0.40 is dangerous at large disagreements.
