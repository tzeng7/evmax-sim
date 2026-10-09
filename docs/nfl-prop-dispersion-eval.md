# NFL prop dispersion — fixed-σ Normal → fixed-scale Gamma (2026-10-09)

**Verdict.** NFL receiving and rushing yards move from a Normal with a fixed σ
(24 / 30 yd) to a **Gamma with a fixed scale θ** (shape μ/θ, so Var = θ·μ):
θ = **25.3** receiving, **15.3** rushing. NFL receptions **keep NegBin k=5**
(re-checked; a refit k is not better out of sample). Passing yards keep the
Normal σ=70 (within 1 SE of realized at every offset, n=130). New prop rows are
tagged `model_version = 'pinnacle-anchor-v2'`.

Reproduce (read-only on both DBs; the first run caches Kalshi settlements from
the public API, ~80 s):

```
python scripts/fit_nfl_prop_dispersion.py --archive-db <archive.db> \
    --pred-db <predictions.db> --kalshi-results <cache.json> --fetch-kalshi-results
```

Shipped code: `evmax/ev/prop_pricing.py` (`GammaProp`, `_GAMMA_STAT_SCALE`).

## Problem

`AgentCoordinator._fetch_props` prices every Kalshi `X+` threshold off ONE
Pinnacle anchor (line + devigged P(over)). With σ fixed, every player got the
same spread regardless of median, so:

- a high-median WR's deep rungs priced near zero — Ja'Marr Chase 2026-10-04,
  line 83.5 @ 50/50: 150+ priced **0.3%**, 160+ **0.08%**; Kalshi asked 9% / 8%
  (phantom NO edges);
- a low-median player put mass below zero (a 12.5-yd rusher had ~25% of its
  mass below −7 yd), distorting the low rungs.

## Data

- **Anchor**: last archived snapshot before kickoff per player-game, 2026
  Weeks 1–5. Legacy rows hold only re-lined rungs, but the anchor is recovered
  exactly: the stored decimals devig (power) to the anchor prob and inverting
  σ=24/30 (k=5) lands every line on a half point to ~1e-14, with all rungs of a
  snapshot implying one μ.
- **Outcome**: Kalshi's official settlement per rung (`result` yes/no; scalar =
  inactive player → void, dropped). `prop_observations.actual_value` is NOT
  enough on its own: a player who played but has no row in any ESPN box-score
  block (zero catches / carries) keeps a NULL stat value while Kalshi settles
  NO (since #374 the resolver writes `outcome` from that settlement; the stat
  value stays NULL). 60 such player-games are recovered here (y = 0).
  Where both sources exist they agree on all 13,364 rungs.
- **Rungs**: the Kalshi thresholds actually listed (`prop_observations`);
  97.8% carry a last-pre-kickoff Kalshi bid/ask mid as a benchmark.

| stat | player-games | rungs |
|---|---|---|
| receiving_yards | 739 | 5,708 |
| rushing_yards | 349 | 3,163 |
| receptions | 727 | 4,828 |

## Method

Rung Brier (and log loss) of P(Y ≥ K) vs the settlement; each family is fit by
minimizing train rung Brier with the anchor reproduced exactly. Validation:
walk-forward by NFL week (fit weeks < w, score w), and a frozen split (fit
Weeks 1–2, score Weeks 3–5 once). SEs are clustered by player-game.

**θ provenance.** The out-of-sample numbers below use θ frozen on Weeks 1–2
(21.2 receiving / 14.2 rushing). The SHIPPED θ is the all-weeks fit (25.3 /
15.3) and has seen Weeks 3–5. The holdout Brier moves ≤1.3/1000 across θ 20–35
(receiving) and ≤0.3/1000 across θ 15–20 (rushing), so the choice costs nothing
measurable.

## Results — yardage (Weeks 3–5 holdout, Brier /1000)

| family | receiving | rushing |
|---|---|---|
| legacy Normal (σ 24 / 30) | 173.75 | 163.14 |
| Normal, σ refit | 170.19 | 161.47 |
| lognormal | 172.48 | 165.55 |
| Gamma, constant CV | 171.19 | 164.54 |
| **Gamma, fixed scale (shipped)** | **169.03** | **160.13** |
| Gamma, Var = φ·μ^p (p 1.30 / 1.37) | 168.97 | 160.82 |
| zero-inflated Gamma | 167.46 | 159.29 |

- Shipped − legacy: receiving **−4.72 ± 1.61**, rushing **−3.01 ± 2.28**.
  Walk-forward agrees (169.49 → 165.02; 159.62 → 156.83).
- Shipped − Kalshi mid, same rungs: receiving −0.34 ± 0.72 (on par with
  Kalshi), rushing +1.21 ± 0.95.

Holdout calibration, receiving (K − line bucket):

| K − line | realized | legacy | shipped | Kalshi mid |
|---|---|---|---|---|
| ≤ −30 | 0.649 ± 0.060 | 0.933 | 0.900 | 0.830 |
| −30…−10 | 0.715 | 0.782 | 0.780 | 0.726 |
| −10…10 | 0.460 | 0.498 | 0.499 | 0.476 |
| 10…30 | 0.275 | 0.215 | 0.248 | 0.233 |
| 30…50 | 0.131 ± 0.016 | 0.057 | 0.119 | 0.106 |
| 50…70 | 0.077 | 0.009 | 0.074 | 0.066 |
| > 70 | 0.062 | 0.001 | 0.050 | 0.045 |

Receiving, P(Y ≥ line + d), one row per player-game, all weeks (shipped θ, so
in-sample):

| d (yd) | +20.5 | +40.5 | +60.5 | +80.5 |
|---|---|---|---|---|
| legacy | 20.3% | 4.8% | 0.6% | 0.0% |
| shipped | 25.3% | 13.0% | 6.7% | 3.4% |
| realized | 22.7% | 10.8% | 4.5% | 2.3% |

## Results — receptions

Holdout Brier: k=5 156.31, refit k 156.78 (+0.46 ± 1.07), NB1 156.83
(+0.51 ± 1.10); log loss also favors k=5. The fitted k is unstable (≈61 on
Weeks 1–2, ≈16 on all weeks). k=5 kept. Its real miss is the deep upper tail:
rungs ≥ line + 3.5 settle YES 4.4% on the holdout, k=5 prices 9.0% (Kalshi
4.4%) — a phantom-YES risk on cheap deep receptions rungs. (With the stat value
alone, k=5 also looked ~3.5 pp low at line − 1; that was the dropped zero-catch
games and mostly disappears with Kalshi settlements: 69.4% vs 70.6% ± 1.7%.)

## Rejected / residuals

- **Zero-inflated Gamma** (dud mass at 0): −1.57 ± 0.67 (receiving) /
  −0.84 ± 1.02 (rushing) vs the shipped Gamma, and it matches Kalshi's mid in
  every receiving bucket. Not shipped: its mass parameter drifted 0.05 → 0.15
  → 0.17 → 0.22 as weeks were added, and it prices the upper tail thinner than
  realized (50+ yd over the line: 5.6% vs 7.3%), the side the Normal got wrong.
  It is the lead candidate once more weeks accrue.
- **Lower tail of high-median receivers** is fatter than the shipped Gamma
  (early exits / blowouts): rungs ≥ 30 yd below the line realize 65% vs 90%
  priced (Kalshi 83%) → phantom-YES risk on deep-below rungs.
- **Mid-median receivers (line 20–55)**: in-sample, the shipped tail runs fat at
  +60 yd (line 20–35: 6.4% vs 3.2% ± 1.3%; 35–55: 8.9% vs 4.5% ± 1.5%) →
  phantom-YES risk on their deep rungs; a free variance power does not fix it.
- **Rushing anchor bias**: overs at the Pinnacle line hit 44%, not 50%. A shape
  cannot fix the anchor; it is the sharp's number.

## Archive labelling (same change)

The re-lined rungs were archived in `archived_sharp_odds` as `book='pinnacle'`.
They now go under `book='pinnacle_derived'` (`derived=1`, real anchor in
`anchor_line` / `anchor_prob_over`), and the raw Pinnacle prop anchors are
archived as `book='pinnacle'` (`derived=0`). Legacy model-output rows are every
`::prop::` row with `derived IS NULL` — 601,664 on 2026-10-09 (NFL 339,223 ·
NBA 213,277, of which ~132k are 2026-03-24 → 05-10 L15-era rows with no prop
columns and 1.0/1.0 decimals · baseball 49,164). `python
scripts/relabel_derived_prop_rungs.py --apply` relabels them (dry-run by
default, batched, `--revert` undoes it) — run it after the change is deployed.

## Follow-ups

- Refit θ as the season accrues (rerun the script; it reads the new
  `anchor_line` columns directly).
- Judge the new pricing on `model_version = 'pinnacle-anchor-v2'` rows; v1 rows
  priced NFL receiving/rushing yards with the fixed-σ Normal.
- Done in #374: the resolver now writes a zero-stat game's `outcome` from
  Kalshi settlement (`actual_value` stays NULL). Re-resolve rows from before
  that fix before reading MODEL-9 metrics.
- A cheap-longshot gate for NFL props: pre-fix shadow rows with EV > 15%
  returned −17% ROI, ≤ 10¢ −32%.
