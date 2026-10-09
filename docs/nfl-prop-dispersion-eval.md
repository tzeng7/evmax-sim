# NFL prop dispersion — fixed-σ Normal → fixed-scale Gamma (2026-10-09)

**Verdict.** NFL receiving and rushing yards move from a Normal with a fixed σ
(24 / 30 yd) to a **Gamma with a fixed scale θ** (shape μ/θ, so Var = θ·μ):
θ = **25.4** receiving, **15.3** rushing. NFL receptions **keep NegBin k=5**
(re-checked, no out-of-sample evidence for a change). Passing yards keep the
Normal σ=70 (within 1 SE of realized at every offset, n=130).

Reproduce: `python scripts/fit_nfl_prop_dispersion.py --archive-db <archive.db>
--pred-db <predictions.db>` (read-only, ~70 s). Shipped code:
`evmax/ev/prop_pricing.py` (`GammaProp`, `_GAMMA_STAT_SCALE`).

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
- **Outcome**: `prop_observations.actual_value` (nflverse-resolved; zero
  disagreements across rows of a player-game).
- **Rungs**: the Kalshi thresholds actually listed (`prop_observations`);
  97.9% carry a last-pre-kickoff Kalshi bid/ask mid (median 36 min pre-kick).

| stat | player-games | rungs |
|---|---|---|
| receiving_yards | 709 | 5,526 |
| rushing_yards | 348 | 3,155 |
| receptions | 698 | 4,683 |

## Method

Rung Brier (and log loss) of P(Y ≥ K) vs 1{y ≥ K}; each family fit by
minimizing train rung Brier with the anchor reproduced exactly. Validation:
walk-forward by NFL week (fit weeks < w, score w) and a frozen split (fit
Weeks 1–2, score Weeks 3–5 once). SEs clustered by player-game.

## Results — yardage (Weeks 3–5 holdout, Brier /1000)

| family | receiving | rushing |
|---|---|---|
| legacy Normal (σ 24 / 30) | 174.19 | 163.32 |
| Normal, σ refit | 170.35 | 161.70 |
| lognormal | 172.15 | 165.84 |
| Gamma, constant CV | 171.26 | 164.85 |
| **Gamma, fixed scale (shipped)** | **168.83** | **160.38** |
| Gamma, Var = φ·μ^p (p 1.26 / 1.37) | 168.82 | 161.09 |
| zero-inflated Gamma | 167.84 | 159.56 |

- Shipped − legacy: receiving **−5.36 ± 1.66**, rushing **−2.94 ± 2.29**.
  Walk-forward agrees (170.17 → 165.13; 159.73 → 156.98).
- Shipped − Kalshi mid, same rungs: receiving −0.61 ± 0.75 (on par with
  Kalshi), rushing +1.23 ± 0.95.
- θ is stable across walk-forward folds (receiving 20.6 → 25.1, rushing
  13.0 → 15.0) and the holdout Brier is flat for θ within ±5 of the shipped
  values, which are the all-weeks fit.

Holdout calibration, receiving (K − line bucket: realized / legacy / shipped /
Kalshi mid):

| K − line | realized | legacy | shipped | Kalshi |
|---|---|---|---|---|
| ≤ −30 | 0.676 ± 0.059 | 0.933 | 0.897 | 0.830 |
| −30…−10 | 0.728 | 0.782 | 0.778 | 0.726 |
| −10…10 | 0.484 | 0.497 | 0.499 | 0.477 |
| 10…30 | 0.284 | 0.215 | 0.251 | 0.234 |
| 30…50 | 0.136 ± 0.017 | 0.057 | 0.121 | 0.107 |
| 50…70 | 0.079 | 0.009 | 0.076 | 0.066 |
| > 70 | 0.062 | 0.001 | 0.051 | 0.045 |

## Results — receptions

Holdout Brier: k=5 155.84, refit k 156.35 (+0.52 ± 1.10), NB1 156.36.
Walk-forward ties (151.93 vs 151.53). The upper tail is unstable: Weeks 1–2
fit k≈50, Weeks 3–5 match k=5. One stable miss: k=5 prices P(≥ line−1)
~3.5 pp too low in both halves. k=5 kept.

## Rejected / residuals

- **Zero-inflated Gamma** (dud mass at 0): −1.0 ± 0.5 (receiving) /
  −0.8 ± 1.0 (rushing) vs the shipped Gamma, but it thins the upper tail and its
  mass parameter swings 0.03 → 0.18 across folds. Revisit with more weeks.
- **Lower tail of high-median receivers** is fatter than any fitted family
  (early exits / blowouts): rungs ≥ 30 yd below the line realize 68% vs 90%
  priced (Kalshi 83%).
- **Mid-median receivers (line 35–55)**: in-sample, the shipped tail runs fat
  at +60 yd (9.0% vs 4.5% ± 1.5%); a free variance power does not fix it. This
  leans toward phantom YES edges on deep rungs for that cohort.
- **Rushing anchor bias**: overs at the Pinnacle line hit 44–45%, not 50%. A
  shape cannot fix the anchor; it is the sharp's number.

## Archive labelling (same change)

The re-lined rungs were archived in `archived_sharp_odds` as `book='pinnacle'`.
They now go under `book='pinnacle_derived'` (`derived=1`, real anchor in
`anchor_line` / `anchor_prob_over`), and the raw Pinnacle prop anchors are
archived as `book='pinnacle'` (`derived=0`). The 466,966 legacy rows (NFL
336,787 · NBA 81,042 · baseball 49,137 on 2026-10-09) are relabelled by
`python scripts/relabel_derived_prop_rungs.py --apply` (dry-run by default,
batched, `--revert` undoes it).

## Follow-ups

- Refit θ as the season accrues (rerun the script; it reads the new
  `anchor_line` columns directly).
- NFL prop rows logged before this change (`model_version` older than the
  deploy SHA) were priced by the fixed-σ Normal; judge the new pricing only on
  rows logged after it.
