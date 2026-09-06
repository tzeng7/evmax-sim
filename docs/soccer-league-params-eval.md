# Per-league soccer model parameters — walk-forward evaluation, 2026-09-05

**Verdict: both REJECTED. Poisson keeps one soccer-wide `league_avg`; Elo keeps the soccer-wide
`HOME_ADVANTAGE_ELO=60`. Neither per-league parameter transfers out of sample, and neither moves
the blended (bettable) probability at all.**

Harness: `scripts/backtest_soccer_league_params.py` — every league (top-5 + MLS, 5,686 rows,
2324–2526 / MLS 2024–2026) replayed chronologically through fresh Elo/Form/Poisson/xG exactly
as `soccer_walkforward.walk_forward_predictions`, with the candidate swapped in. Train = 2425
(MLS: calendar 2025), holdout = 2526 (MLS: 2026). Paired 3-way Brier Δ = candidate − baseline;
negative is better. Blended = tier sharp_weight + the sector disagreement ramp, i.e. what a bet
sees.

## A. Poisson `league_avg` per league (running mean of that league's home/away goals)

Motivation: goals/game differ materially by league on the train window — Bundesliga
1.63/1.50, Serie A 1.34/1.22, MLS 1.66/1.36 — while the walk-forward baseline uses one fixed
1.55/1.15 pair.

| scope | train Δ (z) | holdout Δ (z) |
|---|---|---|
| Poisson standalone, ALL (n=2278 / 1111) | **−3.92/1000 (−3.02)** | −0.16/1000 (−0.08) |
| Bundesliga standalone | −12.28 (−3.43) | +0.61 (+0.09) |
| EPL standalone | −9.17 (−2.01) | +0.63 (+0.11) |
| Serie A standalone | −3.86 (−1.30) | −7.64 (−1.62) |
| Ligue 1 standalone | +2.16 (+0.51) | +6.50 (+1.29) |
| **Blended, ALL** (n=2291 / 1120) | +0.01 (+0.05) | −0.20 (−0.96) |

The in-sample gain is real and significant but does not survive the holdout: pooled Δ
collapses to noise and the per-league signs scatter. In the blend the effect is nil either
way — Poisson's 0.40 share sits behind the ramp and the tier sharp_weight. The λ level
largely cancels in a 3-way ranking (team ratios are computed against the same baseline), so
this is a calibration-of-λ story that the sharp anchor already absorbs.

## B. Elo home-field advantage per league (train-swept, 30 → 110)

| league | train-best HFA | Elo standalone holdout Δ (z) |
|---|---|---|
| EPL | 30 | +2.08 (+0.49) |
| La Liga | 60 (= baseline) | 0 |
| Bundesliga | 30 | **+7.75 (+1.61)** |
| Serie A | 30 | −7.56 (−1.76) |
| Ligue 1 | 75 | −2.27 (−0.93) |
| MLS | 75 | −1.21 (−0.57) |
| ALL standalone (n=1120) | | −0.47 (−0.34) |
| ALL blended | | −0.02 (−0.29) |

Train-optimal HFAs do not transfer: the league that gained most in-sample (Bundesliga at 30)
is the one that loses most out of sample. Season-to-season HFA noise dominates any stable
per-league difference at this sample size, and Elo's 0.15 blend share makes the bettable
probability indifferent regardless.

## What this closes

Together with the MLS ramp evaluation (docs/mls-disagreement-ramp-eval.md), this closes the
"should soccer be separated by league" modelling agenda on close-Brier evidence:

- per-league sharp_weight tier: already in place; the MLS 0.40 base is unsupported on Brier
  (kept, ramp-capped; decide on CLV);
- per-league disagreement ramp: plumbing shipped, MLS-specific value refuted;
- per-league Poisson λ baseline: refuted on holdout;
- per-league Elo HFA: refuted on holdout.

The league dimension's value is on the MEASUREMENT side — `ev_predictions.league`,
`cleanup shadow clv-leagues soccer`, `--league` filters — where per-league CLV decides the
one lever that is a policy rather than a model: how much of each league's blend is the model
(the tier weight). Re-open a per-league MODEL parameter only with a new season of holdout
and a walk-forward that clears both windows.
