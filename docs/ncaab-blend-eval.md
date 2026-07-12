# NCAAB / NCAAW opponent-adjusted efficiency blend — walk-forward validation

**Date:** 2026-07-11 · **Harness:** `scripts/backtest_ncaab_efficiency.py` ·
**Data:** ESPN D1 scoreboard+summary day cache (`scripts/seed_ncaab_efficiency.py`,
~5,800 D1 games/league/season)

## Protocol

- Strictly point-in-time: every prediction for game G uses only games before G.
  Elo/form walk forward from empty state; the efficiency state is re-solved every
  7 days from the current season's games before the cutoff (prior seasons never
  leak in — mirrors the live staleness guard, which darkens prior-season state
  in-season).
- ALL choices (elo K/HA, shrink_k, score_stdev, blend weights) swept on
  **2024-25 ONLY**, frozen, then evaluated once on the **2025-26 holdout**.
- Eval pool: games where the efficiency stack fired (both teams ≥ 4 games,
  post-mid-November) — the subset the new models actually affect. Baselines are
  scored on the identical pool.
- The solver is a ridge least-squares fit of `log(OE) = log(L) + α_team + β_opp
  + h·side/2` (KenPom-style opponent adjustment with jointly-estimated HCA),
  block-coordinate descent, unique optimum. Non-D1 opponents filtered by
  scoreboard-appearance threshold.

## NCAAB (men) — VALIDATED

Frozen train choices: shrink_k=6, score_stdev=10.5, blend eff .60 / sim .20 /
elo .10 / form .10 (train-optimal zeroed elo+sim but every candidate within
0.0004 Brier was equivalent on holdout — below the noise floor — so elo/form
are floored at 0.10 to keep a November fallback blend while the efficiency
stack waits for the current-season re-seed).

| 2025-26 HOLDOUT (n=4878) | Brier | Acc |
|---|---|---|
| **NEW blend (eff .60 / sim .20 / elo .10 / form .10)** | **0.1911** | **70.2%** |
| current blend (elo+form, live cfg K=20/HA=70) | 0.2038 | 68.4% |
| ncaab_efficiency standalone (k=6, sd=10.5) | 0.1917 | 69.8% |
| ncaab_possession_sim standalone (k=6) | 0.1918 | 70.1% |
| elo standalone (current cfg) | 0.2045 | 68.0% |
| form standalone | 0.2136 | 65.3% |

- **Gate: ΔBrier +0.0127 on holdout — PASS.** Both new models are the strongest
  standalone signals (not merely not-harmful). Leave-one-out: dropping
  efficiency costs −0.0225 Brier; no component of the new blend is harmful.
- Train (2024-25, n=4866): NEW 0.1880 vs current 0.2060 — same picture, so the
  edge is not a holdout fluke.
- Solver sanity: final 2024-25 adjusted top-15 (Duke, Houston, Auburn, Florida,
  Tennessee…) matches the season's actual KenPom/seed order; estimated
  HCA ≈ 4.8 pts/100 (~3.3 pts at pace 68). barttorvik T-Rank correlation check
  was skipped — the site now sits behind a JS anti-bot wall.
- Elo K/HA sweep (side result): (K=30, HA=90) beats the live (20, 70) on both
  train (0.2012 vs 0.2057) and holdout (0.2020 vs 0.2045) in the SIMPLE-elo
  harness. NOT wired: the live agent stacks MOV/SOS/recency multipliers on K,
  so a simple-harness K is not transferable, ncaab elo now carries only 0.10
  blend weight, and MODEL-2 scopes elo calibration to NCAAW/NHL.

## NCAAW (women) — VALIDATED

Frozen train choices: shrink_k=6, score_stdev=10.5, elo (K=35, HA=80 — the
MODEL-2 calibration), blend eff .60 / sim .20 / elo .10 / form .10
(deliberately identical weights to NCAAB: the train-optimal combo was
efficiency-only, every candidate within 0.0006 holdout Brier was equivalent,
and the 0.10 floors buy the November fallback).

| 2025-26 HOLDOUT (n=4731) | Brier | Acc |
|---|---|---|
| **NEW blend (eff .60 / sim .20 / elo .10 / form .10)** | **0.1608** | **75.5%** |
| current-doc blend (form only, per categories.yaml) | 0.1942 | 70.8% |
| current-actual blend (elo+form, uncalibrated K=20/HA=0) | 0.1861 | 73.1% |
| ncaaw_efficiency standalone (k=6, sd=10.5) | 0.1602 | 75.5% |
| ncaaw_possession_sim standalone (k=6) | 0.1618 | 75.2% |
| elo standalone (calibrated K=35/HA=80) | 0.1767 | 72.6% |
| elo standalone (uncalibrated K=20/HA=0) | 0.1873 | 72.1% |
| form standalone | 0.1942 | 70.8% |

- **Gate: ΔBrier +0.0253 vs the de-facto elo+form blend, +0.0334 vs the
  documented form-only blend — PASS.** Every new model beats every incumbent
  standalone. (Train, n=4705: NEW 0.1611 vs baselines 0.1998 / 0.1964.)
- **MODEL-2 (NCAAW elo calibration) shipped:** K=35, HOME_ADVANTAGE_ELO=80
  wired into `elo_agent.py`. HA=80 was properly bracketed (won at every K over
  {0, 40, 60, 80, 100} on train and holdout). K: the cold-start harness kept
  improving past K=50 (an extended sweep showed 50/80 at train 0.1853,
  holdout 0.1746) — but a walk from EMPTY state overstates the K appropriate
  for a warm persistent live state that additionally stacks MOV/SOS/recency
  multipliers on the base K, so the in-grid winner K=35 was wired instead of
  chasing the artifact. NHL half of MODEL-2 remains open.
- Solver sanity: final 2025-26 adjusted top-5 (UCLA, UConn, South Carolina,
  Texas, LSU) matches the season's actual hierarchy; estimated
  HCA ≈ 4.1 pts/100.

## Ops notes

- State files ship seeded from the completed 2025-26 season
  (`data/models/ncaab_efficiency_state.json`, `ncaaw_efficiency_state.json`).
- **November rule:** the staleness guard silences prior-season ratings once a
  new season starts (college roster churn makes the WNBA +24pp chalk-bias
  failure mode worse here). Re-seed weekly from opening night:
  `python scripts/seed_ncaab_efficiency.py --league mens` / `--league womens`.
  Until each team reaches MIN_GAMES=4, the blend falls back to elo+form+sharp.
- Recommend a fresh shadow-CLV check (`evmax cleanup shadow clv ncaab` /
  `ncaaw`) over the first weeks of the 2026-27 season before trusting the new
  blend live in November.
- Spread/total pricing unchanged (college sims are NOT wired into the
  spread-dist sim path; they cache margin/total distributions for future
  promotion work).
