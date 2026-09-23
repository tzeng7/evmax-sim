# Multi-sector validation run — 2026-09-22

Scope: an objective-seeking pass over the under-proven sectors (NFL, NCAAF,
soccer by league, UFC, NHL) plus the cross-cutting process they share (blend,
CLV measurement, outcome resolution). Each sector got an independent read-only
audit (live DB + walk-forward/holdout replays + a literature/industry scan);
every change below shipped as its own PR with tests that fail on the old code.
Measured vs inferred is marked where it matters. Nothing here was merged by the
run; the order in §4 is the intended merge order.

## 1. Findings ranked by money impact

| # | Finding | Evidence | PR |
|---|---|---|---|
| 1 | **Dashboard / Discord / portfolio scans blended at `sharp_weight` 0.40**, the CLI at 0.85, for every sector without a per-sector entry (nfl, ncaaf, wnba, ncaab, ncaaw, nhl, worldcup). Rows were stamped 0.85 regardless. | Dashboard ML rows diverge 2.7–3.6pp from sharp vs 0.2–0.4pp for CLI rows; 6 placed live bets (2 NFL, 4 WNBA) since 09-06 sized on 0.40 blends. | #315 |
| 2 | **The ~30%-model-share rows lost.** Replaying resolved rows, every sector where models carried real weight (the 0.40 rows) was significantly worse than sharp. | Pooled ΔBrier +2.97/1000 (z 2.9); NCAAF dashboard +5.6/1000, CLV +0.02pp vs CLI rows +2.11pp (z 3.7); worldcup ROI −47% (z −3.8). No sector shows evidence for more model weight. | #315 (kept 0.85; documented) |
| 3 | **The shipped model share is (1−sw)² ≈ 2.25%, not 15%.** `_flb_correct` re-blends the already sharp-blended prob. Kept deliberately (finding 2); a test pins it. | Algebra verified against `_blend` to 5e-6 on 6,000 cases. | #315 |
| 4 | **CLV measurement biases.** NO-side rows scored NO-ask entry vs `1−yes_ask` (the NO bid); unplaced rows scored against closes that preceded entry; cancelled voids + at-tip entries inside the lenses; the gate counted alt rungs as samples; 164 totals outcomes with `pinnacle_close_prob = 0.0`. | Recompute dry-run: 1,335 / 4,843 rows move. NFL spread NO −0.19→+0.55pp, baseball total NO −0.58→+0.18, WNBA spread NO −1.76→−1.00, UFC +2.38→−0.75 (backward in-play rows). | #316 |
| 5 | **Outcome labels wrong.** (a) nested names (Utah/Utah State, Florida/FAU, …) graded the wrong side; (b) series games: completed-only ESPN fetch + ±1-day window graded rows on YESTERDAY's game of the series. | (a) 5 wrong outcomes + 7 flipped closes (incl. a live placed tennis bet); (b) 194 rows, 188 baseball (5 live). | #319, #322 |
| 6 | **NFL alt spreads priced by a normal CDF (σ 14) ignore key numbers.** | Key-number margin PMF: holdout 2019–25 ΔBrier −1.62/1000 (z −4.95), 7/7 walk-forward seasons, all gain on rungs crossing 3/7; MAE vs Pinnacle's own ladder 0.73pp vs 2.90pp; on our rows CLV +1.41pp where the PMF agrees vs −0.36pp where it doesn't; ~38% of logged NFL spread "+EV" rows phantom. | #317 |
| 7 | **Folded tail gate** leaked "underdog wins by X" rungs 15–21 pts off the main line into LIVE NBA/WNBA pricing. | WNBA live leak rungs priced 11.3%, won 0/11; WNBA shadow 21.5% vs 15.0%, CLV −1.77pp; NBA live CLV −1.62pp. | #320 |
| 8 | **NCAAF training data corrupt** (ESPN scoreboard-delta `score_points`; mislabeled `drive.team`). Reseed blocked two weeks. | 720 illegal deltas / 353 games (2021–26); ~2,100 possession-flipped plays / 800 games. Clean feed: v2 Brier 0.1867→0.1857, open→close slope +0.084→+0.102 (t 12.9). | #314 |
| 9 | **Two sectors silently dead.** UFC: Kalshi short titles "{Name} wins" → 0 matched since 08-23. NHL: no alias map → 0 matched EVER; season opens 2026-09-29. Integrity never alerted (zero baseline). | UFC live replay 0/50 → 28/28 matched (every fight Pinnacle has posted); NHL archive replay 0/236 ML → 182/236 with the map. LoL/CS2 have the same short-title break (0 matched; follow-up task). | #325, #321 |
| 10 | **MLS weight 0.40 never earned.** | clv-leagues MLS 59 games, +0.65pp, 47% positive (fails %pos≥55); taker EV vs close −1.05pp (t −4.0). | #318 |
| 11 | **Model lookups hit the wrong team's state.** Prefix/substring fallbacks: PSG read Paris FC's Poisson row; FCS teams priced as their FBS namesake; 6 NBA teams read a one-game duplicate record and the Clippers had no rating; accented soccer clubs invisible; Elo `update` wrote to a different key than it read (Troy +50). | Old vs new lookup over every archived Pinnacle label: soccer 43, ncaab 27, ncaaf 23, nba 21 changes; ~48 live soccer + 38 live NBA ML rows touched. | #323 |

## 2. Industry baseline (what the literature and pros do)

- **CLV vs the sharp close is the standard edge test.** Pinnacle's soccer close
  is efficient (opening vs closing RPS 0.2059 vs 0.2046 over 162k matches) and
  the pre-close/close odds ratio predicts yield with slope ≈1
  ([Buchdahl/Pinnacle](https://www.pinnacle.com/betting-resources/en/educational/have-pinnacles-soccer-markets-become-more-efficient/qcv2uvnqtqdw98gk),
  [football-data](https://www.football-data.co.uk/blog/pinnacle_efficiency.php)).
  Profitable public systems beat SOFT books using the consensus, not models
  (Kaunitz et al., [arXiv 1710.02824](https://arxiv.org/abs/1710.02824)).
  Our own findings agree: no sector's model beats the Pinnacle close on Brier;
  value is venue timing (Kalshi/PolyUS converging to sharp) and pricing
  correctness, which is exactly what CLV measures — so CLV measurement bugs
  (finding 4) are first-order.
- **Alt lines are priced from an empirical margin distribution conditional on
  the spread, per half point** — Unabated's alt-line tooling, nfelo's normal +
  key-number spikes, Wizard of Odds / Boyd's push charts (3 ≈ 14.5–15.4% of
  games, 7 ≈ 8.7–9.2%; a half point onto/off 3 ≈ 4.8pp). The normal
  approximation (Stern 1991, σ≈13.9; recent residual SD ≈12.7) is a known
  continuous stand-in for a discrete distribution (Glickman & Stern). Totals key
  numbers are weak (~4% each) — no totals change warranted.
- **Linear opinion pools are under-dispersed** and need recalibration
  (Ranjan & Gneiting) — the mechanism behind NCAAF's early-season underdog lean
  once models got real weight.
- **Prediction-market favorite–longshot bias:** Kalshi ≤10c contracts are
  taker-bought and lose; makers out-earn (Bürgi, Deng & Whelan); Kalshi ML is
  best calibrated 30–240 min pre-tip. Exchange/maker execution is a larger EV
  lever than model edge.
- **Sector specifics.** MMA markets show no FLB and are "largely efficient"
  (Miller & Nichols 2026); an open-source Glicko+XGBoost MMA model loses to the
  close. NHL moneylines are among the most efficient markets; goalie
  confirmation is the largest line mover but GSAx repeatability is weak
  (r≈0.22). CFB: preseason priors should carry returning production + transfer
  QB (SP+/FPI); SP+ phases the prior out by ~week 7. Soccer: lower leagues have
  wider margins but no proven lasting edge (Elaad/Reade/Singleton;
  Winkelmann et al.); market-derived ratings beat goals-based Elo (Wunderlich &
  Memmert).

## 3. Per-sector verdicts

- **NFL.** Moneyline models fire for Week 3 (the Week-1 blanks were the
  freshness guard/cold form, by design). Spread pricing → key-number PMF
  (#317) + true-axis gate; the spread CLV sample restarts on `spread_pmf`
  rows. NFL totals PolyUS "PROMOTE-READY" does NOT clear once declustered: 20
  games / 2 weeks with genuine EV ≥2%; net-of-fee CLV +1.34pp (t 4.3), under
  side carries it (+2.33pp) — keep collecting, verify fillability (PolyUS
  depth is never logged). Operational: the Monday 07:04 reseed runs before MNF
  (missed KC–DEN, Rams–Giants) — move the NFL part to Tuesday.
- **NCAAF.** After #314/#315 the sector is effectively sharp-passthrough at
  0.85 (mean |blend−sharp| 0.18pp in the replay) — the board's 2.34pp
  divergence and underdog lean were dashboard 0.40 rows. Early-season models
  are ~2× less dispersed than the close (elo/v2 logit SD 0.83/0.94 vs 1.60).
  Open: returning-production preseason prior (CFBD `/player/returning`, free
  key); FCS-namesake lookups (in the name-resolution PR).
- **Soccer.** No league shows model edge (model−sharp does not predict close
  movement: slope +0.15, CI −0.11..+0.39). At taker fees every live league is
  −EV vs the Pinnacle close (pooled −0.89pp, t −5.2, 197 games); the net-of-fee
  gate already demotes most rows. Model-side bugs: Poisson prices PSG as Paris
  FC; five accented clubs invisible to every model; promoted clubs rated ~0.64
  log-odds too high vs Pinnacle; Elo under-dispersed (market odds ≈1.49× Elo
  odds). Lead worth testing: **maker orders on TIE markets** (TIE gross EV vs
  close +0.51pp, only 7% of TIE bets moved against us; power-devigged
  Pinnacle close under-prices draws by ~1pp).
- **UFC.** No model edge: `ufc_rating` is 21/1000 worse than the DraftKings
  close on 2023–25 (z 3–5); market + all features ≈ market. Keep
  sharp-passthrough shadow; fix the parser so the venue-timing stream accrues.
  Only residual lead: age (t −1.96).
- **NHL.** Unmatched since forever → alias map (#321). xG with a prior ramp +
  elo 0.15 beats the current blend walk-forward (model-side +5.6/1000 confirm,
  +16.4/1000 first six weeks) but nothing beats the close (±0.03/1000); value
  would be CLV (model predicts line movement: slope +0.125, t 9.2). Do NOT
  regress NHL Elo (keep 1.0). Open: nothing reseeds `nhl_xg` weekly yet;
  measure Kalshi lag after goalie confirmations at 5-min resolution before
  building anything.

## 4. Merge order and post-merge operations

1. #314 NCAAF feed → then run `weekly-ncaaf-efficiency-reseed` (gate (c) should pass).
2. #315 sharp-weight source of truth (+ FLB share docs/test).
3. #316 CLV measurement → `evmax cleanup backfill-clv --since 2026-03-01 --recompute --dry-run`, review, then without `--dry-run`.
4. #319 resolver containment → #322 series (stacked) → `python scripts/repair_containment_outcomes.py` and `python scripts/repair_series_game_outcomes.py` (dry-run, then `--apply`).
5. #317 NFL PMF → #320 true-axis gate (stacked).
6. #318 MLS 0.85. 7. #321 NHL (before 2026-09-29; regenerate — never hand-merge — its state JSON if it conflicts). 8. #325 UFC parser + baseline-free zero-match integrity streak. 9. #323 model name resolution (then the NCAAF Elo/form rebuild it lists).
10. **Re-read every promotion verdict** (`cleanup shadow board`, `clv-leagues`, `clv-prices`) — the CLV recompute and outcome repairs change the evidence behind WNBA spread lay/take, baseball totals, UFC and NFL.

## 5. Do-not-re-chase (rejected this run)

- More model weight in any sector (finding 2). Changing the FLB double blend.
- NHL Elo offseason regression / early-K boost.
- UFC debutant prior (~11% more fights at ~2% weight from a model worse than the close).
- NFL totals key-number modeling (the CDF only extrapolates 0.5–2 pts).
- Per-league soccer model parameters (prior run) and a lower MLS weight without CLV evidence.
