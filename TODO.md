# evmax TODO

> Items discovered via full codebase quality sweep (April 2026).
> PLAN.md covers older completed work (Batches A–F ✅). This file tracks what's next.
> Each item has a priority: **P1** (correctness/money), **P2** (quality/coverage), **P3** (nice to have).

---

## Recently Shipped

### Diversification build — soccer/tennis/baseball model independence + promotion board (branch claude/diversify-wnba-moneyline)
Landed 2026-07-19. Motivated by the measured sharp-passthrough diagnosis: 30d ML blend
divergence from sharp was wnba **4.82pp** (the only working sector) vs tennis 0.17 /
baseball 0.10 / soccer **0.00** — every non-WNBA "edge" was venue divergence, not model signal.
- ✅ **Soccer/MLS** — `seed_espn.py` walks MLS (`usa.1`, Feb→current window) + UEL; xG seed
  canonicalization + worldcup-namespace fixes; `MIN_NONSHARP_MODELS` guard (sharp-only soccer
  ML → shadow) + contamination rule for the 11 pre-guard live MLS rows. MLS walk-forward
  validated (blended 0.2269 ≤ sharp 0.2272 pooled; 2026 holdout within the +3/1000 gate)
  via football-data `USA.csv` extra-league support (`--leagues USA`).
- ✅ **Tennis coverage** — `merge_ranking_priors` matchmx supplement (≤90d recency) keeps
  leaderboard-churned players on the 0.48-conf ranking fallback (~29 rows/month recovered);
  advanced agent on shared `resolve_player`; no threshold cuts.
- ✅ **Why-not diagnostics** — `ev_predictions.model_diagnostics` (fired/gated/missing per
  model, captured at the ensemble gate); `evmax cleanup shadow show --why`.
- ✅ **Promotion board** — `evmax cleanup shadow board` + `/api/promotion-board` + dashboard
  Board tab: per (sector, market, venue) clean-n / ΔBrier vs sharp / staleness-filtered CLV
  gates / blend-divergence pp (sharp-passthrough detector) / verdict ladder.
- ✅ **pitcher_v2** — xERA-led starter blend (Savant client), per-pitcher-sample park
  normalization (symmetric venue factors cancel in Pythagenpat — regression-tested), 60/40
  available-bullpen blend (shared `models_ml/bullpen.py` + live fatigue feed), team-offense
  clamp. Rename `pitcher`→`pitcher_v2` resets the clean shadow sample via the contamination
  signature. A/B harness `scripts/backtest_baseball_ab.py` (per-component variants; pen
  carries a prior null result it must beat; frozen prior-season xERA = leak-free).
- ✅ **pitcher_v2 A/B verdict: SHIP** (docs/baseball-pitcher-v2-ab.md) — 2026 holdout
  standalone Δ −5.46/1000, blend −0.19/1000, coverage 100%; all three gates pass. pen
  REVERSED its prior null on the holdout (−7.28/1000; fatigue is a fresh-season signal).
- ⏳ **Post-merge actions** — run the soccer/pitcher/tennis seeds; create the 3
  `ev-scan-light-*` tasks (docs/scheduled-tasks/ev-scan-light.md — gated on the soccer guard
  being on main); baseball promotion re-judged on fresh v2 shadow CLV
  (`--max-staleness-h 3`, n≥30/mean≥0/%pos≥55 — v1's fresh-close baseline was 29% pos).

### UFC/MMA sector — shadow-mode MVP, moneyline only (branch claude/ufc-sector)
Landed 2026-07-11. Validate-first build: Phase 0 confirmed both venues live (Kalshi `KXUFCFIGHT` fight-winner series + Pinnacle sport 22 "UFC"), then models were gated on a walk-forward backtest BEFORE wiring went live.
- ✅ **Data** — `evmax/clients/ufc_espn.py` + `scripts/fetch_ufc_history.py`: ESPN MMA public API (ufcstats.com is behind a JS anti-bot interstitial we don't bypass), 7,959 bouts 2010→present with methods + 2,610 fighter bios, committed to `data/backtest/ufc/*.csv`, raw JSON disk-cached under `data/cache/ufc_espn/`.
- ✅ **Model** — `ufc_rating` (`evmax/agents/models/ufc_rating_agent.py`): Glicko-2 per fighter (`evmax/models_ml/glicko2.py`, validated against Glickman's paper example; per-bout rating periods + inactivity RD inflation) + logistic feature layer on point-in-time differentials (age, layoff, recent-KO, experience, win rate, streak, reach, height, finish rate, KO-absorbed). Fit orientation-symmetrized on ≤2024 (n=4,136).
- ✅ **Gate passed** — `scripts/backtest_ufc_model.py`, holdout 2025→present (n=711): blend Brier **0.2250** / acc 62.9% vs rating-only 0.2463 / 56.8%; calibration sane (no n≥50 bucket off >10pp). Verdict VALIDATED_BLEND → feature layer embedded in `data/models/ufc_rating_state.json` (`scripts/seed_ufc_ratings.py`, weekly `--fetch` reseed). Full tables in `docs/ufc-model-eval.md`.
- ✅ **Wiring** — tennis-pattern individual-competitor sector: surname-canonical matching (`evmax/sectors/aliases/ufc.yaml`; same-surname ambiguity → None via `tennis_common.resolve_player`), UFC title parsing in `kalshi.py`, `ufc` in `_US_SECTORS` (ET game-day alignment), `kalshi_settlement` resolver (removed from `ESPN_SPORT_MAP`), sharp_weight 0.88, ensemble override ufc_rating 1.0 / elo+form 0.0. Live smoke: Kalshi↔Pinnacle exact key match on the McGregor–Holloway card, 2/2 matched, shadow row persisted.
- ⏭ **Next**: accumulate shadow data; promote via `evmax cleanup shadow promote ufc` once n≥30 resolved with CLV ≥ 0. Consider a weekly scheduled task for `seed_ufc_ratings.py --fetch` (currently manual).

### NBA player props calibration — Path A (commits d1eb000, fd2e650, 5f65ae5)
Landed April 2026. Goal stated by user: improve NBA props enough to warrant promotion from shadow to live.
- ✅ **Baseline measurement** — `scripts/backtest_nba_props.py` replays the production prop model math against 26k 2024-25 player game logs pulled via nba_api. Surfaced the actual problem: the model is dramatically under-confident in the 50–70% probability bucket. Predicted 65% bucket realizes 82%; predicted 60% bucket realizes 73%. Symptom of the manual `_BASE_RATE = 0.40, _SHRINKAGE = 0.20` shrinkage chain in `compute_prop_prob_cached` pulling genuine high-confidence overs back toward the league base rate.
- ✅ **Calibration pipeline** — `scripts/calibrate_nba_props.py` precomputes the model's intermediate quantities (raw model_prob, weighted_hit_rate, avg_margin, last5_hits, minutes_volatile) once per row, then grid-searches `base_rate × shrinkage × blend_model` and fits an `IsotonicRegression` post-hoc layer. Train: Jan-Feb 2025 (130k rows). Test: Mar-Apr 2025 (109k rows).
- ✅ **Tuned constants** — `data/models/nba_props_calibration.json` ships `base_rate=0.30`, `shrinkage=0.0`, `blend_model=0.80` plus a 182-breakpoint isotonic mapping. The fitted "shrinkage=0" finding is significant — the manual shrinkage was a workaround for the calibration miss; with isotonic doing real calibration it's no longer needed.
- ✅ **Production wiring** — `evmax/clients/nba_props_cache.py` loads the JSON on first prop computation and caches in module state. Falls back to original constants when missing. `_apply_isotonic()` runs at the end of `compute_prop_prob_cached`, after volatility shrinkage and opp_adj. Implemented as binary-search piecewise-linear interpolation so we don't drag sklearn into the live scan path.
- ✅ **Calibration result** (Mar-Apr 2025 test, n=108,559): Brier 0.1969 → 0.1928 (−2%). Pooled Brier movement is modest because miscalibration cancels at the population level. **The real win is bucket-level shape**: 50–60% bucket goes from +12.9pp miss to +0.1pp; 60–70% from +11.5pp to +0.2pp; every bucket within ±1.6pp of perfect calibration. Matters more for the live scanner's bet selection than for aggregate accuracy — when the calibrated model says 75% on a market priced at 70%, that's a real edge the un-calibrated 65% would have missed.
- ✅ **Path B closed negative** — `scripts/backtest_nba_props_v2.py` tested adding career-mean prior + days-rest factor on top of Path A. Days-rest produced literally zero Brier movement (0.1226 with rest, 0.1226 without). Investigation: the empirical 2024-25 distribution shows long-rest games are load-management cases with minutes restrictions, not "freshness" cases — players with rest=4+ score 6.4 PTS on 14 MIN. The signal direction was wrong and per-36 normalization absorbs the residual. Career-mean was blocked on stats.nba.com rate-limiting; lives in P3 future work.
- ✅ **Ancillary infrastructure** — `scripts/fetch_nba_game_logs.py` (nba_api LeagueGameLog → parquet); `evmax/backtest/sources/shufinskiy.py` + `scripts/fetch_shufinskiy.py` (5 seasons of shotdetail + pbpstats locally cached as parquet). Loader is sub-100ms after the first read. Useful for any future model iteration; not currently consumed by Path A.
- ✅ Tests: 1129 passed, 1 pre-existing fail (`test_flb_correction_extreme_longshot`, unrelated). 0 new regressions.

### WNBA model ensemble for 2026 season (commit 266f152)
Landed April 2026 ahead of the May 16 2026 WNBA season opener. Goal: get WNBA out of shadow-only and to NBA-parity Brier before opening day.
- ✅ **Offseason Elo recalibration** — `scripts/wnba_offseason_regress.py` + `data/models/wnba_2026_offseason.yaml` apply 35% shrinkage toward 1500 plus 38 per-team roster-move deltas (Reese → Atlanta, Plum → LA, Thomas → Phoenix, Bueckers #1 / Fudd / Betts #4 / Amoore #6 / Miles #2 rookie deltas, Sabally → NYL, expansion priors for Fire 1470 / Tempo 1440). 2025-end Elo range (1335-1690) shrunk + churned to a 2026-opener range (1419-1609). Re-runnable per season by editing the YAML move list.
- ✅ **`WNBAEfficiencyModelAgent`** (new file, `name="wnba_efficiency"`, own state file) — Normal-CDF margin model with WNBA-tuned `HOME_EDGE_PTS=2.6`, `SCORE_STDEV=12.5`, `MIN_GAMES=12`. Reads ORTG/DRTG/Pace/eFG%/TOV%/OREB%/FTr from `wnba_efficiency_state.json` seeded by `scripts/seed_wnba_efficiency.py` (walks 2025 WNBA ESPN scoreboard + box scores, derives stats via Dean Oliver formulas, filters All-Star / exhibition contamination). Seed produced 13 teams × ~47 games each; league avg ORtg 100.6, pace 81.1.
- ✅ **`WNBAPossessionSimAgent`** (new file, `name="wnba_possession_sim"`, own cache) — Monte Carlo possession-level sim mirroring NBA's architecture but reading the WNBA efficiency state. 10k sims per matchup, pace clipped to [65, 100], margin σ=12.5, total σ=18.0. Exposes `cover_probability` / `total_probability` ahead of WNBA spread+total live promotion.
- ✅ **Form staleness guard** (`STALE_DAYS=60`) — `FormModelAgent.predict_pair` now returns `None` when the most recent record for either team is older than 60 days relative to the game date. Fixes the WNBA offseason bug where October 2025 records were being read as May 2026 signal. Opponent-quality weighting was tried and backtested net-negative (Form Brier 0.2470 → 0.2615) — reverted; Elo already captures opponent strength, adding it to Form just adds variance.
- ✅ **Ensemble override for WNBA** — `SECTOR_WEIGHT_OVERRIDES["wnba"]` blends wnba_efficiency 0.25 + wnba_possession_sim 0.25 + elo 0.30 + form 0.15, Poisson zeroed (basketball is not a Poisson process; same reason as NBA).
- ✅ **Walk-forward result** (2025 WNBA, 323 games): ensemble Brier 0.2212 → **0.2056** (−0.0156), accuracy 63.7% → **68.8%** (+5.1pp). Gap to NBA ensemble (0.2032) now 0.0024 Brier — inside the sample-size noise floor of 321 games. Both standalone WNBA advanced models are near-tied with their NBA siblings: wnba_efficiency 0.2020 vs NBA efficiency 0.2030; wnba_possession_sim 0.2025 vs NBA possession_sim 0.2022.
- ✅ **NBA regression check** — NBA walk-forward produces byte-identical Brier 0.2032 before and after the WNBA port. All NBA files / constants / state paths untouched by design (zero-shared-file architecture).
- ✅ **Registry + categories wiring** — `KNOWN_MODELS` gained `wnba_efficiency`, `wnba_possession_sim`. `data/categories.yaml::wnba.models` updated to `[elo, form, wnba_efficiency, wnba_possession_sim, sharp]`. `validate_registry()` passes.
- ✅ **Tennis backtest tool** — `scripts/backtest_tennis_model.py` replays tennis-data.co.uk xlsx (2024/2025/2026) through the full tennis ensemble for in-sample calibration diagnostics. Not production infra, but useful for spot-checking tennis-surface drift.
- ✅ Tests: 1091 → 1093 passing (added `test_stale_records_return_none` + `test_is_stale_uses_reference_date`). Pre-existing 2 failures + 16 nfl_props_live errors are environmental, not from this work.

### NFL prop backtest infrastructure (Stage 4 of feat/nfl-prop-backtest)
- ✅ Kalshi historical pull — `scripts/fetch_nfl_prop_history.py` pulls 31,869 settled NFL prop markets across 1,164 events, Dec 1 2025 – Feb 8 2026, via unauthenticated `/historical/markets?event_ticker=...`. Wrote to `data/backtest/nfl_props/kalshi_raw.json` (76 MB).
- ✅ nflverse feature pull — `scripts/fetch_nfl_features.py` pulls weekly stats / schedules / weekly rosters direct from nflverse GitHub releases (ditched `nfl_data_py` — it hardcodes a dead URL). Writes parquet, spot check confirmed point-in-time filtering works.
- ✅ Join + shape report — `scripts/join_nfl_prop_backtest.py` joins Kalshi → nflverse via normalized name matching, 99.7% match rate (30,601 / 30,693) after offensive-position-aware ambiguity resolution, roman-numeral suffix handling, nickname aliases (Hollywood → Marquise Brown, Joshua → Josh Palmer), empty-gsis-id filter, and "* Defense" / "No Touchdown" non-player drops.
- ✅ Pure probability model — `evmax/clients/nfl_props_cache.py` with yardage (normal CDF + decay + empirical blend + streak), poisson (anytime TD + passing TDs), and count (receptions) branches. Per-stat MIN_STD floors. 23 unit tests covering monotonicity, opponent adj, thin-sample gating, poisson threshold stepping.
- ✅ Backtest loader — `evmax/backtest/sources/nfl_props.py` with point-in-time feature extraction, pre-grouped player history, rolling opponent defense per (season, week, team), league-average baselines. `--stats` filter for sub-setting.
- ✅ Prop-specific metrics + display — `metrics_props.py`, `display_props.py`, `PropBacktestRow` / `PropBacktestReport` dataclasses. Engine + CLI dispatch.
- ✅ Closing-price leakage fix — ROI calculation now skips markets with closing price outside [0.05, 0.95] because `last_price_dollars` on settled Kalshi markets converges toward 0/1 near settlement. Market baseline Brier applies the same filter.
- ✅ Full suite: 772 → 795 tests passing.

**✅ Gate status (Stage 4 + MODEL-8):** NFL prop modeling passes all three gates via a **NO-side-only betting strategy**, with a leakage caveat that requires shadow validation before live bets (see MODEL-9).
- QB-only pooled Brier: **0.179** (gate < 0.22, passes).
- Per-stat Brier: passing_tds 0.169, passing_yards 0.180.
- **ROI at ev≥3%, vol≥1000, NO-side only: +79.1%** (247 bets, 74.9% win rate).
- **ROI is monotone in EV threshold** (+43% at ev=2% → +115% at ev=8% vol-gated) and **monotone in volume gate** (+43% ungated → +76% at vol≥1000). Both are consistent with a real edge, not noise.
- **YES-side is catastrophic (−88% ROI)** at every EV threshold — the model's upper-tail overconfidence (still 12pp miss at the 90–100% bucket after MODEL-8) is concentrated on exactly the markets it flags as YES edges. Kalshi NFL prop markets also appear systematically overpriced on YES, likely retail excitement betting.
- **Leakage caveat:** the backtest uses `last_price_dollars` (closing price) not a pre-game snapshot. A market settling NO drifts downward through the game as events unfold. The ROI signal could be partly/wholly retrospective. Volume-gating mitigates but doesn't eliminate this. See MODEL-9 for the shadow-mode validation plan required before live bets.
- Non-QB stats (rec_yds, rush_yds, rec, anytime_td) still have Brier ≥ 0.21 — need usage/target-share/Vegas features, see MODEL-7.

### PR #5 — Tennis surface resolver from Kalshi competition + weight trim
- ✅ MODEL-1 / SECTOR-3 — Tennis surface now detected from Kalshi `event.product_metadata.competition` (structured "{ATP|WTA} {City}" strings) instead of scanning generic market titles. `KalshiClient.get_markets` fetches `/events` in parallel with `/markets` for tennis and joins by `event_ticker`. `PredictionMarket` gained an optional `competition` field (additive, zero SQL migration). Resolver returns `(surface, is_indoor)` with longest-match-first dict ordering so brand aliases like "stuttgart open" (grass) beat ambiguous "stuttgart" (clay). Indoor check gated on `surface == hard` since clay/grass are always outdoor on tour.
- ✅ MODEL-5 — `TennisModelAgent.weight` trimmed 0.45 → 0.35 so tennis no longer dominates the ensemble blend when competing with other models.
- ✅ Resolver is total (try/except → `("hard", False)`), structured `tennis.surface_resolved` log on every call, 1000-run fuzz test verifies never raises.
- ✅ CLI cleanup — `evmax agents update --surface indoor` and `evmax agents seed-tennis surface --surface indoor` now reject with a clear error pointing at MODEL-6. Only `hard/clay/grass` accepted.
- ✅ Live Kalshi fixtures captured 2026-04-13 to `tests/fixtures/kalshi/` (4 JSON files, offline-replayable in CI forever).
- ✅ Sackmann xlsx replay test (`tests/test_tennis_surface_replay.py`) at 99.0% accuracy on both 2024 (2676/2703) and 2025 (2617/2644) — designed as a coverage-expansion loop that drove dict additions until passing the 95% floor.
- ✅ Three-layer Kalshi fixture replay tests (`tests/test_tennis_kalshi_fixtures.py`): resolver replay on captured events, end-to-end `get_markets()` join via monkey-patched `_get`, and explicit `is_indoor` seam verification preventing silent drift before MODEL-6.
- ✅ Semantic correction documented: **MODEL-6** filed with full `(surface, court)` orthogonality context, explanation of the 51 stale legacy `indoor` ratings from manual CLI updates (inert post-merge), and three migration options for when the court-adjustment factor lands. Blocker: needs live indoor-event accumulation.
- ✅ Suite: 705 → 772 (+67 net; 103 tennis-related tests across the three test files, ~36 old `TestSurfaceDetection` tests were replaced by the new `TestResolveSurface` class).

### PR #2 — Poisson bucketing + prop injury boost + pitcher Pythag + TEST-2/TEST-6
- ✅ BUG-4 — Poisson NBA/NFL/NCAAB score matrix now bucketed (NBA=5, NFL=4, NCAAB=5). `lam_h`/`lam_a` scaled by bucket before the matrix is built so Poisson mass fits inside MAX_SCORE.
- ✅ BUG-5 — Prop injury boost now uses `nba_props_cache.lookup_player_team()` to find the player's actual team instead of parsing a nonexistent game slug out of `parts[2]`. The boost was silently never firing for any prop.
- ✅ BUG-9 — Pitcher Pythag semantics corrected. Previously `home_ra = away_era` treated a team's runs allowed as a function of the *opposing* starter, making the team with the better pitcher *less* favored. Fixed to `home_rs = away_era, home_ra = home_era` (each team scores at opponent's rate, allows at its own pitcher's rate). A 2.50 vs 5.50 ERA matchup now produces home_wp ≈ 0.80 instead of ~0.38.
- ✅ TEST-2 — PitcherModelAgent now has 14 tests across sector gating, seeding, Pythag prediction, confidence tiers, and team-name fallback. Writing the tests surfaced BUG-9.
- ✅ TEST-6 — 44 tests added for the prop pipeline: PropMatcher matching logic, `nba_props_cache` (lookup_player_team, compute_prop_prob_cached, _opponent_adjustment), and `prop_resolver` pure helpers (_normalize_for_match, _extract_stat). Network-dependent paths (`fetch_player_stats` body, `resolve_prop_observations`, `refresh_props_cache`) intentionally not covered here — live in the network tier.
- ✅ Suite: 640 → 705 (+65).

### PR #1 — Quality sweep: docs sync, resolution gaps, tennis tests
- ✅ DOC-1 — CLAUDE.md stale facts fixed (rate limiting, NCAAW source, sharp weight, clients tree, tennis weight, resolution table, sectors)
- ✅ DOC-2 — Folder READMEs added for `evmax/clients/` and `evmax/models_ml/` (the two most confusing dirs). Remaining: `evmax/agents/`, `evmax/sectors/aliases/`.
- ✅ DOC-3 — `__init__.py` docstrings upgraded across `clients`, `ev`, `matching`, `sectors`, `models_ml`, `pipeline`, `agents/cleanup`, `agents/models`. Also fixed a missing `TennisModelAgent` / `PitcherModelAgent` export. Remaining: a handful of minor packages (`web`, `cli`, `backtest`, `players`).
- ✅ DOC-4 — Pre-commit `doc-and-test-sync-reminder` hook installed. Two layers: doc-sync warns when source changes without related doc updates; test-sync warns when source changes without test changes (with `[ZERO COVERAGE]` red label for modules with no test file at all).
- ✅ Testing Policy section added to CLAUDE.md
- ✅ `setup.sh` for one-command dev setup
- ❌ BUG-1 — **false positive**: tennis already auto-resolves via Kalshi settlement at `resolver.py:850`. The audit missed that branch because tennis isn't in `ESPN_SPORT_MAP`. Removed.
- ✅ BUG-2 — NHL added to `ESPN_SPORT_MAP` (`hockey/nhl`)
- ✅ BUG-3 — UEL (`uefa.europa`) + MLS (`usa.1`) added to `ESPN_SOCCER_LEAGUES`
- ✅ BUG-6 — `FORM_STATE_PATH` now absolute (was relative, silently failing under non-root CWD)
- ✅ BUG-7 — Dead `"over_under"` string removed from `EVGap.display_label`
- ✅ BUG-8 — All five remaining `datetime.utcnow()` call sites cleaned up (also caught a latent bug in `BankrollSnapshot` where the default was a frozen value)
- ✅ TEST-1 — 51 tennis model tests across 8 test classes; total suite 589 → 640

---

## Section 1 — Documentation & Auto-Sync (remaining)

### DOC-2b Per-Folder READMEs (remaining) [P3]
Still missing dedicated READMEs:
- `evmax/agents/` — explain pub/sub bus, AgentRequest/Response, coordinator lifecycle
- `evmax/sectors/aliases/` — explain YAML alias format and how fuzzy matching uses them

### DOC-3b `__init__.py` Docstrings (remaining) [P3]
Minor packages still with stub docstrings: `evmax/web/`, `evmax/cli/`, `evmax/cli/commands/`, `evmax/backtest/`, `evmax/players/`, `evmax/backtest/sources/`. Low impact since these are leaf packages, but easy wins.

---

## Section 2 — Bugs (Correctness Issues)

_(BUG-4, BUG-5, and BUG-9 shipped — see "Recently Shipped" above.)_

---

## Section 3 — Model Quality

### ~~MODEL-1 Tennis Surface Detection~~ ✅ SHIPPED (PR #5)
Shipped via the Kalshi `event.product_metadata.competition` join — see Recently Shipped above.

### MODEL-2 Uncalibrated Elo K-Factor / Home Advantage [P2] — NCAAW ✅ 2026-07-11 · NHL ✅ 2026-07-18 · NCAAF ✅ 2026-08-07
**File:** `evmax/agents/models/elo_agent.py`
`K_FACTORS` and `HOME_ADVANTAGE_ELO` now carry calibrated entries for every wired game sector — no sector silently uses the NBA-ish fallbacks any more. Latest: **NCAAF**, which was taking `K_FACTORS.get("ncaaf", 20.0)=20` and `HOME_ADVANTAGE_ELO.get("ncaaf", 0.0)=`**`0`** (a ZERO college-football home edge) at both the predict and update sites, while carrying ensemble weight 0.25. Swept via `scripts/backtest_ncaaf_elo.py` (cold-start walk-forward, warmup 2021-22 → rank 2023 → confirm 2024, held out on 2025, n=958): winner **K=40 / home_adv=60** → holdout Brier **0.1836** / acc 72.2% vs the K=20/home_adv=0 fallback at 0.2057 / 68.5% (**Δ+0.0221**, the direct cost of the bug). K=40 is the in-grid winner; grid deliberately not extended past 40 (same cold-start-overstates-K caveat NCAAW gave). Production-faithful "always apply home_adv" beat a neutral-aware variant on holdout (0.1836 vs 0.1845), so no neutral-site change to the agent.
- ~~Add calibrated values for NCAAW~~ ✅ K=35 / home_adv=80 (higher K than NCAAB, as predicted)
- ~~Add NHL: K=16, home_adv=0.04 (puck-line markets exist)~~ ✅ 2026-07-18 landed **K=6 / home_adv=48**, not the guessed K=16/0.04 — swept via `scripts/backtest_nhl_elo.py` over {6,10,14,20,25} × home_adv {0,20,32,48,60}, ranked on 2023-24 and confirmed on 2024-25 (`elo_agent.py:67,133`)
- ~~Add calibrated `ncaaf` K + home_adv~~ ✅ 2026-08-07 **K=40 / home_adv=60** (`scripts/backtest_ncaaf_elo.py`)
- OPEN (blend, not calibration): `SECTOR_WEIGHT_OVERRIDES["nhl"]` still holds `elo: 0.0`. The calibration precondition is met, so raising it is now a walk-forward blend decision.
- **[OPEN · P2] Warm-seed ncaaf elo** — the K=40/home_adv=60 constants are correct but ncaaf elo starts the 2026 season from an EMPTY `elo_state.json` (no key). As of #169 ncaaf IS fed through the resolve-time model-update hook (canonical `ESPN_MODEL_UPDATE_SECTORS` in `model_updater.py`), so elo now warms automatically as games resolve and enters the blend once each team clears the confidence gate (~5 games). A walk-forward seed (the sweep already builds the state) would warm-start it from week 0 instead of ~week 5 — the only remaining activation step. Form is dark for ncaaf until the same hook feeds it likewise.

### MODEL-3 Form Model Draw Normalization Edge Case [P2]
**File:** `evmax/agents/models/form_agent.py:168-176`
When `prob_a` is clamped to 0.95 (dominant team), `prob_a + prob_b ≠ 1.0` before draw scaling, so the final sum `prob_a + prob_b + prob_draw ≠ 1.0`. The ensemble renormalizes so impact is small, but it's architecturally inconsistent with Elo's explicit renormalization.
- After the `min/max` clamp of `prob_a`, set `prob_b = 1.0 - prob_a` before draw scaling

### MODEL-4 Poisson Attack/Defense Updates Use Simple Running Mean [P3]
**File:** `evmax/agents/models/poisson_agent.py`
The attack/defense rating update is a simple running mean. A team's 2020 stats eventually dilute current-season form as game count grows. Should use exponentially weighted update:
```python
alpha = 0.1  # decay factor
team["attack"] = (1 - alpha) * team["attack"] + alpha * actual_goals
```

### ~~MODEL-5 Tennis Model Weight~~ ✅ SHIPPED (PR #5)
Trimmed 0.45 → 0.35. See Recently Shipped above.

### ~~MODEL-10 Tennis Advanced Stats Agent~~ ✅ SHIPPED
**File:** `evmax/agents/models/tennis_advanced_stats_agent.py`
Logistic regression on four advanced stat differentials: BP conversion rate, return points won %, unforced error rate, and winners-to-UE ratio. Trained on 2023-2024 Sackmann ATP/WTA CSVs + Match Charting Project data. Falls back to RPW-only reduced model when MCP coverage unavailable. Weight 0.25, registered in ensemble. Tennis serve/return weight reduced 0.40 → 0.15 (was destructive at higher weight). Sharp weight bumped tennis 0.92 → 0.95.

### MODEL-6 Court-Adjustment Factor for Indoor Hard (Orthogonal to Surface) [P2]
**File:** `evmax/agents/models/tennis_model_agent.py`

**Context — why this exists as a separate item.** The original code treated `indoor` as a peer of `hard`/`clay`/`grass` in the surface dimension. This is a category error: in real tennis (and in every public dataset — Sackmann `tennis_atp`/`tennis_wta`, tennis-data.co.uk), surface and court are **orthogonal axes**:

- **Surface** ∈ {hard, clay, grass} (+ carpet, phased out 2009)
- **Court** ∈ {indoor, outdoor}
- Clay = essentially always outdoor on tour. Grass = always outdoor. Hard = both.

Notable indoor-hard events: Paris Masters (Bercy), Nitto ATP Finals (Turin), WTA Finals, Rotterdam, Basel, Vienna, Stockholm, Antwerp, Metz, Sofia.

**Why it matters for predictions.** Indoor hard plays meaningfully differently from outdoor hard — faster courts, lower bounce, no wind, no sun, controlled temperature. This is a real player-level skill differential, not noise: Medvedev-archetype big servers historically overperform their outdoor-hard baseline on indoor hard, and ATP Finals results systematically deviate from hardcourt-season form. Modeling indoor as a court-adjustment factor (not a surface) would capture this correctly.

**What MODEL-1 set up as a seam.** The surface resolver shipped in MODEL-1 returns both `surface ∈ {hard, clay, grass}` AND a separate `is_indoor: bool`. The boolean is currently unused by `predict_pair()` — it's a read-path hook waiting for MODEL-6 to consume it.

**Design options for MODEL-6:**
1. **Full `(surface, court)` buckets** — 6 separate Elo buckets: `hard_outdoor`, `hard_indoor`, `clay_outdoor` (empty), `grass_outdoor` (empty), plus the existing aggregates. Cleanest, but fragments rating data and bumps `MIN_SURFACE_GAMES` gate failures.
2. **Court-adjustment factor** — single bonus/penalty applied on top of surface Elo when `is_indoor=True`. One scalar per player, or a global constant to start. Simpler, preserves existing rating density, only perturbs the prediction at blend time.

Option 2 is the likely starting point. A global indoor bonus calibrated from Brier improvement would be the minimum viable version; per-player indoor modifiers can come later once data accumulates.

**Stale state data to handle during migration.** As of the MODEL-1 ship date, `data/models/tennis_surface_state.json` contains ~51 ratings in the legacy `indoor` bucket (~288 total game-updates) written via manual `evmax agents update --surface indoor` / `evmax agents seed-tennis surface --surface indoor` calls before those CLI options were removed. **None** of the automated seed pipelines (`scripts/seed_tennis_models.py`, `scripts/seed_espn.py`) ever wrote to this bucket — Sackmann classifies indoor hard as just `Hard`. So the 51 entries are:
- Sparse (avg ~5.6 games/player, below `MIN_SURFACE_GAMES=8`, never trips the confidence gate)
- Inert after MODEL-1 (resolver no longer returns `indoor`, no reader consumes it)
- Frozen (no writer after CLI cleanup, bucket cannot grow)

**When MODEL-6 lands, decide what to do with them:**
- **(a) Fold into `hard`** as bootstrap data for the indoor adjustment factor. Pro: preserves signal. Con: muddies hard ratings with players' indoor-specific skill.
- **(b) Discard** via a one-liner migration (`del state["ratings"]["indoor"]` + corresponding `game_counts`). Pro: clean slate. Con: loses manual-curation work, but the signal is so thin it barely matters.
- **(c) Migrate into a new `hard_indoor` bucket** if going with design option 1.

This is a deliberate deferral — not a cleanup chore. Pick the strategy that fits whichever of the two design options MODEL-6 adopts.

**Blocker:** needs calibration against live outcomes. The Brier-improvement signal from adding an indoor modifier isn't measurable offline — it requires accumulated predictions on both indoor and outdoor hard events. Leave to when the `predictions.db` has enough indoor-event coverage (probably post-Paris Masters / ATP Finals season). Document this as waiting on brother's live accumulation when picking it up.

**Secondary cleanup tracked here:** the frozen 51-entry `indoor` bucket in state is cosmetic debt (~2KB disk, no correctness or perf impact) as long as MODEL-6 is pending. Do not ship a standalone cleanup script — fold it into MODEL-6's migration instead.

### MODEL-7 Non-QB NFL Prop Features (Usage, Target Share, Vegas Totals) [P3]
**Files:** `evmax/clients/nfl_props_cache.py`, `evmax/backtest/sources/nfl_props.py`, `scripts/fetch_nfl_features.py`

**Context.** The Stage 4 backtest (see Recently Shipped) showed non-QB NFL stats (`rushing_yards`, `receiving_yards`, `receptions`, `anytime_td`) have Brier ≥ 0.21 and in some cases perform *worse* than the naive "always predict the base rate" prior. Receiving yards alone is 35% of the dataset and has Brier 0.254 vs prior 0.234. The current model for those stats uses only the player's L8-game rolling mean + Gaussian tail + streak adjustment + team-level opponent allowed-per-game. That feature set is insufficient for stats where per-game variance is dominated by usage shifts and game script, not mean skill.

**Specific features to add (all already in `weekly_stats.parquet` — no new data pulls):**

1. **Usage decomposition.** Model `yards = usage × efficiency` as two separate rolling terms instead of one combined term.
   - RB rushing: `rushing_yards = carries × YPC`. Predict `carries` and `YPC` separately — each is less noisy than their product. A RB whose YPC is stable at 4.2 but whose carries drop 22 → 11/game will have current-model yards drop 92 → 46 silently; a decomposed model catches the usage shift immediately.
   - WR receiving: `receiving_yards = targets × catch_rate × ypr`. Similar decomposition.
   - Columns already in parquet: `carries`, `targets`, `receptions`, `target_share`, `air_yards_share`, `wopr`.

2. **Position-aware opponent adjustment.** Currently `_build_defense_table()` computes a single "pass yards allowed per game" per defense-week. Replace with buckets by target player position/role (WR1 vs slot vs TE vs RB1). A defense that shuts down #1 WRs while getting shredded by slots is currently a lossy single number — tighten to role-matched rolling averages.

3. **Vegas totals + spreads.** Game total is the single best predictor of scoring environment (drives weather, pace, script expectations). Check `schedules.parquet` — it has 46 columns and I only verified 7; look for `spread_line` / `total_line`. If absent, nflverse has a separate `lines` release tag (same direct-fetch pattern as Stage 2). Likely modest impact on QBs, larger impact on rec/rush yards.

4. **Snap count.** nflverse has a separate `snap_counts` dataset (weekly). Out-of-parquet; separate fetch. Would let the model distinguish "played 70% snaps last week" from "played 35% snaps last week" — which is the dominant injury-recovery signal.

**Expected impact.** Pooled Brier on non-QB stats probably drops 0.24–0.25 → 0.21–0.22. Would likely cross the 0.22 pooled gate from Stage 4. **Still insufficient for ROI viability** — would likely still lose to the market's 0.135 Brier on closing prices, because the market has access to practice reports / inactives / weather forecasts that no parquet captures.

**Blocker / priority.** P3 because non-QB prop modeling is downstream of MODEL-8 (calibration fix). Fixing the features without fixing the tail overconfidence won't make the model bet-viable.

### ~~MODEL-8 NFL Prop Tail Calibration~~ ✅ SHIPPED (partial — steps 1-3 of 4)
Doubled MIN_STD floors (passing 35→70, rushing 15→30, receiving 12→24, receptions 1→2), removed the ±0.04 streak adjustment, and capped the empirical-blend disagreement at ±15pp. QB-only Brier improved from 0.1854 → 0.1789. Lower-tail calibration is now nearly perfect (0–10% predicted → 9.1% actual). Upper-tail still misses by ~12pp but the NO-side strategy doesn't depend on upper-tail accuracy. Step 4 (Platt/isotonic post-hoc) was not needed for the gate and is deferred. All 24 NFL prop tests still pass. See Recently Shipped block above for full gate numbers.

### MODEL-8-ORIGINAL — historical context, do not re-run
**File:** `evmax/clients/nfl_props_cache.py`
**Files:** `evmax/clients/nfl_props_cache.py`

**Context.** The QB-only Stage 4 backtest (passing_yards + passing_tds, 3,642 settled markets) produced:
- Pooled Brier 0.1854 (passes 0.22 gate)
- Accuracy 74.1%
- **But ROI at ev≥2%, vol≥1000: −86.7% with 5.1% win rate** — the model systematically loses on the specific markets where it claims the biggest edge vs the market.

The calibration chart is the diagnostic:
```
predicted  actual   n
0–10%     → 12.3%  1,255   (model too confident "under")
20–30%    → 33.1%    296
30–40%    → 35.0%    254   (middle bins calibrate ok)
50–60%    → 56.3%    229
70–80%    → 61.0%    228   (upper tail: model too confident "over")
90–100%   → 78.8%    349   (−18pp miss)
```

Middle bins are well-calibrated. Tails miss by 10–18pp. The ROI filter picks exactly those tail-disagreement markets where the model is wrong, producing the win-rate collapse.

**Root causes (ordered by suspected magnitude):**

1. **Gaussian std floor is too tight.** `MIN_STD_BY_STAT` uses 35 yards for passing, 15 for rushing, 12 for receiving. Real per-game variance is wider because of game-script effects, weather, opponent pressure. Raising the floors 1.5–2× would flatten the tails.

2. **Streak adjustment amplifies overconfidence.** The `+0.04` / `−0.04` nudge for 3-of-3 hot/cold streaks compounds on top of an already-confident Gaussian. For a model predicting 0.92, adding 0.04 → 0.96 moves it into the miscalibrated 90–100% bucket.

3. **Empirical hit-rate blend (60/40).** When the player has beaten the line in 6/8 recent games, `weighted_hit_rate` approaches 0.75 and the blend amplifies the Gaussian's prediction.

4. **Margin adjustment.** `avg_margin × 0.01` can add up to ±8% — another extreme-pushing term.

**Likely fix sequence (do in order, re-measure calibration after each):**

1. **Raise MIN_STD floors by 2×** (passing: 70, rushing: 30, receiving: 24). Re-run backtest. Expected: tails shrink toward reality, middle bins unchanged. Brier may drift up slightly — that's OK, we want a less-confident model.

2. **Kill the streak adjustment entirely.** Streaks in 3-game windows are mostly noise at the NFL's weekly cadence. Re-measure.

3. **Cap the empirical blend at |model − empirical| ≤ 0.15.** If the hit rate disagrees with the Gaussian by more than 15pp, trust the Gaussian (the hit rate is being driven by 1–2 outlier games).

4. **Optional: switch to a Platt-scaling or isotonic post-hoc calibration layer.** Train on (pred, actual) pairs from 2024 season data (out-of-sample from the 2025-26 backtest), apply as a post-processing step. This is the textbook fix for systematic miscalibration.

**Gate to re-run:** target calibration bins of ≤5pp max miss across all buckets AND ROI > 0% at ev≥3%, vol≥1000. Only then proceed to Stage 5 (live pipeline integration).

**Blocker:** none — purely a model iteration using data already on disk. ~4 hours focused work.

### MODEL-9 NFL Prop Shadow-Mode Validation [P1 — blocks Stage 5 live betting]
**Files:** `evmax/agents/coordinator.py`, `evmax/agents/cleanup/prop_resolver.py`, `evmax/clients/kalshi.py`, `evmax/clients/nfl_props_cache.py`

**Status:** infrastructure shipped in `feat/model9-nfl-prop-shadow` (Apr 2026). Validation itself waits on the 2026 NFL regular season.

**Infrastructure shipped:**
- ✅ `kalshi.py` series-name typo fix — `nfl_props` sector now points at the six real Kalshi tickers (`KXNFLPASSYDS`, `KXNFLRSHYDS`, `KXNFLRECYDS`, `KXNFLANYTD`, `KXNFLPASSTDS`, `KXNFLREC`). Zero markets flowed in before this fix.
- ✅ `nfl_props_cache.py` disk-cache layer — wraps the pure compute function from PR #6 Stage 4 with parquet-backed feature lookup. Reuses `data/backtest/nfl_props/{weekly_stats,rosters,schedules}.parquet` directly, point-in-time history per (player, season, week), schedule-based opponent resolution with defense adjustment, lazy module-level memoization. Re-run `scripts/fetch_nfl_features.py` weekly during NFL season to refresh.
- ✅ `coordinator._fetch_props()` NFL branch — mirrors the NBA path, calls `compute_nfl_prop_prob_cached` and emits `SharpOdds`.
- ✅ `prop_resolver` NFL support — ESPN boxscore extraction now knows `passing_yards`, `passing_tds`, `rushing_yards`, `rushing_tds`, `receiving_yards`, `receiving_tds`, `receptions`. `fetch_player_stats` refactored to MERGE stats across stat_groups so a QB's passing + rushing rows land on the same player dict (pre-refactor the second group overwrote the first). `anytime_td` derived post-merge as `rushing_tds + receiving_tds`.
- ✅ Shadow mode config was already in place via ARCH-11 — `data/categories.yaml` ships `nfl_props: mode: shadow`.
- ✅ Tests: 14 new NFL cache-layer tests (parquet load, name normalization, point-in-time history, schedule opponent, coordinator `_fetch_props` integration) + 7 new resolver NFL tests (per-stat extraction, merge-across-groups, anytime_td derivation). Total suite 902 → 923.

**Still pending — validation itself (needs 2026 NFL regular season data):**
1. Capture ≥ 500 shadow bets across ≥ 3 distinct NFL weeks.
2. `evmax cleanup shadow metrics --days N` reads the shadow rows and computes ROI at ev≥3%, vol≥1000.
3. Promote/reject criteria:
   - Shadow ROI NO-only ≥ 65% (backtest was 79%, allow 15pp degradation for leakage + live-price differences).
   - Shadow Brier within 10% of backtest Brier (0.197 max).
   - Calibration tail miss ≤ 15pp at 90–100% bin.
4. If passes: promote with `evmax cleanup shadow promote nfl_props`, then do the NO-side EVGap refactor (see "Deferred" below) as a follow-up PR before Stage 5 live Kelly.
5. If fails: close PROPS-1 with "backtest showed no real edge, ROI was retrospective leakage."

**Deferred out of the infra PR (scope control):**
- **NO-side EVGap emission.** The backtest showed the edge lives on the NO side (YES was catastrophic at −88% ROI). `log_prop_observations` already logs ALL prop rows regardless of EV sign, so the post-hoc NO-side ROI analysis can be done against `captured_yes_price + blended_true_prob + outcome` via `evmax cleanup shadow metrics` without touching EVGap. The full EVGap side refactor (add `side: "yes" | "no"` field OR emit two gaps per market, and propagate through Kelly + exposure guard) lands in a separate PR gated on shadow validation passing — no point designing a refactor for a signal that might be leakage.
- **Opponent-adjustment fallback for schedule gaps.** Current cache uses `opp_adj = 1.0` if the schedule row is missing for the target game_date. During live NFL weeks with fresh `schedules.parquet` this should always resolve; if it becomes a gap we can fall back to season-level team-vs-position defense.
- **MODEL-7** (non-QB features — usage rate, target share, Vegas totals) is still the downstream improvement once MODEL-9 validates QB-only edge.

**Blocker:** NFL regular season starts Sept 2026. Shadow validation requires at least 3-4 weeks of live NFL to be meaningful.

### MODEL-11 WNBA Shadow Validation + Promotion to Live [P1 — blocks 2026 WNBA live betting]
**Files:** `data/categories.yaml` (wnba block), `evmax/agents/cleanup/metrics.py`, `evmax/cli/commands/cleanup.py`

**Context.** The 2026 WNBA ensemble lands in shadow mode (see "Recently Shipped" for the full walk-forward numbers — Brier 0.2056 / Acc 68.8%, within 0.0024 Brier of NBA). WNBA stays in shadow until live Kalshi markets confirm the walk-forward result transfers to real betting. Same MODEL-9 shadow-validation pattern as NFL props.

**Validation steps (once the 2026 WNBA season opens May 16):**
1. Accumulate ≥ 200 shadow bets across ≥ 3 distinct weeks of regular season.
2. `evmax cleanup shadow metrics --days N --category wnba` computes Brier + ROI at ev≥2%.
3. Promote / reject criteria:
   - Shadow ensemble Brier ≤ 0.215 (allow 0.01 degradation from 0.2056 walk-forward for live-price / market-friction differences).
   - Shadow ensemble accuracy ≥ 64%.
   - Calibration tail miss ≤ 10pp at 80–90% and 10–20% buckets.
   - ROI at ev≥2% is positive (doesn't need to be large; just sign-correct).
4. If passes: `evmax cleanup shadow promote wnba` — flips YAML mode to `live` and enables Kelly sizing against bankroll.
5. If fails: leave in shadow, diagnose via per-model Brier breakdown and per-bucket calibration, iterate on weights / tunable constants.

**Weekly data refresh during the season:**
- Re-run `python scripts/seed_wnba_efficiency.py --year 2026` weekly to roll 2026 games into the efficiency/possession_sim inputs. Current seed is 2025 priors only.
- The `form_agent` and `elo_agent` update incrementally via `evmax agents update` + `evmax cleanup resolve`, no manual refresh needed.

**Blocker:** season starts May 16 2026. Validation is meaningless before then.

### MODEL-12 Port `shot_quality` and `matchup` Agents to WNBA [P3]
**Files:** new `evmax/agents/models/wnba_shot_quality_agent.py`, new `evmax/agents/models/wnba_matchup_agent.py`

**Context.** The WNBA ensemble currently runs only 2 of NBA's 4 advanced agents (efficiency + possession_sim). On the NBA walk-forward, `shot_quality` adds Brier 0.2319 and `matchup` adds 0.2355 — each small individually but diverse signal that tightens the ensemble. Porting them would close the last ~0.005 Brier gap to NBA parity.

**Why P3 (not P2):**
- Data access is harder. NBA's `shot_quality_agent` reads `stats.nba.com/LeagueDashTeamShotLocations` for per-zone FGA + FG%. `stats.wnba.com` has an equivalent endpoint but the `nba_api` package doesn't expose it — needs a custom httpx wrapper with the right headers. Same for `matchup_agent` (paint scoring + transition defense + turnover battle).
- Marginal gain is small. Both NBA agents have Brier 0.23+, close to baseline. The 0.005 Brier improvement they unlock on the ensemble only matters once efficiency + possession_sim are already saturated with data, which won't happen until mid-2026 season.
- Post-MODEL-11 ordering. No point building these until WNBA shadow validation clears, because shadow validation may surface different tuning needs that change what "diverse signal" means for the ensemble.

**Design if/when done:** mirror the `wnba_efficiency_agent.py` + `wnba_possession_sim_agent.py` pattern — new standalone files, own state JSON, WNBA-tuned constants, zero shared file with NBA siblings.

### MODEL-13 WNBA `player_impact_agent` [P2 — larger expected impact than MODEL-12]
**Files:** new `evmax/agents/intelligence/wnba_player_impact_agent.py`

**Context.** WNBA star dependency is markedly higher than NBA's because rosters are 12 (vs 15) and season is 40 games (vs 82). A single star out — Caitlin Clark, A'ja Wilson, Napheesa Collier, Breanna Stewart, Alyssa Thomas — is worth roughly an **8-10 point swing** in expected margin, vs NBA's more modest ~5-6 points for a comparable star. The current injury-probability adjustment is sector-flat and caps at -12% per team; WNBA calls for a sector-specific cap closer to -18% AND a per-player impact scaled by on/off net rating (same pattern as NBA's `player_impact_agent`).

**Data source — same story as MODEL-12:** need `stats.wnba.com/LeagueDashPlayerStats?MeasureType=Advanced` which isn't exposed by `nba_api`. Custom httpx wrapper. Alternative: derive the per-player MIN × NET_RATING from aggregated ESPN box-score pulls (same source as efficiency seed), accepting slightly less accurate single-player impact estimates.

**Why P2 (higher than MODEL-12):** the marginal gain is real. Porting `shot_quality` / `matchup` at P3 closes ~0.005 Brier; porting `player_impact` would realistically close 0.01-0.015 Brier during WNBA weeks with active injuries to stars (which is most weeks). Also: the current `InjuryReportAgent` has WNBA wired via ESPN already but the `KNOWN_STARS` set has only ~10 WNBA players and is stale — expanding it is a quick win independent of the full player_impact port.

**Scope for v1:**
1. Refresh `KNOWN_STARS` in `injury_agent.py` with 2026 WNBA top-25 (Bueckers, Clark, Wilson, Collier, Stewart, Ionescu, Thomas, Copper, Sabally, Reese, Howard, Gray, Plum, Ogunbowale, Loyd, Mitchell, Boston, Griner, Hamby, Ogwumike, etc.).
2. Per-sector `MAX_ADJ` dict: NBA 0.20, WNBA 0.28, others 0.20.
3. (Stretch) new agent that mirrors NBA's `player_impact_agent` using ESPN-derived per-player impact minutes.

**Blocker:** needs a data source decision (stats.wnba.com direct vs ESPN aggregation) before implementation begins.

### MODEL-14 NBA props post-calibration validation via shadow [P1]
**Files:** `data/categories.yaml` (flip `nba_props.mode: live → shadow`), `evmax/agents/cleanup/metrics.py`

**Context.** Path A (commit fd2e650) re-tuned the NBA props model end-to-end — the manual shrinkage was zeroed and an isotonic calibration layer added. The Brier reduction is modest in aggregate (0.1969 → 0.1928) but the per-bucket calibration is fundamentally fixed (50–60% bucket from +12.9pp miss to +0.1pp). The retroactive isotonic-only test on real Apr 9-23 prop_observations data confirmed real-world Brier 0.1709 → 0.1683 (small but in the right direction).

**Why shadow now (not just "watch live").** We have 1,107 pre-calibration resolved rows in `prop_observations` from Apr 9-23 (model Brier 0.1709 vs Kalshi 0.1991 — model already beats market by 14% on this sample). Switching to shadow now gives us a clean before/after measurement on the same population (NBA playoffs), with zero bankroll exposure during the validation period. The data side is identical between live and shadow — `prop_observations` keeps logging with `mode='shadow'` tag. Only difference is Kelly stake doesn't touch the bankroll.

**Validation plan (2-3 weeks):**
1. Flip `nba_props.mode` to `shadow` in `data/categories.yaml` (or use `--shadow nba_props` per-scan).
2. Run scans daily as usual. New rows accumulate in `prop_observations` tagged `mode='shadow'`.
3. After ≥ 50 resolved post-calibration bets:
   ```
   evmax cleanup shadow show --days 14 --category nba_props
   evmax cleanup shadow metrics --days 14 --category nba_props
   ```
4. Pass criteria (vs the pre-calibration baseline of Brier 0.1709 / model already beats Kalshi by 0.028):
   - Post-calibration Brier on shadow rows ≤ 0.175 (no regression)
   - Calibration buckets show 60–70% bucket within ±5pp of actual (was +17.8pp pre-calibration)
   - Model still beats Kalshi-price predictor by ≥ 0.02 Brier
   - +EV at ev≥3% remains sign-correct over a 50-bet sample
5. Promote: `evmax cleanup shadow promote nba_props` (flips YAML back to live).

**Status note**: nba_props mode is currently `live` in `data/categories.yaml`. To start MODEL-14 validation, flip to `shadow`. Action item lives in "What needs to occur now" below.

**Blocker:** needs ~50 resolved post-calibration nba_props bets (achievable in 2 weeks of NBA play).

### MODEL-15 NBA props v2 — career-mean prior + shot-type variance [P3]
**Files:** new model code in `evmax/clients/`, new training script in `scripts/`

**Context.** Path B was tested and closed as a partial negative — days-rest signal didn't move Brier (load-management masking + per-36 absorption). But two other v2 ideas remain unbuilt:

1. **Career-mean prior** — blend in a player's prior-4-season per-stat mean to stabilize early-season L15 noise. Helps rookies (no career data → falls back to L15) and age-30+ vets (career mean is more stable than recent L15). Blocked when first attempted because nba_api rate-limited the bulk season pulls. Recoverable with chunked retry logic + exponential backoff. The fetch script (`scripts/fetch_nba_game_logs.py`) already supports 2020-21 through 2024-25.
2. **Shot-type variance from shufinskiy shotdetail** — categorize shots into types (catch-and-shoot 3, pullup 3, paint, mid-range, FT trips), compute per-type rates per player, project total points more precisely than a simple Normal-CDF. Would only affect PTS / FG3M (2 of 9 stats) but those are the most-bet stats. Data is already on disk (`data/historical_nba/shotdetail_*.csv`); the work is the modeling itself.

**Why P3 and not P2:** Path A delivered the actual fix that warranted the live promotion. v2 features are incremental on top. Pursue when (a) there's specific evidence Path A is leaving systematic edges on the table, OR (b) we have a quiet stretch and want to push Brier under 0.18.

**Blocker for career-mean:** needs nba_api rate-limit handling. ~2 hours of retry-loop engineering.
**Blocker for shot-type variance:** none — data is local. ~3-4 days of focused modelling.

### MODEL-16 Refit NBA props isotonic with opp_adj backfilled [P3]
**File:** `scripts/calibrate_nba_props.py`, new team-stats backfill script

**Context.** The Path A calibration was fitted on a row set that did NOT include opp_adj because we don't have backfilled team defensive stats per game date. In production, opp_adj is applied between volatility shrinkage and isotonic — meaning isotonic operates on a slightly different distribution than what was fitted on. Opp_adj is bounded at ±15% so the drift is bounded, but the calibration would be tighter if we re-fit with opp_adj included.

**Plan:** Pull point-in-time team defensive stats from `nba_api.LeagueDashTeamStats` for each game date in the 2024-25 holdout, compute opp_adj per (player, game), include it in the intermediates parquet, re-grid-search constants and re-fit isotonic. Replace `data/models/nba_props_calibration.json`. Same harness, more accurate fit.

**Expected gain:** ~0.001-0.003 additional Brier improvement and slightly better per-bucket calibration. Modest.

**Blocker:** ~half-day of work plus the nba_api rate-limit dance.

---

## Section 4 — Test Coverage Gaps

### TEST-3 PinnacleGuestClient Has Zero Tests [P2]
**File:** `evmax/clients/esports_pinnacle.py`
This is the only live sharp odds provider and has no tests. At minimum, test the response parsing and devigging with a fixture.

### TEST-4 No Integration Test for Full Coordinator Cycle [P2]
There's no test that runs a full `coordinator.run_cycle()` against fixture data for even one sector. This would catch wiring bugs (like the tennis model not being called) before they hit production.

### TEST-5 Live Win Probability Model Untested [P3]
**File:** `evmax/models_ml/live_win_prob.py`
No tests for the live in-game model.

### TEST-6 Prop Probability Pipeline — Partially Covered [P2]
`nba_props_cache.py`, `prop_matcher.py`, and `prop_resolver.py` pure paths are now covered via `tests/test_prop_pipeline.py` (44 tests, PR #2). Still missing: `nba_stats.py` (network-heavy, needs httpx mocking) and the side-effectful paths of the three covered modules (`fetch_player_stats` body, `resolve_prop_observations` DB loop, `refresh_props_cache`).

---

## Section 5 — Architecture / Cleanup

> Skipped for now (your brother's repo — leaving major arch decisions alone).
> Listed here so they don't get forgotten if/when he wants to tackle them.

### ARCH-1 Dead Code: `pipeline/runner.py` and `models_ml/sharp_only.py` [P2]
`pipeline/runner.py` is a Phase 1 legacy module. It imports `SharpBooksModel` from `models_ml/sharp_only.py` and the old `PinnacleClient`. Neither is called by any CLI command.
- Delete `evmax/pipeline/runner.py` (legacy, superseded by coordinator)
- Delete `evmax/models_ml/sharp_only.py` (legacy placeholder model)
- Confirm `evmax/clients/pinnacle.py` (TheOddsAPI) is also unused in live path, and if so, archive or delete it

### ARCH-2 Dual Database Architecture Creates Schema Drift [P2]
Two completely separate storage systems exist:
1. `evmax/db.py` + `evmax/models/` — SQLAlchemy async ORM (`evmax.db`)
2. `evmax/agents/cleanup/db.py` — raw `sqlite3` (`data/predictions.db`)

The ORM models (`EVBetORM`, `SharpOddsORM`, `PredictionMarketORM`) are defined but never used in the live scan pipeline. `evmax.db` is only touched by `simulation/` and `scripts/init_db.py`. The ORM schema is missing all columns that `predictions.db` tracks (`model_sources`, `event_title`, `yes_team`, `blended_true_prob`, etc.).
- Either: migrate the live pipeline to use the ORM (large effort, high value)
- Or: clearly mark `evmax/db.py` + `evmax/models/` as "simulation only" with a header comment, and stop maintaining them as if they're the live schema

### ARCH-3 Sharp Odds Naming Is Confusing [P3]
`esports_pinnacle.py` contains `PinnacleGuestClient` which handles **all** sectors (not just esports). The file name implies esports-only, which confuses new readers. `pinnacle.py` contains `PinnacleClient` (TheOddsAPI) which is never used in live scans.
- Rename `esports_pinnacle.py` → `pinnacle_guest.py`
- Rename `pinnacle.py` → `pinnacle_theodds.py` (or delete if confirmed dead)
- Update all imports

### ARCH-4 NFL Props Fetched But Never Evaluated [P2]
**File:** `evmax/agents/coordinator.py` (~line 570)
NFL prop series (`KXNFLPAS`, `KXNFLREC`, etc.) are fetched from Kalshi but `_fetch_props()` returns `None` for non-NBA sectors. The Kalshi API calls are real, the data is discarded. Either:
- Implement NFL prop probability computation (similar to `nba_stats.py`)
- Or: remove NFL prop series from `KALSHI_PROP_SERIES` until the backend exists

### ARCH-5 Kelly `confidence_discount` Parameter is Dead [P3]
**File:** `evmax/ev/kelly.py:91-93`
`confidence_discount = 1.0` always. The parameter is accepted by `compute_kelly()` but unused.
- Either: implement confidence discount logic (tie to model confidence signal)
- Or: remove the parameter from the signature to prevent confusion

### ARCH-6 `EnsembleModelAgent.avg_conf` Mismatch [P3]
**File:** `evmax/agents/models/ensemble_agent.py:228`
`pred_list = list(model_preds.values())` contains all models, but the confidence averaging loop uses `model_contribs` (only models that passed the gate). The indexing mismatch produces incorrect `avg_conf` display values. This doesn't affect probability predictions, only the displayed confidence.
- Rewrite to compute `avg_conf` directly from the contributing predictions dict rather than by indexing into `pred_list`

### ~~ARCH-7 YES-Team Price Heuristic Fails on Even-Money Games~~ ✅ SHIPPED
**File:** `evmax/agents/odds/ev_gap_agent.py`
Replaced the asymmetric `< 0.05` / `> 0.10` thresholds in the `_resolve_yes_via_market_teams` price fallback with an explicit two-condition rule: the closer side must be within 4pp of the YES ask, AND the gap between the two distances must be ≥ 5pp. Near-coin-flip markets (where both distances are similar) now return `None` instead of being force-aligned to an arbitrary outcome. 5 regression tests in `TestEVGapAgent`.

### ARCH-8 Pinnacle Guest Maintenance Handling + Stale Cache Fallback [P2]
**Files:** `evmax/clients/esports_pinnacle.py`, `evmax/clients/base.py`, `evmax/agents/odds/sharp_agent.py`, `evmax/models/odds.py`

`PinnacleGuestClient` is the single sharp-odds source for every sector (not just tennis — confirmed via `SharpOddsAgent` import and `SECTOR_SPORT_LEAGUES` map). The endpoint is unauthenticated and undocumented, and Pinnacle runs scheduled maintenance windows on it that take **the entire EV pipeline offline across all sectors simultaneously**. Observed 2026-04-13: `guest.api.arcadia.pinnacle.com` returned `503 MAINTENANCE` on every sport ID tested (4, 6, 29, 2, 33) with response body:

```json
{"type": "about:blank", "title": "MAINTENANCE", "detail": "API is currently undergoing maintenance, try again later", "status": 503}
```

The current retry layer (`base.py::_is_retryable`) only runs **2 attempts** with exponential backoff capped at ~10s total — useless against maintenance windows that last minutes to tens of minutes.

**Fix scope:**
1. **Parse the 503 body** — when the response body contains `"MAINTENANCE"`, emit a distinct `pinnacle.guest.maintenance` log event and short-circuit retries instead of burning the retry budget.
2. **Long-TTL last-known-good cache** — populate on every successful fetch, separate from the dev-mode `cache_ttl_secs` cache. TTL = 1-2 hours. On retry exhaustion or maintenance detection, fall through to the cache.
3. **`SharpOdds.is_stale: bool`** — new Pydantic field, default `False`, set `True` when served from fallback cache. Additive optional field (same zero-migration pattern as `PredictionMarket.competition` from MODEL-1). Exclude stale bets from live Kelly sizing by default; include in analysis/reporting.
4. **Bump retry count for genuine transients** — `max_retries=3`, `retry_max_wait=15s` for non-MAINTENANCE 5xx only. Handles rare connection blips without over-retrying maintenance windows.
5. **5xx counter metric** — increment per 5xx response split into `{transient, maintenance}`. Emit as a structured log field and aggregate via `evmax cleanup metrics` so outage frequency becomes visible historically.
6. **Optional pre-scan health check** — `--skip-if-pinnacle-down` CLI flag that probes the endpoint once before running a full cycle. Avoids wasting a scan during known outages.

**Blocker:** none. Uses existing infrastructure. Can ship as a standalone PR.

### ARCH-11 Category Registry + Live / Shadow / Disabled Mode Config [P1]
**Files:** new `evmax/categories.py` (registry module), new `data/categories.yaml` (source of truth), new `evmax/modes.py` (mode lookup API), `evmax/settings.py`, `evmax/agents/coordinator.py`, `evmax/agents/cleanup/db.py`, `evmax/agents/cleanup/logger.py`, `evmax/cli/commands/agents.py`, `evmax/cli/commands/cleanup.py`, new `evmax/cli/commands/categories.py`, `CLAUDE.md`

**Context.** Two related needs in one PR:

1. **A single source of truth for the betting category catalog.** Today the answer to "what categories can we bet on, with which models, via which resolver, and in what state?" is scattered across CLAUDE.md's Key Sectors list, the Modeling table, the Data Sources for Outcome Resolution table, `SECTOR_SERIES_MAP` in `evmax/clients/kalshi.py`, and ad-hoc TODO.md entries. There is no single file a reader can open to see the full product state — and no machine-readable version for the scanner to consult at runtime. This drift is already starting to cost time during reviews (multiple cross-references to answer simple "is NHL modeled?" questions).

2. **Per-category live / shadow / disabled mode.** The user wants a single place to declare, per category, whether the scanner should:
   - `live` — compute edges, produce EVGaps, persist with `mode='live'`, size Kelly against bankroll (current default for every sector)
   - `shadow` — compute edges, log predictions + pre-game prices with `mode='shadow'`, do NOT touch bankroll
   - `disabled` — skip the category entirely during scan (saves API calls for sectors we're not trading)

The immediate use case is MODEL-9 (NFL props need shadow validation before live), but the same toggle is useful across the board: new sector rollouts, vacation bankroll freezes, post-outage recovery, or simply "I don't want to trade NHL anymore." The catalog registry makes the mode the *only* source of truth — no scattered config, no docs-drifting-from-code.

**Design:**

1. **Canonical category keys** — strings of the form `{sector}` for game markets, `{sector}_props` for player props. Examples: `nba`, `nba_props`, `nfl`, `nfl_props`, `tennis`, `mlb`, `nhl`. Matches the Kalshi `_props` suffix convention already in `SECTOR_SERIES_MAP`.

2. **Catalog registry — `data/categories.yaml` is the source of truth.** Schema per category:

   ```yaml
   nba:
     display_name: "NBA"
     market_types: [moneyline, spread, total]
     models: [elo, form, poisson, sharp]
     mode: live
     resolver: espn_scoreboard
     status: shipped
     notes: null

   nfl_props:
     display_name: "NFL player props"
     market_types: [over_under_passing_yards, over_under_rushing_yards, over_under_receiving_yards, over_under_passing_tds, over_under_receptions, anytime_td]
     models: [nfl_props_cache_v1_qb_only]
     mode: shadow                 # the ARCH-11 toggle lives on this line
     resolver: espn_boxscore
     status: "Stage 4 shipped; shadow validation pending MODEL-9"
     notes: "NO-side only per MODEL-8; YES-side systematically overpriced"

   tennis:
     display_name: "Tennis (ATP + WTA)"
     market_types: [moneyline]
     models: [tennis_surface_elo, tennis_serve_return, tennis_h2h, tennis_ranking_trend, sharp]
     mode: live
     resolver: kalshi_settlement
     status: shipped
     notes: "Court adjustment (indoor) deferred — MODEL-6"
   ```

   Every key in `evmax/clients/kalshi.py::SECTOR_SERIES_MAP` must have a corresponding entry — enforced by a startup validator and a test. New sectors that forget to register will fail loudly at import time, not drift silently. The file is editable by humans without code changes; a test verifies the YAML parses and every field is valid.

3. **`evmax/categories.py` registry module** loads `data/categories.yaml` into typed dataclasses (`CategorySpec` with `key`, `display_name`, `market_types`, `models`, `mode`, `resolver`, `status`, `notes`). Exposes:
   - `get_category(key) -> CategorySpec`
   - `all_categories() -> list[CategorySpec]`
   - `categories_in_mode("live" | "shadow" | "disabled") -> list[str]`
   - Validator: `validate_registry()` cross-checks against `SECTOR_SERIES_MAP` and the known model registry — failures are hard errors at import time.

4. **Settings + env overrides.** Env var `EVMAX_CATEGORY_MODES='{"nfl_props":"shadow","nhl":"disabled"}'` (JSON string) overrides the YAML's `mode` field for a single process. Useful for CI, local testing, and CLI wrapper scripts. Overrides compose per-command, not persisted.

5. **`evmax/modes.py`** wraps the mode read specifically (separate from the fuller registry API) with a clean surface: `get_mode(category) -> Literal["live","shadow","disabled"]`, `is_live(category) -> bool`, `is_shadow(category) -> bool`. All coordinator + CLI branches go through this module — no raw dict lookups scattered in code.

4. **CLI overrides.** Every scan-adjacent CLI command accepts `--live X,Y --shadow Z --disabled W` to override settings for a single run. Overrides compose per-command, not persisted.

5. **Persistence — mode column, NOT separate table.** Code trace (2026-04-14) confirmed there is exactly ONE writer to `ev_predictions`: `log_gaps()` in `evmax/agents/cleanup/logger.py`. All ~12 readers live in `evmax/cli/commands/` and `evmax/agents/cleanup/`. Add:
   - `mode TEXT NOT NULL DEFAULT 'live'` column on both `ev_predictions` and `prop_observations`
   - `captured_yes_price REAL` nullable column on both (pre-game YES ask at scan time — distinct from any later value the row might pick up)
   - `model_version TEXT` nullable column on both (short string like `nfl_props_v1_QB_only` so we can re-validate when the model changes and expire stale shadow data)
   
   Every existing SELECT against `ev_predictions` / `prop_observations` in the codebase must add `WHERE mode = 'live'` in the same migration PR. That's the regression risk — all ~12 call sites audited in one pass. Advantage of mode column over a second table: one schema, one migration, promotion from shadow → live is a single `UPDATE` (no row migration), and `evmax cleanup metrics` can compare shadow vs live side-by-side trivially.

6. **`log_gaps()` takes a `mode_resolver: Callable[[str], str]`.** It partitions gaps by category, sets the `mode` column accordingly in the `INSERT`. Exposure guard and Kelly sizing inside the scan CLI still run only for gaps whose `mode == "live"`; shadow gaps log predictions but don't touch the bankroll math. Disabled categories are filtered out before even reaching `log_gaps()`.

7. **New CLI `evmax cleanup shadow`** family:
   - `evmax cleanup shadow show --days 7` — recent shadow predictions with resolved outcomes
   - `evmax cleanup shadow metrics --weeks 4` — Brier + ROI over a window, with a gate table per category
   - `evmax cleanup shadow resolve --date YYYY-MM-DD` — calls `prop_resolver` against shadow rows for a given day
   - `evmax cleanup shadow promote <category>` — after manual review, flips the category from `shadow` → `live` in `data/categories.yaml` and prints a confirmation

8. **New CLI `evmax categories`** family — surfaces the registry to the user:
   - `evmax categories list` — prints the full catalog as a rich table (key, display name, market types, models, mode, resolver, status). One-command answer to "what can we bet on and what are we doing with each?"
   - `evmax categories show <key>` — detail view for one category, including the full model list, the resolver path, and any notes (e.g. the MODEL-8 NO-side-only caveat for nfl_props)
   - `evmax categories validate` — runs `validate_registry()` and prints the result. Used by CI and by the pre-commit hook.

9. **Documentation.** CLAUDE.md gets a new "Betting Categories" section that references `data/categories.yaml` as the source of truth and shows a compact rendered table. The existing scattered docs (Key Sectors list, Modeling table, Data Sources for Outcome Resolution table) either collapse into this new section OR cross-link to it so there's one canonical place to read the product state. "Category modes" explanation is folded into the same section — the three states, the env var / YAML file, the CLI overrides, the promotion workflow.

**Blocker:** none. Pure infrastructure — can ship before MODEL-9 runs, and in fact SHOULD ship first because MODEL-9 depends on the shadow table and the category-mode plumbing existing.

**Test coverage requirements (per CLAUDE.md testing policy):**
- `evmax/categories.py` — YAML parse, dataclass construction, every `SECTOR_SERIES_MAP` key present, every referenced model name exists, every `mode` value is one of the three legal states
- `evmax/modes.py` pure function tests — mode lookup, YAML fallback to default, env var override composition, CLI override composition
- `validate_registry()` — negative tests for missing fields, unknown models, unknown resolvers, illegal mode values
- Coordinator integration test: one sector in `live`, one in `shadow`, one in `disabled`, confirm each takes the right persistence path with the right `mode` column value
- `log_gaps()` mode-column test — verify partitioning and correct column writes
- `evmax categories list / show / validate` CLI output golden tests (typer Runner)
- CLI mode-override test (`--shadow X --live Y` applied for a single run only, not persisted)

**Revised effort estimate:** ~10–11 hours focused work (was ~7 without the catalog registry). The registry adds a YAML schema, a validator module, typed dataclasses, a new CLI command family, and the CLAUDE.md consolidation — but pays back the effort by giving both humans and code one canonical source of truth instead of a mode config + scattered docs.

### ARCH-12 Kalshi Series Drift Detection [P2]
**Files:** new `scripts/check_kalshi_series.py`, optional `.pre-commit-config.yaml` hook

**Context.** `evmax/clients/kalshi.py::SECTOR_SERIES_MAP` is a **static hardcoded dict** that was manually audited once against Kalshi's `/series?category=Sports` endpoint ("Verified against live Kalshi series API (2026-02-23)") and then frozen into source. Kalshi is not re-queried at runtime, and there is no scheduled refresh. Drift goes one direction, silently:

- **New Kalshi series are invisible.** If Kalshi launches a new sport, a new league, a renamed series, or (like real NFL player props during the 2026 season landing on `KXNFLPASSYDS`), `evmax` cannot bet on it until someone edits the dict. The scanner just returns an empty result set — which is what happened to NFL prop markets from day one until PR #6 audited the real series names.
- **Typos / stale entries survive indefinitely.** PR #6 surfaced exactly this failure mode: `SECTOR_SERIES_MAP["nfl_props"]` held `["KXNFLPAS", "KXNFLRSH", "KXNFLREC", "KXNFLTD"]` — four phantom series names that never existed on Kalshi. Every scan fetched zero markets for months and reported success. The ARCH-11 category registry validator will catch drift *between* `SECTOR_SERIES_MAP` and `data/categories.yaml`, but neither side is checked against Kalshi. If both drift together from reality, the validator is happy and we still ship a broken catalog.

**Fix scope:** a standalone script, no runtime dependency on Kalshi from the scan path.

1. **`scripts/check_kalshi_series.py`** — queries `GET /trade-api/v2/series?category=Sports&limit=1000`, paginates the cursor, builds the set of live ticker prefixes. Unauthenticated (Kalshi's `/series` endpoint is public).

2. **Diffs against `SECTOR_SERIES_MAP`** and emits three buckets:
   ```
   [NEW]   KXXFLGAME — Extreme Football League (live on Kalshi, not in SECTOR_SERIES_MAP)
   [STALE] KXNFLPAS  — in SECTOR_SERIES_MAP, but /series returns 404 (retired or typo'd)
   [OK]    KXNBAGAME — matched (category: nba)
   ```

3. **Optional: also probe each `[OK]` entry for market count** across open/closed/settled to flag "live in /series but zero markets ever" cases. That's the signature of a real NFL-prop-style typo where the series exists but the ticker prefix is wrong (e.g. `KXNFLPAS` vs `KXNFLPASSYDS` — both starting with `KX` but the shorter one is a substring match).

4. **Exit code** non-zero if any `[NEW]` or `[STALE]` appears, so the script can run in CI or as a pre-commit hook without human review.

5. **Wire into pre-commit** as a manual-stage hook (`pre-commit run check-kalshi-series`), not a pre-commit auto-hook — it hits the network and shouldn't block every commit. Intent is "run this before releases and once a week" not "run every commit."

6. **Optional: cron it** via `.github/workflows/kalshi-drift.yml` to open a GitHub issue if drift is detected. Weekly cadence is plenty — Kalshi series launches aren't a real-time concern.

**Blocker:** none. Pure infrastructure — can ship any time. Most valuable immediately after ARCH-11 lands, because ARCH-11 locks in the catalog structure that this script audits.

**What this does NOT do:** automatically update `SECTOR_SERIES_MAP` when it detects drift. That would be a runtime behavior change and requires human review (new sectors need category registry entries, models, resolvers, etc. — the ARCH-11 plumbing). The script's job is surfacing drift, not fixing it.

**Estimated effort:** ~2 hours including tests. Small, high-leverage, exactly the kind of automation that would have prevented the PR #6 typo discovery from taking a full research session.

### ARCH-9 Resurrect TheOddsAPI Legacy Client as Paid Fallback [P3]
**Files:** `evmax/clients/pinnacle.py`, `evmax/agents/odds/sharp_agent.py`, `evmax/models/odds.py`

The legacy `PinnacleClient` at `evmax/clients/pinnacle.py` is a fully-implemented TheOddsAPI wrapper (moneyline + spreads + totals + player props + quota tracking, ~900 lines) that was superseded by `PinnacleGuestClient` but never deleted. It's currently dead code — imported only by the dead `pipeline/runner.py` and by one vestigial `get_quota()` display call in `evmax/cli/commands/agents.py:382` that renders an empty string because `_quota` is never populated.

Resurrecting it as a **commercial fallback** when Pinnacle Guest is unavailable is a real option:

- **Data quality:** TheOddsAPI includes Pinnacle in its bookmaker list, so you get the same sharp book — just with ~30-60s latency on the passthrough.
- **Effort:** moderate — client exists, needs wiring into `SharpOddsAgent` with source/latency marking and a fallback-chain strategy. Likely ~150-200 lines net including tests.
- **Cost:** ~$30-100/mo depending on TheOddsAPI quota tier.
- **Requires ARCH-8 first** — the `is_stale` / `source` field plumbing from ARCH-8 is what lets the EV calculator know to discount TheOddsAPI-sourced bets for latency.

**Consider this only if**:
- ARCH-8 observability shows Pinnacle Guest outages are frequent enough to materially affect EV capture (>1 outage/week sustained)
- You're willing to pay for a paid API tier
- The ~60s Pinnacle passthrough latency is acceptable for your betting cadence (pre-game only; useless for in-play)

**Alternative to this item:** ARCH-10 below (authenticated ps3838 API) achieves the same resilience goal without paying TheOddsAPI, at the cost of a Pinnacle account signup. Pick one.

### ARCH-10 Authenticated `api.ps3838.com` as Long-Term Primary [P3]
**Files:** new `evmax/clients/ps3838.py`, `evmax/agents/odds/sharp_agent.py`

Pinnacle operates `api.ps3838.com` as the authenticated version of their sharp-odds API — same sportsbook company, same lines, but served through documented/supported infrastructure that runs separately from the guest-tier Arcadia stack. Verified 2026-04-13: when `guest.api.arcadia.pinnacle.com` returned 503 MAINTENANCE, `api.ps3838.com` returned **403 Forbidden cleanly** (auth challenge, not maintenance) — strong evidence they're on independent maintenance schedules.

**Requires a real Pinnacle account** with API privileges. Historically free for active bettors with a deposit minimum; terms vary. Not a trivial signup — this is a real account with KYC, deposits, etc.

**Fix scope:**
1. New `evmax/clients/ps3838.py::Ps3838Client` wrapping the authenticated Pinnacle API endpoints (`/v1/odds`, `/v1/fixtures`, `/v1/line`).
2. Credentials in `secrets/PS3838_USERNAME` + `secrets/PS3838_PASSWORD` (HTTP Basic), read via settings.
3. Parser adapter — the authenticated API response shape is **different** from the Arcadia guest stack. Not a drop-in swap.
4. Wire as fallback in `SharpOddsAgent` (tier after Pinnacle Guest but before TheOddsAPI).
5. **Long-term path:** promote to primary once confidence is established; retire `PinnacleGuestClient` if the authenticated path proves more stable.

**Blockers:**
- Pinnacle account signup (real-world, one-time)
- Response format research — the published Pinnacle API docs are sparse and the real shape needs verification
- Rate limits on the authenticated tier are different and need respect

**When to prioritize:** only if ARCH-8 + ARCH-9 together aren't enough, OR if you're philosophically uncomfortable with the guest-tier dependency long-term. Good candidate for when the live pipeline starts carrying real bankroll.

### ARCH-11 CLV Is Computed But Not Wired Into Any Feedback Loop [P2]
**Files:** `evmax/agents/cleanup/resolver.py:921 backfill_clv()`, `evmax/cli/commands/cleanup.py:1021 backfill-clv`, `evmax/cli/commands/cleanup.py:148 show`, `evmax/agents/cleanup/db.py:102 clv_pct column`, `evmax/archiver.py:288 get_closing_line()`

CLV (Closing Line Value = `sharp_true_prob − pinnacle_close_prob`) is fully plumbed — column on `ev_predictions`, backfill from `archive.db`, green/red display in `cleanup show` — but **nothing downstream reads `clv_pct` to make a decision.** It's a cosmetic column.

What's missing:
- `backfill-clv` is orphaned. It's not called from `cleanup resolve` and not in the documented daily workflow. You have to remember to run it manually or the column stays NULL.
- No per-sector CLV aggregation. `compute_brier_scores_by_sector` exists; the CLV equivalent does not.
- `cleanup adjust` auto-tunes `sharp_weight` from Brier alone. CLV — which is lower variance and available hours after close rather than after outcome resolution — is a much faster signal that could feed the same tuner.
- No CLV in `cleanup metrics` output or any alert path.

Why it matters: CLV is the leading indicator for model sharpness. Brier needs 100+ resolved bets per sector to converge; CLV is near-deterministic per bet and available at close. A continuous model improvement loop (see the "modelling agent" concept discussed 2026-04-13) should read CLV first and Brier second. Right now it can't, because CLV is just a column nobody queries.

**Fix options:**
1. **Wire it in.** Auto-run `backfill_clv` at the end of `cleanup resolve`. Add `compute_clv_by_sector(weeks)` to `metrics.py`. Add a CLV row to `cleanup metrics` output. Feed avg CLV per sector into `cleanup adjust` as a secondary signal alongside Brier.
2. **Rip it out.** If CLV is not going to drive any decision, delete the column, the backfill function, the CLI command, and the display column. Less code.

**Connection to UNIQUE(market_id) migration:** When `log_gaps` moves to freeze-on-first-insert (see the multi-scan dedup discussion 2026-04-13), CLV gets *better* — `sharp_true_prob` will be the actual first-flag value rather than a refreshed scan closer to close, so `(entry − close)` measures real market movement the model caught. That migration is a prerequisite for #1 being meaningful.

**When to prioritize:** bundle with the modelling-improvement-agent work. Doesn't make sense to build that agent without CLV as one of its inputs.

---

## Section 6 — Player Props (In Progress)

### PROPS-1 Define NFL Props Backend Before Fetching [P1]
Currently NFL prop Kalshi series are fetched but silently discarded (see ARCH-4). Implement or disable.

### PROPS-2 Add Combo Stat Prop Parsing Test [P2]
`KXNBAPRA` (points+rebounds+assists) — verify the Kalshi title regex in `kalshi.py` correctly parses combo stat thresholds (e.g., "Jokic 55.5+ PRA" → player=Jokic, stat=points_rebounds_assists, threshold=55.5).

### PROPS-3 Prop Injury Boosts Broken (see BUG-5) [P1]
Already filed above — fix the team context extraction for props.

### PROPS-4 Add Prop Lines to CLI `show` Output [P3]
Currently `evmax cleanup show` displays game-level bets only. Props are logged to `prop_observations` but there's no CLI command to review prop performance history.
- Add `evmax cleanup show --props` flag to query `prop_observations` with win/loss/pending counts per stat type

### PROPS-5 Prop Model Training Pipeline [P3]
`prop_observations` accumulates data (all lines, not just +EV) with the intent of model training. Currently nothing reads this table for calibration. Define the training loop: what metric to optimize (Brier for over probabilities), how often to retrain, what feature set.

---

## Section 7 — Sector Gaps

### ~~SECTOR-1 Hockey (NHL) Elo Calibration~~ ✅ SHIPPED 2026-07-18
See MODEL-2 above — NHL Elo runs calibrated `K=6` / `home_adv=48` (`elo_agent.py:67,133`), and `elo_state.json['nhl']` is seeded. What remains is a blend decision, not a calibration: `SECTOR_WEIGHT_OVERRIDES["nhl"]` still holds `elo: 0.0`. (NHL outcome resolution is fixed in PR #1.)

### ~~SECTOR-2 NCAAW Elo Calibration~~ ✅ SHIPPED 2026-07-11
Shipped with the NCAAB/NCAAW opponent-adjusted efficiency stack (K=35 / home_adv=80; see MODEL-2 above and `docs/ncaab-blend-eval.md`).

### ~~SECTOR-3 Tennis Tournament Calendar for Surface Lookup~~ ✅ SHIPPED (PR #5)
Solved by reading Kalshi's `event.product_metadata.competition` field instead of a separate calendar lookup. See MODEL-1 in Recently Shipped.

### SECTOR-4 Sectors in Registry Not in CLAUDE.md [P3]
~~`nhl`, `baseball`, `valorant`, `ufc`, `f1` are absent from CLAUDE.md~~ — partially fixed in PR #1 (NHL, Baseball, Valorant added to Key Sectors). UFC and F1 still undocumented but they're long-tail.

---

## Priority Order (open items only)

| Priority | Item | Impact |
|----------|------|--------|
| P1 | ARCH-11 Category mode config (live/shadow/disabled) | Prerequisite to MODEL-9 and general capability across all sectors |
| P1 | MODEL-9 NFL prop shadow validation | Blocks Stage 5 live betting; distinguishes real edge from backtest leakage |
| P1 | MODEL-11 WNBA shadow validation + promote to live | Blocks 2026 WNBA live betting; walk-forward passes but needs live-price confirmation |
| P1 | MODEL-14 NBA props post-calibration validation | Just landed — need 2-3 wks of live data to confirm Path A behaviour holds |
| P1 | PROPS-1 NFL props backend | Merged into MODEL-9 (kalshi.py typo fix ships with the shadow PR) |
| P2 | MODEL-2 Elo calibration — NHL ✅ SHIPPED 2026-07-18, `ncaaf` still open | NHL now K=6 / home_adv=48. `ncaaf` has no entry in either dict and falls through to K=20 / home_adv=**0.0** while carrying elo at 0.25. Separately, the `nhl` ensemble `elo: 0.0` weight is an open blend decision |
| P2 | MODEL-6 Tennis indoor court modifier | Waits on live indoor-event data; `is_indoor` seam already in MODEL-1 |
| P2 | MODEL-13 WNBA player_impact agent | WNBA star-out impact is ~8-10pt swing; bigger gain than MODEL-12. Needs data-source decision first. |
| P2 | TEST-3 PinnacleGuestClient tests | Only live sharp source untested |
| P2 | TEST-4 Coordinator integration test | Catches wiring regressions |
| P2 | TEST-6 Prop pipeline — remaining | nba_stats.py still uncovered |
| P2 | ARCH-8 Pinnacle maintenance + stale cache | Guest API maintenance windows take whole pipeline offline; retry layer currently useless against multi-minute outages |
| P2 | ARCH-12 Kalshi series drift detection | Standalone script + optional pre-commit hook; would have caught the PR #6 NFL prop typo on day one. Best landed right after ARCH-11. |
| P3 | DOC-2b / DOC-3b remaining doc polish | — |
| P3 | MODEL-3 Form draw normalization | Cosmetic precision |
| P3 | MODEL-4 Poisson EWMA | Long-term staleness |
| P3 | MODEL-7 Non-QB NFL prop features | Downstream of MODEL-9 — usage decomposition + Vegas totals + position-aware defense |
| P3 | MODEL-12 Port shot_quality/matchup to WNBA | ~0.005 Brier close-out; defer until post-MODEL-11 validation. |
| P3 | MODEL-15 NBA props v2 features | Career-mean prior + shot-type variance. Days-rest tested, doesn't help. |
| P3 | MODEL-16 Refit NBA props calibration with opp_adj | Modest gain (~0.002 Brier); needs team-stat backfill |
| P3 | ARCH-9 Resurrect TheOddsAPI as paid fallback | Alternative to ARCH-10; pay-to-resilience path |
| P3 | ARCH-10 Authenticated ps3838 API | Long-term alternative to guest tier; requires real Pinnacle account |
| P3 | All other ARCH-* (1–6) | Skipped for now |

---

## What needs to occur now (April 25, 2026)

The active list — ordered by deadline / blocker.

### Now → next 2 weeks (NBA props shadow validation + WNBA opener prep)

1. **Flip nba_props to shadow (~30 sec).** Edit `data/categories.yaml`, change `nba_props.mode: live → shadow`. Run `evmax categories validate` to confirm. This stops the new calibrated model from sizing real Kelly stakes during the validation window. Existing `prop_observations` rows (1,107 pre-calibration) stay untouched; new rows from this point forward will be tagged `mode='shadow'`. **MODEL-14 step 1.**
2. **Run scans daily as usual.** Each scan adds calibrated-model rows to `prop_observations`. We have a clean before/after on the same population (NBA playoffs).
3. **Check shadow metrics weekly.** After ~50 resolved post-calibration bets (probably 7-10 days of play):
   ```
   evmax cleanup shadow show --days 14 --category nba_props
   evmax cleanup shadow metrics --days 14 --category nba_props
   ```
   Compare to the pre-calibration baseline (model Brier 0.1709 vs Kalshi 0.1991). Pass criteria in MODEL-14 above.
4. **If validation passes, promote back to live**: `evmax cleanup shadow promote nba_props`. Otherwise, stay shadow and diagnose.
5. **Stop running** `scripts/fetch_nba_game_logs.py` for prior seasons. nba_api is rate-limiting; recovering that needs proper backoff which is P3. Career-mean prior is closed-tabled until then.
6. **WNBA opener prep — verify the Elo state holds**. We applied the offseason regression on Apr 23. If any major roster moves happened since (notable signings, injuries that materially change a team), rerun `python scripts/wnba_offseason_regress.py --dry-run` after editing `data/models/wnba_2026_offseason.yaml` to confirm the picture. Otherwise no action.
7. **WNBA seed refresh once opening night happens (May 16).** Rerun `python scripts/seed_wnba_efficiency.py --year 2026` after Day 1, then weekly. Without this the agent's predictions stay frozen at 2025 stats.

### May 16 onward (WNBA shadow validation)

5. **Watch WNBA in shadow.** Same drill as MODEL-9 for NFL props. Aim for ≥ 200 shadow bets across ≥ 3 distinct weeks. Then `evmax cleanup shadow metrics --days 30 --category wnba` and check the gate criteria in MODEL-11.
6. **Promote WNBA to live** via `evmax cleanup shadow promote wnba` once it passes the gate.

### Nothing happening here for a while

- **NFL props** — season starts Sept 2026. MODEL-9 work is queued for then.
- **NBA games** — already live, no action needed.
- **Tennis** — live. Stable.
- **Soccer / NHL / NCAAB** — live, stable.
- **NCAAW** — live; Elo calibrated (K=35/HA=80) + opponent-adjusted efficiency stack shipped 2026-07-11 (see `docs/ncaab-blend-eval.md`).

### What's NOT actionable right now

These are tracked but waiting on something:
- MODEL-15 (career-mean prior) — wait for nba_api rate-limit cooldown or write the retry loop.
- MODEL-12 (WNBA shot_quality / matchup) — defer until post-MODEL-11 validation.
- MODEL-13 (WNBA player_impact) — needs data-source decision, can wait until 2026 season has 30 games of injury data.
- MODEL-16 (refit calibration with opp_adj) — modest gain, can wait.
- ARCH-8 (Pinnacle maintenance handling) — only acutely needed during the next outage. File-and-wait.

---

## Things to keep in mind (WNBA 2026 season)

Collected from the April 2026 WNBA prep pass — not "todo" items but recurring constraints / operational reminders.

1. **The WNBA 2025 seed data has contamination.** ESPN's WNBA scoreboard surface includes the All-Star Game (Team Clark / Team Collier) and occasional international exhibitions (Team Brazil / Team Antelopes / Toyota Antelopes). `scripts/seed_wnba_efficiency.py` filters via `REAL_WNBA_TEAMS` allow-list; any new seed-style script must do the same or the league averages skew (Collier's 139.8 ORtg in a 1-game sample would distort the pace average badly).
2. **Form staleness is reference-date-aware.** The guard uses `market.event_date` (or `game_date` in the walk-forward harness) as the reference, not wall-clock today. Historical replays stay meaningful. Do NOT revert this to `date.today()` — it silently breaks every walk-forward against past seasons.
3. **Quality weighting in form does not work.** We tried multiplying each win by `(1 + elo_gap / 400)` and backtested net-negative on WNBA (Brier 0.2470 → 0.2615). Elo already captures opponent strength; adding it to Form double-counts. Do not re-propose this — refer to the April 2026 walk-forward if someone tries to revive it.
4. **Offseason regression is manually triggered.** `scripts/wnba_offseason_regress.py` must be run once before each WNBA season opener. It backs up the current Elo state and applies 35% shrinkage + the YAML move list in one pass. Edit `data/models/wnba_2026_offseason.yaml` for 2027 — regeneration isn't automated.
5. **`wnba_efficiency_state.json` is a seed, not a running update.** The agent's `update()` method is a no-op — game-level score pairs don't carry the FGA / FTA / TO / OREB detail needed to recompute ORtg/DRtg. Re-run `python scripts/seed_wnba_efficiency.py --year 2026` weekly during the 2026 season to roll new games into the efficiency inputs. Otherwise predictions stay frozen at 2025 stats.
6. **WNBA playoff tightening is NOT calibrated.** `WNBAPossessionSimAgent._simulate_game` omits the `is_playoff` / `PLAYOFF_ORTG_FACTOR` branch that NBA uses. The NBA factor (0.9623) was derived from a specific sample of 17 NBA playoff games; blindly porting it to WNBA would add unmeasured bias. Leave off until WNBA has a comparable playoff-sample measurement.
7. **Pace clip must match game length.** NBA clip `[80, 120]` and WNBA clip `[65, 100]` differ because WNBA games are 40 min vs NBA's 48 — same clip on both would push WNBA predictions into impossibly fast territory. If anyone adds a new basketball-league sim (EuroLeague? NCAAW?), adjust the clip by `game_minutes / 40` from WNBA or `/ 48` from NBA.
8. **Expansion priors (Fire 1470 / Tempo 1440) are guesses based on Valkyries' Y1 trajectory.** Verify against their actual opening-day Pinnacle lines once those post. If Fire opens as a 1400-level team on Pinnacle, shrink the prior and re-seed before scanning.
9. **WNBA is `shadow` mode on launch.** Kelly sizing against bankroll is disabled until MODEL-11 validation closes. The scanner still produces predictions and logs them to `prop_observations` / `ev_predictions` with `mode='shadow'` and `captured_yes_price` — that's the data MODEL-11 reads.
10. **The existing NFL-prop shadow tooling (`evmax cleanup shadow metrics`) handles WNBA automatically.** No new CLI needed — just pass `--category wnba`. Promotion flips the YAML mode to `live` and re-enables bankroll sizing in one command.
