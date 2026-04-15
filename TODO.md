# evmax TODO

> Items discovered via full codebase quality sweep (April 2026).
> PLAN.md covers older completed work (Batches A–F ✅). This file tracks what's next.
> Each item has a priority: **P1** (correctness/money), **P2** (quality/coverage), **P3** (nice to have).

---

## Recently Shipped

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

### MODEL-2 NCAAW and NHL Have No Calibrated K-Factor / Home Advantage [P2]
**File:** `evmax/agents/models/elo_agent.py`
`K_FACTORS` and `HOME_ADVANTAGE_ELO` have entries for `nba`, `nfl`, `ncaab`, `soccer`, `lol`, `cs2` but not for `ncaaw` or `nhl`. Both silently use NBA defaults (K=20.0, home_adv=0.055).
- Add calibrated values for NCAAW: home advantage is similar to NCAAB, K-factor should be higher (more variance)
- Add NHL: K=16, home_adv=0.04 (puck-line markets exist)

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
**Files:** `evmax/agents/coordinator.py`, `evmax/agents/cleanup/db.py`, `evmax/cli/commands/agents.py`, `evmax/cli/commands/cleanup.py`

**Context.** MODEL-8 produced a backtest that passes all three Stage 4 gates — Brier 0.179, log-loss improved over prior, ROI +79% at ev≥3% vol≥1000 — **BUT** the ROI number uses Kalshi `last_price_dollars` (closing price), which for settled NO markets drifts downward through the game as events unfold. The ROI signal is either (a) real retail-overprices-YES edge, (b) retrospective leakage, or (c) a mix. We can't distinguish these three using only Stage 1's historical dataset.

**Distinguishing the three requires capturing pre-game prices at scan time and resolving outcomes separately.** That's what shadow mode is for: run the NFL prop scanner during live NFL weeks, log (model_prob, pre_game_yes_price, threshold, player, game) tuples to a shadow table, resolve outcomes via ESPN boxscore, compute ROI using prices the live bettor actually could have gotten. If shadow ROI holds within ~15pp of the backtest's +79%, the edge is real and we ship Stage 5 with real Kelly sizing. If it collapses below 0%, we know it was leakage and close PROPS-1.

**Fix scope:** depends on ARCH-11 shipping first (the general category-mode config). Assuming that's in place, MODEL-9 only needs to:
1. Set `nfl_props` category default to `shadow` in `data/category_modes.yaml` (or equivalent).
2. **kalshi.py typo fix** — fold the phantom series names into this PR since NFL prop markets won't appear in scan output until the fix lands.
3. **Wire the NFL prop probability compute into the coordinator** — add the NBA-mirrored branch in `_fetch_props()` that calls `compute_nfl_prop_prob` with the features pre-computed from a daily-refreshed disk cache (new `data/nfl_props_cache.json`, schema mirrors NBA).
4. **Resolver:** confirm `evmax/agents/cleanup/prop_resolver.py` auto-resolves NFL props via ESPN boxscore (exploration suggests it already does — no code change expected, just verify with a fixture test).
5. **NO-side betting logic:** per the MODEL-8 / Stage 4 finding, Kalshi NFL prop markets are systematically YES-overpriced and the model's edge is on the NO side. The coordinator needs to handle both sides — either extend EVGap to carry a `side: "yes" | "no"` field or produce two gaps per market (one YES, one NO) and rely on the EV filter. Whatever the mechanism, the NFL prop shadow path must NOT only emit YES gaps (that would hide the real edge). This is a one-sector generalization of the existing coordinator logic.
6. **Minimum sample gate to promote nfl_props from `shadow` → `live`:**

   - ≥ 500 shadow bets captured, ≥ 3 distinct NFL weeks
   - Shadow ROI at ev≥3%, vol≥1000, NO-only ≥ 65% (backtest was 79%, allow 15pp degradation)
   - Shadow Brier within 10% of backtest Brier (0.197 max)
   - Calibration chart tail miss ≤ 15pp at 90–100% bin (the MODEL-8 residual)

**Blocker:** NFL regular season starts Sept 2026. Shadow validation requires at least 3-4 weeks of live NFL to be meaningful. Until then, build the infrastructure only — don't attempt to run shadow during the offseason (no markets).

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

### SECTOR-1 Hockey (NHL) Elo Calibration [P2]
See MODEL-2 above. NHL is in the registry and Kalshi series but Elo uses NBA defaults. (NHL outcome resolution is fixed in PR #1.)

### SECTOR-2 NCAAW Elo Calibration [P2]
See MODEL-2 above. NCAAW has a `REST_ELO_ADJ` entry but no K-factor or home advantage.

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
| P1 | PROPS-1 NFL props backend | Merged into MODEL-9 (kalshi.py typo fix ships with the shadow PR) |
| P2 | MODEL-2 NCAAW/NHL Elo calibration | Uncalibrated K + home adv |
| P2 | MODEL-6 Tennis indoor court modifier | Waits on live indoor-event data; `is_indoor` seam already in MODEL-1 |
| P2 | TEST-3 PinnacleGuestClient tests | Only live sharp source untested |
| P2 | TEST-4 Coordinator integration test | Catches wiring regressions |
| P2 | TEST-6 Prop pipeline — remaining | nba_stats.py still uncovered |
| P2 | ARCH-8 Pinnacle maintenance + stale cache | Guest API maintenance windows take whole pipeline offline; retry layer currently useless against multi-minute outages |
| P2 | ARCH-12 Kalshi series drift detection | Standalone script + optional pre-commit hook; would have caught the PR #6 NFL prop typo on day one. Best landed right after ARCH-11. |
| P3 | DOC-2b / DOC-3b remaining doc polish | — |
| P3 | MODEL-3 Form draw normalization | Cosmetic precision |
| P3 | MODEL-4 Poisson EWMA | Long-term staleness |
| P3 | MODEL-7 Non-QB NFL prop features | Downstream of MODEL-9 — usage decomposition + Vegas totals + position-aware defense |
| P3 | ARCH-9 Resurrect TheOddsAPI as paid fallback | Alternative to ARCH-10; pay-to-resilience path |
| P3 | ARCH-10 Authenticated ps3838 API | Long-term alternative to guest tier; requires real Pinnacle account |
| P3 | All other ARCH-* (1–6) | Skipped for now |
