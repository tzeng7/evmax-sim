# evmax TODO

> Items discovered via full codebase quality sweep (April 2026).
> PLAN.md covers older completed work (Batches A–F ✅). This file tracks what's next.
> Each item has a priority: **P1** (correctness/money), **P2** (quality/coverage), **P3** (nice to have).

---

## Recently Shipped

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
| P1 | PROPS-1 NFL props backend | Dead API calls |
| P2 | MODEL-2 NCAAW/NHL Elo calibration | Uncalibrated K + home adv |
| P2 | MODEL-6 Tennis indoor court modifier | Waits on live indoor-event data; `is_indoor` seam already in MODEL-1 |
| P2 | TEST-3 PinnacleGuestClient tests | Only live sharp source untested |
| P2 | TEST-4 Coordinator integration test | Catches wiring regressions |
| P2 | TEST-6 Prop pipeline — remaining | nba_stats.py still uncovered |
| P3 | DOC-2b / DOC-3b remaining doc polish | — |
| P3 | MODEL-3 Form draw normalization | Cosmetic precision |
| P3 | MODEL-4 Poisson EWMA | Long-term staleness |
| P3 | All ARCH-* | Skipped for now |
