# evmax TODO

> Items discovered via full codebase quality sweep (April 2026).
> PLAN.md covers older completed work (Batches A–F ✅). This file tracks what's next.
> Each item has a priority: **P1** (correctness/money), **P2** (quality/coverage), **P3** (nice to have).

---

## Recently Shipped

### PR #2 — Poisson bucketing + prop injury boost + TEST-2/TEST-6
- ✅ BUG-4 — Poisson NBA/NFL/NCAAB score matrix now bucketed (NBA=5, NFL=4, NCAAB=5). `lam_h`/`lam_a` scaled by bucket before the matrix is built so Poisson mass fits inside MAX_SCORE.
- ✅ BUG-5 — Prop injury boost now uses `nba_props_cache.lookup_player_team()` to find the player's actual team instead of parsing a nonexistent game slug out of `parts[2]`. The boost was silently never firing for any prop.
- ✅ TEST-2 — PitcherModelAgent now has 14 tests across sector gating, seeding, Pythag prediction, confidence tiers, and team-name fallback. Writing the tests exposed **BUG-9** (filed below).
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

_(BUG-4 and BUG-5 shipped — see "Recently Shipped" above.)_

### BUG-9 Pitcher Pythag Semantics Are Inverted [P1]
**File:** `evmax/agents/models/pitcher_agent.py:169-178`
The Pythagorean block assigns `home_ra = away_era` and `away_ra = home_era`, treating "runs the home team allows" as a function of the *opposing* starter. Runs allowed is driven by a team's *own* pitcher, so the mapping is backwards: a team with the better starter ends up *less* favored after normalization. In a 2.50-vs-5.50 ERA matchup the home ace produces `prob_a ≈ 0.38` (before home bonus) instead of the ~0.80 it should.

Pinned in `tests/test_pitcher_agent.py::test_known_bug_better_home_pitcher_is_less_favored`. Fix should flip that test assertion.

Suggested fix: either use `home_rs=away_era, home_ra=home_era` (both pitchers enter via each team's own ERA) or simplify to the equivalent `home_wp = away_era^e / (away_era^e + home_era^e)`. `league_avg_era` drops out of the matchup formula under both options — keep it as a fallback for unknown-ERA pitchers.

---

## Section 3 — Model Quality

### MODEL-1 Tennis Surface Detection is Too Fragile [P1]
**File:** `evmax/agents/models/tennis_model_agent.py`
Surface is inferred from keywords in `market.title`. In practice, Kalshi market titles for ATP/WTA are typically just "Player A vs Player B" with no tournament context. The surface defaults to `hard` almost always, meaning surface-specific Elo (clay/grass advantages) is rarely exercised despite existing in the state.

> **Note:** `tests/test_tennis_model.py::test_known_bug_title_without_tournament_silently_defaults_to_hard` pins the current buggy behavior. Fixing MODEL-1 will require updating that test, making the regression visible at code-review time.

Fix options (in order of preference):
1. Enrich `TennisSectorHandler` to carry a `surface` field from a tournament calendar lookup (ATP tour schedule is public)
2. As a fallback, add a lookup table of known tournament names → surface (Roland Garros → clay, Wimbledon → grass, US Open → hard, Australian Open → hard, etc.) and check if the Kalshi series ticker or league name contains a tournament keyword

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

### MODEL-5 Tennis Model Weight Exceeds All Other Models Combined [P3]
**File:** `evmax/agents/models/ensemble_agent.py`
`TennisModelAgent.weight = 0.45` is higher than any other model weight. For tennis events where the agent contributes, it dominates the blend, which is aggressive given the surface detection weakness in MODEL-1. Consider reducing to 0.35 (matching Elo) after MODEL-1 is fixed and surface signals are reliable.

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

### ARCH-7 YES-Team Price Heuristic Fails on Even-Money Games [P2]
**File:** `evmax/agents/odds/ev_gap_agent.py:345-349`
The `_resolve_yes_via_market_teams` fallback aligns YES to outcome_a when `|kalshi_ask - sharp.true_prob_a| < 0.05`. When both teams have near-50% probability, both conditions can be true simultaneously, producing ambiguous alignment.
- Add a tiebreaker: prefer the alignment that results in a higher EV (more conservative: require `< 0.03` threshold, or require one distance to be at least 2x the other)

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

### SECTOR-3 Tennis Tournament Calendar for Surface Lookup [P1]
See MODEL-1 above. This is the highest-leverage model improvement for tennis.

### SECTOR-4 Sectors in Registry Not in CLAUDE.md [P3]
~~`nhl`, `baseball`, `valorant`, `ufc`, `f1` are absent from CLAUDE.md~~ — partially fixed in PR #1 (NHL, Baseball, Valorant added to Key Sectors). UFC and F1 still undocumented but they're long-tail.

---

## Priority Order (open items only)

| Priority | Item | Impact |
|----------|------|--------|
| P1 | BUG-9 Pitcher Pythag inversion | Better pitcher → less favored (backwards) |
| P1 | MODEL-1 / SECTOR-3 Tennis surface detection | Surface Elo rarely fires in practice |
| P1 | PROPS-1 NFL props backend | Dead API calls |
| P2 | MODEL-2 NCAAW/NHL Elo calibration | Uncalibrated K + home adv |
| P2 | TEST-3 PinnacleGuestClient tests | Only live sharp source untested |
| P2 | TEST-4 Coordinator integration test | Catches wiring regressions |
| P2 | TEST-6 Prop pipeline — remaining | nba_stats.py still uncovered |
| P3 | DOC-2b / DOC-3b remaining doc polish | — |
| P3 | MODEL-3 Form draw normalization | Cosmetic precision |
| P3 | MODEL-4 Poisson EWMA | Long-term staleness |
| P3 | MODEL-5 Tennis weight tuning | After MODEL-1 |
| P3 | All ARCH-* | Skipped for now |
