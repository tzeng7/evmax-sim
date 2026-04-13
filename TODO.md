# evmax TODO

> Items discovered via full codebase quality sweep (April 2026).
> PLAN.md covers completed work (Batches A–F ✅). This file tracks what's next.
> Each item has a priority: **P1** (correctness/money), **P2** (quality/coverage), **P3** (nice to have).

---

## Section 1 — Documentation & Auto-Sync

The codebase has drifted from its docs. CLAUDE.md has at least three stale facts, and there are
no module-level docstrings or per-folder READMEs. Goal: make it easy for Claude Code to
stay oriented in future sessions without needing to re-explore everything from scratch.

### DOC-1 Fix Stale CLAUDE.md Facts [P2]
Three confirmed inaccuracies:
- **Kalshi rate limiting**: says `asyncio.Semaphore(3)` — actual code uses `AsyncLimiter(10, 1.0)` (token bucket, 10 req/s from `aiolimiter`)
- **NCAAW sharp source**: says "uses TheOddsAPI `basketball_wncaab`" — code uses `PinnacleGuestClient` (league_id=583) just like every other sector
- **Sharp weight default**: says 0.85 — `AgentCoordinator.__init__` defaults to 0.40; 0.85 lives in CLI default + `_DEFAULT_CONFIG` only

### DOC-2 Add Per-Folder README.md Files [P3]
Key folders that are confusing without orientation:
- `evmax/agents/` — explain pub/sub bus, AgentRequest/Response, coordinator lifecycle
- `evmax/clients/` — explain that `pinnacle.py` (TheOddsAPI) is legacy/unused; `esports_pinnacle.py` handles ALL sectors
- `evmax/models_ml/` — explain what's live vs dead (`sharp_only.py` is dead)
- `evmax/sectors/aliases/` — explain YAML alias format and how fuzzy matching uses them

### DOC-3 Add `__init__.py` Docstrings to All Packages [P3]
Many packages have empty `__init__.py` files. Add one-line module docstrings so `help(evmax.agents)` is useful and Claude Code can orient quickly.

### DOC-4 Pre-commit Hook: Remind to Update Docs [P3]
Add a `.pre-commit-config.yaml` with a custom hook that:
- On any commit touching `evmax/agents/`, `evmax/clients/`, `evmax/sectors/`, `evmax/ev/`, or `evmax/models_ml/`, prints a reminder: `"Remember: update CLAUDE.md, README.md, and relevant folder READMEs if public interfaces changed"`
- Use `hooks.PostToolUse` in Claude Code settings (`.claude/settings.json`) so Claude itself gets this reminder after editing key files

Example `.claude/settings.json` hook:
```json
{
  "hooks": {
    "PostToolUse": [{
      "matcher": "Edit|Write",
      "hooks": [{
        "type": "command",
        "command": "python scripts/check_doc_sync.py \"$CLAUDE_TOOL_INPUT_FILE_PATH\""
      }]
    }]
  }
}
```
`check_doc_sync.py` would print a warning when a changed file falls in a tracked folder.

---

## Section 2 — Bugs (Correctness Issues)

### BUG-1 Tennis Outcome Resolution is Missing [P1]
**File:** `evmax/agents/cleanup/resolver.py`
Tennis bets are found, logged, and never resolved. `ESPN_SPORT_MAP` has no tennis entry; neither does `BO3_DISCIPLINE`. There is no other resolution source.
- Add ATP/WTA resolution via the ESPN tennis scoreboard endpoint (sport=`tennis`, league=`atp`/`wta`)
- Or add a manual-resolve CLI fallback for sectors with no data source

### BUG-2 NHL Outcome Resolution is Missing [P1]
**File:** `evmax/agents/cleanup/resolver.py`
Same as BUG-1 but for hockey. `ESPN_SPORT_MAP` has no `nhl` entry.
- Add `"nhl": ("hockey", "nhl", {})` to `ESPN_SPORT_MAP`
- Verify `elo_agent.py` K-factor and home advantage have NHL entries (currently defaults to NBA values silently)

### BUG-3 UEL + MLS Bets Never Auto-Resolve [P1]
**File:** `evmax/agents/cleanup/resolver.py:51-58`
`ESPN_SOCCER_LEAGUES` covers EPL, La Liga, Bundesliga, Serie A, Ligue 1, UCL — UEL and MLS are missing. Any +EV bets on these competitions accumulate without resolution indefinitely.
- Add UEL (`uefa.uel`) and MLS (`usa.1`) ESPN league slugs to `ESPN_SOCCER_LEAGUES`

### BUG-4 Poisson NBA/NFL Score Matrix is Truncated [P1]
**File:** `evmax/agents/models/poisson_agent.py`
`MAX_SCORE` maps `nba→25`, `nfl→20`, `ncaab→20`. The comment says "buckets of 5 pts → effectively 0–125 range" but **no bucketing code exists**. Each matrix cell represents exactly 1 scoring unit, so the matrix covers 0–25 points total — absurdly low for basketball/football.
The normalization in `_win_draw_probs()` partially recovers the win probability direction but the distribution is wrong and produces overconfident predictions for large favorites.
- Implement the intended bucketing: divide `lam_h` and `lam_a` by bucket_size (5 for NBA, 4 for NFL), use `max_g` buckets, then check `h_score_bucket > a_score_bucket` for win

### BUG-5 Prop Injury Boosts Are Silently Never Applied [P1]
**File:** `evmax/agents/coordinator.py` (prop evaluation block)
The injury boost lookup splits `prop.event_id` on `"::"` expecting a `game_slug` segment, but prop event_ids are formatted as `{sector}::{date}::prop::{player}::{stat}::{threshold}`. The `parts[2]` segment is always `"prop"`, not a game slug. So `team_boosts` is always empty for props.
- Extract the correct team context for props from the `PropMatch.player_team` field (it's already populated via `nba_stats.py`)

### BUG-6 Relative Path for Elo Rest-Day Adjustment [P2]
**File:** `evmax/agents/models/elo_agent.py:92`
```python
FORM_STATE_PATH = Path("data/models/form_state.json")
```
This is a relative path. Every other state path in the codebase uses `Path(__file__).resolve().parents[N]`. When the process runs from a directory other than the project root, rest-day Elo adjustments silently fail.
- Change to: `FORM_STATE_PATH = Path(__file__).resolve().parents[3] / "data" / "models" / "form_state.json"`

### BUG-7 `EVGap.display_label` Checks Dead String "over_under" [P3]
**File:** `evmax/agents/odds/ev_gap_agent.py:90-92`
Checks `if self.market_type in ("over_under", "total")`. `MarketType` has no `over_under` member — only `total`. The string `"over_under"` is dead and will never match.
- Remove `"over_under"` from the tuple

### BUG-8 `datetime.utcnow()` Deprecation Not Fully Fixed [P2]
PLAN.md marks A1 as ✅ but these files were missed:
- `evmax/models/odds.py:48` — `fetched_at` default
- `evmax/models/bankroll.py:26` — `recorded_at` default
- `evmax/models/market.py:52` — `fetched_at` default
- `evmax/agents/models/form_agent.py:208` — `update()` event_date fallback
- `evmax/simulation/resolver.py:68` — `resolved_at` assignment

Replace all remaining `datetime.utcnow()` with `datetime.now(timezone.utc)`.

---

## Section 3 — Model Quality

### MODEL-1 Tennis Surface Detection is Too Fragile [P1]
**File:** `evmax/agents/models/tennis_model_agent.py`
Surface is inferred from keywords in `market.title`. In practice, Kalshi market titles for ATP/WTA are typically just "Player A vs Player B" with no tournament context. The surface defaults to `hard` almost always, meaning surface-specific Elo (clay/grass advantages) is rarely exercised despite existing in the state.

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

### TEST-1 TennisModelAgent Has Zero Tests [P1]
The entire tennis model agent — surface lookup, Elo computation, probability output — has no unit tests. This is the most important untested model given that the tennis prediction bug was recently discovered.
- Add `tests/test_tennis_model.py` covering: surface detection, player lookup, surface Elo vs overall Elo fallback, unknown player handling, non-tennis sector returns None

### TEST-2 PitcherModelAgent Has Zero Tests [P2]
No tests for pitcher ERA lookup, park factor application, or Pythagorean win probability.

### TEST-3 PinnacleGuestClient Has Zero Tests [P2]
**File:** `evmax/clients/esports_pinnacle.py`
This is the only live sharp odds provider and has no tests. At minimum, test the response parsing and devigging with a fixture.

### TEST-4 No Integration Test for Full Coordinator Cycle [P2]
There's no test that runs a full `coordinator.run_cycle()` against fixture data for even one sector. This would catch wiring bugs (like the tennis model not being called) before they hit production.

### TEST-5 Live Win Probability Model Untested [P3]
**File:** `evmax/models_ml/live_win_prob.py`
No tests for the live in-game model.

### TEST-6 Prop Probability Pipeline Untested [P2]
`nba_stats.py`, `nba_props_cache.py`, `prop_matcher.py`, and `prop_resolver.py` have no tests. These cover the end-to-end prop workflow.

---

## Section 5 — Architecture / Cleanup

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
See MODEL-2 above. NHL is in the registry and Kalshi series but Elo uses NBA defaults.

### SECTOR-2 NCAAW Elo Calibration [P2]  
See MODEL-2 above. NCAAW has a `REST_ELO_ADJ` entry but no K-factor or home advantage.

### SECTOR-3 Tennis Tournament Calendar for Surface Lookup [P1]
See MODEL-1 above. This is the highest-leverage model improvement for tennis.

### SECTOR-4 Sectors in Registry Not in CLAUDE.md [P3]
`nhl`, `baseball`, `valorant`, `ufc`, `f1` are in `registry.py` and `esports_pinnacle.py` but absent from CLAUDE.md "Key Sectors". Either document them or remove them if they're experimental.

---

## Priority Order

| Priority | Item | Impact |
|----------|------|--------|
| P1 | BUG-1 Tennis resolution | Bets never resolve |
| P1 | BUG-2 NHL resolution | Bets never resolve |
| P1 | BUG-3 UEL/MLS resolution | Bets never resolve |
| P1 | BUG-4 Poisson score matrix | Wrong probabilities for NBA/NFL |
| P1 | BUG-5 Prop injury boosts broken | Props underestimated |
| P1 | MODEL-1 Tennis surface detection | Model rarely used as intended |
| P1 | SECTOR-3 Tennis tournament calendar | Same as MODEL-1 |
| P1 | PROPS-1 NFL props backend | Dead API calls |
| P2 | BUG-6 Relative Elo path | Silent failures on non-root CWD |
| P2 | BUG-8 utcnow remaining | Deprecation warnings |
| P2 | MODEL-2 NCAAW/NHL calibration | Uncalibrated Elo |
| P2 | ARCH-1 Delete dead code | Confusion + maintenance |
| P2 | ARCH-2 Dual DB documentation | Schema drift confusion |
| P2 | ARCH-4 NFL props or remove | Wasted API calls |
| P2 | TEST-1 Tennis model tests | No coverage for critical model |
| P2 | TEST-4 Integration test | No end-to-end test |
| P3 | Everything else | Quality / polish |
