# weekly-clv-backfill — spec

Claude scheduled task (created 2026-09-20). Runs the two CLV backfills that no
other schedule already covers.

| Field | Value |
|---|---|
| Task id | `weekly-clv-backfill` |
| Schedule | Mon 10:00 PT (`0 10 * * 1`) — after the crowded Mon 07:04–09:01 maintenance cluster |
| Checkout | MAIN (`/Users/ktzeng/Projects/evmax`) — never a `.claude/worktrees/*` copy (those carry near-empty DBs) |
| Git | none — both backfills write only gitignored SQLite DBs, so there is no branch/commit/PR and no `sched_worktree.py` |

## Why this task exists — and what is deliberately NOT in it

"CLV backfill" is three distinct mechanisms. Two are already scheduled; this
task owns the two gaps that were not.

| Mechanism | What it writes | Already scheduled? |
|---|---|---|
| `backfill_clv()` (`evmax/agents/cleanup/resolver.py`) — core: `ev_predictions.pinnacle_drift_pct` / `kalshi_clv_pct`, `ev_outcomes.pinnacle_close_prob` | `predictions.db` | ✅ runs at the end of every `cleanup resolve` (daily-resolve 07:32, daily-evening-resolve 23:06). **Do not re-run here.** |
| WNBA spread/total candle backfill | `archive.db` | ✅ Mondays inside `weekly-wnba-total-anchored-backfill-check`. **Do not re-run WNBA here.** |
| **`backfill_outcome_closes.py`** — repairs orphaned `ev_outcomes` closes (the `ev_predictions INNER JOIN ev_outcomes` blind spot: outcomes whose prediction partner was pruned by maintenance keep `pinnacle_close_prob = NULL`) | `predictions.db` | ❌ → **this task**, all sectors |
| **`backfill_kalshi_candles.py --sector nfl`** — reconstructs NFL spread/total candle trails for the anchored-entry / listings-eval CLV lens (`--entry-sectors wnba,nfl`) | `archive.db` (repointed to the main checkout, script L43–46) | ❌ → **this task** |

At creation (2026-09-20 `--dry-run`), `backfill_outcome_closes` had **467
orphaned outcome rows** pending across all sectors — the concrete backlog that
justifies automating it.

## Not the worktree/rolling-PR pattern — on purpose

The request that created this task framed it as the `sched_worktree.py` +
rolling `bot/model-state` PR pattern. That pattern isolates git working-tree
collisions and produces mergeable PRs from **checked-in state files**
(`data/models/*.json`). Both backfills here commit nothing — `predictions.db`
and `archive.db` are gitignored (`.gitignore` `*.db`), and neither script runs
any git command — so `ship`/`--rolling` would stage zero owned files and push an
empty PR. A plain in-place run in the main checkout (as the resolve tasks and
the `watch-*` launchd agents do for their DB writes) is the correct vehicle.

## Task body

Run from the main checkout with `.venv/bin/python` (no bare `python` on PATH).

1. `.venv/bin/python scripts/backfill_outcome_closes.py --dry-run` then
   `.venv/bin/python scripts/backfill_outcome_closes.py` (idempotent, fills
   NULLs only). Record `filled=` / `no_archive_close=`.
2. `.venv/bin/python scripts/backfill_kalshi_candles.py --sector nfl`
   (defaults to `--market-types spread,total`; idempotent, skips
   already-backfilled tickers unless `--refresh`; grows through the season).
   Record tickers-backfilled / snapshots-written. To widen coverage to another
   laddered sector that later enters the anchored-entry pipeline, add another
   `--sector <name>` run — but never `--sector wnba` (owned elsewhere).
3. Stay quiet on a normal successful run (routine maintenance). Send a
   proactive notification ONLY on a failure: a missing/renamed script, a script
   error, or a wholesale Kalshi-fetch failure (0 snapshots when tickers were
   pending). Keep running weekly indefinitely.
