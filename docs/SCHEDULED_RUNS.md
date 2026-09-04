# Scheduled Runs

Inventory of everything that runs on a schedule for evmax. Since the
2026-06-28 crontab teardown (`crontab -r`; backups in `/tmp/evmax-cron*`),
scheduling lives in exactly **two places** — there is no user crontab, and
the claude.ai cloud routines this doc used to describe have been superseded
by local scheduled tasks (see History below):

1. **Claude Code scheduled tasks** (`~/.claude/scheduled-tasks/<task-id>/SKILL.md`)
   — run only while the Claude desktop app is open; missed runs fire on next
   launch. Manage via the app or the `scheduled-tasks` MCP tools.
2. **launchd** (`~/Library/LaunchAgents/com.evmax.*.plist`) — robust,
   app-independent, for the always-up watchers.

All times America/Los_Angeles. Each task carries a small random jitter.

**PR-within-the-run policy:** any scheduled run that creates code changes and commits
(today: `weekly-drift-audit` and `biweekly-model-improve-graph` — every other task is
forbidden from editing code) must finish the job inside the same run: work off the **live
remote tip** (`origin/main`) in an **isolated git worktree**, `git push` the branch, then
`gh pr create` with explicit `--title`/`--body` (a bare `gh pr create` prompts interactively
and hangs a headless run). If `gh pr create` fails, the push already preserved the work — the
run reports the branch + compare URL so the PR can be opened manually. Commits must never end
a run stranded local-only, and no scheduled run ever merges its own PR.

### Git isolation for state/code-altering tasks (`scripts/sched_worktree.py`)

Several tasks mutate git-tracked artifacts — model-state tasks rewrite
`data/models/*.json` / `data/model_config.json`; the two code tasks edit `evmax/` + docs.
They all run against the **one shared checkout** at `/Users/ktzeng/Projects/evmax`. Two
failure modes used to follow, and this is how they are now prevented:

1. **Working-tree collision.** A checkout has one checked-out branch and one working tree.
   When two tasks overlapped, task A's `git switch main` / `git pull` yanked the tree out
   from under task B mid-run, silently discarding uncommitted state (the 2026-07-21
   collision; the 2026-08-20 pre-commit stash-timeout loss).
2. **Un-mergeable PRs.** The state artifacts are whole-file regenerated JSON — two PRs that
   both rewrite `elo_state.json` cannot three-way-merge, and each branched off a possibly
   stale *local* main. Concurrent PRs deadlock on branch-protection "require up-to-date"
   → reads as "won't pass CI/CD". (The CI test job itself does not fail on a JSON reseed;
   the block is PR *mergeability*.)

**The fix** is `scripts/sched_worktree.py`, called by every state/code task instead of
switching branches in the shared checkout:

- `open` — `git fetch origin`, then create an **isolated worktree** whose branch is cut from
  `origin/main` (the live remote tip), never local main. Worktrees share the object store
  (cheap) but have independent working trees + HEADs, so parallel runs can never fight over
  one checked-out branch. Prints the worktree path.
- `ship` — stage **only** the owned files (never `git add -A`), commit (`--no-verify` by
  default so a slow pre-commit stash cannot swallow another task's tree; `--run-hooks` for
  code tasks), push, ensure exactly one open PR, remove the worktree.

**State tasks pass `--rolling`:** they all accumulate onto ONE long-lived branch
(`bot/model-state`) so there is only ever a single open state PR. Each task owns a **disjoint**
set of files and the branch is always re-anchored to `origin/main`, so a merged PR leaves the
branch clean (next run rebuilds from live main) and an unmerged PR is preserved (the new
task's owned files are added on top). Regenerated state is idempotent, so on any rebase
trouble the safe fallback is to re-anchor to `origin/main` and let the seed script reproduce
the current-truth file. **Code tasks omit `--rolling`** → a fresh dated branch off
`origin/main` each run (their diffs are real code that must pass review + CI on its own).

The shared checkout now **stays permanently on a clean `main`** — all work happens in
worktrees — so no task ever runs `git switch` / `git reset --hard` / `git checkout -- .` /
`git clean` there. Canonical shape a task's SYNC/SHIP steps reduce to:

```bash
REPO=/Users/ktzeng/Projects/evmax
# STEP 0 — SYNC: refresh the helper from the live remote (safe on clean main),
# then open an isolated worktree on origin/main. No branch switch, no reset.
git -C "$REPO" pull --ff-only --quiet          # keeps scripts/sched_worktree.py current
WT=$(python "$REPO/scripts/sched_worktree.py" open --rolling \
        --branch bot/model-state --print-path)
cd "$WT"
# ... run this task's seed scripts here; they rewrite ONLY this task's owned files in $WT ...

# STEP N — SHIP: stage only the owned files, single rolling PR, remove worktree.
python "$REPO/scripts/sched_worktree.py" ship --rolling \
    --branch bot/model-state --worktree "$WT" \
    --title "chore(models): <what was reseeded>" \
    --body  "Automated reseed." \
    -- data/models/<owned-file-a>.json data/models/<owned-file-b>.json
```

**Rollout status:** the helper + its tests landed first (this is a repo file, so it must be
on `main` before any task calls it). The per-task SKILL.md SYNC/SHIP blocks are converted to
the shape above once the helper is merged — do NOT flip a task config before merge, or the
task cannot find the helper. Tasks to convert: `weekly-seasonal-model-reseed`,
`weekly-tennis-surface-elo-refresh`, `daily-resolve-and-model-update`, `daily-evening-resolve`,
`weekly-model-calibration` (all `--rolling`), plus `weekly-drift-audit` and the
`biweekly-model-improve-graph` graph (dated branches, `--run-hooks`).

---

## Claude scheduled tasks

### Daily

| Task | Local time | What it does |
|---|---|---|
| `daily-resolve-and-model-update` | 07:32 | `evmax update scores` + `cleanup resolve --date YESTERDAY` + `archive resolve --date YESTERDAY` — feeds ESPN finals into Elo/Form/Poisson/xG state and resolves outcomes |
| `ev-scan-light-midday` | 07:04 | Light scan #1 (10:04 ET): tonight's full US slate is listed AND sharp-anchored by now (Kalshi lists ~24h out, Pinnacle posts T-17–24h). Spec: `docs/scheduled-tasks/ev-scan-light.md` |
| `ev-scan-light-afternoon` | 13:04 | Light scan #2 (16:04 ET): T-3h re-scan for the 19:00 ET wave — MLB lineups post, injuries land, stale morning edges re-checked |
| `ev-scan-light-evening` | 15:34 | Light scan #3 (18:34 ET): T-0.5–2h from the main slate — the only entry window with demonstrated +CLV; these are the rows the fresh-close CLV gates score |
| `ev-scan-nfl-sunday-early` | Sun 09:15 (NFL season only, 09-04→02-15; self-skips otherwise) | Same light-scan body, added 2026-09-02: NFL 1pm-ET Sunday games kick at 10:00 PT, so their T-1h pick window (09:00–09:45 PT) fell between the 07:04 and 13:04 scans. The 4:25pm-ET wave (13:04 = T-21m) and SNF/MNF/TNF (15:34) were already covered. Spec: `docs/scheduled-tasks/ev-scan-light.md` |
| `daily-evening-resolve` | 23:06 | `evmax cleanup resolve --date TODAY` + `cleanup show --date TODAY` — day's P/L |
| `prune-stale-open-positions` | 11:00–23:00, every 2h (`0 11,13,15,17,19,21,23 * * *`) | `evmax cleanup prune-stale --lookahead 12 --once` — re-prices every un-picked LIVE candidate (`placed=0`) against the current venue ask + a fresh Pinnacle re-blend (the same engine `agents pick --live` uses) and VOIDs (`void_reason='stale_reverted'`) any whose edge reverted below the flag threshold, so the dashboard "Open Positions" list + `cleanup show` stop showing phantom edges the live line already erased. Un-voids its own rows on a bounce-back (hysteresis). Fail-safe: never voids on a missing/empty-book quote or an in-play game, and only touches `placed=0` live rows. Writes only the gitignored `predictions.db` — no git/PR step (forbidden from editing code) |

The three `ev-scan-light-*` tasks are the **2026-07-19 replacement** for the removed 90-min pair
(`ev-scan-90min-on-hour`/`half-hour`, removed 2026-07-18): 3×/day, evening-weighted, enough
shadow-sample volume for the sector promotion gates without the old churn. They share one spec
(`docs/scheduled-tasks/ev-scan-light.md`) and are **forbidden from editing code** (the
PR-within-the-run policy covers only the two audit tasks). **Activated 2026-07-19** after the
soccer sharp-only guard (MIN_NONSHARP_MODELS, PR #117) merged — a scheduled scan running before
that guard would have logged sharp-passthrough MLS rows as live plays 3×/day. `daily-morning-scan`/
`daily-updated-scan` remain disabled below (their functionality is now covered by the light scans).

### Weekly

| Task | Local time | What it does |
|---|---|---|
| `weekly-seasonal-model-reseed` | Mon 07:04 | Season-aware reseed of the manual-seed states: WNBA efficiency (May–Oct), NFL efficiency + QB Elo (Sep–Feb), NCAAF efficiency EPA (Aug–Jan), MLB pitcher_v2 (xERA+bullpen+FIP, Mar–Nov), UFC Glicko-2 ratings (weekly, no offseason) |
| `weekly-tennis-surface-elo-refresh` | Mon 07:15 | Refreshes ALL six tennis models from Tennis Abstract: surface Elo (leaderboards) + serve/return, advanced, form, h2h, ranking_trend (matchmx) |
| `weekly-drift-audit` | Mon 07:56 | This audit — doc↔code drift, SAFE fixes on a branch + PR (spec: `.claude/commands/drift-audit.md`) |
| `weekly-model-calibration` | Mon 08:07 | `cleanup metrics --weeks 4` + `cleanup adjust` (sharp_weight auto-tune, bounded 0.40–0.95) + shadow metrics review (reports promotion readiness, never promotes) |
| `weekly-calibration-tripwire` | Mon 08:34 | Read-only alert: fires only when a sector's blend shows a real, consistent calibration bias (n≥30, one-signed, ≥4pp) and names the `cleanup recalibrate` fix |
| `weekly-wnba-listings-robustness-check` | Mon 08:35 | Checks whether the WNBA spread lay anchored-entry live shadow gate (n≥30, mean CLV≥0, %pos≥55%) has cleared for promotion (read-only tripwire; no promotion) |
| `biweekly-model-improve-graph` | Mon + Thu 08:51 | Runs the versioned model-improve Workflow graph: value-audit gap → ONE model-side change → walk-forward + integrity/signal gates → PR or revert. Propose-only, never merges (graph: `.claude/workflows/model-improve.js`; ledger: `.claude/improvement-ledger.jsonl`) |
| `weekly-wnba-total-anchored-backfill-check` | Mon 09:01 | Watches the WNBA total over/under anchored-entry backfill sample for a PROMOTE or KILL verdict (read-only) |

### Disabled (kept for reference)

| Task | Status |
|---|---|
| `daily-morning-scan` | Disabled 2026-07-02 — the fixed 01:01 scan was replaced by the interleaved `ev-scan-90min-*` pair (rolling 90-min scan + push-notify), which was itself removed (see below) |
| `daily-updated-scan` | Disabled 2026-07-02 — the fixed 09:05 re-scan was likewise folded into the `ev-scan-90min-*` pair, which was itself removed (see below) |
| `ev-scan-90min-on-hour` + `ev-scan-90min-half-hour` | Removed (confirmed by user 2026-07-18) — was the rolling ~90-min `evmax agents scan` + PushNotification pair that replaced the two fixed daily scans above; no automated scan currently runs, see the note under Daily |
| `weekly-nba-props-shadow-metrics` | Disabled 2026-07-01 per user request; re-enable when NBA season restarts if nba_props promotion tracking resumes |
| `weekly-value-audit` | Disabled 2026-08-03 — SUPERSEDED by `biweekly-model-improve-graph` (the versioned Workflow graph subsumes the audit → one-change → gate → PR loop). Paused rather than deleted so it can be restored if the graph is rolled back; `.claude/commands/value-audit.md` is still the spec the graph's audit stage reads |
| `weekly-tennis-rankings-refresh` | Deprecated 2026-06-27 (Sackmann repos offline); ranking_trend now rides the Mon tennis task. Task DELETED — deregistered and its `~/.claude/scheduled-tasks/` folder removed (confirmed gone 2026-08-03) |

---

## launchd agents

| Agent | Cadence | What it does |
|---|---|---|
| `com.evmax.watch-closes` | `StartInterval` 300 s → `--once` per firing | `evmax cleanup watch-closes --lookahead 40 --once` — near-tip Kalshi + Pinnacle close capture so placed-bet CLV has a genuine post-entry anchor. `--lookahead` bumped 30 → 40 on 2026-07-12 so the T-30 close-target window has qualifying snapshots (`get_kalshi_close_price` selects the latest snapshot at or before tipoff−30min). Converted from an always-up loop 2026-07-05: launchd coalesces firings missed during sleep into one run on wake |
| `com.evmax.watch-listings` | `StartCalendarInterval` Minute=0 (hourly) → `--once` per firing | `evmax cleanup watch-listings --once --log-entries --entry-sectors wnba,baseball` — LISTING→scan window capture (all game sectors, spread+total ladders): Kalshi snapshots + order-book depth + as-of Pinnacle anchor. Logs: `logs/launchd.watch-listings.{out,err}`. Converted from an always-up `--interval 3600` loop 2026-07-05: the in-process sleep froze across lid-close/forced sleep (1–17 sweeps/day vs 24 in archive.db), while calendar firings coalesce on wake. Each firing is a fresh process, so code changes are picked up automatically (no `launchctl kickstart` needed) and the cross-loop AsyncLimiter RuntimeWarning is gone. `caffeinate -i` is now per-invocation only |
| `com.evmax.heartbeat` | `StartCalendarInterval` 09:00 + 21:00 daily | `evmax cleanup heartbeat --check-pinnacle --notify` — pipeline liveness alert; surfaces a down Pinnacle anchor (the sole sharp source, which fails CLEAR rather than serving stale prices) as a critical notification. Logs: `logs/launchd.heartbeat.{out,err}` |
| `com.evmax.clv-monitor` | `StartCalendarInterval` Mon 09:20 | `evmax cleanup clv-monitor --notify` — weekly CLV tripwire on the live streams. Logs: `logs/launchd.clv-monitor.{out,err}` |
| `com.evmax.nba-resolve` | — | STOPPED (plist renamed `.plist.disabled`) — was a no-op in the NBA offseason |

---

## Still manual by design

- `scripts/wnba_offseason_regress.py` — needs a human-reviewed roster YAML; run once before each WNBA season opener.
- All shadow → live promotions (`evmax cleanup shadow promote <category>`) — a human decision, never automated.

---

## History

- **2026-08-28** the three `ev-scan-light-*` tasks switched from a hardcoded
  `--bankroll 500` to `--bankroll-venue kalshi --bankroll 500`: they now size
  Kelly against the **live Kalshi balance** (cash + open-position value, `GET
  /portfolio/balance`), with the `--bankroll 500` retained only as the
  fail-soft fallback if the authenticated balance call fails. `--bankroll-venue
  kalshi` also scopes live plays to Kalshi; PolyUS was effectively unfunded
  ($0.73 vs Kalshi $1,575.96 at the time), so the scoping is intended and
  PolyUS rows simply log as shadow. Shared spec updated
  (`docs/scheduled-tasks/ev-scan-light.md`); the SKILL.md prompts for
  `ev-scan-light-{midday,afternoon,evening}` were updated via the
  `scheduled-tasks` MCP tools.
- **2026-08-08** `evmax cleanup prune-stale` added — dynamic stale-candidate pruner (voids
  un-picked live candidates whose edge reverted below the flag threshold; un-voids on recovery).
  Scheduled as the Claude task `prune-stale-open-positions` (every 2h, 11:00–23:00 PT, listed under
  Daily above) rather than a launchd agent — it only writes the gitignored `predictions.db`, so it
  needs no always-up daemon, just the periodic CLI run. It becomes functional once the feature branch
  (`claude/stale-positions-cleanup-bbec57`) merges to `main` (the task runs from the shared main
  checkout). Preview any time with `evmax cleanup prune-stale --once --dry-run`.
- **2026-08-03** (weekly drift audit) three enabled tasks were missing from this doc and are
  now listed: `weekly-calibration-tripwire`, `biweekly-model-improve-graph`, and
  `weekly-wnba-total-anchored-backfill-check`. `weekly-value-audit` moved to Disabled (superseded
  by the model-improve graph, so the PR-within-the-run policy now names that graph instead), the
  `weekly-tennis-rankings-refresh` row was corrected (the task folder is gone, not merely
  disabled), `weekly-seasonal-model-reseed` gained the NCAAF EPA reseed it has covered since
  2026-08-01, and `daily-close-lines-capture` was annotated as a structural no-op.
- **≤ 2026-06** this doc described claude.ai cloud routines (`mlb-fip-reseed`,
  `mlb-calibration-refit`, `Weekly Model Calibration`, `nightly-nba-proj`,
  tennis shadow validation, one-shots). Their duties were absorbed by the
  local tasks above (calibration/reseed/resolve automated locally 2026-06-10).
  If any cloud routine is still enabled at <https://claude.ai/code/routines>,
  it is redundant — disable/delete it there (that page is not manageable from
  the CLI).
- **2026-06-28** user crontab fully removed. The `*/5 watch-closes --once`
  cron line was a duplicate of the launchd agent and was dropped; the 1am scan
  cron had been silently broken (no `cd` into the repo).
- **2026-07-01** `com.evmax.watch-listings` launchd agent added (hourly,
  all-sector defaults from PR #65); `weekly-nba-props-shadow-metrics` disabled.
- **2026-07-02** the two fixed daily scans (`daily-morning-scan` 01:01 and
  `daily-updated-scan` 09:05) were disabled and replaced by two interleaved
  3-hourly tasks — `ev-scan-90min-on-hour` (`:00` on 0/3/6/9/12/15/18/21) and
  `ev-scan-90min-half-hour` (`:30` on 1/4/7/10/13/16/19/22) — for a rolling
  ~90-min `evmax agents scan` + PushNotification cadence through the day instead
  of two fixed morning runs.
- **2026-07-18** `ev-scan-90min-on-hour`/`ev-scan-90min-half-hour` removed (confirmed
  intentional by user during the weekly drift audit — the SKILL.md folders were left on
  disk but deregistered from the scheduler). No automated `evmax agents scan` currently
  runs; the two fixed scans they replaced remain disabled.
- **2026-07-05** both watchers converted from KeepAlive always-up loops
  (`caffeinate -i` + in-process `time.sleep`) to launchd-driven `--once`
  firings (`StartCalendarInterval` hourly for watch-listings, `StartInterval`
  300 s for watch-closes). The old loops froze across lid-close/forced sleep
  — `caffeinate -i` only blocks *idle* sleep — so archive.db showed 1–17
  listing sweeps/day instead of 24, starving the WNBA spread lay-side CLV
  gate of listing-window density. launchd coalesces sleep-missed firings
  into one run at wake, and each sweep being a fresh process also fixed the
  recurring AsyncLimiter cross-event-loop RuntimeWarning.
- **2026-08-07** `daily-close-lines-capture` retired (scheduled task deleted)
  and the underlying `evmax cleanup close-lines` CLI command removed. The
  command was a permanent no-op: it queried `ev_outcomes WHERE outcome IS NULL
  AND pinnacle_close_prob IS NULL`, but the only path that INSERTs an
  `ev_outcomes` row is `_write_outcome` (resolver.py), always called with a
  concrete `outcome`. `ev_outcomes` is a resolution table — a row exists only
  after a market resolves — so the pre-tipoff query matched 0 rows and the
  task printed "No unresolved markets without closing lines" every day
  (verified 0-match on 2026-07-19/07-30/08-01/08-03, and 0 of 3946 rows on
  08-07). `ev_outcomes.pinnacle_close_prob` is already populated from genuine
  near-tip data by `watch-closes` (`com.evmax.watch-closes`, near-tip Pinnacle
  write) and `backfill_clv` (`resolver.py`, post-resolution write), so no
  close capture was lost. A repoint to `ev_predictions` was rejected: the
  08:04 slot would snapshot a stale morning line, not a close, and moving it
  near tip would duplicate `watch-closes`.
