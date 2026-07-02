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

---

## Claude scheduled tasks

### Daily

| Task | Local time | What it does |
|---|---|---|
| `daily-morning-scan` | 01:01 | `evmax agents scan --bankroll 500 --kelly 0.5 --date TODAY` — morning watchlist (night-before edges usually revert by tip-off) |
| `daily-resolve-and-model-update` | 07:32 | `evmax update scores` + `cleanup resolve --date YESTERDAY` + `archive resolve --date YESTERDAY` — feeds ESPN finals into Elo/Form/Poisson/xG state and resolves outcomes |
| `daily-close-lines-capture` | 08:04 | `evmax cleanup close-lines` — snapshots Pinnacle closing lines pre-tipoff (timing-sensitive; `watch-closes` under launchd covers this even when the app is closed) |
| `daily-updated-scan` | 09:05 | re-runs the scan against refreshed lines (the 1am scan often predates Kalshi listing next-day markets) |
| `daily-evening-resolve` | 23:06 | `evmax cleanup resolve --date TODAY` + `cleanup show --date TODAY` — day's P/L |

### Weekly

| Task | Local time | What it does |
|---|---|---|
| `weekly-seasonal-model-reseed` | Mon 07:04 | Season-aware reseed of the manual-seed states: WNBA efficiency (May–Oct), NFL efficiency + QB Elo (Sep–Feb), MLB pitcher FIP (Mar–Nov) |
| `weekly-drift-audit` | Mon 07:56 | This audit — doc↔code drift, SAFE fixes on a branch + PR (spec: `.claude/commands/drift-audit.md`) |
| `weekly-model-calibration` | Mon 08:07 | `cleanup metrics --weeks 4` + `cleanup adjust` (sharp_weight auto-tune, bounded 0.40–0.95) + shadow metrics review (reports promotion readiness, never promotes) |
| `weekly-value-audit` | Mon 08:22 | Model-blend VALUE audit — Brier vs sharp/close + CLV per sector, model-side fixes only, propose-only PR (spec: `.claude/commands/value-audit.md`) |
| `weekly-tennis-surface-elo-refresh` | Tue 07:25 | Refreshes ALL six tennis models from Tennis Abstract: surface Elo (leaderboards) + serve/return, advanced, form, h2h, ranking_trend (matchmx) |

### Disabled (kept for reference)

| Task | Status |
|---|---|
| `weekly-nba-props-shadow-metrics` | Disabled 2026-07-01 per user request; re-enable when NBA season restarts if nba_props promotion tracking resumes |
| `weekly-tennis-rankings-refresh` | Deprecated 2026-06-27 (Sackmann repos offline); ranking_trend now rides the Tue tennis task. Still present as a disabled task — safe to delete the task folder |

---

## launchd agents

| Agent | Cadence | What it does |
|---|---|---|
| `com.evmax.watch-closes` | every 300 s, always-up | `evmax cleanup watch-closes` — near-tip Kalshi + Pinnacle close capture so placed-bet CLV has a genuine post-entry anchor |
| `com.evmax.watch-listings` | hourly, always-up | `evmax cleanup watch-listings --interval 3600` — LISTING→scan window capture (all game sectors, spread+total ladders): Kalshi snapshots + order-book depth + as-of Pinnacle anchor. Logs: `logs/launchd.watch-listings.{out,err}`. After code changes restart with `launchctl kickstart -k gui/$(id -u)/com.evmax.watch-listings` (KeepAlive keeps the old process otherwise) |
| `com.evmax.nba-resolve` | — | STOPPED (plist renamed `.plist.disabled`) — was a no-op in the NBA offseason |

---

## Still manual by design

- `scripts/wnba_offseason_regress.py` — needs a human-reviewed roster YAML; run once before each WNBA season opener.
- All shadow → live promotions (`evmax cleanup shadow promote <category>`) — a human decision, never automated.

---

## History

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
