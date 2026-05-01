# Scheduled Runs

Inventory of remote Claude Code routines (cloud agents) that run on a cron
schedule for the evmax project. These routines run in Anthropic's cloud
sandbox — they have a fresh checkout of `tzeng7/evmax-sim` but **no access**
to your local `predictions.db` or `.env`.

Manage all routines (enable / disable / view logs / delete): <https://claude.ai/code/routines>

All times listed in **America/Los_Angeles** (PDT in summer, PST in winter).
The cron expressions stored in claude.ai are UTC; this doc shows the local
equivalent so you don't have to convert in your head. **DST drift:** when
clocks fall back in November, every routine fires 1 hour later in local
time than what's listed below.

---

## Active routines

### Auto-maintenance (no action needed unless something breaks)

| Routine | Cadence | Local time | What it does | Output |
|---|---|---|---|---|
| `mlb-fip-reseed` | Weekly | Mon 03:00 PDT | Pulls fresh BR pitcher stats via pybaseball, computes FIP from box-score components, writes `data/models/pitcher_state.json` | Auto-commits to `main` only when FIP/ERA values changed (no-op most weeks) |
| `mlb-calibration-refit` | Weekly | Mon 04:00 PDT | Re-fits isotonic calibration on combined 2024+2025 walk-forward predictions | Auto-commits `data/models/calibration.json` if PROMOTE; auto-reverts if Δ < +0.001 promotion bar |
| `Weekly Model Calibration` | Weekly | Sun 08:00 PDT | Re-seeds Elo/Form/Poisson via `seed_espn.py` for nba/ncaab/soccer/baseball/tennis, runs `evmax cleanup metrics`, runs `evmax cleanup adjust` to retune `sharp_weight` | Reports output in routine logs; auto-tunes `data/model_config.json` |
| `nightly-nba-proj` | Daily | 21:00 PT (previous-day kickoff) | Runs `evmax project slate --sector nba` for tonight's NBA games | Markdown table in routine log — read at <https://claude.ai/code/routines/trig_01Y5UuzMTmvd9E8GAhdNCFbR> |
| `Tennis ensemble calibration refresh` | Quarterly | 25th of Jan/Apr/Jul/Oct, 02:00 PT | Re-fits tennis isotonic calibration on prior year, validates on current | Opens PR if PROMOTE — review the Brier delta before merging |

### Requires you to read output and decide

| Routine | Cadence | Local time | What you do |
|---|---|---|---|
| `evmax-tennis-shadow-validation-weekly` | Weekly | Mon 09:00 PDT | Run on your laptop: `evmax cleanup shadow metrics --category tennis --days 30` and `evmax cleanup shadow show --category tennis --days 7`. Promote tennis back to live (`evmax cleanup shadow promote tennis`) if **all three** hold: ≥50 resolved shadow bets in the rolling 30d window, positive ROI vs implied price, Brier ≤ 0.20. Otherwise let it run another week. |

### One-shot (fires once then auto-disables)

| Routine | Fires | What it does |
|---|---|---|
| `mlb-2week-checkup` | **Fri May 15 2026, 08:00 PDT** | Re-runs `fit_baseball_calibration.py` to verify the pipeline hasn't drifted, audits FIP-reseed commit history, commits a `docs/mlb_2week_checkup_2026-05-15.md` report with go/no-go criteria + a local-commands checklist. Read the report after `git pull` and run the three local commands it lists. |
| `evmax: WNBA spread path validation (post-launch)` | **Thu May 21 2026, 09:00 PDT** | Validates WNBA spread/total handling 5 days into the 2026 season opener. Probes Kalshi for KXWNBASPREAD/KXWNBATOTAL tickers, runs the WNBA test suite, opens a GitHub issue summarizing findings. Address the issue before WNBA volume ramps up. |

---

## Conventions

- **Pull pattern.** Most auto-maintenance routines commit directly to `main`. To pick up changes locally: `git pull` from `/Users/ktzeng/Projects/evmax` whenever convenient.
- **No-op weeks.** Both `mlb-fip-reseed` and `mlb-calibration-refit` are designed to no-op when underlying data hasn't changed (BR returned identical pitcher lines, calibration didn't beat promotion bar). No-ops still consume compute but leave state untouched. This is normal, not a failure.
- **Commit messages from routines.** All committed by `Claude Sonnet 4.6` co-author tag. Search `git log --grep="MLB:"` to find their changes.
- **Failure handling.** Routines self-protect — bad calibration fits revert, BR scrapes that fail leave state intact, ESPN 5xx errors are retried once. A single failed run is rarely catastrophic; check the routines page if you suspect drift.

---

## Calendar — next 4 weeks

| Date (PT) | What |
|---|---|
| Sun May 3, 08:00 | Weekly Model Calibration runs |
| Mon May 4, 03:00 | MLB FIP reseed |
| Mon May 4, 04:00 | MLB calibration refit |
| Mon May 4, 09:00 | Tennis shadow validation reminder |
| Daily 21:00 | NBA slate projection |
| Sat May 16 | WNBA season opens — watch for Kalshi `KXWNBASPREAD` / `KXWNBATOTAL` markets to appear |
| **Fri May 15, 08:00** | MLB 2-week checkup report |
| **Thu May 21, 09:00** | WNBA spread launch validation |
| Mon May 11, 18, 25 | Tennis shadow validation reminder (repeats weekly) |

---

## Cleanup TODO

Three disabled-but-not-deleted "Weekly Model Calibration" duplicates from
April 2026 are still listed in the routines page. Claude Code's API can't
delete them — go to <https://claude.ai/code/routines> and delete manually:

- `trig_01JckAPJemdrBjsajqMa9yFb` (disabled 2026-04-14)
- `trig_01R1ze4noxvaWmcGBCWHPTtS` (disabled 2026-04-14)
- `trig_01M4a5sG5axrRhp5XcbWLvUX` (disabled 2026-04-14)

Also consider whether these still earn their slots:

- `Weekly Model Calibration` (`trig_01KiEHHf4V2SYsQhXBMKrq5o`) — overlaps with `mlb-fip-reseed` for baseball. Useful for non-MLB sectors (NBA/NCAAB/soccer/tennis) but worth pruning if any of those sectors get their own dedicated routines.
- `evmax-tennis-shadow-validation-weekly` — once you make the promote/no-promote call, disable this routine. Indefinite weekly reminders accumulate noise.

---

## Routine IDs (for `gh` / API access)

| Name | ID |
|---|---|
| mlb-fip-reseed | `trig_01XYavtVvRjGhZfqERtVXHQb` |
| mlb-calibration-refit | `trig_01U8USfUfcBdphqEo5aS6e7j` |
| mlb-2week-checkup | `trig_01QbUNkeBH1DZZzwRpowFL4N` |
| evmax-tennis-shadow-validation-weekly | `trig_01PWZ6wnxEAww9bynuVU2yHh` |
| evmax: WNBA spread path validation (post-launch) | `trig_013p6bDz7cJT5j1RiPKSfQWT` |
| nightly-nba-proj | `trig_01Y5UuzMTmvd9E8GAhdNCFbR` |
| Tennis ensemble calibration refresh | `trig_01LDTfQimGCY3SNxS8jT3AUM` |
| Weekly Model Calibration (active) | `trig_01KiEHHf4V2SYsQhXBMKrq5o` |
