# ev-scan-light — shared spec for the 3×/day light scan tasks

The approved (2026-07-19) replacement for the removed 90-min scan pair.
Three Claude scheduled tasks share this body; they differ only in their slot:

| Task id | PT | ET | Slot rationale |
|---|---|---|---|
| `ev-scan-light-midday` | 07:04 | 10:04 | Tonight's full US slate is listed AND sharp-anchored (Kalshi lists ~24h out; Pinnacle posts T-17–24h for US sports — see project_pinnacle_posting_windows). The volume workhorse: MLB day+night, WNBA, MLS, in-progress Euro tennis, weekend Euro soccer listed for tomorrow (from Aug). |
| `ev-scan-light-afternoon` | 13:04 | 16:04 | T-3h re-scan for the 19:00 ET wave: MLB lineups post ~3-4h pre-game (feeds the pitcher_v2 pen/lineup signals), injury news lands, stale morning edges get re-checked. |
| `ev-scan-light-evening` | 15:34 | 18:34 | T-0.5–2h from the main MLB/WNBA slate and T-1–4h from MLS. The only entry window with demonstrated positive CLV (see the Placed-bet CLV notes) — these rows are what the fresh-close CLV promotion gates actually score. |

## ⚠️ Activation gate

Create these tasks (via the `scheduled-tasks` MCP tools) only AFTER the
soccer sharp-only guard (`MIN_NONSHARP_MODELS` in `ev_gap_agent.py`) is on
`main`. Before it, a scheduled scan logs sharp-passthrough MLS rows as LIVE
plays three times a day — the exact failure the guard exists to stop.

## Task body (identical for all three)

1. `cd ~/Projects/evmax`
2. Run: `uv run evmax agents scan --bankroll-venue kalshi --bankroll 500 --kelly 0.5 --date TODAY`
   (default in-season sectors; the mode registry + full-blend/sharp-only
   guards apply automatically at persistence — shadow sectors log shadow,
   disabled market types drop). `--bankroll-venue kalshi` sizes Kelly against
   the **live Kalshi balance** (cash + open-position value, `GET
   /portfolio/balance`) and scopes live plays to Kalshi; the `--bankroll 500`
   is the fail-soft fallback used only if the authenticated balance call fails
   (no creds / network), so real money is never sized against a fabricated
   figure. PolyUS is effectively unfunded, so Kalshi-only scoping is intended —
   any PolyUS rows log as shadow. The scan echoes which bankroll it used.
3. Report, in the task summary:
   - the bankroll used (live Kalshi balance vs the $500 fallback)
   - gaps found per sector, split live/shadow (the scan output's mode counts)
   - top 3 live plays by EV (Event · Outcome · EV% · Kelly)
   - any `prediction_demoted_partial_blend` / no-pitcher-skip counts (these
     are the guards working, not errors)
4. If any LIVE play has EV ≥ 5%, send a push notification with the play list
   (same pattern as the removed 90-min pair).
5. **Never edit code, commit, or open PRs from this task** (the
   PR-within-the-run policy covers only the two audit tasks).

## Why 3×/day and these slots

Every sector promotion path (soccer, baseball v2, tennis) needs shadow-sample
volume: the gates want n≥30 clean resolved rows and CLV %pos≥55 on genuine
near-tip closes. `watch-closes` (launchd, 5-min) already snapshots closes for
live AND shadow rows — each scan multiplies the rows that get a genuine close
anchor. Two of the three runs sit in the pre-evening window (evening-weighted
per the user's decision); no 4th run — weekend Euro-morning kickoffs are
pre-listed 24h out, so the Fri/Sat runs cover them.
