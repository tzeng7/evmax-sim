---
description: Generate or resolve NBA point projections for a date, then render a unified plays table. Usage: /nba-proj [YYYY-MM-DD]
argument-hint: "[YYYY-MM-DD]"
---

Run the NBA standalone projection workflow for the date in `$ARGUMENTS` (default: today if empty).

## Steps

1. **Parse the date argument.**
   - `$ARGUMENTS` contains the user-supplied text (possibly empty).
   - If empty or whitespace → use today's date in `YYYY-MM-DD`.
   - Otherwise validate it matches `YYYY-MM-DD`. On mismatch, tell the user the expected format and stop.
   - Compare to today.

2. **Run the right `evmax project` command.** Use Bash with `.venv/bin/evmax`:
   - If the target date is **today or later** → `.venv/bin/evmax project slate --sector nba --log`
   - If the target date is **in the past** → `.venv/bin/evmax project resolve --date <YYYY-MM-DD> --sector nba`
   - Both commands are idempotent, so always run — never skip based on DB state.
   - If the command fails, surface stderr and stop before step 3.

3. **Query `data/projections.db` and render one unified table.** Use `.venv/bin/python` with the template below. Pull every row where `sector='nba'` and `game_date=<target>`.

   If zero rows → print `No NBA projections logged for <date>.` and stop (no empty table).

   Otherwise, render a GitHub-flavored markdown table **in the assistant response** (not stdout) with these columns:

   | Column | Source |
   |---|---|
   | Matchup | `{away_team} @ {home_team}` (shorten each team to the last word) |
   | Proj Score | `{away_last} {proj_away:.0f}–{proj_home:.0f} {home_last}` |
   | Actual Score | `{away_last} {actual_away:.0f}–{actual_home:.0f} {home_last}` or `—` if unresolved |
   | Proj Spread | `proj_spread` with sign |
   | Actual Spread | `actual_away − actual_home` or `—` |
   | Proj Total | `proj_total` |
   | Actual Total | `actual_home + actual_away` or `—` |
   | Spread Play | `spread_play` + ` HIT`/` MISS`/`` based on `spread_hit` (1/0/NULL) |
   | Total Play | `total_play` + ` HIT`/` MISS`/`` based on `total_hit` |

4. **Plays summary line** after the table:
   `ATS: {s_hits}–{s_misses} | O/U: {t_hits}–{t_misses} | {unresolved} unresolved`
   Only count rows where the play column is non-null.

## Spread sign convention (do not get this wrong)

The schema stores `proj_spread = proj_away − proj_home`. So:
- `proj_spread = +2.8` → **away favored by 2.8** (book line: `away −2.8` / `home +2.8`)
- `proj_spread = −0.1` → **home favored by 0.1** (book line: `home −0.1` / `away +0.1`)

Never write "Knicks +2.8" when Knicks were the *favored* away team — that's backwards. Use `Knicks −2.8` / `Hawks +2.8`.

## Python template for step 3

```python
import sqlite3, sys

date_arg = sys.argv[1]
con = sqlite3.connect("data/projections.db")
rows = con.execute("""
    SELECT home_team, away_team,
           proj_home_pts, proj_away_pts, proj_spread, proj_total,
           spread_play, total_play,
           actual_home_pts, actual_away_pts, spread_hit, total_hit
    FROM projections
    WHERE sector='nba' AND game_date=?
    ORDER BY home_team
""", (date_arg,)).fetchall()

if not rows:
    print(f"No NBA projections logged for {date_arg}.")
    raise SystemExit
# Use the rows to build the markdown table in your reply.
```

## Reference

- CLI source: `evmax/cli/commands/project.py`
- DB schema: see `_PROJ_SCHEMA` in `project.py`
- Engine: `evmax/models_ml/point_projection.py`
