# nfl-projections-* — spec

Three Claude scheduled tasks (created 2026-10-10) run the standalone NFL
projection engine (`evmax/nfl_projections/`) through the season. They write
only gitignored files, so there is no branch, commit or PR.

| Task id | Schedule (PT) | Command | Posts to Discord |
|---|---|---|---|
| `nfl-projections-weekly` | Tue 09:00 (`0 9 * * 2`) | `evmax project nfl-run --post` (grades last week first) + `nfl-track` readout | yes |
| `nfl-projections-friday-refresh` | Fri 14:30 (`30 14 * * 5`) | `evmax project nfl-run --no-resolve` | no |
| `nfl-projections-sunday-refresh` | Sun 08:45 (`45 8 * * 0`) | `evmax project nfl-run --no-resolve --post` | yes |

All three:
- run in the MAIN checkout (`/Users/ktzeng/Projects/evmax`), never a worktree
  (a worktree has its own near-empty `data/projections.db` and no nflverse cache);
- self-skip outside the NFL window 09-04 → 02-15;
- self-skip quietly while `evmax project nfl-run` does not exist in the main
  checkout (the code gate);
- notify only on failure.

## Why these times

- **Tuesday 09:00.** Monday night is final, so the whole past week grades. The
  coming week, including Thursday night, is projected before its first kickoff.
- **Friday 14:30.** Teams publish game-status designations (Out / Doubtful /
  Questionable) on Friday afternoon. Thursday's game is already frozen.
- **Sunday 08:45.** The live ESPN injury feed carries game-day downgrades before
  the 1pm ET (10:00 PT) kickoffs. Late games are refreshed too, but their
  inactives are not announced yet (known limitation).

## What a run does

`evmax project nfl-run` (`evmax/nfl_projections/pipeline.py`):

1. Grades stored rows whose game is final (`store.resolve`). Games get scores
   and the closing consensus line. Players get official stat lines once both the
   weekly stats and the snap counts for that game are published.
2. Projects the next week with an unplayed game: games (`live.project_week`) and
   players (`live.project_week_players`). Ruled-out players come from the
   nflverse injury report plus the live ESPN feed (`InjuryReportAgent`).
3. Upserts both into `data/projections.db` (`nfl_game_projections`,
   `nfl_player_projections`).
   - A row is rewritten on each run until its game kicks off, then frozen.
   - A player missing from a later pre-kickoff run is deleted from that game.
4. With `--post`, posts the stored week to the Discord channel/DM
   (`discord_bot.embeds.nfl_projection_embeds`).

Readers: the dashboard **NFL** tab (`GET /api/nfl-projections`), the `/nfl`
slash command, and `evmax project nfl-track` (tracked accuracy).
