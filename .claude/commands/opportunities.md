---
description: Find, scope and offline-test new +EV opportunities with the opportunity-scout multi-agent workflow, or build one supported opportunity as a shadow-only PR. Usage: /opportunities [focus] [--research 2] [--top-k N] [--top-n N] [--offline] | /opportunities build <opp-id> | /opportunities status
argument-hint: "[focus text] [--research 2] [--top-k N] [--top-n N] [--offline] | build <opp-id> | status"
---

You are the **evmax opportunity orchestrator's front end**. The orchestration itself lives in two
versioned Workflow graphs — `.claude/workflows/opportunity-scout.js` (discovery) and
`.claude/workflows/opportunity-build.js` (build). This command prepares their inputs, runs them,
and writes their results through `scripts/opportunity_ledger.py`. Design and gates:
`docs/opportunity-workflow-scope.md`.

**This command is the user's explicit opt-in to run those Workflow graphs.** Run from the repo
root of the current checkout with `uv run`. Use the session scratchpad (or
`.claude/opportunity-scratch/<date>/`, which is gitignored) for temporary files.

Parse `$ARGUMENTS`:
- starts with `status` → **Status**
- starts with `build ` → **Build** with the given `<opp-id>`
- anything else → **Discovery**; the free text (minus flags) is the focus. Flags:
  `--research 2` (two research lenses), `--top-k N` (briefs sent to the validator, default 5),
  `--top-n N` (briefs backtested, default 2), `--offline` (skip network sections of the snapshot).

## Status

```bash
uv run python scripts/opportunity_ledger.py status
```

Show the result as a short list (id, status, title, PR link).

## Discovery

1. `DATE=$(date +%F)`.
2. Build the shared context snapshot (read-only against the main checkout's databases):

   ```bash
   uv run python scripts/opportunity_context.py --date "$DATE" --focus "<focus>" [--offline]
   ```

   The last stdout line is the snapshot path. Stop and report if BOTH `categories` and
   `promotion_board` errored (stderr lists every section); otherwise continue and mention the
   failed sections. Read `meta.db_dir` from the snapshot (`jq -r .meta.db_dir <path>`).
3. Run the Workflow by name `opportunity-scout` (if the name does not resolve, use
   `scriptPath` = the absolute path of `.claude/workflows/opportunity-scout.js`) with args as a
   JSON object:
   `{"date": DATE, "focus": "<focus>", "snapshot_path": "<abs snapshot path>", "db_dir": "<db_dir>",
   "scratch_dir": ".claude/opportunity-scratch/<DATE>", "research_agents": 1|2, "top_k": 5,
   "top_n": 2}`.
   It runs in the background; wait for its completion notification. Do not predict its result.
4. Ingest the result straight from the notification's `<output-file>` (never retype the JSON):

   ```bash
   uv run python scripts/opportunity_ledger.py ingest "<output-file>" --date "$DATE"
   ```

   This writes the report `docs/opportunities/<DATE>.md`, ledger rows, graveyard additions
   (refuted ideas and permanent access blocks), research-journal sources and the competitive
   landscape.
5. Reply with: the report link, a table of briefs (id, status, pre-registered metric, result),
   the dropped count with the top reasons, and the next step — `/opportunities build <id>` for
   `SUPPORTED` briefs (`UNDERPOWERED` briefs can only be built as shadow-collect). Say plainly
   when nothing was supported; that is a normal outcome.
6. The run artifacts are left uncommitted in this checkout. List them and offer to open a docs PR.

## Build

1. `DATE=$(date +%F)`, `ID=<opp-id>`, `S=.claude/opportunity-scratch/$DATE`.
2. Gate G0 (status SUPPORTED / UNDERPOWERED, evidence ≤ 14 days old):

   ```bash
   uv run python scripts/opportunity_ledger.py get --id "$ID" --gate --today "$DATE" --row-out "$S/build-$ID-row.json"
   ```

   Exit code 2 means not buildable — report the reason and stop. On success stdout is the
   compact gate JSON (`opp_id`, `title`, `build_mode`, `preregistration`, `evidence`,
   `brief_summary`).
3. Open the isolated build worktree off `origin/main`:

   ```bash
   WT=$(python scripts/sched_worktree.py open --branch "opp/$ID" --print-path)
   uv run python scripts/opportunity_ledger.py record --id "$ID" --status BUILDING --branch "opp/$ID" --date "$DATE"
   ```

   `db_dir` = the main checkout's `data/` (the same value `opportunity_context.py` prints).
4. Run the Workflow `opportunity-build` (or its `scriptPath`) with args:
   `{"date": DATE, "opp_id": ID, "build_mode": <gate.build_mode>, "title": <gate.title>,
   "row_path": "<abs path of $S/build-$ID-row.json>", "gate": <the gate JSON object>,
   "worktree": "$WT", "branch": "opp/$ID", "db_dir": "<db_dir>", "scratch_dir": "$S"}`.
   Pass `gate` as a JSON object, not a string. Wait for the completion notification.
5. Read `result` from the `<output-file>` with `jq`.
   - `ready_to_ship: true` → ship ONLY the files the build changed, never live state:

     ```bash
     jq -r .result.pr_body "<output-file>" > "$S/pr-body-$ID.md"
     FILES=$(git -C "$WT" status --porcelain --untracked-files=all | awk '{print $NF}' | grep -vE '^data/(models/.*\.json|model_config\.json|.*\.db)')
     python scripts/sched_worktree.py ship --branch "opp/$ID" --worktree "$WT" \
         --title "$(jq -r .result.pr_title "<output-file>")" --body "$(cat "$S/pr-body-$ID.md")" \
         --run-hooks -- $FILES
     ```

     Use a long Bash timeout (pre-commit hooks run). Never pass `--merge-when-green`. Then
     record `--status PR_OPEN --pr-url <url>` and bind the PR with the ccd_pr tools.
   - otherwise → `record --status BLOCKED --note "<reason>"`, keep the worktree, and report its
     path, the reason, and the reviewer / test issues from `result.history`.

## Rules
- Never merge a PR, flip a category mode, or edit live `data/models/*_state.json` /
  `data/model_config.json` from this command.
- Never edit the workflow graphs mid-run to change a verdict. A gate result is final for the run.
- If a Workflow agent returns empty or odd results, read the run's `journal.jsonl` before
  diagnosing.
