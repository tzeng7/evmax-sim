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
root of the current checkout with `uv run`. Temporary files go in
`.claude/opportunity-scratch/<date>/` (gitignored).

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

1. Set `DATE=$(date +%F)` and `S=.claude/opportunity-scratch/$DATE`, then `mkdir -p "$S"`.
   `RUN_SEQ` = 1 + the number of existing `docs/opportunities/$DATE*.md` reports.
2. Build the shared context snapshot. It reads the main checkout's databases read-only:

   ```bash
   uv run python scripts/opportunity_context.py --date "$DATE" --focus "<focus>" [--offline]
   ```

   The last stdout line is the snapshot path. The web digest sits next to it as `<date>.web.json`
   (also in `meta.web_digest_path`). Stop and report if BOTH `categories` and
   `promotion_board` errored (stderr lists every section). Otherwise continue and mention the
   sections that failed. Read `meta.db_dir` with `jq -r .meta.db_dir <snapshot>`.
3. Record the git baseline. The integrity reviewer uses it to tell pre-existing changes from
   anything a backtester touched:

   ```bash
   git status --porcelain=v1 --untracked-files=all > "$S/git-baseline-discovery.txt"
   ```

4. Run the Workflow by name `opportunity-scout`. If the name does not resolve, use `scriptPath`
   with the absolute path of `.claude/workflows/opportunity-scout.js`. Pass args as a JSON
   object:
   `{"date": DATE, "focus": "<focus>", "snapshot_path": "<abs snapshot path>", "db_dir": "<db_dir>",
   "web_context": <the object printed by jq -c . <web digest path>>, "git_baseline_path": "<abs path>",
   "run_seq": RUN_SEQ, "scratch_dir": "<abs $S>", "research_agents": 1|2, "top_k": 5, "top_n": 2}`.
   `web_context` is the ONLY repo context the web-facing agents get (they have no file access).
   Copy it verbatim from `jq -c` output. It is context, not gate data.
   The workflow runs in the background. Wait for its completion notification and do not predict
   its result.
5. Ingest the result straight from the notification's `<output-file>`. Never retype the JSON.

   ```bash
   uv run python scripts/opportunity_ledger.py ingest "<output-file>" --date "$DATE"
   ```

   This writes the report `docs/opportunities/<DATE>.md`, the ledger rows, graveyard additions
   (refuted ideas and permanent access blocks), research-journal sources and the competitive
   landscape. A non-zero exit means the payload was not a scout result: report it and stop.
6. Reply with:
   - the report link;
   - a table of briefs (id, status, pre-registered metric, result);
   - the dropped count with the top reasons;
   - the next step: `/opportunities build <id>` for `SUPPORTED` briefs (`UNDERPOWERED` briefs
     can only be built as shadow-collect; `INCONCLUSIVE` briefs cannot be built).

   Say plainly when nothing was supported. That is a normal outcome.
7. The run artifacts are left uncommitted in this checkout. List them and offer to open a docs PR.

## Build

1. Set `DATE=$(date +%F)`, `ID=<opp-id>` and `S=.claude/opportunity-scratch/$DATE`, then
   `mkdir -p "$S"`.
2. Gate G0: the brief must be SUPPORTED or UNDERPOWERED, have no schema errors, and its evidence
   must be at most 14 days old.

   ```bash
   uv run python scripts/opportunity_ledger.py get --id "$ID" --gate --today "$DATE" --row-out "$S/build-$ID-row.json"
   ```

   Exit code 2 means not buildable: report the reason and stop. On success stdout is the compact
   gate JSON (`opp_id`, `title`, `build_mode`, `preregistration`, `evidence`, `brief_summary`).
3. Check for a previous worktree. If `/tmp/evmax-sched/opp-$ID` exists, an earlier BLOCKED
   build left it for inspection, and `open` would replace it. Ask the user before continuing.
   Then open the isolated build worktree off `origin/main`, record BUILDING, and only THEN take
   the session baseline:

   ```bash
   WT=$(python scripts/sched_worktree.py open --branch "opp/$ID" --print-path)
   uv run python scripts/opportunity_ledger.py record --id "$ID" --status BUILDING --branch "opp/$ID" --date "$DATE"
   git status --porcelain=v1 --untracked-files=all > "$S/session-baseline-$ID.txt"
   ```

   `db_dir` is the main checkout's `data/` (the same value `opportunity_context.py` prints).
4. Run the Workflow `opportunity-build` (or its `scriptPath`) with these args:
   `{"date": DATE, "opp_id": ID, "build_mode": <gate.build_mode>, "title": <gate.title>,
   "row_path": "<abs path of $S/build-$ID-row.json>", "gate": <the gate JSON object>,
   "worktree": "$WT", "branch": "opp/$ID", "db_dir": "<db_dir>"}`.
   Pass `gate` as a JSON object, not a string. Wait for the completion notification.
5. Run the deterministic ship gate. It recomputes G4 from the ledger and re-checks the build
   mode, the files that actually changed (deny list plus the build's declared allowlist), any
   widening of live modes, and that this checkout was not touched:

   ```bash
   uv run python scripts/opportunity_ledger.py verify-build "<output-file>" --id "$ID" --worktree "$WT" \
       --today "$DATE" --session-root "$PWD" --session-baseline "$S/session-baseline-$ID.txt" \
       --files-out "$S/ship-files-$ID.txt"
   ```

   - Exit 0: ship exactly the files `verify-build` listed:

     ```bash
     TITLE=$(jq -r .result.pr_title "<output-file>")
     jq -r .result.pr_body "<output-file>" > "$S/pr-body-$ID.md"
     python scripts/sched_worktree.py ship --branch "opp/$ID" --worktree "$WT" \
         --title "$TITLE" --body "$(cat "$S/pr-body-$ID.md")" \
         --message "$(printf '%s\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>' "$TITLE")" \
         --run-hooks -- $(cat "$S/ship-files-$ID.txt")
     ```

     Use a long Bash timeout because pre-commit hooks run. Never pass `--merge-when-green`.
     Then run `record --status PR_OPEN --pr-url <url>` and bind the PR with the ccd_pr tools.
   - Exit 3, or the workflow returned `ready_to_ship: false`: run
     `record --status BLOCKED --note "<first reasons>"`. Keep the worktree, and report its path,
     every reason, and the reviewer / test issues from `result.history`.

## Rules
- Never merge a PR, flip a category mode, or edit live `data/models/*_state.json` /
  `data/model_config.json` from this command.
- Never ship without a passing `verify-build`, and never ship files it did not list.
- Never edit the workflow graphs, the ledger script or the graveyard mid-run to change a
  verdict. A gate result is final for the run.
- If a Workflow agent returns empty or odd results, read the run's `journal.jsonl` before
  diagnosing.
