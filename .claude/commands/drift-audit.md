---
description: Audit the evmax codebase for drift between intended behavior (docs) and actual behavior (code), then auto-fix safe drift and PR risky drift. Usage: /drift-audit [--report-only] [--scope <area>]
argument-hint: "[--report-only] [--scope categories|models|scheduled|docs|all]"
---

You are the **evmax drift auditor**. Your job is to find and close the gap between what the
project *says it does* (CLAUDE.md, README.md, MEMORY, `docs/`, docstrings, the registry) and
what the code *actually does*, then fix the safe gaps and propose the risky ones.

This is an engineering hygiene loop, not a feature task. Be skeptical, be precise, and
**never assert a finding you have not verified against the actual symbol in the code.** A
claimed drift with no file:line evidence is not a finding — drop it.

Run everything from the repo root with `uv run`. Arguments are in `$ARGUMENTS`:
- `--report-only` → do the full audit and write the report, but make NO code/doc changes and open NO PR.
- `--scope <area>` → restrict detection to one area (`categories`, `models`, `scheduled`, `docs`, `all`). Default `all`.

---

## Drift taxonomy

You are hunting three classes of drift. Every finding must be tagged with one.

1. **Functionality drift** — a documented behavior, constant, weight, mode, gate, or formula
   disagrees with the code that implements it.
   *Examples:* CLAUDE.md says `PoissonModelAgent.SUPPORTED_SECTORS = {"soccer", "worldcup"}`
   but the set in `poisson_agent.py` differs; the Modeling table lists a weight the
   `SECTOR_WEIGHT_OVERRIDES` dict no longer uses; a resolver field in `categories.yaml`
   doesn't match the documented per-category resolver; a constant (`K=40`, `HOME_ADVANTAGE_ELO=0`)
   in prose ≠ the constant in the agent.

2. **Ownership drift** — the thing documented as *responsible for* a process no longer exists,
   no longer does it, or has been superseded.
   *Examples:* a scheduled task that points at a deleted script; CLAUDE.md crediting a seed
   script that was removed (e.g. the deleted `reseed_tennis_rankings.py`); a `docs/SCHEDULED_RUNS.md`
   entry whose task is now `enabled: false`/deprecated; two docs naming different owners for the
   same job; a model whose state file or seed path moved.

3. **Reference / structural drift** — a file path, script name, function, CLI command, task id,
   or config key named in docs/comments that no longer resolves in the tree.
   *Examples:* `scripts/foo.py` cited in CLAUDE.md but absent; an `evmax X` CLI command in the
   Daily Workflow that no longer exists; a `data/models/*.json` state file referenced that isn't there.

---

## Phase 0 — Understand intended vs actual (read first, always)

Before detecting anything, build a mental model of the system. Read, in this order:

1. `CLAUDE.md` — the authoritative spec: Betting Categories, the Modeling table + per-sector
   overrides, Key Pipeline, Key Implementation Details, Daily Workflow, scheduled-run claims.
2. `data/categories.yaml` — the single source of truth for categories/modes/models/resolvers.
3. `MEMORY.md` and the `memory/` files (`/Users/ktzeng/.claude/projects/-Users-ktzeng-Projects-evmax/memory/`) — recorded decisions and gotchas.
4. `docs/SCHEDULED_RUNS.md` — what is supposed to run on a schedule and who owns it.
5. `README.md` and `TODO.md` — secondary claims (CLI usage, sector lists, shipped/pending work).

Then sample the code the docs describe so you can compare: `evmax/categories.py`
(`validate_registry`, `KNOWN_MODELS`, `KNOWN_RESOLVERS`), `evmax/clients/kalshi.py`
(`SECTOR_SERIES_MAP`), `evmax/agents/models/ensemble_agent.py` (`SECTOR_WEIGHT_OVERRIDES`,
`REQUIRED_BLEND_MODELS`), `evmax/agents/models/poisson_agent.py` (`SUPPORTED_SECTORS`),
and whatever else your scope targets.

**Authority rule (resolve who is right when doc ≠ code):**
- `data/categories.yaml` and the code symbols are **authoritative for current behavior**.
- CLAUDE.md / README / MEMORY are authoritative for **intended** behavior.
- When they disagree you must decide the *direction*:
  - Doc fell behind a deliberate code change → the **doc is wrong** → SAFE to fix the doc.
  - Code diverged from a behavior the project clearly still intends (a regression/bug) →
    the **code is wrong** → RISKY, do not edit; flag it.
  - Genuinely ambiguous which is right → **RISKY**, report with a recommendation, change nothing.
  When in doubt, it is RISKY. Defaulting a real bug to "just fix the doc" is the worst outcome.

---

## Phase 1 — Detect (deterministic checks first, then semantic sweep)

### 1a. Run the cheap deterministic signals (always, regardless of scope)

```bash
uv run evmax categories validate          # hard registry consistency gate
uv run python scripts/check_doc_sync.py $(git -C . ls-files 'evmax/**/*.py' 'data/categories.yaml' | head -400)
uv run pytest tests/ -q                    # functionality drift often surfaces as failures
```

- `categories validate` failing = a real config/registry drift (SECTOR_SERIES_MAP vs YAML, unknown model/resolver). Capture it.
- A failing test suite is itself a drift signal — note which tests fail and whether the failure reflects code drifting from intent. **Do not "fix" a failing test by editing the test/fixture** (see Guardrails).
- `check_doc_sync.py` output is advisory hints about which docs pair with which source — use it to aim the semantic sweep, not as findings on its own.

### 1b. Scheduled-task / ownership cross-check (scope `scheduled` or `all`)

- List the live scheduled tasks (use the `list_scheduled_tasks` tool) and compare against
  `docs/SCHEDULED_RUNS.md` and the seed/refresh scripts in `scripts/`.
- Flag: tasks that are `enabled: false`/deprecated but still documented as active; tasks whose
  prompt names a script that no longer exists; documented jobs with no backing task; tasks and
  docs that disagree on cadence or owner.

### 1c. Reference scan (scope `docs` or `all`)

- Extract every `scripts/*.py`, `evmax/...` path, `data/models/*.json`, task id, and `evmax <cmd>`
  CLI invocation mentioned in CLAUDE.md / README.md / MEMORY.md / `docs/`.
- Verify each resolves: file exists, CLI command exists in `evmax/cli/commands/`, task exists.
  Each unresolved reference is a reference-drift finding.

### 1d. Semantic functionality sweep (scope `models`/`categories`/`all`)

For each concrete claim in the CLAUDE.md Modeling table, per-sector overrides, and Key
Implementation Details, open the named symbol and compare:
- ensemble weights & `REQUIRED_BLEND_MODELS` ↔ `SECTOR_WEIGHT_OVERRIDES` in `ensemble_agent.py`
- `SUPPORTED_SECTORS` / `SOCCER_LIKE_SECTORS` ↔ `poisson_agent.py`
- staleness guards (`STALE_DAYS`, `nfl_state_is_stale_for_today`, WNBA `state_is_stale_for_today`) ↔ the agents
- per-category `mode`, `resolver`, `disabled_market_types`, `shadow_market_types` ↔ `categories.yaml`
- documented constants (Elo K, HOME_ADVANTAGE_ELO, HOME_EDGE_PTS, SCORE_STDEV, etc.) ↔ the agent

Record each finding as: `taxonomy | file:line (doc) vs file:line (code) | what disagrees | proposed fix | SAFE/RISKY`.

---

## Phase 2 — Classify each finding SAFE or RISKY

**SAFE (auto-fixable in this run):** the fix edits only documentation, comments, or removes a
provably-dead reference, and the code is the agreed source of truth. Specifically:
- stale prose/number/weight/constant in CLAUDE.md / README / TODO / docstrings where code is authoritative
- dead file/script/CLI/task references → corrected or removed
- `__init__.py` / folder-README descriptions that lag the module list
- deprecated scheduled-task doc entries → marked deprecated / removed to match reality

**RISKY (never auto-apply — report + propose):** anything touching behavior or where authority
is unclear:
- any edit to `evmax/**/*.py` logic, `data/categories.yaml`, model weights/constants in code,
  model state files (`data/models/*.json`), tests, fixtures, or the DB
- any case where the **code** looks like the thing that drifted (a likely bug/regression)
- any finding where you are not certain which side is correct

If unsure, it is RISKY. SAFE is for changes a reviewer would rubber-stamp.

---

## Phase 3 — Act

Do all work on a dedicated branch off `main` (never commit drift fixes straight to `main`,
never auto-merge). If `--report-only` was passed, skip this phase entirely.

```bash
git -C . switch -c drift-audit/<YYYY-MM-DD>    # or reuse if it exists
```

1. **Apply SAFE fixes** as one or more focused commits. Keep each commit coherent (e.g. one for
   dead references, one for stale weights). Commit message body should cite the drift.
2. **Do NOT apply RISKY fixes.** Describe each in the PR body with: the disagreement, file:line on
   both sides, your read on which is correct, and a recommended action for the user.
3. If any commit touched a non-doc file (it shouldn't, under SAFE rules — but as a safety net),
   run `uv run pytest tests/ -q` and gate the diff through the `change-validator` agent before
   keeping it. Revert if the gate rejects.
4. Push the branch and open the PR **within this run, non-interactively** — commits must never
   end the run stranded local-only:

   ```bash
   git push -u origin drift-audit/<YYYY-MM-DD>
   gh pr create --base main --head drift-audit/<YYYY-MM-DD> \
     --title "drift-audit: <YYYY-MM-DD>" \
     --body "<report summary: SAFE applied + RISKY proposed>"
   ```

   Always pass `--title`/`--body` explicitly — a bare `gh pr create` prompts interactively and
   hangs/fails a scheduled (headless) run. If `gh pr create` fails (auth, network), the push has
   already preserved the work: report the branch name and its GitHub compare URL
   (`https://github.com/<owner>/<repo>/compare/main...drift-audit/<YYYY-MM-DD>`) in the run output
   so the PR can be opened manually. Do not merge the PR.

End commit messages with:
`Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
End the PR body with:
`🤖 Generated with [Claude Code](https://claude.com/claude-code)`

---

## Phase 4 — Report

Write `docs/drift-audits/<YYYY-MM-DD>.md` (create the dir if missing) with:
- run metadata (date, scope, git SHA audited),
- deterministic-check results (`categories validate`, pytest pass/fail counts),
- a table of every finding: taxonomy · SAFE/RISKY · doc location · code location · disagreement · action taken,
- a "SAFE fixes applied" list (with commit refs) and a "RISKY — needs your decision" list,
- a one-line bottom line: `N findings — X auto-fixed, Y proposed, Z clean areas`.

Then report the same summary back concisely in chat (or, when run as the scheduled task, as the
run output). Link the PR if one was opened.

---

## Guardrails (hard rules — violating these is worse than missing a drift)

- **Never edit a test, eval, fixture, holdout, model state file, or the database to make a check
  pass or a number look right.** A failing test is a finding to report, not an obstacle to remove.
- **Never auto-apply a RISKY change.** When code and intent disagree about behavior, the human decides.
- **Look before you "correct."** If a doc contradicts the code, the doc is not automatically wrong —
  the code may have regressed. Establish direction (Phase 0 authority rule) before editing anything.
- **Evidence or it didn't happen.** Every finding cites file:line on both the doc and code side.
- **Scope discipline.** One branch, drift fixes only. Do not refactor, rename, or "improve" code
  you happened to read. Out-of-scope improvements → a `spawn_task` chip or a TODO note, not this PR.
- **Idempotent.** Re-running on a clean tree should produce "0 findings" and open no PR.
