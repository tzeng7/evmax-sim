export const meta = {
  name: 'model-improve',
  description: 'Gated model-improvement loop: value-audit finds actionable sector gaps; for each (most-significant first) an agent proposes ONE model-side change, it is measured walk-forward and integrity- and signal-gated; the FIRST change that validates ships as a PR, the rest revert. Never edits live model state directly. Safe to run headless in the SHARED main checkout.',
  whenToUse: 'Run manually or on a schedule to improve one sector blend against a walk-forward Brier signal.',
  phases: [
    { title: 'Observe', detail: 'value-audit --json → actionable findings' },
    { title: 'Hypothesize', detail: 'rank one novel model-side change per finding (checks ledger)' },
    { title: 'Implement', detail: 'branch + ONE change, committed to the branch' },
    { title: 'Measure', detail: 'walk-forward backtest, train≤Y / holdout=Y+1' },
    { title: 'Integrity', detail: 'iteration-reviewer: honest number?' },
    { title: 'Dispose', detail: 'keep→PR (ship first validated) or revert; append ledger row' },
  ],
}

// ---- edge contracts (validated at the tool-call layer) ----

const AUDIT_SCHEMA = {
  type: 'object',
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          sector: { type: 'string' },
          market_type: { type: 'string' },
          gap_type: { type: 'string', description: 'e.g. model_subtracts | calibration_bias | none' },
          z_score: { type: 'number', description: 'paired z of blend vs entry-sharp Brier; negative = blend worse' },
          calib_bias: { type: 'boolean', description: 'consistent calibration bias across buckets' },
          n_clean: { type: 'integer' },
          actionable: { type: 'boolean', description: 'z <= -1.64 OR calib_bias, per the value-audit rule' },
          evidence: { type: 'string' },
        },
        required: ['sector', 'gap_type', 'actionable', 'evidence'],
      },
    },
  },
  required: ['findings'],
}

const CHANGE_SPEC_SCHEMA = {
  type: 'object',
  properties: {
    sector: { type: 'string' },
    target: { type: 'string', description: 'which model/weight/calibration is touched (file:symbol)' },
    kind: { type: 'string', enum: ['reweight', 'recalibrate', 'add_feature', 'none'] },
    params: { type: 'string', description: 'the concrete edit, e.g. "efficiency 0.30→0.35, elo 0.10→0.05"' },
    rationale: { type: 'string' },
    novel: { type: 'boolean', description: 'true only if this exact change is NOT a reverted row in the ledger' },
    note: { type: 'string' },
  },
  required: ['sector', 'kind', 'params', 'novel'],
}

const IMPLEMENT_SCHEMA = {
  type: 'object',
  properties: {
    branch: { type: 'string' },
    committed: { type: 'boolean', description: 'true only if OUR change is committed to the throwaway branch' },
    files_changed: { type: 'array', items: { type: 'string' } },
    diff_summary: { type: 'string' },
    aborted_reason: { type: 'string', description: 'set if a git-safety guard stopped the run' },
    note: { type: 'string' },
  },
  required: ['committed', 'diff_summary'],
}

const METRICS_SCHEMA = {
  type: 'object',
  properties: {
    ran: { type: 'boolean', description: 'false if no valid walk-forward backtest could be run' },
    command: { type: 'string' },
    train_seasons: { type: 'string' },
    holdout_season: { type: 'string' },
    baseline_holdout_brier: { type: 'number', description: 'holdout Brier of the UNCHANGED blend' },
    candidate_holdout_brier: { type: 'number', description: 'holdout Brier WITH the change' },
    train_brier: { type: 'number' },
    coverage_pct: { type: 'number' },
    n: { type: 'integer' },
    note: { type: 'string' },
  },
  required: ['ran', 'note'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    accept: { type: 'boolean' },
    reward_hack_suspected: { type: 'boolean' },
    reason: { type: 'string' },
  },
  required: ['accept', 'reason'],
}

const DISPOSE_SCHEMA = {
  type: 'object',
  properties: {
    action: { type: 'string', enum: ['kept', 'reverted', 'blocked'] },
    pr_url: { type: 'string' },
    ledger_written: { type: 'boolean' },
    note: { type: 'string' },
  },
  required: ['action', 'ledger_written'],
}

// Brier noise floor: a holdout improvement below this is treated as within-noise and reverted.
// ~2 Brier points / 1000, matching the project's own "below the noise floor" language.
const NOISE_FLOOR = 0.002
// How many actionable findings to try per run before giving up. Bounds cost; we still ship at
// most ONE change (the first that validates) — this just stops a single non-novel/failing top
// finding from suppressing a real, validatable fix waiting behind it.
const MAX_ATTEMPTS = 3
const LEDGER = '.claude/improvement-ledger.jsonl'

// Operational safety rules for every agent that touches git in the SHARED main checkout.
// The checkout is shared with daily-resolve-and-model-update, weekly-model-calibration,
// weekly-seasonal-model-reseed and others, which routinely leave legitimate UNCOMMITTED
// changes sitting here between runs. Destroying that state broke a task on 2026-07-21.
const GIT_SAFETY = `GIT SAFETY (SHARED checkout — mandatory):
  - NEVER run: git reset --hard, git checkout -- ., git restore ., git clean, git stash drop.
    These destroy other scheduled tasks' uncommitted work. There is no exception.
  - Stage and commit ONLY the specific files this change touched (git add <those paths>),
    never 'git add -A' / 'git add .' — other tasks' stray uncommitted files must stay untouched.
  - Do NOT run the full test suite ('pytest tests/' with no filter): it MUTATES
    data/models/*_state.json. Run targeted tests only (pytest -k / a specific test file).
  - Always end on main via 'git switch main'. If 'git switch' reports a conflict or would
    overwrite changes, STOP and report it — never force, never discard to get past it.`

// ===========================================================================

// ---- N1 · OBSERVE ---------------------------------------------------------
phase('Observe')
const audit = await agent(
  `Run the evmax model-value audit and return its actionable findings.

  From the repo root, run:
    uv run evmax cleanup value-audit --weeks 12 --json

  Parse the JSON. For each sector/market it reports Brier of the blend vs entry-sharp
  (with paired z), calibration bias, and n. A finding is ACTIONABLE only when
  z <= -1.64 (blend significantly WORSE than the sharp line) OR there is a consistent
  calibration bias across buckets — the project's hard rule (see .claude/commands/value-audit.md).
  Beating the CLOSE is rare and healthy; CLV is context only, never a reason to act.
  "No actionable gap" is a correct, expected outcome — do NOT manufacture one.

  Return every finding, with actionable set correctly. Read-only: change nothing.`,
  { label: 'value-audit', phase: 'Observe', schema: AUDIT_SCHEMA, agentType: 'general-purpose' },
)

const findings = (audit?.findings || []).filter((f) => f && f.actionable)

// ---- G1 · actionable? -----------------------------------------------------
if (findings.length === 0) {
  log('G1: no actionable gap — all blends within noise. Stopping (this is a correct outcome).')
  return { stopped: 'no_actionable_gap', reviewed: (audit?.findings || []).length }
}

// most-significant first (most-negative z; calibration-bias-only sorts after)
findings.sort((a, b) => (a.z_score ?? 0) - (b.z_score ?? 0))
const queue = findings.slice(0, MAX_ATTEMPTS)
log(`G1: ${findings.length} actionable gap(s); will try up to ${queue.length} until one validates → [${queue.map((f) => f.sector).join(', ')}]`)

// One attempt against a single finding. Returns { kept, dispose, spec, metrics, verdict } or a
// { skipped } reason. Ships (PR) and returns kept=true for the FIRST change that clears G2+G3.
async function tryFinding(target, idx) {
  const tag = `${target.sector}#${idx + 1}`

  // ---- N2 · HYPOTHESIZE (folds G0 novelty check) --------------------------
  phase('Hypothesize')
  const spec = await agent(
    `Propose ONE model-side change to close this evmax value gap.

    Finding: ${JSON.stringify(target)}

    First read the ledger at ${LEDGER}. Each line is a prior attempt. Do NOT re-propose any
    change whose "decision" was "revert/no-change" for this sector — set novel=false if the only
    fix you can think of is already a reverted row, and explain in note.

    Allowed change kinds ONLY (gating is forbidden — see value-audit.md rule 1):
      - reweight: adjust this sector's entry in SECTOR_WEIGHT_OVERRIDES (ensemble_agent.py)
      - recalibrate: refresh the sector's isotonic curve (evmax cleanup recalibrate --sector S)
      - add_feature: add one signal to the sector's model
    Propose the smallest change that addresses the finding's mechanism. ONE change only.
    Do NOT implement anything yet — just return the ChangeSpec. Read-only.`,
    { label: `hypothesize:${tag}`, phase: 'Hypothesize', schema: CHANGE_SPEC_SCHEMA, agentType: 'general-purpose' },
  )

  if (!spec || spec.kind === 'none' || spec.novel === false) {
    log(`G0[${tag}]: no novel change (${spec?.note || 'none proposed'}) — next finding.`)
    return { skipped: 'no_novel_change', spec }
  }

  // ---- N3 + N4 · ISOLATE + IMPLEMENT (commit to the branch immediately) ----
  phase('Implement')
  const impl = await agent(
    `Implement exactly ONE model-side change in the evmax repo, on a fresh git branch, and COMMIT it
    to that branch so the working tree is never used to hold the diff.

    ChangeSpec: ${JSON.stringify(spec)}

    ${GIT_SAFETY}

    Steps:
      1. Record the starting branch. Create the throwaway branch:
           git switch -c model-improve/${spec.sector}-${spec.kind}
         (uncommitted files from other tasks harmlessly ride along — do not touch them.)
      2. Make ONLY the change in the spec. Do NOT touch tests, fixtures, backtest scripts,
         holdout data, or any success metric — that is reward hacking and the next stage rejects it.
         If the change needs a state re-seed, run the sector's seed script; never hand-edit state JSON.
      3. Commit ONLY the files you changed:
           git add <exact paths>   &&   git commit -m "<summary>  (model-improve, pre-validation)"
         Commit message ends with:  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
      4. Leave the checkout ON the branch with the change committed (clean working tree for our paths).
         Return committed=true only if the commit succeeded; else committed=false with aborted_reason,
         and switch back to the starting branch WITHOUT any destructive command.`,
    { label: `implement:${tag}`, phase: 'Implement', schema: IMPLEMENT_SCHEMA, agentType: 'implementer' },
  )

  if (!impl || !impl.committed) {
    log(`Implement[${tag}] did not commit (${impl?.aborted_reason || impl?.note || 'no diff'}) — next finding.`)
    return { skipped: 'implement_failed', spec, impl }
  }

  // ---- N5 · MEASURE (leak-free walk-forward; change committed on branch) ----
  phase('Measure')
  const metrics = await agent(
    `Measure this evmax change walk-forward. The change is committed on branch ${impl.branch}, which is
    currently checked out and active. The signal must not see the future.

    Sector: ${spec.sector}. Change under test: ${JSON.stringify(spec)}.

    Preferred: run the general walk-forward with an explicit train/holdout split, e.g.
      uv run evmax backtest run --sectors ${spec.sector} --seasons <train>,<holdout>
    If a dedicated script fits better, use scripts/backtest_${spec.sector}*.py (e.g.
    backtest_ncaaf_efficiency.py, backtest_nfl_efficiency.py, backtest_baseball_ab.py). Any
    prior/seed the model uses MUST be derived from seasons <= the train year only.

    Report the holdout Brier of the UNCHANGED blend (baseline_holdout_brier) AND with the change
    (candidate_holdout_brier) on the SAME holdout split, plus coverage and n. If no valid
    walk-forward can be run (missing data, coverage < 90%), set ran=false and explain — do not
    fabricate numbers. Do not run the full pytest suite. Read-only w.r.t. git.`,
    { label: `measure:${tag}`, phase: 'Measure', schema: METRICS_SCHEMA, agentType: 'general-purpose' },
  )

  // ---- G2 · INTEGRITY (before signal, always) -----------------------------
  phase('Integrity')
  const verdict = metrics?.ran
    ? await agent(
        `Adversarially review this evmax model-improvement iteration for reward hacking.

        ChangeSpec: ${JSON.stringify(spec)}
        Branch: ${impl.branch} (inspect the actual committed diff: git diff main...${impl.branch})
        Measurement: ${JSON.stringify(metrics)}

        REJECT (accept=false) if the diff edits any test, fixture, backtest script, holdout data,
        or the metric definition; if the model leaks holdout/future data into training or the prior;
        if outputs are hardcoded; or if the change is a disguised gating/threshold move rather than a
        genuine model/blend improvement. Inspect the real diff, do not trust the summary alone.
        Default to reject when uncertain. Read-only.`,
        { label: `integrity:${tag}`, phase: 'Integrity', schema: VERDICT_SCHEMA, agentType: 'iteration-reviewer' },
      )
    : { accept: false, reason: `measurement did not run: ${metrics?.note || 'unknown'}` }

  // ---- G3 · SIGNAL (pure arithmetic) --------------------------------------
  const improvement = metrics?.ran
    ? (metrics.baseline_holdout_brier ?? Infinity) - (metrics.candidate_holdout_brier ?? Infinity)
    : -Infinity
  const signalPass = metrics?.ran && improvement >= NOISE_FLOOR
  const keep = Boolean(verdict?.accept) && signalPass

  log(
    `Gates[${tag}] → integrity=${verdict?.accept ? 'ACCEPT' : 'REJECT'} · ` +
      `signal Δbrier=${metrics?.ran ? improvement.toFixed(4) : 'n/a'} (floor ${NOISE_FLOOR}) → ${signalPass ? 'PASS' : 'FAIL'} · ` +
      `decision=${keep ? 'KEEP' : 'REVERT'}`,
  )

  // ---- N6 · DISPOSE (no destructive working-tree commands, ever) -----------
  phase('Dispose')
  const decisionCtx = {
    sector: spec.sector,
    loop: `model-improve-${spec.sector}`,
    branch: impl.branch,
    spec,
    metrics,
    verdict,
    improvement: metrics?.ran ? improvement : null,
  }

  const dispose = await agent(
    keep
      ? `KEEP this evmax change and open a PR. The change is committed on branch ${impl.branch}.

         Context: ${JSON.stringify(decisionCtx)}

         ${GIT_SAFETY}

         Steps:
           1. Run TARGETED tests only for the touched area, e.g.
                uv run pytest tests/ -q -k "${spec.sector} or ensemble"
              (never the full suite — it mutates state files). If they fail, do NOT open the PR:
              fall through to the revert path (git switch main; git branch -D ${impl.branch}) and
              record decision "revert/no-change" with note "targeted tests failed"; return action=blocked.
           2. Push and open ONE PR non-interactively (a bare 'gh pr create' hangs a headless run):
                git push -u origin ${impl.branch}
                gh pr create --base main --head ${impl.branch} --title "<title>" --body "<body>"
              The body LEADS with the model change: which weight/calibration/feature moved, the
              before→after walk-forward holdout Brier, the integrity verdict, why it is not overfitting.
              PR body ends with:  🤖 Generated with [Claude Code](https://claude.com/claude-code)
              If the push succeeds but 'gh pr create' fails (auth/network), report the branch name +
              GitHub compare URL so the PR can be opened by hand — the work is preserved. Do NOT merge.
           3. Append one line to ${LEDGER} (stamp ts via: date +%F):
              {"ts":"<today>","loop":"model-improve-${spec.sector}","phase":"attempt",
               "hyp":"<rationale>","change":"<params>","decision":"keep-PR",
               "metric":"holdout_brier","value":<candidate>,"note":"Δ<improvement> vs baseline; PR <url>","sha":"<git rev-parse --short HEAD>"}
           3b. AUTO-FIX CI (one pass, never merge — model-improve PRs stay human-reviewed):
                 BUCKET=$(python3 /Users/ktzeng/Projects/evmax/scripts/sched_worktree.py watch-ci --branch ${impl.branch})
               - 'pass' -> done; leave the PR open for review.
               - 'fail' -> make ONE corrective attempt on THIS branch: read the failing check's logs
                 (gh run view <run-id> --log-failed; the id/URL is printed in the watch-ci line), fix the
                 cause with a focused commit (same Co-Authored-By trailer), git push, then run watch-ci
                 ONCE more. Whatever the second result, STOP -- leave the PR open, never merge, never loop again.
               - 'pending'/'error'/'none' -> report the bucket and leave the PR open.
           4. git switch main. If it conflicts, report and STOP — do not force. Leave branch ${impl.branch} pushed.
         Return action=kept with the PR url.`
      : `REVERT this evmax change — it did not clear the gates. The change is committed on branch
         ${impl.branch}; discard it as a pure branch operation, with NO working-tree destruction.

         Context: ${JSON.stringify(decisionCtx)}

         ${GIT_SAFETY}

         Steps:
           1. git switch main   (our change was COMMITTED on the branch, so this leaves the working
              tree without it; other tasks' uncommitted files ride along untouched). If 'git switch'
              reports a conflict, STOP and report — never force, never 'git checkout -- .'.
           2. git branch -D ${impl.branch}   (deletes the throwaway branch and its commit).
           3. Append one line to ${LEDGER} (stamp ts via: date +%F):
              {"ts":"<today>","loop":"model-improve-${spec.sector}","phase":"attempt",
               "hyp":"<rationale>","change":"<params>","decision":"revert/no-change",
               "note":"<why: integrity reject reason OR within-noise Δbrier>","sha":"<current short sha on main>"}
         Return action=reverted.`,
    { label: `dispose:${keep ? 'keep' : 'revert'}:${tag}`, phase: 'Dispose', schema: DISPOSE_SCHEMA, agentType: 'implementer' },
  )

  const shipped = keep && dispose?.action === 'kept'
  return { kept: shipped, dispose, spec, metrics, verdict, improvement: metrics?.ran ? improvement : null }
}

// ---- attempt actionable findings until one validates + ships --------------
const attempts = []
for (let i = 0; i < queue.length; i++) {
  const r = await tryFinding(queue[i], i)
  attempts.push({ sector: queue[i].sector, gap: queue[i].gap_type, outcome: r.kept ? 'kept' : r.skipped || 'reverted' })
  if (r.kept) {
    return {
      decision: 'kept',
      sector: r.spec.sector,
      change: r.spec.params,
      holdout_delta: r.improvement,
      pr_url: r.dispose?.pr_url || null,
      ledger_written: Boolean(r.dispose?.ledger_written),
      attempts,
    }
  }
}

log(`All ${attempts.length} actionable finding(s) tried; none produced a validated change. No PR this run.`)
return { decision: 'no_validated_change', attempts }
