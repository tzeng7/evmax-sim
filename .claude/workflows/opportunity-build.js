export const meta = {
  name: 'opportunity-build',
  description: 'Build run for one opportunity-scout brief: implementer writes the smallest shadow-only / default-off slice in an isolated worktree, test-runner proves it, the backtester re-runs the pre-registered test through the built code (G4), and change-validator reviews the FINAL diff last (up to 2 fix loops). Returns PR title/body; the /opportunities command re-checks everything deterministically (opportunity_ledger.py verify-build) before shipping. Never pushes, never merges.',
  whenToUse: 'Invoked by `/opportunities build <opp-id>` after the ledger confirms the brief is buildable and an isolated worktree exists.',
  phases: [
    { title: 'Implement', detail: 'implementer: smallest shadow-only / default-off slice + tests' },
    { title: 'Test', detail: 'test-runner: targeted + full suite in the isolated worktree' },
    { title: 'Verify', detail: 'backtester post-build: reproduce the pre-registered result (G4)' },
    { title: 'Review', detail: 'change-validator on the FINAL diff (implementer + test + verify edits)' },
  ],
}

// Design: docs/opportunity-workflow-scope.md §3.7, §4.2. The worktree and the PR are created by
// the /opportunities command (scripts/sched_worktree.py open / ship) — this graph only edits files
// inside args.worktree and returns a verdict plus PR text. The reviewer runs LAST and is read-only,
// so the diff it accepts is the diff that ships; verify-build then re-checks the actual changed
// files, the deny list, live-mode widening and G4 against the ledger, deterministically.

const IMPLEMENT_SCHEMA = {
  type: 'object',
  properties: {
    files_changed: { type: 'array', items: { type: 'string' }, description: 'paths relative to the worktree root' },
    diff_summary: { type: 'string' },
    flags_added: { type: 'array', items: { type: 'string' }, description: 'default-off flags / shadow lanes that gate the change' },
    shadow_guarantee: { type: 'string', description: 'why every LIVE lane prices, sizes and logs exactly as before' },
    tests_added: { type: 'array', items: { type: 'string' } },
    docs_updated: { type: 'array', items: { type: 'string' } },
    aborted_reason: { type: 'string', description: 'set if the change could not be made safely' },
    note: { type: 'string' },
  },
  required: ['files_changed', 'diff_summary', 'shadow_guarantee'],
}

const REVIEW_SCHEMA = {
  type: 'object',
  properties: {
    verdict: { type: 'string', enum: ['ACCEPT', 'REJECT'] },
    live_behavior_unchanged: { type: 'boolean', description: 'true only if every live lane prices, sizes and logs exactly as before with defaults' },
    issues: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          severity: { type: 'string', enum: ['blocker', 'major', 'minor'] },
          file: { type: 'string' }, line: { type: 'integer' }, problem: { type: 'string' }, fix: { type: 'string' },
        },
        required: ['severity', 'problem'],
      },
    },
    files_reviewed: { type: 'array', items: { type: 'string' }, description: 'every path in git status that you reviewed' },
    reward_hack_suspected: { type: 'boolean' },
    reason: { type: 'string' },
  },
  required: ['verdict', 'live_behavior_unchanged', 'issues', 'files_reviewed', 'reason'],
}

const TEST_SCHEMA = {
  type: 'object',
  properties: {
    passed: { type: 'boolean' },
    commands_run: { type: 'array', items: { type: 'string' } },
    tests_added: { type: 'array', items: { type: 'string' }, description: 'test files you created or edited' },
    failures: { type: 'array', items: { type: 'string' } },
    state_files_restored: { type: 'array', items: { type: 'string' }, description: 'data/ files the suite mutated and were restored inside the worktree' },
    note: { type: 'string' },
  },
  required: ['passed', 'commands_run', 'tests_added', 'failures'],
}

const VERIFY_SCHEMA = {
  type: 'object',
  properties: {
    ran: { type: 'boolean' },
    metric: { type: 'string' },
    value: { type: ['number', 'null'], description: 'in the metric’s units (METRIC_RULES)' },
    z_improvement: { type: ['number', 'null'] },
    n_games: { type: ['integer', 'null'] },
    ci_low: { type: ['number', 'null'] },
    ci_high: { type: ['number', 'null'] },
    command: { type: 'string' },
    code_path_used: { type: 'string', description: 'the implemented function / flag / CLI path the test went through' },
    shadow_collect_ok: { type: ['boolean', 'null'], description: 'shadow_collect builds: the new lane logs mode=shadow rows with zero Kelly' },
    leakage_checks: {
      type: 'object',
      properties: {
        utc_et_day: { type: 'boolean' }, point_in_time: { type: 'boolean' },
        no_future_close: { type: 'boolean' }, declustered_by_game: { type: 'boolean' },
      },
    },
    note: { type: 'string' },
  },
  required: ['ran', 'command', 'code_path_used', 'note'],
}

// METRIC_RULES:BEGIN (strict JSON — scripts/opportunity_ledger.py parses this block)
// threshold = value the metric must strictly beat (null → the brief must set one).
// value_range = plausible open interval; a value or threshold outside it is a units error.
const METRIC_RULES = {
  "clv_pp_net_fee": {"direction": "higher", "threshold": 0.0, "z_min": 1.64, "min_n_games": 30, "value_range": [-50, 50], "units": "percentage points of price, net of fees (0.8 = +0.8pp)"},
  "roi_net_fee": {"direction": "higher", "threshold": 0.0, "z_min": 1.64, "min_n_games": 30, "value_range": [-100, 100], "units": "percent ROI per unit staked, net of fees (3 = +3%)"},
  "open_close_slope": {"direction": "higher", "threshold": 0.0, "z_min": 2.0, "min_n_games": 30, "value_range": [-5, 5], "units": "OLS slope of (close - open) on (model - open), unitless"},
  "brier_delta_per_1000": {"direction": "lower", "threshold": -2.0, "z_min": 1.64, "min_n_games": 200, "value_range": [-100, 100], "units": "(candidate Brier - baseline Brier) x 1000; negative = better", "needs_promotion_plan": true},
  "match_rate": {"direction": "higher", "threshold": null, "z_min": null, "min_n_games": 30, "value_range": [0, 1], "units": "fraction 0-1"},
  "coverage": {"direction": "higher", "threshold": null, "z_min": null, "min_n_games": 30, "value_range": [0, 1], "units": "fraction 0-1"}
}
// METRIC_RULES:END

// DENY_PATHS:BEGIN (strict JSON regex list — scripts/opportunity_ledger.py verify-build applies the same list)
// A build may never change live model state, model config, databases, holdouts, secrets, or the
// opportunity machinery itself (graphs, agents, command, ledger, graveyard, their tests).
const DENY_PATHS = [
  "^data/models/.*\\.json$",
  "^data/model_config\\.json$",
  "^data/.*\\.db(-wal|-shm|-journal)?$",
  "^tests/fixtures/.*holdout",
  "(^|/)\\.env",
  "^\\.claude/",
  "^scripts/opportunity_[a-z_]+\\.py$",
  "^docs/opportunities/",
  "^tests/test_opportunity_[a-z_]+\\.py$",
  "^tests/workflow_harness\\.mjs$"
]
// DENY_PATHS:END

const BUILD_MODES = ['shadow_feature', 'shadow_collect']
const MAX_FIX_LOOPS = 2

// ---------------------------------------------------------------------------
// Pure gate functions (pinned by tests; verify-build re-implements them in Python)
// ---------------------------------------------------------------------------

function hasRule(metric) {
  return typeof metric === 'string' && Object.prototype.hasOwnProperty.call(METRIC_RULES, metric)
}

function effectiveRule(prereg) {
  const base = METRIC_RULES[prereg.metric]
  const t = typeof prereg.threshold === 'number' ? prereg.threshold : base.threshold
  const threshold = base.threshold === null ? t
    : (base.direction === 'higher' ? Math.max(base.threshold, t) : Math.min(base.threshold, t))
  const minN = Math.max(base.min_n_games, Number.isInteger(prereg.min_n_games) ? prereg.min_n_games : base.min_n_games)
  return { direction: base.direction, threshold, min_n_games: minN, value_range: base.value_range }
}

function forbiddenFiles(files) {
  const res = DENY_PATHS.map(s => new RegExp(s))
  return (files || []).filter(f => {
    const p = String(f).replace(/^\.\//, '')
    return res.some(r => r.test(p))
  })
}

// G4. shadow_collect: the new lane must log shadow rows. shadow_feature: the built code path must
// reproduce the pre-build result — same metric, plausible units, no leakage flag false, same side
// of the threshold, enough games, and inside the pre-build 95% CI (or within 25% of the pre-build
// margin over the threshold when no CI exists).
function reproduces(mode, prereg, pre, post) {
  if (!post) return { ok: false, reason: 'verification agent returned nothing' }
  if (mode === 'shadow_collect') {
    return post.shadow_collect_ok === true
      ? { ok: true, reason: 'shadow-collect lane logs mode=shadow rows' }
      : { ok: false, reason: `shadow-collect lane not confirmed: ${post.note || 'shadow_collect_ok is not true'}` }
  }
  if (mode !== 'shadow_feature') return { ok: false, reason: `unknown build mode ${mode}` }
  if (!prereg || !hasRule(prereg.metric)) return { ok: false, reason: 'pre-registration missing or unknown metric' }
  if (post.ran !== true) return { ok: false, reason: `post-build test did not run: ${post.note || ''}` }
  if (post.metric !== prereg.metric) return { ok: false, reason: `post-build metric ${JSON.stringify(post.metric)} ≠ pre-registered ${prereg.metric}` }
  const leaky = Object.entries(post.leakage_checks || {}).filter(([, v]) => v === false).map(([k]) => k)
  if (leaky.length) return { ok: false, reason: `post-build leakage checks failed: ${leaky.join(', ')}` }
  if (!pre || typeof pre.value !== 'number') return { ok: false, reason: 'no pre-build value to reproduce' }
  if (typeof post.value !== 'number' || !Number.isFinite(post.value)) return { ok: false, reason: 'post-build test returned no numeric value' }
  const rule = effectiveRule(prereg)
  const [lo, hi] = rule.value_range
  if (!(post.value > lo && post.value < hi)) return { ok: false, reason: `post-build value ${post.value} outside the plausible range (${lo}, ${hi})` }
  if (rule.threshold === null) return { ok: false, reason: 'no threshold to compare against' }
  const beats = v => (rule.direction === 'higher' ? v > rule.threshold : v < rule.threshold)
  if (!beats(post.value)) return { ok: false, reason: `post-build ${post.value} does not beat ${rule.threshold} (pre-build ${pre.value})` }
  if (!Number.isInteger(post.n_games) || post.n_games < rule.min_n_games) {
    return { ok: false, reason: `post-build n ${post.n_games} < ${rule.min_n_games} games` }
  }
  if (typeof pre.ci_low === 'number' && typeof pre.ci_high === 'number') {
    if (post.value < pre.ci_low || post.value > pre.ci_high) {
      return { ok: false, reason: `post-build ${post.value} outside pre-build CI [${pre.ci_low}, ${pre.ci_high}]` }
    }
    return { ok: true, reason: `post-build ${post.value} inside pre-build CI [${pre.ci_low}, ${pre.ci_high}]` }
  }
  const tol = 0.25 * Math.abs(pre.value - rule.threshold)
  if (Math.abs(post.value - pre.value) > tol) {
    return { ok: false, reason: `post-build ${post.value} differs from pre-build ${pre.value} by more than 25% of the margin (${tol})` }
  }
  return { ok: true, reason: `post-build ${post.value} within ${tol} of pre-build ${pre.value}` }
}

// A review only counts as ACCEPT when it is internally consistent.
function reviewAccepted(review) {
  if (!review || review.verdict !== 'ACCEPT') return false
  if (review.live_behavior_unchanged !== true || review.reward_hack_suspected === true) return false
  return !(review.issues || []).some(i => i && i.severity === 'blocker')
}

// A test run only counts as passed when it reports no failures.
function testPassed(test) {
  return !!test && test.passed === true && !(test.failures || []).length
}

function fmt(v) {
  if (v === null || v === undefined || v === '') return '—'
  return Array.isArray(v) ? v.join(', ') : String(v)
}

function prBody(A, brief, pre, impl, review, test, post, g4) {
  const p = brief.preregistration || {}
  return [
    `## Opportunity \`${A.opp_id}\` — ${brief.title}`,
    '',
    `Built by the opportunity-build workflow from discovery evidence dated ${A.date}. Build mode: **${A.build_mode}**.`,
    '',
    `- **Hypothesis:** ${fmt(brief.hypothesis)}`,
    `- **Edge mechanism:** ${fmt(brief.edge_mechanism)}`,
    `- **Lever / sectors:** ${fmt(brief.lever)} · ${fmt(brief.sectors)}`,
    '',
    '## Live behavior',
    '',
    `${fmt(impl.shadow_guarantee)}`,
    '',
    `Flags / shadow lanes: ${fmt(impl.flags_added)}. Final reviewer (read-only, last stage): live behavior unchanged = **${review.live_behavior_unchanged}**.`,
    'This PR promotes nothing. Promotion stays with the normal shadow → live gates (CLV net of fees, n ≥ 30 games).',
    '',
    '## Pre-registration',
    '',
    '| Metric | Units | Threshold | z min | Min games | Train | Holdout | Comparator |',
    '|---|---|---|---|---|---|---|---|',
    `| ${fmt(p.metric)} | ${fmt(hasRule(p.metric) ? METRIC_RULES[p.metric].units : null)} | ${fmt(p.threshold)} | ${fmt(p.z_min)} | ${fmt(p.min_n_games)} | ${fmt(p.train_window)} | ${fmt(p.holdout_window)} | ${fmt(p.comparator)} |`,
    '',
    '## Evidence',
    '',
    '| | Value | z | Games | 95% CI | Command |',
    '|---|---|---|---|---|---|',
    `| Pre-build (discovery) | ${fmt(pre && pre.value)} | ${fmt(pre && pre.z_improvement)} | ${fmt(pre && pre.n_games)} | ${pre && typeof pre.ci_low === 'number' ? `[${pre.ci_low}, ${pre.ci_high}]` : '—'} | \`${fmt(pre && pre.command)}\` |`,
    `| Post-build (this code) | ${fmt(post && post.value)} | ${fmt(post && post.z_improvement)} | ${fmt(post && post.n_games)} | ${post && typeof post.ci_low === 'number' ? `[${post.ci_low}, ${post.ci_high}]` : '—'} | \`${fmt(post && post.command)}\` |`,
    '',
    `G4 reproduction: **${g4.ok ? 'pass' : 'fail'}** — ${g4.reason}. Code path: ${fmt(post && post.code_path_used)}.`,
    'The /opportunities command re-ran G4, the deny list, the live-mode widening check and the changed-file allowlist deterministically (`scripts/opportunity_ledger.py verify-build`) before shipping.',
    '',
    '## Change',
    '',
    `${fmt(impl.diff_summary)}`,
    '',
    `- Files: ${fmt(impl.files_changed)}`,
    `- Tests added: ${fmt([...(impl.tests_added || []), ...((test && test.tests_added) || [])])}`,
    `- Docs updated: ${fmt(impl.docs_updated)}`,
    `- Test commands: ${fmt(test && test.commands_run)}`,
    `- Review: ${review.verdict} — ${fmt(review.reason)}`,
    '',
    '## Risks',
    '',
    ...((brief.risks || []).length ? brief.risks.map(r => `- ${r}`) : ['- none recorded']),
    '',
    '🤖 Generated with [Claude Code](https://claude.com/claude-code)',
  ].join('\n')
}

// ==== WORKFLOW BODY ====

const A = args || {}
const missing = ['date', 'opp_id', 'build_mode', 'row_path', 'gate', 'worktree', 'branch', 'db_dir'].filter(k => !A[k])
if (missing.length) return { ready_to_ship: false, status: 'BLOCKED', reason: `missing args: ${missing.join(', ')}` }
if (!BUILD_MODES.includes(A.build_mode)) return { ready_to_ship: false, status: 'BLOCKED', reason: `unknown build_mode ${A.build_mode}` }
const prereg = (A.gate && A.gate.preregistration) || null
const preEvidence = (A.gate && A.gate.evidence) || null
if (!prereg || !hasRule(prereg.metric)) return { ready_to_ship: false, status: 'BLOCKED', reason: 'gate.preregistration missing or has an unknown metric' }

const WT = A.worktree
const DB_ENV = `EVMAX_DB_DIR=${A.db_dir} EVMAX_DB_READONLY=1`
const SCRATCH = `${WT}/.claude/opportunity-scratch/${A.opp_id}`
const CONTEXT = `Opportunity \`${A.opp_id}\` (build mode ${A.build_mode}). The full ledger row — brief, scope verdict,
pre-build evidence — is at ${A.row_path}; read it first. Design: docs/opportunity-workflow-scope.md.
Isolated build worktree: ${WT} (branch ${A.branch}, cut from origin/main).
WORKING DIRECTORY RULES (mandatory):
- Use ABSOLUTE paths under ${WT} for every Read/Edit/Write, and run every shell command as \`cd ${WT} && …\`
  (the shell's cwd resets between calls). Never edit or run git in any other checkout — in particular not the
  shared checkout /Users/ktzeng/Projects/evmax.
- Never run: git stash (the stash stack is shared by every worktree), git reset --hard, git clean, git checkout
  outside ${WT}, git commit, git push, gh pr. The /opportunities command ships.
- Never edit data/models/*_state.json, data/model_config.json, any database, any holdout fixture, .env, or the
  opportunity machinery (.claude/, scripts/opportunity_*.py, docs/opportunities/, tests/test_opportunity_*).
  A deterministic deny list blocks the ship if you do.
Databases (read-only) from the worktree: prefix evmax commands with ${DB_ENV}
Throwaway files go under ${SCRATCH}/ (gitignored).`

const MODE_RULES = A.build_mode === 'shadow_collect'
  ? `shadow_collect: build ONLY what is needed to log mode='shadow' rows (Kelly zero) that will produce the
pre-registered metric over time. No pricing, sizing or mode change for any live lane.`
  : `shadow_feature: implement the lever behind a default-off flag, a new shadow market type / lane, or
shadow-only rows. With defaults, every LIVE lane must price, size and log byte-for-byte as before.`

const implementPrompt = fixes => `${CONTEXT}

You are implementing this opportunity as the smallest correct vertical slice.
${MODE_RULES}
Rules:
- Follow CLAUDE.md conventions and its Testing Policy: new logic gets tests in tests/ (happy path + edge case).
- Update CLAUDE.md only where documented behavior changes.
- No category / market / venue / league may become MORE live: data/categories.yaml modes,
  shadow_market_types, disabled_market_types, data/soccer_league_tiers.yaml shadow_leagues and the *_live
  settings defaults may only stay or tighten (verify-build checks this). New categories enter as shadow.
- New model agents need KNOWN_MODELS + data/categories.yaml entries (validate_registry).
- Run targeted tests only (cd ${WT} && uv run pytest tests/test_<x>.py -q); the test stage runs the full suite.
- Leave the changes UNCOMMITTED in the worktree. List EVERY path you created or edited in files_changed.
${fixes ? `\nThis is a FIX round. Address every issue below, then report the full current change:\n${JSON.stringify(fixes, null, 1)}` : ''}`

const testPrompt = impl => `${CONTEXT}

Prove the change empirically, inside the worktree only:
1. Run the targeted tests for every changed module; add any missing test for the new behavior and its edge
   cases (a test that would fail without the change). You may ADD tests; never weaken, skip or delete an
   existing assertion. List every test file you created or edited in tests_added.
2. Run the full suite: \`cd ${WT} && uv run pytest tests/ -q\`. It is safe here because the worktree is
   isolated, BUT the suite mutates data/models/*_state.json: afterwards run \`git -C ${WT} status --porcelain data/\`
   and restore every mutated tracked data/ file with \`git -C ${WT} checkout -- <file>\` (inside ${WT} only).
   Report what you restored.
3. Report passed=true only if every test passes; list every failure otherwise.
Implementer's summary:
${JSON.stringify(impl, null, 1)}`

const verifyPrompt = impl => `${CONTEXT}

POST-BUILD MODE. Re-run the pre-registered test THROUGH THE IMPLEMENTED CODE PATH in ${WT} (call the new
function / flag / CLI path — not a prototype). Pre-registration:
${JSON.stringify(prereg, null, 1)}
Units: ${METRIC_RULES[prereg.metric].units}
Pre-build evidence to reproduce:
${JSON.stringify(preEvidence, null, 1)}
${A.build_mode === 'shadow_collect' ? 'This is a shadow_collect build: confirm the new lane logs mode=\'shadow\' rows with zero Kelly (unit-level or a replay into a temporary database — never the real databases) and set shadow_collect_ok.' : 'Report value / z / n / CI exactly as in pre-build mode, with leakage_checks.'}
Do not edit any tracked file; scratch scripts go under ${SCRATCH}/.
Implementer's summary:
${JSON.stringify(impl, null, 1)}`

const reviewPrompt = (impl, test, post) => `${CONTEXT}

You are the FINAL reviewer: nothing runs after you, so the diff you accept is the diff that ships.
Review EVERY change in the worktree — \`git -C ${WT} status --porcelain --untracked-files=all\`,
\`git -C ${WT} diff\`, and every untracked file — including edits made by the test stage and the verification
stage, not just the implementer's. List every path you reviewed in files_reviewed.

Build mode rule: ${MODE_RULES}
Beyond correctness, regressions, conventions and reward hacking, check the evmax specifics: YES-side
alignment (matching/alignment.py is the only place that decides it), ET vs UTC game day, ':no'-side
conventions, ev_pct stored as a FRACTION, the venue / league shadow firewalls, mode='shadow' handling in
log_gaps, contamination rules for any newly-priced rows, a declared state_filename for renamed models,
MIN_NONSHARP_MODELS / REQUIRED_BLEND_MODELS (never zero a required model), parallel model stacks sharing no
files, and categories.yaml / KNOWN_MODELS consistency. A weakened, skipped or deleted existing test
assertion is a blocker. Any category/market/venue/league becoming more live is a blocker. Set
live_behavior_unchanged=true ONLY if, with defaults, every live lane prices, sizes and logs exactly as before.

Implementer: ${JSON.stringify(impl, null, 1)}
Test stage: ${JSON.stringify(test, null, 1)}
Verification: ${JSON.stringify(post, null, 1)}`

const history = []
let impl = null
let test = null
let post = null
let review = null
let g4 = null
let issues = null
let loop = 0
while (true) {
  phase('Implement')
  impl = await agent(implementPrompt(issues), { label: loop === 0 ? 'implement' : `implement:fix-${loop}`, phase: 'Implement', schema: IMPLEMENT_SCHEMA, agentType: 'implementer' })
  if (!impl) return { ready_to_ship: false, status: 'BLOCKED', reason: 'implementer returned nothing', history }
  if (impl.aborted_reason) return { ready_to_ship: false, status: 'BLOCKED', reason: `implementer aborted: ${impl.aborted_reason}`, impl, history }
  if (!(impl.files_changed || []).length) return { ready_to_ship: false, status: 'BLOCKED', reason: 'implementer reported no changed files', impl, history }

  issues = forbiddenFiles(impl.files_changed).map(f => ({ severity: 'blocker', file: f, problem: 'forbidden path (deny list): builds may not change live model state, model_config, databases, holdouts, secrets or the opportunity machinery' }))
  test = null
  post = null
  review = null
  if (!issues.length) {
    phase('Test')
    test = await agent(testPrompt(impl), { label: `test:${loop + 1}`, phase: 'Test', schema: TEST_SCHEMA, agentType: 'test-runner' })
    if (!testPassed(test)) {
      issues = test
        ? [...(test.failures || []).map(f => ({ severity: 'blocker', problem: `test failure: ${f}` })),
           ...(test.passed === true ? [{ severity: 'blocker', problem: 'test stage reported passed=true with failures listed' }] : []),
           ...(!(test.failures || []).length && test.passed !== true ? [{ severity: 'blocker', problem: `tests did not pass: ${test.note || ''}` }] : [])]
        : [{ severity: 'blocker', problem: 'test stage returned nothing' }]
    } else {
      phase('Verify')
      post = await agent(verifyPrompt(impl), { label: `verify:${loop + 1}`, phase: 'Verify', schema: VERIFY_SCHEMA, agentType: 'evmax-backtester' })
      g4 = reproduces(A.build_mode, prereg, preEvidence, post)
      log(`G4: ${g4.ok ? 'reproduced' : 'NOT reproduced'} — ${g4.reason}`)
      // A non-reproducing build is reported, never iterated on: tuning code until the number
      // matches would be fitting the implementation to the metric.
      if (!g4.ok) return { ready_to_ship: false, status: 'BLOCKED', reason: `G4: ${g4.reason}`, build_mode: A.build_mode, impl, test, verification: post, history }

      phase('Review')
      review = await agent(reviewPrompt(impl, test, post), { label: `review:${loop + 1}`, phase: 'Review', schema: REVIEW_SCHEMA, agentType: 'change-validator' })
      if (reviewAccepted(review)) break
      issues = [
        ...((review && review.issues) || []),
        ...(review && review.verdict === 'ACCEPT' ? [{ severity: 'blocker', problem: 'review said ACCEPT but was internally inconsistent (blocker issue, live behavior unconfirmed, or reward hack suspected)' }] : []),
        ...(!review ? [{ severity: 'blocker', problem: 'review stage returned nothing' }] : []),
      ]
    }
  }
  history.push({ loop: loop + 1, test, verification: post, review, issues })
  if (loop >= MAX_FIX_LOOPS) {
    return { ready_to_ship: false, status: 'BLOCKED', reason: `not accepted after ${MAX_FIX_LOOPS} fix loop(s)`, build_mode: A.build_mode, impl, test, review, history }
  }
  loop++
  log(`Fix loop ${loop}/${MAX_FIX_LOOPS}: ${issues.length} issue(s)`)
}

const allowed = [...new Set([...(impl.files_changed || []), ...((test && test.tests_added) || [])])]
const title = (A.title || '').slice(0, 60)
const prefix = A.build_mode === 'shadow_collect' ? 'shadow-collect' : 'shadow'
return {
  ready_to_ship: true,
  status: 'READY',
  reason: g4.reason,
  opp_id: A.opp_id,
  build_mode: A.build_mode,
  branch: A.branch,
  worktree: WT,
  files_changed: impl.files_changed,
  allowed_files: allowed,
  impl, review, test, verification: post, history,
  pr_title: `opportunity(${prefix}): ${title || A.opp_id}`,
  pr_body: prBody(A, { title: A.title, ...((A.gate && A.gate.brief_summary) || {}), preregistration: prereg }, preEvidence, impl, review, test, post, g4),
}
