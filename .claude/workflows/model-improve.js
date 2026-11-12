export const meta = {
  name: 'model-improve',
  description: 'Gated model-improvement loop: value-audit (plus opt-in coverage / close-capture / calibration-bucket triggers) finds gaps; for each queued finding an agent proposes ONE change, it is measured on the trigger\'s own metric and integrity- and signal-gated; the FIRST change that validates ships as a PR, the rest revert. Never edits live model state directly. Safe to run headless in the SHARED main checkout.',
  whenToUse: 'Run manually or on a schedule to improve one sector blend. Optional args: {triggers: [...], dry_run: true}.',
  phases: [
    { title: 'Observe', detail: 'value-audit --json + improve-triggers --json → findings' },
    { title: 'Hypothesize', detail: 'rank one novel change per finding (checks ledger, allowed kinds per trigger)' },
    { title: 'Implement', detail: 'branch + ONE change, committed to the branch' },
    { title: 'Measure', detail: 'walk-forward / replay on the trigger\'s own metric' },
    { title: 'Integrity', detail: 'iteration-reviewer: honest number?' },
    { title: 'Dispose', detail: 'keep→PR (ship first validated) or revert; append ledger row' },
  ],
}

// Args (all optional; the scheduled task passes none):
//   triggers: string[]  — extra triggers that may QUEUE work, on top of 'value_audit' (always on,
//                         the original behavior). Every trigger not listed runs shadow-only: it is
//                         measured and reported as `shadow_queue`, but no agent acts on it.
//   dry_run: boolean     — stop after Observe and return the queue (no Hypothesize/Implement).

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

const TRIGGERS_SCHEMA = {
  type: 'object',
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          trigger: { type: 'string' },
          sector: { type: 'string' },
          market_type: { type: ['string', 'null'] },
          venue: { type: ['string', 'null'] },
          model: { type: ['string', 'null'] },
          bucket: { type: ['string', 'null'] },
          n: { type: 'integer' },
          value: { type: 'number' },
          threshold: { type: 'number' },
          z: { type: ['number', 'null'] },
          fires: { type: 'boolean' },
          judge_metric: { type: 'string' },
          priority: { type: 'number' },
          evidence: { type: 'string' },
        },
        required: ['trigger', 'sector', 'n', 'value', 'fires', 'judge_metric', 'evidence'],
      },
    },
    ran: { type: 'array', items: { type: 'string' } },
    failed: { type: 'object' },
  },
  required: ['findings'],
}

const CHANGE_SPEC_SCHEMA = {
  type: 'object',
  properties: {
    sector: { type: 'string' },
    target: { type: 'string', description: 'which model/weight/calibration/data path is touched (file:symbol)' },
    kind: { type: 'string', enum: ['reweight', 'recalibrate', 'add_feature', 'coverage_fix', 'data_fix', 'none'] },
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
    ran: { type: 'boolean', description: 'false if no valid measurement could be run' },
    command: { type: 'string' },
    // holdout_brier (value_audit) and the calibration_bucket no-harm guard
    train_seasons: { type: 'string' },
    holdout_season: { type: 'string' },
    baseline_holdout_brier: { type: 'number', description: 'holdout Brier of the UNCHANGED blend' },
    candidate_holdout_brier: { type: 'number', description: 'holdout Brier WITH the change' },
    train_brier: { type: 'number' },
    // rows_priced_frac (model_coverage, passthrough) and clv_coverage_frac (close_capture)
    coverage_before: { type: 'number', description: 'fraction 0–1 on the finding\'s rows, unchanged code' },
    coverage_after: { type: 'number', description: 'fraction 0–1 on the SAME rows, changed code' },
    newly_priced_n: { type: 'integer', description: 'rows the change newly prices (coverage) or recovers (close capture)' },
    harm_z: { type: 'number', description: 'z of (Brier with − Brier without) on newly priced rows; positive = worse' },
    rule_violations: { type: 'integer', description: 'recovered closes that break the forward-only / near-tip CLV rules' },
    // bucket_calib_err_pp (calibration_bucket)
    bucket_err_before_pp: { type: 'number', description: '|blend − realized| in the bucket on the holdout, unchanged' },
    bucket_err_after_pp: { type: 'number', description: '|blend − realized| in the bucket on the holdout, changed' },
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
// How many queued findings to try per run before giving up. Bounds cost; we still ship at
// most ONE change (the first that validates) — this just stops a single non-novel/failing top
// finding from suppressing a real, validatable fix waiting behind it.
const MAX_ATTEMPTS = 3
const LEDGER = '.claude/improvement-ledger.jsonl'

// ---- triggers ----------------------------------------------------------------
// value_audit is the original rule. The other four come from
// evmax/agents/cleanup/improve_triggers.py (`evmax cleanup improve-triggers --json`); the
// thresholds below are RE-CHECKED here so an agent's transcription can never widen them, and
// tests/test_model_improve_workflow.py pins them equal to the Python constants.
//   judge  — the metric that decides keep/revert for a fix of this trigger. A coverage or data
//            fix is judged on its own metric (rows priced, CLV coverage), never on Brier.
//   kinds  — the only change kinds a hypothesis may propose for this trigger.
const DEFAULT_TRIGGERS = ['value_audit']
const TRIGGER_ORDER = ['model_coverage', 'passthrough', 'close_capture', 'calibration_bucket']
const TRIGGER_RULES = {
  value_audit: { judge: 'holdout_brier', kinds: ['reweight', 'recalibrate', 'add_feature'] },
  // LIVE moneyline lane whose blend is the sharp line. sharp_weight is owned by `cleanup adjust`
  // (judged on CLV) and demotion is a manual promotion call, so only a coverage fix is allowed.
  passthrough: { judge: 'rows_priced_frac', kinds: ['coverage_fix'], min_n: 10, max_value: 0.5 },
  model_coverage: { judge: 'rows_priced_frac', kinds: ['coverage_fix'], min_n: 20, min_value: 0.5 },
  close_capture: { judge: 'clv_coverage_frac', kinds: ['data_fix'], min_n: 10, min_value_exclusive: 0.3 },
  calibration_bucket: { judge: 'bucket_calib_err_pp', kinds: ['recalibrate'], min_n: 50, min_abs_value: 3.0, min_abs_z: 2.5 },
}
// Signal floors for the non-Brier judges.
const COVERAGE_MIN_GAIN = 0.10      // rows-priced fraction must rise ≥ 10 points
const COVERAGE_MIN_NEW_N = 20       // …on ≥ 20 newly priced rows
const HARM_Z_MAX = 1.64             // newly priced rows must not be significantly WORSE (guard, not judge)
const CLV_COVERAGE_MIN_GAIN = 0.10  // CLV-coverage fraction must rise ≥ 10 points
const CLV_MIN_RECOVERED = 5         // …recovering ≥ 5 closes, none breaking the CLV rules
const CALIB_MIN_DROP_PP = 1.0       // bucket |blend − realized| must fall ≥ 1pp on the holdout
// Mode, weight, firewall and league-shadow config: changing any of them is a promotion or a
// sharp_weight decision, never a model-improve change. Enforced on the committed file list.
const DENY_FILES = ['data/categories.yaml', 'data/model_config.json', 'data/soccer_league_tiers.yaml', 'evmax/settings.py']

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

const HYPOTHESIS_GUIDANCE = {
  value_audit: `Allowed change kinds ONLY (gating is forbidden — see value-audit.md rule 1):
      - reweight: adjust this sector's entry in SECTOR_WEIGHT_OVERRIDES (ensemble_agent.py)
      - recalibrate: refresh the sector's isotonic curve (evmax cleanup recalibrate --sector S)
      - add_feature: add one signal to the sector's model`,
  model_coverage: `A configured model is missing or gated on most of this sector's recent rows.
    First DIAGNOSE why with \`uv run evmax cleanup shadow show --why --category <sector>\` and the model's
    state file / seed date. Allowed kind ONLY: coverage_fix — a name-resolution alias, a seed/state
    loading bug, a stale state_filename, a season stamp, a missing seed-script league, etc.
    Return kind='none' (with the reason in note) when the gap is BY DESIGN or SELF-HEALING: an
    early-season MIN_GAMES / staleness guard that the weekly reseed will clear, a model that has
    weight 0 here, or an opponent the model correctly cannot price (FCS, ITF). Loosening a gate,
    a MIN_GAMES floor or a staleness guard is NOT a coverage fix here — that is a model-parameter
    change that needs its own Brier walk-forward (the value_audit path), so return kind='none'.`,
  passthrough: `This LIVE lane's blend sits within 0.5pp of the sharp line: no independent model
    signal. Do NOT propose a sharp_weight change (owned by \`cleanup adjust\`, judged on CLV) and do
    NOT propose a demotion (promotions/demotions are manual). The only allowed kind is coverage_fix,
    and only when a configured model SHOULD fire but does not (see the finding's top blockers and
    \`uv run evmax cleanup shadow show --why --category <sector>\`). If every configured model fires and
    the blend is passthrough because of sharp_weight dominance, that is by design: return kind='none'.
    The same gate rules as model_coverage apply (no gate / floor / guard loosening).`,
  close_capture: `Resolved Kalshi rows in this sector have NULL CLV (no Kalshi close was captured).
    DIAGNOSE first: is it a code bug in close capture / close lookup (resolver.close_lookup_ticker,
    archiver.get_kalshi_close_price, watch-closes sector/series coverage, backfill_clv selection), or
    an operational outage (launchd agent down, no archive snapshots)? Allowed kind ONLY: data_fix, and
    only for a code bug. An operational outage is not fixable by a PR: return kind='none'. Never relax
    the CLV rules (forward-only from entry, no close inside (T-30, tip) for unplaced Kalshi rows, no
    synthesized NO closes, no 0/1 devigged closes) to recover rows.`,
  calibration_bucket: `The blend is miscalibrated in this entry-price bucket AND worse than the sharp line
    there. Allowed kind ONLY: recalibrate — the sector's isotonic curve (evmax cleanup recalibrate
    --sector S, or the sector's leakage-guarded scripts/fit_<sector>_calibration.py). Never gate plays
    by price bucket and never change the EV threshold.`,
}

const MEASURE_GUIDANCE = {
  holdout_brier: `Preferred: run the general walk-forward with an explicit train/holdout split, e.g.
      uv run evmax backtest run --sectors <sector> --seasons <train>,<holdout>
    If a dedicated script fits better, use scripts/backtest_<sector>*.py (e.g.
    backtest_ncaaf_efficiency.py, backtest_nfl_efficiency.py, backtest_baseball_ab.py). Any
    prior/seed the model uses MUST be derived from seasons <= the train year only.
    Report baseline_holdout_brier (UNCHANGED blend) AND candidate_holdout_brier (WITH the change)
    on the SAME holdout split, plus coverage and n.`,
  rows_priced_frac: `This is a COVERAGE fix. It is judged on rows priced, NOT on Brier.
    Replay the finding's affected rows (the sector's moneyline-family ev_predictions rows from the
    finding's window: teams, event_date, market) through the target model agent's predict path, once
    on main (git stash is FORBIDDEN — use 'git worktree add' of main in your scratch dir, or
    'git show main:<file>') and once on the branch. A row is "priced" when the model returns a
    prediction at or above the ensemble confidence gate (0.45).
    Report coverage_before / coverage_after as fractions 0–1 of the SAME rows, newly_priced_n, and
    harm_z: on the newly priced rows that have resolved, the paired z of (blend Brier with the model −
    blend Brier without it), positive = worse (0 if none resolved). harm_z is a no-harm guard only.`,
  clv_coverage_frac: `This is a DATA fix. It is judged on CLV coverage, NOT on Brier.
    On the finding's resolved Kalshi rows, compute the fraction with a usable Kalshi close under the
    unchanged code (coverage_before) and under the changed code (coverage_after), WITHOUT writing any
    database: call the close-lookup functions directly from a Python script, or use
    'uv run evmax cleanup backfill-clv --recompute --dry-run' if it covers these rows. Report
    newly_priced_n = closes recovered, and rule_violations = recovered closes that break the CLV rules
    (a close earlier than the row's own entry, a close inside (T-30, tip) for an unplaced Kalshi row,
    an in-play print, a synthesized NO close, a 0/1 devigged close).`,
  bucket_calib_err_pp: `This is a RECALIBRATION. Fit the curve on the train window ONLY and score the
    holdout window: report bucket_err_before_pp / bucket_err_after_pp = |mean blend − realized win rate|
    in pp for the finding's entry-price bucket on the holdout, AND baseline_holdout_brier /
    candidate_holdout_brier on the whole holdout (a no-harm guard).`,
}

const INTEGRITY_CHECKS = {
  value_audit: `the change is a disguised gating/threshold move rather than a genuine model/blend improvement`,
  model_coverage: `the coverage gain comes from loosening a confidence gate, a MIN_GAMES floor, a staleness
    guard, or a name-resolution fallback that maps a team onto a DIFFERENT team (a namesake, an FBS
    school for an FCS one, a sibling roster) rather than fixing a genuine lookup/seed/state bug; or the
    replay rows were not the finding's rows`,
  passthrough: `the change moves sharp_weight / ensemble weights / modes, or the coverage gain comes from
    loosening a gate, floor or guard, or from a fuzzy name mapping onto a different team`,
  close_capture: `recovered closes violate the CLV rules (earlier than entry, inside (T-30, tip) for
    unplaced Kalshi rows, in-play, synthesized NO side = 1 − yes, 0/1 devigged), or the measurement
    wrote to a database`,
  calibration_bucket: `the curve was fitted on holdout rows, the change gates plays by price or edits the
    EV threshold, or the bucket gain comes with a worse whole-holdout Brier`,
}

// ---- pure gate functions (tested via tests/workflow_harness.mjs call mode) ----

// Exposes the constants to tests/test_model_improve_workflow.py (parity with improve_triggers.py).
function triggerConfig() {
  return {
    DEFAULT_TRIGGERS, TRIGGER_ORDER, TRIGGER_RULES, DENY_FILES, NOISE_FLOOR, MAX_ATTEMPTS,
    COVERAGE_MIN_GAIN, COVERAGE_MIN_NEW_N, HARM_Z_MAX, CLV_COVERAGE_MIN_GAIN, CLV_MIN_RECOVERED, CALIB_MIN_DROP_PP,
  }
}

// value_audit is always enabled (the original behavior can never be switched off by args);
// args.triggers ADDS triggers on top of it.
function resolveTriggers(args) {
  const raw = args && Array.isArray(args.triggers) ? args.triggers : []
  const unknown = raw.filter((t) => !(t in TRIGGER_RULES))
  const enabled = [...new Set([...DEFAULT_TRIGGERS, ...raw.filter((t) => t in TRIGGER_RULES)])]
  return { enabled, unknown }
}

// Re-check a trigger finding against TRIGGER_RULES. Never trusts the agent's `fires` alone.
function triggerEligible(f) {
  if (!f || typeof f !== 'object') return { ok: false, reason: 'not an object' }
  const rule = TRIGGER_RULES[f.trigger]
  if (!rule || f.trigger === 'value_audit') return { ok: false, reason: `unknown trigger ${f && f.trigger}` }
  if (!f.sector) return { ok: false, reason: 'no sector' }
  if (f.fires !== true) return { ok: false, reason: 'fires=false' }
  if (f.judge_metric !== rule.judge) return { ok: false, reason: `judge_metric ${f.judge_metric} != ${rule.judge}` }
  const n = Number(f.n)
  const v = Number(f.value)
  if (!Number.isFinite(n) || n < rule.min_n) return { ok: false, reason: `n=${f.n} < ${rule.min_n}` }
  if (!Number.isFinite(v)) return { ok: false, reason: 'value not numeric' }
  if (rule.max_value !== undefined && !(v < rule.max_value)) return { ok: false, reason: `value ${v} not < ${rule.max_value}` }
  if (rule.min_value !== undefined && !(v >= rule.min_value)) return { ok: false, reason: `value ${v} < ${rule.min_value}` }
  if (rule.min_value_exclusive !== undefined && !(v > rule.min_value_exclusive)) {
    return { ok: false, reason: `value ${v} not > ${rule.min_value_exclusive}` }
  }
  if (rule.min_abs_value !== undefined && !(Math.abs(v) >= rule.min_abs_value)) {
    return { ok: false, reason: `|value| ${Math.abs(v)} < ${rule.min_abs_value}` }
  }
  if (rule.min_abs_z !== undefined) {
    const z = Number(f.z)
    if (!Number.isFinite(z) || Math.abs(z) < rule.min_abs_z) return { ok: false, reason: `|z| ${f.z} < ${rule.min_abs_z}` }
  }
  return { ok: true, reason: 'ok' }
}

function laneKey(f) {
  return [f.trigger, f.sector, f.market_type || '', f.venue || '', f.model || '', f.bucket || ''].join('|')
}

// Build the attempt queue.
//   audit:     value-audit findings (only `actionable` ones are kept, most-negative z first)
//   triggers:  improve-triggers findings
//   enabled:   trigger names allowed to queue work; the rest go to `shadow`
// Returns {queue, shadow, rejected}. Value-audit findings lead; trigger findings follow round-robin
// across TRIGGER_ORDER so one run tries different classes. One finding per (trigger, sector) — the
// others for that sector ride along as `related` (e.g. ncaab_efficiency + ncaab_possession_sim share
// one state file). A passthrough lane whose sector already has a queued model_coverage finding is the
// same fix, so it rides along too.
function buildQueue(audit, triggers, enabled, max) {
  const on = new Set(enabled || DEFAULT_TRIGGERS)
  const rejected = []
  const va = (audit || [])
    .filter((f) => f && f.actionable)
    .map((f) => ({ ...f, trigger: 'value_audit' }))
    .sort((a, b) => (a.z_score ?? 0) - (b.z_score ?? 0))

  const perTrigger = Object.fromEntries(TRIGGER_ORDER.map((t) => [t, []]))
  const seen = new Set()
  for (const f of triggers || []) {
    const e = triggerEligible(f)
    if (!e.ok) { rejected.push({ trigger: f && f.trigger, sector: f && f.sector, reason: e.reason }); continue }
    const k = laneKey(f)
    if (seen.has(k)) continue
    seen.add(k)
    perTrigger[f.trigger].push({ ...f })
  }
  // collapse to one finding per (trigger, sector), highest priority first
  for (const t of TRIGGER_ORDER) {
    const bySector = new Map()
    for (const f of perTrigger[t].sort((a, b) => (b.priority ?? 0) - (a.priority ?? 0))) {
      if (!bySector.has(f.sector)) bySector.set(f.sector, { ...f, related: [] })
      else bySector.get(f.sector).related.push(f.evidence)
    }
    perTrigger[t] = [...bySector.values()]
  }
  const coverageSectors = new Set(perTrigger.model_coverage.map((f) => f.sector))
  if (on.has('model_coverage')) {
    perTrigger.passthrough = perTrigger.passthrough.filter((f) => {
      if (!coverageSectors.has(f.sector)) return true
      const host = perTrigger.model_coverage.find((c) => c.sector === f.sector)
      host.related.push(f.evidence)
      return false
    })
  }

  const ordered = []
  const maxLen = Math.max(0, ...TRIGGER_ORDER.map((t) => perTrigger[t].length))
  for (let i = 0; i < maxLen; i++) {
    for (const t of TRIGGER_ORDER) if (perTrigger[t][i]) ordered.push(perTrigger[t][i])
  }
  const live = [...va, ...ordered.filter((f) => on.has(f.trigger))]
  const shadow = ordered.filter((f) => !on.has(f.trigger))
  return { queue: live.slice(0, max ?? MAX_ATTEMPTS), shadow, rejected }
}

function kindAllowed(trigger, kind) {
  const rule = TRIGGER_RULES[trigger]
  return Boolean(rule && rule.kinds.includes(kind))
}

function deniedFiles(files) {
  return (files || []).map((p) => String(p).replace(/^\.\//, '')).filter((p) => DENY_FILES.includes(p))
}

const num = (x) => (typeof x === 'number' && Number.isFinite(x) ? x : null)

// G3: keep/revert from the measurement, by the trigger's own judge metric.
function signalVerdict(trigger, m) {
  const rule = TRIGGER_RULES[trigger]
  if (!rule) return { pass: false, delta: null, reason: `unknown trigger ${trigger}` }
  if (!m || m.ran !== true) return { pass: false, delta: null, reason: `measurement did not run: ${(m && m.note) || 'unknown'}` }
  if (rule.judge === 'holdout_brier') {
    const base = num(m.baseline_holdout_brier)
    const cand = num(m.candidate_holdout_brier)
    if (base === null || cand === null) return { pass: false, delta: null, reason: 'missing holdout Brier' }
    const d = base - cand
    return { pass: d >= NOISE_FLOOR, delta: d, reason: `Δbrier ${d.toFixed(4)} vs floor ${NOISE_FLOOR}` }
  }
  if (rule.judge === 'rows_priced_frac') {
    const b = num(m.coverage_before)
    const a = num(m.coverage_after)
    const n = num(m.newly_priced_n)
    const h = num(m.harm_z)
    if (b === null || a === null || n === null || h === null) return { pass: false, delta: null, reason: 'missing coverage_before/after, newly_priced_n or harm_z' }
    if (b < 0 || b > 1 || a < 0 || a > 1) return { pass: false, delta: null, reason: 'coverage is not a 0–1 fraction' }
    const d = a - b
    const reasons = []
    if (d < COVERAGE_MIN_GAIN) reasons.push(`gain ${d.toFixed(3)} < ${COVERAGE_MIN_GAIN}`)
    if (n < COVERAGE_MIN_NEW_N) reasons.push(`newly priced ${n} < ${COVERAGE_MIN_NEW_N}`)
    if (h >= HARM_Z_MAX) reasons.push(`harm_z ${h} ≥ ${HARM_Z_MAX}`)
    return { pass: reasons.length === 0, delta: d, reason: reasons.join('; ') || `rows priced +${d.toFixed(3)} on ${n} new rows` }
  }
  if (rule.judge === 'clv_coverage_frac') {
    const b = num(m.coverage_before)
    const a = num(m.coverage_after)
    const n = num(m.newly_priced_n)
    const v = num(m.rule_violations)
    if (b === null || a === null || n === null || v === null) return { pass: false, delta: null, reason: 'missing coverage_before/after, newly_priced_n or rule_violations' }
    if (b < 0 || b > 1 || a < 0 || a > 1) return { pass: false, delta: null, reason: 'coverage is not a 0–1 fraction' }
    const d = a - b
    const reasons = []
    if (d < CLV_COVERAGE_MIN_GAIN) reasons.push(`gain ${d.toFixed(3)} < ${CLV_COVERAGE_MIN_GAIN}`)
    if (n < CLV_MIN_RECOVERED) reasons.push(`recovered ${n} < ${CLV_MIN_RECOVERED}`)
    if (v !== 0) reasons.push(`${v} recovered close(s) break the CLV rules`)
    return { pass: reasons.length === 0, delta: d, reason: reasons.join('; ') || `CLV coverage +${d.toFixed(3)} (${n} closes)` }
  }
  if (rule.judge === 'bucket_calib_err_pp') {
    const eb = num(m.bucket_err_before_pp)
    const ea = num(m.bucket_err_after_pp)
    const base = num(m.baseline_holdout_brier)
    const cand = num(m.candidate_holdout_brier)
    if (eb === null || ea === null || base === null || cand === null) return { pass: false, delta: null, reason: 'missing bucket error or holdout Brier' }
    const d = eb - ea
    const reasons = []
    if (d < CALIB_MIN_DROP_PP) reasons.push(`bucket error drop ${d.toFixed(2)}pp < ${CALIB_MIN_DROP_PP}pp`)
    if (cand - base > NOISE_FLOOR) reasons.push(`holdout Brier worse by ${(cand - base).toFixed(4)}`)
    return { pass: reasons.length === 0, delta: d, reason: reasons.join('; ') || `bucket error −${d.toFixed(2)}pp` }
  }
  return { pass: false, delta: null, reason: `no signal rule for ${rule.judge}` }
}

// ==== WORKFLOW BODY ====

const { enabled, unknown } = resolveTriggers(args)
if (unknown.length) log(`ignoring unknown trigger(s): ${unknown.join(', ')}`)
log(`triggers enabled: [${enabled.join(', ')}] (others shadow-only)`)

// ---- N1 · OBSERVE ---------------------------------------------------------
phase('Observe')
const [audit, trig] = await parallel([
  () => agent(
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
  ),
  () => agent(
    `Run the evmax model-improve trigger sweep and return its findings VERBATIM.

    From the repo root, run:
      uv run evmax cleanup improve-triggers --json

    It prints one JSON object {findings, ran, failed}. Return it exactly: copy every finding with its
    fields unchanged (trigger, sector, market_type, venue, model, bucket, n, value, threshold, z, fires,
    judge_metric, priority, evidence). Do not add, drop, merge, re-rank or re-judge findings — the
    workflow re-checks every threshold itself. If the command fails, return {findings: [], failed:
    {"command": "<error>"}}. Read-only: change nothing.`,
    { label: 'improve-triggers', phase: 'Observe', schema: TRIGGERS_SCHEMA, agentType: 'general-purpose' },
  ),
])

const auditFindings = audit?.findings || []
const triggerFindings = trig?.findings || []
if (!trig) log('improve-triggers agent returned nothing — trigger findings unavailable this run.')
const { queue, shadow, rejected } = buildQueue(auditFindings, triggerFindings, enabled, MAX_ATTEMPTS)
const brief = (f) => ({ trigger: f.trigger, sector: f.sector, lane: [f.market_type, f.venue, f.model, f.bucket].filter(Boolean).join(' '), evidence: f.evidence })
if (shadow.length) log(`shadow triggers would queue ${shadow.length}: ${shadow.map((f) => `${f.trigger}:${f.sector}`).join(', ')}`)
if (rejected.length) log(`trigger findings rejected by the JS re-check: ${rejected.length}`)

if (args && args.dry_run) {
  return {
    decision: 'dry_run',
    enabled,
    queue: queue.map(brief),
    shadow_queue: shadow.map(brief),
    rejected,
    reviewed: { value_audit: auditFindings.length, triggers: triggerFindings.length },
  }
}

// ---- G1 · anything to act on? ---------------------------------------------
if (queue.length === 0) {
  log('G1: no queued gap — value-audit blends within noise and no enabled trigger fired. Stopping (a correct outcome).')
  return {
    stopped: 'no_actionable_gap',
    reviewed: auditFindings.length,
    enabled,
    shadow_queue: shadow.map(brief),
  }
}
log(`G1: will try up to ${queue.length} finding(s) until one validates → [${queue.map((f) => `${f.trigger}:${f.sector}`).join(', ')}]`)

// One attempt against a single finding. Returns { kept, dispose, spec, metrics, verdict } or a
// { skipped } reason. Ships (PR) and returns kept=true for the FIRST change that clears G2+G3.
async function tryFinding(target, idx) {
  const tag = `${target.trigger}:${target.sector}#${idx + 1}`
  const rule = TRIGGER_RULES[target.trigger]

  // ---- N2 · HYPOTHESIZE (folds G0 novelty check) --------------------------
  phase('Hypothesize')
  const spec = await agent(
    `Propose ONE change to close this evmax gap.

    Trigger: ${target.trigger} — judged on ${rule.judge}.
    Finding: ${JSON.stringify(target)}

    First read the ledger at ${LEDGER}. Each line is a prior attempt. Do NOT re-propose any
    change whose "decision" was "revert/no-change" for this sector — set novel=false if the only
    fix you can think of is already a reverted row, and explain in note.

    ${HYPOTHESIS_GUIDANCE[target.trigger]}

    Never touch: ${DENY_FILES.join(', ')} (modes, sharp_weight, firewalls, league shadow).
    Propose the smallest change that addresses the finding's mechanism. ONE change only.
    Do NOT implement anything yet — just return the ChangeSpec. Read-only.`,
    { label: `hypothesize:${tag}`, phase: 'Hypothesize', schema: CHANGE_SPEC_SCHEMA, agentType: 'general-purpose' },
  )

  if (!spec || spec.kind === 'none' || spec.novel === false) {
    log(`G0[${tag}]: no novel change (${spec?.note || 'none proposed'}) — next finding.`)
    return { skipped: 'no_novel_change', spec }
  }
  if (!kindAllowed(target.trigger, spec.kind)) {
    log(`G0[${tag}]: kind '${spec.kind}' is not allowed for trigger ${target.trigger} (allowed: ${rule.kinds.join(', ')}) — next finding.`)
    return { skipped: 'kind_not_allowed', spec }
  }

  // ---- N3 + N4 · ISOLATE + IMPLEMENT (commit to the branch immediately) ----
  phase('Implement')
  const impl = await agent(
    `Implement exactly ONE change in the evmax repo, on a fresh git branch, and COMMIT it
    to that branch so the working tree is never used to hold the diff.

    ChangeSpec: ${JSON.stringify(spec)}

    ${GIT_SAFETY}

    Steps:
      1. Record the starting branch. Create the throwaway branch:
           git switch -c model-improve/${spec.sector}-${spec.kind}
         (uncommitted files from other tasks harmlessly ride along — do not touch them.)
      2. Make ONLY the change in the spec. Do NOT touch tests, fixtures, backtest scripts,
         holdout data, or any success metric — that is reward hacking and the next stage rejects it.
         Never edit ${DENY_FILES.join(', ')}.
         If the change needs a state re-seed, run the sector's seed script; never hand-edit state JSON.
      3. Commit ONLY the files you changed:
           git add <exact paths>   &&   git commit -m "<summary>  (model-improve, pre-validation)"
         Commit message ends with:  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
      4. Leave the checkout ON the branch with the change committed (clean working tree for our paths).
         Return committed=true only if the commit succeeded; else committed=false with aborted_reason,
         and switch back to the starting branch WITHOUT any destructive command.
         Return files_changed = the exact committed paths.`,
    { label: `implement:${tag}`, phase: 'Implement', schema: IMPLEMENT_SCHEMA, agentType: 'implementer' },
  )

  if (!impl || !impl.committed) {
    log(`Implement[${tag}] did not commit (${impl?.aborted_reason || impl?.note || 'no diff'}) — next finding.`)
    return { skipped: 'implement_failed', spec, impl }
  }

  const denied = deniedFiles(impl.files_changed)

  // ---- N5 · MEASURE (change committed on branch) ---------------------------
  phase('Measure')
  const metrics = denied.length
    ? { ran: false, note: `committed a denied file: ${denied.join(', ')}` }
    : await agent(
        `Measure this evmax change. The change is committed on branch ${impl.branch}, which is
        currently checked out and active. The signal must not see the future.

        Sector: ${spec.sector}. Trigger: ${target.trigger}. Judge metric: ${rule.judge}.
        Finding: ${JSON.stringify(target)}
        Change under test: ${JSON.stringify(spec)}

        ${MEASURE_GUIDANCE[rule.judge]}

        Point at the main checkout's databases READ-ONLY when you need them:
          EVMAX_DB_DIR=/Users/ktzeng/Projects/evmax/data EVMAX_DB_READONLY=1
        If no valid measurement can be run (missing data, coverage < 90%), set ran=false and
        explain — do not fabricate numbers. Do not run the full pytest suite. Read-only w.r.t. git.`,
        { label: `measure:${tag}`, phase: 'Measure', schema: METRICS_SCHEMA, agentType: 'general-purpose' },
      )

  // ---- G2 · INTEGRITY (before signal, always) -----------------------------
  phase('Integrity')
  const verdict = metrics?.ran
    ? await agent(
        `Adversarially review this evmax model-improvement iteration for reward hacking.

        Trigger: ${target.trigger} (judged on ${rule.judge}).
        ChangeSpec: ${JSON.stringify(spec)}
        Branch: ${impl.branch} (inspect the actual committed diff: git diff main...${impl.branch})
        Measurement: ${JSON.stringify(metrics)}

        REJECT (accept=false) if the diff edits any test, fixture, backtest script, holdout data,
        or the metric definition; if the model leaks holdout/future data into training or the prior;
        if outputs are hardcoded; if it touches ${DENY_FILES.join(', ')}; or if
        ${INTEGRITY_CHECKS[target.trigger]}.
        Inspect the real diff, do not trust the summary alone. Default to reject when uncertain. Read-only.`,
        { label: `integrity:${tag}`, phase: 'Integrity', schema: VERDICT_SCHEMA, agentType: 'iteration-reviewer' },
      )
    : { accept: false, reason: `measurement did not run: ${metrics?.note || 'unknown'}` }

  // ---- G3 · SIGNAL (pure arithmetic, per judge metric) --------------------
  const sig = signalVerdict(target.trigger, metrics)
  const keep = Boolean(verdict?.accept) && sig.pass

  log(
    `Gates[${tag}] → integrity=${verdict?.accept ? 'ACCEPT' : 'REJECT'} · ` +
      `signal ${rule.judge}: ${sig.reason} → ${sig.pass ? 'PASS' : 'FAIL'} · ` +
      `decision=${keep ? 'KEEP' : 'REVERT'}`,
  )

  // ---- N6 · DISPOSE (no destructive working-tree commands, ever) -----------
  phase('Dispose')
  const decisionCtx = {
    trigger: target.trigger,
    judge_metric: rule.judge,
    sector: spec.sector,
    loop: `model-improve-${spec.sector}`,
    branch: impl.branch,
    spec,
    metrics,
    verdict,
    signal: sig,
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
              The body LEADS with the change and its trigger: what moved, the judge metric
              (${rule.judge}) before→after, the integrity verdict, and why it is not overfitting or a
              disguised gate change.
              PR body ends with:  🤖 Generated with [Claude Code](https://claude.com/claude-code)
              If the push succeeds but 'gh pr create' fails (auth/network), report the branch name +
              GitHub compare URL so the PR can be opened by hand — the work is preserved. Do NOT merge.
           3. Append one line to ${LEDGER} (stamp ts via: date +%F):
              {"ts":"<today>","loop":"model-improve-${spec.sector}","phase":"attempt","trigger":"${target.trigger}",
               "hyp":"<rationale>","change":"<params>","decision":"keep-PR",
               "metric":"${rule.judge}","value":<measured after-value>,"note":"Δ<signal delta>; PR <url>","sha":"<git rev-parse --short HEAD>"}
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
              {"ts":"<today>","loop":"model-improve-${spec.sector}","phase":"attempt","trigger":"${target.trigger}",
               "hyp":"<rationale>","change":"<params>","decision":"revert/no-change",
               "note":"<why: integrity reject reason OR the signal reason>","sha":"<current short sha on main>"}
         Return action=reverted.`,
    { label: `dispose:${keep ? 'keep' : 'revert'}:${tag}`, phase: 'Dispose', schema: DISPOSE_SCHEMA, agentType: 'implementer' },
  )

  const shipped = keep && dispose?.action === 'kept'
  return { kept: shipped, dispose, spec, metrics, verdict, signal: sig }
}

// ---- attempt queued findings until one validates + ships -------------------
const attempts = []
for (let i = 0; i < queue.length; i++) {
  const r = await tryFinding(queue[i], i)
  attempts.push({
    trigger: queue[i].trigger,
    sector: queue[i].sector,
    gap: queue[i].gap_type || queue[i].trigger,
    outcome: r.kept ? 'kept' : r.skipped || 'reverted',
  })
  if (r.kept) {
    return {
      decision: 'kept',
      trigger: queue[i].trigger,
      sector: r.spec.sector,
      change: r.spec.params,
      judge_metric: TRIGGER_RULES[queue[i].trigger].judge,
      delta: r.signal?.delta ?? null,
      holdout_delta: TRIGGER_RULES[queue[i].trigger].judge === 'holdout_brier' ? r.signal?.delta ?? null : null,
      pr_url: r.dispose?.pr_url || null,
      ledger_written: Boolean(r.dispose?.ledger_written),
      attempts,
      shadow_queue: shadow.map(brief),
    }
  }
}

log(`All ${attempts.length} queued finding(s) tried; none produced a validated change. No PR this run.`)
return { decision: 'no_validated_change', attempts, shadow_queue: shadow.map(brief) }
