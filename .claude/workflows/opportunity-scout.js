export const meta = {
  name: 'opportunity-scout',
  description: 'Discovery run: research-journal, competitive-analysis and modeling agents propose +EV opportunities; a synthesizer writes pre-registered briefs; a scope validator gates them (hard fails enforced in code); a backtester tests the top allowed briefs offline, an integrity reviewer checks each test, and a deterministic signal gate grades it. Writes nothing to the repo — returns structured results for scripts/opportunity_ledger.py ingest.',
  whenToUse: 'Invoked by the /opportunities command. Needs args {date, snapshot_path, db_dir} (+ optional focus, scratch_dir, top_k, top_n, research_agents).',
  phases: [
    { title: 'Propose', detail: 'research journal ║ competitive analysis ║ modeling' },
    { title: 'Synthesize', detail: 'dedup, graveyard G0, pre-registered briefs, rank, top-K cap' },
    { title: 'Scope', detail: 'validator G1; hard fails + pre-registration enforced in code; one revise round' },
    { title: 'Backtest', detail: 'pre-registered offline test per allowed brief (top-N)' },
    { title: 'Integrity', detail: 'iteration-reviewer G2a, then signal G2b in code' },
  ],
}

// Design: docs/opportunity-workflow-scope.md. Gate logic below the schemas is pure JS so the
// orchestrator — not an LLM — decides every transition. tests/test_opportunity_workflows.py
// runs this file under a stub harness (tests/workflow_harness.mjs) to pin that logic.

// ---------------------------------------------------------------------------
// Edge contracts (validated at the tool-call layer)
// ---------------------------------------------------------------------------

const LEVERS = ['model', 'pricing', 'execution', 'coverage', 'venue', 'data', 'reliability', 'sizing']

const CANDIDATE = {
  type: 'object',
  properties: {
    title: { type: 'string' },
    lever: { type: 'string', enum: LEVERS },
    sectors: { type: 'array', items: { type: 'string' } },
    market_types: { type: 'array', items: { type: 'string' } },
    venues: { type: 'array', items: { type: 'string' } },
    hypothesis: { type: 'string' },
    edge_mechanism: { type: 'string', description: 'who is on the other side and why the price is wrong' },
    evidence: { type: 'string', description: 'the numbers / claim that support it' },
    refs: { type: 'array', items: { type: 'string' }, description: 'URLs, DOIs, file:line, or exact commands' },
    data_needed: { type: 'array', items: { type: 'string' } },
    graveyard_matches: { type: 'array', items: { type: 'string' } },
    why_different: { type: 'string', description: 'required when graveyard_matches is non-empty' },
  },
  required: ['title', 'lever', 'hypothesis', 'edge_mechanism', 'refs'],
}

const PROPOSER_SCHEMA = {
  type: 'object',
  properties: {
    candidates: { type: 'array', items: CANDIDATE },
    notes: { type: 'string' },
    journal_entries: {
      type: 'array',
      description: 'research agents only: every source read',
      items: {
        type: 'object',
        properties: {
          url: { type: 'string' }, title: { type: 'string' }, venue_year: { type: 'string' },
          claim: { type: 'string' }, relevance: { type: 'string', enum: ['high', 'medium', 'low'] },
          maps_to_lever: { type: 'string' }, fetched: { type: 'boolean' },
        },
        required: ['url', 'title', 'fetched'],
      },
    },
    landscape: {
      type: 'object',
      description: 'competitive agent only',
      properties: {
        competitors: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              name: { type: 'string' }, type: { type: 'string' }, url: { type: 'string' },
              approach: { type: 'string' }, has_we_lack: { type: 'string' }, we_have_they_lack: { type: 'string' },
            },
            required: ['name'],
          },
        },
        venue_gaps: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              market: { type: 'string' }, venue: { type: 'string' },
              observed_activity: { type: 'string' }, wiring_cost: { type: 'string' },
            },
            required: ['market'],
          },
        },
        diff_vs_previous: { type: 'string' },
      },
    },
  },
  required: ['candidates', 'notes'],
}

const PREREG = {
  type: 'object',
  properties: {
    metric: { type: 'string', enum: ['clv_pp_net_fee', 'roi_net_fee', 'open_close_slope', 'brier_paired_vs_sharp', 'match_rate', 'coverage'] },
    threshold: { type: ['number', 'null'], description: 'the value the metric must strictly beat; null = use the default' },
    z_min: { type: ['number', 'null'] },
    min_n_games: { type: 'integer' },
    train_window: { type: 'string' },
    holdout_window: { type: 'string' },
    comparator: { type: 'string', description: 'what the candidate is compared against (e.g. scan-time entry, current blend, Pinnacle close)' },
    command: { type: 'string', description: 'the exact evmax command or throwaway-script plan that produces the metric' },
    declustering: { type: 'string', description: 'unit of independence, normally game (shadow.game_key)' },
  },
  required: ['metric', 'min_n_games', 'holdout_window', 'comparator', 'command'],
}

const BRIEF = {
  type: 'object',
  properties: {
    id: { type: 'string', description: 'keep an existing id unchanged; new briefs may leave it empty' },
    title: { type: 'string' },
    lever: { type: 'string', enum: LEVERS },
    sectors: { type: 'array', items: { type: 'string' } },
    market_types: { type: 'array', items: { type: 'string' } },
    venues: { type: 'array', items: { type: 'string' } },
    hypothesis: { type: 'string' },
    edge_mechanism: { type: 'string' },
    sources: {
      type: 'array',
      items: { type: 'object', properties: { kind: { type: 'string', enum: ['research', 'competitive', 'internal'] }, ref: { type: 'string' } }, required: ['kind', 'ref'] },
    },
    data: {
      type: 'array',
      items: {
        type: 'object',
        properties: { name: { type: 'string' }, public: { type: 'boolean' }, auth_required: { type: 'boolean' }, access_method: { type: 'string' } },
        required: ['name', 'public', 'auth_required'],
      },
    },
    preregistration: PREREG,
    size: { type: 'string', enum: ['S', 'M', 'L'] },
    build_plan: { type: 'string' },
    files_likely_touched: { type: 'array', items: { type: 'string' } },
    risks: { type: 'array', items: { type: 'string' } },
    blast_radius: { type: 'string', description: 'which live lanes this could touch, and how it stays shadow / default-off' },
    graveyard_check: {
      type: 'object',
      properties: { matched_ids: { type: 'array', items: { type: 'string' } }, why_different: { type: 'string' } },
      required: ['matched_ids'],
    },
    revisit_if: { type: 'string', description: 'what new evidence would justify retrying if this is refuted' },
    rank_score: { type: 'number', description: '0-10 expected value per unit of build effort' },
  },
  required: ['title', 'lever', 'sectors', 'hypothesis', 'edge_mechanism', 'sources', 'preregistration', 'size', 'graveyard_check', 'rank_score'],
}

const BRIEFS_SCHEMA = {
  type: 'object',
  properties: {
    briefs: { type: 'array', items: BRIEF },
    dropped: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          title: { type: 'string' }, source: { type: 'string' }, reason: { type: 'string' },
          lever: { type: 'string' },
          graveyard_id: { type: 'string', description: 'set when dropped because it matches this graveyard entry' },
        },
        required: ['title', 'reason'],
      },
    },
  },
  required: ['briefs', 'dropped'],
}

const HARD_FAILS = [
  'requires_auth_or_account',
  'requires_antibot_bypass',
  'requires_tos_violation_or_paywall',
  'touches_live_pricing_without_flag',
  'changes_bankroll_or_mode',
  'no_measurable_signal',
  'in_graveyard_without_new_evidence',
  'edits_eval_or_holdout',
]
const SCORE_KEYS = ['edge_mechanism', 'net_of_fee', 'data_availability', 'time_to_evidence', 'build_size', 'architecture_fit', 'upside']

const SCOPE_SCHEMA = {
  type: 'object',
  properties: {
    verdicts: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          verdict: { type: 'string', enum: ['allow', 'revise', 'reject'] },
          hard_fails: {
            type: 'object',
            properties: Object.fromEntries(HARD_FAILS.map(k => [k, { type: 'boolean' }])),
            required: HARD_FAILS,
          },
          scores: {
            type: 'object',
            properties: Object.fromEntries(SCORE_KEYS.map(k => [k, { type: 'integer', minimum: 0, maximum: 3 }])),
            required: SCORE_KEYS,
          },
          required_changes: { type: 'array', items: { type: 'string' } },
          graveyard_ids: { type: 'array', items: { type: 'string' } },
          reason: { type: 'string' },
        },
        required: ['id', 'verdict', 'hard_fails', 'scores', 'reason'],
      },
    },
  },
  required: ['verdicts'],
}

const METRICS_SCHEMA = {
  type: 'object',
  properties: {
    ran: { type: 'boolean', description: 'false when the registered test could not run as written' },
    metric: { type: 'string', description: 'echo the pre-registered metric' },
    value: { type: ['number', 'null'], description: 'the metric on the holdout (net of fees for CLV/ROI)' },
    z_improvement: { type: ['number', 'null'], description: 'game-clustered z (t for open_close_slope); POSITIVE = better in the metric’s good direction' },
    n_games: { type: ['integer', 'null'] },
    n_rows: { type: ['integer', 'null'] },
    ci_low: { type: ['number', 'null'] },
    ci_high: { type: ['number', 'null'] },
    train_window: { type: 'string' },
    holdout_window: { type: 'string' },
    command: { type: 'string', description: 'exactly what was run' },
    leakage_checks: {
      type: 'object',
      properties: {
        utc_et_day: { type: 'boolean' }, point_in_time: { type: 'boolean' },
        no_future_close: { type: 'boolean' }, declustered_by_game: { type: 'boolean' },
      },
      required: ['utc_et_day', 'point_in_time', 'no_future_close', 'declustered_by_game'],
    },
    scripts_written: { type: 'array', items: { type: 'string' } },
    secondary: { type: 'array', items: { type: 'string' }, description: 'extra diagnostics, never gated' },
    note: { type: 'string' },
  },
  required: ['ran', 'metric', 'command', 'leakage_checks', 'note'],
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

// METRIC_RULES:BEGIN (strict JSON — scripts/opportunity_ledger.py parses this block)
const METRIC_RULES = {
  "clv_pp_net_fee": {"direction": "higher", "threshold": 0.0, "z_min": 1.64, "min_n_games": 30},
  "roi_net_fee": {"direction": "higher", "threshold": 0.0, "z_min": 1.64, "min_n_games": 30},
  "open_close_slope": {"direction": "higher", "threshold": 0.0, "z_min": 2.0, "min_n_games": 30},
  "brier_paired_vs_sharp": {"direction": "lower", "threshold": -0.002, "z_min": 1.64, "min_n_games": 200},
  "match_rate": {"direction": "higher", "threshold": null, "z_min": null, "min_n_games": 30},
  "coverage": {"direction": "higher", "threshold": null, "z_min": null, "min_n_games": 30}
}
// METRIC_RULES:END

// ---------------------------------------------------------------------------
// Pure gate functions (no agent calls; pinned by tests)
// ---------------------------------------------------------------------------

function clampInt(v, dflt, lo, hi) {
  const n = Number.isInteger(v) ? v : dflt
  return Math.max(lo, Math.min(hi, n))
}

function slugify(s) {
  return String(s || 'opportunity').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 48).replace(/-+$/g, '') || 'opportunity'
}

function compactDate(d) {
  return String(d).replace(/-/g, '').slice(0, 8)
}

// Deterministic ids: <slug(title)>-<YYYYMMDD>, de-duplicated with -2, -3 …
function assignIds(briefs, date, taken) {
  const used = new Set(taken || [])
  return briefs.map(b => {
    const base = `${slugify(b.title)}-${compactDate(date)}`
    let id = base
    let k = 2
    while (used.has(id)) { id = `${slugify(b.title)}-${k}-${compactDate(date)}`; k++ }
    used.add(id)
    return { ...b, id }
  })
}

// Mirrors scripts/opportunity_ledger.py::preregistration_errors. A brief may tighten a
// default but never loosen it; a null-default metric needs an explicit threshold.
function preregErrors(prereg) {
  if (!prereg || typeof prereg !== 'object') return ['preregistration missing']
  const rule = METRIC_RULES[prereg.metric]
  if (!rule) return [`metric ${JSON.stringify(prereg.metric)} is not one of ${Object.keys(METRIC_RULES).sort().join(', ')}`]
  const errors = []
  for (const key of ['holdout_window', 'command', 'comparator']) if (!prereg[key]) errors.push(`${key} missing`)
  const t = prereg.threshold
  if (t === null || t === undefined) {
    if (rule.threshold === null) errors.push(`${prereg.metric} needs an explicit numeric threshold`)
  } else if (typeof t !== 'number' || !Number.isFinite(t)) {
    errors.push('threshold must be a number')
  } else if (rule.threshold !== null) {
    const looser = rule.direction === 'higher' ? t < rule.threshold : t > rule.threshold
    if (looser) errors.push(`threshold ${t} is looser than the default ${rule.threshold}`)
  }
  const z = prereg.z_min
  if (rule.z_min !== null && z !== null && z !== undefined && z < rule.z_min) errors.push(`z_min ${z} is looser than the default ${rule.z_min}`)
  const n = prereg.min_n_games
  if (!Number.isInteger(n)) errors.push('min_n_games must be an integer')
  else if (n < rule.min_n_games) errors.push(`min_n_games ${n} is looser than the default ${rule.min_n_games}`)
  return errors
}

// The rule actually applied at G2b: defaults, tightened by the brief, never loosened
// (clamped here even if validation was somehow bypassed).
function effectiveRule(prereg) {
  const base = METRIC_RULES[prereg.metric]
  const t = typeof prereg.threshold === 'number' ? prereg.threshold : base.threshold
  const threshold = base.threshold === null ? t
    : (base.direction === 'higher' ? Math.max(base.threshold, t) : Math.min(base.threshold, t))
  const zMin = base.z_min === null ? (typeof prereg.z_min === 'number' ? prereg.z_min : null)
    : Math.max(base.z_min, typeof prereg.z_min === 'number' ? prereg.z_min : base.z_min)
  const minN = Math.max(base.min_n_games, Number.isInteger(prereg.min_n_games) ? prereg.min_n_games : base.min_n_games)
  return { direction: base.direction, threshold, z_min: zMin, min_n_games: minN }
}

function hardFailsOf(verdict) {
  const hf = (verdict && verdict.hard_fails) || {}
  return HARD_FAILS.filter(k => hf[k] === true)
}

function scoreSum(verdict) {
  const s = (verdict && verdict.scores) || {}
  return SCORE_KEYS.reduce((acc, k) => acc + (Number.isInteger(s[k]) ? s[k] : 0), 0)
}

// G1. Returns { decision: 'allow'|'revise'|'reject', status, reason }. The validator's
// verdict is advisory; hard fails, pre-registration errors and the one-revision limit are not.
function applyScopeGate(verdict, preErrs, round) {
  if (!verdict) return { decision: 'reject', status: 'REJECTED_SCOPE', reason: 'validator returned no verdict for this brief' }
  const fails = hardFailsOf(verdict)
  if (fails.length) {
    const novelty = fails.length === 1 && fails[0] === 'in_graveyard_without_new_evidence'
    return { decision: 'reject', status: novelty ? 'REJECTED_NOVELTY' : 'REJECTED_SCOPE', reason: `hard fail: ${fails.join(', ')}` }
  }
  let decision = verdict.verdict
  let reason = verdict.reason || ''
  if (!['allow', 'revise', 'reject'].includes(decision)) return { decision: 'reject', status: 'REJECTED_SCOPE', reason: `invalid verdict ${JSON.stringify(decision)}` }
  if (decision === 'allow' && preErrs && preErrs.length) {
    decision = 'revise'
    reason = `pre-registration errors: ${preErrs.join('; ')}`
  }
  if (decision === 'revise' && round >= 2) return { decision: 'reject', status: 'REJECTED_SCOPE', reason: `still needs revision after one round: ${reason}` }
  if (decision === 'reject') return { decision, status: 'REJECTED_SCOPE', reason }
  return { decision, status: decision === 'allow' ? 'ALLOWED_UNTESTED' : 'PROPOSED', reason }
}

// G2b. metrics come from the backtester; integrity (G2a) has already accepted them.
function signalVerdict(prereg, m) {
  if (!m || m.ran !== true) return { status: 'NOT_RUN', reason: (m && m.note) || 'backtest did not run' }
  if (m.metric && m.metric !== prereg.metric) return { status: 'INVALID', reason: `backtest reported ${m.metric}; pre-registered ${prereg.metric}` }
  if (typeof m.value !== 'number' || !Number.isFinite(m.value)) return { status: 'NOT_RUN', reason: 'backtest returned no numeric value' }
  if (!Number.isInteger(m.n_games)) return { status: 'INVALID', reason: 'backtest returned no game count' }
  const rule = effectiveRule(prereg)
  if (rule.threshold === null) return { status: 'INVALID', reason: 'no threshold to compare against' }
  if (rule.z_min !== null && (typeof m.z_improvement !== 'number' || !Number.isFinite(m.z_improvement))) {
    return { status: 'INVALID', reason: 'backtest returned no z for a metric that requires one' }
  }
  const beats = rule.direction === 'higher' ? m.value > rule.threshold : m.value < rule.threshold
  const side = `${m.value} vs ${rule.direction === 'higher' ? '>' : '<'} ${rule.threshold}`
  if (m.n_games < rule.min_n_games) return { status: 'UNDERPOWERED', reason: `${m.n_games} games < ${rule.min_n_games} required (${side})` }
  if (!beats) return { status: 'REFUTED', reason: `${side} on ${m.n_games} games` }
  if (rule.z_min !== null && m.z_improvement < rule.z_min) {
    return { status: 'UNDERPOWERED', reason: `${side} but z ${m.z_improvement} < ${rule.z_min}` }
  }
  return { status: 'SUPPORTED', reason: `${side}${rule.z_min !== null ? `, z ${m.z_improvement} ≥ ${rule.z_min}` : ''} on ${m.n_games} games` }
}

function finalize(brief, fields) {
  return { ...brief, ...fields }
}

// ==== WORKFLOW BODY ====

const A = args || {}
if (!A.date || !A.snapshot_path || !A.db_dir) {
  return { error: 'args.date, args.snapshot_path and args.db_dir are required (run via /opportunities)' }
}
const DATE = A.date
const FOCUS = A.focus || ''
const TOP_K = clampInt(A.top_k, 5, 1, 8)
const TOP_N = clampInt(A.top_n, 2, 0, 4)
const N_RESEARCH = clampInt(A.research_agents, 1, 1, 2)
const SCRATCH = A.scratch_dir || `.claude/opportunity-scratch/${DATE}`
const DB_ENV = `EVMAX_DB_DIR=${A.db_dir} EVMAX_DB_READONLY=1`
const notes = []

const CONTEXT = `Run date: ${DATE}. Focus: ${FOCUS || 'the whole project'}.
Context snapshot (JSON, pretty-printed — Read it with offsets; it is the shared ground truth for this run):
  ${A.snapshot_path}
It holds: meta (db_dir, how_to_query_dbs), categories (base/effective modes), promotion_board, value_audit,
integrity, kalshi_series (wired/stale/unwired), open_prs, recent_commits, eval_docs, graveyard (ideas already
tested — read it), ledger (earlier opportunity runs), research_sources_seen, landscape_previous, memory_index.
Design of this workflow: docs/opportunity-workflow-scope.md.`

// ---- Propose (barrier: the synthesizer needs every proposer's output) ----
phase('Propose')
const LENSES = N_RESEARCH === 2
  ? [
      { key: 'research', lens: 'market microstructure and prediction-market pricing: favorite–longshot bias, maker/taker economics, closing-line efficiency, liquidity and stale quotes, cross-venue price formation' },
      { key: 'research_2', lens: 'sports modeling and information timing: ratings, player props, injuries and lineup news, in-season priors, calibration of specific price buckets' },
    ]
  : [{ key: 'research', lens: 'both (1) market microstructure and prediction-market pricing and (2) sports modeling and information timing' }]

const researchPrompt = l => `${CONTEXT}

You are the Research Journal agent. Lens: ${l.lens}.
Find published or practitioner evidence for +EV mechanisms on Kalshi / Polymarket US that evmax does not
use yet. Skip every URL in research_sources_seen. Verify each source by fetching it. Return journal_entries
for every source you read and candidates only for findings that name an edge mechanism, survive fees,
and are measurable with evmax data or a shadow lane. Respect the graveyard.`

const competitivePrompt = `${CONTEXT}

You are the EV Competitive Analysis agent. Compare evmax against commercial +EV tools, open-source
prediction-market / sports-betting projects, and the venue landscape (use the snapshot's kalshi_series —
do not re-probe Kalshi). Report changes vs landscape_previous. Public pages only. Return landscape and
candidates for gaps that name an edge mechanism and a way to measure it here.`

const modelingPrompt = `${CONTEXT}

You are the Modeling agent. Mine evmax's own measurements for model, pricing, coverage and execution
opportunities. Drill down with read-only commands prefixed by: ${DB_ENV}
Every candidate needs internal evidence (exact command + numbers, or snapshot section + field).`

const proposerCalls = [
  ...LENSES.map(l => () => agent(researchPrompt(l), { label: `propose:${l.key}`, phase: 'Propose', schema: PROPOSER_SCHEMA, agentType: 'opportunity-researcher' })),
  () => agent(competitivePrompt, { label: 'propose:competitive', phase: 'Propose', schema: PROPOSER_SCHEMA, agentType: 'opportunity-competitor' }),
  () => agent(modelingPrompt, { label: 'propose:modeling', phase: 'Propose', schema: PROPOSER_SCHEMA, agentType: 'opportunity-modeler' }),
]
const proposerKeys = [...LENSES.map(l => l.key), 'competitive', 'modeling']
const proposedRaw = await parallel(proposerCalls)
const proposers = {}
proposerKeys.forEach((k, i) => {
  proposers[k] = proposedRaw[i] || null
  if (!proposedRaw[i]) notes.push(`${k} agent returned nothing`)
})

const candidates = proposerKeys.flatMap(k => ((proposers[k] && proposers[k].candidates) || []).map(c => ({ ...c, source: k })))
log(`Proposers returned ${candidates.length} candidate(s): ` + proposerKeys.map(k => `${k} ${((proposers[k] && proposers[k].candidates) || []).length}`).join(', '))

const RUN = { date: DATE, focus: FOCUS, snapshot_path: A.snapshot_path, db_dir: A.db_dir, top_k: TOP_K, top_n: TOP_N, research_agents: N_RESEARCH }
if (!candidates.length) {
  return { run: RUN, proposers, briefs: [], dropped: [], notes: [...notes, 'no candidates proposed — nothing to synthesize'] }
}

// ---- Synthesize ----
phase('Synthesize')
const synthPrompt = `${CONTEXT}

You are the synthesizer — the orchestrator's only judgment node. You receive every proposer candidate:
${JSON.stringify(candidates, null, 1)}

1. Merge duplicates (same lever + same mechanism + same lane) into one brief; keep every source.
2. Drop candidates that match a graveyard entry without new evidence meeting its revisit_if — put them in
   "dropped" with graveyard_id set. Drop anything already shipped in evmax (check CLAUDE.md) with a reason.
3. Write every survivor as an Opportunity Brief with a COMPLETE pre-registration chosen BEFORE any test:
   - metric: clv_pp_net_fee | roi_net_fee | open_close_slope | brier_paired_vs_sharp | match_rate | coverage.
     Model levers prefer open_close_slope or clv_pp_net_fee; brier_paired_vs_sharp is screening only (the
     0.85 sharp anchor absorbs standalone Brier gains).
   - Defaults (you may TIGHTEN, never loosen): ${JSON.stringify(METRIC_RULES)}
     "threshold" is the value the metric must strictly beat; match_rate/coverage need an explicit threshold.
   - train_window / holdout_window (walk-forward, untouched holdout), comparator, declustering (game), and a
     concrete command using existing evmax lenses or a scoped throwaway script. DB-reading commands use: ${DB_ENV}
4. rank_score 0–10 = expected value per unit of build effort. Be honest; most ideas here fail.
Return briefs (leave id empty) and dropped. Do not invent sources: every brief source must come from a candidate's refs.`

const synth = await agent(synthPrompt, { label: 'synthesize', phase: 'Synthesize', schema: BRIEFS_SCHEMA, agentType: 'general-purpose' })
const dropped = []
for (const d of (synth && synth.dropped) || []) dropped.push(d)
let briefs = assignIds((synth && synth.briefs) || [], DATE)
assignIds(dropped, DATE, briefs.map(b => b.id)).forEach((d, i) => { dropped[i] = { ...dropped[i], id: d.id } })
if (!synth) notes.push('synthesizer returned nothing')

briefs.sort((a, b) => (b.rank_score || 0) - (a.rank_score || 0))
const capped = briefs.slice(TOP_K)
briefs = briefs.slice(0, TOP_K)
for (const b of capped) dropped.push({ id: b.id, title: b.title, lever: b.lever, source: 'synthesizer', reason: `top-K cap (${TOP_K}); rank_score ${b.rank_score}` })
if (capped.length) log(`Top-K cap dropped ${capped.length} brief(s): ${capped.map(b => b.id).join(', ')}`)
if (!briefs.length) {
  return { run: RUN, proposers, briefs: [], dropped, notes: [...notes, 'synthesizer produced no briefs'] }
}

// ---- Scope (G1) ----
async function runScope(list, round) {
  const preErrs = Object.fromEntries(list.map(b => [b.id, preregErrors(b.preregistration)]))
  const res = await agent(`${CONTEXT}

You are the Scope Validator (gate G1), round ${round} of at most 2. Judge each brief below allow / revise /
reject. Return one verdict per id, with all hard_fails booleans and all scores. Read-only commands that
touch the databases use: ${DB_ENV}
Pre-registration problems the orchestrator already found (it will force at least "revise" for these):
${JSON.stringify(preErrs, null, 1)}

Briefs:
${JSON.stringify(list, null, 1)}`, { label: round === 1 ? 'scope' : 'scope:revise', phase: 'Scope', schema: SCOPE_SCHEMA, agentType: 'opportunity-validator' })
  const byId = Object.fromEntries(((res && res.verdicts) || []).map(v => [v.id, v]))
  return list.map(b => ({ brief: b, verdict: byId[b.id] || null, gate: applyScopeGate(byId[b.id], preErrs[b.id], round) }))
}

phase('Scope')
let judged = await runScope(briefs, 1)

const toRevise = judged.filter(j => j.gate.decision === 'revise')
if (toRevise.length) {
  phase('Synthesize')
  const rev = await agent(`${CONTEXT}

You are the synthesizer, revision round. The scope validator asked for changes to these briefs. Apply every
required change, keep each brief's "id" EXACTLY as given, and keep the pre-registration honest (tighten,
never loosen: ${JSON.stringify(METRIC_RULES)}). Return only the revised briefs (dropped may be empty).
${JSON.stringify(toRevise.map(j => ({ brief: j.brief, required_changes: (j.verdict && j.verdict.required_changes) || [], reason: j.gate.reason })), null, 1)}`,
    { label: 'synthesize:revise', phase: 'Synthesize', schema: BRIEFS_SCHEMA, agentType: 'general-purpose' })
  const revised = (rev && rev.briefs) || []
  const byIdOrTitle = j => revised.find(r => r.id === j.brief.id) || revised.find(r => r.title === j.brief.title)
  const resubmit = toRevise.map(j => { const r = byIdOrTitle(j); return r ? { ...r, id: j.brief.id } : null }).filter(Boolean)
  phase('Scope')
  const second = resubmit.length ? await runScope(resubmit, 2) : []
  const secondById = Object.fromEntries(second.map(j => [j.brief.id, j]))
  judged = judged.map(j => {
    if (j.gate.decision !== 'revise') return j
    return secondById[j.brief.id] || { ...j, gate: { decision: 'reject', status: 'REJECTED_SCOPE', reason: 'synthesizer did not return a revision' } }
  })
}

const allowed = judged.filter(j => j.gate.decision === 'allow')
  .sort((a, b) => (scoreSum(b.verdict) - scoreSum(a.verdict)) || ((b.brief.rank_score || 0) - (a.brief.rank_score || 0)))
const toTest = allowed.slice(0, TOP_N)
const untested = allowed.slice(TOP_N)
if (untested.length) log(`Top-N cap: ${untested.length} allowed brief(s) not backtested this run: ${untested.map(j => j.brief.id).join(', ')}`)
log(`Scope: ${allowed.length} allowed, ${judged.length - allowed.length} rejected; backtesting ${toTest.length}`)

// ---- Backtest → Integrity (G2a) → Signal (G2b) ----
const tested = toTest.length ? await pipeline(
  toTest,
  j => agent(`${CONTEXT}

PRE-BUILD MODE. Run this brief's pre-registered test exactly as registered and report one honest number.
Databases: prefix every evmax command / script with ${DB_ENV}
Throwaway scripts go under ${SCRATCH}/${j.brief.id}/ (gitignored). Do not edit any tracked file.

Brief:
${JSON.stringify(j.brief, null, 1)}`, { label: `backtest:${j.brief.id}`, phase: 'Backtest', schema: METRICS_SCHEMA, agentType: 'evmax-backtester' }),
  (metrics, j) => (metrics && metrics.ran === true)
    ? agent(`You are the integrity gate (G2a) for one PRE-REGISTERED offline test in evmax's opportunity-scout
workflow. The backtester claims the result below. Decide whether the number is honest BEFORE anyone compares
it to the threshold. REJECT (accept=false) when any of these hold:
- the metric, train/holdout windows, comparator or declustering differ from the pre-registration;
- n or z counts rows/rungs instead of independent games;
- leakage: UTC-vs-ET game day, state/ratings that already contain the priced game, closes from after entry;
- CLV/ROI not net of fees when the metric says net_fee;
- the backtester edited a tracked file (run \`git status --porcelain\`; only paths under .claude/ are allowed),
  touched the databases outside read-only mode, or hard-coded / cherry-picked results.
Re-run or spot-check the command if needed (DB commands use: ${DB_ENV}). Read-only.

Pre-registration:
${JSON.stringify(j.brief.preregistration, null, 1)}

Backtester result:
${JSON.stringify(metrics, null, 1)}`, { label: `integrity:${j.brief.id}`, phase: 'Integrity', schema: VERDICT_SCHEMA, agentType: 'iteration-reviewer' })
        .then(v => ({ metrics, integrity: v }))
    : { metrics, integrity: null },
) : []

const finalBriefs = []
judged.forEach(j => {
  const scope = j.verdict ? { ...j.verdict, gate_reason: j.gate.reason } : { verdict: 'reject', reason: j.gate.reason }
  if (j.gate.decision !== 'allow') {
    finalBriefs.push(finalize(j.brief, { status: j.gate.status, status_reason: j.gate.reason, scope }))
    return
  }
  const ti = toTest.indexOf(j)
  if (ti < 0) {
    finalBriefs.push(finalize(j.brief, { status: 'ALLOWED_UNTESTED', status_reason: `allowed; beyond top-N cap (${TOP_N})`, scope }))
    return
  }
  const r = tested[ti]
  let sig
  if (!r || !r.metrics) sig = { status: 'NOT_RUN', reason: 'backtest agent failed' }
  else if (r.metrics.ran !== true) sig = { status: 'NOT_RUN', reason: r.metrics.note || 'backtest could not run as registered' }
  else if (!r.integrity) sig = { status: 'INVALID', reason: 'integrity reviewer returned nothing' }
  else if (r.integrity.accept !== true) sig = { status: 'INVALID', reason: `integrity: ${r.integrity.reason}` }
  else sig = signalVerdict(j.brief.preregistration, r.metrics)
  finalBriefs.push(finalize(j.brief, {
    status: sig.status, status_reason: sig.reason, scope,
    evidence: r && r.metrics ? r.metrics : null,
    integrity: r ? r.integrity : null,
  }))
})

const counts = {}
finalBriefs.forEach(b => { counts[b.status] = (counts[b.status] || 0) + 1 })
log(`Outcome: ${Object.entries(counts).map(([k, v]) => `${k} ${v}`).join(', ')}`)

return { run: RUN, proposers, briefs: finalBriefs, dropped, notes }
