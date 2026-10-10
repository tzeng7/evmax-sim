// Stub harness for the Workflow graphs in .claude/workflows/ (driven by
// tests/test_opportunity_workflows.py). It runs a graph with fake agent() /
// parallel() / pipeline() / phase() / log() so the deterministic orchestration —
// gates, caps, loops, status transitions — can be tested without spawning agents.
//
// Usage: node tests/workflow_harness.mjs <scenario.json>
// Scenario:
//   { "script": ".claude/workflows/opportunity-scout.js",
//     "mode": "run" | "call",
//     "args": {...},                                  // run mode: the workflow's args
//     "responses": { "<label or label prefix>": [r1, r2, ...] },  // run mode
//     "calls": [ { "fn": "signalVerdict", "args": [...] } ] }     // call mode
// Responses are matched by exact label, else by the longest key that the label
// starts with; each list is consumed in order and its last entry repeats. An
// unmatched label returns null (a dead agent); a response {"__throw__": "msg"} makes
// agent() reject (an agent that errors out).
// Output (stdout, JSON): run → {result, calls:[{label, agentType, phase}], prompts:{label: text},
// logs, phases, unmatched}; call → {results:[...]}.

import { readFileSync } from 'node:fs'

const scenario = JSON.parse(readFileSync(process.argv[2], 'utf8'))
const source = readFileSync(scenario.script, 'utf8').replace(/^export const meta\s*=/m, 'const meta =')
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor
const clone = v => (v === undefined ? undefined : JSON.parse(JSON.stringify(v)))

if (scenario.mode === 'call') {
  const marker = '// ==== WORKFLOW BODY ===='
  const idx = source.indexOf(marker)
  if (idx < 0) throw new Error(`no "${marker}" in ${scenario.script}`)
  const names = [...new Set(scenario.calls.map(c => c.fn))]
  const prelude = source.slice(0, idx)
  const fns = new Function(`${prelude}\nreturn { ${names.join(', ')} }`)()
  const results = scenario.calls.map(c => clone(fns[c.fn](...(c.args || []))))
  process.stdout.write(JSON.stringify({ results }))
} else {
  const queues = Object.fromEntries(Object.entries(scenario.responses || {}).map(([k, v]) => [k, clone(v)]))
  const calls = []
  const logs = []
  const phases = []
  const unmatched = []
  const prompts = {}
  const pick = label => {
    if (label in queues) return label
    let best = null
    for (const k of Object.keys(queues)) if (label.startsWith(k) && (!best || k.length > best.length)) best = k
    return best
  }
  const agent = async (prompt, opts = {}) => {
    const label = opts.label || '(unlabeled)'
    calls.push({ label, agentType: opts.agentType || null, phase: opts.phase || null, prompt_chars: String(prompt).length })
    prompts[label] = String(prompt)
    const key = pick(label)
    if (!key) { unmatched.push(label); return null }
    const q = queues[key]
    const r = q.length > 1 ? q.shift() : q[0]
    if (r && typeof r === 'object' && '__throw__' in r) throw new Error(String(r.__throw__))
    return clone(r)
  }
  const parallel = thunks => Promise.all(thunks.map(t => Promise.resolve().then(t).catch(() => null)))
  const pipeline = (items, ...stages) => Promise.all(items.map(async (item, i) => {
    let prev = item
    for (const stage of stages) {
      try { prev = await stage(prev, item, i) } catch (e) { return null }
    }
    return prev
  }))
  const phase = t => { phases.push(t) }
  const log = m => { logs.push(m) }
  const run = new AsyncFunction('args', 'agent', 'parallel', 'pipeline', 'phase', 'log', 'budget', source)
  const budget = { total: null, spent: () => 0, remaining: () => Infinity }
  const result = await run(clone(scenario.args), agent, parallel, pipeline, phase, log, budget)
  process.stdout.write(JSON.stringify({ result, calls, prompts, logs, phases, unmatched }))
}
