// Run: npx --yes tsx --test tests/reconcileSelection.test.ts
// (no test runner is installed in frontend; this uses node:test via tsx.)
import test from 'node:test'
import assert from 'node:assert/strict'
import { reconcileRowState } from '../src/lib/reconcileSelection.ts'

type G = { market_id: string; mode?: string }
const legsOf = (g: G): G[] => [g]
const def = (g: G) => ({ odds: 'def-' + g.market_id, stake: '1' })
const empty = { selected: new Set<string>(), venueSel: {}, fills: {} }

test('first load selects live rows only', () => {
  const { state } = reconcileRowState(empty, new Set(), [{ market_id: 'a' }, { market_id: 'b', mode: 'shadow' }], legsOf, def)
  assert.deepEqual([...state.selected], ['a'])
  assert.deepEqual(Object.keys(state.fills).sort(), ['a', 'b'])
})

test('keeps user state for present ids, seeds new, drops vanished', () => {
  const prev = {
    selected: new Set(['a']), // user deselected b, c vanishes
    venueSel: { a: 'poly', c: 'kalshi' },
    fills: { a: { odds: '55', stake: '9' }, b: { odds: '40', stake: '2' }, c: { odds: '1', stake: '1' } },
  }
  const { state, ids } = reconcileRowState(prev, new Set(['a', 'b', 'c']),
    [{ market_id: 'a' }, { market_id: 'b' }, { market_id: 'd' }, { market_id: 'e', mode: 'shadow' }], legsOf, def)
  assert.deepEqual([...state.selected].sort(), ['a', 'd'])
  assert.deepEqual(state.venueSel, { a: 'poly' })
  assert.equal(state.fills.a.odds, '55')
  assert.equal(state.fills.d.odds, 'def-d')
  assert.ok(!('c' in state.fills))
  assert.deepEqual([...ids].sort(), ['a', 'b', 'd', 'e'])
})
