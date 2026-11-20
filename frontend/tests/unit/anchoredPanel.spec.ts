import { expect, test } from '@playwright/test'
import { placeAnchored } from '../../src/lib/anchoredPanel'

const vp = { width: 1024, height: 768 }
const panel = { width: 440, height: 300 }
const chip = (left: number, top = 60) => ({ left, right: left + 90, top, bottom: top + 32 })

test('end-aligned panel near the left edge is pushed back inside (the clipped health popup)', () => {
  // The chip wrapped to x=30 on a 1024px window; `right: 0` put the panel at -321.
  const p = placeAnchored(chip(30), panel, vp, { align: 'end' })
  expect(p.left).toBe(16)
  expect(p.left + p.width).toBeLessThanOrEqual(vp.width - 16)
})

test('end-aligned panel with room keeps its right edge on the anchor', () => {
  const a = chip(900)
  const p = placeAnchored(a, panel, vp, { align: 'end' })
  expect(p.left + p.width).toBe(a.right)
  expect(p.top).toBe(a.bottom + 8)
})

test('start-aligned panel near the right edge is pulled back inside', () => {
  const p = placeAnchored(chip(950), panel, vp, { align: 'start' })
  expect(p.left + p.width).toBe(vp.width - 16)
})

test('panel wider than the viewport is narrowed to fit', () => {
  const phone = { width: 390, height: 844 }
  const p = placeAnchored(chip(10), panel, phone, { align: 'end' })
  expect(p.width).toBe(390 - 32)
  expect(p.left).toBe(16)
})

test('opens above when it does not fit below and above has more room', () => {
  const a = chip(400, 700)
  const p = placeAnchored(a, panel, vp)
  expect(p.top + Math.min(panel.height, p.maxHeight)).toBe(a.top - 8)
  expect(p.top).toBeGreaterThanOrEqual(16)
})

test('stays below and scrolls when below still has the most room', () => {
  const a = chip(400, 300)
  const tall = { width: 440, height: 2000 }
  const p = placeAnchored(a, tall, vp)
  expect(p.top).toBe(a.bottom + 8)
  expect(p.top + p.maxHeight).toBe(vp.height - 16)
})
