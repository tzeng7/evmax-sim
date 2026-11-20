import { expect, test } from '@playwright/test'
import type { Locator, Page } from '@playwright/test'
import { mockApi } from './mockApi'

// Every popup must open fully visible at every window size (see the layout-*
// projects in playwright.config.ts). Regression: the health popup was pinned
// `right: 0` to its chip; when the chip wrapped to the left of the header the
// panel ran ~320px off the left edge.

/**
 * Fails unless `panel` is entirely inside the viewport AND is the topmost
 * element at its four (inset) corners. The second check catches what a bounding
 * box cannot: clipping by an `overflow` ancestor and panels covered by others.
 */
async function expectFullyVisible(page: Page, panel: Locator) {
  await expect(panel).toBeVisible()
  const vp = page.viewportSize()!
  const box = (await panel.boundingBox())!
  const where = `panel box ${JSON.stringify(box)} in viewport ${vp.width}x${vp.height}`
  expect(box.x, where).toBeGreaterThanOrEqual(0)
  expect(box.y, where).toBeGreaterThanOrEqual(0)
  expect(box.x + box.width, where).toBeLessThanOrEqual(vp.width + 0.5)
  expect(box.y + box.height, where).toBeLessThanOrEqual(vp.height + 0.5)

  const hidden = await panel.evaluate((el) => {
    const r = el.getBoundingClientRect()
    // Probe just inside the rounded corner, not on the transparent curve.
    const radius = parseFloat(getComputedStyle(el).borderTopLeftRadius) || 0
    const inset = Math.min(radius + 3, r.width / 4, r.height / 4)
    const corners: [number, number][] = [
      [r.left + inset, r.top + inset], [r.right - inset, r.top + inset],
      [r.left + inset, r.bottom - inset], [r.right - inset, r.bottom - inset],
    ]
    return corners.filter(([x, y]) => {
      const hit = document.elementFromPoint(x, y)
      return !hit || !(el === hit || el.contains(hit))
    }).map(([x, y]) => `(${Math.round(x)},${Math.round(y)})`)
  })
  expect(hidden, `corners not showing the panel (clipped or covered): ${where}`).toEqual([])
}

let unmocked: string[] = []
let pageErrors: string[] = []

test.beforeEach(async ({ page }) => {
  pageErrors = []
  page.on('pageerror', (e) => pageErrors.push(e.message))
  ;({ unmocked } = await mockApi(page))
  await page.goto('/')
})

test.afterEach(() => {
  expect(pageErrors, 'uncaught errors in the page').toEqual([])
  expect(unmocked, 'API calls with no fixture in tests/e2e/mockApi.ts').toEqual([])
})

test('integrity health popup opens fully on screen', async ({ page }) => {
  const chip = page.getByTitle('System integrity')
  await expect(chip).toContainText('critical')
  await chip.click()
  const panel = page.getByRole('dialog', { name: 'Integrity checks' })
  await expectFullyVisible(page, panel)
  // The first and the last check rows are reachable (the panel scrolls, it is not cut).
  await expect(panel.getByText('match_rate')).toBeVisible()
  await panel.getByText('drawdown').scrollIntoViewIfNeeded()
  await expect(panel.getByText('drawdown')).toBeVisible()
})

test('integrity popup stays on screen when the window shrinks while open', async ({ page }) => {
  await page.getByTitle('System integrity').click()
  const panel = page.getByRole('dialog', { name: 'Integrity checks' })
  await expect(panel).toBeVisible()
  const vp = page.viewportSize()!
  await page.setViewportSize({ width: Math.max(360, Math.round(vp.width * 0.6)), height: vp.height })
  await expectFullyVisible(page, panel)
})

test('model-diagnostics popups in scan results open fully on screen', async ({ page }) => {
  await page.getByRole('button', { name: 'Scan', exact: true }).click()
  const badges = page.locator('details > summary.badge')
  await expect(badges.first()).toBeVisible()
  const n = await badges.count()
  expect(n).toBeGreaterThan(0)
  for (let i = 0; i < n; i++) {
    const badge = badges.nth(i)
    await badge.scrollIntoViewIfNeeded()
    await badge.click()
    const panel = page.locator('.diag-panel') // portalled to <body>; one open at a time
    await expectFullyVisible(page, panel)
    await badge.click() // close before the next one
    await expect(panel).toHaveCount(0)
  }
})

test('sector filter menu opens fully on screen', async ({ page }) => {
  const toggle = page.locator('.sector-ms-toggle').first()
  await toggle.click()
  await expectFullyVisible(page, page.locator('.sector-ms-menu').first())
})
