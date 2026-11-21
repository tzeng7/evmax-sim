import { defineConfig, devices } from '@playwright/test'

// Frontend tests (`npm test`).
//
// - unit:    pure logic under tests/unit, no browser.
// - layout-*: the real app in Chromium at four window sizes, every /api call
//            answered by tests/e2e/mockApi.ts, so no Python backend is needed.
//            These catch what typecheck/build cannot: popups clipped by the
//            viewport, broken layouts at narrow widths, runtime errors.
//
// The dev server is started on its own port so it never collides with a
// dashboard you already have running on 3000.
const PORT = 4317

const sizes = [
  { name: 'layout-phone', viewport: { width: 390, height: 844 } },
  { name: 'layout-tablet', viewport: { width: 768, height: 1024 } },
  { name: 'layout-laptop', viewport: { width: 1024, height: 768 } },
  { name: 'layout-desktop', viewport: { width: 1440, height: 900 } },
]

export default defineConfig({
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [['list'], ['html', { open: 'never' }]] : 'list',
  use: {
    baseURL: `http://localhost:${PORT}`,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [
    { name: 'unit', testDir: 'tests/unit', testMatch: '**/*.spec.ts' },
    ...sizes.map((s) => ({
      name: s.name,
      testDir: 'tests/e2e',
      testMatch: '**/*.spec.ts',
      use: { ...devices['Desktop Chrome'], viewport: s.viewport },
    })),
  ],
  webServer: {
    command: `npx vite --port ${PORT} --strictPort`,
    url: `http://localhost:${PORT}`,
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
})
