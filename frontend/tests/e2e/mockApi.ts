import type { Page, Route } from '@playwright/test'
import type { ScanGap, ScanResult } from '../../src/lib/types'

// Synthetic API fixtures for the layout tests. Shapes follow src/lib/types.ts
// (TypeScript checks the scan fixtures against them). Content is made up and
// deliberately worst-case for layout: long titles, long diagnostic notes, a
// critical integrity result with many long messages.

const summary = { total_bets: 0, wins: 0, losses: 0, win_rate: 0, total_pnl: 0, roi_pct: 0, avg_ev: 0, total_staked: 0 }

const longNote = 'Elo_a=1716 Elo_b=1568 n=100 rest=2d/4d — a deliberately long gate note so the panel has to wrap inside the viewport'

export const integrityCritical = {
  ok: false,
  status: 'critical',
  generated_at: new Date().toISOString(),
  failed: [],
  checks: [
    { check: 'cadence', severity: 'ok', issues: [] },
    { check: 'match_rate', severity: 'critical', issues: [{ severity: 'critical', detail: 'nhl: fetched 1149 venue markets today but matched 0 Pinnacle events (14d median matched 470) — codes/pagination/title-format break? (`scripts/check_kalshi_series.py --probe`, `scripts/check_pinnacle_leagues.py`)' }] },
    { check: 'state:tennis_surface', severity: 'warning', issues: [{ severity: 'warning', detail: 'tennis_surface state last updated 2026-10-12 (39d ago > 10d) — weekly reseed may be silently failing' }] },
    { check: 'passthrough', severity: 'warning', issues: Array.from({ length: 5 }, (_, i) => ({ severity: 'warning', detail: `sector${i} moneyline [kalshi] — LIVE with blend divergence 0.2${i}pp (<0.5): staking bankroll with no independent model signal; any EV is venue divergence. Fix seeding/models or demote.` })) },
    { check: 'drawdown', severity: 'warning', issues: [{ severity: 'warning', detail: 'ncaab: live flat ROI -46.9% over n=21 resolved rows (3 wins) in 30d — below the -30% floor' }] },
  ],
}

function gap(i: number, over: Partial<ScanGap> = {}): ScanGap {
  return {
    event_title: `Southern Illinois Salukis vs Liberty Flames (game ${i})`,
    yes_team: 'liberty flames',
    market_type: 'moneyline',
    display_label: 'Liberty ML',
    line: null,
    sector: 'ncaab',
    kalshi_price: 0.45,
    true_prob: 0.481,
    ev_pct: 6.0,
    kelly_pct: 1.2,
    kelly_fraction: 0.012,
    stake: 19.04,
    model_sources: 'elo+form+sharp',
    model_diagnostics: {
      fired: { elo: { conf: 0.8, n: 100 }, form: { conf: 0.8, n: 20 } },
      gated: { ncaab_efficiency: { conf: 0.3, note: longNote } },
      missing: ['ncaab_possession_sim'],
    },
    sharp_only: false,
    market_id: `KXNCAABGAME-26NOV20SIULIB-LIB-${i}`,
    event_date: '2026-11-20',
    tip_time: new Date(Date.now() + (i + 2) * 3600_000).toISOString(),
    volume: 5000,
    mode: 'live',
    venue: 'kalshi',
    ...over,
  }
}

export const scanResult: ScanResult = {
  gaps: [
    gap(1),
    gap(2, { sharp_only: true, model_sources: 'sharp', model_diagnostics: null, display_label: 'Under 156.5', market_type: 'total', line: 156.5 }),
    gap(3, { venue: 'polymarket_us', market_id: 'polymarket_us:aec-cbb-siu-lib-2026-11-20' }),
  ],
  markets_fetched: 120,
  markets_matched: 40,
  sectors: ['ncaab'],
}

const fixtures: Record<string, unknown> = {
  '/api/integrity': integrityCritical,
  '/api/categories': { categories: [
    { key: 'nba', display_name: 'NBA', mode: 'live' },
    { key: 'ncaab', display_name: 'NCAAB', mode: 'live' },
    { key: 'nfl', display_name: 'NFL', mode: 'live' },
    { key: 'nhl', display_name: 'NHL', mode: 'shadow' },
  ] },
  '/api/dashboard': {
    summary_all: summary, summary_placed: summary,
    series_all: [], series_placed: [], sectors: [],
    placed_bets: [], unplaced_bets: [], recent: [],
  },
  '/api/resting-orders': { orders: [], readout: {
    sector: 'nfl', enabled: true, kelly_mult: 0.25, checkpoint_games: 15, gate_games: 30,
    candidates_resolved: 0, candidates_filled: 0, candidates_open: 0, fill_rate: null,
    games: 0, rows: 0, mean_net_pp: null, z: null, frac_positive: null, verdict: 'COLLECTING', action: '',
  } },
  '/api/balance': { balances: { kalshi: null, polymarket_us: null } },
  '/api/scan': scanResult,
}

/**
 * Answer every /api request from the fixtures above (overridable per test).
 * A request with no fixture is recorded in `unmocked` and answered 501, and
 * the test fails on it: a new endpoint must get a fixture, never hit a backend.
 */
export async function mockApi(page: Page, overrides: Record<string, unknown> = {}) {
  const table = { ...fixtures, ...overrides }
  const unmocked: string[] = []
  await page.route('**/api/**', async (route: Route) => {
    const path = new URL(route.request().url()).pathname
    if (path in table) {
      await route.fulfill({ json: table[path] })
    } else {
      unmocked.push(`${route.request().method()} ${path}`)
      await route.fulfill({ status: 501, json: { error: 'no fixture' } })
    }
  })
  return { unmocked }
}
