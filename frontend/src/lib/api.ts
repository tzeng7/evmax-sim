import type { Summary, ProfitPoint, ScanResult, MetricsResult, Bet, SectorRow, Category, Portfolio, PortfolioDetail, PortfolioScanResult, ArbScanResult, PromotionBoardResult } from './types'

const json = (r: Response) => r.json()

export async function fetchCategories(): Promise<Category[]> {
  return fetch('/api/categories').then(json).then((d: { categories: Category[] }) => d.categories)
}

export async function fetchDashboard(): Promise<{
  summary_all: Summary
  summary_placed: Summary
  series_all: ProfitPoint[]
  series_placed: ProfitPoint[]
  sectors: SectorRow[]
  placed_bets: Bet[]
  unplaced_bets: Bet[]
  recent: Bet[]
}> {
  return fetch('/api/dashboard').then(json)
}

export async function fetchSummary(days: number, view: string): Promise<Summary> {
  return fetch(`/api/summary?days=${days}&view=${view}`).then(json)
}

export async function fetchProfit(): Promise<ProfitPoint[]> {
  return fetch('/api/profit').then(json)
}

export async function runScan(params: {
  sectors: string
  bankroll: number
  kelly: number
  date_from: string
  date_to: string
}): Promise<ScanResult> {
  return fetch('/api/scan', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  }).then(json)
}

export interface PickSkip { market_id: string | null; reason: string }
export interface PickResult { placed: number; skipped?: PickSkip[] }

export async function pickBets(bets: { market_id: string; fill_price: number | null; fill_stake: number | null }[]): Promise<PickResult> {
  return fetch('/api/pick', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ bets }),
  }).then(json)
}

export interface MakerFillRow {
  market_id: string
  fill_price: number
  fill_stake: number
  contracts: number
  maker_fee: number
  venue: string
  prior_mode: string
  event_title: string
}
export interface MakerFillResult { filled: number; fills: MakerFillRow[]; skipped?: PickSkip[] }

// Record filled maker limit orders. A maker-only play is not crossable at the
// ask, so it logs as shadow and /api/pick refuses it — this is the path that
// turns one into a real position once your resting order actually fills.
// fill_price accepts cents or a fraction; the server normalizes.
export async function recordMakerFills(
  fills: { market_id: string; fill_price: number | null; fill_stake: number | null }[],
): Promise<MakerFillResult> {
  return fetch('/api/fill', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ fills }),
  }).then(json)
}

export async function pickByIds(market_ids: string[]): Promise<PickResult> {
  return fetch('/api/pick', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ market_ids }),
  }).then(json)
}

export async function resolve(
  date: string,
  syncPortfolios = true,
): Promise<{ resolved: number; portfolio_resolved?: number }> {
  return fetch('/api/resolve', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ date, sync_portfolios: syncPortfolios }),
  }).then(json)
}

export async function fetchMetrics(weeks: number): Promise<MetricsResult> {
  return fetch('/api/metrics', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ weeks }),
  }).then(json)
}

export async function updatePlaced(edits: { market_id: string; fill_price: number | null; fill_stake: number | null }[]): Promise<{ updated: number }> {
  return fetch('/api/update-placed', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ edits }),
  }).then(json)
}

export async function unplaceBets(market_ids: string[]): Promise<{ removed: number }> {
  return fetch('/api/unplace', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ market_ids }),
  }).then(json)
}

export async function fetchPortfolios(): Promise<Portfolio[]> {
  return fetch('/api/portfolios').then(json)
}

export async function fetchPortfolioDetail(id: string): Promise<PortfolioDetail> {
  return fetch(`/api/portfolios/${id}`).then(json)
}

export async function createDefaultPortfolios(): Promise<{ created: number; portfolios: Portfolio[] }> {
  return fetch('/api/portfolios/create-defaults', { method: 'POST' }).then(json)
}

export async function deletePortfolio(id: string): Promise<{ deleted: boolean }> {
  return fetch(`/api/portfolios/${id}`, { method: 'DELETE' }).then(json)
}

export async function scanPortfolios(portfolioIds?: string[]): Promise<PortfolioScanResult> {
  return fetch('/api/portfolios/scan', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ portfolio_ids: portfolioIds || [] }),
  }).then(json)
}

export async function syncPortfolioOutcomes(): Promise<{ resolved: number }> {
  return fetch('/api/portfolios/sync-outcomes', { method: 'POST' }).then(json)
}

export async function runArbScan(params: {
  sectors: string
  max_cost: number
  include_in_play: boolean
}): Promise<ArbScanResult> {
  return fetch('/api/arb/scan', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  }).then(json)
}

export async function fetchPromotionBoard(days: number, sector?: string): Promise<PromotionBoardResult> {
  const params = new URLSearchParams({ days: String(days) })
  if (sector) params.set('sector', sector)
  return fetch(`/api/promotion-board?${params}`).then(json)
}
