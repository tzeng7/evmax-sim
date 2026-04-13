import type { Summary, ProfitPoint, ScanResult, MetricsResult, Bet, SectorRow } from './types'

const json = (r: Response) => r.json()

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

export async function pickBets(bets: { market_id: string; fill_price: number | null; fill_stake: number | null }[]): Promise<{ placed: number }> {
  return fetch('/api/pick', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ bets }),
  }).then(json)
}

export async function pickByIds(market_ids: string[]): Promise<{ placed: number }> {
  return fetch('/api/pick', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ market_ids }),
  }).then(json)
}

export async function resolve(date: string): Promise<{ resolved: number }> {
  return fetch('/api/resolve', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ date }),
  }).then(json)
}

export async function fetchMetrics(weeks: number): Promise<MetricsResult> {
  return fetch('/api/metrics', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ weeks }),
  }).then(json)
}

export async function updatePlaced(edits: { market_id: string; fill_price: number | null; fill_stake: number }[]): Promise<{ updated: number }> {
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
