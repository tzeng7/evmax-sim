export interface Summary {
  total_bets: number
  wins: number
  losses: number
  win_rate: number
  total_pnl: number
  roi_pct: number
  avg_ev: number
  total_staked: number
}

export interface ProfitPoint {
  date: string
  daily: number
  cumulative: number
}

export interface SectorRow {
  sector: string
  bets: number
  wins: number
  win_rate: number
  pnl: number
  roi_pct: number
}

export interface Category {
  key: string
  display_name: string
  mode: string
}

export interface Bet {
  id: number
  scan_date: string
  event_date: string
  sector: string
  event_title: string
  yes_team: string
  market_type: string
  display_label?: string
  kalshi_yes_price: number
  blended_true_prob: number
  sharp_true_prob?: number
  ev_pct: number
  kelly_fraction: number
  bankroll_used: number
  model_sources: string
  placed: number
  placed_stake?: number
  placed_price?: number
  market_id: string
  line?: number
  volume_usd?: number
  outcome?: number
  status?: 'upcoming' | 'in_progress'
  venue?: string
}

export interface ScanGap {
  event_title: string
  yes_team: string
  market_type: string
  display_label?: string
  line: number | string | null
  sector: string
  kalshi_price: number
  true_prob: number
  ev_pct: number
  kelly_pct: number
  kelly_fraction: number
  stake: number
  model_sources: string
  market_id: string
  event_date: string
  volume: number
  mode?: string
  venue?: string
}

export interface ScanResult {
  gaps: ScanGap[]
  markets_fetched: number
  markets_matched: number
  sectors: string[]
  portfolio_results?: { portfolio_id: string; portfolio_name: string; gaps_logged: number }[]
}

export interface Portfolio {
  id: string
  name: string
  sectors: string[]
  initial_bankroll: number
  current_bankroll: number
  kelly_fraction: number
  scenario: string
  active: boolean
  total_bets: number
  open_bets: number
  settled_bets: number
  wins: number
  losses: number
  win_rate: number
  total_pnl: number
  total_staked: number
  roi_pct: number
  avg_ev: number
}

export interface PortfolioDetail extends Portfolio {
  bets: PortfolioBet[]
}

export interface PortfolioBet {
  id: number
  portfolio_id: string
  market_id: string
  scan_date: string
  event_date: string
  sector: string
  yes_team: string
  market_type: string
  event_title: string
  display_label: string
  kalshi_yes_price: number
  blended_true_prob: number
  ev_pct: number
  kelly_fraction: number
  stake: number
  bankroll_at_scan: number
  line: number | null
  volume_usd: number
  model_sources: string
  outcome: number | null
  pnl: number | null
}

export interface PortfolioScanResult {
  portfolios_scanned: number
  markets_fetched: number
  markets_matched: number
  results: { portfolio_id: string; portfolio_name: string; gaps_logged: number }[]
}

export interface SectorMetricRow {
  sector: string
  bets: number
  win_rate: number
  pnl: number
  roi_pct: number
  avg_ev_pct: number | null
  avg_clv_pct: number | null
  clv_n: number
  brier_model: number
  brier_sharp: number
}

export interface MetricsResult {
  weeks: number
  brier_model: number
  brier_sharp: number
  n: number
  win_rate: number
  calibration: { bucket: string; n: number; predicted: number; actual: number }[]
  sectors?: SectorMetricRow[]
  error?: string
}

export interface ArbLeg {
  venue: string // 'kalshi' | 'polymarket_us'
  market_id: string
  side: string // 'yes' | 'no'
  outcome: string
  ask: number
  fee: number
  volume_usd: number
}

export interface ArbBasket {
  sector: string
  event_title: string
  event_date: string | null
  market_desc: string
  legs: ArbLeg[]
  gross_cost: number
  net_cost: number
  net_edge: number
  cross_venue: boolean
  in_play: boolean
}

export interface ArbScanResult {
  baskets: ArbBasket[]
  sector_stats: { sector: string; kalshi_markets: number; polyus_markets: number; baskets: number }[]
  scanned_at: string
}
