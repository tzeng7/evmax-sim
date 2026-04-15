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
  stake: number
  model_sources: string
  market_id: string
  event_date: string
  volume: number
}

export interface ScanResult {
  gaps: ScanGap[]
  markets_fetched: number
  markets_matched: number
  sectors: string[]
}

export interface MetricsResult {
  brier_model: number
  brier_sharp: number
  n: number
  win_rate: number
  calibration: { bucket: string; n: number; predicted: number; actual: number }[]
  error?: string
}
