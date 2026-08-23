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
  maker_fill?: number  // 1 = placed as a maker fill (P&L uses the maker fee)
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
  // Maker execution (net of the maker fee, not the taker fee):
  maker_ev_pct?: number | null      // EV % if opened as a resting limit order
  maker_only?: boolean              // clears the EV floor ONLY as a maker
  maker_limit_price?: number | null // ceiling: max price (prob 0-1) still +EV as a maker
  maker_bid_price?: number | null   // actionable: the price (prob 0-1) to REST the buy at
  maker_bid_ev_pct?: number | null  // EV % if the resting order fills at maker_bid_price
  maker_bid_kelly_fraction?: number | null // Kelly fraction sized at that fill (suggested stake)
  // Best-execution alternative (GAP 3): when the same bet is also +EV on the
  // OTHER venue, this row is the better book and these carry the alternative so
  // you can still line-shop. null when the bet is quoted on only one venue.
  alt_venue?: string | null
  alt_venue_price?: number | null   // the other venue's YES ask (prob 0-1)
  alt_venue_ev_pct?: number | null  // the other venue's EV %
}

export interface ScanResult {
  gaps: ScanGap[]
  markets_fetched: number
  markets_matched: number
  sectors: string[]
  portfolio_results?: { portfolio_id: string; portfolio_name: string; gaps_logged: number }[]
  // The bankroll Kelly was actually sized against, plus where it came from.
  // bankroll_source is "manual", "live:{venue}", or "manual_fallback" (a venue
  // was requested but its live balance was unavailable). Present when the scan
  // was asked to size against a venue balance (bankroll_venue).
  bankroll?: number
  bankroll_source?: string
}

// venue -> live total-wealth USD (cash + open positions), or null when the
// venue has no credentials / the fetch failed. Returned by GET /api/balance.
export type VenueBalances = Record<string, number | null>

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
  venue: string
  market_id: string
  side: string
  outcome: string
  ask: number
  fee: number
  volume_usd: number
}

export interface ArbRow {
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
  min_volume_usd: number
}

export interface ArbSectorCount {
  sector: string
  kalshi: number
  polymarket_us: number
  baskets: number
}

export interface ArbScanResult {
  arbs: ArbRow[]
  sectors: ArbSectorCount[]
  max_cost: number
}

export interface BoardGate {
  value: number
  required: number
  ok: boolean
}

export interface BoardClv {
  n: number
  mean_clv_pp: number
  frac_positive: number
  clears: boolean
  excluded_stale: number
}

export interface BoardRow {
  sector: string
  market_type: string
  venue: string
  mode: string | null
  mode_split: Record<string, number>
  n_logged: number
  n_resolved: number
  n_clean_resolved: number
  brier_blend: number | null
  brier_sharp: number | null
  brier_delta_per_1000: number | null
  brier_delta_z: number | null
  blend_divergence_pp: number | null
  blend_divergence_resolved_pp: number | null
  sharp_passthrough: boolean
  clv: BoardClv
  gates: Record<string, BoardGate>
  verdict: string
  top_blockers: string[]
}

export interface PromotionBoardResult {
  days: number
  rows: BoardRow[]
}
