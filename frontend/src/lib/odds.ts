export function probToAmerican(p: number | null | undefined): string {
  // Bets that come back from the dashboard after Pick can carry null
  // kalshi_yes_price for the brief window before the server backfills it,
  // and React would happily render the resulting "-NaN" string. Guard at
  // the conversion site so downstream rows render "-" instead of crashing
  // on subsequent math.
  if (p == null || !Number.isFinite(p)) return '-'
  if (p <= 0 || p >= 1) return '+100'
  if (p < 0.5) return '+' + Math.round((1 / p - 1) * 100)
  return '-' + Math.round((p / (1 - p)) * 100)
}

export function americanToProb(s: string): number | null {
  s = s.toString().trim()
  const n = parseFloat(s.replace('+', ''))
  if (isNaN(n) || n === 0) return null
  if (n > 0) return 100 / (n + 100)
  return Math.abs(n) / (Math.abs(n) + 100)
}

export function fmtDate(): string {
  return new Date().toISOString().slice(0, 10)
}

export function fmtTomorrow(): string {
  const d = new Date()
  d.setDate(d.getDate() + 1)
  return d.toISOString().slice(0, 10)
}

export function yesterday(): string {
  return new Date(Date.now() - 86400000).toISOString().slice(0, 10)
}

// Fallback only — the server sends a pre-formatted `display_label` for every
// row. Used when an older cached payload comes back without it.
const STAT_LABELS: Record<string, string> = {
  points: 'PTS', assists: 'AST', rebounds: 'REB', threes: '3PM',
  pra: 'PRA', pts_reb: 'P+R', pts_ast: 'P+A', reb_ast: 'R+A',
  steals: 'STL', blocks: 'BLK', strikeouts: 'K', hits: 'H', total_bases: 'TB',
}

export function outcomeLabel(row: {
  display_label?: string
  yes_team?: string
  market_type?: string
  line?: number | string | null
  prop_stat_type?: string
  prop_threshold?: number | null
  prop_player_name?: string
}): string {
  if (row.display_label) return row.display_label

  const mt = (row.market_type || '').toLowerCase()
  const team = (row.yes_team || '?').replace(/^./, c => c.toUpperCase())
  const line = row.line

  if (mt === 'player_prop') {
    const raw = (row.prop_player_name || row.yes_team || '?').replace(/_/g, ' ')
    const player = raw.replace(/\b\w/g, c => c.toUpperCase())
    const statRaw = (row.prop_stat_type || '').toLowerCase()
    const stat = STAT_LABELS[statRaw] || statRaw.toUpperCase() || 'PROP'
    const thr = row.prop_threshold ?? (typeof line === 'number' ? line : null)
    return thr != null ? `${player} ${thr}+ ${stat}` : `${player} ${stat}`
  }
  if (mt === 'moneyline') return `${team} ML`
  if (mt === 'spread' && line != null) {
    const n = typeof line === 'number' ? line : parseFloat(String(line))
    return Number.isFinite(n) ? `${team} ${n.toFixed(1).replace(/\.0$/, '')}` : `${team} ${line}`
  }
  if ((mt === 'over_under' || mt === 'total') && line != null) {
    const n = typeof line === 'number' ? line : parseFloat(String(line))
    const lineStr = Number.isFinite(n) ? n.toFixed(1) : String(line)
    // Kalshi totals are YES = OVER (the per-line market threshold lives in
    // floor_strike, not in the outcome code). yes_team is set to "over"
    // upstream in kalshi.py — fall back to "Under" if a future source ever
    // posts an under-side YES market.
    const side = (row.yes_team || '').toLowerCase().startsWith('u') ? 'Under' : 'Over'
    return `${side} ${lineStr}`
  }
  return team
}
