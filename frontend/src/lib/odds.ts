export function probToAmerican(p: number): string {
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
    return Number.isFinite(n) ? `O/U ${n.toFixed(1)}` : `O/U ${line}`
  }
  return team
}
