import { useState, useMemo, useCallback, useEffect } from 'react'
import type { ScanGap } from '../lib/types'
import { probToCents, centsToProb, outcomeLabel } from '../lib/odds'
import { SectorFilter } from './SectorFilter'
import { VenueLogo } from './VenueLogo'
import { pickBets } from '../lib/api'

interface Props {
  gaps: ScanGap[]
  meta: { markets_fetched: number; markets_matched: number } | null
  bankroll: number
  kelly: number
  scanKelly: number
  toast: (msg: string, type?: 'info' | 'ok' | 'err') => void
  onPicked: () => void
}

// Approximate re-sizing of a gap's stake when bankroll / Kelly multiplier
// diverge from the scan-time snapshot. The stored kelly_fraction is
// K_full × scan_kelly × liquidity_discount, capped at 5%, so scaling by
// (current_kelly / scan_kelly) recovers the position size for the new knob
// without a rescan. Re-applies the 5% per-bet cap. The 8% per-event exposure
// guard isn't reproduced here — for canonical sizing, rescan.
function recomputedStake(g: ScanGap, bankroll: number, kelly: number, scanKelly: number): number {
  const scaled = scanKelly > 0 ? g.kelly_fraction * (kelly / scanKelly) : g.kelly_fraction
  return bankroll * Math.min(scaled, 0.05)
}

export function ScanResults({ gaps, meta, bankroll, kelly, scanKelly, toast, onPicked }: Props) {
  const [sector, setSector] = useState('')
  const [dateFilter, setDateFilter] = useState('')
  const [selected, setSelected] = useState<Set<string>>(() => new Set(gaps.map(g => g.market_id)))
  const [fills, setFills] = useState<Record<string, { odds: string; stake: string }>>({})
  const [picking, setPicking] = useState(false)

  // Reset selection when gaps change
  useMemo(() => {
    setSelected(new Set(gaps.filter(g => (g.mode || 'live') === 'live').map(g => g.market_id)))
    const f: Record<string, { odds: string; stake: string }> = {}
    for (const g of gaps) {
      f[g.market_id] = { odds: probToCents(g.kalshi_price), stake: recomputedStake(g, bankroll, kelly, scanKelly).toFixed(2) }
    }
    setFills(f)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gaps])

  // When the user changes bankroll or Kelly after a scan, refresh the stake
  // column to the rescaled value. Manually-edited odds are preserved.
  useEffect(() => {
    if (!gaps.length) return
    setFills(prev => {
      const next: Record<string, { odds: string; stake: string }> = {}
      for (const g of gaps) {
        const stake = recomputedStake(g, bankroll, kelly, scanKelly).toFixed(2)
        next[g.market_id] = {
          odds: prev[g.market_id]?.odds ?? probToCents(g.kalshi_price),
          stake,
        }
      }
      return next
    })
  }, [bankroll, kelly, scanKelly, gaps])

  const filtered = useMemo(() => {
    return gaps.filter(g => {
      if (sector && g.sector !== sector) return false
      if (dateFilter && g.event_date !== dateFilter) return false
      return true
    })
  }, [gaps, sector, dateFilter])

  const toggle = (mid: string) => {
    setSelected(prev => {
      const s = new Set(prev)
      s.has(mid) ? s.delete(mid) : s.add(mid)
      return s
    })
  }

  const selectAll = () => setSelected(new Set(filtered.filter(g => (g.mode || 'live') === 'live').map(g => g.market_id)))
  const deselectAll = () => setSelected(new Set())

  const updateFill = (mid: string, field: 'odds' | 'stake', value: string) => {
    setFills(prev => ({ ...prev, [mid]: { ...prev[mid], [field]: value } }))
  }

  const handlePick = useCallback(async () => {
    const bets = [...selected].map(mid => ({
      market_id: mid,
      fill_price: centsToProb(fills[mid]?.odds || ''),
      fill_stake: parseFloat(fills[mid]?.stake || '0'),
    }))
    if (!bets.length) return
    setPicking(true)
    try {
      const res = await pickBets(bets)
      if (res.placed > 0) {
        const skippedNote = res.skipped?.length ? ` · skipped ${res.skipped.length}` : ''
        toast(`Placed ${res.placed} bet(s)${skippedNote}`, 'ok')
      } else if (res.skipped?.length) {
        const first = res.skipped[0]
        const more = res.skipped.length > 1 ? ` (+${res.skipped.length - 1} more)` : ''
        toast(`Placed 0 — ${first.reason}${more}`, 'err')
      } else {
        toast('Placed 0 bet(s)', 'err')
      }
      onPicked()
    } catch (e) {
      toast('Pick failed: ' + (e as Error).message, 'err')
    } finally {
      setPicking(false)
    }
  }, [selected, fills, toast, onPicked])

  if (!gaps.length) return null

  const checkedCount = [...selected].filter(mid => filtered.some(g => g.market_id === mid)).length

  return (
    <div className="panel">
      <div className="panel-header">
        <h2>Scan Results — {gaps.length} plays ({meta?.markets_fetched} markets, {meta?.markets_matched} matched)</h2>
        <div className="actions" style={{ gap: 4 }}>
          <input type="date" value={dateFilter} onChange={e => setDateFilter(e.target.value)}
            style={{ fontSize: 11 }} />
          <SectorFilter value={sector} onChange={setSector} />
          <button className="btn btn-sm" onClick={() => { setSector(''); setDateFilter('') }}>Clear</button>
          <button className="btn btn-sm" onClick={selectAll}>Select All</button>
          <button className="btn btn-sm" onClick={deselectAll}>Deselect All</button>
        </div>
      </div>
      <table>
        <thead>
          <tr>
            <th style={{ width: 30 }}></th>
            <th>Date</th><th>Sector</th><th style={{ width: 34 }}>Venue</th><th>Event</th><th>Outcome</th>
            <th className="num">Ask</th><th className="num">Fair Value</th><th className="num">Model</th><th className="num">EV</th>
            <th className="num">Fill ¢</th><th className="num">Stake ($)</th><th>Models</th>
          </tr>
        </thead>
        <tbody>
          {filtered.map(g => {
            const askOdds = probToCents(g.kalshi_price)
            const outcome = outcomeLabel(g)
            const mode = g.mode || 'live'
            const isLive = mode === 'live'
            return (
              <tr key={g.market_id} style={isLive ? undefined : { opacity: 0.55 }}>
                <td>
                  <input type="checkbox" className="check" checked={selected.has(g.market_id)}
                    disabled={!isLive}
                    title={isLive ? '' : `${mode} mode — not pickable`}
                    onChange={() => toggle(g.market_id)} />
                </td>
                <td className="muted">{g.event_date}</td>
                <td>
                  <span className="badge">{g.sector}</span>
                  {!isLive && (
                    <span
                      className="badge"
                      style={{ marginLeft: 4, background: 'rgba(224,179,65,0.13)', color: '#e0b341', borderColor: 'rgba(224,179,65,0.32)' }}
                      title={`mode=${mode} — logged for tracking, not pickable`}
                    >{mode}</span>
                  )}
                </td>
                <td><VenueLogo venue={g.venue} /></td>
                <td>{g.event_title}</td>
                <td>{outcome}</td>
                <td className="num">{askOdds}</td>
                <td className="num">{probToCents(g.true_prob)}</td>
                <td className="num">{(g.true_prob * 100).toFixed(1)}%</td>
                <td className="num green">{g.ev_pct.toFixed(1)}%</td>
                <td className="num">
                  <input type="text" value={fills[g.market_id]?.odds || askOdds}
                    onChange={e => updateFill(g.market_id, 'odds', e.target.value)}
                    style={{ width: 64 }} />
                </td>
                <td className="num">
                  <input type="number" value={fills[g.market_id]?.stake || ''}
                    onChange={e => updateFill(g.market_id, 'stake', e.target.value)}
                    style={{ width: 70 }} min="0.01" step="0.01" />
                </td>
                <td className="muted" style={{ fontSize: 10 }}>{g.model_sources}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <div style={{ marginTop: 8 }}>
        <button className="btn success" disabled={checkedCount === 0 || picking} onClick={handlePick}>
          {picking ? 'Placing...' : `Pick Selected (${checkedCount})`}
        </button>
      </div>
    </div>
  )
}
