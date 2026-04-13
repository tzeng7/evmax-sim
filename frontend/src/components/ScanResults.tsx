import { useState, useMemo, useCallback } from 'react'
import type { ScanGap } from '../lib/types'
import { probToAmerican, americanToProb } from '../lib/odds'
import { SectorFilter } from './SectorFilter'
import { pickBets } from '../lib/api'

interface Props {
  gaps: ScanGap[]
  meta: { markets_fetched: number; markets_matched: number } | null
  toast: (msg: string, type?: 'info' | 'ok' | 'err') => void
  onPicked: () => void
}

export function ScanResults({ gaps, meta, toast, onPicked }: Props) {
  const [sector, setSector] = useState('')
  const [dateFilter, setDateFilter] = useState('')
  const [selected, setSelected] = useState<Set<string>>(() => new Set(gaps.map(g => g.market_id)))
  const [fills, setFills] = useState<Record<string, { odds: string; stake: string }>>({})
  const [picking, setPicking] = useState(false)

  // Reset selection when gaps change
  useMemo(() => {
    setSelected(new Set(gaps.map(g => g.market_id)))
    const f: Record<string, { odds: string; stake: string }> = {}
    for (const g of gaps) {
      f[g.market_id] = { odds: probToAmerican(g.kalshi_price), stake: g.stake.toFixed(2) }
    }
    setFills(f)
  }, [gaps])

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

  const selectAll = () => setSelected(new Set(filtered.map(g => g.market_id)))
  const deselectAll = () => setSelected(new Set())

  const updateFill = (mid: string, field: 'odds' | 'stake', value: string) => {
    setFills(prev => ({ ...prev, [mid]: { ...prev[mid], [field]: value } }))
  }

  const handlePick = useCallback(async () => {
    const bets = [...selected].map(mid => ({
      market_id: mid,
      fill_price: americanToProb(fills[mid]?.odds || ''),
      fill_stake: parseFloat(fills[mid]?.stake || '0'),
    }))
    if (!bets.length) return
    setPicking(true)
    try {
      const res = await pickBets(bets)
      toast(`Placed ${res.placed} bet(s)`, 'ok')
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
            style={{ fontSize: 11, padding: '4px 6px', background: '#1a1a2e', color: '#e0e0e0', border: '1px solid #333', borderRadius: 4 }} />
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
            <th>Date</th><th>Sector</th><th>Event</th><th>Outcome</th>
            <th className="num">Ask</th><th className="num">Fair Value</th><th className="num">Model</th><th className="num">EV</th>
            <th className="num">Fill Odds</th><th className="num">Stake ($)</th><th>Models</th>
          </tr>
        </thead>
        <tbody>
          {filtered.map(g => {
            const askOdds = probToAmerican(g.kalshi_price)
            const outcome = g.yes_team + ' ' + g.market_type + (g.line ? ' ' + g.line : '')
            return (
              <tr key={g.market_id}>
                <td>
                  <input type="checkbox" className="check" checked={selected.has(g.market_id)}
                    onChange={() => toggle(g.market_id)} />
                </td>
                <td className="muted">{g.event_date}</td>
                <td><span className="badge">{g.sector}</span></td>
                <td>{g.event_title}</td>
                <td>{outcome}</td>
                <td className="num">{askOdds} <span className="muted">({Math.round(g.kalshi_price * 100)}c)</span></td>
                <td className="num">{probToAmerican(g.true_prob)} <span className="muted">({Math.round(g.true_prob * 100)}c)</span></td>
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
