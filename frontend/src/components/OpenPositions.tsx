import { useState, useMemo, useCallback } from 'react'
import type { Bet, ScanGap } from '../lib/types'
import { probToCents, outcomeLabel } from '../lib/odds'
import { SectorFilter } from './SectorFilter'
import { VenueLogo } from './VenueLogo'
import { pickBets } from '../lib/api'
import { resizeStake } from '../lib/resizeStake'

interface Props {
  bets: Bet[]
  scanGaps: ScanGap[]
  bankroll: number
  kelly: number
  scanKelly: number
  toast: (msg: string, type?: 'info' | 'ok' | 'err') => void
  onPicked: () => void
}

const PAGE = 40

type SortKey = 'event_date' | 'kalshi_yes_price' | 'ev_pct' | 'stake'

export function OpenPositions({ bets, scanGaps, bankroll, kelly, scanKelly, toast, onPicked }: Props) {
  const [sector, setSector] = useState('')
  const [selected, setSelected] = useState<Set<string>>(new Set())

  const scanMids = useMemo(() => new Set(scanGaps.map(g => g.market_id)), [scanGaps])

  const [sortKey, setSortKey] = useState<SortKey | null>(null)
  const [sortDir, setSortDir] = useState<1 | -1>(1)
  const [limit, setLimit] = useState(PAGE)

  const stakeOf = useCallback(
    (b: Bet) => resizeStake(b.kelly_fraction, bankroll, kelly, scanKelly),
    [bankroll, kelly, scanKelly],
  )

  const filtered = useMemo(() => {
    const rows = sector ? bets.filter(b => b.sector === sector) : bets
    if (!sortKey) return rows
    const val = (b: Bet): number | string =>
      sortKey === 'stake' ? stakeOf(b)
        : sortKey === 'event_date' ? (b.event_date || '')
          : (b[sortKey] as number) || 0
    return [...rows].sort((x, y) => {
      const a = val(x), c = val(y)
      return (a < c ? -1 : a > c ? 1 : 0) * sortDir
    })
  }, [bets, sector, sortKey, sortDir, stakeOf])

  const visible = filtered.slice(0, limit)
  const allVisibleSelected = visible.length > 0 && visible.every(b => selected.has(b.market_id))

  const toggleAllVisible = () => {
    setSelected(prev => {
      const s = new Set(prev)
      if (allVisibleSelected) visible.forEach(b => s.delete(b.market_id))
      else visible.forEach(b => s.add(b.market_id))
      return s
    })
  }

  const sortBy = (k: SortKey) => {
    if (sortKey === k) setSortDir(d => (d === 1 ? -1 : 1))
    else { setSortKey(k); setSortDir(k === 'event_date' ? 1 : -1) }
  }
  const sortTh = (k: SortKey, label: string, cls?: string) => (
    <th className={cls} style={{ cursor: 'pointer' }} onClick={() => sortBy(k)}
      aria-sort={sortKey === k ? (sortDir === 1 ? 'ascending' : 'descending') : 'none'}>
      {label}{sortKey === k ? (sortDir === 1 ? ' ▲' : ' ▼') : ''}
    </th>
  )

  const toggle = (mid: string) => {
    setSelected(prev => {
      const s = new Set(prev)
      if (s.has(mid)) {
        s.delete(mid)
      } else {
        s.add(mid)
      }
      return s
    })
  }

  const handlePick = useCallback(async () => {
    const ids = [...selected]
    if (!ids.length) { toast('Select positions to pick first', 'info'); return }
    // Place at the CURRENT dashboard bankroll/kelly — i.e. the stake the preview
    // shows — not the scan-time bankroll_used × kelly_fraction the backend falls
    // back to when sent bare market_ids. Without this, a bet scanned at $250/0.5
    // and re-sized to $1100/0.75 in the UI would silently revert to the $250
    // stake on placement.
    const byId = new Map(bets.map(b => [b.market_id, b]))
    const payload = ids.map(mid => {
      const b = byId.get(mid)
      return {
        market_id: mid,
        fill_price: b ? b.kalshi_yes_price : null,
        fill_stake: b ? Number(stakeOf(b).toFixed(2)) : null,
      }
    })
    try {
      const res = await pickBets(payload)
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
      setSelected(new Set())
      onPicked()
    } catch (e) {
      toast('Pick failed: ' + (e as Error).message, 'err')
    }
  }, [selected, bets, stakeOf, toast, onPicked])

  return (
    <div className="panel">
      <div className="panel-header">
        <h2>Open Positions ({filtered.length})</h2>
        <div className="actions" style={{ gap: 4 }}>
          <SectorFilter value={sector} onChange={setSector} />
          {bets.length > 0 && <button className="btn btn-sm success" onClick={handlePick}>Pick All</button>}
        </div>
      </div>
      <div className="scroll-area">
        <table>
          <thead>
            <tr>
              <th style={{ width: 28 }}>
                <input type="checkbox" aria-label="Select all visible" checked={allVisibleSelected} onChange={toggleAllVisible} />
              </th><th></th>
              {sortTh('event_date', 'Date')}<th>Sector</th><th style={{ width: 34 }}>Venue</th><th>Event</th><th>Outcome</th>
              {sortTh('kalshi_yes_price', 'Ask', 'num')}<th className="num">Fair Value</th>{sortTh('ev_pct', 'EV', 'num')}{sortTh('stake', 'Stake', 'num')}
            </tr>
          </thead>
          <tbody>
            {visible.map(b => {
              const stake = stakeOf(b)
              return (
                <tr key={b.market_id}>
                  <td><input type="checkbox" checked={selected.has(b.market_id)} onChange={() => toggle(b.market_id)} /></td>
                  <td>
                    {scanMids.has(b.market_id) && <span className="badge new">NEW</span>}
                    {b.status === 'in_progress' && <span className="badge warn" title="Game has started — awaiting resolution">LIVE</span>}
                  </td>
                  <td className="muted">{b.event_date}</td>
                  <td><span className="badge">{b.sector}</span></td>
                  <td><VenueLogo venue={b.venue} /></td>
                  <td>{b.event_title}</td>
                  <td>{outcomeLabel(b)}</td>
                  <td className="num">{probToCents(b.kalshi_yes_price)}</td>
                  <td className="num">{probToCents(b.blended_true_prob)}</td>
                  <td className="num green">{((b.ev_pct || 0) * 100).toFixed(1)}%</td>
                  <td className="num">${stake.toFixed(2)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
        <div className="muted" style={{ padding: 8, display: 'flex', gap: 8, alignItems: 'center' }}>
          <span>Showing {visible.length} of {filtered.length}</span>
          {visible.length < filtered.length && (
            <>
              <button className="btn btn-sm" onClick={() => setLimit(l => l + PAGE)}>Show {Math.min(PAGE, filtered.length - visible.length)} more</button>
              <button className="btn btn-sm" onClick={() => setLimit(filtered.length)}>Show all</button>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
