import { useState, useMemo, useCallback } from 'react'
import type { Bet, ScanGap } from '../lib/types'
import { probToCents, outcomeLabel } from '../lib/odds'
import { SectorFilter } from './SectorFilter'
import { VenueLogo } from './VenueLogo'
import { pickBets } from '../lib/api'

interface Props {
  bets: Bet[]
  scanGaps: ScanGap[]
  bankroll: number
  kelly: number
  toast: (msg: string, type?: 'info' | 'ok' | 'err') => void
  onPicked: () => void
}

// Stored kelly_fraction = K_full × scan_kelly × discounts, capped at 5%.
// Dashboard scans default to kelly=0.5, so scale by (current_kelly / 0.5) to
// preview a different kelly multiplier. Bankroll scales linearly.
const SCAN_KELLY_BASELINE = 0.5

function previewStake(b: Bet, bankroll: number, kelly: number): number {
  const stored = b.kelly_fraction || 0
  const scaled = stored * (kelly / SCAN_KELLY_BASELINE)
  return bankroll * Math.min(scaled, 0.05)
}

export function OpenPositions({ bets, scanGaps, bankroll, kelly, toast, onPicked }: Props) {
  const [sector, setSector] = useState('')
  const [selected, setSelected] = useState<Set<string>>(new Set())

  const scanMids = useMemo(() => new Set(scanGaps.map(g => g.market_id)), [scanGaps])

  const filtered = useMemo(() => {
    return sector ? bets.filter(b => b.sector === sector) : bets
  }, [bets, sector])

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
        fill_stake: b ? Number(previewStake(b, bankroll, kelly).toFixed(2)) : null,
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
  }, [selected, bets, bankroll, kelly, toast, onPicked])

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
              <th style={{ width: 28 }}></th><th></th>
              <th>Date</th><th>Sector</th><th style={{ width: 34 }}>Venue</th><th>Event</th><th>Outcome</th>
              <th className="num">Ask</th><th className="num">Fair Value</th><th className="num">EV</th><th className="num">Stake</th>
            </tr>
          </thead>
          <tbody>
            {filtered.slice(0, 40).map(b => {
              const stake = previewStake(b, bankroll, kelly)
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
      </div>
    </div>
  )
}
