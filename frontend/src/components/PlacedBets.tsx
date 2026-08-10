import { useState, useCallback } from 'react'
import type { Bet } from '../lib/types'
import type { ToastOptions } from '../hooks/useToast'
import { probToCents, centsToProb, outcomeLabel } from '../lib/odds'
import { SectorFilter } from './SectorFilter'
import { VenueLogo } from './VenueLogo'
import { updatePlaced, unplaceBets, pickBets } from '../lib/api'

interface Props {
  bets: Bet[]
  toast: (msg: string, type?: 'info' | 'ok' | 'err', opts?: ToastOptions) => void
  onChanged: () => void
}

export function PlacedBets({ bets, toast, onChanged }: Props) {
  const [sector, setSector] = useState('')
  const [edits, setEdits] = useState<Record<string, { odds: string; stake: string }>>({})
  const [dirty, setDirty] = useState(false)

  const filtered = sector ? bets.filter(b => b.sector === sector) : bets

  const getFill = (b: Bet) => b.placed_price || b.kalshi_yes_price || 0.5
  const getStake = (b: Bet) => b.placed_stake || (b.bankroll_used || 500) * (b.kelly_fraction || 0)

  const updateField = (mid: string, field: 'odds' | 'stake', value: string) => {
    setEdits(prev => {
      const existing = prev[mid] || {}
      return { ...prev, [mid]: { ...existing, [field]: value } }
    })
    setDirty(true)
  }

  const getOdds = (b: Bet) => edits[b.market_id]?.odds ?? probToCents(getFill(b))
  const getStakeVal = (b: Bet) => edits[b.market_id]?.stake ?? getStake(b).toFixed(2)

  const toWin = (b: Bet) => {
    const odds = getOdds(b)
    const stake = parseFloat(getStakeVal(b)) || 0
    const prob = centsToProb(odds)
    if (prob && prob > 0 && prob < 1) return stake * (1 / prob - 1)
    return 0
  }

  const handleSave = useCallback(async () => {
    // Only send fields the user actually edited (non-empty string). Backend
    // does a partial UPDATE so untouched columns keep their existing values
    // — sending null/0 for an un-edited field would clobber it.
    const items = Object.entries(edits).map(([mid, e]) => ({
      market_id: mid,
      fill_price: e.odds && e.odds.trim() !== '' ? centsToProb(e.odds) : null,
      fill_stake: e.stake !== undefined && e.stake !== '' ? parseFloat(e.stake) : null,
    })).filter(it => it.fill_price !== null || it.fill_stake !== null)
    if (!items.length) return
    try {
      const res = await updatePlaced(items)
      toast(`Updated ${res.updated} bet(s)`, 'ok')
      setDirty(false)
      setEdits({})
      onChanged()
    } catch (e) {
      toast('Save failed: ' + (e as Error).message, 'err')
    }
  }, [edits, toast, onChanged])

  // Removal is reversible, so skip the blocking confirm() and offer Undo in the
  // toast instead (§16.2 forgiveness over friction). Undo re-places the bet at
  // the exact fill it was removed at — captured before the async call so the
  // refreshed list can't strip it out from under us.
  const handleRemove = useCallback(async (b: Bet) => {
    const mid = b.market_id
    const fillPrice = b.placed_price || b.kalshi_yes_price || 0.5
    const fillStake = b.placed_stake || (b.bankroll_used || 500) * (b.kelly_fraction || 0)
    try {
      const res = await unplaceBets([mid])
      if (res.removed > 0) {
        onChanged()
        toast('Bet removed', 'ok', {
          action: {
            label: 'Undo',
            onClick: async () => {
              try {
                await pickBets([{ market_id: mid, fill_price: fillPrice, fill_stake: fillStake }])
                toast('Bet restored', 'ok')
                onChanged()
              } catch (e) {
                toast('Undo failed: ' + (e as Error).message, 'err')
              }
            },
          },
        })
      }
    } catch (e) {
      toast('Remove failed: ' + (e as Error).message, 'err')
    }
  }, [toast, onChanged])

  const totalExposure = filtered.reduce((sum, b) => sum + (parseFloat(getStakeVal(b)) || 0), 0)

  // Guard placed AFTER every hook so the hook count is identical whether bets
  // is empty or populated. An early return above the useCallbacks made React
  // run fewer hooks on the empty render and more once a bet was placed →
  // error #310 "rendered more hooks than during the previous render".
  if (!bets.length) return null

  return (
    <div className="panel">
      <div className="panel-header">
        <h2>Placed Bets ({filtered.length})</h2>
        <div className="actions" style={{ gap: 4 }}>
          <SectorFilter value={sector} onChange={setSector} />
          {dirty && <button className="btn btn-sm" onClick={handleSave}>Save Changes</button>}
        </div>
      </div>
      <table>
        <thead>
          <tr>
            <th style={{ width: 30 }}></th>
            <th>Date</th><th>Sector</th><th style={{ width: 34 }}>Venue</th><th>Event</th><th>Outcome</th>
            <th className="num">Model</th><th className="num">EV</th>
            <th className="num">Fill ¢</th><th className="num">Stake ($)</th><th className="num">To Win</th>
          </tr>
        </thead>
        <tbody>
          {filtered.map(b => (
            <tr key={b.market_id}>
              <td><button className="btn-del" onClick={() => handleRemove(b)}>&times;</button></td>
              <td className="muted">{b.event_date}</td>
              <td>
                <span className="badge">{b.sector}</span>
                {b.maker_fill === 1 && (
                  <span
                    className="badge"
                    style={{ marginLeft: 4, background: 'rgba(198,120,221,0.13)', color: '#c678dd', borderColor: 'rgba(198,120,221,0.32)' }}
                    title="Placed as a maker fill — P&L uses the maker fee"
                  >MAKER</span>
                )}
              </td>
              <td><VenueLogo venue={b.venue} /></td>
              <td>{b.event_title}</td>
              <td>{outcomeLabel(b)}</td>
              <td className="num">{Math.round((b.blended_true_prob || 0) * 100)}%</td>
              <td className="num green">{((b.ev_pct || 0) * 100).toFixed(1)}%</td>
              <td className="num">
                <input type="text" value={getOdds(b)}
                  onChange={e => updateField(b.market_id, 'odds', e.target.value)}
                  style={{ width: 64 }} />
              </td>
              <td className="num">
                <input type="number" value={getStakeVal(b)}
                  onChange={e => updateField(b.market_id, 'stake', e.target.value)}
                  style={{ width: 70 }} min="0.01" step="0.01" />
              </td>
              <td className="num green">${toWin(b).toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div style={{ marginTop: 8, fontSize: 11, color: 'var(--muted)' }}>
        Total exposure: ${totalExposure.toFixed(2)}
      </div>
    </div>
  )
}
