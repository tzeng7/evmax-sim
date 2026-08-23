import { useState, useMemo, useCallback, useEffect } from 'react'
import type { ScanGap } from '../lib/types'
import { probToCents, centsToProb, outcomeLabel } from '../lib/odds'
import { SectorFilter } from './SectorFilter'
import { VenueLogo } from './VenueLogo'
import { pickBets, recordMakerFills } from '../lib/api'

interface Props {
  gaps: ScanGap[]
  meta: { markets_fetched: number; markets_matched: number } | null
  bankroll: number
  kelly: number
  scanKelly: number
  toast: (msg: string, type?: 'info' | 'ok' | 'err') => void
  onPicked: () => void
}

// Short venue label for the best-execution "· also {venue}" annotation.
function venueShort(v?: string | null): string {
  if (v === 'polymarket_us') return 'Poly'
  if (v === 'kalshi') return 'Kalshi'
  return v ?? ''
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

// Default Fill ¢ / Stake ($) for a row. Maker-only plays are not crossable at
// the ask (taker Kelly is zeroed), so seed them to the actionable resting bid
// and the maker-sized Kelly stake instead — the numbers you'd actually place
// and later record with `evmax agents fill`. All other rows use the taker ask
// and the taker Kelly stake.
function defaultFill(g: ScanGap, bankroll: number, kelly: number, scanKelly: number): { odds: string; stake: string } {
  if (g.maker_only && g.maker_bid_price != null) {
    const frac = g.maker_bid_kelly_fraction ?? 0
    return { odds: probToCents(g.maker_bid_price), stake: (bankroll * Math.min(frac, 0.05)).toFixed(2) }
  }
  return { odds: probToCents(g.kalshi_price), stake: recomputedStake(g, bankroll, kelly, scanKelly).toFixed(2) }
}

export function ScanResults({ gaps, meta, bankroll, kelly, scanKelly, toast, onPicked }: Props) {
  const [sector, setSector] = useState('')
  const [dateFilter, setDateFilter] = useState('')
  const [selected, setSelected] = useState<Set<string>>(() => new Set(gaps.map(g => g.market_id)))
  const [fills, setFills] = useState<Record<string, { odds: string; stake: string }>>({})
  const [picking, setPicking] = useState(false)
  const [filling, setFilling] = useState(false)

  // Reset selection when gaps change
  useMemo(() => {
    setSelected(new Set(gaps.filter(g => (g.mode || 'live') === 'live').map(g => g.market_id)))
    const f: Record<string, { odds: string; stake: string }> = {}
    for (const g of gaps) {
      f[g.market_id] = defaultFill(g, bankroll, kelly, scanKelly)
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
        const def = defaultFill(g, bankroll, kelly, scanKelly)
        next[g.market_id] = {
          odds: prev[g.market_id]?.odds ?? def.odds,
          stake: def.stake,
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
      if (s.has(mid)) {
        s.delete(mid)
      } else {
        s.add(mid)
      }
      return s
    })
  }

  const selectAll = () => setSelected(new Set(filtered.filter(g => (g.mode || 'live') === 'live').map(g => g.market_id)))
  const deselectAll = () => setSelected(new Set())

  const updateFill = (mid: string, field: 'odds' | 'stake', value: string) => {
    setFills(prev => ({ ...prev, [mid]: { ...prev[mid], [field]: value } }))
  }

  const handlePick = useCallback(async () => {
    // Maker-only rows are excluded: they are not crossable at the ask, so
    // /api/pick rejects them. They go through handleFill instead.
    const bets = gaps
      .filter(g => !g.maker_only && selected.has(g.market_id))
      .map(g => ({
        market_id: g.market_id,
        fill_price: centsToProb(fills[g.market_id]?.odds || ''),
        fill_stake: parseFloat(fills[g.market_id]?.stake || '0'),
      }))
    if (!bets.length) return
    setPicking(true)
    try {
      const res = await pickBets(bets)
      if (res.placed > 0) {
        const skippedNote = res.skipped?.length ? ` · skipped ${res.skipped.length}` : ''
        const cappedNote = res.cash_capped ? ` · capped to venue cash (${Object.keys(res.cash_capped).join(', ')})` : ''
        toast(`Placed ${res.placed} bet(s)${skippedNote}${cappedNote}`, 'ok')
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
  }, [gaps, selected, fills, toast, onPicked])

  const handleFill = useCallback(async () => {
    const rows = gaps
      .filter(g => g.maker_only && selected.has(g.market_id))
      .map(g => ({
        market_id: g.market_id,
        fill_price: centsToProb(fills[g.market_id]?.odds || ''),
        fill_stake: parseFloat(fills[g.market_id]?.stake || '0'),
      }))
    if (!rows.length) return
    setFilling(true)
    try {
      const res = await recordMakerFills(rows)
      if (res.filled > 0) {
        const fee = res.fills.reduce((a, f) => a + f.maker_fee, 0)
        const skippedNote = res.skipped?.length ? ` \u00b7 skipped ${res.skipped.length}` : ''
        toast(`Recorded ${res.filled} maker fill(s) \u00b7 fees $${fee.toFixed(2)}${skippedNote}`, 'ok')
      } else {
        const first = res.skipped?.[0]
        const more = (res.skipped?.length ?? 0) > 1 ? ` (+${(res.skipped!.length) - 1} more)` : ''
        toast(first ? `Recorded 0 \u2014 ${first.reason}${more}` : 'Recorded 0 maker fill(s)', 'err')
      }
      onPicked()
    } catch (e) {
      toast('Maker fill failed: ' + (e as Error).message, 'err')
    } finally {
      setFilling(false)
    }
  }, [gaps, selected, fills, toast, onPicked])

  if (!gaps.length) return null

  const checkedRows = filtered.filter(g => selected.has(g.market_id))
  const pickCount = checkedRows.filter(g => !g.maker_only).length
  const fillCount = checkedRows.filter(g => g.maker_only).length

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
            <th className="num" title="EV if opened as a resting limit order (maker fee)">Maker EV</th>
            <th className="num" title="Ceiling — rest at or below this and you stay ≥ the EV floor as a maker. Can sit ABOVE the ask; it is not a place-order price. Use Bid ¢.">Limit ¢</th>
            <th className="num" title="The bid to SET: rest your maker buy here (one tick above the current best bid, still below the ask). Fills as a maker, not a taker. Record the fill with `evmax agents fill`.">Bid ¢</th>
            <th className="num">Fill ¢</th><th className="num">Stake ($)</th><th>Models</th>
          </tr>
        </thead>
        <tbody>
          {filtered.map(g => {
            const askOdds = probToCents(g.kalshi_price)
            const outcome = outcomeLabel(g)
            const mode = g.mode || 'live'
            const isLive = mode === 'live'
            // A maker-only row logs as shadow (fill-contingent) but is still
            // selectable — its selection arms "Record Maker Fill", never the
            // taker pick, which the ask price would make -EV.
            const isMaker = !!g.maker_only
            const selectable = isLive || isMaker
            return (
              <tr key={g.market_id} style={isLive ? undefined : { opacity: 0.55 }}>
                <td>
                  <input type="checkbox" checked={selected.has(g.market_id)}
                    disabled={!selectable}
                    title={
                      isMaker
                        ? 'Maker-only — select, set Fill ¢ / Stake to what actually filled, then "Record Maker Fill"'
                        : isLive ? '' : `${mode} mode — not pickable`
                    }
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
                  {g.maker_only && (
                    <span
                      className="badge"
                      style={{ marginLeft: 4, background: 'rgba(198,120,221,0.13)', color: '#c678dd', borderColor: 'rgba(198,120,221,0.32)' }}
                      title="Clears the EV floor only as a resting limit order (maker fee), not crossable at the ask. Rest your buy at the Bid ¢ price (Fill ¢ / Stake are pre-seeded to it). Once it fills, tick this row and hit Record Maker Fill (or run `evmax agents fill`)."
                    >MAKER</span>
                  )}
                </td>
                <td><VenueLogo venue={g.venue} /></td>
                <td>{g.event_title}</td>
                <td>
                  {outcome}
                  {g.alt_venue && (
                    <span
                      className="muted"
                      style={{ marginLeft: 6, fontSize: 10 }}
                      title={`Same bet also +EV on ${venueShort(g.alt_venue)}`
                        + (g.alt_venue_price != null ? ` at ${probToCents(g.alt_venue_price)}` : '')
                        + (g.alt_venue_ev_pct != null ? ` (EV ${g.alt_venue_ev_pct.toFixed(1)}%)` : '')
                        + ' — this row is the better book'}
                    >
                      · also {venueShort(g.alt_venue)}
                      {g.alt_venue_price != null ? ` ${probToCents(g.alt_venue_price)}` : ''}
                    </span>
                  )}
                </td>
                <td className="num">{askOdds}</td>
                <td className="num">{probToCents(g.true_prob)}</td>
                <td className="num">{(g.true_prob * 100).toFixed(1)}%</td>
                <td className="num green">{g.ev_pct.toFixed(1)}%</td>
                <td className="num" style={{ color: '#c678dd' }}>
                  {g.maker_ev_pct != null ? `${g.maker_ev_pct.toFixed(1)}%` : '—'}
                </td>
                <td className="num" style={{ color: '#c678dd' }}>
                  {g.maker_limit_price != null ? probToCents(g.maker_limit_price) : '—'}
                </td>
                <td className="num" style={{ color: '#c678dd', fontWeight: 600 }}
                  title={g.maker_bid_ev_pct != null ? `+${g.maker_bid_ev_pct.toFixed(1)}% EV if filled here` : undefined}>
                  {g.maker_bid_price != null ? probToCents(g.maker_bid_price) : '—'}
                </td>
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
      <div style={{ marginTop: 8, display: 'flex', gap: 6, alignItems: 'center' }}>
        <button className="btn success" disabled={pickCount === 0 || picking} onClick={handlePick}>
          {picking ? 'Placing...' : `Pick Selected (${pickCount})`}
        </button>
        <button
          className="btn"
          style={{ borderColor: 'rgba(198,120,221,0.45)', color: '#c678dd' }}
          disabled={fillCount === 0 || filling}
          onClick={handleFill}
          title="Record selected MAKER rows as filled limit orders — placed at the Fill ¢ / Stake shown, charged the maker fee."
        >
          {filling ? 'Recording...' : `Record Maker Fill (${fillCount})`}
        </button>
      </div>
    </div>
  )
}
