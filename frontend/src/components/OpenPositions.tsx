import { usePersistedState } from '../hooks/usePersistedState'
import { useState, useMemo, useCallback } from 'react'
import type { Bet, ScanGap } from '../lib/types'
import { probToCents, outcomeLabel } from '../lib/odds'
import { SectorFilter } from './SectorFilter'
import { VenueLogo } from './VenueLogo'
import { useRowKeyboard } from '../hooks/useRowKeyboard'
import { pickBets, verifyBets, verifySummary } from '../lib/api'
import type { VerifyRow } from '../lib/api'
import { VerifyHeaderCells, VerifyCells } from './VerifyCells'
import {
  BUCKET_LABEL, BUCKET_ORDER, bucketStyle, formatTimeToTip, minutesToTip, tipBucket, useNow,
} from '../lib/tipTime'
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
  const [sector, setSector] = usePersistedState('open.sector', '')
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [verified, setVerified] = useState<Record<string, VerifyRow>>({})
  const [verifying, setVerifying] = useState(false)
  const [showLate, setShowLate] = useState(false)
  const now = useNow()

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

  const bucketOf = useCallback((b: Bet) => tipBucket(minutesToTip(b.tip_time, now)), [now])
  // Late / in-play rows are hidden unless toggled on; hidden rows are never picked.
  const active = useMemo(
    () => filtered.filter(b => showLate || bucketOf(b) !== 'late'),
    [filtered, showLate, bucketOf],
  )
  const hiddenCount = filtered.length - active.length
  const visible = useMemo(() => active.slice(0, limit), [active, limit])
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

  const handleVerify = useCallback(async () => {
    const ids = bets.filter(b => b.status !== 'in_progress').map(b => b.market_id).slice(0, 200)
    if (!ids.length) return
    setVerifying(true)
    try {
      const res = await verifyBets(ids, { bankroll, kelly })
      setVerified(Object.fromEntries(res.results.map(r => [r.market_id, r])))
      // Rows below the EV floor (or in-play / no quote) drop out of the pick set.
      setSelected(prev => new Set([...prev].filter(m => res.results.find(r => r.market_id === m)?.keep !== false)))
      toast(`Verified live: ${verifySummary(res.results)}${res.warnings.length ? ' · ' + res.warnings[0] : ''}`, res.live ? 'ok' : 'err')
    } catch (e) {
      toast('Verify failed: ' + (e as Error).message, 'err')
    } finally {
      setVerifying(false)
    }
  }, [bets, bankroll, kelly, toast])

  const handlePick = useCallback(async () => {
    const ok = new Set(active.map(b => b.market_id))
    const ids = [...selected].filter(id => ok.has(id))
    if (!ids.length) { toast('Select positions to pick first', 'info'); return }
    // Place at the CURRENT dashboard bankroll/kelly — i.e. the stake the preview
    // shows — not the scan-time bankroll_used × kelly_fraction the backend falls
    // back to when sent bare market_ids. Without this, a bet scanned at $250/0.5
    // and re-sized to $1100/0.75 in the UI would silently revert to the $250
    // stake on placement.
    const byId = new Map(bets.map(b => [b.market_id, b]))
    const payload = ids.map(mid => {
      const b = byId.get(mid)
      // A verified row is placed at its LIVE ask and live-sized stake; an
      // unverified one falls back to the scan-time ask (the old behaviour).
      const v = verified[mid]
      return {
        market_id: mid,
        fill_price: v?.live_ask ?? (b ? b.kalshi_yes_price : null),
        fill_stake: v?.keep && v.stake > 0
          ? v.stake
          : b ? Number(stakeOf(b).toFixed(2)) : null,
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
  }, [selected, bets, active, stakeOf, verified, toast, onPicked])

  const kbIds = useMemo(() => visible.map(b => b.market_id), [visible])
  const kb = useRowKeyboard({ ids: kbIds, onToggle: toggle, onPick: handlePick })

  return (
    <div className="panel">
      <div className="panel-header">
        <h2>Open Positions ({active.length})</h2>
        <div className="actions" style={{ gap: 4 }}>
          <SectorFilter value={sector} onChange={setSector} />
          {bets.length > 0 && (
            <button className="btn btn-sm" disabled={verifying} onClick={handleVerify}
              title="Re-fetch live asks + the Pinnacle line and re-gate each row at the current price (read-only)">
              {verifying ? 'Verifying...' : 'Verify live'}
            </button>
          )}
          <label className="muted" style={{ fontSize: 11 }}
            title="Games within 15 min of tip or already started. Hidden rows are excluded from Pick All.">
            <input type="checkbox" checked={showLate} onChange={e => setShowLate(e.target.checked)} />
            {' '}Late/in-play{hiddenCount > 0 && !showLate ? ` (${hiddenCount} hidden)` : ''}
          </label>
          {bets.length > 0 && <button className="btn btn-sm success" onClick={handlePick}>Pick All</button>}
        </div>
      </div>
      <div {...kb.containerProps} className="scroll-area kb-table">
        <table className="card-tbl st-open">
          <thead>
            <tr>
              <th style={{ width: 28 }}>
                <input type="checkbox" aria-label="Select all visible" checked={allVisibleSelected} onChange={toggleAllVisible} />
              </th><th></th>
              {sortTh('event_date', 'Date')}<th className="num" title="Time until the game starts (from scan-time minutes to tip)">To tip</th><th>Sector</th><th style={{ width: 34 }}>Venue</th><th>Event</th><th>Outcome</th>
              {sortTh('kalshi_yes_price', 'Scan Ask', 'num')}<VerifyHeaderCells /><th className="num">Fair Value</th>{sortTh('ev_pct', 'EV', 'num')}{sortTh('stake', 'Stake', 'num')}
            </tr>
          </thead>
          {BUCKET_ORDER.map(bk => {
            const rows = visible.filter(b => bucketOf(b) === bk)
            if (!rows.length) return null
            return (
          <tbody key={bk}>
            <tr><td colSpan={99} style={{ fontSize: 11, fontWeight: 600, paddingTop: 8 }}>
              {BUCKET_LABEL[bk]} <span className="muted">({rows.length})</span>
            </td></tr>
            {rows.map(b => {
              const stake = stakeOf(b)
              const v = verified[b.market_id]
              return (
                <tr key={b.market_id} {...kb.rowProps(b.market_id)} style={v && !v.keep ? { opacity: 0.55 } : undefined}
                  title={v && !v.keep ? `Excluded from Pick: ${v.reason}` : undefined}>
                  <td><input type="checkbox" aria-label={`Select ${b.event_title} ${outcomeLabel(b)}`} checked={selected.has(b.market_id)} onChange={() => toggle(b.market_id)} /></td>
                  <td>
                    {scanMids.has(b.market_id) && <span className="badge new">NEW</span>}
                    {b.status === 'in_progress' && <span className="badge warn" title="Game has started — awaiting resolution">LIVE</span>}
                  </td>
                  <td className="muted">{b.event_date}</td>
                  <td className="num" style={bucketStyle(bucketOf(b))}
                    title={b.tip_time ? new Date(b.tip_time).toLocaleString() : 'Tip time unknown'}>
                    {formatTimeToTip(minutesToTip(b.tip_time, now))}
                  </td>
                  <td><span className="badge">{b.sector}</span></td>
                  <td><VenueLogo venue={b.venue} /></td>
                  <td>{b.event_title}</td>
                  <td>{outcomeLabel(b)}</td>
                  <td className="num">{probToCents(b.kalshi_yes_price)}</td>
                  <VerifyCells v={v} />
                  <td className="num">{probToCents(b.blended_true_prob)}</td>
                  <td className="num green">{((b.ev_pct || 0) * 100).toFixed(1)}%</td>
                  <td className="num">${(v?.keep && v.stake > 0 ? v.stake : stake).toFixed(2)}</td>
                </tr>
              )
            })}
          </tbody>
            )
          })}
        </table>
        <div className="muted" style={{ padding: 8, display: 'flex', gap: 8, alignItems: 'center' }}>
          <span>Showing {visible.length} of {active.length}</span>
          {visible.length < active.length && (
            <>
              <button className="btn btn-sm" onClick={() => setLimit(l => l + PAGE)}>Show {Math.min(PAGE, active.length - visible.length)} more</button>
              <button className="btn btn-sm" onClick={() => setLimit(active.length)}>Show all</button>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
