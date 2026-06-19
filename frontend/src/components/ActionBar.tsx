import { useEffect, useState } from 'react'
import { runScan, resolve } from '../lib/api'
import { useCategories } from '../lib/useCategories'
import { fmtDate, fmtTomorrow, yesterday } from '../lib/odds'
import type { ScanGap } from '../lib/types'

interface Props {
  bankrollStr: string
  setBankrollStr: (s: string) => void
  kelly: number
  setKelly: (k: number) => void
  onScanComplete: (
    gaps: ScanGap[],
    meta: { markets_fetched: number; markets_matched: number },
    snapshot: { bankroll: number; kelly: number },
  ) => void
  onResolve: () => void
  onMetrics: () => void
  toast: (msg: string, type?: 'info' | 'ok' | 'err') => void
}

export function ActionBar({ bankrollStr, setBankrollStr, kelly, setKelly, onScanComplete, onResolve, onMetrics, toast }: Props) {
  const cats = useCategories()
  const allSectors = cats.map(c => c.key).join(',')
  // Sector list comes from the registry via /api/categories. Default the
  // selection to "all sectors" once the list loads, but never clobber a choice
  // the user has already made.
  const [sectors, setSectors] = useState('')
  useEffect(() => {
    setSectors(prev => prev || allSectors)
  }, [allSectors])
  // Bankroll stored as a string so leading zeros the user types get stripped
  // explicitly. The previous Number-typed state had a known React bug where
  // typing "0" before an existing number left the visual "0…" in place
  // because the parsed value didn't change, so React never re-rendered the
  // input. String state with onChange-time sanitization fixes that.
  const bankroll = Number(bankrollStr) || 0
  const [dateFrom, setDateFrom] = useState(fmtDate)
  const [dateTo, setDateTo] = useState(fmtTomorrow)
  const [scanning, setScanning] = useState(false)
  const [resolving, setResolving] = useState(false)

  const handleBankrollChange = (raw: string) => {
    // Strip leading zeros unless the user is typing "0" or "0.xxx" — keep
    // empty string allowed so the field can be cleared during editing.
    if (raw === '' || raw === '0' || raw.startsWith('0.')) {
      setBankrollStr(raw)
      return
    }
    setBankrollStr(raw.replace(/^0+/, ''))
  }

  const handleScan = async () => {
    setScanning(true)
    toast('Scanning markets...', 'info')
    try {
      const data = await runScan({ sectors, bankroll, kelly, date_from: dateFrom, date_to: dateTo })
      onScanComplete(
        data.gaps,
        { markets_fetched: data.markets_fetched, markets_matched: data.markets_matched },
        { bankroll, kelly },
      )
      const portfolioBets = (data.portfolio_results || []).reduce((s, r) => s + r.gaps_logged, 0)
      const portfolioSuffix = data.portfolio_results?.length
        ? ` · logged ${portfolioBets} to ${data.portfolio_results.length} portfolio(s)`
        : ''
      toast(`Found ${data.gaps.length} +EV plays across ${data.markets_fetched} markets${portfolioSuffix}`, 'ok')
    } catch (e) {
      toast('Scan failed: ' + (e as Error).message, 'err')
    } finally {
      setScanning(false)
    }
  }

  const handleResolve = async () => {
    setResolving(true)
    toast('Resolving outcomes...', 'info')
    try {
      // Resolve yesterday + today, then sync portfolios once on the second
      // call (after both dates' ev_outcomes are populated). One click now
      // covers both dashboard and portfolio outcomes.
      const r1 = await resolve(yesterday(), false)
      const r2 = await resolve(fmtDate(), true)
      const total = (r1.resolved || 0) + (r2.resolved || 0)
      const portfolio = r2.portfolio_resolved || 0
      const portfolioSuffix = portfolio ? ` · synced ${portfolio} portfolio bet(s)` : ''
      toast(`Resolved ${total} outcome(s)${portfolioSuffix}`, 'ok')
      onResolve()
    } catch (e) {
      toast('Resolve failed: ' + (e as Error).message, 'err')
    } finally {
      setResolving(false)
    }
  }

  return (
    <div className="actions">
      <select value={sectors} onChange={e => setSectors(e.target.value)}>
        <option value={allSectors}>All Sectors</option>
        {cats.map(c => (
          <option key={c.key} value={c.key} title={c.display_name}>{c.key.toUpperCase()}</option>
        ))}
      </select>
      <input
        type="number"
        value={bankrollStr}
        onChange={e => handleBankrollChange(e.target.value)}
        style={{ width: 80 }}
        title="Bankroll ($)"
      />
      <select value={kelly} onChange={e => setKelly(+e.target.value)} title="Kelly fraction" style={{ width: 112 }}>
        <option value={0.1}>0.1× Kelly</option>
        <option value={0.25}>0.25× Kelly</option>
        <option value={0.5}>0.5× Kelly</option>
        <option value={0.75}>0.75× Kelly</option>
        <option value={1}>Full Kelly</option>
      </select>
      <input type="date" value={dateFrom} onChange={e => setDateFrom(e.target.value)}
        style={{ width: 138, fontSize: 11 }} title="From date" />
      <span className="muted">–</span>
      <input type="date" value={dateTo} onChange={e => setDateTo(e.target.value)}
        style={{ width: 138, fontSize: 11 }} title="To date" />
      <button className="btn primary" onClick={handleScan} disabled={scanning}>
        {scanning ? 'Scanning...' : 'Scan'}
      </button>
      <button className="btn" onClick={handleResolve} disabled={resolving}>
        {resolving ? 'Resolving...' : 'Resolve'}
      </button>
      <button className="btn" onClick={onMetrics}>Metrics</button>
    </div>
  )
}
