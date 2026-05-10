import { useState } from 'react'
import { runScan, resolve } from '../lib/api'
import { fmtDate, fmtTomorrow, yesterday } from '../lib/odds'
import type { ScanGap } from '../lib/types'

interface Props {
  onScanComplete: (gaps: ScanGap[], meta: { markets_fetched: number; markets_matched: number }) => void
  onResolve: () => void
  onMetrics: () => void
  toast: (msg: string, type?: 'info' | 'ok' | 'err') => void
}

export function ActionBar({ onScanComplete, onResolve, onMetrics, toast }: Props) {
  const [sectors, setSectors] = useState('nba,wnba,soccer,tennis,ncaab')
  const [bankroll, setBankroll] = useState(250)
  const [kelly, setKelly] = useState(0.5)
  const [dateFrom, setDateFrom] = useState(fmtDate)
  const [dateTo, setDateTo] = useState(fmtTomorrow)
  const [scanning, setScanning] = useState(false)
  const [resolving, setResolving] = useState(false)

  const handleScan = async () => {
    setScanning(true)
    toast('Scanning markets...', 'info')
    try {
      const data = await runScan({ sectors, bankroll, kelly, date_from: dateFrom, date_to: dateTo })
      onScanComplete(data.gaps, { markets_fetched: data.markets_fetched, markets_matched: data.markets_matched })
      toast(`Found ${data.gaps.length} +EV plays across ${data.markets_fetched} markets`, 'ok')
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
      const r1 = await resolve(yesterday())
      const r2 = await resolve(fmtDate())
      const total = (r1.resolved || 0) + (r2.resolved || 0)
      toast(`Resolved ${total} outcome(s)`, 'ok')
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
        <option value="nba,wnba,soccer,tennis,ncaab">All Sectors</option>
        <option value="nba">NBA</option>
        <option value="wnba">WNBA</option>
        <option value="soccer">Soccer</option>
        <option value="tennis">Tennis</option>
        <option value="ncaab">NCAAB</option>
        <option value="baseball">Baseball</option>
        <option value="lol,cs2">Esports</option>
      </select>
      <input type="number" value={bankroll} onChange={e => setBankroll(+e.target.value)} style={{ width: 80 }} title="Bankroll ($)" />
      <select value={kelly} onChange={e => setKelly(+e.target.value)} title="Kelly fraction" style={{ width: 90 }}>
        <option value={0.1}>0.1× Kelly</option>
        <option value={0.25}>0.25× Kelly</option>
        <option value={0.5}>0.5× Kelly</option>
        <option value={0.75}>0.75× Kelly</option>
        <option value={1}>Full Kelly</option>
      </select>
      <input type="date" value={dateFrom} onChange={e => setDateFrom(e.target.value)}
        style={{ width: 130, fontSize: 11, padding: '4px 6px', background: '#1a1a2e', color: '#e0e0e0', border: '1px solid #333', borderRadius: 4 }} />
      <span style={{ color: '#888' }}>–</span>
      <input type="date" value={dateTo} onChange={e => setDateTo(e.target.value)}
        style={{ width: 130, fontSize: 11, padding: '4px 6px', background: '#1a1a2e', color: '#e0e0e0', border: '1px solid #333', borderRadius: 4 }} />
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
