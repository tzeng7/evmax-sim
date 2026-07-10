import { useEffect, useRef, useState } from 'react'
import { fetchArbSectors, runArbScan } from '../lib/api'
import type { ArbBasket, ArbScanResult } from '../lib/types'

interface Props {
  toast: (msg: string, type?: 'info' | 'ok' | 'err') => void
}

/**
 * Cross-venue arbitrage tab. Runs the read-only /api/arb/scan (Kalshi vs
 * Polymarket US complete-outcome baskets, fee-netted) on demand — nothing is
 * persisted and no bankroll is touched. Steady-state arbs between the venues
 * essentially don't exist, so an empty result is the expected outcome; the
 * tab exists to catch transient dislocations.
 */
export function ArbsPage({ toast }: Props) {
  const [sectors, setSectors] = useState('') // comma-separated; '' = all
  const [maxCost, setMaxCost] = useState('1.02')
  const [includeInPlay, setIncludeInPlay] = useState(false)
  const [includeSingleVenue, setIncludeSingleVenue] = useState(false)
  const [scanning, setScanning] = useState(false)
  const [result, setResult] = useState<ArbScanResult | null>(null)

  const scan = async () => {
    setScanning(true)
    try {
      const r = await runArbScan({
        sectors,
        max_cost: Number(maxCost) || 1.02,
        include_in_play: includeInPlay,
        include_single_venue: includeSingleVenue,
      })
      setResult(r)
      const positive = r.baskets.filter(b => b.net_edge > 0).length
      toast(
        positive > 0
          ? `${positive} net-positive arb basket(s) found!`
          : `Scan complete — ${r.baskets.length} basket(s) under cost ceiling, none net-positive`,
        positive > 0 ? 'ok' : 'info',
      )
    } catch (e) {
      toast('Arb scan failed: ' + (e as Error).message, 'err')
    } finally {
      setScanning(false)
    }
  }

  return (
    <>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16, flexWrap: 'wrap' }}>
        <h2 style={{ margin: 0 }}>Cross-Venue Arbs</h2>
        <span className="muted">Kalshi vs Polymarket US — complete-outcome baskets under $1 after taker fees</span>
        <div style={{ flex: 1 }} />
        <ArbSectorSelect value={sectors} onChange={setSectors} />
        <label className="muted" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          Max cost
          <input
            className="input"
            style={{ width: 64 }}
            value={maxCost}
            onChange={e => setMaxCost(e.target.value)}
            title="Show baskets whose fee-inclusive cost is at most this (1.0 = true arbs only)"
          />
        </label>
        <label className="muted" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <input type="checkbox" className="check" checked={includeInPlay} onChange={e => setIncludeInPlay(e.target.checked)} />
          In-play
        </label>
        <label className="muted" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <input type="checkbox" className="check" checked={includeSingleVenue} onChange={e => setIncludeSingleVenue(e.target.checked)} />
          Single-venue
        </label>
        <button className="btn success" disabled={scanning} onClick={scan}>
          {scanning ? 'Scanning…' : 'Scan for Arbs'}
        </button>
      </div>

      {result && (
        <div className="muted" style={{ marginBottom: 12, fontSize: 12 }}>
          {result.sector_stats
            .filter(s => s.kalshi_markets + s.polyus_markets > 0)
            .map(s => `${s.sector} K:${s.kalshi_markets}/P:${s.polyus_markets}`)
            .join(' · ') || 'No markets fetched'}
          {' — asks are REST snapshots; verify depth and price live before acting.'}
        </div>
      )}

      {result && <ArbTable baskets={result.baskets} />}
      {!result && !scanning && (
        <div className="panel" style={{ padding: 24, textAlign: 'center' }}>
          <span className="muted">
            Run a scan to fetch both venues' books. An empty result is normal —
            the venues run ~1pp of combined vig against ~3pp of round-trip taker
            fees, so this tab is for catching transient dislocations.
          </span>
        </div>
      )}
    </>
  )
}

function ArbTable({ baskets }: { baskets: ArbBasket[] }) {
  if (baskets.length === 0) {
    return (
      <div className="panel" style={{ padding: 24, textAlign: 'center' }}>
        <span className="muted">No baskets under the cost ceiling.</span>
      </div>
    )
  }
  return (
    <div className="panel">
      <div className="panel-header">
        <h3>Arb Baskets</h3>
        <span className="muted">{baskets.length} under ceiling · sorted by net cost</span>
      </div>
      <table>
        <thead>
          <tr>
            <th>Event</th><th>Outcome</th><th>Legs</th>
            <th className="num">Gross</th><th className="num">Net</th><th className="num">Edge</th>
            <th>Start (UTC)</th><th className="num">Min Vol $</th>
          </tr>
        </thead>
        <tbody>
          {baskets.map((b, i) => (
            <tr key={i} style={b.net_edge > 0 ? { background: 'rgba(46, 200, 102, 0.08)' } : undefined}>
              <td>
                <span className="badge">{b.sector}</span>{' '}
                {b.event_title}
                {b.in_play && <span className="badge" style={{ marginLeft: 6 }}>in-play</span>}
              </td>
              <td>{b.market_desc}</td>
              <td style={{ fontSize: 11 }}>
                {b.legs.map((leg, j) => (
                  <div key={j}>
                    <span className={leg.venue === 'kalshi' ? 'green' : 'yellow'}>
                      {leg.venue === 'kalshi' ? 'K' : 'P'}
                    </span>
                    {`:${leg.side} ${leg.outcome} @${leg.ask.toFixed(2)}`}
                  </div>
                ))}
              </td>
              <td className="num">{b.gross_cost.toFixed(3)}</td>
              <td className="num">{b.net_cost.toFixed(3)}</td>
              <td className={`num ${b.net_edge > 0 ? 'green' : 'muted'}`}>
                {(b.net_edge * 100).toFixed(2)}pp
              </td>
              <td className="muted">{b.event_date ? b.event_date.slice(5, 16).replace('T', ' ') : '?'}</td>
              <td className="num muted">{Math.min(...b.legs.map(l => l.volume_usd)).toFixed(0)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/** Sector picker fed by /api/arb/sectors (ARB_LEAGUE_MAP — a superset of the
 * betting categories, so the scan-bar SectorMultiSelect can't be reused). */
function ArbSectorSelect({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const [all, setAll] = useState<string[]>([])
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    fetchArbSectors().then(setAll).catch(() => setAll([]))
  }, [])

  useEffect(() => {
    if (!open) return
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onDoc)
    return () => document.removeEventListener('mousedown', onDoc)
  }, [open])

  const selected = new Set(value ? value.split(',').filter(Boolean) : [])
  const label = selected.size === 0
    ? 'All Sectors'
    : selected.size === 1
      ? [...selected][0].toUpperCase()
      : `${selected.size} sectors`

  const toggle = (key: string) => {
    const next = new Set(selected)
    if (next.has(key)) next.delete(key)
    else next.add(key)
    onChange(all.filter(k => next.has(k)).join(','))
  }

  return (
    <div className="sector-ms" ref={ref}>
      <button className="btn btn-sm" onClick={() => setOpen(o => !o)}>{label} ▾</button>
      {open && (
        <div className="sector-ms-menu">
          <label className="sector-ms-item">
            <input
              type="checkbox"
              className="check"
              checked={selected.size === 0}
              onChange={() => onChange('')}
            />
            All (default)
          </label>
          {all.map(k => (
            <label key={k} className="sector-ms-item">
              <input type="checkbox" className="check" checked={selected.has(k)} onChange={() => toggle(k)} />
              {k}
            </label>
          ))}
        </div>
      )}
    </div>
  )
}
