import { useState } from 'react'
import { runArbScan } from '../lib/api'
import type { ArbRow, ArbScanResult } from '../lib/types'
import { probToCents } from '../lib/odds'
import { VenueLogo } from './VenueLogo'
import { SectorMultiSelect } from './SectorMultiSelect'

interface Props {
  toast: (msg: string, type?: 'info' | 'ok' | 'err') => void
}

/**
 * Arb page — dashboard front-end for `evmax arb scan`. Read-only: fetches
 * both venues' live books and shows complete-outcome baskets whose
 * fee-inclusive cost is under the ceiling. Steady-state arbs between the
 * venues essentially don't exist, so hits are transient dislocations to
 * verify by hand — the asks are REST snapshots without depth confirmation.
 */
export function ArbPage({ toast }: Props) {
  const [sectors, setSectors] = useState('')
  const [maxCostStr, setMaxCostStr] = useState('1.02')
  const [includeInPlay, setIncludeInPlay] = useState(false)
  const [scanning, setScanning] = useState(false)
  const [result, setResult] = useState<ArbScanResult | null>(null)

  const handleScan = async () => {
    setScanning(true)
    try {
      const res = await runArbScan({
        sectors,
        max_cost: Number(maxCostStr) || 1.02,
        include_in_play: includeInPlay,
      })
      setResult(res)
      const pos = res.arbs.filter(a => a.net_edge > 0).length
      if (pos > 0) toast(`${pos} net-positive basket(s) found — verify depth live before acting`, 'ok')
      else toast(res.arbs.length ? `No net arbs — ${res.arbs.length} near-miss(es) under the ceiling` : 'No baskets under the cost ceiling', 'info')
    } catch (e) {
      toast('Arb scan failed: ' + (e as Error).message, 'err')
    } finally {
      setScanning(false)
    }
  }

  return (
    <>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16, flexWrap: 'wrap' }}>
        <h2 style={{ margin: 0 }}>Cross-Venue Arb</h2>
        <span className="muted">Kalshi ↔ Polymarket US complete-outcome baskets under $1 after taker fees</span>
        <div style={{ flex: 1 }} />
        <SectorMultiSelect value={sectors} onChange={setSectors} emptyLabel="Default sectors" />
        <input
          type="number"
          step="0.01"
          min="0.9"
          max="1.1"
          value={maxCostStr}
          onChange={e => setMaxCostStr(e.target.value)}
          style={{ width: 72 }}
          title="Max fee-inclusive basket cost (1.0 = true arbs only; 1.02 keeps 2pp near-misses visible)"
        />
        <label className="muted" style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={includeInPlay}
            onChange={e => setIncludeInPlay(e.target.checked)}
            style={{ accentColor: 'var(--amber)', cursor: 'pointer' }}
          />
          In-play
        </label>
        <button className="btn primary" onClick={handleScan} disabled={scanning}>
          {scanning ? 'Scanning...' : 'Scan'}
        </button>
      </div>

      {scanning && <div className="skeleton" style={{ height: 240 }} />}

      {!scanning && result && (
        <>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 14 }}>
            {result.sectors.map(s => (
              <span key={s.sector} className="badge" title={`Kalshi ${s.kalshi} · Polymarket US ${s.polymarket_us} markets fetched`}>
                {s.sector}: {s.baskets} basket{s.baskets === 1 ? '' : 's'}
              </span>
            ))}
          </div>

          {result.arbs.length === 0 && (
            <div className="panel">
              <p className="muted" style={{ margin: 0 }}>
                No baskets under the {result.max_cost.toFixed(2)} cost ceiling. Steady-state arbs
                don't exist between the venues (~1pp combined vig vs ~3.25pp round-trip taker fees) —
                re-scan around Kalshi's T-24h listing window for transient dislocations.
              </p>
            </div>
          )}

          {result.arbs.length > 0 && (
            <div className="panel">
              <div className="panel-header">
                <h2>Baskets ≤ {result.max_cost.toFixed(2)}</h2>
                <span className="muted">sorted cheapest first · edge = $1.00 payout − fee-inclusive cost</span>
              </div>
              <div className="scroll-area">
                <table>
                  <thead>
                    <tr>
                      <th>Event</th><th>Outcome</th><th>Legs</th>
                      <th className="num">Gross</th><th className="num">Net</th><th className="num">Edge</th>
                      <th>Start</th><th className="num">MinVol$</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.arbs.map((a, i) => <ArbTableRow key={i} arb={a} />)}
                  </tbody>
                </table>
              </div>
              <p className="muted" style={{ marginBottom: 0, fontSize: 11.5 }}>
                Asks are REST snapshots — verify depth/price live before acting.
                Fees: Kalshi taker 0.07·p·(1−p), Polymarket US taker 0.06·p·(1−p).
              </p>
            </div>
          )}
        </>
      )}

      {!scanning && !result && (
        <div className="panel">
          <p className="muted" style={{ margin: 0 }}>
            Run a scan to fetch both venues' live books. Read-only — nothing is persisted
            and no orders are placed.
          </p>
        </div>
      )}
    </>
  )
}

function fmtStart(iso: string | null): string {
  if (!iso) return '?'
  const d = new Date(iso)
  if (isNaN(d.getTime())) return '?'
  return d.toLocaleString(undefined, { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })
}

function ArbTableRow({ arb }: { arb: ArbRow }) {
  const edge = arb.net_edge
  const rowClass = edge > 0 ? 'green' : ''
  return (
    <tr style={edge > 0 ? { background: 'var(--green-soft)' } : arb.gross_cost >= 1 ? { opacity: 0.6 } : undefined}>
      <td>
        <span className="badge">{arb.sector}</span>{' '}
        {arb.event_title}
        {arb.in_play && <span className="badge warn" style={{ marginLeft: 6 }}>LIVE</span>}
      </td>
      <td>{arb.market_desc}</td>
      <td>
        {arb.legs.map((leg, i) => (
          <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 5, whiteSpace: 'nowrap' }}>
            <VenueLogo venue={leg.venue} />
            <span className={leg.side === 'yes' ? 'green' : 'red'} style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase' }}>{leg.side}</span>
            <span>{leg.outcome}</span>
            <span className="muted">@{probToCents(leg.ask)}</span>
          </div>
        ))}
      </td>
      <td className="num">{arb.gross_cost.toFixed(3)}</td>
      <td className="num">{arb.net_cost.toFixed(3)}</td>
      <td className={`num ${rowClass}`} style={edge > 0 ? { fontWeight: 700 } : undefined}>
        {(edge * 100).toFixed(2) === '-0.00' ? '0.00' : `${edge > 0 ? '+' : ''}${(edge * 100).toFixed(2)}`}pp
      </td>
      <td className="muted">{fmtStart(arb.event_date)}</td>
      <td className="num">{Math.round(arb.min_volume_usd)}</td>
    </tr>
  )
}
