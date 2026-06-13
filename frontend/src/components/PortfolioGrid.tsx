import { useState, useEffect, useCallback } from 'react'
import { fetchPortfolios, createDefaultPortfolios, scanPortfolios, syncPortfolioOutcomes } from '../lib/api'
import type { Portfolio } from '../lib/types'

interface Props {
  onSelect: (id: string) => void
  toast: (msg: string, type?: 'info' | 'ok' | 'err') => void
}

// Risk-escalation ramp, neutralized: safe → green, mid → slate, hot → coral.
const SCENARIO_COLORS: Record<string, string> = {
  conservative: '#00d082',
  moderate: '#b3b8c0',
  aggressive: '#f0616d',
}

export function PortfolioGrid({ onSelect, toast }: Props) {
  const [portfolios, setPortfolios] = useState<Portfolio[]>([])
  const [loading, setLoading] = useState(true)
  const [scanning, setScanning] = useState(false)

  const refresh = useCallback(async () => {
    try {
      const data = await fetchPortfolios()
      setPortfolios(data)
    } catch (e) {
      console.error('Failed to load portfolios', e)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { refresh() }, [refresh])

  const handleCreateDefaults = async () => {
    toast('Creating default portfolios...', 'info')
    try {
      const r = await createDefaultPortfolios()
      toast(`Created ${r.created} portfolios`, 'ok')
      refresh()
    } catch (e) {
      toast('Failed: ' + (e as Error).message, 'err')
    }
  }

  const handleScanAll = async () => {
    setScanning(true)
    toast('Scanning all portfolios...', 'info')
    try {
      const r = await scanPortfolios()
      const total = r.results.reduce((s, x) => s + x.gaps_logged, 0)
      toast(`Scanned ${r.portfolios_scanned} portfolios, logged ${total} bets across ${r.markets_fetched} markets`, 'ok')
      refresh()
    } catch (e) {
      toast('Scan failed: ' + (e as Error).message, 'err')
    } finally {
      setScanning(false)
    }
  }

  const handleSync = async () => {
    toast('Syncing outcomes...', 'info')
    try {
      const r = await syncPortfolioOutcomes()
      toast(`Resolved ${r.resolved} portfolio bet(s)`, 'ok')
      refresh()
    } catch (e) {
      toast('Sync failed: ' + (e as Error).message, 'err')
    }
  }

  if (loading) return <div style={{ color: '#767b85', padding: 20 }}>Loading portfolios...</div>

  const grouped = new Map<string, Portfolio[]>()
  for (const p of portfolios) {
    const key = p.sectors.sort().join(',')
    if (!grouped.has(key)) grouped.set(key, [])
    grouped.get(key)!.push(p)
  }

  return (
    <div>
      <div className="panel-header" style={{ marginBottom: 16 }}>
        <h2>Portfolios</h2>
        <div style={{ display: 'flex', gap: 8 }}>
          {portfolios.length === 0 && (
            <button className="btn primary" onClick={handleCreateDefaults}>Create Defaults</button>
          )}
          <button className="btn primary" onClick={handleScanAll} disabled={scanning || portfolios.length === 0}>
            {scanning ? 'Scanning...' : 'Scan All'}
          </button>
          <button className="btn" onClick={handleSync} disabled={portfolios.length === 0}>
            Sync Outcomes
          </button>
        </div>
      </div>

      {portfolios.length === 0 ? (
        <div className="panel" style={{ textAlign: 'center', padding: 40 }}>
          <p style={{ color: '#767b85', marginBottom: 12 }}>No portfolios yet. Create default portfolios to get started.</p>
          <button className="btn primary" onClick={handleCreateDefaults}>Create Default Portfolios</button>
        </div>
      ) : (
        Array.from(grouped.entries()).map(([key, group]) => (
          <div key={key} style={{ marginBottom: 24 }}>
            <h2 style={{ marginBottom: 10 }}>
              {group[0].name.replace(/ (Conservative|Moderate|Aggressive)$/, '')}
              <span style={{ color: '#767b85', fontSize: 11, marginLeft: 8 }}>{group[0].sectors.join(', ')}</span>
            </h2>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10 }}>
              {group.sort((a, b) => a.initial_bankroll - b.initial_bankroll).map(p => (
                <div
                  key={p.id}
                  className="panel card-clickable"
                  onClick={() => onSelect(p.id)}
                  style={{ cursor: 'pointer', borderLeft: `3px solid ${SCENARIO_COLORS[p.scenario] || '#00d082'}`, marginBottom: 0 }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                    <span style={{ fontWeight: 600, textTransform: 'capitalize', color: SCENARIO_COLORS[p.scenario] }}>
                      {p.scenario}
                    </span>
                    <span style={{ fontSize: 10, color: '#767b85' }}>
                      ${p.initial_bankroll} / {(p.kelly_fraction * 100).toFixed(0)}% Kelly
                    </span>
                  </div>

                  <div style={{ fontSize: 23, fontWeight: 600, marginBottom: 6, fontFamily: 'var(--mono)', letterSpacing: '-0.02em', fontVariantNumeric: 'tabular-nums' }}>
                    <span className={p.current_bankroll >= p.initial_bankroll ? 'green' : 'red'}>
                      ${p.current_bankroll.toFixed(2)}
                    </span>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4, fontSize: 11, fontFamily: 'var(--mono)' }}>
                    <div>
                      <span className="muted">P&L </span>
                      <span className={p.total_pnl >= 0 ? 'green' : 'red'}>
                        {p.total_pnl >= 0 ? '+' : ''}{p.total_pnl.toFixed(2)}
                      </span>
                    </div>
                    <div>
                      <span className="muted">ROI </span>
                      <span className={p.roi_pct >= 0 ? 'green' : 'red'}>
                        {p.roi_pct.toFixed(1)}%
                      </span>
                    </div>
                    <div>
                      <span className="muted">Bets </span>
                      <span>{p.total_bets}</span>
                    </div>
                    <div>
                      <span className="muted">Win </span>
                      <span>{p.win_rate.toFixed(0)}%</span>
                    </div>
                    <div>
                      <span className="muted">Open </span>
                      <span>{p.open_bets}</span>
                    </div>
                    <div>
                      <span className="muted">Avg EV </span>
                      <span className="green">{p.avg_ev.toFixed(1)}%</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))
      )}
    </div>
  )
}
