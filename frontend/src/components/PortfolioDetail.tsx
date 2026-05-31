import { useState, useEffect, useCallback } from 'react'
import { fetchPortfolioDetail, scanPortfolios, syncPortfolioOutcomes } from '../lib/api'
import { outcomeLabel, probToAmerican } from '../lib/odds'
import type { PortfolioDetail as PortfolioDetailType, PortfolioBet } from '../lib/types'

interface Props {
  portfolioId: string
  onBack: () => void
  toast: (msg: string, type?: 'info' | 'ok' | 'err') => void
}

export function PortfolioDetail({ portfolioId, onBack, toast }: Props) {
  const [data, setData] = useState<PortfolioDetailType | null>(null)
  const [scanning, setScanning] = useState(false)
  const [tab, setTab] = useState<'open' | 'settled' | 'all'>('open')

  const refresh = useCallback(async () => {
    try {
      const d = await fetchPortfolioDetail(portfolioId)
      setData(d)
    } catch (e) {
      toast('Failed to load portfolio', 'err')
    }
  }, [portfolioId, toast])

  useEffect(() => { refresh() }, [refresh])

  const handleScan = async () => {
    setScanning(true)
    toast('Scanning this portfolio...', 'info')
    try {
      const r = await scanPortfolios([portfolioId])
      const gaps = r.results[0]?.gaps_logged || 0
      toast(`Logged ${gaps} bets from ${r.markets_fetched} markets`, 'ok')
      refresh()
    } catch (e) {
      toast('Scan failed: ' + (e as Error).message, 'err')
    } finally {
      setScanning(false)
    }
  }

  const handleSync = async () => {
    try {
      const r = await syncPortfolioOutcomes()
      toast(`Resolved ${r.resolved} bet(s)`, 'ok')
      refresh()
    } catch (e) {
      toast('Sync failed: ' + (e as Error).message, 'err')
    }
  }

  if (!data) return <div style={{ color: '#7a8aa0', padding: 20 }}>Loading...</div>

  const filteredBets = data.bets.filter(b => {
    if (tab === 'open') return b.outcome === null
    if (tab === 'settled') return b.outcome !== null
    return true
  })

  const pnlColor = data.total_pnl >= 0 ? 'green' : 'red'

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <button className="btn btn-sm" onClick={onBack}>&larr; Back</button>
          <h2 style={{ margin: 0, textTransform: 'none', fontSize: 16 }}>{data.name}</h2>
          <span className="badge">{data.scenario}</span>
          <span style={{ color: '#7a8aa0', fontSize: 11 }}>{data.sectors.join(', ')}</span>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn primary" onClick={handleScan} disabled={scanning}>
            {scanning ? 'Scanning...' : 'Scan'}
          </button>
          <button className="btn" onClick={handleSync}>Sync</button>
        </div>
      </div>

      <div className="grid-kpi">
        <div className="kpi">
          <div className="label">Balance</div>
          <div className={`value ${pnlColor}`}>${data.current_bankroll.toFixed(2)}</div>
        </div>
        <div className="kpi">
          <div className="label">P&L</div>
          <div className={`value ${pnlColor}`}>{data.total_pnl >= 0 ? '+' : ''}{data.total_pnl.toFixed(2)}</div>
        </div>
        <div className="kpi">
          <div className="label">ROI</div>
          <div className={`value ${pnlColor}`}>{data.roi_pct.toFixed(1)}%</div>
        </div>
        <div className="kpi">
          <div className="label">Win Rate</div>
          <div className="value">{data.win_rate.toFixed(0)}%</div>
        </div>
        <div className="kpi">
          <div className="label">Bets</div>
          <div className="value">{data.total_bets}</div>
        </div>
        <div className="kpi">
          <div className="label">Kelly</div>
          <div className="value">{(data.kelly_fraction * 100).toFixed(0)}%</div>
        </div>
        <div className="kpi">
          <div className="label">Initial</div>
          <div className="value muted">${data.initial_bankroll}</div>
        </div>
        <div className="kpi">
          <div className="label">Avg EV</div>
          <div className="value green">{data.avg_ev.toFixed(1)}%</div>
        </div>
      </div>

      <div className="panel">
        <div className="panel-header">
          <div style={{ display: 'flex', gap: 6 }}>
            {(['open', 'settled', 'all'] as const).map(t => (
              <button key={t} className={`btn btn-sm pnl-tab ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>
                {t === 'open' ? `Open (${data.bets.filter(b => b.outcome === null).length})` :
                 t === 'settled' ? `Settled (${data.bets.filter(b => b.outcome !== null).length})` :
                 `All (${data.bets.length})`}
              </button>
            ))}
          </div>
        </div>

        {filteredBets.length === 0 ? (
          <div style={{ color: '#7a8aa0', textAlign: 'center', padding: 20 }}>No bets in this view</div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Date</th>
                <th>Event</th>
                <th>Outcome</th>
                <th>Sector</th>
                <th className="num">Odds</th>
                <th className="num">Fair Value</th>
                <th className="num">EV</th>
                <th className="num">Stake</th>
                {tab !== 'open' && <th className="num">P&L</th>}
                {tab !== 'open' && <th>Result</th>}
              </tr>
            </thead>
            <tbody>
              {filteredBets.map(bet => (
                <BetRow key={bet.id} bet={bet} showResult={tab !== 'open'} />
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}

function BetRow({ bet, showResult }: { bet: PortfolioBet; showResult: boolean }) {
  const label = bet.display_label || outcomeLabel(bet as any) || bet.yes_team || '?'

  return (
    <tr>
      <td style={{ whiteSpace: 'nowrap' }}>{bet.event_date || bet.scan_date}</td>
      <td style={{ maxWidth: 220 }}>{bet.event_title}</td>
      <td>{label}</td>
      <td><span className="badge">{bet.sector}</span></td>
      <td className="num">{probToAmerican(bet.kalshi_yes_price || 0)}</td>
      <td className="num">{probToAmerican(bet.blended_true_prob || 0)}</td>
      <td className="num green">{((bet.ev_pct || 0) * 100).toFixed(1)}%</td>
      <td className="num">${(bet.stake || 0).toFixed(2)}</td>
      {showResult && (
        <td className={`num ${(bet.pnl || 0) >= 0 ? 'green' : 'red'}`}>
          {bet.pnl != null ? `${bet.pnl >= 0 ? '+' : ''}$${bet.pnl.toFixed(2)}` : '-'}
        </td>
      )}
      {showResult && (
        <td>
          {bet.outcome === 1 ? <span className="green">WON</span> :
           bet.outcome === 0 ? <span className="red">LOST</span> :
           <span className="muted">PENDING</span>}
        </td>
      )}
    </tr>
  )
}
