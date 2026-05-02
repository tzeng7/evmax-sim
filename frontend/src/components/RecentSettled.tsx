import { useState } from 'react'
import type { Bet } from '../lib/types'
import { outcomeLabel } from '../lib/odds'

interface Props { bets: Bet[] }

type View = 'all' | 'placed'

function pnl(b: Bet): number {
  const stake = b.placed_stake || (b.bankroll_used || 250) * (b.kelly_fraction || 0)
  const price = b.placed_price || b.kalshi_yes_price || 0.5
  if (price <= 0 || price >= 1) return 0
  return b.outcome === 1 ? stake * (1 / price - 1) : -stake
}

export function RecentSettled({ bets }: Props) {
  const [view, setView] = useState<View>('all')
  if (!bets.length) return null
  const visible = view === 'placed' ? bets.filter(b => b.placed === 1) : bets
  return (
    <div className="panel">
      <div className="panel-header">
        <h2>Recent Settled Bets</h2>
        <div style={{ display: 'flex', gap: 6 }}>
          <button
            className={`btn btn-sm settled-tab ${view === 'all' ? 'active' : ''}`}
            onClick={() => setView('all')}
          >All</button>
          <button
            className={`btn btn-sm settled-tab ${view === 'placed' ? 'active' : ''}`}
            onClick={() => setView('placed')}
          >Placed Only</button>
        </div>
      </div>
      <div style={{ maxHeight: 400, overflowY: 'auto' }}>
        <table>
          <thead>
            <tr>
              <th>Date</th><th>Sector</th><th>Event</th><th>Outcome</th>
              <th className="num">Ask</th><th className="num">Model</th><th className="num">EV</th>
              <th>Result</th><th className="num">P&L</th>
            </tr>
          </thead>
          <tbody>
            {visible.map((b, i) => {
              const p = pnl(b)
              return (
                <tr key={i}>
                  <td className="muted">{b.event_date}</td>
                  <td><span className="badge">{b.sector}</span></td>
                  <td>{b.event_title}</td>
                  <td>{outcomeLabel(b)}</td>
                  <td className="num">{Math.round(b.kalshi_yes_price * 100)}c</td>
                  <td className="num">{Math.round((b.blended_true_prob || 0) * 100)}%</td>
                  <td className="num">{((b.ev_pct || 0) * 100).toFixed(1)}%</td>
                  <td><span className={`badge ${b.outcome === 1 ? 'won' : 'lost'}`}>{b.outcome === 1 ? 'WON' : 'LOST'}</span></td>
                  <td className={`num ${p >= 0 ? 'green' : 'red'}`}>${p.toFixed(2)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
