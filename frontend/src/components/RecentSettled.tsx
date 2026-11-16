import { useState } from 'react'
import type { Bet } from '../lib/types'
import { outcomeLabel } from '../lib/odds'
import { fmtDay, fmtPnl, prettyOutcome } from '../lib/format'
import { VenueLogo } from './VenueLogo'

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
  const net = visible.reduce((a, b) => a + pnl(b), 0)
  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title">
          <h2>Recent settled</h2>
          <span className="panel-sub">
            {visible.length} shown · <span className={net >= 0 ? 'green' : 'red'}>{fmtPnl(net)}</span>
          </span>
        </div>
        <div className="segmented" role="group" aria-label="Settled scope">
          <button className={`seg ${view === 'all' ? 'active' : ''}`} onClick={() => setView('all')}>All</button>
          <button className={`seg ${view === 'placed' ? 'active' : ''}`} onClick={() => setView('placed')}>Placed</button>
        </div>
      </div>
      {visible.length === 0 ? (
        <div className="empty-inline">No placed bets have settled yet.</div>
      ) : (
        <div className="scroll-area table-scroll">
          <table className="plays-table">
            <thead>
              <tr>
                <th>Event</th><th>Outcome</th>
                <th className="num" title="Entry price. Hover a value for the fair price and EV at entry.">Entry</th>
                <th>Result</th><th className="num">P&L</th>
              </tr>
            </thead>
            <tbody>
              {visible.map((b, i) => {
                const p = pnl(b)
                const entry = b.placed_price || b.kalshi_yes_price
                return (
                  <tr key={`${b.market_id}-${i}`}>
                    <td className="cell-event">
                      <div className="cell-title">{b.event_title}</div>
                      <div className="cell-meta">
                        <span>{fmtDay(b.event_date)}</span>
                        <span className="badge">{b.sector}</span>
                        <VenueLogo venue={b.venue} />
                        {b.placed === 1 && <span className="badge tone-blue" title="You placed this bet">Placed</span>}
                      </div>
                    </td>
                    <td className="cell-outcome">{prettyOutcome(outcomeLabel(b), b.event_title)}</td>
                    <td className="num price" title={`Fair ${Math.round((b.blended_true_prob || 0) * 100)}¢ · EV at entry ${((b.ev_pct || 0) * 100).toFixed(1)}%`}>{Math.round(entry * 100)}¢</td>
                    <td><span className={`badge ${b.outcome === 1 ? 'won' : 'lost'}`}>{b.outcome === 1 ? 'Won' : 'Lost'}</span></td>
                    <td className={`num ${p >= 0 ? 'green' : 'red'}`}>{fmtPnl(p)}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
