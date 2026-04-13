import type { SectorRow } from '../lib/types'

interface Props { sectors: SectorRow[] }

export function SectorPerformance({ sectors }: Props) {
  return (
    <div className="panel">
      <h2>Sector Performance</h2>
      <table>
        <thead>
          <tr><th>Sector</th><th className="num">Bets</th><th className="num">Wins</th><th className="num">Win%</th><th className="num">P&L</th><th className="num">ROI</th></tr>
        </thead>
        <tbody>
          {sectors.map(s => (
            <tr key={s.sector}>
              <td><span className="badge">{s.sector}</span></td>
              <td className="num">{s.bets}</td>
              <td className="num">{s.wins}</td>
              <td className="num">{s.win_rate}%</td>
              <td className={`num ${s.pnl >= 0 ? 'green' : 'red'}`}>${s.pnl.toFixed(2)}</td>
              <td className={`num ${s.roi_pct >= 0 ? 'green' : 'red'}`}>{s.roi_pct}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
