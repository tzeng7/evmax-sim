import type { Summary } from '../lib/types'

interface Props {
  summary: Summary | null
}

export function KpiCards({ summary }: Props) {
  if (!summary) return null
  const s = summary
  const pnlCls = s.total_pnl >= 0 ? 'green' : 'red'
  const roiCls = s.roi_pct >= 0 ? 'green' : 'red'

  return (
    <div className="grid-kpi">
      <div className="kpi"><div className="label">Total P&L</div><div className={`value ${pnlCls}`}>${s.total_pnl.toFixed(2)}</div></div>
      <div className="kpi"><div className="label">ROI</div><div className={`value ${roiCls}`}>{s.roi_pct.toFixed(1)}%</div></div>
      <div className="kpi"><div className="label">Win Rate</div><div className="value">{s.win_rate}%</div></div>
      <div className="kpi"><div className="label">Bets</div><div className="value">{s.total_bets}</div></div>
      <div className="kpi"><div className="label">W / L</div><div className="value"><span className="green">{s.wins}</span><span className="muted"> / </span><span className="red">{s.losses}</span></div></div>
      <div className="kpi"><div className="label">Avg EV</div><div className="value">{s.avg_ev}%</div></div>
    </div>
  )
}
