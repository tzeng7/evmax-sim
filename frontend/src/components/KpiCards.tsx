import type { Summary } from '../lib/types'
import { AnimatedNumber } from './AnimatedNumber'

interface Props {
  summary: Summary | null
}

export function KpiCards({ summary }: Props) {
  if (!summary) return null
  const s = summary
  const pnlCls = s.total_pnl >= 0 ? 'green' : 'red'
  const roiCls = s.roi_pct >= 0 ? 'green' : 'red'

  const pnlCaret = s.total_pnl >= 0 ? '▲' : '▼'
  const roiCaret = s.roi_pct >= 0 ? '▲' : '▼'

  return (
    <div className="grid-kpi">
      <div className="kpi">
        <div className="label">Total P&L</div>
        <div className={`value ${pnlCls}`}>
          <span style={{ fontSize: 13, marginRight: 4 }}>{pnlCaret}</span>
          <AnimatedNumber value={s.total_pnl} format={(n) => `$${n.toFixed(2)}`} />
        </div>
      </div>
      <div className="kpi">
        <div className="label">ROI</div>
        <div className={`value ${roiCls}`}>
          <span style={{ fontSize: 13, marginRight: 4 }}>{roiCaret}</span>
          <AnimatedNumber value={s.roi_pct} format={(n) => `${n.toFixed(1)}%`} />
        </div>
      </div>
      <div className="kpi">
        <div className="label">Win Rate</div>
        <div className="value"><AnimatedNumber value={s.win_rate} format={(n) => `${Math.round(n)}%`} /></div>
      </div>
      <div className="kpi">
        <div className="label">Bets</div>
        <div className="value"><AnimatedNumber value={s.total_bets} format={(n) => String(Math.round(n))} /></div>
      </div>
      <div className="kpi">
        <div className="label">W / L</div>
        <div className="value">
          <AnimatedNumber value={s.wins} className="green" format={(n) => String(Math.round(n))} />
          <span className="muted"> / </span>
          <AnimatedNumber value={s.losses} className="red" format={(n) => String(Math.round(n))} />
        </div>
      </div>
      <div className="kpi">
        <div className="label">Avg EV</div>
        <div className="value"><AnimatedNumber value={s.avg_ev} format={(n) => `${n.toFixed(1)}%`} /></div>
      </div>
    </div>
  )
}
