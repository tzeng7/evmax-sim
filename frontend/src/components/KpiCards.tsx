import type { Summary } from '../lib/types'
import { AnimatedNumber } from './AnimatedNumber'
import { NotEvalLens } from './NotEvalLens'
import { fmtInt, fmtPnl, fmtSignedPct } from '../lib/format'

interface Props {
  summary: Summary | null
  onOpenEval?: () => void
}

export function KpiCards({ summary, onOpenEval }: Props) {
  if (!summary) return null
  const s = summary
  const pnlCls = s.total_pnl >= 0 ? 'green' : 'red'
  const roiCls = s.roi_pct >= 0 ? 'green' : 'red'

  return (
    <>
    <div className="grid-kpi">
      <div className="kpi">
        <div className="label">Total P&L</div>
        <div className={`value ${pnlCls}`}>
          <AnimatedNumber value={s.total_pnl} format={fmtPnl} />
        </div>
      </div>
      <div className="kpi">
        <div className="label">ROI</div>
        <div className={`value ${roiCls}`}>
          <AnimatedNumber value={s.roi_pct} format={(n) => fmtSignedPct(n)} />
        </div>
      </div>
      <div className="kpi">
        <div className="label">Win Rate</div>
        <div className="value"><AnimatedNumber value={s.win_rate} format={(n) => `${Math.round(n)}%`} /></div>
      </div>
      <div className="kpi">
        <div className="label">Bets</div>
        <div className="value"><AnimatedNumber value={s.total_bets} format={fmtInt} /></div>
      </div>
      <div className="kpi">
        <div className="label">W / L</div>
        <div className="value">
          <AnimatedNumber value={s.wins} className="green" format={fmtInt} />
          <span className="muted"> / </span>
          <AnimatedNumber value={s.losses} className="red" format={fmtInt} />
        </div>
      </div>
      <div className="kpi">
        <div className="label">Avg EV at entry</div>
        <div className="value"><AnimatedNumber value={s.avg_ev} format={(n) => `${n.toFixed(1)}%`} /></div>
      </div>
    </div>
    <NotEvalLens onOpenEval={onOpenEval} />
    </>
  )
}
