import type { MetricsResult } from '../lib/types'

interface Props { data: MetricsResult | null }

export function MetricsPanel({ data }: Props) {
  if (!data) return null
  if (data.error) return <div className="panel"><h2>Model Calibration</h2><p className="muted">{data.error}</p></div>

  return (
    <div className="panel">
      <h2>Model Calibration</h2>
      <div className="grid-kpi" style={{ marginBottom: 12 }}>
        <div className="kpi"><div className="label">Brier (Model)</div><div className="value">{data.brier_model.toFixed(4)}</div></div>
        <div className="kpi"><div className="label">Brier (Sharp)</div><div className="value">{data.brier_sharp.toFixed(4)}</div></div>
        <div className="kpi"><div className="label">Sample</div><div className="value">{data.n}</div></div>
        <div className="kpi"><div className="label">Win Rate</div><div className="value">{data.win_rate}%</div></div>
      </div>
      <h2 style={{ marginTop: 8 }}>Calibration Buckets</h2>
      <table>
        <thead><tr><th>Bucket</th><th className="num">N</th><th className="num">Predicted</th><th className="num">Actual</th><th className="num">Gap</th></tr></thead>
        <tbody>
          {data.calibration.map((b, i) => {
            const gap = b.actual - b.predicted
            const cls = Math.abs(gap) < 5 ? 'green' : Math.abs(gap) < 10 ? 'yellow' : 'red'
            return (
              <tr key={i}>
                <td>{b.bucket}</td>
                <td className="num">{b.n}</td>
                <td className="num">{b.predicted}%</td>
                <td className="num">{b.actual}%</td>
                <td className={`num ${cls}`}>{gap > 0 ? '+' : ''}{gap.toFixed(1)}%</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
